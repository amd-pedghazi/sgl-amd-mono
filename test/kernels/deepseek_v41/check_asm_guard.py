# SPDX-License-Identifier: Apache-2.0
"""The asm collectives' deadline and fail-stop (asm/guard.py) on 4 gfx950 GPUs, in the
shape of check_guard.py and test_guard_multi_rank.py: one worker process a rank, each
spawned and waited for on its own (a torchrun-style launcher would tear the group down
when a rank dies and hide whether each rank stopped by itself).

    python check_asm_guard.py --out DIR [--kinds comb_ar_bf16,...] [--cases clean,skip,...]

Each worker runs LAUNCHES launches of one collective at ROWS rows; every completed
launch's output is checked exactly against its expected value (the inputs change per
launch and row, so a stale or mismatched output fails) and appended to the rank's
progress file. The first launch runs under the default deadline, then every rank sets
SHORT_S. Cases:
- clean: every launch completes on every rank, no guard word set, exit 0;
- delay: rank 1 starts launch SKIP_AT DELAY_S late (below the deadline): still clean;
- skip: rank 1 skips launch SKIP_AT and holds HOLD_S (above the deadline), then goes on.
  The peers' waits pass their deadline and stop every rank; rank 1's next launch stops
  on the stop word. Every rank exits non-zero, completes exactly SKIP_AT launches, and
  the peers stop within SHORT_S + STOP_MARGIN_S of the skip;
- skip_words: skip with halt=False kernels: a failed wave ends at once, the worker reads
  the guard words after each launch and exits 3; the peers' site is the kernel's wait,
  rank 1's stop is the stop word;
- graph, skip_graph: clean and skip with the launch captured in one HIP graph.
Writes DIR/summary.json and per case DIR/<kind>.<case>/ (progress, stderr, words).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROWS = 64
LAUNCHES = 8
SKIP_AT = 3
SHORT_S = 6.0  # the guard floor is 5 s
HOLD_S = SHORT_S + 4.0  # rank 1's hold after its skip: the peers have stopped by then
DELAY_S = SHORT_S / 2
STOP_MARGIN_S = 15.0  # host slack: the 64-poll check, the fault and the teardown
KINDS = (
    "comb_ar_bf16",
    "comb_ar_fp8",
    "wob_ar_bf16",
    "wob_ar_fp8",
    "ar2",
    "ar2f8",
    "eg_ag",
)
EG_COLS = 512  # the Engram all-gather's bf16 columns a rank (1024-byte rows)
CASES = ("clean", "delay", "skip", "skip_words", "graph", "skip_graph")
EXIT_WORDS, EXIT_BAD = 3, 5


def _no_core():
    """Keep this rank's deliberate GPU fault from dumping a core (check_guard.py's)."""
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE = 4
        raise SystemExit(
            f"prctl(PR_SET_DUMPABLE, 0) failed: errno {ctypes.get_errno()}"
        )


def _launcher(kind, rank, cpu, guard, halt, dev):
    """(runner, set_inputs(i), launch(out) -> out, expected(i), out_given, ordered) for this
    kind: set_inputs writes launch i's inputs in place (outside a graph), launch makes the
    one collective launch (captured once in graph mode); out_given: launch writes a given
    ``out``, which the worker fills with NaN first (a launch that skips a row leaves NaN);
    ordered: the ranks meet after each launch (the Engram all-gather's single-buffered dst,
    which the main stream's all-reduces order in serving)."""
    import torch

    from sgl_amd_mono.deepseek_v41.asm import ar_runner

    H = 5120
    t = torch.arange(ROWS, device=dev).view(ROWS, 1)

    def value(i, r):  # exact in bf16 and in e4m3 with an E8M0 block scale
        return ((r + 1) * torch.pow(2.0, ((i + t) % 5).float())).expand(ROWS, H)

    def expected(i):
        return (
            (10.0 * torch.pow(2.0, ((i + t) % 5).float()))
            .expand(ROWS, H)
            .to(torch.bfloat16)
        )

    if kind.startswith("comb_ar"):
        wire = kind.split("_")[-1]
        run = ar_runner.CombAllReduce(rank, cpu, 4, guard=guard, halt=halt)
        # part = bf16(bf16(Σ y) + add) = add: the all-reduce of add
        y = torch.zeros(ROWS * 6, H, dtype=torch.float32, device=dev)
        ids = torch.zeros(ROWS * 6, dtype=torch.int32, device=dev)
        add = torch.empty(ROWS, H, dtype=torch.bfloat16, device=dev)
        return (run, lambda i: add.copy_(value(i, rank)),
                lambda out: run(y, ids, add, ROWS, wire), expected, False, False)  # fmt: skip
    if kind == "eg_ag":
        run = ar_runner.EngramAllGather(rank, cpu, EG_COLS * 2, guard=guard, halt=halt)
        src = torch.empty(ROWS, EG_COLS, dtype=torch.bfloat16, device=dev)

        def gathered(i):  # rank r's row at columns r * EG_COLS
            return torch.cat([value(i, r)[:, :EG_COLS] for r in range(4)], 1).to(
                torch.bfloat16
            )

        return (run, lambda i: src.copy_(value(i, rank)[:, :EG_COLS]),
                lambda out: run(src), gathered, False, True)  # fmt: skip
    if kind.startswith("wob_ar"):
        from sgl_amd_mono.deepseek_v41.asm.kernels import gemv8

        wire = kind.split("_")[-1]
        cfg = gemv8.WO_B1
        run = ar_runner.WoBAllReduce(rank, cpu, 4, guard=guard, halt=halt)
        # zero codes and E8M0 scales (sized generously for the frag layouts): out is 0
        w = torch.zeros(2 * cfg.N * cfg.K, dtype=torch.uint8, device=dev)
        ws = torch.zeros(2 * cfg.N * cfg.K // 32, dtype=torch.uint8, device=dev)
        z = torch.zeros(2 * 256 * cfg.K, dtype=torch.uint8, device=dev)
        zs = torch.zeros(2 * 256 * cfg.K // 32, dtype=torch.uint8, device=dev)
        zero = torch.zeros(ROWS, H, dtype=torch.bfloat16, device=dev)
        return (run, lambda i: None, lambda out: run(w, ws, z, zs, ROWS, wire, out=out),
                lambda i: zero, True, False)  # fmt: skip
    run = (ar_runner.BF16AllReduce if kind == "ar2" else ar_runner.FP8AllReduce)(
        rank, cpu, 4, guard=guard, halt=halt
    )
    part = torch.empty(ROWS, H, dtype=torch.bfloat16, device=dev)
    return (run, lambda i: part.copy_(value(i, rank)), lambda out: run(part), expected,
            False, False)  # fmt: skip


def worker(a):
    import torch
    import torch.distributed as dist

    _no_core()
    rank = a.rank
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{a.init_file}", rank=rank, world_size=4
    )
    cpu = dist.group.WORLD
    from sgl_amd_mono.deepseek_v41.asm.guard import AsmGuard

    halt = a.case != "skip_words"
    guard = AsmGuard(rank, cpu, 4)  # the default deadline covers the first launch
    run, set_inputs, launch, expected, out_given, ordered = _launcher(
        a.kind, rank, cpu, guard, halt, dev
    )
    graph = a.case in ("graph", "skip_graph")
    skip = a.case in ("skip", "skip_words", "skip_graph")
    pdir = Path(a.progress_dir)
    words_path = pdir / f"rank{rank}.json"
    g = out_g = None
    with open(pdir / f"rank{rank}.log", "a", buffering=1) as progress:
        for i in range(a.launches):
            if rank == 1 and i == SKIP_AT and skip:
                (pdir / "skip_time").write_text(f"{time.time()}\n")
                time.sleep(HOLD_S)
                continue
            if rank == 1 and i == SKIP_AT and a.case == "delay":
                time.sleep(DELAY_S)
            set_inputs(i)
            if graph and i >= 1:
                if g is None:
                    # captured after the eager first launch; capture launches nothing
                    out_g = (
                        torch.empty(ROWS, 5120, dtype=torch.bfloat16, device=dev)
                        if out_given
                        else None
                    )
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        out_g = launch(out_g)
                if out_given:
                    out_g.fill_(float("nan"))
                g.replay()
                out = out_g
            else:
                out = (
                    torch.empty(ROWS, 5120, dtype=torch.bfloat16, device=dev)
                    if out_given
                    else None
                )
                if out_given:
                    out.fill_(float("nan"))
                out = launch(out)
            torch.cuda.synchronize()
            if not halt:
                e = guard.errors()
                if e:
                    words_path.write_text(
                        json.dumps(dict(launch=i, errors=e, timeouts=run.errors()))
                    )
                    os._exit(EXIT_WORDS)
            if not torch.equal(out, expected(i)):
                bad = (out.float() - expected(i).float()).abs()
                progress.write(f"BAD {i} max {bad.nan_to_num(1e30).max().item()}\n")
                os._exit(EXIT_BAD)
            progress.write(f"{i}\n")
            if i == 0:
                dist.barrier(group=cpu)
                guard.set_deadline(SHORT_S)
            elif ordered:
                dist.barrier(group=cpu)
    words_path.write_text(
        json.dumps(
            dict(launch=a.launches, errors=guard.errors(), timeouts=run.errors())
        )
    )
    dist.destroy_process_group()


def _spawn(kind, case, root):
    d = root / f"{kind}.{case}"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        f.unlink()
    procs, errs = {}, {}
    for r in range(4):
        errs[r] = open(d / f"rank{r}.err", "w")
        procs[r] = subprocess.Popen(
            [sys.executable, __file__, "--worker", "--kind", kind, "--case", case,
             "--rank", str(r), "--init-file", str(d / "init"), "--launches", str(LAUNCHES),
             "--progress-dir", str(d)],
            stdout=errs[r], stderr=subprocess.STDOUT,
        )  # fmt: skip
    t0 = time.time()
    stop = {}
    hard = 600
    while len(stop) < 4 and time.time() - t0 < hard:
        for r, p in procs.items():
            if r not in stop and p.poll() is not None:
                stop[r] = time.time()
        time.sleep(0.1)
    for r, p in procs.items():
        if r not in stop:
            p.kill()
            p.wait()
        errs[r].close()
    res = dict(kind=kind, case=case, ranks={})
    skip_t = (
        float((d / "skip_time").read_text()) if (d / "skip_time").exists() else None
    )
    res["skip_time"] = skip_t
    for r in range(4):
        lines = (
            (d / f"rank{r}.log").read_text().split()
            if (d / f"rank{r}.log").exists()
            else []
        )
        words = (
            json.loads((d / f"rank{r}.json").read_text())
            if (d / f"rank{r}.json").exists()
            else None
        )
        err = (d / f"rank{r}.err").read_text(errors="replace")
        res["ranks"][r] = dict(
            rc=procs[r].returncode,
            hung=r not in stop,
            stop_after_skip_s=(stop[r] - skip_t) if (r in stop and skip_t) else None,
            done=sum(1 for x in lines if x.isdigit()),
            bad=[
                x
                for x in (d / f"rank{r}.log").read_text().splitlines()
                if x.startswith("BAD")
            ]
            if (d / f"rank{r}.log").exists()
            else [],
            words=words,
            null_page_fault=("Memory access fault" in err)
            and ("(nil)" in err or "0x0" in err),
            err_tail=err.strip().splitlines()[-3:],
        )
    return res


def judge(res):
    """The case's pass rules; a list of failures."""
    kind, case, R = res["kind"], res["case"], res["ranks"]
    sites = {
        f"{kind.rsplit('_', 1)[0] if kind.startswith(('comb', 'wob')) else kind}.{h}"
        for h in ("rs", "ag")
    }
    f = []
    for r, x in R.items():
        if x["hung"]:
            f.append(f"rank {r} hung")
        if x["bad"]:
            f.append(f"rank {r} returned a wrong output {x['bad']}")
    if case in ("clean", "delay", "graph"):
        for r, x in R.items():
            if x["rc"] != 0 or x["done"] != LAUNCHES:
                f.append(f"rank {r}: rc {x['rc']} done {x['done']}")
            if not x["words"] or x["words"]["errors"] or x["words"]["timeouts"]:
                f.append(f"rank {r}: guard words {x['words']}")
        return f
    for r, x in R.items():
        if x["done"] != SKIP_AT:
            f.append(f"rank {r} completed {x['done']} launches, not {SKIP_AT}")
        if x["rc"] == 0:
            f.append(f"rank {r} exited 0")
        lim = (HOLD_S if r == 1 else SHORT_S) + STOP_MARGIN_S
        if x["stop_after_skip_s"] is None or x["stop_after_skip_s"] > lim:
            f.append(
                f"rank {r} stopped {x['stop_after_skip_s']} s after the skip (> {lim})"
            )
    if case == "skip_words":
        # the first peer past its deadline stops every rank: the other peers may stop on its stop
        # word before their own deadline; rank 1 (the skipper) stops on it at its next launch
        timed_out = 0
        for r, x in R.items():
            e = (x["words"] or {}).get("errors", {})
            if x["rc"] != EXIT_WORDS:
                f.append(f"rank {r}: rc {x['rc']} (words exit {EXIT_WORDS})")
            if not e.get("stop"):
                f.append(f"rank {r}: no stop word {e}")
            if r == 1 and (e.get("peer") not in sites or e.get("site")):
                f.append(f"rank 1: not stopped on the stop word alone {e}")
            if r != 1 and not ({e.get("site"), e.get("peer")} & sites):
                f.append(f"rank {r}: no wait of {sites} stopped {e}")
            if 1 in {int(k) for k in e.get("by", {})}:
                f.append(f"rank {r}: rank 1 recorded as timed out {e}")
            timed_out += r != 1 and e.get("site") in sites
        if not timed_out:
            f.append("no peer's wait timed out")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--kind", default="comb_ar_bf16")
    ap.add_argument("--case", default="clean")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--init-file")
    ap.add_argument("--launches", type=int, default=LAUNCHES)
    ap.add_argument("--progress-dir")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--out", default="/mb_logs/guard")
    a = ap.parse_args()
    if a.worker:
        worker(a)
        return
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    summary, failed = [], 0
    for kind in a.kinds.split(","):
        for case in a.cases.split(","):
            t = time.time()
            res = _spawn(kind, case, root)
            res["failures"] = judge(res)
            res["wall_s"] = round(time.time() - t, 1)
            failed += bool(res["failures"])
            summary.append(res)
            print(
                f"{kind:13s} {case:11s} {'PASS' if not res['failures'] else 'FAIL'} {res['wall_s']:6.1f} s  "
                + "; ".join(
                    f"r{r} rc={x['rc']} done={x['done']}"
                    for r, x in res["ranks"].items()
                )
                + (f"  {res['failures']}" if res["failures"] else ""),
                flush=True,
            )
            (root / "summary.json").write_text(
                json.dumps(summary, indent=1, default=str)
            )
    print(f"{len(summary)} cases, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
