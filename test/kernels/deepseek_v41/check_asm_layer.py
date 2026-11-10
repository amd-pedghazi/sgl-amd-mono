# SPDX-License-Identifier: Apache-2.0
"""The asm decode layers (asm/layer.py) through the real model, as vLLM's
test_dsv41_asm_layers.py checks them: DeepSeek-V4.1-Flash at TP4 on gfx950 with the
wide path on (SGLANG_AMD_MONO_ASM_WIDE=1), decode steps of 64, 128, 192 and 256 rows.

--mode check (eager decode): the plugin's check mode (SGLANG_AMD_MONO_ASM_CHECK) runs
SGLang's FFN half on each wide layer's K1 outputs next to the asm K2 and records the
per-row differences of every layer's output and residual. vLLM's bounds: the residual
within 2 % of SGLang's; the output within 12 % unless the token's top-6 routing is a
near tie on SGLang's path (then the token is exempt from that layer on); at most a
quarter of the tokens exempt. Each requested row count must have run.
--mode serve (the control's decode graphs, check mode off): the startup log routes the
wide batch sizes to the monokernel forward, and batches of each row count complete.
--mode kv: the control itself (no wide path, eager), for its KV pool; every mode reports
the KV pool (max_total_num_tokens) and the monokernel build's memory from the log.
--dspark: DSpark verify steps (block 5: 6 rows a request); rows are a step's rows.

    python check_asm_layer.py --mode check --out DIR [--rows 64,128,192,256]
    python check_asm_layer.py --mode serve --out DIR [--rows 64,128]
    python check_asm_layer.py --mode kv --out DIR
Writes DIR/summary.json and DIR/server.{out,err}; the check's records in DIR/records/.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# the control's server flags (host, port, parsers and tokenizer left out)
CONTROL_FLAGS = [
    "--tp-size", "4", "--trust-remote-code", "--chunked-prefill-size", "16384",
    "--context-length", "135168", "--cuda-graph-backend-prefill", "disabled",
    "--cuda-graph-max-bs-decode", "256", "--enable-amd-monokernel",
    "--enable-decoder-swa-bounded-replay", "--enforce-shared-experts-fusion",
    "--fp8-gemm-backend", "aiter", "--max-prefill-tokens", "16384",
    "--max-running-requests", "256", "--mem-fraction-static", "0.87",
]  # fmt: skip
DSPARK_FLAGS = [
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
]
DSPARK_ROWS = 6  # a verify step's rows a request (block 5)
# decode tokens a request: more than the batch's prefill chunks, so that every request
# still decodes when the last one starts (steps of the full row count)
STEPS = 64
# prompts past the ratio-4 sources' 512 compressed entries (2048 tokens)
PROMPT_LEN = (2100, 3100)
RES_REL, X_REL, TIE_FIRST, TIE_LATER = 0.02, 0.12, 5e-3, 1e-2


def model_path(name: str) -> str:
    if os.path.isdir(name):
        return name
    hub = os.environ.get("HF_HUB_CACHE", "/root/.cache/huggingface/hub")
    snaps = sorted(
        glob.glob(f"{hub}/models--{name.replace('/', '--')}/snapshots/*/config.json")
    )
    if not snaps:
        raise SystemExit(f"{name} not in {hub}")
    return os.path.dirname(snaps[-1])


def prompts(n: int) -> list:
    import torch

    g = torch.Generator().manual_seed(0)
    lens = torch.randint(*PROMPT_LEN, (n,), generator=g).tolist()
    return [torch.randint(1000, 60000, (k,), generator=g).tolist() for k in lens]


def serve(args, out: Path) -> dict:
    import requests
    from sglang.test.test_utils import (
        popen_launch_server,
        terminate_and_kill_process_tree,
    )

    url = f"http://127.0.0.1:{args.port}"
    rows = [int(r) for r in args.rows.split(",")] if args.mode != "kv" else []
    per_req = DSPARK_ROWS if args.dspark else 1
    env = dict(os.environ)
    env.pop("SGLANG_AMD_MONO_ASM_WIDE", None)
    if args.mode != "kv":
        env["SGLANG_AMD_MONO_ASM_WIDE"] = "1"
    env["SGLANG_AMD_MONO_ASM_K2"] = "v1" if args.k2_v1 else "v3"
    flags = list(CONTROL_FLAGS) + ["--watchdog-timeout", "600"]
    if args.no_shared_fusion:
        # SGLang's reference then runs the checkpoint's FP8 shared expert, as K2 does
        flags.remove("--enforce-shared-experts-fusion")
    if args.dspark:
        flags += DSPARK_FLAGS
    if args.mode in ("check", "kv"):
        flags += ["--cuda-graph-backend-decode", "disabled"]
    if args.mode == "check":
        env["SGLANG_AMD_MONO_ASM_CHECK"] = str(out / "records")
    logs = (open(out / "server.out", "w+"), open(out / "server.err", "w+"))
    t0 = time.time()
    proc = popen_launch_server(
        model_path(args.model), url, timeout=3600,
        other_args=flags, env=env, return_stdout_stderr=logs,
    )  # fmt: skip
    res = dict(mode=args.mode, rows=rows, dspark=args.dspark, k2_chain="v1" if args.k2_v1 else "v3",
               shared_fusion=not args.no_shared_fusion,
               startup_s=round(time.time() - t0, 1), batches=[])  # fmt: skip
    try:
        # the same prompts in every run (the graph run's tokens against the eager run's)
        p = prompts(256)
        for m in rows:
            assert m % per_req == 0, (m, per_req)
            requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=40)
            t = time.time()
            r = requests.post(
                url + "/generate",
                json={
                    "input_ids": p[: m // per_req],
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": STEPS,
                        "ignore_eos": True,
                    },
                },
                timeout=3600,
            )
            r.raise_for_status()
            outs = r.json()
            done = sum(
                1
                for o in outs
                if o.get("meta_info", {}).get("completion_tokens") == STEPS
            )
            # greedy tokens of the first requests: the graph run's against the eager check run's
            first = [list(o.get("output_ids") or [])[:16] for o in outs[:8]]
            res["batches"].append(dict(rows=m, requests=m // per_req, completed=done,
                                       wall_s=round(time.time() - t, 1), first_tokens=first))  # fmt: skip
    finally:
        terminate_and_kill_process_tree(proc)
        for f in logs:
            f.close()
    log = (out / "server.out").read_text(errors="replace") + (
        out / "server.err"
    ).read_text(errors="replace")
    res["routed_line"] = next(
        (
            ln.strip()
            for ln in log.splitlines()
            if "monokernel" in ln and " runs " in ln
        ),
        None,
    )
    kv = re.findall(r"max_total_num_tokens=(\d+)", log)
    res["max_total_num_tokens"] = int(kv[-1]) if kv else None
    built = re.findall(r"widths .*? built in ([\d.]+) s, ([\d.]+) MiB", log)
    res["monokernel_build"] = (
        dict(s=float(built[-1][0]), mib=float(built[-1][1])) if built else None
    )
    res["memory_fault_lines"] = [
        ln.strip() for ln in log.splitlines() if "Memory access fault" in ln
    ][:4]
    return res


def judge_records(rec_dir: Path, rows: list) -> dict:
    """vLLM's bounds over the check's records: per (rank, step), layers in order."""
    by_step = defaultdict(list)
    for path in sorted(rec_dir.glob("rank*.jsonl")):
        rank = int(path.stem[4:])
        for line in path.read_text().splitlines():
            r = json.loads(line)
            by_step[(rank, r["step"])].append(r)
    stats = defaultdict(lambda: dict(steps=0, tokens=0, flipped=0, res_max=0.0, x_max=0.0,
                                     x_max_any=0.0, mix_max=0.0, failures=[]))  # fmt: skip
    first = min((r["layer"] for recs in by_step.values() for r in recs), default=0)
    for (rank, step), recs in sorted(by_step.items()):
        recs.sort(key=lambda r: r["layer"])
        m = recs[0]["rows"]
        s = stats[m]
        s["steps"] += 1
        s["tokens"] += m
        flipped = set()
        for r in recs:
            tie = TIE_FIRST if r["layer"] == first else TIE_LATER
            s["mix_max"] = max(s["mix_max"], r["mix_rel"])
            for t in range(m):
                s["x_max_any"] = max(s["x_max_any"], r["x_rel"][t])
                if t in flipped:
                    continue
                s["res_max"] = max(s["res_max"], r["res_rel"][t])
                if r["res_rel"][t] >= RES_REL:
                    s["failures"].append(
                        (rank, step, r["layer"], t, "residual", r["res_rel"][t])
                    )
                if r["x_rel"][t] >= X_REL:
                    if r["gap"][t] < tie:
                        flipped.add(t)
                    else:
                        s["failures"].append(
                            (rank, step, r["layer"], t, "x", r["x_rel"][t])
                        )
                else:
                    s["x_max"] = max(s["x_max"], r["x_rel"][t])
        s["flipped"] += len(flipped)
    out = {}
    for m, s in sorted(stats.items()):
        s["failures"] = s["failures"][:20] + (
            [f"... {len(s['failures']) - 20} more"] if len(s["failures"]) > 20 else []
        )
        s["flip_bound_ok"] = s["flipped"] <= s["tokens"] // 4
        out[m] = s
    out["missing_rows"] = [m for m in rows if m not in stats]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("check", "serve", "kv"), default="check")
    ap.add_argument(
        "--dspark", action="store_true", help="DSpark verify steps (6 rows a request)"
    )
    ap.add_argument(
        "--k2-v1",
        action="store_true",
        help="K2Runner's v1 chain (the norm before the MXFP8)",
    )
    ap.add_argument(
        "--no-shared-fusion",
        action="store_true",
        help="without --enforce-shared-experts-fusion (SGLang's shared expert in FP8, not MXFP4)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="64,128,192,256")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--port", type=int, default=30001, help="the slot's server port")
    ap.add_argument("--judge-only", action="store_true", help="re-judge DIR/records")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    res = {} if args.judge_only else serve(args, out)
    rows = [int(r) for r in args.rows.split(",")] if args.mode != "kv" else []
    per_req = DSPARK_ROWS if args.dspark else 1
    fails = []
    if args.mode == "kv" and not res.get("max_total_num_tokens"):
        fails.append("no max_total_num_tokens in the server log")
    if args.mode == "check":
        res["judge"] = j = judge_records(out / "records", rows)
        fails += [f"no wide step of {m} rows recorded" for m in j["missing_rows"]]
        for m in rows:
            s = j.get(m)
            if s and (s["failures"] or not s["flip_bound_ok"]):
                fails.append(
                    f"{m} rows: {len(s['failures'])} out of bounds, {s['flipped']} of {s['tokens']} flipped"
                )
    for b in res.get("batches", []):
        if b["completed"] != b["requests"]:
            fails.append(
                f"{b['rows']}-row batch: {b['completed']} of {b['requests']} requests completed"
            )
    if args.mode == "serve":
        line = res.get("routed_line") or ""
        sizes = (
            {
                int(x)
                for x in re.findall(
                    r"\d+", line.split("batch size", 1)[-1].split("]")[0]
                )
            }
            if "batch size" in line
            else set()
        )
        missing = [m // per_req for m in rows if m // per_req not in sizes]
        if missing:
            fails.append(
                f"batch sizes {missing} not routed to the monokernel forward: {line!r}"
            )
    res["failures"] = fails
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "judge"}, indent=1, default=str
        )
    )
    if "judge" in res:
        for m, s in res["judge"].items():
            if isinstance(s, dict):
                print(
                    f"{m:4d} rows: steps {s['steps']}, res max {s['res_max']:.4f}, x max {s['x_max']:.4f} "
                    f"(any {s['x_max_any']:.4f}), mix max {s['mix_max']:.2e}, flipped {s['flipped']}/{s['tokens']}, "
                    f"out of bounds {len(s['failures'])}"
                )
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
