# SPDX-License-Identifier: Apache-2.0
"""Multi-rank worker for the guarded-wait fail-stop test (RFC #43949 section 8): run a
production launch in a loop on this rank. The harness (test_guard_multi_rank.py) spawns one
worker a rank with an explicit file init method and waits for each on its own, so a
fault on one rank does not take the siblings down with it.

With a fault injected on one rank (SGLANG_AMD_MONO_FAULT_INJECT=rank:site:after, read by the
runner at construction) that rank's wait at ``site``, once ``after`` launches have run, times
out at its deadline, stops the launch with a GPU fault at the null page and flags every peer,
so each peer's next guarded wait stops too and no rank reaches the end. Every completed launch
appends a line to this rank's progress file, so the harness sees how far each rank got before
it stopped; a clean run completes all of them. The first launch runs under the runner's default
deadline and ends in a barrier, so the short deadline below times a steady state rather than one
rank waiting on another's first compile.

``--launch k2`` drives K2 (the attention back and its TP reduce, then the FFN seam, MoE and
all-reduce: both cross-rank reduces); ``--launch ffn`` the FFN launch alone; ``--mode step``
adds the step's closing TP collective, as a serving step does."""

import argparse
import os

import torch
import torch.distributed as dist

ROWS = 8
# the guard floor is 5 s; a short wait so an injected fault fires fast
SHORT_DEADLINE_S = 6.0


def _no_core():
    """PR_SET_DUMPABLE 0 so a deliberate GPU fault writes no core whatever the host
    core_pattern is. These ranks fault on purpose; fail loudly if the call does not take,
    rather than pipe cores to a shared disk."""
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE = 4
        raise SystemExit(
            f"prctl(PR_SET_DUMPABLE, 0) failed: errno {ctypes.get_errno()}"
        )


def _k2_inputs(tp, dev):
    """Correctly shaped K2 attention-back weights and a window-only SWA cache (zeros: this
    test checks fail-stop at a wait, not numerics)."""
    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import Dims
    from sgl_amd_mono.deepseek_v41.gfx950.runner import K2Weights

    d = Dims(tp)
    g = d.groups
    e4m3, u8, f32 = torch.float8_e4m3fn, torch.uint8, torch.float32
    aw = K2Weights(
        woa=torch.zeros(g * 1024, d.group_k, dtype=e4m3, device=dev),
        woa_s=torch.zeros(g * 32, d.group_k // 32, dtype=u8, device=dev),
        wob=torch.zeros(5120, g * 1024, dtype=e4m3, device=dev),
        wob_s=torch.zeros(160, g * 32, dtype=u8, device=dev),
        sink=torch.zeros(d.heads, dtype=f32, device=dev),
        cos_sin=torch.zeros(4096, 64, dtype=f32, device=dev),
    )
    rec = 584  # a V4 KV record: 576 B of data, then 8 B of scales
    page = -(-64 * rec // 576) * 576
    swa = torch.zeros(48, page, dtype=u8, device=dev)
    pos = torch.arange(ROWS, dtype=torch.int64, device=dev)
    return aw, swa, 64, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", default="k2", choices=("ffn", "k2"))
    ap.add_argument("--mode", default="launch", choices=("launch", "step"))
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--init-file", required=True)
    ap.add_argument("--launches", type=int, required=True)
    ap.add_argument("--progress-dir", required=True)
    args = ap.parse_args()
    _no_core()
    rank, tp = args.rank, args.world_size
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    # the harness's explicit init method, a file: a port picked in advance can be taken before
    # rank 0 listens on it. No torchrun, so one rank's fault never signals the others, and the
    # harness waits for each rank on its own
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{args.init_file}",
        rank=rank,
        world_size=tp,
    )
    cpu = dist.new_group(backend="gloo")
    from check_ffn import make_weights, mono_weights, seam_inputs

    from sgl_amd_mono.deepseek_v41.gfx950.runner import DSV41MonoRunner

    # the default deadline covers the first compile
    runner = DSV41MonoRunner(tp, rank, cpu, dev)
    w, _, _ = make_weights(tp, rank, dev, seed=0)
    mw = mono_weights(w)
    del w
    part, res, post, comb, pre = seam_inputs(ROWS, tp, rank, dev)
    logits = torch.ones(ROWS, 256, device=dev)
    if args.launch == "k2":
        aw, swa, swa_block, pos = _k2_inputs(tp, dev)
        runner.res_mid[:ROWS], runner.post_a[:ROWS] = res, post
        runner.comb_a[:ROWS], runner.pre_a[:ROWS] = comb, pre
        # Site 1 is K2's srdy wait and exists only when a token has keys.
        # Give each token ROWS window keys; the zeroed key table points to SWA slot 0.
        runner.klen[:ROWS].fill_(ROWS)

        def one():
            runner.k2(aw, mw, ROWS, pos, swa, swa_block, None, swa_block)

    else:

        def one():
            runner.ffn(mw, part.clone(), res.clone(), post, comb, pre)

    # one flushed line a completed launch, so the count survives the process's own stop: the
    # faulted rank writes `after` of them before it stalls, a clean rank writes every one
    path = os.path.join(args.progress_dir, f"rank{rank}.log")
    with open(path, "a", buffering=1) as progress:
        for i in range(args.launches):
            one()
            if args.mode == "step":
                # the step's closing TP collective (the logits gather)
                dist.all_reduce(logits)
            torch.cuda.synchronize()
            progress.write(f"{i}\n")
            if i == 0:
                # every rank built its launch under the default deadline
                dist.barrier(group=cpu)
                # time the rest under the short deadline
                runner.set_deadline(SHORT_DEADLINE_S)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
