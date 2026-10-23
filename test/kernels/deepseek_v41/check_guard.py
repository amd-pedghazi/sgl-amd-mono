# SPDX-License-Identifier: Apache-2.0
"""One rank's worker for ``test_guard_multi_rank.py``: a production launch in a loop.

With ``SGLANG_AMD_MONO_FAULT_INJECT=rank:site:after``, that rank's wait at ``site`` times out
once ``after`` launches have run, stops its launch and flags every peer, so each peer's next
guarded wait stops too. Each completed launch appends a line to this rank's progress file. The
first launch runs under the default deadline and ends in a barrier, so the short deadline times
a steady state, not a peer's first compile. ``--launch k2`` drives K2, which carries both
cross-rank reduces; ``--launch ffn`` the FFN launch; ``--mode step`` adds the step's closing TP
collective."""

import argparse
import os

import torch
import torch.distributed as dist

ROWS = 8
# the guard floor is 5 s; a short wait so an injected fault fires fast
SHORT_DEADLINE_S = 6.0


def _no_core():
    """Keep this rank's deliberate GPU fault from dumping a core, whatever the host's
    ``core_pattern``; refuse to run where that fails."""
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
    # a file init method: a port picked in advance can be taken before rank 0 listens on it
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
        # site 1, K2's srdy wait, needs keys: ROWS window keys a token, all at SWA slot 0
        runner.klen[:ROWS].fill_(ROWS)

        def one():
            runner.k2(aw, mw, ROWS, pos, swa, swa_block, None, swa_block)

    else:

        def one():
            runner.ffn(mw, part.clone(), res.clone(), post, comb, pre)

    # a flushed line a launch, so the count survives the process's own stop
    path = os.path.join(args.progress_dir, f"rank{rank}.log")
    with open(path, "a", buffering=1) as progress:
        for i in range(args.launches):
            one()
            if args.mode == "step":
                # the step's closing TP collective, as the logits gather
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
