# SPDX-License-Identifier: Apache-2.0
"""Kernel-only time of the Kimi-K3 layer launch (K2) at TP8, synthetic weights.

    torchrun --nproc-per-node 8 test/kernels/kimi_k3/time_k2.py --rows 6 --flush 1024

Each iteration bumps the step epoch, optionally writes a ``--flush`` MB buffer
first (1024 evicts L2 and MALL: serving reads a layer's weights cold), and times
one eager launch with events. Prints every rank's median and the max over ranks
(the step waits for the slowest rank).
"""

import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
import check_common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--flush",
        type=int,
        default=1024,
        help="MB written before each launch, 0 = warm",
    )
    ap.add_argument(
        "--fold", type=int, default=1, help="K2Build.fold (1: the serving form)"
    )
    ap.add_argument(
        "--sync",
        type=int,
        default=0,
        help="1: an all-reduce right before K2 in its graph, so ranks start together",
    )
    ap.add_argument(
        "--opt",
        type=int,
        default=-1,
        help="K2Build.opt (the OPT_* bits; -1: K2_OPT, as served)",
    )
    ap.add_argument(
        "--timeline",
        default="",
        help="dir: a timeline build, its last dumps saved as r<rank>_<i>.pt",
    )
    args = ap.parse_args()
    check_common.ROWS = args.rows
    import check_k2  # noqa: E402 -- reads ROWS at import

    rank, dev, cpu = check_common.setup()
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2

    peers = PeerBuffer(k2.peer_alloc_bytes(), cpu, rank, check_common.TP, dev)
    k2.init_peer(peers.bytes)
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(30.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    scratch = torch.zeros(k2.scratch_alloc_bytes(), dtype=torch.uint8, device=dev)
    w = check_common.moe_weights(rank, dev, guint=False)
    w.update(check_k2.layer_weights(rank, dev))
    junk = torch.empty(max(args.flush, 1) << 20, dtype=torch.uint8, device=dev)

    from sgl_amd_mono.common.plan import BLOCKS

    tl = (
        torch.zeros(BLOCKS, k2.TL_POINTS, dtype=torch.int64, device=dev)
        if args.timeline
        else None
    )

    def launch():
        check_k2.run(
            k2,
            w,
            scratch,
            reset=False,
            peers=peers,
            epoch=epoch,
            rank=rank,
            layer=1,
            guint=False,
            fold=bool(args.fold),
            tl=tl,
            opt=k2.K2_OPT if args.opt < 0 else args.opt,
        )

    # Graph replays, as serving runs it: an eager @flyc.jit launch rebuilds its
    # cache key on the host (~150 us at K2's argument count), which events
    # around an eager call would time. One graph bumps the epoch (and flushes),
    # the other is the launch alone.
    epoch[0].add_(1)
    launch()  # compile, outside any capture
    torch.cuda.synchronize()
    pre, g = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(pre):
        if args.flush:
            junk.add_(1)
        epoch[0].add_(1)
    one = torch.ones(1, device=dev)
    if args.sync:
        dist.all_reduce(one)  # the communicator, outside the capture
        torch.cuda.synchronize()
    with torch.cuda.graph(g):
        if args.sync:
            # ranks leave this together: K2's waits then measure its own chain,
            # not the replays' start skew (events then include the all-reduce)
            dist.all_reduce(one)
        launch()
    times = []
    for it in range(args.iters + 5):
        pre.replay()
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        if it >= 5:
            times.append(a.elapsed_time(b) * 1e3)
        if tl is not None and it >= args.iters:
            os.makedirs(args.timeline, exist_ok=True)
            torch.save(tl.cpu(), f"{args.timeline}/r{rank}_{it}.pt")
            tl.zero_()
    med = statistics.median(times)
    out = [None] * check_common.TP
    dist.all_gather_object(out, med, group=cpu)
    if rank == 0:
        print(
            f"K2 rows={args.rows} fold={args.fold} opt={args.opt} flush={args.flush}MB: "
            "per-rank median us "
            + " ".join(f"{t:.1f}" for t in out)
            + f" | max {max(out):.1f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
