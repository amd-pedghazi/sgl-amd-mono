# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 MoE launch against SGLang's unfused decode path, at TP8.

This launch runs in no forward: the layer launch contains the same stages, and
this is the reference that isolates them. When ``check_k2``'s ``out`` fails and
this passes, the fault is in the seam, not the MoE.

Both of aiter's A16W4 expert layouts run. They hold the same logical weights,
so they have to give the same output. Reading the bytes with the wrong layout
does not fail. It returns wrong numbers.
"""

import argparse

import torch
from check_common import (
    ROWS,
    TP,
    cos,
    finish,
    moe_reference,
    moe_weights,
    randn,
    rel,
    setup,
)

from sgl_amd_mono.kimi_k3.spec import NUMERICS


def run(moe, w, x, *, peers, epoch, rank, layer, guint):
    out = torch.empty_like(x)
    key = moe.MoeBuild(tokens=x.size(0), guint=guint)
    moe.moe(
        key,
        x=x,
        w_gate=w["w_gate"],
        bias=w["bias"],
        w_ld=w["w_ld"],
        w_sgu=w["w_sgu"],
        w_sd=w["w_sd"],
        w13=w["w13"],
        w13s=w["w13s"],
        w2=w["w2"],
        w2s=w["w2s"],
        ln_w=w["ln_w"],
        w_up=w["w_up_shard"],
        out=out,
        scratch=torch.zeros(moe.scratch_bytes(key), dtype=torch.uint8, device=x.device),
        queue=torch.zeros(moe.QUEUE_BYTES // 4, dtype=torch.int32, device=x.device),
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        layer=layer,
    )
    torch.cuda.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 4, ROWS])
    args = ap.parse_args()

    rank, dev, cpu = setup()
    import torch.distributed as dist

    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.stages import moe as k2b
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import HIDDEN

    peers = PeerBuffer(k2b.peer_bytes(ROWS), cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=dev)

    fails, outs = [], {}
    # aiter's fused_moe reads the plain layout, so it is the reference for that
    # one only. The interleaved layout is checked against the plain kernel
    # output: the two hold the same logical weights.
    for guint in (False, True):
        w = moe_weights(rank, dev, guint=guint)
        for rows in args.widths:
            x = randn(torch.Generator(device=dev).manual_seed(11 + rows), rows,
                      HIDDEN, scale=0.3)  # fmt: skip
            epoch.add_(1)
            out = run(
                k2b, w, x, peers=peers, epoch=epoch, rank=rank,
                layer=1, guint=guint,
            )  # fmt: skip
            gathered = [torch.empty_like(out) for _ in range(TP)]
            dist.all_gather(gathered, out.contiguous())
            checks = {
                "finite": bool(torch.isfinite(out).all()),
                "ranks agree": all(torch.equal(gathered[0], g) for g in gathered[1:]),
            }
            if not guint:
                ref = moe_reference(x, w, rank)
                checks["out"] = cos(out, ref) > NUMERICS.bound(
                    "moe.out.cosine"
                ) and rel(out, ref) < NUMERICS.bound("moe.out.rel")
                if rank == 0:
                    print(
                        f"rows={rows}: cos {cos(out, ref):.6f} rel {rel(out, ref):.4g}",
                        flush=True,
                    )
            fails += [
                f"guint{int(guint)}/{rows}/{n}" for n, ok in checks.items() if not ok
            ]
            outs.setdefault(rows, {})[guint] = out.clone()

    # Reading the expert bytes with the wrong interleave does not fail. It
    # returns wrong numbers. So the two layouts agreeing is the only check
    # the interleaved one gets.
    for rows, by_layout in outs.items():
        c = cos(by_layout[False], by_layout[True])
        if rank == 0:
            print(
                f"rows={rows}: the two expert layouts agree to cos {c:.6f}", flush=True
            )
        if c <= NUMERICS.bound("moe.out.cosine"):
            fails.append(f"{rows}: the two expert layouts disagree")

    peers.close()
    finish(fails, dev)


if __name__ == "__main__":
    main()
