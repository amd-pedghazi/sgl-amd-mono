# SPDX-License-Identifier: Apache-2.0
"""What both Kimi-K3 kernel checks need: the metrics, the synthetic weights and
the MoE reference SGLang's unfused decode path computes.

Adapted from the Kimi-K3 mono decode tests in SGLang (test/registered/amd).
"""

import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

TP = 8  # the launches are built for TP8 only
ROWS = 8  # the x rows fit one CTA's LDS
BF = torch.bfloat16
# The bank the launch is built for, and the live rows a check uses: keeping
# them apart exercises the production bank stride.
MAX_BLOCKS = 8
NVB = 3


def cos(a, b):
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0, eps=1e-12
    ).item()


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


def randn(gen, *shape, scale=1.0, dtype=BF):
    t = torch.randn(*shape, generator=gen, device=gen.device, dtype=torch.float32)
    return (t * scale).to(dtype)


def all_reduce_fp32(t):
    """The ranks' bf16 partials summed in fp32, as the in-kernel reduce does."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(BF)


def rms(x, weight, eps):
    """RMSNorm over the last dimension, in fp32 as the kernel computes it."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def situ(g, u):
    g, u = g.float(), u.float()
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return (a * (25.0 * torch.tanh(u * 0.04))).to(BF)


def _expert_bytes(gen, device, *, guint):
    """The routed experts in one of aiter's two A16W4 layouts.

    The logical bytes come from a fixed generator, so the two layouts hold the
    same weights and have to give the same MoE output. Scale bytes stay near
    E8M0 1.0: the exponent is raw with bias 127, and uniform bytes would reach
    2^128 and make the layer non-finite.
    """
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import LAT, RI, E

    def u8(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=gen, device=device, dtype=torch.int32
        ).to(torch.uint8)

    return {
        "w13": shuffle_weight_a16w4(u8(E, 2 * RI, LAT // 2), 16, guint),
        "w13s": shuffle_scale_a16w4(
            u8(E, 2 * RI, LAT // 32, lo=118, hi=123).view(-1, LAT // 32), E, guint
        ),
        "w2": shuffle_weight_a16w4(u8(E, LAT, RI // 2), 16, False),
        "w2s": shuffle_scale_a16w4(
            u8(E, LAT, RI // 32, lo=118, hi=123).view(-1, RI // 32), E, False
        ),
    }


def moe_weights(rank, device, *, guint):
    """The MoE half's weights. The router, the latent down and the latent up are
    replicated across ranks; the shared and routed experts are TP shards."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import HIDDEN, LAT, SI, UP_N, E

    gc = torch.Generator(device=device).manual_seed(7)
    gr = torch.Generator(device=device).manual_seed(100 + rank)
    ge = torch.Generator(device=device).manual_seed(100 + rank)

    w_up = randn(gc, HIDDEN, LAT, scale=0.02)
    out = {
        "w_gate": randn(gc, E, HIDDEN, scale=0.02),
        "bias": randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(BF),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": randn(gr, HIDDEN, SI, scale=0.03),
    }
    out.update(_expert_bytes(ge, device, guint=guint))
    return out


def moe_reference(x, w, rank):
    """The MoE as SGLang's unfused decode path computes it."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import SI, TOPK, UP_N

    sig = torch.sigmoid(torch.mm(x.float(), w["w_gate"].float().t()))
    ids = torch.topk(sig + w["bias"], TOPK, dim=-1, sorted=True).indices
    wts = sig.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    gu = F.linear(x, w["w_sgu"])
    shared = F.linear(situ(gu[:, :SI], gu[:, SI:]), w["w_sd"])
    # The kernel takes the experts as raw bytes. That is how SGLang holds
    # them. fused_moe dispatches on the dtype and on an is_shuffled marker,
    # and a view drops that marker. So tag the same storage, for the
    # reference only.
    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = w["w13"].view(fp4), w["w2"].view(fp4)
    w13.is_shuffled = w2.is_shuffled = True
    y = fused_moe(
        F.linear(x, w["w_ld"]),
        w13,
        w2,
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Situv2,
        QuantType.per_1x32,
        False,
        w["w13s"],
        w["w2s"],
        None,
        None,
        dtype=BF,
        gate_mode="separated",
        beta=4.0,
        linear_beta=25.0,
    )
    lat = all_reduce_fp32(y).float()
    ln = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5) * w["ln_w"]
    lo = rank * UP_N
    up = ln.to(BF).float() @ w["w_up_shard"].float().t()
    shared[:, lo : lo + UP_N] = (shared[:, lo : lo + UP_N].float() + up).to(BF)
    return all_reduce_fp32(shared)


def setup(port_offset: int = 0):
    """This rank's device and the gloo group the peer handshake needs."""
    import os

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    return rank, torch.device("cuda", rank), dist.new_group(backend="gloo")


def finish(fails, dev):
    """Print this rank's failures, then exit nonzero on every rank if any failed."""
    if fails:
        print(f"rank {dist.get_rank()} FAILED: {fails}", flush=True)
    bad = torch.tensor([int(bool(fails))], device=dev)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print("ALL PASS" if not bad.item() else "FAILED", flush=True)
    dist.destroy_process_group()
    if bad.item():
        sys.exit(1)
