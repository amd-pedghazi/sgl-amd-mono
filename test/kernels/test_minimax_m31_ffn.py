# SPDX-License-Identifier: Apache-2.0
"""The M3.1 FFN launch against native ops on one GPU (TP=1 build): o_proj (AITER's
per-token FP8 quant, SGLang's rowwise-FP8 weight), the seam residual, Gemma
post-attention norm, FP32 router, top-4, the routed A16W4 experts against AITER on
SGLang's post-load MXFP4 layout (AITER (16, 16) shuffle, shuffled E8M0 group-32 scales,
separated gate / up), and the MXFP8 shared expert against torch. Adapted from vLLM's
tests/models/minimax_m3/amd_specialized_decode/test_ops.py (Apache-2.0)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl")
if (
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
):
    pytest.skip("gfx950 only", allow_module_level=True)

from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.minimax_m31 import a16w8, ffn  # noqa: E402
from sgl_amd_mono.minimax_m31.runner import FfnRunner  # noqa: E402

R = ffn.EXPERTS
H, INTER = ffn.HIDDEN, ffn.INTER


def _metrics(got, want):
    got, want = got.float().flatten(), want.float().flatten()
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0).item()
    rel = ((got - want).norm() / want.norm()).item()
    return cos, rel


@pytest.fixture(scope="module")
def weights():
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)

    def codes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scales(*shape):
        # e8m0 codes around 2^-7: gate / up pre-activations of a few units
        return torch.randint(
            119, 123, shape, dtype=torch.uint8, device=dev, generator=g
        )

    def kernel_format(w, s):
        # what SGLang's process_requantized_weights leaves for aiter's fused MoE
        rows = w.shape[1]
        s = e8m0_shuffle(s.view(R * rows, -1)).view(R, rows, -1)
        w = shuffle_weight(w.contiguous(), (16, 16)).view(torch.float4_e2m1fn_x2)
        w.is_shuffled = True
        return w, s

    w13, s13 = kernel_format(codes(R, 2 * INTER, H // 2), scales(R, 2 * INTER, H // 32))
    w2, s2 = kernel_format(codes(R, H, INTER // 2), scales(R, H, INTER // 32))

    def mxfp8(rows, k):
        q = torch.randn(rows, k, device=dev, generator=g).mul(2).to(torch.float8_e4m3fn)
        return q, scales(rows, k // 32)

    gate_up, gate_up_scale = mxfp8(2 * INTER, H)
    down, down_scale = mxfp8(H, INTER)
    w13s, s13s = a16w8.layout(gate_up, gate_up_scale)
    w2s, s2s = a16w8.layout(down, down_scale)
    w = SimpleNamespace(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        w13s=w13s,
        s13s=s13s,
        w2s=w2s,
        s2s=s2s,
        shared=SimpleNamespace(
            gate_up=gate_up,
            gate_up_scale=gate_up_scale,
            down=down,
            down_scale=down_scale,
        ),
        post_norm=torch.randn(H, device=dev, generator=g).mul(0.1).bfloat16(),
        router=torch.randn(ffn.EXPERTS, H, device=dev, generator=g).mul(0.02),
        router_bias=torch.randn(ffn.EXPERTS, device=dev, generator=g).mul(0.01),
    )
    # o_proj as SGLang keeps it for decode on gfx950: rowwise FP8, (16, 16) shuffled
    dense = torch.randn(H, ffn.ATTN_K, device=dev, generator=g).mul(0.02)
    w.wos = dense.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
    w.oproj = (dense / w.wos).clamp(-448, 448).to(torch.float8_e4m3fn)
    w.wo = shuffle_weight(w.oproj, (16, 16))
    return w


def _native_experts(x, w, ids, wts):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    return fused_moe(
        x,
        w.w13,
        w.w2,
        wts,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w.s13,
        w2_scale=w.s2,
        gate_mode="separated",
        swiglu_limit=ffn.LIMIT,
    )


def _native_shared(x, w):
    """The shared expert on its MXFP8 values, MID in bf16 as the launch keeps it."""

    def values(q, scale):
        return q.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)

    h = x.float() @ values(w.gate_up, w.gate_up_scale).T
    gate = h[:, :INTER].clamp(max=ffn.LIMIT)
    up = h[:, INTER:].clamp(-ffn.LIMIT, ffn.LIMIT)
    mid = (gate * torch.sigmoid(gate * ffn.ALPHA) * (up + ffn.BETA)).bfloat16()
    return mid.float() @ values(w.down, w.down_scale).T


def _native_oproj(attn, w):
    """AITER's per-token FP8 quant, then the rowwise-FP8 GEMM in f32."""
    import aiter

    q, xs = aiter.per_token_quant_hip(attn, quant_dtype=aiter.dtypes.fp8)
    return ((q.float() @ w.oproj.float().T) * xs * w.wos.T).bfloat16()


def _native_norm(part, residual, weight):
    total = part.float() + residual.float()
    var = total.pow(2).mean(-1, keepdim=True)
    x = total * torch.rsqrt(var + ffn.EPS) * (1.0 + weight.float())
    return x.bfloat16(), total.bfloat16()


def _native_route(logits, bias):
    scores = logits.sigmoid()
    ids = (scores + bias).topk(ffn.TOPK, dim=-1).indices
    wts = scores.gather(1, ids)
    wts = wts / wts.sum(-1, keepdim=True) * ffn.ROUTED_SCALE
    return ids.int(), wts


def _clear_choices(logits, bias, margin):
    top = (logits.sigmoid() + bias).topk(ffn.TOPK + 1, dim=-1).values
    return (top[:, ffn.TOPK - 1] - top[:, ffn.TOPK]) > margin


def _pairs(region, n):
    return region[: n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()


def _pushed(runner, s):
    """The launch's o_proj partial, from the triples it pushed (the first launch's
    parity half, this rank's rows)."""
    triples = runner.peer_bufs[0][: s * ffn.TRIPLES * 8].view(torch.int64)
    bits = torch.stack([(triples >> (16 * k)) & 0xFFFF for k in range(3)], -1)
    return bits.to(torch.int16).view(torch.bfloat16).view(s, H)


def _runner(weights, s):
    dev = torch.device("cuda")
    buf = torch.zeros(ffn.peer_bytes(1), dtype=torch.uint8, device=dev)
    peers = torch.tensor([buf.data_ptr()], dtype=torch.int64, device=dev)
    runner = FfnRunner(1, 0, buf.data_ptr(), peers, dev, (s,))
    runner.warm(weights)
    runner.peer_bufs = [buf]
    return runner


def _run(weights, s, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    attn = torch.randn(s, ffn.ATTN_K, device="cuda", generator=g).mul(3).bfloat16()
    residual = torch.randn(s, H, device="cuda", generator=g).mul(4).bfloat16()
    runner = _runner(weights, s)
    slots = torch.arange(s, dtype=torch.int64, device="cuda")
    out, res_out = runner(attn, residual, weights, slots)
    torch.cuda.synchronize()
    err = runner.epoch[list(guard.ERROR_WORDS)].tolist()
    assert not any(err), err
    x = runner.region(s, "xn")[: s * H * 2].view(torch.bfloat16).view(s, H)
    parts = _pairs(runner.region(s, "logit"), s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s, ffn.EXPERTS, ffn.ROUTER_PARTS)
    route = _pairs(runner.region(s, "route"), s * 2 * ffn.TOPK).view(s, 2 * ffn.TOPK)
    return SimpleNamespace(
        attn=attn,
        part=_pushed(runner, s),
        residual=residual,
        out=out,
        res_out=res_out,
        x=x,
        logits=parts.sum(-1),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
    )


@pytest.mark.parametrize("s", [1, 3, 8, 16])
def test_ffn_stages_match_native(weights, s):
    r = _run(weights, s)
    cos, rel = _metrics(r.part, _native_oproj(r.attn, weights))
    assert cos > 0.99999 and rel < 2e-3, (cos, rel)
    want_x, want_res = _native_norm(r.part, r.residual, weights.post_norm)
    torch.testing.assert_close(r.res_out, want_res, rtol=0, atol=0)
    cos, rel = _metrics(r.x, want_x)
    assert cos > 0.99999 and rel < 4e-3, (cos, rel)

    want_logits = r.x.float() @ weights.router.T
    torch.testing.assert_close(r.logits, want_logits, rtol=1e-4, atol=1e-4)

    ids, wts = _native_route(r.logits, weights.router_bias)
    clear = _clear_choices(r.logits, weights.router_bias, 1e-4)
    assert clear.any()
    got_ids, order = r.ids.sort(-1)
    want_ids, want_order = ids.sort(-1)
    assert torch.equal(got_ids[clear], want_ids[clear])
    torch.testing.assert_close(
        r.wts.gather(1, order)[clear],
        wts.gather(1, want_order)[clear],
        rtol=1e-5,
        atol=1e-6,
    )

    want = _native_experts(r.x, weights, r.ids, r.wts).float()
    want = want + _native_shared(r.x, weights.shared)
    cos, rel = _metrics(r.out, want)
    assert cos > 0.9999 and rel < 1e-2, (cos, rel)


@pytest.mark.parametrize("s", [8, 16])
def test_o_proj_is_sglangs_to_the_bit(weights, s):
    """The launch's o_proj partial is SGLang's (AITER's per-token FP8 quant, then
    ``smallm_fp8_gemm``) bit for bit, in both of that GEMM's wave splits."""
    import aiter

    smallm = pytest.importorskip("sglang.kernels.ops.gemm.smallm_fp8_gfx950")
    if not smallm.smallm_fp8_gemm_enabled():
        pytest.skip("smallm_fp8_gemm is off")
    r = _run(weights, s)
    q, xs = aiter.per_token_quant_hip(r.attn, quant_dtype=aiter.dtypes.fp8)
    want = smallm.smallm_fp8_gemm(q, weights.wo, xs, weights.wos, torch.bfloat16)
    assert torch.equal(r.part, want)


_STALLED_LAUNCH = """
import torch
import test_minimax_m31_ffn as t
from sgl_amd_mono.minimax_m31 import ffn
from sgl_amd_mono.minimax_m31.runner import FfnRunner

w = t.weights._get_wrapped_function()()
dev = torch.device("cuda")
buf = torch.zeros(ffn.peer_bytes(1), dtype=torch.uint8, device=dev)
peers = torch.tensor([buf.data_ptr()], dtype=torch.int64, device=dev)
runner = FfnRunner(1, 0, buf.data_ptr(), peers, dev, (1,), drop_mark=5)
runner.set_deadline(0.2)
runner.warm(w)
attn = torch.randn(1, ffn.ATTN_K, device=dev).bfloat16()
res = torch.randn(1, ffn.HIDDEN, device=dev).bfloat16()
runner(attn, res, w, torch.zeros(1, dtype=torch.int64, device=dev))
torch.cuda.synchronize()
print("LAUNCH_RETURNED", flush=True)
"""


def test_a_timed_out_wait_stops_the_launch():
    """A CTA that never marks the epoch barrier must stop the launch at the deadline
    (the guard's store to the null page), not let the step return."""
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([os.path.dirname(__file__), *sys.path]),
        # the stop faults the GPU on purpose: no core dump of its VRAM
        "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _STALLED_LAUNCH],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    log = proc.stdout + proc.stderr
    assert "LAUNCH_RETURNED" not in log and proc.returncode != 0, log[-2000:]


def test_a_tag_cycle_ends_with_its_receive_half_cleared(weights):
    """Across the 16-bit triple tags' wrap every launch matches one at epoch 0, and
    the last launch of each parity's cycle leaves that parity's receive half zero
    (a stale pair there could otherwise carry the next cycle's tags)."""
    s = 16
    g = torch.Generator(device="cuda").manual_seed(3)
    attn = torch.randn(s, ffn.ATTN_K, device="cuda", generator=g).bfloat16()
    residual = torch.randn(s, H, device="cuda", generator=g).mul(4).bfloat16()
    slots = torch.arange(s, dtype=torch.int64, device="cuda")
    want, want_res = _runner(weights, s)(attn, residual, weights, slots)

    runner = _runner(weights, s)
    half = ffn.peer_half_bytes(1)
    last = 2 * (ffn.TAG16_CYCLE - 1)  # parity 0's last launch of the first cycle
    runner.epoch[0] = last
    for ep in range(last, last + 4):
        out, res_out = runner(attn, residual, weights, slots)
        torch.cuda.synchronize()
        assert torch.equal(out, want) and torch.equal(res_out, want_res), ep
        if ep < last + 2:  # a cycle's end: the parity's half was cleared
            p = ep & 1
            assert not runner.peer_bufs[0][p * half : (p + 1) * half].any(), ep
    assert runner.epoch[0].item() == last + 4
