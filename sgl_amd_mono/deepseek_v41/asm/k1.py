# SPDX-License-Identifier: Apache-2.0
"""K1, the attention half of a DeepSeek-V4.1 decoder layer at 17 to 256 rows, on the asm
kernels of ``k1_runner`` (vllm-project/vllm#65421), on SGLang's caches and metadata.

``AsmK1.__call__`` takes what the monokernel's K1 takes, the previous layer's pending FFN
post (x, residual, post, comb) and pre, and returns what K2 takes: the all-reduced
attention output, the residual with that post applied, and the attention's post, comb
and pre mixes. Inside, in order:

- ``seam_v3``: the post-mix, the pre-collapse with the previous pre, MXFP8 of the
  collapse times the attention norm (the row's rstd is folded in downstream);
- ``front_v3``: wqkv (its extra waves finish the seam's mixes), then wq_b with the q
  RoPE, and the kv norm, RoPE and fp8_ds_mla record written to the SWA cache at ``slot``;
- on an index source, its index step (``sglang_index``: SGLang's compressor and decode
  indexer) on the normed input and qr the front leaves, which rewrites the layer's top-k;
- ``attend``: the sparse attention over the token's SWA window and top-k keys;
- ``back_ar``: wo_a, then wo_b with all-reduce #1 in one launch (``WoBAllReduce``).

SGLang's key lists feed the attention without a copy. Token t's window is row t of
``swa_page_indices`` (128 slots, newest first, -1 past ``swa_topk_lengths[t]``), so its
indptr is 128 t: the kernel reads all 128 and skips the negative ones. Its top-k is row t
of ``sparse_page_indices(ratio)`` (512 slots) with the count in
``sparse_topk_lengths(ratio)``, the kernel's fixed-stride mode. A window-only layer
(ratio 0: layers 0 and 1) reads no compressed keys.

The asm indexer (vLLM's ``idx_logits``, ``front_v3(idx=True)``'s MXFP4 query) stays out:
it reads AITER's preshuffled index pages, not SGLang's split plane. So one launch turns
the front's scratch into the inputs SGLang's own chain gives its compressor and indexer,
the normed input and qr (the q norm's output), both bf16. v3: the seam keeps
y' = bf16(x1 · attn_norm.w) without rstd, so normed = bf16(y' · rstd); wqkv32n keeps no
bf16 q latent, only wq_b's MXFP8 x = bf16(q') · q_norm.w and the 40 partial sums of q'²,
so qr = bf16(x · rr), rr = rstd · rsq(rstd² Σ / 1280 + eps): vLLM's v3 indexer reads the
same MXFP8 q. v1: normed is the fin's y, and qr the q norm of wqkv's bf16 q latent.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sgl_amd_mono.deepseek_v41 import HF_CONFIG
from sgl_amd_mono.deepseek_v41.asm.k1_runner import HEADS, MAX_M, H

# by file under sglang/ (``f(p)``: a parameter, ``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    "srt/layers/quantization/fp8_utils.py": ("unshuffle_aiter_fp8_weight",),
    "srt/layers/quantization/fp8_hip.py": ("mxfp8_aiter_preshuffled",),
    "srt/models/deepseek_v4.py": (
        "self_attn",
        "input_layernorm",
        "rms_norm_eps",
        "hc_eps",
        "hc_sinkhorn_iters",
        "wqkv_a",
        "wq_b",
        "wo_b",
        "eps",
        "softmax_scale",
    ),
    "srt/layers/layernorm.py": ("variance_epsilon",),
    # an index source's compressor and decode indexer, between the front and the attention
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/deepseek_v4_backend.py": (
        "forward_low_ratio_sources(layer)",
        "forward_low_ratio_sources(x)",
        "forward_low_ratio_sources(q_lora)",
        "forward_low_ratio_sources(positions)",
        "forward_low_ratio_sources(forward_batch)",
    ),
}

HC = HF_CONFIG["hc_mult"]
SWA_WIDTH = HF_CONFIG["sliding_window"]  # slots a row of swa_page_indices holds
TOPK = HF_CONFIG["index_topk"]
Q_RANK = 1280
QKV = 1792  # the q latent's 1280 columns, then the 512 of kv
RECORD = 584  # fp8_ds_mla: 448 e4m3, 64 bf16, 7 scale bytes and a pad byte
# SGLang's mHC kernels fix the post mix's scale: post = 2 sigmoid(.)
POST_ALPHA = 2.0


@triton.jit
def _pow2(n):
    """2^(n - 127) for integer n in 1 .. 254, exactly (the f32 with exponent field n)."""
    return (n << 23).to(tl.float32, bitcast=True)


@triton.jit
def _index_rows_v3(y, rstd, ss, qx, qs, normed, qr, eps, H_: tl.constexpr,
                   QR: tl.constexpr, NSS: tl.constexpr, BH: tl.constexpr,
                   BQ: tl.constexpr):  # fmt: skip
    """Row t of an index source's normed input and qr on v3 (the module docstring's
    rules): wq_b's x read in gemv8.frag_x's order, [j][kg][h][q][r][16 B] codes and
    [j][kg][q32][r] E8M0 scales (token t = 16 j + r, column 128 kg + 64 h + 16 q + b)."""
    t = tl.program_id(0).to(tl.int64)
    rd = tl.load(rstd + t)
    for c in tl.static_range(0, H_, BH):
        o = c + tl.arange(0, BH)
        v = tl.load(y + t * H_ + o).to(tl.float32) * rd
        tl.store(normed + t * H_ + o, v.to(tl.bfloat16))
    p = tl.arange(0, 64)
    sq = tl.sum(tl.load(ss + t * NSS + p, mask=p < NSS, other=0.0), axis=0)
    rr = rd / tl.sqrt(rd * rd * (sq / QR) + eps)
    k = tl.arange(0, BQ)
    m = k < QR
    tile = (t // 16) * (QR // 128) + k // 128
    r = t % 16
    off = (((tile * 2 + (k // 64) % 2) * 4 + (k // 16) % 4) * 16 + r) * 16 + k % 16
    code = tl.load(qx + off, mask=m, other=0).to(tl.int32)
    sc = tl.load(qs + (tile * 4 + (k // 32) % 4) * 16 + r, mask=m, other=127)
    e, mant = (code >> 3) & 15, code & 7
    # e4m3 (OCP): (8 + m) 2^(e - 10), subnormal m 2^-9; the quantizer clamps, so no NaN
    v = tl.where(
        e == 0,
        mant.to(tl.float32) * 0.001953125,
        (mant + 8).to(tl.float32) * _pow2(e + 117),
    )
    v = tl.where((code & 128) != 0, -v, v) * _pow2(sc.to(tl.int32))
    tl.store(qr + t * QR + k, (v * rr).to(tl.bfloat16), mask=m)


@triton.jit
def _index_qr_v1(q, qw, qr, eps, QKV_: tl.constexpr, QR: tl.constexpr,
                 BQ: tl.constexpr):  # fmt: skip
    """Row t of qr on v1: the q norm of wqkv's bf16 q latent, as qkv_norm computes it,
    bf16((x · rsq(Σx² / 1280 + eps)) · w) in f32."""
    t = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BQ)
    m = k < QR
    v = tl.load(q + t * QKV_ + k, mask=m, other=0.0).to(tl.float32)
    rr = 1.0 / tl.sqrt(tl.sum(v * v, axis=0) / QR + eps)
    wq = tl.load(qw + k, mask=m, other=0.0).to(tl.float32)
    tl.store(qr + t * QR + k, (v * rr * wq).to(tl.bfloat16), mask=m)


def sglang_index(L, positions, forward_batch):
    """Layer ``L``'s index step for ``AsmK1.__call__`` (an index source, ``L.index``):
    SGLang's compressor and decode indexer on K1's normed input and qr, as SGLang runs
    them before the attention (``forward_low_ratio_sources``). They write this step's
    compressed keys and rewrite the ratio's ``sparse_page_indices`` in place, which the
    attention then reads. Graph-capturable as SGLang's decode graphs capture it."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend, attn = get_attn_backend(), L.module.self_attn

    def index(normed: torch.Tensor, qr: torch.Tensor) -> None:
        backend.forward_low_ratio_sources(
            layer=attn,
            x=normed,
            q_lora=qr,
            positions=positions,
            forward_batch=forward_batch,
        )

    return index


def _check_cache(tag: str, cache: torch.Tensor, block: int) -> None:
    """Raise unless ``cache`` is uint8 pages of ``block`` V4 records (the kernels address a
    record by shifts: ``block`` a power of two)."""
    pages = cache.view(cache.shape[0], -1)
    if not (
        cache.dtype == torch.uint8
        and pages.stride(1) == 1
        and pages.stride(0) >= block * RECORD
        and block > 0
        and block & (block - 1) == 0
    ):
        raise RuntimeError(
            f"asm K1 {tag}: {cache.dtype} {tuple(cache.shape)} {cache.stride()}, want "
            f"uint8 pages of {block} (a power of two) x {RECORD} B"
        )


def _row_major(weight: torch.Tensor, linear) -> torch.Tensor:
    """An FP8 linear's weight [N, K] row-major: undone where SGLang's MXFP8 aiter route
    stored it in aiter's (16, 16) preshuffle (a copy, freed after the frag copy)."""
    if not getattr(linear, "mxfp8_aiter_preshuffled", False):
        return weight
    from sglang.srt.layers.quantization.fp8_utils import unshuffle_aiter_fp8_weight

    return unshuffle_aiter_fp8_weight(weight)


class AsmK1:
    """The process's K1 on the asm kernels: one ``K1Runner`` (its scratch serves every
    layer, which run one after another on one stream) and the all-reduce #1 launch
    (``ar_runner.WoBAllReduce``, collective; None for checks of the front alone).

    ``chain`` "v3" (vLLM's): the seam quantizes the norm's input before the row's rstd is
    known and the q norm's rrms is folded into the attention, 3 launches before it; "v1":
    the seam's fin launch normalizes first and qkv_norm normalizes q and kv, 5 launches."""

    def __init__(self, runner, wob=None, chain: str = "v3"):
        if chain not in ("v3", "v1") or (chain == "v3" and not runner.v3):
            raise ValueError(f"asm K1 chain {chain!r} on a K1Runner(v3={runner.v3})")
        self.run, self.wob, self.v3 = runner, wob, chain == "v3"
        dev = runner.dev
        self.sw_ptr = torch.arange(
            0, (MAX_M + 1) * SWA_WIDTH, SWA_WIDTH, dtype=torch.int32, device=dev
        )
        # a settled residual (no post owed): the seam's post-mix as the identity
        self.zero_x = torch.zeros(MAX_M, H, dtype=torch.bfloat16, device=dev)
        self.zero_post = torch.zeros(MAX_M, HC, dtype=torch.float32, device=dev)
        # an index source's normed input and qr (shared by the layers, as the scratch)
        self.normed = torch.empty(MAX_M, H, dtype=torch.bfloat16, device=dev)
        self.qr = torch.empty(MAX_M, Q_RANK, dtype=torch.bfloat16, device=dev)
        self.index_rows: tuple | None = None

    def weights(self, L) -> dict:
        """Layer ``L``'s kernel tensors (the spec's ``_Layer``, after ``_weights``): the
        frag-shuffled copies of wqkv, wq_b, wo_a and wo_b (about 40 MB) from the e4m3
        weights and E8M0 codes the monokernel reads, row-major again where SGLang
        preshuffled them; the seam's mHC tensors, norms, sinks and RoPE table as the
        monokernel keeps them. None of the indexer's: SGLang's chain runs it."""
        layer, a, k1, k2 = L.module, L.module.self_attn, L.k1, L.k2
        if k1.wqkv.shape != (1792, H) or k1.wqb.shape != (HEADS * 512, 1280):
            raise RuntimeError(
                f"asm K1 layer {L.i}: wqkv {tuple(k1.wqkv.shape)}, wq_b "
                f"{tuple(k1.wqb.shape)}: needs the fused wqkv_a and {HEADS} local heads"
            )
        t = dict(
            hc_fn=k1.hc_fn,
            hc_scale=k1.hc_scale,
            hc_base=k1.hc_base,
            norm=k1.norm,
            rms_eps=layer.rms_norm_eps,
            hc_eps=layer.hc_eps,
            post_alpha=POST_ALPHA,
            norm_eps=layer.input_layernorm.variance_epsilon,
            sinkhorn=layer.hc_sinkhorn_iters,
            q_norm=k1.q_norm,
            kv_norm=k1.kv_norm,
            sink=k2.sink,
            cos_sin=k1.cos_sin,
            scale=a.softmax_scale,
            eps=a.eps,
            wqkv=(_row_major(k1.wqkv, a.wqkv_a), k1.wqkv_s),
            wq_b=(_row_major(k1.wqb, a.wq_b), k1.wqb_s),
            wo_a=(k2.woa, k2.woa_s),
            wo_b=(_row_major(k2.wob, a.wo_b), k2.wob_s),
        )
        w = self.run.weights(t, v3=self.v3)
        w["index_eps"] = float(a.eps)  # the q norm's, as SGLang's q_norm has it
        return w

    def front(self, w, x, residual, post, comb, prev_pre, pos, slot, swa, swa_block,
              index=False):  # fmt: skip
        """The seam and the front: the step's SWA records written at ``slot`` (int64,
        -1: none), and (q [M, 16, 512] bf16, a view of the runner's scratch, on v3
        without q_norm's rrms; residual'; post [M, 4, 1], comb [M, 4, 4], pre [M, 4] f32).
        ``index``: also the index step's inputs, ``index_rows`` = (normed [M, 5120], qr
        [M, 1280]) bf16."""
        run, M = self.run, residual.shape[0]
        # the kv wave reads both as int64 (an int32 slot list would address other records)
        for name, v in (("positions", pos), ("slots", slot)):
            if not (v.dtype == torch.int64 and v.is_contiguous() and v.numel() >= M):
                raise RuntimeError(
                    f"asm K1 {name}: {tuple(v.shape)} {v.dtype}, want int64 [>= {M}]"
                )
        _check_cache("SWA cache", swa, swa_block)
        if x is None:
            x, post, comb = self.zero_x[:M], self.zero_post[:M], run.s["eye"][:M]
        seam_in = (residual.contiguous(), x.contiguous(), post.contiguous(),
                   comb.contiguous(), prev_pre.contiguous())  # fmt: skip
        if self.v3:
            # idx: the seam keeps y' for the index step; the front's indexer CTAs stay off
            r, post_a, comb_a, y, pre_a = run.seam_v3(w, *seam_in, idx=index)
            q = run.front_v3(w, M, pos, slot, swa, swa_block)
        else:
            r, post_a, comb_a, y, pre_a = run.seam(w, *seam_in)
            q = run.front(w, M, pos, slot, swa, swa_block)
        self.index_rows = self._index_rows(w, M, y) if index else None
        return q, r, post_a, comb_a, pre_a

    def _index_rows(self, w, M: int, y: torch.Tensor) -> tuple:
        """An index source's normed input and qr from the front's scratch (one launch)."""
        s, eps = self.run.s, w["index_eps"]
        if self.v3:
            _index_rows_v3[(M,)](
                y, s["rstd"], s["ss"], s["qrq"], s["qrs"], self.normed, self.qr, eps,
                H_=H, QR=Q_RANK, NSS=Q_RANK // 32, BH=1024, BQ=2048, num_warps=4,
            )  # fmt: skip
            return self.normed[:M], self.qr[:M]
        _index_qr_v1[(M,)](
            s["qr_kv"], w["qw"], self.qr, eps, QKV_=QKV, QR=Q_RANK, BQ=2048,
            num_warps=4,
        )  # fmt: skip
        return y, self.qr[:M]

    def attend(self, w, L, q, pos, caches, keys) -> None:
        """The sparse attention of the front's q over the token's window and, on a layer
        with compressed keys, its top-k: its output stays in the runner's scratch."""
        swa, swa_block, comp, comp_block = caches
        swa_idx, _, tk_idx, tk_lens = keys
        M = q.shape[0]
        if not (
            swa_idx.dtype == torch.int32
            and swa_idx.is_contiguous()
            and swa_idx.shape[1:] == (SWA_WIDTH,)
            and swa_idx.shape[0] >= M
        ):
            raise RuntimeError(
                f"asm K1: window slots {tuple(swa_idx.shape)} {swa_idx.dtype}, want "
                f"int32 [>= {M}, {SWA_WIDTH}]"
            )
        comp_keys = L.ratio > 0
        if comp_keys:
            _check_cache("compressed cache", comp, comp_block)
        if comp_keys and not (
            tk_idx.dtype == tk_lens.dtype == torch.int32
            and tk_idx.is_contiguous()
            and tk_lens.is_contiguous()
            and tk_idx.shape[1:] == (TOPK,)
            and min(tk_idx.shape[0], tk_lens.shape[0]) >= M
        ):
            raise RuntimeError(
                f"asm K1: top-k slots {tuple(tk_idx.shape)} {tk_idx.dtype}, lengths "
                f"{tuple(tk_lens.shape)}, want int32 [>= {M}, {TOPK}] and [>= {M}]"
            )
        self.run.attend(
            w, q, pos, swa, swa_block, swa_idx, self.sw_ptr[: M + 1],
            comp if comp_keys else None, comp_block,
            tk_idx if comp_keys else None, tk_lens if comp_keys else None,
            fixed=True, v3=self.v3,
        )  # fmt: skip

    def __call__(self, w, L, x, residual, post, comb, prev_pre, pos, slot, caches,
                 keys, index, wire):  # fmt: skip
        """K1 of layer ``L``: (x_attn [M, 5120] bf16 all-reduced, residual', post_a
        [M, 4, 1], comb_a [M, 4, 4], pre_a [M, 4]). ``caches`` = (swa, swa_block, comp,
        comp_block); ``keys`` = (window slots, window lengths, top-k slots, top-k
        lengths) as the monokernel's K1 takes them; ``slot`` int64. ``index``: None (or
        False) on a layer without an index source, else its index step
        (``sglang_index``), called once after the front with ``index_rows``; it writes
        the top-k slots of ``keys`` in place before the attention reads them."""
        if index is False:
            index = None
        if index is not None and not callable(index):
            raise TypeError(
                f"asm K1 layer {L.i}: index {index!r}, want None or the layer's index "
                "step (sglang_index)"
            )
        swa, swa_block = caches[0], caches[1]
        q, r, post_a, comb_a, pre_a = self.front(
            w, x, residual, post, comb, prev_pre, pos, slot, swa, swa_block,
            index=index is not None,
        )  # fmt: skip
        if index is not None:
            index(*self.index_rows)
        self.attend(w, L, q, pos, caches, keys)
        out = self.run.back_ar(w, residual.shape[0], pos, self.wob, wire)
        return out, r, post_a, comb_a, pre_a
