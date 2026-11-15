# SPDX-License-Identifier: Apache-2.0
"""DSv4.1's decode layers for steps of MONO_MAX + 1 to MAX_M rows at TP4 (the wide path):
vllm#65421's asm K2 (the FFN half: mHC seam, router, shared and routed experts, the
combine with all-reduce #2) after a K1 (the attention half with all-reduce #1).

K1 is ``k1.AsmK1``, the asm K1's chain that ``SGLANG_AMD_MONO_ASM_K1`` names ("v3", the
default, or "v1"), on SGLang's caches and key lists, with SGLang's compressor and decode
indexer as an index source's index step; "0" keeps SGLang's own attention half of the
layer (the fused boundary, the input norm and ``self_attn``). Either way the
state between the halves is the fused boundary's: (x_attn, residual, post, comb, pre),
and K2 returns the layer's (x, residual, post, comb, pre) as the monokernel's K2 does.

One ``AsmLayers`` per process, built in the spec's build before SGLang sizes the KV
cache: the guard block (``guard.py``), one K2 (one K2Runner, its scratch sized for
MAX_M rows and shared by the layers, and one CombAllReduce), with the asm K1 one
AsmK1 (one K1Runner and one WoBAllReduce), and each layer's weights. Every rank makes
the same launches in the same order.

``SGLANG_AMD_MONO_ASM_WIRE``: "bf16" (default, exact as the custom all-reduce) or
"fp8" (MXFP8) for the asm all-reduces. ``SGLANG_AMD_MONO_ASM_K2`` picks K2's chain
(``k2.py``). ``SGLANG_AMD_MONO_ASM_CHECK=<dir>`` (tests, eager only): every wide
layer also runs SGLang's FFN half on the same K1 outputs and appends their per-token
differences to ``<dir>/rank<r>.jsonl``.
"""

import json
import os

import torch

MAX_M = 256
HC = 4

# by file under sglang/ (``f(p)``: a parameter, ``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    # the fused boundary forward's halves (forward_hc_pre_from_prev_fused_boundary)
    "srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py": (
        "hc_boundary",
        "_gfx95_dense_post_attention_norm",
    ),
    "kernels/ops/layernorm/mhc_boundary_hip.py": ("tensors",),
    # DSpark's captures: the mean of a target layer's input residual streams
    "kernels/ops/layernorm/mhc.py": ("mhc_stream_mean", "mhc_stream_mean(out)"),
    # the asm K1's per-step int64 SWA slots (``k1_keys``) and key lists
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "get_swa_out_cache_loc",
        "forward_metadata",
        "core_attn_metadata",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
    ),
    "srt/models/deepseek_v4.py": (
        "self_attn",
        "mlp",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "_input_norm",
        "_input_norm(allow_aiter_quant)",
        "_input_norm(coefficients)",
        "maybe_use_decode_attn_tp",
        "forward(x_quant)",
        "forward(defer_all_reduce)",
        '"input_ids_global"',
    ),
    "srt/models/deepseek_v4_platform_stage.py": (
        "_run_moe_ffn_dp_sync",
        "_run_moe_ffn_dp_sync(input_ids)",
        "_run_moe_ffn_dp_sync(input_ids_global)",
    ),
    "srt/models/deepseek_v2.py": ("gate", "e_score_correction_bias"),
}
# vLLM's test bounds for a layer's outputs (test_dsv41_asm_layers.py): the residual
# within 2 %, the FFN output within 12 % unless the token's top-6 is a near tie on
# SGLang's path (the 6th and 7th scores closer than TIE_FIRST at the first layer,
# TIE_LATER past it); such a token may route elsewhere and is exempt from later layers
RES_REL, X_REL, TIE_FIRST, TIE_LATER = 0.02, 0.12, 5e-3, 1e-2


def wire() -> str:
    w = os.environ.get("SGLANG_AMD_MONO_ASM_WIRE", "bf16")
    if w not in ("bf16", "fp8"):
        raise ValueError(f"SGLANG_AMD_MONO_ASM_WIRE={w!r}: bf16 or fp8")
    return w


def k1_chain() -> str | None:
    """The asm K1's chain ("v3" by default), or None for SGLang's attention half ("0")."""
    c = os.environ.get("SGLANG_AMD_MONO_ASM_K1", "v3")
    if c not in ("0", "v3", "v1"):
        raise ValueError(f"SGLANG_AMD_MONO_ASM_K1={c!r}: 0, v3 or v1")
    return None if c == "0" else c


def k1_keys(plan, positions, forward_batch, m: int):
    """The asm K1's int64 positions and SWA slots of the step (its kv wave reads 8-byte
    slots): one copy a step in ``plan.state``, which every layer reads."""
    keys = plan.state.get("asm_k1")
    if keys is None:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        slot = get_attn_backend().get_swa_out_cache_loc(forward_batch)[:m]
        keys = plan.state["asm_k1"] = (
            positions[:m].to(torch.int64).contiguous(),
            slot.to(torch.int64).contiguous(),
        )
    return keys


def attention_half(L, positions, forward_batch, pending_post, prev_pre, capture=None):
    """SGLang's attention half of layer ``L`` (the stand-in K1), as its fused-boundary
    forward runs it (``forward_hc_pre_from_prev_fused_boundary``) above the all-reduce
    fusions' rows: the pending post (x, residual, post, comb) and pre in, (x_attn,
    residual, post, comb, pre) out, x_attn all-reduced. ``capture(L, rows)``: where the
    layer's DSpark capture goes (the mean of its input residual's streams), or None."""
    from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        hc_boundary,
    )

    layer = L.module
    residual, y, co = hc_boundary(
        layer, *pending_post, prev_pre,
        layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
    )  # fmt: skip
    aux = None if capture is None else capture(L, residual.shape[0])
    if aux is not None:
        mhc_stream_mean(residual, out=aux)
    # the boundary's reduce + sinkhorn rides in the norm launch
    y, y_quant = layer._input_norm(y, allow_aiter_quant=False, coefficients=co)
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        x = layer.self_attn(
            x=y,
            positions=positions,
            forward_batch=forward_batch,
            x_quant=y_quant,
            defer_all_reduce=False,
        )
    pre, post, comb = co.tensors()
    return x, residual, post, comb, pre


def _ffn_half(L, forward_batch, x_attn, residual, post, comb, pre):
    """SGLang's FFN half on the same K1 outputs (the check's reference): (x, residual,
    post, comb, pre) and the router's normed input."""
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        _gfx95_dense_post_attention_norm,
        hc_boundary,
    )

    layer = L.module
    residual, y, co = hc_boundary(
        layer, x_attn, residual, post, comb, pre,
        layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
    )  # fmt: skip
    y = _gfx95_dense_post_attention_norm(layer, y, co)
    ids = forward_batch.input_ids
    x = layer._run_moe_ffn_dp_sync(
        y,
        forward_batch,
        input_ids=ids,
        input_ids_global=getattr(forward_batch, "input_ids_global", ids),
    )
    pre, post, comb = co.tensors()
    return (x, residual, post, comb, pre), y


def _rel_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row ||a - b|| / ||b||."""
    a, b = a.flatten(1).float(), b.flatten(1).float()
    return (a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)


def _tie_gaps(L, y: torch.Tensor) -> torch.Tensor:
    """Per row, the 6th minus the 7th routing score of SGLang's router on its normed
    input (sqrtsoftplus + the correction bias, as noaux_tc selects)."""
    gate = L.module.mlp.gate
    s = torch.nn.functional.softplus(y.float() @ gate.weight.float().t()).sqrt()
    s = s + gate.e_score_correction_bias.float()
    top = s.topk(7, dim=1).values
    return top[:, 5] - top[:, 6]


class AsmLayers:
    """The wide path's launches of one process (TP4, gfx950)."""

    def __init__(self, rank: int, cpu_group, tp: int, *, deadline_s: float):
        from sgl_amd_mono.deepseek_v41.asm.guard import AsmGuard

        if tp != 4:
            raise ValueError(f"the asm decode layers are written for TP4, not TP{tp}")
        self.rank, self.wire = rank, wire()
        self.guard = AsmGuard(rank, cpu_group, 4, deadline_s=deadline_s)
        from sgl_amd_mono.deepseek_v41.asm.k2 import AsmK2

        self.k2 = AsmK2(rank, cpu_group, guard=self.guard)
        self.k1, chain = None, k1_chain()
        if chain is not None:
            from sgl_amd_mono.deepseek_v41.asm.ar_runner import WoBAllReduce
            from sgl_amd_mono.deepseek_v41.asm.k1 import AsmK1
            from sgl_amd_mono.deepseek_v41.asm.k1_runner import K1Runner

            wob = WoBAllReduce(rank, cpu_group, 4, guard=self.guard)
            self.k1 = AsmK1(K1Runner(v3=chain == "v3"), wob, chain)
            # the first layer's pre (no previous pre): copy 0, as the monokernel's
            self.first_pre = torch.zeros(MAX_M, HC, dtype=torch.float32, device="cuda")
            self.first_pre[:, 0] = 1.0
        self._w: dict[int, dict] = {}
        self._w1: dict[int, dict] = {}
        self.check_dir = os.environ.get("SGLANG_AMD_MONO_ASM_CHECK") or None
        self._step = 0

    def add_layer(self, i: int, L) -> None:
        """Layer i's K2 weights, and its K1 weights with the asm K1 (in the build,
        before SGLang sizes the KV cache)."""
        self._w[i] = self.k2.weights(L.ffn)
        if self.k1 is not None:
            self._w1[i] = self.k1.weights(L)

    def set_deadline(self, seconds: float) -> None:
        self.guard.set_deadline(seconds)

    def errors(self) -> dict:
        """The guard's sticky words and the combine's timed-out CTAs. Syncs (tests and
        failure reports only)."""
        return dict(guard=self.guard.errors(), comb_ar_timeouts=self.k2.errors())

    def layer(
        self, L, plan, positions, forward_batch, pending_post, prev_pre, capture=None
    ):
        """K1 then K2 of layer ``L``: what ``spec._launch`` returns."""
        m = pending_post[1].shape[0]
        assert m <= MAX_M, m
        if self.k1 is None:
            k1 = attention_half(
                L, positions, forward_batch, pending_post, prev_pre, capture
            )
        else:
            k1 = self._asm_k1(
                L, plan, positions, forward_batch, pending_post, prev_pre, capture
            )
        out = self.k2(self._w[L.i], *(t.contiguous() for t in k1), self.wire)
        if self.check_dir is not None:
            self._check(L, forward_batch, k1, out)
        return out

    def _asm_k1(
        self, L, plan, positions, forward_batch, pending_post, prev_pre, capture
    ):
        """The asm K1 of layer ``L`` on SGLang's caches and key lists, as
        ``spec._launch`` hands them to the monokernel's K1 (the caches set there at the
        layer's first launch); an index source's index step is SGLang's
        (``k1.sglang_index``)."""
        from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_amd_mono.deepseek_v41.asm.k1 import sglang_index

        m = pending_post[1].shape[0]
        pos, slot = k1_keys(plan, positions, forward_batch, m)
        md = get_attn_backend().forward_metadata.core_attn_metadata
        if L.ratio:
            topk = (md.sparse_page_indices(L.ratio), md.sparse_topk_lengths(L.ratio))
        else:
            topk = (None, None)  # a window-only layer reads no compressed keys
        keys = (md.swa_page_indices, md.swa_topk_lengths, *topk)
        if pending_post[0] is None and prev_pre is None:
            prev_pre = self.first_pre[:m]
        out = self.k1(
            self._w1[L.i], L, *pending_post, prev_pre, pos, slot,
            (L.swa, L.swa_block, L.comp, L.comp_block), keys,
            sglang_index(L, positions, forward_batch) if L.index else None, self.wire,
        )  # fmt: skip
        aux = None if capture is None else capture(L, m)
        if aux is not None:
            # the mean of the layer's input residual streams: K1's residual (post in)
            mhc_stream_mean(out[1], out=aux)
        return out

    def _check(self, L, forward_batch, k1, out) -> None:
        """SGLang's FFN half on the same K1 outputs; one record of the per-row
        differences a layer (host syncs: eager only)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SGLANG_AMD_MONO_ASM_CHECK runs eagerly only")
        ref, y = _ffn_half(L, forward_batch, *k1)
        x_rel = _rel_rows(out[0], ref[0])
        res_rel = _rel_rows(out[1], ref[1])
        mix_rel = max(
            _rel_rows(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)).max().item()
            for a, b in zip(out[2:], ref[2:])
        )
        if L.i == min(self._w):
            self._step += 1
        rec = dict(
            step=self._step,
            layer=L.i,
            rows=int(out[0].shape[0]),
            wire=self.wire,
            x_rel=[round(v, 5) for v in x_rel.tolist()],
            res_rel=[round(v, 5) for v in res_rel.tolist()],
            gap=[round(v, 5) for v in _tie_gaps(L, y).tolist()],
            mix_rel=round(mix_rel, 6),
        )
        os.makedirs(self.check_dir, exist_ok=True)
        with open(os.path.join(self.check_dir, f"rank{self.rank}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
