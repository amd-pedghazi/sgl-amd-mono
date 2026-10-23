# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): each covered layer also runs SGLang's
layer on its input, on the caches as they were, and checks the kernel's outputs
against it: the hidden stream it writes, the cache rows the layer writes, and on
an indexer layer the top-k it selects, each against its ``spec.NUMERICS`` bound.
Every check is logged; a miss is logged as a violation.

On an indexer layer the index keys the step writes stay in the cache as SGLang
writes them, with their scales set to POISON_SCALE. A launch must score those keys
from its rows' pairs; one that reads the cache drops each of them that SGLang
selects with a positive score from its top-k (``layer.topk_own``)."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": (
        "fold",
        "take_output",
        "set_written",
    ),
    "srt/layers/moe/topk.py": ("topk_ids", "topk_weights"),
    "srt/mem_cache/memory_pool.py": ("get_key_buffer", "get_index_k_with_scale_buffer"),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc",),
    "srt/model_executor/forward_context.py": ("get_token_to_kv_pool",),
    "srt/models/deepseek_v2.py": ("mlp", "gate", "topk", "e_score_correction_bias"),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/utils/common.py": ("BumpAllocator",),
}

PAGE_TOKENS = 64  # tokens an index-cache page covers
INDEX_VALUE_BYTES = PAGE_TOKENS * 128  # an index page's FP8 values; f32 scales follow
# the kernel scales a key's head-weighted ReLU sum: a positive sum then scores below
# any top-k threshold; finite, so a zero sum scores 0, not NaN
POISON_SCALE = -1e30
SHARED_EXPERT = 256  # SGLang's id for the fused shared expert
TOP_K, ROUTE_SCALE = 8, 2.5


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    return {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _index_keys(page: torch.Tensor):
    """An index page's keys as scored (FP8 values times their token's power-of-two
    scale) and its scales. A value byte b of the 16 x 16 tiles is token
    (b // 2048) * 16 + (b % 256) // 16."""
    b = torch.arange(INDEX_VALUE_BYTES, device=page.device)
    token = (b // 2048) * 16 + (b % 256) // 16
    values = page[:, :INDEX_VALUE_BYTES].contiguous().view(torch.float8_e4m3fn).float()
    scales = page[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32)
    return values * scales[:, token], scales


def _outputs(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, dict]:
    if name != "index":
        return {name: _metrics(a, b)}
    # a key whose amax sits at a power of two may take the other scale and values
    (ka, sa), (kb, sb) = _index_keys(a), _index_keys(b)
    return {"index": _metrics(ka, kb), "index_scale": _metrics(sa, sb)}


def poison_new_index_scales(pages: torch.Tensor, slots: torch.Tensor) -> None:
    """Set the cached scales of the index keys at ``slots`` (this step's, 0 = pad)
    to POISON_SCALE in the uint8 index pages."""
    slots = slots[slots != 0].long()
    scales = pages.view(pages.shape[0], -1)[:, INDEX_VALUE_BYTES:].view(torch.float32)
    scales[slots // PAGE_TOKENS, slots % PAGE_TOKENS] = POISON_SCALE


def _topk_outputs(ref_topk: torch.Tensor, step, slots: torch.Tensor, S: int) -> dict:
    """Per row, SGLang's selected KV slots against the kernel's (``step.topk`` at
    the row's ``indptr``): the largest share of SGLang's a row misses, and the most
    of this step's own slots (a verify row's earlier drafts and itself) one misses."""
    bounds = step.indptr[: S + 1].tolist()
    picked = step.topk.tolist()
    own = set(slots[slots != 0].tolist())
    rel, own_missed = 0.0, 0
    for s, row in enumerate(ref_topk[:S].tolist()):
        want = {k for k in row if k >= 0}
        if not want:
            continue
        missed = want - set(picked[bounds[s] : bounds[s + 1]])
        rel = max(rel, len(missed) / len(want))
        own_missed = max(own_missed, len(missed & own))
    return {"topk": {"rel": rel}, "topk_own": {"max_abs": float(own_missed)}}


def _regions(L, forward_batch, S: int) -> dict:
    """The cache entries the layer writes this step: name -> (tensor, index)."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    loc = forward_batch.out_cache_loc[:S]
    slots = loc[loc != 0].long()
    kv = pool.get_key_buffer(L.i)
    regions = {"kv": (kv.view(kv.shape[0], -1), slots)}
    if L.w.index is not None:
        regions["index"] = (
            pool.get_index_k_with_scale_buffer(L.i),
            (slots // PAGE_TOKENS).unique(),
        )
    return regions


def _f32_routing(mlp):
    """A top-k hook that routes in f32 as the model defines it and the kernel does
    (SGLang's AITER top-k adds the ~34 correction bias in bf16): the 8 experts of
    the largest sigmoid + bias, weighted by their sigmoid over its sum, times 2.5."""

    seen = {}

    def gate_in(mod, args):
        seen["x"] = args[0]

    def route(mod, args, out):
        x = seen.pop("x")
        scores = torch.sigmoid(x.float() @ mlp.gate.weight.float().t())
        ids = (
            (scores + mlp.gate.e_score_correction_bias.float()).topk(TOP_K, -1).indices
        )
        raw = scores.gather(1, ids)
        routed = out.topk_ids != SHARED_EXPERT
        new_ids, new_w = out.topk_ids.clone(), out.topk_weights.clone()
        new_ids[routed] = ids.to(new_ids.dtype).reshape(-1)
        new_w[routed] = (
            (raw / raw.sum(-1, keepdim=True) * ROUTE_SCALE).to(new_w.dtype).reshape(-1)
        )
        return out._replace(topk_weights=new_w, topk_ids=new_ids)

    return mlp.register_forward_pre_hook(gate_in), mlp.topk.register_forward_hook(route)


def reference(L, positions, h_in, forward_batch, S: int, prev_topk) -> dict:
    """SGLang's layer on ``h_in`` [S, HIDDEN] (a copy), routed in f32: its output
    stream, its top-k and the cache rows it writes. The KV cache is restored after;
    the index keys keep SGLang's values with POISON_SCALE scales."""
    from sglang.srt.layers.layer_boundary.residual.batch import (
        fold,
        set_written,
        take_output,
    )
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S)
    kv, kv_slots = regions["kv"]
    saved = kv[kv_slots].clone()
    hidden = set_written(h_in, forward_batch)
    zero = BumpAllocator(buffer_size=4, dtype=torch.float32, device=h_in.device)
    hooks = _f32_routing(L.module.mlp)
    try:
        out, topk = L.module(
            positions[:S],
            hidden,
            forward_batch,
            zero,
            None,
            None,
            prev_topk_indices=prev_topk,
        )
    finally:
        for hook in hooks:
            hook.remove()
    ref = {
        "streams": take_output(fold(out, forward_batch), forward_batch).clone(),
        "topk": topk,
    }
    for name, (t, idx) in regions.items():
        ref[name] = t[idx].clone()
    kv[kv_slots] = saved
    if "index" in regions:
        poison_new_index_scales(regions["index"][0], forward_batch.out_cache_loc[:S])
    return ref


def compare(
    L, ref: dict, streams: torch.Tensor, positions, forward_batch, S: int, *, step
) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53.spec import NUMERICS

    if get_parallel().tp_rank != 0:
        return
    outputs = _outputs("streams", streams, ref["streams"])
    for name, (t, idx) in _regions(L, forward_batch, S).items():
        outputs.update(_outputs(name, t[idx], ref[name]))
    # SGLang's layer returns no top-k when no later layer reuses it
    if L.w.index is not None and ref["topk"] is not None:
        slots = forward_batch.out_cache_loc[:S]
        outputs.update(_topk_outputs(ref["topk"], step, slots, S))
    values = {
        f"layer.{out}.{metric}": v
        for out, ms in outputs.items()
        for metric, v in ms.items()
    }
    pos = positions[:S]
    logger.info(
        "monokernel check layer %d (%s, %d rows, positions %d..%d): %s",
        L.i,
        "indexer" if L.w.index is not None else "shared",
        S,
        pos.min().item(),
        pos.max().item(),
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                L.i,
                name,
                v,
                bound,
            )
