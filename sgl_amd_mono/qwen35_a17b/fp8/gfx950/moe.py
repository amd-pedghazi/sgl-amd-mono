# SPDX-License-Identifier: Apache-2.0
"""Router and block-FP8 MoE for one TP4 rank.

Routed experts are a softmax top-10. When the checkpoint asks for it, those
weights are renormalized. The shared expert, fused as expert ``num_experts``,
is weighted by ``sigmoid(shared_gate(x))`` and is not part of that
renormalization. Expert GEMMs are the block-FP8 weights on this rank
(intermediate 256, fused gate|up N = 512).
"""

from __future__ import annotations

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/moe/smallm_router_gfx950/__init__.py": (
        "smallm_router",
        "smallm_router_supported",
    ),
    "srt/layers/moe/topk.py": ("StandardTopKOutput",),
}


def _smallm_router_topk_output(mlp, hidden: torch.Tensor):
    """SGLang's one-launch gfx950 router with the fused shared expert, or None.

    None sends the caller to the standard gate and top-k, which give the same
    ids. The helper was renamed from ``qwen_moe_topk_output``; this follows
    SGLang's ``qwen2_moe._smallm_router_topk_output``.
    """
    from sglang.kernels.ops.moe.smallm_router_gfx950 import (
        smallm_router,
        smallm_router_supported,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    cfg = mlp.topk.topk_config
    if (
        mlp.num_fused_shared_experts != 1
        or cfg.top_k != 10
        or not cfg.renormalize
        or not smallm_router_supported(hidden)
    ):
        return None
    out = smallm_router(
        hidden,
        mlp.gate.weight,
        mlp.shared_expert_gate.weight,
        mlp._shared_expert_scale(),
    )
    if out is None:
        return None
    return StandardTopKOutput(out[0], out[1], None)


def _tensor(out: torch.Tensor) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def route_and_moe(mlp, hidden: torch.Tensor) -> torch.Tensor:
    """Local MoE partial, before the tensor-parallel all-reduce.

    The routed ids are SGLang's own top-k, including the fused shared expert
    when that fusion is on, so the expert GEMM sees the same ids the standard
    block does. The shared expert is not part of the routed renormalization.
    """
    fused = bool(getattr(mlp, "enable_shared_expert_fusion", False))
    if fused:
        chosen = _smallm_router_topk_output(mlp, hidden)
        if chosen is not None:
            return _tensor(mlp.experts(hidden, chosen))
    logits, _ = mlp.gate(hidden)
    chosen = mlp.topk(hidden, logits)
    if fused:
        chosen = mlp._append_shared_to_topk_output(chosen, hidden)
    return _tensor(mlp.experts(hidden, chosen))
