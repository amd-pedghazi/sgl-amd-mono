# SPDX-License-Identifier: Apache-2.0
"""Check mode (``SGLANG_AMD_MONO_CHECK=1``) of the MiniMax-M3 architecture's launches:
compare a sparse layer's launch with SGLang's modules on a copy of its input, as both
MiniMax-M3 and M3.1 log it."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def routed_experts(layer, residual: torch.Tensor, top_k: int) -> torch.Tensor:
    """SGLang's routed experts of each row (sorted top-k ids: sigmoid scores plus the
    correction bias) for a post-attention residual."""
    out = layer.post_attention_layernorm.forward_native(residual.clone())
    x = out[0] if isinstance(out, tuple) else out
    moe = layer.mlp
    scores = (x.float() @ moe.gate.weight.float().t()).sigmoid()
    scores = scores + moe.e_score_correction_bias.float()
    return scores.topk(top_k, dim=-1).indices.sort(dim=-1).values


def log_check(layer, lid: int, live: torch.Tensor, got, ref, top_k: int) -> None:
    """Log the launch's (MLP output, residual) against SGLang's on the step's live rows.

    Per output, the largest difference over the largest reference value and the least
    row cosine. The MLP output is compared on the rows whose residuals route to the
    same experts: a row routed elsewhere differs by construction, and is counted."""
    if not bool(live.any()):
        return
    alike = (
        routed_experts(layer, got[1][live], top_k)
        == routed_experts(layer, ref[1][live], top_k)
    ).all(dim=-1)
    stats = []
    for name, o, r in zip(("mlp_out", "residual"), got, ref):
        o, r = o[live].float(), r[live].float()
        if name == "mlp_out":
            o, r = o[alike], r[alike]
        if o.numel() == 0:  # every row routed elsewhere: nothing to compare
            stats.append(f"{name} rel 0.00000 cosine 1.000000")
            continue
        rel = ((o - r).abs().amax() / r.abs().amax().clamp_min(1e-30)).item()
        cos = torch.nn.functional.cosine_similarity(o, r, dim=-1).min().item()
        stats.append(f"{name} rel {rel:.5f} cosine {cos:.6f}")
    logger.info(
        "monokernel check layer %d, %d rows (%d routed alike): %s",
        lid, int(live.sum()), int(alike.sum()), "; ".join(stats),
    )  # fmt: skip
