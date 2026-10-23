# SPDX-License-Identifier: Apache-2.0
"""Views of one sparse MoE layer, checked against what the kernels read.

The kernels read tensors through raw pointers, where a differently laid out weight
would not fault but give plausible garbage, so ``layer_reasons`` checks every
assumption once, a refusal at startup, before the views are taken. The MXFP4 experts,
the norms and the K / V scales are read in place. The copies: the qkv / index
projection and o_proj, which the public Quark export leaves in bf16, quantized per
channel (amax / 448) on aiter's (16, 16) preshuffled layout for the kernels'
per-token x per-channel FP8 GEMVs; the router gate as bf16 for the kernels' bf16 GEMV
(SGLang holds it in fp32); the routing bias as fp32.
"""

from __future__ import annotations

import msgspec
import torch

from sgl_amd_mono.minimax_m3.gfx950.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ONE_INDEX_HEAD,
    ROTARY_DIM,
    TOP_K,
    TOPK_BLOCKS,
    qkv_rows,
)

# by file under sglang/ (``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    "srt/models/minimax_m3.py": (
        "self_attn",
        "mlp",
        "is_layer_sparse",
        "is_sparse_attention_layer",
        "disable_index_value",
        "num_heads",
        "num_kv_heads",
        "num_idx_heads",
        "head_dim",
        "idx_head_dim",
        "rotary_dim",
        "use_gemma_norm",
        "qk_norm_type",
        "input_layernorm",
        "post_attention_layernorm",
        "q_norm",
        "k_norm",
        "index_q_norm",
        "index_k_norm",
        "_fused_qkv_index",
        "qkv_proj",
        "index_qkv_proj",
        "o_proj",
        "experts",
        "gate",
        "e_score_correction_bias",
        "rotary_emb",
        "attn",
    ),
    "srt/layers/layernorm.py": ("GemmaRMSNorm",),
    "srt/layers/moe/fused_moe_triton/layer.py": (
        "moe_ep_rank",
        "num_local_experts",
        "num_fused_shared_experts",
        "top_k",
        "intermediate_size_per_partition",
        "moe_runner_config",
    ),
    "srt/layers/moe/moe_runner/base.py": ("activation",),
    "srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py": (
        "is_shuffled",
        '"w13_weight"',
        '"w2_weight"',
        '"w13_weight_scale"',
        '"w2_weight_scale"',
    ),
    "srt/layers/rotary_embedding/*.py": ("cos_sin_cache",),
    "srt/layers/radix_attention.py": ("k_scale", "v_scale"),
}

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
EXPERTS = N_ROUTED + 1  # the routed experts and the fused shared expert


def linear_to_ptpc_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 [rows, cols] weight as per-channel FP8 on aiter's (16, 16) shuffle,
    and its fp32 [rows] scales."""
    from aiter.ops.shuffle import shuffle_weight

    deq = weight.to(torch.float32)
    amax = deq.abs().amax(dim=1).clamp_(min=torch.finfo(torch.float32).tiny)
    scale = amax / FP8_MAX
    q = (deq / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return shuffle_weight(q.contiguous(), layout=(16, 16)).contiguous(), scale


def _qkv_weight(attn) -> torch.Tensor:
    """qkv | index_q | index_k: the fused projection SGLang builds after loading,
    else the two bf16 projections concatenated in that order."""
    fused = attn._fused_qkv_index
    if fused is not None:
        return fused.weight.data
    return torch.cat([attn.qkv_proj.weight.data, attn.index_qkv_proj.weight.data])


def _norm_reasons(module, n: int, what: str) -> list[str]:
    """The kernels apply ``w + 1``: a GemmaRMSNorm's contiguous bf16 [n] weight."""
    if type(module).__name__ != "GemmaRMSNorm":
        return [f"{what}: {type(module).__name__}, the kernels apply GemmaRMSNorm"]
    w = module.weight
    if not (w.dtype == torch.bfloat16 and w.numel() == n and w.is_contiguous()):
        return [f"{what}: {w.dtype} {tuple(w.shape)}, want contiguous bf16 [{n}]"]
    return []


def layer_reasons(layer, i: int) -> list[str]:
    """Every reason the kernels cannot read sparse MoE layer ``i`` (empty: they can)."""
    attn, moe = layer.self_attn, layer.mlp
    if not (layer.is_layer_sparse and attn.is_sparse_attention_layer):
        return [f"layer {i}: not a sparse-attention MoE layer"]
    why = []
    if not attn.disable_index_value:
        why.append(f"layer {i}: the indexer carries values; the kernels read K only")
    shape = (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim,
             attn.idx_head_dim)  # fmt: skip
    want = (LOCAL_Q_HEADS, 1, ONE_INDEX_HEAD.count, HEAD_DIM, HEAD_DIM)
    if shape != want:
        why.append(f"layer {i}: heads / dims {shape} a rank, built for {want}")
    if attn.rotary_dim != ROTARY_DIM:
        why.append(f"layer {i}: rotary_dim {attn.rotary_dim}, built for {ROTARY_DIM}")
    if not (attn.use_gemma_norm and attn.qk_norm_type == "per_head"):
        why.append(f"layer {i}: the kernels apply per-head gemma q / k norms")
    for module, n, what in (
        (layer.input_layernorm, HIDDEN, "input_layernorm"),
        (layer.post_attention_layernorm, HIDDEN, "post_attention_layernorm"),
        (attn.q_norm, HEAD_DIM, "q_norm"),
        (attn.k_norm, HEAD_DIM, "k_norm"),
        (attn.index_q_norm, HEAD_DIM, "index_q_norm"),
        (attn.index_k_norm, HEAD_DIM, "index_k_norm"),
    ):
        why += _norm_reasons(module, n, f"layer {i} {what}")
    fused = attn._fused_qkv_index
    if fused is not None:
        qkv = (fused.weight.dtype, tuple(fused.weight.shape))
    else:  # the two projections _qkv_weight concatenates
        qp, ip = attn.qkv_proj.weight, attn.index_qkv_proj.weight
        same = qp.dtype if qp.dtype == ip.dtype else (qp.dtype, ip.dtype)
        qkv = (same, (qp.shape[0] + ip.shape[0], qp.shape[1]))
    o = (attn.o_proj.weight.dtype, tuple(attn.o_proj.weight.shape))
    for got, want, what in (
        (qkv, (qkv_rows(ONE_INDEX_HEAD.count), HIDDEN), "qkv"),
        (o, (HIDDEN, O_K), "o_proj"),
    ):
        if got != (torch.bfloat16, want):
            why.append(f"layer {i} {what}: {got}, want bf16 {want}")
    experts = moe.experts
    if not (experts.moe_ep_rank == 0 and experts.num_local_experts == EXPERTS
            and experts.num_fused_shared_experts == 1):  # fmt: skip
        why.append(
            f"layer {i}: experts {experts.num_local_experts} local / "
            f"{experts.num_fused_shared_experts} fused shared; the kernels need "
            f"{N_ROUTED} routed + the fused shared expert, no expert parallelism"
        )
    # FusedMoE counts the fused shared expert in its top_k
    routed = int(experts.top_k) - int(experts.num_fused_shared_experts)
    if routed != TOP_K:
        why.append(f"layer {i}: {routed} routed experts a token, built for {TOP_K}")
    if experts.intermediate_size_per_partition != INTER:
        why.append(
            f"layer {i}: expert intermediate {experts.intermediate_size_per_partition}"
            f" a rank, built for {INTER}"
        )
    w13, w2 = experts.w13_weight, experts.w2_weight
    # aiter's MXFP4 method marks its shuffled tensors; no mark is no shuffle
    if not (getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False)):
        why.append(f"layer {i}: expert weights are not AITER-shuffled MXFP4")
    # SiTU interleaves gate and up rows (A16W4), still marked shuffled; the kernel reads them apart
    activation = experts.moe_runner_config.activation
    if activation != "silu":
        why.append(f"layer {i}: expert activation {activation!r}, built for 'silu'")
    if not (w13.dtype == w2.dtype == torch.uint8
            and w13.numel() == EXPERTS * 2 * INTER * HIDDEN // 2
            and w2.numel() == EXPERTS * HIDDEN * INTER // 2):  # fmt: skip
        why.append(f"layer {i}: expert weights {w13.dtype} {tuple(w13.shape)}")
    # an MXFP4 quant method registers block scales; another method has none
    s13 = getattr(experts, "w13_weight_scale", None)
    s2 = getattr(experts, "w2_weight_scale", None)
    if not (s13 is not None and s2 is not None
            and s13.dtype == s2.dtype == torch.uint8
            and s13.numel() == EXPERTS * 2 * INTER * HIDDEN // 32
            and s2.numel() == EXPERTS * HIDDEN * INTER // 32):  # fmt: skip
        why.append(f"layer {i}: expert block scales are not MXFP4 E8M0")
    if tuple(moe.gate.weight.shape) != (N_ROUTED, HIDDEN):
        why.append(f"layer {i}: router gate {tuple(moe.gate.weight.shape)}")
    bias = moe.e_score_correction_bias
    if bias is None or bias.numel() != N_ROUTED:
        why.append(f"layer {i}: no routing bias correction")
    table = attn.rotary_emb.cos_sin_cache
    if table.dim() != 2 or table.shape[1] != ROTARY_DIM:
        why.append(f"layer {i}: cos / sin cache {tuple(table.shape)}")
    for scale, what in ((attn.attn.k_scale, "k_scale"), (attn.attn.v_scale, "v_scale")):
        if isinstance(scale, torch.Tensor) and scale.numel() != 1:
            why.append(f"layer {i} {what}: {tuple(scale.shape)}, want one scale")
    return why


def selection_reasons(config) -> list[str]:
    """Reasons the model's sparse-attention selection is not the kernels'."""
    cfg = getattr(config, "sparse_attention_config", None) or {}
    got = tuple(cfg.get(k) for k in ("sparse_topk_blocks", "sparse_init_block",
                                     "sparse_local_block"))  # fmt: skip
    why = []
    if None in got or tuple(int(x) for x in got) != (TOPK_BLOCKS, 0, 1):
        why.append(f"sparse selection {got}, the kernels are built for (16, 0, 1)")
    if (int(cfg.get("sparse_block_size", 0)), str(cfg.get("sparse_score_type", "max"))) != (128, "max"):  # fmt: skip
        why.append(f"sparse block / score {cfg.get('sparse_block_size')} / "
                   f"{cfg.get('sparse_score_type')}, built for 128 / max")  # fmt: skip
    return why


def sparse_selection(config) -> tuple[int, int, int]:
    """(top-k, init, local) block counts of a model ``selection_reasons`` accepts."""
    cfg = config.sparse_attention_config
    return tuple(int(cfg[k]) for k in ("sparse_topk_blocks", "sparse_init_block",
                                       "sparse_local_block"))  # fmt: skip


def _scale(value, device) -> torch.Tensor:
    """A cache scale as the one-fp32 device tensor the kernels dereference."""
    if value is None:
        return torch.ones(1, dtype=torch.float32, device=device)
    if isinstance(value, torch.Tensor):
        return value.data.to(device=device, dtype=torch.float32).reshape(1)
    return torch.tensor([float(value)], dtype=torch.float32, device=device)


class SparseMoeLayer(msgspec.Struct, frozen=True):
    """Everything the launch reads for one sparse MoE layer that ``layer_reasons``
    accepted."""

    layer_id: int
    g_in: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor
    g_q: torch.Tensor
    g_k: torch.Tensor
    g_iq: torch.Tensor
    g_ik: torch.Tensor
    cos_sin: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor
    g_post: torch.Tensor
    gate: torch.Tensor
    bias: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @staticmethod
    def build(layer, i: int, copies: dict) -> SparseMoeLayer:
        """The layer's views, and its copies kept in ``copies`` (logged at startup):
        the FP8 projections, the bf16 router gate, the fp32 bias, one bf16 RoPE
        table for every layer that shares SGLang's cached rotary module."""
        attn, moe, experts = layer.self_attn, layer.mlp, layer.mlp.experts
        w_qkv, s_qkv = linear_to_ptpc_fp8(_qkv_weight(attn))
        w_o, s_o = linear_to_ptpc_fp8(attn.o_proj.weight.data)
        table = attn.rotary_emb.cos_sin_cache
        key = ("rope table", table.data_ptr())
        if key not in copies:
            copies[key] = table.data.to(torch.bfloat16).contiguous()
        dev = experts.w13_weight.device
        L = SparseMoeLayer(
            layer_id=i,
            g_in=layer.input_layernorm.weight.data,
            w_qkv=w_qkv,
            s_qkv=s_qkv,
            g_q=attn.q_norm.weight.data,
            g_k=attn.k_norm.weight.data,
            g_iq=attn.index_q_norm.weight.data,
            g_ik=attn.index_k_norm.weight.data,
            cos_sin=copies[key],
            w_o=w_o,
            s_o=s_o,
            g_post=layer.post_attention_layernorm.weight.data,
            gate=moe.gate.weight.data.to(torch.bfloat16).contiguous(),
            bias=moe.e_score_correction_bias.data.to(torch.float32).contiguous(),
            w13=experts.w13_weight.data,
            s13=experts.w13_weight_scale.data,
            w2=experts.w2_weight.data,
            s2=experts.w2_weight_scale.data,
            k_scale=_scale(attn.attn.k_scale, dev),
            v_scale=_scale(attn.attn.v_scale, dev),
        )
        for kind, t in (("fp8 qkv", w_qkv), ("fp8 qkv scale", s_qkv),
                        ("fp8 o_proj", w_o), ("fp8 o_proj scale", s_o),
                        ("bf16 router gate", L.gate), ("fp32 routing bias", L.bias)):  # fmt: skip
            copies[kind, i] = t
        return L
