# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash on gfx950: the FFN monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed decode or DSpark
verify step runs ``_Built.forward``, which takes and returns what the model's forward
does. The embedding, the dense layers and each sparse layer's input norm and attention
run SGLang's modules; the rest of every sparse layer is one launch (``ffn.py``), from
the attention output (o_proj on SGLang's rowwise-FP8 copy of its weight) to the
reduced MoE output, whose residual add the next layer's input norm applies. SGLang's
final norm and logits processor close the step.

The launch reads SGLang's post-load MXFP4 routed experts in place, and the shared
expert in the checkpoint's own MXFP8: the load hook keeps this rank's shard, and the
build lays it out for the launch (``a16w8``). ``SGLANG_AMD_MONO_CHECK=1`` runs SGLang's
layer on a copy of every sparse layer's input in eager steps and logs both outputs'
differences on the live rows; the bounds in ``NUMERICS`` come from these runs.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager

import torch
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_amd_mono.minimax_m31 import ffn

logger = logging.getLogger(__name__)

TP = 4
WIDTHS = tuple(range(1, ffn.MAX_TOKENS + 1))
BLOCKS = 256

# The SGLang names this glue reads, by file under the sglang package
SGLANG_READS = {
    "srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/models/minimax_m3.py": (
        "is_layer_sparse",
        '"_is_layer_to_capture"',
        "_norm_quant",
        "attn_boundary",
        "self_attn",
        "mlp",
        "experts",
        "gate",
        "e_score_correction_bias",
        "post_attention_layernorm",
        "layer_id",
        "o_proj",
        "disable_index_value",
        "embed_tokens",
        "norm",
        "layers",
        "end_layer",
        "layers_to_capture",
        "finalize_norm",
    ),
    "srt/models/minimax_common/amd/minimax_m3_hip.py": ("attn_quant_format",),
    "srt/distributed/parallel_state.py": ("get_tp_group", "rank_in_group", "cpu_group"),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/environ.py": ("envs",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "stream_of",
        "final_norm",
        "final_norm(capture)",
        "final_norm(finalize_norm)",
    ),
    "srt/layers/layer_boundary/residual/stream.py": (
        "ResidualStream",
        "write",
        "record",
        "residual",
        "snapshot",
        "from_handoff",
        "export",
    ),
    "srt/layers/layer_boundary/residual/add_norm.py": ("PLAIN_ADD",),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList", "capture"),
    "srt/layers/quantization/fp8.py": ("_ptpc_weight", "_ptpc_scale"),
    "srt/models/minimax_m3_vl.py": (
        "logits_processor",
        "lm_head",
        "forward(get_embedding)",
    ),
}

# The checkpoint names of the shared experts' MXFP8 tensors
_CKPT_SHARED = re.compile(
    r"(?:language_model\.)?model\.layers\.(?P<i>\d+)\.block_sparse_moe\.shared_experts\."
    r"(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)"
)


def _text(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def is_m31(hf_config) -> bool:
    """MiniMax-M3.1-Flash's checkpoint: NVFP4 experts next to MXFP8 dense layers."""
    q = getattr(hf_config, "quantization_config", None) or getattr(
        _text(hf_config), "quantization_config", None
    )
    return isinstance(q, dict) and str(q.get("moe_quant_algo", "")).upper() == "NVFP4"


def config_disable_reasons(cfg, hf_config) -> list[str]:
    """What the shared MiniMax spec's requirements cannot state for M3.1."""
    import importlib.util

    from sglang.srt.environ import envs

    t = _text(hf_config)
    reasons = []
    for key, want in (
        ("hidden_size", ffn.HIDDEN),
        ("num_local_experts", ffn.EXPERTS),
        ("num_experts_per_tok", ffn.TOPK),
        ("routed_scaling_factor", ffn.ROUTED_SCALE),
        ("scoring_func", "sigmoid"),
        ("hidden_act", "swigluoai"),
    ):
        if getattr(t, key, None) != want:
            reasons.append(
                f"config {key}={getattr(t, key, None)!r}, built for {want!r}"
            )
    if getattr(t, "intermediate_size", None) != TP * ffn.INTER:
        reasons.append(f"intermediate_size {getattr(t, 'intermediate_size', None)}")
    if cfg.speculative_algorithm not in (None, "DSPARK"):
        reasons.append(
            f"speculative algorithm {cfg.speculative_algorithm}: needs DSPARK or none"
        )
    if cfg.moe_runner_backend not in ("auto", "aiter"):
        reasons.append(f"--moe-runner-backend {cfg.moe_runner_backend}: needs aiter")
    if getattr(cfg, "enable_hisparse", False):
        reasons.append("HiSparse")
    if getattr(cfg, "expert_distribution_recorder_mode", None) is not None:
        reasons.append("the launches record no expert distribution")
    for name in (
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ):
        var = getattr(envs, name, None)
        if var is not None and var.get():
            reasons.append(f"{name}: the launches route with the model's own router")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each sparse layer's MXFP8 shared expert, by its
    checkpoint name (``...layers.{i}.block_sparse_moe.shared_experts.{gate,up,down}_proj
    .{weight,weight_scale_inv}``): the rows of gate / up, the columns of down."""
    from sglang.srt.runtime_context import get_parallel

    m = _CKPT_SHARED.fullmatch(name)
    if m is None:
        return
    p = get_parallel()
    dim = 1 if m["proj"] == "down" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    copies["shared expert mxfp8", int(m["i"]), m["proj"], m["kind"]] = (
        shard.contiguous()
    )


def _sparse_ids(model) -> list[int]:
    return [i for i, layer in enumerate(model.model.layers) if layer.is_layer_sparse]


def model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    reasons = []
    sparse = _sparse_ids(model)
    if not sparse or sparse != list(range(sparse[0], len(model.model.layers))):
        return [f"sparse layers {sparse}: the forward needs them after the dense ones"]
    for i in sparse:
        layer = model.model.layers[i]
        experts = layer.mlp.experts
        w13, w2 = (
            getattr(experts, "w13_weight", None),
            getattr(experts, "w2_weight", None),
        )
        if (
            w13 is None
            or w13.dtype != torch.uint8
            or tuple(w13.shape) != (ffn.EXPERTS, 2 * ffn.INTER, ffn.HIDDEN // 2)
            or not getattr(w13, "is_shuffled", False)
            or tuple(w2.shape) != (ffn.EXPERTS, ffn.HIDDEN, ffn.INTER // 2)
            or not getattr(w2, "is_shuffled", False)
        ):
            reasons.append(f"layer {i}: routed experts are not SGLang's shuffled MXFP4")
            break
        if tuple(layer.mlp.gate.weight.shape) != (ffn.EXPERTS, ffn.HIDDEN):
            reasons.append(f"layer {i}: router {tuple(layer.mlp.gate.weight.shape)}")
            break
        if layer.mlp.e_score_correction_bias is None:
            reasons.append(f"layer {i}: no routing bias")
            break
        if not getattr(layer.self_attn, "disable_index_value", False):
            reasons.append(f"layer {i}: index values add to o_proj's output")
            break
        wo, wos = _o_proj_rowwise(layer)
        if (
            wo is None
            or wo.dtype != torch.float8_e4m3fn
            or tuple(wo.shape) != (ffn.HIDDEN, ffn.ATTN_K)
            or wos is None
            or tuple(wos.shape) != (ffn.HIDDEN, 1)
        ):
            reasons.append(
                f"layer {i}: o_proj has no rowwise-FP8 copy (MXFP8 on aiter)"
            )
            break
        missing = [
            (proj, kind)
            for proj in ("gate", "up", "down")
            for kind in ("weight", "weight_scale_inv")
            if ("shared expert mxfp8", i, proj, kind) not in copies
        ]
        if missing:
            reasons.append(f"layer {i}: no checkpoint shared expert {missing}")
            break
    return reasons


def _o_proj_rowwise(layer):
    """The rowwise-FP8 o_proj SGLang keeps for decode-sized steps on gfx950: the
    (16, 16)-shuffled weight and its per-row scale."""
    w = layer.self_attn.o_proj.weight
    return getattr(w, "_ptpc_weight", None), getattr(w, "_ptpc_scale", None)


@contextmanager
def _o_proj_input(o_proj):
    """``o_proj`` returns its input while the attention runs: the launch applies it."""
    o_proj.forward = lambda x: (x, None)
    try:
        yield
    finally:
        del o_proj.forward


def _shared_mxfp8(copies: dict, i: int):
    """Layer i's shared expert in the launch's MXFP8 layout (``a16w8.layout``), from the
    checkpoint values the load hook kept: (w13, s13, w2, s2), gate / up separated."""
    from sgl_amd_mono.minimax_m31 import a16w8

    def take(proj):
        return (
            copies.pop(("shared expert mxfp8", i, proj, "weight")),
            copies.pop(("shared expert mxfp8", i, proj, "weight_scale_inv")),
        )

    (gw, gs), (uw, us), (dw, ds) = take("gate"), take("up"), take("down")
    return (
        *a16w8.layout(torch.cat([gw, uw]), torch.cat([gs, us])),
        *a16w8.layout(dw, ds),
    )


class _Built:
    """One rank's FFN launcher, every sparse layer's weights, and the forward."""

    def __init__(self, lm, runner, peer, weights: dict, sparse: list[int]):
        self.lm, self.model = lm, lm.model
        self.runner, self.peer, self.weights = runner, peer, weights
        self.first, self.sparse = sparse[0], sparse
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        lm = self.lm
        hidden, aux = self._backbone(plan, input_ids, positions, forward_batch)
        if kwargs.get("get_embedding", False):
            return hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, plan, input_ids, positions, forward_batch):
        """``MiniMaxM3Model.forward`` with each sparse layer's post-attention work as
        one launch."""
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
        from sglang.srt.layers.layer_boundary.residual.batch import (
            final_norm,
            start,
            stream_of,
        )

        m, rows = self.model, plan.key.rows
        hidden = m.embed_tokens(input_ids)
        start(forward_batch)
        aux = AuxHiddenStateList()
        for i in range(self.first):
            layer = m.layers[i]
            hidden = layer(
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden,
                captured_last_layer_outputs=aux
                if getattr(layer, "_is_layer_to_capture", False)
                else None,  # fmt: skip
            )
        slots = forward_batch.out_cache_loc
        stream = stream_of(forward_batch)
        check = self.check and not torch.cuda.is_current_stream_capturing()
        for lid in self.sparse:
            layer = m.layers[lid]
            ref = self._reference(layer, positions, forward_batch, hidden) if check else None  # fmt: skip
            call = {}
            if getattr(layer, "_norm_quant", None) is not None:
                call["quant_format"] = layer._norm_quant.attn_quant_format(rows)
            x = layer.attn_boundary.prepare(
                hidden,
                forward_batch,
                capture_gathered=aux
                if getattr(layer, "_is_layer_to_capture", False)
                else None,  # fmt: skip
                **call,
            )
            with _o_proj_input(layer.self_attn.o_proj):
                attn = layer.self_attn(
                    positions=positions, hidden_states=x, forward_batch=forward_batch
                )
            out, residual = self.runner(
                attn.contiguous(),
                stream.residual.contiguous(),
                self.weights[lid],
                slots,
            )
            if ref is not None:
                from sgl_amd_mono.minimax.check import log_check

                log_check(layer, lid, slots[:rows] > 0, (out, residual), ref, ffn.TOPK)
            # the next layer's input norm applies the residual add
            stream.write(residual)
            hidden = stream.record(out, PLAIN_ADD)
        hidden = final_norm(
            hidden,
            forward_batch,
            m.norm,
            # the DSpark draft reads the last layer's updated residual
            capture=aux.capture if m.end_layer in m.layers_to_capture else None,
            finalize_norm=m.finalize_norm,
            skip_empty=True,
        )
        return hidden, (aux if len(aux) else None)

    def _reference(self, layer, positions, forward_batch, hidden):
        """Check mode: SGLang's layer on a copy of the residual entering it, on a
        stream of its own; returns (MLP output, residual after attention) as the launch
        does. It writes the same KV slots, which the launch's layer writes again."""
        from sglang.srt.layers.layer_boundary.residual.batch import stream_of
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        entering = stream_of(forward_batch).snapshot(hidden)
        hidden, stream = ResidualStream.from_handoff(entering, None, None)
        original = forward_batch.residual_stream
        forward_batch.residual_stream = stream
        try:
            out = layer(
                positions=positions, forward_batch=forward_batch, hidden_states=hidden
            )
            return stream.export(out)
        finally:
            forward_batch.residual_stream = original


def build(model, hf_config, ctx):
    from types import SimpleNamespace

    from sglang.srt.distributed import get_tp_group

    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.minimax_m31.runner import FfnRunner

    group = get_tp_group()
    rank, device = (
        group.rank_in_group,
        torch.device("cuda", torch.cuda.current_device()),
    )
    peer = PeerBuffer(ffn.peer_bytes(TP), group.cpu_group, rank, TP, device)
    runner = FfnRunner(TP, rank, peer.local, peer.addresses, device, tuple(ctx.widths))
    runner.set_deadline(ctx.deadline_s)
    sparse = _sparse_ids(model)
    weights = {}
    for i in sparse:
        layer = model.model.layers[i]
        moe, experts = layer.mlp, layer.mlp.experts
        w13s, s13s, w2s, s2s = _shared_mxfp8(ctx.copies, i)
        wo, wos = _o_proj_rowwise(layer)
        weights[i] = SimpleNamespace(
            post_norm=layer.post_attention_layernorm.weight.data,
            router=moe.gate.weight.data.float().contiguous(),
            router_bias=moe.e_score_correction_bias.data.float().contiguous(),
            w13=experts.w13_weight.data.view(torch.float4_e2m1fn_x2),
            s13=experts.w13_weight_scale.data,
            w2=experts.w2_weight.data.view(torch.float4_e2m1fn_x2),
            s2=experts.w2_weight_scale.data,
            w13s=w13s,
            s13s=s13s,
            w2s=w2s,
            s2s=s2s,
            wo=wo,
            wos=wos.contiguous(),
        )
    # every width compiles alike; its residency is checked against the grid
    runner.warm(
        weights[sparse[0]],
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    logger.info(
        "MiniMax-M3.1 FFN monokernels: %d sparse layers, rows %s",
        len(sparse),
        tuple(ctx.widths),
    )
    return _Built(model, runner, peer, weights, sparse)


# check mode, 2 runs of 16 GSM8K prompts (49,476 and 44,004 launches); worst case:
# residual, every row: rel 0.042, cosine 0.99831
# MLP output, rows routed alike (94%): rel 0.28, cosine 0.66
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.residual.rel": 0.08,
        "layer.residual.cosine": 0.995,
        "layer.mlp_out.rel": 0.5,
        "layer.mlp_out.cosine": 0.5,
    },
)

SPEC = MonoSpec(
    name="minimax_m31_ffn_hip",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(TP,), resident_blocks=BLOCKS)
    },
    # each row's FFN is independent and reads no KV: any verify layout, any KV tier
    features=frozenset(
        {
            Feature.SPECULATIVE_DECODING,
            Feature.RAGGED_VERIFY,
            Feature.EAGLE3_HIDDEN_STATES,
            Feature.KV_OFFLOAD,
        }
    ),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}), env={"SGLANG_USE_AITER": True}
    ),
    numerics=NUMERICS,
    model_disable_reasons=model_disable_reasons,
    build=build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=config_disable_reasons,
    load_tensor=load_tensor,
)
