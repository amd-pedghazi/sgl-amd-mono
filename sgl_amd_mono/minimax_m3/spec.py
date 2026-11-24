# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 on gfx950: the monokernel spec and its forward (RFC #43949).

A routed decode or EAGLE3 verify step runs ``_Built.forward``, which takes and returns
what the model's forward does: SGLang's modules run the embedding and the leading
dense-MLP layers, each sparse MoE layer is one launch (``gfx950/runner.py``) that takes
and hands on the residual stream's pending pair (the all-reduced MLP output, its
residual add not yet applied), and SGLang's final norm and logits processor close the
step. The kernels read SGLang's weights in place except the copies logged at startup
(``weights.py``). Pad rows write KV slot 0, the reserved dummy slot, as the normal path
does. ``SGLANG_AMD_MONO_CHECK=1`` runs SGLang's modules on a copy of every sparse
layer's input in eager steps and logs both outputs' differences on the live rows; the
bounds in ``NUMERICS`` come from these runs.
"""

from __future__ import annotations

import importlib.util
import logging
import os

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.minimax_m3 import HF_CONFIG, LAYER_LIMIT, SWIGLU_ALPHA

logger = logging.getLogger(__name__)

MAX_ROWS = 16  # the rows a launch takes at most (``gfx950/config.MAX_TOKENS``)
TP = 4
# SGLang's sum(routed) / routed_scaling_factor, scaled back after the MoE, is 1 when renormalized
SHARED_WEIGHT = 1.0

# by file under sglang/ (``f(p)``: a parameter, ``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "rows",
        "tokens_per_request",
        "forward_mode",
    ),
    "srt/kernel_plugins/amd/binding.py|srt/monokernels/runtime.py": (
        '"target_verify"',
    ),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "page_size",
        "moe_runner_backend",
        "disable_shared_experts_fusion",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "enable_hierarchical_cache",
        "hicache_io_backend",
        "hicache_mem_layout",
        "enable_lmcache",
        "enable_flexkv",
        "disaggregation_decode_enable_offload_kvcache",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
    ),
    "srt/environ.py": (
        "SGLANG_USE_AITER",
        "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ",
        "SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
        "envs",
    ),
    "srt/runtime_context.py": (
        "get_parallel",
        '"tp_size"',
        '"tp_rank"',
        '"tp_group"',
        '"attn_tp_size"',
        '"moe_tp_size"',
        '"moe_ep_size"',
    ),
    "srt/distributed/parallel_state.py": ("world_size",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/minimax_sparse_backend.py": (
        "MiniMaxHybridAttnBackend",
        "sparse",
        "kv_pool",
    ),
    "srt/mem_cache/memory_pool.py": ("_wait_for_layer",),
    "srt/model_executor/forward_batch_info.py": ("residual_stream", "out_cache_loc"),
    "srt/layers/layernorm.py": ("forward_native",),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "stream_of",
        "final_norm",
        "final_norm(skip_empty)",
    ),
    "srt/layers/layer_boundary/residual/stream.py": (
        "ResidualStream",
        "from_handoff",
        "export",
        "write",
        "record",
    ),
    "srt/layers/layer_boundary/residual/__init__.py": ("is_plain_add",),
    "srt/layers/layer_boundary/*.py": ("declaration", "update"),
    "srt/layers/aux_hidden_states.py": ("AuxHiddenStateList", "capture"),
    "srt/models/minimax_m3.py": (
        "is_layer_sparse",
        '"_is_layer_to_capture"',
        "layers",
        "norm",
        "ffn_boundary",
        "embed_tokens",
        "start_layer",
        "end_layer",
        "self_attn",
        "mlp",
        "input_layernorm",
        "forward(captured_last_layer_outputs)",
        "e_score_correction_bias",
        "gate",
        "post_attention_layernorm",
        "routed_scaling_factor",
        "scaling",
    ),
    "srt/models/minimax_m3_vl.py": (
        "is_mrope_enabled",
        "logits_processor",
        "lm_head",
        "forward(get_embedding)",
    ),
    "srt/configs/minimax_vl.py": ("text_config",),
}

# check mode, 2 x ~60k launches, FP8 attention projections (SGLang's: bf16); median / p99 / worst:
# residual, every row: rel 0.006 / 0.021 / 0.063, cosine 0.99989 / p1 0.99953 / 0.99858
# MLP output, rows routed alike: rel 0.038 / 0.097 / 0.33, cosine 0.998 / p1 0.937 / 0.64
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "layer.residual.rel": 0.1,
        "layer.residual.cosine": 0.995,
        "layer.mlp_out.rel": 0.5,
        "layer.mlp_out.cosine": 0.4,
    },
)


def _text(hf_config):
    """The text model's config: the VL checkpoint nests it."""
    return getattr(hf_config, "text_config", None) or hf_config


def _has_flydsl() -> bool:
    return importlib.util.find_spec("flydsl") is not None


# switches of sgl-project/sglang#41998 (on #41708) for the page16 SHUFFLE K / V, not in main yet:
# read by name, so an SGLang without them is refused with a reason; pin in SGLANG_READS once merged
PENDING_ENV = {
    # prefill writes the page16 SHUFFLE K / V the kernels read
    "SGLANG_M3_USE_AITER_SPARSE_PA": True,
    "SGLANG_M3_USE_AITER_PREFILL_INDEXER": True,
    "SGLANG_M3_USE_AITER_INDEXER": True,
}


def _pending_env_reasons() -> list[str]:
    from sglang.srt.environ import envs

    reasons = []
    for name, want in PENDING_ENV.items():
        var = getattr(envs, name, None)
        if var is None:
            reasons.append(
                f"needs SGLang with {name} (the page16 SHUFFLE K / V of "
                "sgl-project/sglang#41998 and #41708)"
            )
        elif var.get() != want:
            reasons.append(f"{name} {var.get()!r} is not {want!r}")
    return reasons


def _sparse_ids(model) -> list[int]:
    return [i for i, layer in enumerate(model.model.layers) if layer.is_layer_sparse]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the nested text config, the draft chain,
    KV offload, FlyDSL and the SHUFFLE K / V switches (``PENDING_ENV``)."""
    from sgl_amd_mono.minimax_m3.weights import selection_reasons

    t = _text(hf_config)
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in HF_CONFIG.items()
        if getattr(t, k, None) != v
    ]
    if abs(float(getattr(t, "swiglu_alpha", 0.0)) - SWIGLU_ALPHA) > 1e-6:
        reasons.append(
            f"swiglu_alpha {getattr(t, 'swiglu_alpha', None)}, built for 1.702"
        )
    if t.num_hidden_layers > LAYER_LIMIT:
        reasons.append(
            f"{t.num_hidden_layers} layers; the mailbox tags hold {LAYER_LIMIT}"
        )
    reasons += selection_reasons(t)
    if cfg.speculative_algorithm is not None and (
        cfg.speculative_eagle_topk != 1
        or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        reasons.append(
            "EAGLE3 verifies one draft chain: topk 1, draft tokens = steps + 1"
        )
    if cfg.enable_lmcache or cfg.enable_flexkv or cfg.disaggregation_decode_enable_offload_kvcache:  # fmt: skip
        reasons.append("KV offload other than HiCache")
    if cfg.enable_hierarchical_cache and (
        cfg.hicache_io_backend != "kernel" or cfg.hicache_mem_layout != "layer_first"
    ):
        reasons.append(
            "HiCache keeps the page16 SHUFFLE K / V only with --hicache-io-backend "
            "kernel and --hicache-mem-layout layer_first"
        )
    if not _has_flydsl():
        reasons.append("needs FlyDSL")
    return reasons + _pending_env_reasons()


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.minimax_m3.weights import layer_reasons

    m, p = model.model, get_parallel()
    reasons = []
    if getattr(model, "is_mrope_enabled", False):  # the VL wrapper's rope config
        reasons.append("mrope positions; the kernels apply 1-D RoPE")
    if (p.attn_tp_size, p.moe_tp_size, p.moe_ep_size) != (TP, TP, 1):
        reasons.append(
            f"attention TP {p.attn_tp_size} / MoE TP {p.moe_tp_size} / EP "
            f"{p.moe_ep_size}: the kernels shard attention and experts over TP {TP}"
        )
    if m.start_layer != 0 or m.end_layer != len(m.layers):
        reasons.append(f"layers {m.start_layer}..{m.end_layer} of {len(m.layers)}")
    sparse = _sparse_ids(model)
    if not sparse or sparse != list(range(sparse[0], len(m.layers))):
        return reasons + [f"MoE layers {sparse} are not a trailing run of the stack"]
    if sparse[0] == 0:
        return reasons + [
            "layer 0 is a MoE layer; the first launch needs a dense layer before it"
        ]
    # the first launch's input norm applies the previous layer's pending add, which must be plain
    if not m.layers[sparse[0] - 1].ffn_boundary.declaration.update.is_plain_add:
        reasons.append(f"layer {sparse[0] - 1} hands on a non-additive residual")
    attn = m.layers[sparse[0]].self_attn
    if abs(float(attn.scaling) - 128**-0.5) > 1e-12:
        reasons.append(f"softmax scale {attn.scaling}")
    for i in sparse:
        reasons += layer_reasons(m.layers[i], i)
    return reasons


class _Built:
    """One rank's monokernels: the runner, the sparse layers, and the forward."""

    def __init__(self, lm, runner, sparse: list[int]):
        self.lm, self.model, self.runner = lm, lm.model, runner
        self.first = sparse[0]
        self.sparse = sparse
        self.backend = None  # the sparse attention backend, bound at the first forward
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        """The model's forward for a routed step: the backbone, then SGLang's
        logits processor (or the hidden states, the VL wrapper's embedding mode)."""
        lm = self.lm
        hidden, aux = self._backbone(plan, input_ids, positions, forward_batch)
        if kwargs.get("get_embedding", False):
            return hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, plan, input_ids, positions, forward_batch):
        """``MiniMaxM3Model.forward``: the embedding and the dense layers on SGLang's
        modules, every sparse layer as one launch, and SGLang's final norm."""
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
        from sglang.srt.layers.layer_boundary.residual.batch import (
            final_norm,
            start,
            stream_of,
        )

        m, r = self.model, self.runner
        if self.backend is None:
            self._bind()
        rows, query_len = plan.key.rows, plan.key.tokens_per_request
        is_verify = plan.key.forward_mode == "target_verify"
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
        stream = stream_of(forward_batch)
        # the dense stack's MLP output with its residual add owed: the kernels apply it
        ar, residual = stream.export(hidden)
        tables = r.prepare_step(forward_batch, rows, query_len, is_verify)
        slots = forward_batch.out_cache_loc
        check = self.check and not torch.cuda.is_current_stream_capturing()
        for n, lid in enumerate(self.sparse):
            # HiCache's per-layer H2D loads, as the pool's own getters wait for them
            r.pool._wait_for_layer(lid)
            ref = self._reference(lid, positions, forward_batch, ar, residual) if check else None  # fmt: skip
            ar, residual = r.launch(
                i=n, rows=rows, query_len=query_len, positions=positions,
                slot_mapping=slots, hidden=ar, residual=residual, tables=tables,
            )  # fmt: skip
            if ref is not None:
                from sgl_amd_mono.minimax.check import log_check
                from sgl_amd_mono.minimax_m3.gfx950.config import TOP_K

                log_check(
                    m.layers[lid], lid, slots[:rows] > 0, (ar, residual), ref, TOP_K
                )
            if getattr(m.layers[lid], "_is_layer_to_capture", False):
                # the pre-norm residual the attention boundary hands EAGLE3; capture() copies it
                aux.capture(r.h[:rows])
        stream.write(residual)
        last = m.layers[self.sparse[-1]]
        hidden = stream.record(ar, last.ffn_boundary.declaration.update)
        hidden = final_norm(hidden, forward_batch, m.norm, skip_empty=True)
        return hidden, (aux if len(aux) else None)

    def _reference(self, lid, positions, forward_batch, ar, residual):
        """Check mode: SGLang's layer ``lid`` on a copy of the launch's input pair.

        Runs on a residual stream of its own and returns (MLP output, residual) as the
        launch does. It writes the same KV slots, which the launch then writes again."""
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        layer = self.model.layers[lid]
        update = self.model.layers[lid - 1].ffn_boundary.declaration.update
        hidden, stream = ResidualStream.from_handoff(
            ar.clone(), residual.clone(), update
        )
        original = forward_batch.residual_stream
        forward_batch.residual_stream = stream
        try:
            out = layer(
                positions=positions, forward_batch=forward_batch, hidden_states=hidden
            )
            return stream.export(out)
        finally:
            forward_batch.residual_stream = original

    def _bind(self) -> None:
        """Check and bind SGLang's caches at the first routed forward.

        Every rank runs that forward eagerly together, after SGLang sized the caches;
        the ranks gather each other's reasons, so all refuse or none does."""
        import torch.distributed as dist
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxHybridAttnBackend,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.runtime_context import get_parallel

        backend = get_attn_backend()
        if isinstance(backend, MiniMaxHybridAttnBackend):
            backend = backend.sparse  # the sparse backend owns the MiniMax pool
        pool = backend.kv_pool
        reasons = self.runner.cache_reasons(backend, pool)
        group = get_parallel().tp_group
        every = [None] * group.world_size
        dist.all_gather_object(every, reasons, group=group.cpu_group)
        bad = [f"rank {k}: {why}" for k, rs in enumerate(every) for why in rs]
        if bad:
            raise ValueError("MiniMax-M3 monokernels: " + "; ".join(bad))
        self.runner.bind(backend, pool)
        self.backend = backend


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.minimax_m3.gfx950.config import ONE_INDEX_HEAD
    from sgl_amd_mono.minimax_m3.gfx950.runner import M3MonoRunner
    from sgl_amd_mono.minimax_m3.weights import SparseMoeLayer, sparse_selection

    t, m = _text(hf_config), model.model
    sparse = _sparse_ids(model)
    layers = [SparseMoeLayer.build(m.layers[i], i, ctx.copies) for i in sparse]
    _, init_blocks, local_blocks = sparse_selection(t)
    k4_build = dict(
        npes=TP,
        sm_scale=float(m.layers[sparse[0]].self_attn.scaling),
        eps=float(t.rms_norm_eps),
        route_scale=float(m.layers[sparse[0]].mlp.routed_scaling_factor),
        shared_weight=SHARED_WEIGHT,
        swiglu_limit=float(t.swiglu_limit),
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        heads=ONE_INDEX_HEAD,
    )
    # the peer handshake is collective: every rank gets here (attach agreed first)
    p = get_parallel()
    device = m.layers[sparse[0]].input_layernorm.weight.device
    runner = M3MonoRunner(
        tp=p.tp_size,
        rank=p.tp_rank,
        cpu_group=p.tp_group.cpu_group,
        device=device,
        layers=layers,
        k4_build=k4_build,
    )
    runner.set_deadline(ctx.deadline_s)
    # FlyDSL's disk cache hands a rank its peers' builds, so each rank starts at its own share
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // p.tp_size
    for rows in widths[start:] + widths[:start]:
        runner.precompile(rows)
    return _Built(model, runner, sparse)


SPEC = MonoSpec(
    name="minimax_m3_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
            tp_sizes=(TP,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset(
        {Feature.SPECULATIVE_DECODING, Feature.EAGLE3_HIDDEN_STATES, Feature.KV_OFFLOAD}
    ),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "page_size": frozenset({128}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # the shared expert runs as the fused expert of the MXFP4 MoE
            "disable_shared_experts_fusion": frozenset({False}),
            "speculative_algorithm": frozenset({None, "EAGLE3"}),
            # HiSparse maps the logical KV slots to device rows; the kernels write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            "SGLANG_USE_AITER": True,
            # the kernels select blocks in every layer
            "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ": 1,
            # the index-K cache the kernels score is FP8
            "SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE": True,
            # benchmark routing overrides: the kernels route with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
