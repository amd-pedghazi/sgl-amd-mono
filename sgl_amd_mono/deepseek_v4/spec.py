# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Pro on gfx950: the monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed decode or DSpark
verify step runs ``_Built.forward`` here, which takes and returns what
``DeepseekV4ForCausalLM.forward`` does: SGLang's input preparation, the embedding and
its mHC broadcast, every decoder layer in one launch (``gfx950/kernels/layer.py``),
the head's mHC combine and norm, and SGLang's logits processor.

A launch runs the whole layer: both mHC mixes, the attention with its KV compressor
and, on a CSA layer, the lightning indexer (its compressor, scores and an exact
top-k), the sparse attention, the output projections, the router (hash-routed on
layers 0-2), the shared and routed experts, and both all-reduces. It reads and writes
SGLang's caches in place (``gfx950/runner.py``), so a request can take either path
from one step to the next: the two-pool fp8 unified KV, the compressor and indexer
states and the FP4 index keys.

The routed experts are SGLang's MXFP4 bank, read in place (aiter's shuffled gate /
up interleaved layout, checked against the checkpoint's expert 0 at load) and
computed as SGLang's decode computes them, on MXFP8 activations. The rest of each
layer (about 0.1 GB a rank) is packed from the checkpoint at build, before SGLang
sizes the KV cache, and logged. A launch holds at most 8 rows: a decode step of up to
8 requests, or with DSpark one request's verify; a wider step runs SGLang's path.
"""

from __future__ import annotations

import importlib.util
import os
import time

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
from sgl_amd_mono.deepseek_v4.gfx950.config import MAX_TOKENS

# the step rows a launch is built for: a decode step pads to 1, 2, 4 or 8 requests,
# and a DSpark verify of 7 tokens (block size 6) is one request's
WIDTHS = (1, 2, 4, 7, 8)
# the longest context the indexer's top-k registers cover
MAX_CONTEXT = 1 << 20

# The SGLang names this glue reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/arg_groups/fields/*.py": (
        "kv_cache_dtype",
        "page_size",
        "enable_deepseek_v4_fp4_indexer",
        "moe_runner_backend",
        "speculative_algorithm",
        "speculative_num_draft_tokens",
        "enable_encoder_swa_bounded_replay",
        "expert_distribution_recorder_mode",
        "load_format",
        "disaggregation_mode",
        "disaggregation_transfer_backend",
    ),
    "srt/environ.py": (
        "SGLANG_HACK_FLASHMLA_BACKEND",
        "SGLANG_DSV4_UNIFIED_KV_FP8",
        "SGLANG_DSV4_FP4_EXPERTS",
        "SGLANG_DSV4_FP4_DEQUANT",
        "SGLANG_USE_AITER_MOE_GU_ITLV",
        "SGLANG_DSV4_COMPRESS_STATE_DTYPE",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/mem_cache/deepseek_v4_memory_pool.py": ("resolve_compress_state_ring_size",),
    "srt/model_executor/forward_context.py": ("get_token_to_kv_pool",),
    "srt/runtime_context.py": (
        "get_parallel",
        "get_spec",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "get_attn_tp_context",
        "maybe_input_scattered",
    ),
    "kernels/ops/layernorm/mhc.py": ("hc_broadcast",),
    "srt/layers/quantization/fp8.py": (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        '"w2_weight_scale_inv"',
        "is_shuffled",
    ),
    "srt/layers/logits_processor.py": ("forward(hidden_states_before_norm)",),
    # DeepseekV4ForCausalLM.forward, which the routed forward stands in for
    "srt/models/deepseek_v4.py": (
        "prepare_language_model_inputs",
        "logits_processor",
        "lm_head",
        "capture_aux_hidden_states",
        "model",
        "embed_tokens",
        "hc_mult",
        "hc_head",
        "hc_head_fn",
        "hc_head_scale",
        "hc_head_base",
        "hc_pre_from_prev_sublayer",
        "norm",
        "layers",
        "dspark_layers_to_capture",
        "self_attn",
        "mlp",
        "freqs_cis",
        "compress_ratio",
    ),
    "srt/models/deepseek_v2.py": ("experts",),
}

# Per launch output against SGLang's layer on the same GPU and inputs
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # the layer's residual streams out (the kernel rounds as the original V4 path)
        "layer.x_out.cosine": 0.999,
        "layer.x_out.rel": 0.05,
    },
)


def _tokens_per_request() -> int:
    """Rows a request takes in a served step: a DSpark verify's, or 1."""
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1
    return int(spec.speculative_num_draft_tokens or 1)


def _max_seq(hf_config) -> int:
    """The positions the kernels size their compressed caches for: SGLang's context."""
    return min(int(getattr(hf_config, "context_len", MAX_CONTEXT)), MAX_CONTEXT)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the checkpoint's shapes and schedule,
    aiter's activation rule, the verify width and FlyDSL."""
    from sgl_amd_mono.deepseek_v4.gfx950.config import checkpoint_refusal

    reasons = []
    mode = cfg.disaggregation_mode
    if mode not in (None, "null"):
        if mode != "decode":
            reasons.append("PD monokernels support the decode role only")
        elif cfg.disaggregation_transfer_backend != "mori":
            reasons.append("PD monokernels require the mori transfer backend")
    why = checkpoint_refusal(hf_config)
    if why is not None:
        reasons.append(f"the kernels are built for DeepSeek-V4-Pro: {why}")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    if os.environ.get("AITER_BF16_FP8_MOE_BOUND") != "0":
        reasons.append(
            "SGLang's decode must run the routed experts on MXFP8 activations, as the "
            "kernels do (AITER_BF16_FP8_MOE_BOUND=0)"
        )
    if os.environ.get("AITER_FORCE_A8W4", "").lower() in ("1", "true"):
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout, not "
            "AITER_FORCE_A8W4"
        )
    if cfg.load_format == "dummy":
        reasons.append("the kernels pack the attention weights from the checkpoint")
    if cfg.speculative_algorithm is not None:
        k = int(cfg.speculative_num_draft_tokens or 1)
        if not 2 * k > MAX_TOKENS >= k:
            reasons.append(
                f"a verify of {k} tokens a request: a launch takes one request's "
                f"verify of {MAX_TOKENS // 2 + 1} to {MAX_TOKENS} tokens"
            )
    if int(getattr(hf_config, "context_len", 0) or 0) > MAX_CONTEXT:
        reasons.append(
            f"a context of {hf_config.context_len}: the indexer covers {MAX_CONTEXT}"
        )
    return reasons


def _expert_bank(layer, cfg) -> dict:
    """Layer ``layer``'s routed experts as SGLang holds them (aiter's shuffled MXFP4,
    gate / up interleaved; a fused shared expert past them is left out), viewed in the
    kernel's terms: no copy."""
    e, inter, hidden = cfg.n_experts, cfg.inter, cfg.hidden
    fm = layer.mlp.experts

    def pad8(n):
        return -(-n // 8) * 8

    def scales(t, rows, cols):
        s = t.data.view(torch.uint8)
        return s.reshape(-1, s.shape[-1])[: e * rows].view(e, rows, cols)

    return {
        "w_ug": fm.w13_weight.data.view(torch.uint8)[:e],
        "s_ug": scales(fm.w13_weight_scale_inv, 2 * inter, pad8(hidden // 32)),
        "w_dn": fm.w2_weight.data.view(torch.uint8)[:e],
        "s_dn": scales(fm.w2_weight_scale_inv, hidden, pad8(inter // 32)),
    }


def _expert_reasons(model, hf_config, ck) -> list[str]:
    """The routed experts the kernels read in place: aiter's shuffled MXFP4 bank of the
    checkpoint's shapes, its expert 0 (first and last layer) byte for byte the
    checkpoint's, laid out by the kernels' packer."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v4.gfx950.packing import (
        pack_mxfp4,
        pack_mxfp4_scales,
    )
    from sgl_amd_mono.deepseek_v4.gfx950.weights import config_for_layer, load_expert

    p = get_parallel()
    reasons = []
    n = hf_config.num_hidden_layers
    for i in range(n):
        fm = model.model.layers[i].mlp.experts
        if (
            not (
                getattr(fm.w13_weight, "is_shuffled", False)
                and getattr(fm.w2_weight, "is_shuffled", False)
            )
            or fm.w13_weight.element_size() != 1
        ):
            reasons.append(
                f"layer {i}: the routed experts are not aiter's shuffled MXFP4"
            )
            return reasons
    for i in (0, n - 1):
        cfg = config_for_layer(ck.path, i, p.tp_size)
        try:
            bank = _expert_bank(model.model.layers[i], cfg)
        except RuntimeError as err:
            return reasons + [f"layer {i}: the routed experts' views: {err}"]
        dev = bank["w_ug"].device
        ug, ugs, dn, dns = load_expert(ck, i, 0, p.tp_rank, p.tp_size, dev)
        want = {
            "w_ug": pack_mxfp4(ug[None], True)[0],
            "s_ug": pack_mxfp4_scales(ugs[None], True)[0],
            "w_dn": pack_mxfp4(dn[None], False)[0],
            "s_dn": pack_mxfp4_scales(dns[None], False)[0],
        }
        for name, w in want.items():
            if tuple(bank[name][0].shape) != tuple(w.shape) or not torch.equal(
                bank[name][0], w
            ):
                reasons.append(
                    f"layer {i} expert 0 {name}: SGLang's expert layout is not the one "
                    "the kernels read"
                )
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sgl_amd_mono.deepseek_v4.gfx950.weights import Checkpoint

    path = getattr(hf_config, "_name_or_path", None)
    try:
        ck = Checkpoint(path)
    except Exception as err:  # every rank reports it rather than crash on one
        return [
            f"the kernels pack weights from the checkpoint, which {path!r} is not: {err}"
        ]
    m = model.model
    reasons = []
    if getattr(m, "hc_pre_from_prev_sublayer", False) or m.hc_head_fn is None:
        reasons.append(
            "needs DeepSeek-V4's mHC: the same sublayer's pre and the hc head"
        )
    need = _max_seq(hf_config)
    for i in range(hf_config.num_hidden_layers):
        a = m.layers[i].self_attn
        if a.freqs_cis.dtype != torch.complex64 or a.freqs_cis.shape[0] < need:
            reasons.append(
                f"layer {i}: a RoPE table {a.freqs_cis.dtype} "
                f"{tuple(a.freqs_cis.shape)}, the kernels read {need} positions"
            )
            break
    return reasons + _expert_reasons(model, hf_config, ck)


def _rope(freqs_cis: torch.Tensor, tables: list) -> tuple:
    """The kernels' RoPE tables, f32 cos and sin [positions, rope_dim / 2], from SGLang's
    complex64 table; equal tables share one copy (``tables``: (freqs_cis, cos, sin))."""
    for f, cos, sin in tables:
        if f.shape == freqs_cis.shape and torch.equal(f, freqs_cis):
            return cos, sin
    cos = freqs_cis.real.float().contiguous()
    sin = freqs_cis.imag.float().contiguous()
    tables.append((freqs_cis, cos, sin))
    return cos, sin


def _layers(model, hf_config, ck, copies: dict) -> list:
    """Every layer's packed weights (this rank's shard, from the checkpoint), SGLang's
    routed experts in place, and its RoPE tables."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v4.gfx950.packing import pack_layer_weights
    from sgl_amd_mono.deepseek_v4.gfx950.runner import Layer
    from sgl_amd_mono.deepseek_v4.gfx950.weights import load_layer

    p = get_parallel()
    m = model.model
    dev = m.layers[0].input_layernorm.weight.device
    max_seq, tables, out = _max_seq(hf_config), [], []
    threads = torch.get_num_threads()
    torch.set_num_threads(4)  # 8 ranks x all cores oversubscribe the node
    try:
        for i in range(hf_config.num_hidden_layers):
            W = load_layer(
                ck, i, rank=p.tp_rank, tp=p.tp_size, device=dev, experts=False
            )
            cfg = W.cfg
            cfg.max_seq = max_seq
            cfg.validate()
            packed = pack_layer_weights(W.t, _expert_bank(m.layers[i], cfg))
            # the raw copies of what was packed are dead weight
            t = {k: v for k, v in W.t.items() if k not in packed}
            hc_sb = {
                side: torch.cat(
                    [
                        t.pop(f"hc_{side}_scale").float(),
                        t.pop(f"hc_{side}_base").float(),
                    ]
                ).contiguous()
                for side in ("attn", "ffn")
            }
            t.update(packed)
            for name, tensor in list(t.items()) + list(hc_sb.items()):
                if name not in ("w_ug", "s_ug", "w_dn", "s_dn"):  # SGLang's, in place
                    copies["packed layer weights", i, name] = tensor
            cos, sin = _rope(m.layers[i].self_attn.freqs_cis, tables)
            out.append(Layer(i, cfg, t, hc_sb, cos, sin, use_hash="tid2eid" in t))
    finally:
        torch.set_num_threads(threads)
    for n, (_, cos, sin) in enumerate(tables):
        copies["rope table", n, "cos"], copies["rope table", n, "sin"] = cos, sin
    return out


class _Built:
    """One rank's monokernels: the runner and the forward."""

    def __init__(self, lm, runner):
        self.lm, self.model, self.runner = lm, lm.model, runner

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__
    ):
        """``DeepseekV4ForCausalLM.forward`` for a routed step: SGLang's input
        preparation and logits processor around the backbone."""
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        lm = self.lm
        input_ids, input_embeds = lm.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            out = self._backbone(plan, input_ids, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            out, aux = out
        hidden, pre_hc_head = out
        return lm.logits_processor(
            input_ids,
            hidden,
            lm.lm_head,
            forward_batch,
            aux,
            hidden_states_before_norm=None if aux is not None else pre_hc_head,
        )

    def _backbone(self, plan, input_ids, forward_batch, input_embeds):
        """``DeepseekV4Model.forward``: the embedding and its mHC broadcast, each layer
        in one launch, then the hc head and the norm."""
        from sglang.kernels.ops.layernorm.mhc import hc_broadcast
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        m, runner = self.model, self.runner
        if runner.pool is None:  # the caches, at the first forward (eager)
            pool = get_token_to_kv_pool()
            why = runner.cache_reasons(pool)
            if why:
                raise RuntimeError("DeepSeek-V4 monokernels: " + "; ".join(why))
            runner.bind(pool)
        rows = plan.key.rows
        x = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        h = hc_broadcast(x, m.hc_mult)
        inp = runner.step_inputs(forward_batch, input_ids, rows)
        # DSpark reads the mean of each target layer's output residual
        targets = m.dspark_layers_to_capture
        aux = None if targets is None else []
        for L in runner.layers:
            h = runner.launch(L, h, inp)
            if aux is not None and L.i in targets:
                aux.append(h.mean(dim=1))
        runner.end_step(rows)
        pre_hc_head = h.flatten(1)
        hidden = m.norm(m.hc_head(h, m.hc_head_fn, m.hc_head_scale, m.hc_head_base))
        if aux is not None:
            return (hidden, pre_hc_head), aux
        return hidden, pre_hc_head


def _build(model, hf_config, ctx) -> _Built:
    import logging

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        resolve_compress_state_ring_size,
    )
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v4.gfx950.runner import Dsv4MonoRunner
    from sgl_amd_mono.deepseek_v4.gfx950.weights import Checkpoint

    log = logging.getLogger(__name__)
    p = get_parallel()
    t0 = time.perf_counter()
    layers = _layers(model, hf_config, Checkpoint(hf_config._name_or_path), ctx.copies)
    log.info(
        "DeepSeek-V4 monokernels: rank %d packed %d layers in %.0f s",
        p.tp_rank,
        len(layers),
        time.perf_counter() - t0,
    )
    tokens = _tokens_per_request()
    ratios = sorted({L.ratio for L in layers if L.ratio})
    runner = Dsv4MonoRunner(
        tp=p.tp_size,
        rank=p.tp_rank,
        cpu_group=p.tp_group.cpu_group,
        device=layers[0].cos.device,
        layers=layers,
        spec_steps=tokens - 1,
        c_rings={r: resolve_compress_state_ring_size(r) for r in ratios},
        tokens_per_request=tokens,
    )
    runner.set_deadline(ctx.deadline_s)
    widths = list(ctx.widths)
    # the peer handshakes are collective: every rank builds the widths in one order
    for S in widths:
        runner.build_width(S)
    # FlyDSL's disk cache hands a rank the kernels another rank built (a lock per
    # kernel), so each rank starts compiling at its own share
    start = len(widths) * p.tp_rank // p.tp_size
    for S in widths[start:] + widths[:start]:
        runner.precompile(S)
    return _Built(model, runner)


SPEC = MonoSpec(
    name="deepseek_v4_hip",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(8,), resident_blocks=BLOCKS)
    },
    # MoRI transfers into the same KV and compressor-state buffers the kernels use.
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.PD_DISAGGREGATION}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsv4"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "page_size": frozenset({256}),
            "enable_deepseek_v4_fp4_indexer": frozenset({True}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # DSpark's target steps are uniform verify steps; other drafts differ
            "speculative_algorithm": frozenset({None, "DSPARK"}),
            # the SWA ring of the unified KV, not a request window
            "enable_encoder_swa_bounded_replay": frozenset({False}),
            # the launches record no expert distribution (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        hf_config={"model_type": "deepseek_v4"},
        env={
            # the two-pool fp8 unified KV (512 B nope + 128 B rope a row)
            "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
            "SGLANG_DSV4_UNIFIED_KV_FP8": True,
            # the MXFP4 experts in aiter's gate / up interleaved layout
            "SGLANG_DSV4_FP4_EXPERTS": True,
            "SGLANG_DSV4_FP4_DEQUANT": False,
            "SGLANG_USE_AITER_MOE_GU_ITLV": True,
            # the compressor states the kernels pool in f32
            "SGLANG_DSV4_COMPRESS_STATE_DTYPE": "float32",
            # benchmark routing overrides: the launches route with the model's router
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
