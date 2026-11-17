# SPDX-License-Identifier: Apache-2.0
"""The monokernel spec of the MiniMax-M3 architecture (``MiniMaxM3Sparse*``), which
MiniMax-M3 (Quark MXFP4, EAGLE3; ``minimax_m3``) and MiniMax-M3.1-Flash (NVFP4 experts,
DSpark; ``minimax_m31``) share. SGLang registers one spec an architecture, so this one
selects each checkpoint's own: its requirements, load hook, checks and build.

Only what both models need is stated in ``requires``; each model's own requirements are
checked in ``config_disable_reasons`` as SGLang checks a spec's.
"""

from __future__ import annotations

from sglang.srt.monokernels.spec import MonoSpec, Requirements

from sgl_amd_mono.minimax_m3 import spec as m3
from sgl_amd_mono.minimax_m31 import spec as m31

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/monokernels/spec.py": ("MonoSpec", "Requirements"),
    "srt/monokernels/guards.py": ("features_in_use",),
    "srt/arg_groups/overrides.py": ("attention_backends_of",),
    "srt/environ.py": ("envs",),
}


def _requirement_reasons(spec: MonoSpec, cfg, hf_config) -> list[str]:
    """``spec``'s requirements and features, as SGLang's monokernel guard checks them
    (``monokernel_disable_reasons``); dtype and TP are this spec's."""
    from sglang.srt.arg_groups.overrides import attention_backends_of
    from sglang.srt.environ import envs
    from sglang.srt.monokernels.guards import features_in_use

    req = spec.requires
    reasons = [
        f"{f.value} is in use and {spec.name} does not accept it"
        for f in sorted(features_in_use(cfg) - spec.features, key=lambda f: f.name)
    ]
    _, decode_backend = attention_backends_of(cfg)
    one_of = [("decode attention backend", decode_backend, req.attention_backends)]
    one_of += [(arg, getattr(cfg, arg), ok) for arg, ok in req.server_args.items()]
    exactly = [
        (f"config.{key}", getattr(hf_config, key, None), want)
        for key, want in req.hf_config.items()
    ] + [(name, getattr(envs, name).get(), want) for name, want in req.env.items()]
    reasons += [
        f"{what} {value!r} is not one of {sorted(map(str, allowed))}"
        for what, value, allowed in one_of
        if allowed and value not in allowed
    ]
    reasons += [
        f"{what} {value!r} is not {want!r}"
        for what, value, want in exactly
        if value != want
    ]
    return reasons


def _model(hf_config):
    return m31 if m31.is_m31(hf_config) else m3


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    spec = _model(hf_config).SPEC
    reasons = _requirement_reasons(spec, cfg, hf_config)
    if spec.config_disable_reasons is not None:
        reasons += spec.config_disable_reasons(cfg, hf_config)
    return reasons


def _load_tensor(copies, model, name, tensor) -> None:
    load = _model(model.config).SPEC.load_tensor
    if load is not None:
        load(copies, model, name, tensor)


def _model_disable_reasons(model, hf_config, copies) -> list[str]:
    return _model(hf_config).SPEC.model_disable_reasons(model, hf_config, copies)


def _build(model, hf_config, ctx):
    return _model(hf_config).SPEC.build(model, hf_config, ctx)


# SGLang routes both models by these limits; test_minimax_m31_spec checks they agree
SPEC = MonoSpec(
    name="minimax_m3_family_hip",
    devices=m3.SPEC.devices,
    features=m3.SPEC.features | m31.SPEC.features,
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}), env={"SGLANG_USE_AITER": True}
    ),
    numerics=m3.SPEC.numerics,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
    load_tensor=_load_tensor,
)
