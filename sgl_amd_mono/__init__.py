# SPDX-License-Identifier: Apache-2.0
"""Monokernels for SGLang on AMD Instinct GPUs (RFC sgl-project/sglang#43949).

Persistent kernels run most of a decoder layer, its TP collectives included, in one
or two launches. SGLang holds the interface (``sglang.kernels.plugin_kernels.amd``);
this package holds the HIP device runtime (``common``) and, per model, its specs and
their forwards (``<model>/spec.py``, ``<model>/draft.py``) and its kernels per ISA
(``<model>/<isa>/``). SGLang loads it as a general plugin (the ``sglang.srt.plugins``
entry point in ``pyproject.toml``), which registers each spec.
"""

__version__ = "0.1.0.dev0"

# the backend name every spec registers under
BACKEND = "amd"
# each spec by HF architecture; its module imports only when a server runs this backend
SPECS = {
    "DeepseekV4ForCausalLM": "sgl_amd_mono.deepseek_v41.spec:SPEC",
    "DeepseekV4ForCausalLMDSpark": "sgl_amd_mono.deepseek_v41.draft:SPEC",
    "Glm5NextForConditionalGeneration": "sgl_amd_mono.glm53_flash.spec:SPEC",
    "GlmMoeDsaForCausalLM": "sgl_amd_mono.glm53.spec:SPEC",
    "MiniMaxM3SparseForCausalLM": "sgl_amd_mono.minimax_m3.spec:SPEC",
    "MiniMaxM3SparseForConditionalGeneration": "sgl_amd_mono.minimax_m3.spec:SPEC",
}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": ("register",)
}


def register_specs() -> None:
    """The plugin entry point, which SGLang runs in every process."""
    try:
        from sglang.kernels.plugin_kernels.amd.spec import register
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.monokernels.spec import register

    for architecture, path in SPECS.items():
        register(backend=BACKEND, architecture=architecture, path=path)
