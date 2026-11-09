# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/__init__.py (changed: docstring)
"""Hand-written gfx950 (MI355X) decode kernels for DeepSeek-V4.1-Flash at TP4:
the decoder layers' decode steps of 17-256 rows (``decode.AsmDecodeLayer``)
next to the mono decode layer, which keeps those of up to 16. The kernels are
generated as assembly at first use, built with LLVM (clang, ld.lld) into code
objects cached under $SGLANG_AMD_MONO_ASM_CACHE (default ~/.cache/sgl_amd_mono/dsv41_asm) and launched through HIP."""
