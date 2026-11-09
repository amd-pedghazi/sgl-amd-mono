# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/__init__.py (unchanged)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""gfx950 asm kernel generators (Python → AMDGPU assembly text), built and launched by ``kmod`` / ``build``."""
