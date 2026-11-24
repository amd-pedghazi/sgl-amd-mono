# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 4cd75a2
# (ROCm/ATOM#2559), MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v4/mono/kernels/stages/__init__.py
"""The layer kernel's stages (``layer``), a module per group: each takes the
kernel's context dict (the build's constants, the kernel arguments and the values
earlier stages returned) and returns the values it defines."""
