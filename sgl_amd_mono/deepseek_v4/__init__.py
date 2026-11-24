# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Pro monokernels (RFC #43949): the spec and its SGLang glue in ``spec``,
and the gfx950 kernels and host runner in ``gfx950``.

Adapted from ROCm/ATOM's DeepSeek-V4 mono decode (ROCm/ATOM#2559, at 4cd75a2; MIT
License, Copyright (C) 2026, Advanced Micro Devices, Inc.); each file names its origin.
The checkpoint the kernels build in is ``gfx950.config.V4_PRO_CONFIG``.
"""
