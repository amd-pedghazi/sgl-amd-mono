# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 4cd75a2
# (ROCm/ATOM#2559), MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v4/mono/kernels/__init__.py
"""The DeepSeek-V4 mono layer kernel: ``layer`` builds the one launch from the
stages (``stages/``: ``hc``, ``qkv``, ``indexer``, ``attention``, ``ffn``) over the
device helpers ``common`` binds to it (``helpers``, ``ops``)."""
