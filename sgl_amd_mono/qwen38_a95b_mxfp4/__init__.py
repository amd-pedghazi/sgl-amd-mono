# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 MoE monokernels for amd/Qwen3.8-2.4T-A95B-Quark-MXFP4.

Routed experts are MXFP4. Attention and GDN projections stay bf16.
The spec, forward, and kernels land in the model PR.
"""
