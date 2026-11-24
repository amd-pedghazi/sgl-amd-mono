# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash decode FFN monokernel on gfx950: one persistent launch per sparse
layer from the attention output to the reduced MoE output (o_proj, attention TP
all-reduce and residual add, Gemma post-attention norm, FP32 router and top-4, the
shared and routed A16W4 experts, MoE TP all-reduce).

The kernel comes from vLLM (Apache-2.0, Copyright contributors to the vLLM project) at
vllm-project/vllm@a50259bb325366e7dd7c24461e1eb48e9b9ed7cf:
``vllm/models/minimax_m3/amd/specialized_decode/mono``. It builds on the plugin's
``common`` primitives, from which vLLM's DeepSeek-V4.1 copies were taken.
"""
