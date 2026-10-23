"""The MiniMax-M3 servers the serving tests launch: MI35x, TP4, MXFP4 experts,
with EAGLE3 (verify steps of 4 rows a request) or without it (decode steps, with
HiCache)."""

import os

MODEL = os.environ.get("MINIMAX_M3_MODEL_PATH", "amd/MiniMax-M3-MXFP4")
DRAFT = os.environ.get("MINIMAX_M3_EAGLE3_PATH", "Inferact/MiniMax-M3-EAGLE3-GQA")
# The MI355X recipe's environment (the GSM8K gate's server)
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
    "SGLANG_OPT_MINIMAX_M3_FP8_INDEX_CACHE": "1",
    "SGLANG_M3_ALLOW_CUSTOM_AR": "1",
    "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
    "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "0",
    # the kernels select blocks in every layer
    "SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ": "1",
}
COMMON = [
    "--trust-remote-code",
    "--quantization",
    "quark",
    "--tp",
    "4",
    "--attention-backend",
    "aiter",
    "--moe-runner-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "128",
    "--context-length",
    "131072",
    "--mem-fraction-static",
    "0.85",
    "--enable-aiter-allreduce-fusion",
]
# Without --monokernel-backend, so a test can launch the normal path too.
# EAGLE3: verify steps of 4 rows a request, 4 requests at the largest width (16 rows)
ARGS = COMMON + [
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    DRAFT,
    "--speculative-draft-attention-backend",
    "aiter",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--cuda-graph-max-bs-decode",
    "8",
    "--max-running-requests",
    "8",
]
# a row a request: 24 requests step past the 16-row width; HiCache loads layers during launches
DECODE_ARGS = COMMON + [
    "--cuda-graph-max-bs-decode",
    "24",
    "--max-running-requests",
    "24",
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "2",
    "--hicache-io-backend",
    "kernel",
    "--hicache-mem-layout",
    "layer_first",
]
