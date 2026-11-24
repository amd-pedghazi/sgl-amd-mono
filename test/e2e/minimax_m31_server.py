"""The MiniMax-M3.1-Flash servers the serving tests launch: MI35x, TP4, NVFP4 experts,
with DSpark (verify steps of 8 rows a request) or without it (decode steps)."""

import os

MODEL = os.environ.get("MINIMAX_M31_FLASH_MODEL_PATH", "MiniMaxAI/MiniMax-M3.1-Flash")
ENV = {"SGLANG_USE_AITER": "1"}
COMMON = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--fp8-gemm-backend",
    "aiter",
    "--chunked-prefill-size",
    "32768",
    "--mem-fraction-static",
    "0.8",
]
# Without --monokernel-backend, so a test can launch the normal path too.
# DSpark: verify steps of 8 rows a request, 2 requests at the largest width (16 rows)
ARGS = COMMON + [
    "--speculative-algorithm",
    "DSPARK",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--cuda-graph-max-bs-decode",
    "8",
    "--max-running-requests",
    "8",
]
# a row a request: 20 requests step past the 16-row width
DECODE_ARGS = COMMON + [
    "--cuda-graph-max-bs-decode",
    "24",
    "--max-running-requests",
    "20",
]
