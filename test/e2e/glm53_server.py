"""The GLM-5.3 servers that the monokernel serving test launches: MI35x, TP8,
decode steps, and with ``MTP`` the NextN draft's verify steps."""

import os

MODEL = os.environ.get("GLM_53_MODEL_PATH", "zai-org/GLM-5.3")
ENV = {"SGLANG_USE_AITER": "1", "AITER_ONLINE_TUNE": "0"}
# The cookbook's MI355X low-latency cell without MTP, capped at 64 running requests.
# Without --monokernel-backend, so a test can launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "8",
    "--mem-fraction-static",
    "0.80",
    "--max-running-requests",
    "64",
    "--cuda-graph-max-bs-decode",
    "64",
    "--chunked-prefill-size",
    "131072",
    "--dsa-prefill-backend",
    "tilelang",
    "--dsa-decode-backend",
    "tilelang",
    "--reasoning-parser",
    "glm45",
]
# SGLang's MTP default for GLM-5.3: one draft chain of 3 tokens, 4 verify rows a
# request
MTP = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
