"""The Kimi-K3 gfx950 monokernels against SGLang's ops (RFC #43949).

Each check runs under torchrun at TP8 (``kimi_k3/check_*.py``). ``check_k2``
covers the layer launch: its o_proj and attention-TP reduce, its
attention-residual seam, its MoE, and then a graph replay. ``check_moe``
covers the MoE launch on its own, on both of aiter's expert layouts.

TP8 is not a choice here. The all-reduce and the expert sharding are built for
it, so these need 8 GPUs where DeepSeek-V4.1's need 4."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


CHECKS = Path(__file__).with_name("kimi_k3")


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonoKimiK3Kernels(CustomTestCase):
    def _check(self, module: str, *args: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=8",
            "--master-port=29608",
            str(CHECKS / f"{module}.py"),
            *args,
        ]
        result = subprocess.run(cmd, timeout=1800)
        self.assertEqual(result.returncode, 0, f"{module} at TP8")

    def test_layer_launch(self):
        self._check("check_k2", "--replays", "50")

    def test_moe_launch(self):
        self._check("check_moe")


if __name__ == "__main__":
    unittest.main()
