"""The DeepSeek-V4.1 gfx950 monokernels against SGLang's ops (RFC #43949).

Each ``deepseek_v41/check_*.py`` (K1, K2, the FFN launch on both routes, the sampled Markov block)
under torchrun at TP2 and TP4: outputs within the spec's bound, bitwise graph replays with every
rank agreeing; and ``check_glue.py`` (the step glue) on one GPU."""

import os
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


CHECKS = Path(__file__).with_name("deepseek_v41")
# The PR test (MONO_CI_QUICK_KERNELS=1) runs K1, K2 and the target FFN launch at
# these widths, not all 48: both edges, each side of the pad row (from 6) and of
# token parallel (from 12), and a two-request verify step. The draft route keeps
# its own 8 widths (DRAFT_WIDTHS). The nightly runs every width.
QUICK_WIDTHS = "1,2,5,6,7,11,12,13,24,47,48"


def _widths() -> tuple:
    return ("--widths", QUICK_WIDTHS) if os.environ.get("MONO_CI_QUICK_KERNELS") else ()


@unittest.skipIf(_gfx950_gpus() < 4, "needs 4 gfx950 GPUs")
class TestMonoDeepseekV41Kernels(CustomTestCase):
    def _check(self, module: str, *args: str) -> None:
        for tp in (2, 4):
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={tp}",
                f"--master-port={29600 + tp}",
                str(CHECKS / f"{module}.py"),
                *args,
            ]
            result = subprocess.run(cmd, timeout=1800)
            self.assertEqual(result.returncode, 0, f"{module} at TP{tp}")

    def test_ffn_launch(self):
        self._check("check_ffn", "--replays", "200", *_widths())

    def test_ffn_launch_draft_route(self):
        self._check("check_ffn", "--replays", "100", "--draft")

    def test_k1(self):
        self._check("check_k1", "--trials", "2", *_widths())

    def test_k1_preshuffled(self):
        self._check("check_k1", "--trials", "2", "--shuffled", *_widths())

    def test_k2(self):
        self._check("check_k2", "--replays", "100", *_widths())

    def test_k2_preshuffled(self):
        self._check("check_k2", "--replays", "100", "--shuffled", *_widths())

    def test_sampled_markov_block(self):
        self._check("check_markov", "--replays", "20")

    def test_step_glue(self):
        cmd = [sys.executable, str(CHECKS / "check_glue.py"), "--trials", "8"]
        self.assertEqual(subprocess.run(cmd, timeout=1800).returncode, 0, "check_glue")


if __name__ == "__main__":
    unittest.main()
