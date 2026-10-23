# SPDX-License-Identifier: Apache-2.0
"""A fault on one rank stops every rank's launch (RFC #43949 section 8), not only its own GPU's.

An injected fault (``SGLANG_AMD_MONO_FAULT_INJECT``) stalls rank 1's K2 or FFN launch at a
wait; at the deadline rank 1 stops and flags its peers, so each rank must stop within one more
deadline without finishing the step, and no token of it can be sent. The harness spawns and
waits for each rank itself: a torchrun-style launcher tears the group down when a rank dies,
which would hide whether each rank stopped on its own."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase

CHECKS = Path(__file__).with_name("deepseek_v41")
LAUNCHES = 12  # a clean run's launches; more than the latest injected `after`
DEADLINE_S = 6.0  # the worker's short wait deadline (check_guard.SHORT_DEADLINE_S)
# host slack over one deadline for a peer to see the flag and tear down
STOP_MARGIN_S = 15.0


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


@unittest.skipIf(_gfx950_gpus() < 4, "needs 4 gfx950 GPUs")
class TestGuardMultiRank(CustomTestCase):
    def _spawn(self, tp, launch, mode, fault, hard_timeout):
        """Spawn one worker a rank and wait for each on its own; return per-rank dicts of the
        stop time in seconds from the spawn (absent for a rank killed at the hang limit), the
        return code and the completed launches."""
        progress = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, progress, ignore_errors=True)
        procs = {}
        for r in range(tp):
            env = {**os.environ}
            if fault:
                env["SGLANG_AMD_MONO_FAULT_INJECT"] = fault  # rank:site:after
            procs[r] = subprocess.Popen(
                [sys.executable, str(CHECKS / "check_guard.py"),
                 "--launch", launch, "--mode", mode,
                 "--rank", str(r), "--world-size", str(tp),
                 "--init-file", str(progress / "init"), "--launches", str(LAUNCHES),
                 "--progress-dir", str(progress)],
                env=env,
            )  # fmt: skip
        t0 = time.monotonic()
        stop = {}
        while len(stop) < tp and time.monotonic() - t0 < hard_timeout:
            for r, p in procs.items():
                if r not in stop and p.poll() is not None:
                    stop[r] = time.monotonic() - t0
            time.sleep(0.1)
        # the hang limit only: a rank that never stopped is killed
        for r, p in procs.items():
            if r not in stop:
                p.kill()
                p.wait()
        done = {}
        for r in range(tp):
            f = progress / f"rank{r}.log"
            done[r] = sum(1 for _ in open(f)) if f.exists() else 0
        rc = {r: procs[r].returncode for r in range(tp)}
        return stop, rc, done

    def _assert_fault(self, stop, rc, done, tp, after, tag):
        self.assertEqual(
            set(stop),
            set(range(tp)),
            f"{tag}: a rank hung {sorted(set(range(tp)) - set(stop))}",
        )
        for r in range(tp):
            self.assertGreaterEqual(
                done[r], after, f"{tag}: rank {r} ran {done[r]} < {after}"
            )
            self.assertLess(
                done[r], LAUNCHES, f"{tag}: rank {r} finished all {done[r]}"
            )
            self.assertNotEqual(rc[r], 0, f"{tag}: rank {r} exited 0")
        bound = stop[1] + DEADLINE_S + STOP_MARGIN_S  # the faulted rank is rank 1
        for r in range(tp):
            self.assertLessEqual(
                stop[r],
                bound,
                f"{tag}: rank {r} stopped at {stop[r]:.1f}s > {bound:.1f}s",
            )

    def test_clean_run_finishes_every_rank(self):
        # satisfied cross-rank waits: no fault, every rank finishes every K2 launch and exits 0
        stop, rc, done = self._spawn(4, "k2", "launch", None, hard_timeout=900)
        self.assertEqual(set(stop), set(range(4)), "a rank hung on the clean run")
        for r in range(4):
            self.assertEqual(rc[r], 0, f"rank {r} clean exit {rc[r]}")
            self.assertEqual(done[r], LAUNCHES, f"rank {r} did {done[r]} launches")

    def test_fault_stops_every_rank_tp4(self):
        # early (after 3) and late (after 6) faults, on K2 and the FFN launch, in either mode
        for launch in ("k2", "ffn"):
            for after in (3, 6):
                for mode in ("launch", "step"):
                    stop, rc, done = self._spawn(
                        4, launch, mode, f"1:1:{after}", hard_timeout=420
                    )
                    self._assert_fault(
                        stop, rc, done, 4, after, f"{launch}/{mode}/after{after}"
                    )

    def test_fault_stops_every_rank_tp2_cross_gpu(self):
        # the other rank sees rank 1's flags at its next wait and stops on its own
        stop, rc, done = self._spawn(2, "k2", "launch", "1:1:3", hard_timeout=420)
        self._assert_fault(stop, rc, done, 2, 3, "tp2/k2/after3")


if __name__ == "__main__":
    unittest.main()
