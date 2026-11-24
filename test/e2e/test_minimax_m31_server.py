"""MiniMax-M3.1-Flash with --monokernel-backend amd on MI35x, TP4, across the switch points.

DSpark verify steps (2 requests of 8 rows at the largest width) and decode steps without
it (16 rows): each server's startup line shows its tested batches run the FFN
launches, and a chat GSM8K run steps past the largest width and back (gross failures).
Check mode holds every sparse layer's launch to the spec's bounds against SGLang's
modules on the same input, and a fault on one rank stops the server before any token of
the failed step is sent."""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from minimax_m31_server import ARGS, DECODE_ARGS, ENV, MODEL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND
from sgl_amd_mono.minimax_m31.spec import NUMERICS

LAUNCH_TIMEOUT = 3600
# chat GSM8K, 200 questions, greedy: normal 0.975 (verify server); monokernels 0.975 and
# 0.985 (verify), 0.975 (decode). 8-shot completion prompts score under 0.1 on either path.
GSM8K_BOUND = 0.93
# check mode: rows routed to other experts than SGLang's, 6.0% and 6.1% measured
ROUTED_ELSEWHERE_BOUND = 0.1


def _launch(args=ARGS, extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args + ["--monokernel-backend", BACKEND, *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


def _logs(files) -> str:
    return "".join(Path(f.name).read_text() for f in files)


def _gsm8k(parallel):
    args = SimpleNamespace(
        base_url=DEFAULT_URL_FOR_TEST,
        model=MODEL,
        eval_name="gsm8k",
        num_examples=200,
        num_threads=parallel,
        max_tokens=4096,
        temperature=0.0,
    )
    return run_eval(args)


def _short_traffic(parallel, questions):
    """Steps to check: 8-shot GSM8K prompts, 128 tokens each (the answers do not matter)."""
    args = SimpleNamespace(
        num_shots=8,
        data_path=None,
        num_questions=questions,
        parallel=parallel,
        max_new_tokens=128,
        host="127.0.0.1",
        port=int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
    )
    run_eval_few_shot_gsm8k(args)


class _Serving:
    """A server's switch points: ``PARALLEL`` requests at a time step past the
    largest covered width as the batch fills, and back as it drains."""

    SERVER_ARGS: list
    PARALLEL: int
    RUNS: str  # the startup line: the steps the FFN launches run

    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(
            cls.SERVER_ARGS, extra_args=["--watchdog-timeout", "600"], log=cls.logs
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    def test_the_tested_batches_run_the_monokernels(self):
        # routing is fixed at startup; else the other tests could pass on the normal path alone
        self.assertIn(self.RUNS, _logs(self.logs))

    def test_gsm8k_past_the_largest_width(self):
        metrics = _gsm8k(self.PARALLEL)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)


class TestMonokernelVerify(_Serving, CustomTestCase):
    """DSpark verify steps: above 2 requests (16 rows) the normal path."""

    SERVER_ARGS, PARALLEL = ARGS, 4
    RUNS = (
        "monokernel minimax_m3_family_hip on gfx950 runs target_verify steps of batch "
        "size [1, 2] x 8 tokens"
    )


class TestMonokernelDecode(_Serving, CustomTestCase):
    """Decode steps without DSpark: above 16 rows the normal path."""

    SERVER_ARGS, PARALLEL = DECODE_ARGS, 20
    RUNS = (
        "monokernel minimax_m3_family_hip on gfx950 runs decode steps of batch size "
        f"{list(range(1, 17))}"
    )


class TestMonokernelCheck(CustomTestCase):
    def test_every_launch_is_within_the_bounds(self):
        # check mode: eager steps, every launch against SGLang's modules on a copy of its input
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                extra_args=["--disable-cuda-graph"],
                log=(out, err),
            )
            try:
                _short_traffic(parallel=2, questions=16)
            finally:
                terminate_and_kill_process_tree(process)
            stats = re.findall(
                r"monokernel check layer (\d+), (\d+) rows \((\d+) routed alike\): "
                r"mlp_out rel ([\d.]+) cosine ([\d.-]+); "
                r"residual rel ([\d.]+) cosine ([\d.-]+)",
                _logs((out, err)),
            )
        self.assertGreater(len(stats), 57 * 10, "too few checked launches")
        rows = sum(int(s[1]) for s in stats)
        alike = sum(int(s[2]) for s in stats)
        print(
            f"{len(stats)} checked launches, {rows - alike} of {rows} rows routed elsewhere"
        )
        # the MLP output on rows that route alike; the residual on every row
        values = {
            "mlp_out.rel": [float(s[3]) for s in stats if int(s[2])],
            "mlp_out.cosine": [float(s[4]) for s in stats if int(s[2])],
            "residual.rel": [float(s[5]) for s in stats],
            "residual.cosine": [float(s[6]) for s in stats],
        }
        for name, vals in values.items():
            bound = NUMERICS.bound(f"layer.{name}")
            if name.endswith("rel"):
                print(f"{name}: worst {max(vals)}, bound {bound}")
                self.assertLessEqual(max(vals), bound, name)
            else:
                print(f"{name}: worst {min(vals)}, bound {bound}")
                self.assertGreaterEqual(min(vals), bound, name)
        self.assertLessEqual(rows - alike, ROUTED_ELSEWHERE_BOUND * rows)


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # rank 1 stalls at its first wait after 1000 launches (startup runs about 230); at
        # the deadline (watchdog / 2) all stop. --watchdog-timeout 120 still leaves startup's
        # JIT builds room
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={
                    "SGLANG_AMD_MONO_FAULT_INJECT": "1:1:1000",
                    # the stop faults every GPU on purpose: no crash dumps (each
                    # rank's VRAM), which would hold the ranks up for minutes
                    "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
                },
                extra_args=["--watchdog-timeout", "120"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 2048,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=600,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # a failed wait stops the launch with a null-page store: faults only at 0x0 / (nil)
                faults = re.findall(
                    r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))", log
                )
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0", "(nil)"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
