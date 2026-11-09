# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/build.py (changed: cache directory, docstring)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Assemble and link the generated gfx950 kernels, and reach HIP through ctypes.

Kernels are generated per process at first use (some depend on the TP rank or the model's max length), assembled
with clang and linked with ld.lld into code objects cached by source hash under $SGLANG_AMD_MONO_ASM_CACHE (default ~/.cache/sgl_amd_mono/dsv41_asm).
vLLM's TP workers build the same kernels at once: every build writes per-process files and renames atomically,
and the runner construction is serialised with ``build_lock()`` so one rank builds and the others hit the cache."""

import contextlib
import ctypes
import fcntl
import glob
import hashlib
import os
import shutil
import subprocess


def cache_dir() -> str:
    d = os.environ.get("SGLANG_AMD_MONO_ASM_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "sgl_amd_mono", "dsv41_asm"
    )
    os.makedirs(d, exist_ok=True)
    return d


@contextlib.contextmanager
def build_lock():
    """Cross-process lock around kernel builds (vLLM's TP ranks build the same kernels at once)."""
    with open(os.path.join(cache_dir(), ".build.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _tool(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    # the pip ROCm SDK (vLLM's ROCm images), then a system ROCm
    for c in glob.glob(
        f"/usr/local/lib/python3*/dist-packages/_rocm_sdk_devel/lib/llvm/bin/{name}"
    ) + glob.glob(f"/opt/rocm*/llvm/bin/{name}"):
        return c
    return None


_tools: dict = {}


def toolchain_missing() -> str | None:
    """None if clang and ld.lld (LLVM with the AMDGPU target) are found, else what is missing."""
    if not _tools:
        _tools.update(clang=_tool("clang"), lld=_tool("ld.lld"))
    missing = [k for k, v in _tools.items() if v is None]
    return " and ".join(missing) or None


def build(src: str, name: str) -> bytes:
    """Assemble + link ``src`` → code object bytes, cached by source hash."""
    if toolchain_missing():
        raise FileNotFoundError(
            f"building the asm decode kernels needs {toolchain_missing()}"
        )
    h = hashlib.sha1(src.encode()).hexdigest()[:12]
    base = os.path.join(cache_dir(), f"{name}-{h}")
    co = base + ".co"
    if not os.path.exists(co):
        # per-process files, then an atomic rename (several ranks may build the same new kernel at once)
        tmp = f"{base}.{os.getpid()}"
        with open(tmp + ".s", "w") as f:
            f.write(src)
        r = subprocess.run(
            [
                _tools["clang"],
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcpu=gfx950",
                "-c",
                tmp + ".s",
                "-o",
                tmp + ".o",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(
                f"assemble failed ({tmp}.s):\n" + "\n".join(r.stderr.splitlines()[:12])
            )
        r = subprocess.run(
            [_tools["lld"], "-shared", tmp + ".o", "-o", tmp + ".co"],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(f"link failed:\n{r.stderr}")
        os.remove(tmp + ".s")
        os.remove(tmp + ".o")
        os.replace(tmp + ".co", co)
    with open(co, "rb") as f:
        return f.read()


_hip = None


def hip():
    """The HIP runtime torch loaded (found in the process's maps, so it is the same library instance)."""
    global _hip
    if _hip is None:
        import torch

        torch.zeros(1, device="cuda")  # make torch load and initialise the runtime
        lib = None
        with open("/proc/self/maps") as f:
            for line in f:
                if "libamdhip64.so" in line:
                    lib = line.split()[-1]
                    break
        _hip = ctypes.CDLL(lib or "libamdhip64.so")
        _hip.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def _check(err: int, what: str) -> None:
    if err:
        raise RuntimeError(
            f"{what}: hip error {err} {hip().hipGetErrorString(err).decode()}"
        )
