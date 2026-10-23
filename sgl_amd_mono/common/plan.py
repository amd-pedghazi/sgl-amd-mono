# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/plan/execution.py
# atom/mono/plan/layout.py
# atom/mono/plan/arith.py
# atom/mono/plan/shard.py
# atom/mono/plan/build_key.py
"""Plain-Python planning of the mono kernels: execution model, layouts, TP shards and build keys."""

import ast
import dataclasses
import enum
import functools
import hashlib
import sys
from pathlib import Path

import msgspec

# ---------------------------------------------------------------- execution model
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64


def first_task(bid, base):
    """Return CTA ``bid``'s first task of a stage placed from CTA ``base``."""
    return (bid - base % BLOCKS + BLOCKS) % BLOCKS


# ---------------------------------------------------------------- mailbox layouts
PAIR_BYTES = 8  # a (value, tag) pair


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """Lay out ``items`` (name, pairs) in order: name -> (byte offset, bytes), ``align``-aligned."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


# ---------------------------------------------------------------- arithmetic
def sel(pred, a, b):
    """Return a if pred else b, as ``select`` on traced values."""
    if isinstance(pred, bool):
        return a if pred else b
    out = pred.select(a, b)
    traced = [x for x in (a, b) if not isinstance(x, int)]
    return type(traced[0])(out) if traced else out


def imin(a, b):
    return sel(a < b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


# ---------------------------------------------------------------- TP shards
class ShardError(ValueError):
    """A width the TP size does not divide, or a TP size the model refuses."""


class Shard(msgspec.Struct, frozen=True):
    tp: int

    def __post_init__(self):
        if self.tp < 1:
            raise ShardError(f"TP {self.tp}")

    def split(self, full: int, what: str) -> int:
        """``full`` / tp, refusing a remainder: a rank's share of ``what``."""
        if full % self.tp:
            raise ShardError(f"{what} {full} is not divisible by TP {self.tp}")
        return full // self.tp


def padded(n: int, align: int) -> int:
    return cdiv(n, align) * align


# ---------------------------------------------------------------- build keys
_SCALARS = (int, float, bool, str, enum.Enum, type(None))
# FlyDSL keys a kernel only on its own directory's helpers, so its imports are hashed in
_TRACED = ("sgl_amd_mono", "aiter.ops.flydsl.kernels")
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _module_file(name: str):
    """The source file of module ``name`` in a traced package, found on disk without
    importing anything; None for an attribute, or for a package not imported."""
    root = next((p for p in _TRACED if name == p or name.startswith(f"{p}.")), None)
    init = getattr(sys.modules.get(root or ""), "__file__", None)
    if init is None:  # not traced, or no aiter (nothing compiles)
        return None
    base = Path(init).parent.joinpath(*name.split(".")[root.count(".") + 1 :])
    files = (base / "__init__.py", base.with_suffix(".py"))
    return next((f for f in files if f.is_file()), None)


def _imported(name: str, tree: ast.Module, package: bool):
    """Every module name ``tree`` can import, at any depth; for ``from m import x``
    both ``m`` and ``m.x``, as ``x`` may be a submodule."""
    parent = name if package else name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = parent.rsplit(".", node.level - 1)[0]
                base = f"{anchor}.{base}" if base else anchor
            yield base
            yield from (f"{base}.{a.name}" for a in node.names)


def _code(tree: ast.Module) -> str:
    """``tree`` without its docstrings; comments and layout are already gone."""
    for node in ast.walk(tree):
        if isinstance(node, _DOCUMENTED) and ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


@functools.cache
def _module_code(name: str):
    """Module ``name``'s code and the module names it can import, or None when
    ``name`` is not a module in a traced package."""
    f = _module_file(name)
    if f is None:
        return None
    tree = ast.parse(f.read_bytes())
    imported = tuple(_imported(name, tree, f.name == "__init__.py"))
    return _code(tree), imported


@functools.cache
def kernel_sources(module: str) -> str:
    """The digest of the code a kernel built in ``module`` traces.

    ``module`` and every plugin and aiter FlyDSL module it imports, transitively,
    without comments or docstrings (FlyDSL hashes each jit and kernel function's own
    source itself). Take it at import, as ``SOURCES = kernel_sources(__name__)``: a
    digest read at build time would pair an edit made since import with the code
    already loaded."""
    codes, todo = {}, [module]
    while todo:
        name = todo.pop()
        found = None if name in codes else _module_code(name)
        if found is not None:
            codes[name], imported = found
            todo.extend(imported)
    if module not in codes:
        raise FileNotFoundError(f"no source for kernel module {module!r}")
    h = hashlib.sha256()
    for name in sorted(codes):
        h.update(f"{name}\0{codes[name]}\0".encode())
    return h.hexdigest()[:16]


def _fields(value) -> list:
    if isinstance(value, msgspec.Struct):
        return [(f.name, getattr(value, f.name)) for f in msgspec.structs.fields(value)]
    return [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]


def _key_pairs(prefix: str, items):
    for name, value in items:
        if isinstance(value, msgspec.Struct) or dataclasses.is_dataclass(value):
            yield from _key_pairs(f"{prefix}{name}.", _fields(value))
        elif isinstance(value, _SCALARS):
            yield f"{prefix}{name}", value
        else:
            raise TypeError(
                f"{prefix}{name} = {value!r}: a build parameter must be a scalar, or a"
                " Struct or dataclass of them, to reach the JIT cache key"
            )


def key_tuple(key, sources: str) -> tuple:
    """Every build parameter in ``key`` as a (name, value) pair, then ("sources", ``sources``).

    ``key`` is a build Struct, or a builder's ``locals()`` before its first
    assignment; a nested Struct's or dataclass's fields are flattened, as FlyDSL
    leaves Structs, dataclasses and dicts a kernel closes over out of its key."""
    items = key.items() if isinstance(key, dict) else _fields(key)
    return (*_key_pairs("", items), ("sources", sources))
