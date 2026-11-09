# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/ar_runner.py (changed: import paths, the guard block (asm/guard.py) of
# every launch, BF16AllReduce, EngramAllGather)
"""The fp8-wire all-reduce (kernels/ar2.py ``source_fp8``) across the 4 TP
processes: landing buffers and flags are hipMalloc'd per rank and shared with
the peers through IPC handles (exchanged over a CPU process group), as vLLM's
custom all-reduce does. ``out`` is written only by its own rank, so it is a
plain torch tensor (graph-pool friendly).

    # once per process, outside graph capture
    ar = FP8AllReduce(rank, cpu_group)
    # part [M, 5120] bf16, M ≤ 256 → out [M, 5120] bf16
    out = ar(part)

Every rank must make the same calls in the same order (the per-CTA sequence
counters live in the flags buffer and must agree). A rank that stops calling
makes its peers' waits pass their deadline (``asm/guard.py``): they record the
site, stop every rank and their own launch, and the process with it. Pass one
``AsmGuard`` to all of a process's collectives (``guard=``) so that a failure in
one stops the waits of all; without it each makes its own. ``errors()`` counts
the CTAs whose wait timed out (readable with ``halt=False`` kernels, tests only).
The other runners here: BF16AllReduce (ar2's bf16 wire), WoBAllReduce,
CombAllReduce and EngramAllGather.
"""

import ctypes

import torch

from sgl_amd_mono.deepseek_v41.asm.guard import AsmGuard
from sgl_amd_mono.deepseek_v41.asm.kernels import ar2
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import Kernel

H = ar2.H


class _IpcHandle(ctypes.Structure):
    _fields_ = [
        ("reserved", ctypes.c_ubyte * 64)
    ]  # not c_char: its field reads stop at the first NUL


def _hip():
    from sgl_amd_mono.deepseek_v41.asm.kernels.build import hip

    h = hip()
    h.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
    h.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _IpcHandle,
        ctypes.c_uint,
    ]
    return h


def flag_state(ptr: int, nb: int, g: int) -> tuple[list, list]:
    """The last 2g dwords of a flags buffer (ar2 / eg_ag layout: ... ctr [g], err [g])
    as (ctr, err) lists. Syncs."""
    f = torch.empty(nb // 4, dtype=torch.int32, device="cuda")
    torch.accelerator.synchronize()
    h = _hip()
    h.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    h.hipMemcpy(ctypes.c_void_p(f.data_ptr()), ctypes.c_void_p(ptr), nb, 3)
    t = f[-2 * g :].tolist()
    return t[:g], t[g:]


def ipc_alloc(sizes: dict, rank: int, cpu_group, world: int = 4) -> tuple[dict, dict]:
    """hipMalloc'd, zeroed buffers {name: bytes} on every rank, shared through
    IPC handles (collective over cpu_group) → (mine {name: ptr}, all {name:
    [ptr of rank d's buffer in this process, d < world]})."""
    from sgl_amd_mono.deepseek_v41.asm.kernels.build import _check

    h = _hip()
    torch.accelerator.synchronize()
    mine, handles = {}, {}
    # plain hipMalloc (coarse-grained, cached): peers write these buffers and we read
    # them; an uncached allocation (as vLLM's custom all-reduce uses for its signals)
    # costs ~20 µs per decode step
    for n, sz in sizes.items():
        p = ctypes.c_void_p()
        _check(h.hipMalloc(ctypes.byref(p), sz), f"hipMalloc {n}")
        _check(h.hipMemset(p, 0, sz), f"hipMemset {n}")
        hd = _IpcHandle()
        _check(h.hipIpcGetMemHandle(ctypes.byref(hd), p), f"hipIpcGetMemHandle {n}")
        mine[n], handles[n] = p.value, ctypes.string_at(ctypes.addressof(hd), 64)
    torch.accelerator.synchronize()
    gathered: list = [None] * world
    torch.distributed.all_gather_object(gathered, handles, group=cpu_group)
    every: dict[int, list] = {n: [] for n in sizes}
    for n in sizes:
        for d in range(world):
            if d == rank:
                every[n].append(mine[n])
                continue
            hd = _IpcHandle()
            ctypes.memmove(ctypes.addressof(hd), gathered[d][n], 64)
            p = ctypes.c_void_p()
            _check(
                h.hipIpcOpenMemHandle(ctypes.byref(p), hd, 1),
                f"hipIpcOpenMemHandle {n} of rank {d}",
            )
            every[n].append(p.value)
    _ipc_check(h, mine, every, rank, cpu_group, world)
    return mine, every


_ipc_seq = [0]


def _ipc_check(h, mine: dict, every: dict, rank: int, cpu_group, world: int) -> None:
    """Every rank writes a signature (magic, rank, allocation #, buffer #) at
    the start of each of its new buffers; every rank reads its peers' through
    the opened IPC pointers. A mismatch means a mapping lands in some other
    allocation (peer flag writes that never arrived).
    Prints the pointers' page offsets too (suballocation). Buffers are re-zeroed."""
    import torch.distributed as dist

    h.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _ipc_seq[0] += 1
    names = list(mine)
    sig = lambda r, i: (0x5A5A0000 | i, r, _ipc_seq[0], 0x1234)
    for i, n in enumerate(names):
        buf = (ctypes.c_int32 * 4)(*sig(rank, i))
        h.hipMemcpy(ctypes.c_void_p(mine[n]), ctypes.cast(buf, ctypes.c_void_p), 16, 1)
    torch.accelerator.synchronize()
    dist.barrier(group=cpu_group)
    bad = []
    for i, n in enumerate(names):
        for d in range(world):
            if d == rank:
                continue
            buf = (ctypes.c_int32 * 4)()
            h.hipMemcpy(
                ctypes.cast(buf, ctypes.c_void_p), ctypes.c_void_p(every[n][d]), 16, 2
            )
            got = tuple(v & 0xFFFFFFFF for v in buf)
            want = tuple(v & 0xFFFFFFFF for v in sig(d, i))
            if got != want:
                bad.append(f"{n}@r{d} read {[hex(v) for v in got]}")
    dist.barrier(group=cpu_group)
    for n in names:
        h.hipMemset(ctypes.c_void_p(mine[n]), 0, 16)
    torch.accelerator.synchronize()
    dist.barrier(group=cpu_group)
    if bad:
        raise RuntimeError(
            f"asm decode: IPC buffers don't map to the peers' (rank {rank}): "
            + "; ".join(bad)
        )


class FP8AllReduce:
    def __init__(
        self,
        rank: int,
        cpu_group,
        world: int = 4,
        g: int | None = None,
        knock: frozenset = frozenset(),
        guard: AsmGuard | None = None,
        halt: bool = True,
    ):
        assert world == 4, "ar2 is written for TP4"
        self.rank = rank
        self.guard = guard or AsmGuard(rank, cpu_group, world)
        self.g = g or ar2.G
        sizes = dict(
            land=4 * ar2.QMAX, land2=4 * ar2.QMAX, flags=ar2.flag_bytes(self.g)
        )
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for n in sizes:
            for d in range(world):
                key = {"land": f"land{d}", "land2": f"land2_{d}", "flags": f"flags{d}"}[
                    n
                ]
                self.ptrs[key] = every[n][d]
        self.flags_mine = mine["flags"]
        self.flag_layout = (
            ar2.flag_bytes(self.g),
            2,
        )  # (bytes, hops), for IPC diagnostics
        k = ar2.source_fp8(rank, g=self.g, knock=knock, halt=halt)
        self.k = Kernel(k["name"], k["src"], k["ka"])
        torch.distributed.barrier(group=cpu_group)

    def __call__(self, part: torch.Tensor) -> torch.Tensor:
        m = part.shape[0]
        assert (
            part.dtype == torch.bfloat16 and part.shape[1] == H and 1 <= m <= ar2.MAX_M
        ), (part.dtype, part.shape)
        part = part.contiguous()
        out = torch.empty_like(part)
        self.k(
            self.g,
            64,
            **self.ptrs,
            **{f"out{d}": out for d in range(4)},
            part=part,
            m=m,
            guard=self.guard.ptr,
        )
        return out

    def errors(self) -> int:
        """Poll timeouts so far (CTAs with a nonzero err word)."""
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per CTA: launches so far, and the seq of the last timed-out launch
        (0: none). Syncs."""
        return flag_state(self.flags_mine, ar2.flag_bytes(self.g), self.g)


class BF16AllReduce:
    """The bf16-wire all-reduce (kernels/ar2.py ``source``): hop 2 stores each reduced
    quarter into every rank's ``out``, so ``out`` is an IPC buffer per rank; a call
    returns a fresh tensor copied from it on the stream (before the next call's pushes,
    which wait for this rank's flags)."""

    def __init__(
        self,
        rank: int,
        cpu_group,
        world: int = 4,
        guard: AsmGuard | None = None,
        halt: bool = True,
    ):
        assert world == 4, "ar2 is written for TP4"
        self.rank = rank
        self.guard = guard or AsmGuard(rank, cpu_group, world)
        sizes = dict(land=4 * ar2.QMAX, flags=ar2.FLAG_BYTES, out=ar2.MAX_M * H * 2)
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for d in range(world):
            self.ptrs.update(
                {
                    f"land{d}": every["land"][d],
                    f"flags{d}": every["flags"][d],
                    f"out{d}": every["out"][d],
                }
            )
        self.flags_mine, self.out_mine = mine["flags"], mine["out"]
        self.flag_layout = (ar2.FLAG_BYTES, 2)  # (bytes, hops), for IPC diagnostics
        k = ar2.source(rank, halt=halt)
        self.k = Kernel(k["name"], k["src"], k["ka"])
        self.h = _hip()
        self.h.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        torch.distributed.barrier(group=cpu_group)

    def __call__(self, part: torch.Tensor) -> torch.Tensor:
        from sgl_amd_mono.deepseek_v41.asm.kernels.build import _check

        m = part.shape[0]
        assert (
            part.dtype == torch.bfloat16 and part.shape[1] == H and 1 <= m <= ar2.MAX_M
        ), (part.dtype, part.shape)
        part = part.contiguous()
        self.k(ar2.G, 64, **self.ptrs, part=part, m=m, guard=self.guard.ptr)
        out = torch.empty_like(part)
        _check(
            self.h.hipMemcpyAsync(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(self.out_mine),
                m * H * 2,
                3,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            ),
            "hipMemcpyAsync ar2 out",
        )
        return out

    def errors(self) -> int:
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per CTA. Syncs."""
        return flag_state(self.flags_mine, ar2.FLAG_BYTES, ar2.G)


class EngramAllGather:
    """The Engram side chain's all-gather (kernels/eg_ag.py, phase "all"): every rank's
    src [m, row_bytes] lands in every rank's dst [m, 4 row_bytes] at byte column
    rank * row_bytes, so dst is an IPC buffer per rank; a call returns a fresh tensor
    copied from it on the stream. dst is single-buffered: a rank's next call may push
    into its peers' dst once its own call is done, so the steps between two calls
    must order the ranks (in serving, the main stream's all-reduces do)."""

    def __init__(
        self,
        rank: int,
        cpu_group,
        row_bytes: int,
        world: int = 4,
        max_m: int = 256,
        guard: AsmGuard | None = None,
        halt: bool = True,
    ):
        from sgl_amd_mono.deepseek_v41.asm.kernels import eg_ag

        assert world == 4, "eg_ag is written for TP4"
        self.rank, self.eg_ag, self.row_bytes, self.max_m = (
            rank,
            eg_ag,
            row_bytes,
            max_m,
        )
        self.guard = guard or AsmGuard(rank, cpu_group, world)
        sizes = dict(dst=max_m * 4 * row_bytes, flags=eg_ag.flag_bytes())
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for d in range(world):
            self.ptrs.update(
                {f"dst{d}": every["dst"][d], f"flags{d}": every["flags"][d]}
            )
        self.flags_mine, self.dst_mine = mine["flags"], mine["dst"]
        self.flag_layout = (eg_ag.flag_bytes(), 1)  # (bytes, hops), for IPC diagnostics
        k = eg_ag.source(rank, row_bytes, halt=halt)
        self.k = Kernel(k["name"], k["src"], k["ka"])
        self.h = _hip()
        self.h.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        torch.distributed.barrier(group=cpu_group)

    def __call__(self, src: torch.Tensor) -> torch.Tensor:
        from sgl_amd_mono.deepseek_v41.asm.kernels.build import _check

        m = src.shape[0]
        assert src.dim() == 2 and 1 <= m <= self.max_m, src.shape
        assert src.shape[1] * src.element_size() == self.row_bytes, src.shape
        src = src.contiguous()
        self.k(self.eg_ag.G, 64, **self.ptrs, src=src, m=m, guard=self.guard.ptr)
        out = torch.empty(m, 4 * src.shape[1], dtype=src.dtype, device=src.device)
        _check(
            self.h.hipMemcpyAsync(
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(self.dst_mine),
                m * 4 * self.row_bytes,
                3,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            ),
            "hipMemcpyAsync eg_ag dst",
        )
        return out

    def errors(self) -> int:
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per CTA. Syncs."""
        return flag_state(self.flags_mine, self.eg_ag.flag_bytes(), self.eg_ag.G)


class WoBAllReduce:
    """K1's wo_b with all-reduce #1 folded in (kernels/wob_ar.py): one launch
    from wo_a's z (MXFP8 frag) to the all-reduced [M, 5120] bf16, wire "bf16"
    (= vLLM's custom all-reduce, M ≤ 128) or "fp8" (= FP8AllReduce, M ≥ 192).

        # once per process, outside graph capture
        wob = WoBAllReduce(rank, cpu_group)
        out = wob(w_frag, ws_frag, z, zs, m, wire)

    Collective like FP8AllReduce: every rank makes the same calls (same M, same
    wire) in the same order; both wires share the per-wave sequence counters."""

    def __init__(
        self,
        rank: int,
        cpu_group,
        world: int = 4,
        knock: frozenset = frozenset(),
        guard: AsmGuard | None = None,
        halt: bool = True,
    ):
        from sgl_amd_mono.deepseek_v41.asm.kernels import wob_ar

        assert world == 4, "wob_ar is written for TP4"
        self.rank, self.wob_ar = rank, wob_ar
        self.guard = guard or AsmGuard(rank, cpu_group, world)
        sizes = dict(
            land=wob_ar.LAND_BYTES, land2=wob_ar.LAND_BYTES, flags=wob_ar.FLAG_BYTES
        )
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for d in range(world):
            self.ptrs.update(
                {
                    f"land{d}": every["land"][d],
                    f"land2_{d}": every["land2"][d],
                    f"flags{d}": every["flags"][d],
                }
            )
        self.flags_mine = mine["flags"]
        self.g, self.flag_layout = (
            wob_ar.NW,
            (wob_ar.FLAG_BYTES, 2),
        )  # for IPC diagnostics
        self.k = {}
        for wire in ("bf16", "fp8"):
            d = wob_ar.source(rank, wire, knock=knock, halt=halt)
            self.k[wire] = Kernel(d["name"], d["src"], d["ka"])
        self.dummy = torch.zeros(64, dtype=torch.uint8, device="cuda")
        torch.distributed.barrier(group=cpu_group)

    def __call__(
        self, w, ws, z, zs, m: int, wire: str, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert 1 <= m <= 256
        out = (
            torch.empty(m, H, dtype=torch.bfloat16, device=z.device)
            if out is None
            else out
        )
        d = self.dummy
        self.k[wire](
            self.wob_ar.NCTA,
            256,
            w=w,
            ws=ws,
            x=z,
            xs=zs,
            out=out,
            outs=d,
            part=d,
            cnt=d,
            cs=d,
            pos=d,
            dbg=d,
            m=m,
            **self.ptrs,
            guard=self.guard.ptr,
        )
        return out

    def errors(self) -> int:
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per wave (CTA·4 + wave). Syncs."""
        return flag_state(self.flags_mine, self.wob_ar.FLAG_BYTES, self.wob_ar.NW)


class CombAllReduce:
    """K2's MoE combine with all-reduce #2 folded in (kernels/comb_ar.py): one launch
    from the weighted expert rows y [M·6, 5120] f32 (or bf16: K2Runner v3), the picks'
    ids [M·6] and the shared expert's output add [M, 5120] bf16 to the all-reduced
    bf16(bf16(Σ y) + add), wire "bf16" (= vLLM's custom all-reduce) or "fp8" (=
    moe_combine + FP8AllReduce, bitwise). Collective like FP8AllReduce; both wires share
    the per-wave counters."""

    def __init__(
        self,
        rank: int,
        cpu_group,
        world: int = 4,
        stop: int = 99,
        guard: AsmGuard | None = None,
        halt: bool = True,
    ):
        from sgl_amd_mono.deepseek_v41.asm.kernels import comb_ar

        assert world == 4, "comb_ar is written for TP4"
        self.rank, self.comb_ar, self.g = rank, comb_ar, comb_ar.NW
        self.guard = guard or AsmGuard(rank, cpu_group, world)
        sizes = dict(land=4 * ar2.QMAX, land2=4 * ar2.QMAX, flags=comb_ar.FLAG_BYTES)
        mine, every = ipc_alloc(sizes, rank, cpu_group, world)
        self.ptrs = {}
        for d in range(world):
            self.ptrs.update(
                {
                    f"land{d}": every["land"][d],
                    f"land2_{d}": every["land2"][d],
                    f"flags{d}": every["flags"][d],
                }
            )
        self.flags_mine = mine["flags"]
        self.flag_layout = (comb_ar.FLAG_BYTES, 2)  # for IPC diagnostics
        self.k = {}
        for wire in ("bf16", "fp8"):
            for yb in (False, True):
                d = comb_ar.source(rank, wire, stop=stop, y_bf16=yb, halt=halt)
                self.k[wire, yb] = Kernel(d["name"], d["src"], d["ka"])
        torch.distributed.barrier(group=cpu_group)

    def __call__(self, y, ids, add, m: int, wire: str) -> torch.Tensor:
        assert (
            1 <= m <= ar2.MAX_M
            and y.dtype in (torch.float32, torch.bfloat16)
            and ids.dtype == torch.int32
        )
        assert add.dtype == torch.bfloat16
        assert (
            y.is_contiguous()
            and ids.is_contiguous()
            and add.is_contiguous()
            and add.shape == (m, H)
        )
        out = torch.empty(m, H, dtype=torch.bfloat16, device=add.device)
        yb = y.dtype == torch.bfloat16
        self.k[wire, yb](
            self.comb_ar.grid(m),
            256,
            **self.ptrs,
            out=out,
            y=y,
            ids=ids,
            add=add,
            m=m,
            y_bytes=m * 6 * H * (2 if yb else 4),
            ids_bytes=m * 6 * 4,
            add_bytes=m * H * 2,
            guard=self.guard.ptr,
        )
        return out

    def errors(self) -> int:
        return sum(e != 0 for e in self.state()[1])

    def state(self) -> tuple[list, list]:
        """(ctr, err) per wave (CTA·4 + wave). Syncs."""
        return flag_state(self.flags_mine, self.flag_layout[0], self.g)
