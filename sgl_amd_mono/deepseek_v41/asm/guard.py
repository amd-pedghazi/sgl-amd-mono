# SPDX-License-Identifier: Apache-2.0
"""The deadline and fail-stop of the asm collectives' waits (``kernels/ar2.py``,
``comb_ar.py``, ``wob_ar.py``, ``eg_ag.py``), with the plugin's guard semantics
(``common/guard.py``).

A wait polls its flags; every ``CHECK_POLLS`` polls it reads the deadline and the
stop word of this process's guard block and the 100 MHz ``s_memrealtime`` clock, with
the deadline running from that first check (a wait the data reaches within
``CHECK_POLLS`` polls checks nothing). A wait past its deadline sets the sticky site
word, writes its site and the stop word into every rank's block (its own too),
fences and stops the launch with a store to the null page; a wait that sees the stop
word set stops too. A stopped launch never ends, so no wait returns stale data and
no later launch on the stream runs; the fault takes the process down. ``halt=False``
(tests only) ends a failed wave at once instead, so the host can read the words.

The guard block (int32 words, ``GUARD_BYTES``), one per process, IPC-shared:

=====  ================================================================================
word   meaning
=====  ================================================================================
0      the deadline in ``2**TICK_SHIFT`` clock ticks (host, ``AsmGuard.set_deadline``)
1      stop: 1 + the rank of a failed wait, on any rank (every wait stops at its check)
2      sticky: the site of a wait on this GPU that passed its deadline
3      sticky: the site of a wait on this GPU that stopped on the stop word
4-7    the site of rank r's timed-out wait (rank r writes word 4 + r of every block)
8-15   the 4 ranks' block addresses in this process (int64, host, at build)
=====  ================================================================================

The asm collectives' per-CTA ``err`` words (the seq of a timed-out launch) stay.
"""

import ctypes

CHECK_POLLS = 64  # common/sync.py's: a pending wait checks this often
CLOCK_HZ = 100_000_000  # s_memrealtime: the constant 100 MHz reference clock
TICK_SHIFT = 8  # common/guard.py's: a check compares 32-bit times in 2.56 us units
# common/guard.py's MAX_DEADLINE_TICKS: a longer deadline does not fit the check
MAX_DEADLINE_TICKS = (1 << 31 << TICK_SHIFT) - 1
DEFAULT_DEADLINE_S = 60.0  # the plugin runner's, until the spec sets SGLang's

W_DEADLINE, W_STOP, W_SITE, W_PEER, W_BY, W_PEERS = 0, 1, 2, 3, 4, 8
GUARD_BYTES = 256

# a wait's site: the kernel and its hop ("rs": hop 0, "ag": hop 1)
SITES = {
    "ar2.rs": 1,
    "ar2.ag": 2,
    "ar2f8.rs": 3,
    "ar2f8.ag": 4,
    "comb_ar.rs": 5,
    "comb_ar.ag": 6,
    "wob_ar.rs": 7,
    "wob_ar.ag": 8,
    "eg_ag.ag": 9,
}
SITE_NAMES = {v: k for k, v in SITES.items()}


def deadline_units(seconds: float) -> int:
    """``seconds`` in the units a wait compares (``2**TICK_SHIFT`` ticks)."""
    ticks = int(seconds * CLOCK_HZ)
    if not 0 < ticks <= MAX_DEADLINE_TICKS:
        raise ValueError(
            f"a {seconds} s wait deadline does not fit the wait's 32-bit check; "
            f"use 0 < s <= {MAX_DEADLINE_TICKS // CLOCK_HZ}"
        )
    return ticks >> TICK_SHIFT


class WaitGuard:
    """One kernel's guarded waits (generator side). ``A`` appends a line of asm.

    ``start()`` before a poll loop, ``check(tag, site)`` as the loop's first lines (it
    falls through to the poll), ``tails(err_store)`` once after the kernel's end. Register
    pairs: ``polls`` the loop's poll count; ``t0``, ``now``, ``words`` SGPR pairs free in
    the loop; ``gptr`` an SGPR pair no fail path writes; ``table`` 8 SGPRs (aligned to 4)
    dead on a failure; ``vp`` an even VGPR pair dead at the top of a poll iteration.
    ``err_store``: the kernel's own per-CTA err store (exec = 1, may write ``words``).
    """

    def __init__(
        self, A, R, ka_off, *, polls, t0, now, words, gptr, table, vp, halt=True
    ):
        assert all(r % 2 == 0 for r in (t0, now, words, gptr, vp)) and table % 4 == 0
        self.A, self.R, self.ka_off, self.halt = A, R, ka_off, halt
        self.polls, self.t0, self.now, self.words = polls, t0, now, words
        self.gptr, self.table, self.vp = gptr, table, vp
        self.sites: list[tuple[str, int]] = []

    def start(self):
        self.A(f"  s_mov_b32 s{self.polls}, 0")

    def check(self, tag: str, site: int):
        A, p, t0, now, w, g, vp = (
            self.A,
            self.polls,
            self.t0,
            self.now,
            self.words,
            self.gptr,
            self.vp,
        )
        self.sites.append((tag, site))
        A(f"  s_add_u32 s{p}, s{p}, 1")
        A(f"  s_and_b32 s{now}, s{p}, {CHECK_POLLS - 1}")
        A(f"  s_cbranch_scc1 .Lq{tag}")  # not a check poll
        A(f"  s_load_dwordx2 s[{g}:{g + 1}], s[0:1], {self.ka_off:#x}")
        A(f"  s_memrealtime s[{now}:{now + 1}]")
        A(f"  v_mov_b32 v{vp}, 0")
        A("  s_waitcnt lgkmcnt(0)")
        # the deadline and the stop word, as the flags are polled (peers set the stop word)
        A(f"  global_load_dwordx2 v[{vp}:{vp + 1}], v{vp}, s[{g}:{g + 1}] sc1")
        A(f"  s_cmp_eq_u32 s{p}, {CHECK_POLLS}")
        A(f"  s_cselect_b64 s[{t0}:{t0 + 1}], s[{now}:{now + 1}], s[{t0}:{t0 + 1}]")
        A(f"  s_sub_u32 s{now}, s{now}, s{t0}")
        A(f"  s_subb_u32 s{now + 1}, s{now + 1}, s{t0 + 1}")
        A(f"  s_lshr_b64 s[{now}:{now + 1}], s[{now}:{now + 1}], {TICK_SHIFT}")
        A("  s_waitcnt vmcnt(0)")
        A(f"  v_readfirstlane_b32 s{w}, v{vp}")
        A(f"  v_readfirstlane_b32 s{w + 1}, v{vp + 1}")
        A(f"  s_cmp_lg_u32 s{w + 1}, 0")
        A(f"  s_cbranch_scc1 .Lgp_{tag}")  # a wait failed somewhere: stop
        A(f"  s_cmp_gt_u32 s{now}, s{w}")
        A(f"  s_cbranch_scc1 .Lgf_{tag}")  # this wait passed its deadline
        A(f".Lq{tag}:")

    def tails(self, err_store: list[str]):
        A, R, w, g, t, vp = self.A, self.R, self.words, self.gptr, self.table, self.vp
        for tag, site in self.sites:
            A(f".Lgf_{tag}:")
            A(f"  s_mov_b32 s{w}, {site}")
            A("  s_branch .Lgfail")
            A(f".Lgp_{tag}:")
            A(f"  s_mov_b32 s{w}, {site}")
            A("  s_branch .Lgpeer")
        A(".Lgfail:")
        A("  s_mov_b64 exec, 1")
        A(f"  v_mov_b32 v{vp + 1}, s{w}")
        for line in err_store:
            A(line)
        A(f"  v_mov_b32 v{vp}, {W_SITE * 4}")
        A(f"  global_store_dword v{vp}, v{vp + 1}, s[{g}:{g + 1}] sc0 sc1")
        A(f"  s_load_dwordx8 s[{t}:{t + 7}], s[{g}:{g + 1}], {W_PEERS * 4:#x}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_mov_b32 v{vp}, {(W_BY + R) * 4}")
        for d in range(4):
            A(
                f"  global_store_dword v{vp}, v{vp + 1}, s[{t + 2 * d}:{t + 2 * d + 1}] sc0 sc1"
            )
        A(f"  v_mov_b32 v{vp}, {W_STOP * 4}")
        A(f"  v_mov_b32 v{vp + 1}, {R + 1}")
        for d in range(4):
            A(
                f"  global_store_dword v{vp}, v{vp + 1}, s[{t + 2 * d}:{t + 2 * d + 1}] sc0 sc1"
            )
        A("  s_waitcnt vmcnt(0)")
        A("  s_branch .Lgstop")
        A(".Lgpeer:")
        A("  s_mov_b64 exec, 1")
        A(f"  v_mov_b32 v{vp}, {W_PEER * 4}")
        A(f"  v_mov_b32 v{vp + 1}, s{w}")
        A(f"  global_store_dword v{vp}, v{vp + 1}, s[{g}:{g + 1}] sc0 sc1")
        A("  s_waitcnt vmcnt(0)")
        A(".Lgstop:")
        if not self.halt:
            A("  s_endpgm")
            return
        # the null page: the fault tells a deliberate stop apart and tears the queue down
        A(f"  v_mov_b32 v{vp}, 0")
        A(f"  v_mov_b32 v{vp + 1}, 0")
        A(f"  global_store_dword v[{vp}:{vp + 1}], v{vp}, off sc0 sc1")
        A("  s_waitcnt vmcnt(0)")
        A(".Lghalt:")
        A("  s_sleep 127")
        A("  s_branch .Lghalt")


def _hip_copy():
    from sgl_amd_mono.deepseek_v41.asm.kernels.build import _check, hip

    h = hip()
    h.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    return h, _check


class AsmGuard:
    """One process's guard block, shared by its asm collectives: hipMalloc'd and shared
    with the peers through IPC handles (collective over ``cpu_group``, like the
    collectives' own buffers). ``ptr`` is the kernarg every guarded launch takes."""

    def __init__(
        self,
        rank: int,
        cpu_group,
        world: int = 4,
        deadline_s: float = DEFAULT_DEADLINE_S,
    ):
        import torch

        from sgl_amd_mono.deepseek_v41.asm.ar_runner import ipc_alloc

        assert world == 4, "the asm collectives are written for TP4"
        mine, every = ipc_alloc({"guard": GUARD_BYTES}, rank, cpu_group, world)
        self.rank, self.world, self.ptr, self.blocks = (
            rank,
            world,
            mine["guard"],
            every["guard"],
        )
        h, check = _hip_copy()
        table = (ctypes.c_uint64 * world)(*self.blocks)
        check(
            h.hipMemcpy(
                ctypes.c_void_p(self.ptr + W_PEERS * 4),
                ctypes.cast(table, ctypes.c_void_p),
                8 * world,
                1,
            ),
            "hipMemcpy guard table",
        )
        self.set_deadline(deadline_s)
        torch.distributed.barrier(group=cpu_group)

    def set_deadline(self, seconds: float) -> None:
        """Every guarded wait's deadline (below the server's watchdog timeout). Outside
        graph capture; the launches before it on this stream should be done."""
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("AsmGuard.set_deadline under graph capture")
        units = deadline_units(seconds)
        torch.accelerator.synchronize()
        h, check = _hip_copy()
        v = ctypes.c_uint32(units)
        check(
            h.hipMemcpy(
                ctypes.c_void_p(self.ptr + W_DEADLINE * 4), ctypes.byref(v), 4, 1
            ),
            "hipMemcpy guard deadline",
        )
        self.deadline_s = seconds

    def words(self) -> list[int]:
        """The block's first 8 words. Syncs (tests and failure reports only)."""
        import torch

        torch.accelerator.synchronize()
        h, check = _hip_copy()
        buf = (ctypes.c_uint32 * 8)()
        check(
            h.hipMemcpy(
                ctypes.cast(buf, ctypes.c_void_p), ctypes.c_void_p(self.ptr), 32, 2
            ),
            "hipMemcpy guard words",
        )
        return list(buf)

    def errors(self) -> dict:
        """The sticky words by name ({} when clean): ``stop`` (1 + the failed rank),
        ``site`` / ``peer`` (this GPU's wait that timed out / stopped on the stop word)
        and ``by`` (rank: its timed-out site)."""
        w = self.words()
        out: dict = {}
        if w[W_STOP]:
            out["stop"] = w[W_STOP]
        for key, i in (("site", W_SITE), ("peer", W_PEER)):
            if w[i]:
                out[key] = SITE_NAMES.get(w[i], w[i])
        by = {
            r: SITE_NAMES.get(w[W_BY + r], w[W_BY + r])
            for r in range(self.world)
            if w[W_BY + r]
        }
        if by:
            out["by"] = by
        return out
