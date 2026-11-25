# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/eg_ag.py (changed: import paths, the wait's deadline and fail-stop
# (asm/guard.py) instead of the 2^30-poll timeout, a guard kernarg)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Engram: an all-gather across the 4 TP ranks, pushed over xGMI,
for the Engram side chain (lookup rows and wkv output). Replaces vLLM's NCCL
``all_gather(x, dim=-1)``: every rank's ``src`` [m, B bytes] lands in every
rank's ``dst`` [m, 4 B] at byte column R·B, so the result is laid out as
vLLM's.

- Grid G CTAs of one wave. Flat 16-byte granules i < m·B/16 of src; CTA c
  takes i = (k·G + c)·64 + lane for k = 0, 1, ... (U per loop pass), and only
  talks to CTA c of the peers.
- Each granule goes to my dst (plain store) and the 3 peers' dst (sc1 stores)
  at 16 i + 3 B ⌊i / (B/16)⌋ + R B; then vmcnt(0), one flag per (source, CTA)
  to the peers, poll the 3 peers' flags (ar2's hop). When the kernel ends,
  every peer's rows are in my dst.
- Seq numbers in device memory (ar2): CTA c reads ctr[c], uses ctr + 1, writes
  it back, so graph replays and eager calls need no host input. Every rank must
  make the same launches.
- No double buffering: peer d writes my dst for launch s + 1 only after its
  stream has passed launch s, and the main stream's all-reduces order the
  steps (a rank starts the next step's side chain only after every rank
  reached its last all-reduce, i.e. consumed this step's dst).

- The wait has the guard's deadline and fail-stop (asm/guard.py).

Kernargs: 8 pointers [dst 0..3, flags 0..3] in this rank's address space,
then src, m and the guard block; flags: flag_bytes(g) (``[4 sources][g]``
dwords, ctr [g], err [g]).
"""

from sgl_amd_mono.deepseek_v41.asm.guard import SITES, WaitGuard
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import KArgs, kernel_source

G, U = 128, 4
POLL = frozenset()  # sc1 polls (no system-scope loads, invalidates or write-backs)
POLL_LD = "sc1"

KA = KArgs(
    *[(f"dst{d}", "ptr") for d in range(4)],
    *[(f"flags{d}", "ptr") for d in range(4)],
    ("src", "ptr"),
    ("m", "u32"),
    ("guard", "ptr"),
)


def flag_bytes(g: int = G) -> int:
    return (4 * g + 2 * g) * 4


def magic(d: int, nmax: int = 1 << 21) -> tuple[int, int]:
    """(mul, sh): ⌊i / d⌋ = mulhi(i, mul) >> sh for every i < nmax (checked)."""
    import numpy as np

    for sh in range(32):
        mul = -(-(1 << (32 + sh)) // d)
        if mul >= 1 << 32:
            break
        i = np.arange(nmax, dtype=np.uint64)
        if np.array_equal(
            (i * np.uint64(mul)) >> np.uint64(32 + sh), i // np.uint64(d)
        ):
            return mul, sh
    raise ValueError(f"no magic for {d}")


def source(
    R: int,
    B: int,
    g: int = G,
    u: int = U,
    name: str | None = None,
    phase: str = "all",
    halt: bool = True,
):
    """Phase "all": the all-gather above. The push split (no cross-rank wait on the main stream): "push" stores the rows and the flags and returns without waiting (ctr + 1); "wait" stores
    nothing and polls the peers' flags for seq = ctr (as the push before it on this rank left it).
    halt=False (tests only): a failed wait ends its wave instead of stopping the launch."""
    assert B % 16 == 0 and 0 <= R < 4 and phase in ("all", "push", "wait")
    gpr = B // 16
    pow2 = gpr & (gpr - 1) == 0
    mul, sh = (0, gpr.bit_length() - 1) if pow2 else magic(gpr)
    name = name or (
        f"eg_ag_r{R}_b{B}"
        + (f"_g{g}" if g != G else "")
        + (f"_u{u}" if u != U else "")
        + ("" if phase == "all" else f"_{phase}")
        + ("" if halt else "_nohalt")
    )
    peers = [d for d in range(4) if d != R]
    ctr_off, err_off = 4 * g * 4, 5 * g * 4
    VI, VO, VD = 2, 2 + u, 2 + 2 * u  # granule index, dst offset, data quads
    T = VD + 4 * u
    FV = T + 2
    EX = FV + 8  # the guard words read as the wait starts (a VGPR pair of its own)
    c = []
    A = c.append
    # free SGPRs (t0, now, words, gptr, table), polls s18
    wg = WaitGuard(
        A, R, KA.off["guard"], polls=18, t0=92, now=94, words=48, gptr=4, table=52,
        vp=FV + 6, halt=halt,
    )  # fmt: skip

    def fl(d):
        return f"s[{28 + 2 * d}:{29 + 2 * d}]"

    def ds(d):
        return f"s[{20 + 2 * d}:{21 + 2 * d}]"

    A("  s_load_dwordx16 s[20:35], s[0:1], 0")  # dst 0..3, flags 0..3
    A(f"  s_load_dwordx2 s[6:7], s[0:1], {KA.off['src']:#x}")
    A(f"  s_load_dword s8, s[0:1], {KA.off['m']:#x}")
    A(f"  s_load_dwordx2 s[4:5], s[0:1], {KA.off['guard']:#x}")
    A("  v_and_b32 v1, 63, v0")
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mul_i32 s9, s8, {gpr}")  # N granules
    A("  s_lshl_b32 s14, s2, 2")
    A(f"  s_add_u32 s14, s14, {ctr_off}")
    A(f"  s_load_dword s13, {fl(R)}, s14")
    A("  s_lshl_b32 s10, s2, 6")  # this pass's first granule (unit 0)
    A("  s_cmp_ge_u32 s10, s9")
    A("  s_cbranch_scc1 .Lidle")  # idle CTA (small m), as on every rank
    if phase == "wait":
        A("  s_branch .Lhop")
    A(f"  s_mov_b32 s11, {R * B}")
    if not pow2:
        A(f"  s_mov_b32 s12, {mul:#x}")  # (VOP3 takes no literal on gfx9)
    A(".Lpass:")
    for k in range(u):
        A(f"  v_add_u32 v{VI + k}, s10, v1")
        if k:
            A(f"  v_add_u32 v{VI + k}, {k * g * 64}, v{VI + k}")
        A(f"  v_cmp_gt_u32 s[{40 + 2 * k}:{41 + 2 * k}], s9, v{VI + k}")
    for k in range(u):
        A(f"  s_mov_b64 exec, s[{40 + 2 * k}:{41 + 2 * k}]")
        A(f"  v_lshlrev_b32 v{T}, 4, v{VI + k}")
        A(f"  global_load_dwordx4 v[{VD + 4 * k}:{VD + 4 * k + 3}], v{T}, s[6:7]")
    for k in range(u):  # 16 i + 3 B t + R B
        A(f"  s_mov_b64 exec, s[{40 + 2 * k}:{41 + 2 * k}]")
        if pow2:
            A(f"  v_lshrrev_b32 v{T}, {sh}, v{VI + k}")
        else:
            A(f"  v_mul_hi_u32 v{T}, s12, v{VI + k}")
            if sh:
                A(f"  v_lshrrev_b32 v{T}, {sh}, v{T}")
        A(f"  v_mul_u32_u24 v{T}, {3 * B}, v{T}")
        A(f"  v_lshlrev_b32 v{VO + k}, 4, v{VI + k}")
        A(f"  v_add3_u32 v{VO + k}, v{VO + k}, v{T}, s11")
    A("  s_waitcnt vmcnt(0)")
    for k in range(u):
        A(f"  s_mov_b64 exec, s[{40 + 2 * k}:{41 + 2 * k}]")
        for d in peers:
            A(
                f"  global_store_dwordx4 v{VO + k}, v[{VD + 4 * k}:{VD + 4 * k + 3}], {ds(d)} sc1"
            )
        A(
            f"  global_store_dwordx4 v{VO + k}, v[{VD + 4 * k}:{VD + 4 * k + 3}], {ds(R)}"
        )
    A("  s_mov_b64 exec, -1")
    A(f"  s_add_u32 s10, s10, {u * g * 64}")
    A("  s_cmp_lt_u32 s10, s9")
    A("  s_cbranch_scc1 .Lpass")
    # ---- hop (ar2): flags to the peers, poll theirs
    A(".Lhop:")
    A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
    if phase != "wait":
        A("  s_add_u32 s13, s13, 1")  # this launch's seq
    A("  s_mov_b64 exec, 1")
    A("  s_lshl_b32 s16, s2, 2")
    A(f"  s_add_u32 s17, s16, {R * g * 4}")
    A(f"  v_mov_b32 v{FV}, s17")
    A(f"  v_mov_b32 v{FV + 1}, s13")
    if phase != "wait":
        for d in peers:
            A(f"  global_store_dword v{FV}, v{FV + 1}, {fl(d)} sc0 sc1")
        if "wb" in POLL:
            A("  buffer_wbl2 sc0 sc1")
    if phase == "push":
        A("  s_branch .Ldone")
    for k, s_ in enumerate(peers):
        A(f"  s_add_u32 s17, s16, {s_ * g * 4}")
        A(f"  v_mov_b32 v{FV + 2 + k}, s17")
    # the guard words as the wait starts: a single hop's flags can be the peers' of a launch
    # they failed in, so the wait also stops on a stop word set before it began
    A(f"  v_mov_b32 v{EX}, 0")
    A(f"  global_load_dwordx2 v[{EX}:{EX + 1}], v{EX}, s[4:5] sc1")
    wg.start()
    A(".Lw:")
    wg.check("ag", SITES["eg_ag.ag"])
    if "inv" in POLL:
        A("  buffer_inv sc1")
    for k in range(3):
        A(f"  global_load_dword v{FV + 5 + k}, v{FV + 2 + k}, {fl(R)} {POLL_LD}")
    A("  s_waitcnt vmcnt(0)")
    for k in range(3):
        A(f"  v_readfirstlane_b32 s{88 + k}, v{FV + 5 + k}")
    for k in range(3):
        A(f"  s_cmp_eq_u32 s{88 + k}, s13")
        A("  s_cbranch_scc0 .Lw")
    # the loop's vmcnt(0) waited for the guard words too
    A(f"  v_readfirstlane_b32 s49, v{EX + 1}")
    A("  s_cmp_lg_u32 s49, 0")
    A("  s_cbranch_scc1 .Lgp_ag")
    A("  s_branch .Ldone")
    A(".Lidle:")
    if phase == "wait":
        A("  s_endpgm")  # the push already counted this launch
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_add_u32 s13, s13, 1")
    A(".Ldone:")
    if phase == "wait":
        A("  s_endpgm")
    A("  s_mov_b64 exec, 1")
    A(f"  v_mov_b32 v{FV}, s14")
    A(f"  v_mov_b32 v{FV + 1}, s13")
    A(f"  global_store_dword v{FV}, v{FV + 1}, {fl(R)}")
    A("  s_endpgm")
    # a wait past its deadline: the err word (this launch's seq), then the guard's stop
    wg.tails(
        [
            "  s_lshl_b32 s16, s2, 2",
            f"  s_add_u32 s16, s16, {err_off}",
            f"  v_mov_b32 v{FV}, s16",
            f"  v_mov_b32 v{FV + 1}, s13",
            f"  global_store_dword v{FV}, v{FV + 1}, {fl(R)} sc1",
        ]
    )
    src = kernel_source(name, c, KA, lds=0, vgprs=EX + 2, agprs=0, sgprs=96, block=64)
    return dict(name=name, src=src, ka=KA, grid=g)
