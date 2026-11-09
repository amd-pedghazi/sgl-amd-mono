# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/ar2.py (changed: import paths, the waits' deadline and fail-stop
# (asm/guard.py) instead of the 2^30-poll timeout, a guard kernarg)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""K2 v2a: all-reduce #2 of the MoE output across the 4 TP ranks,
reduce-scatter + all-gather, pushed over xGMI. Replaces vLLM's all-reduce of
``part`` [m, 5120] bf16.

- The payload is 4 flat quarters (m * 160 granules of 16 B each); rank q
  reduces quarter q. Grid G CTAs of one wave; CTA c owns granules
  [c S 64, (c + 1) S 64) of every quarter, S = ceil(m 160 / (G 64)) <= 4, and
  only talks to CTA c of the peers (one flag per (hop, source, CTA)).
- Hop 1: push quarter d of my part to rank d's landing slot R (sc1 stores),
  vmcnt(0), flag. Poll the 3 peers' flags, sum the 4 partials in fp32 in rank
  order (mine from registers), round to bf16 (RNE).
- Hop 2: store the reduced quarter into my ``out`` and every peer's ``out``
  (sc1), vmcnt(0), flag; poll the 3 peers' flags. ``out`` is then complete on
  every rank (the next kernel reads it).
- Seq numbers live in device memory: CTA c reads ctr[c], uses ctr + 1 as this
  launch's flag value and writes it back, so graph replays need no host input.
  Every rank runs the same launches, so the counters agree.
- No double buffering: peer d writes my landing (or out) chunk c for launch
  s + 1 only after it saw my CTA c's flags for launch s + 1, i.e. after I read
  the landing of launch s (hop 2 flag) and after my stream consumed out of
  launch s (it ran before this launch).
- Rules from xgmi.py: stores to peers sc1, polls sc1; plain stores to a peer
  stay in our L2.

- Each wait has the guard's deadline (asm/guard.py): on a timeout it records its
  site, stops every rank and its own launch; it never ends with stale data.

Kernargs: 12 pointers [land 0..3, flags 0..3, out 0..3] in this rank's
address space, then part, m and the guard block; land [4][QMAX], flags
(FLAG_BYTES: [2 hops][4 sources][G] dwords, then ctr [G], then err [G]), out
[256, 5120] bf16.
"""

from sgl_amd_mono.deepseek_v41.asm.guard import SITES, WaitGuard
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import KArgs, kernel_source

H, MAX_M = 5120, 256
G = 160
GL = G * 64  # granules per quarter one pass of the grid covers
QMAX = MAX_M * H * 2 // 4  # bytes of a quarter at m = 256
CTR_OFF = 2 * 4 * G * 4
ERR_OFF = CTR_OFF + G * 4
FLAG_BYTES = ERR_OFF + G * 4
POLL = frozenset()  # sc1 polls (no system-scope loads, invalidates or write-backs)
POLL_LD = "sc1"

KA = KArgs(
    *[(f"land{d}", "ptr") for d in range(4)],
    *[(f"flags{d}", "ptr") for d in range(4)],
    *[(f"out{d}", "ptr") for d in range(4)],
    ("part", "ptr"),
    ("m", "u32"),
    ("guard", "ptr"),
)


def source(R: int, name: str | None = None, stop: int = 99, halt: bool = True):
    """Stop (debug): end after 0 the prologue, 1 the part loads, 2 the pushes, 3 hop 1, 4 the reduce, 5 the
    all-gather stores. halt=False (tests only): a failed wait ends its wave instead of stopping the launch."""
    name = name or f"ar2_r{R}" + (f"_stop{stop}" if stop < 99 else "") + (
        "" if halt else "_nohalt"
    )
    peers = [d for d in range(4) if d != R]
    vA = 2  # v2..v5: byte offset of step j inside a quarter
    P = 8  # P[q][j] = quad P + 4 (4 q + j)
    L = 72  # L[k][j] = quad L + 4 (4 k + j), peer k
    Q = 120  # Q[j]
    T = 136  # temps
    FV = 141  # flag offset, value, 3 poll offsets, 3 polled values
    c = []
    A = c.append
    # free SGPRs (t0, now, gptr, table) or free in the loop (words s16:17, polls s18)
    wg = WaitGuard(
        A, R, KA.off["guard"], polls=18, t0=92, now=94, words=16, gptr=4, table=80,
        vp=FV + 5, halt=halt,
    )  # fmt: skip

    def pq(q, j):
        return P + 4 * (4 * q + j)

    def lq(k, j):
        return L + 4 * (4 * k + j)

    def fl(d):
        return f"s[{28 + 2 * d}:{29 + 2 * d}]"

    A("  s_load_dwordx16 s[20:35], s[0:1], 0")  # land 0..3, flags 0..3
    A("  s_load_dwordx8 s[36:43], s[0:1], 64")  # out 0..3
    A(f"  s_load_dwordx2 s[6:7], s[0:1], {KA.off['part']:#x}")
    A(f"  s_load_dword s8, s[0:1], {KA.off['m']:#x}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_mul_i32 s9, s8, 160")  # n: granules per quarter
    A("  s_lshl_b32 s10, s9, 4")  # quarter bytes
    A("  s_mov_b32 s11, 1")  # S
    for k in (1, 2, 3):
        A(f"  s_cmp_gt_u32 s9, {k * GL}")
        A("  s_addc_u32 s11, s11, 0")
    A("  s_mul_i32 s12, s2, s11")
    A("  s_lshl_b32 s12, s12, 6")  # first granule of this CTA
    A("  v_and_b32 v1, 63, v0")
    for j in range(4):
        A(f"  v_add_u32 v{vA + j}, s12, v1")
        A(f"  v_add_u32 v{vA + j}, {j * 64}, v{vA + j}")
        A(f"  v_cmp_gt_u32 s[{44 + 2 * j}:{45 + 2 * j}], s9, v{vA + j}")
        A(f"  s_cmp_lt_u32 {j}, s11")
        A(
            f"  s_cselect_b64 s[{44 + 2 * j}:{45 + 2 * j}], s[{44 + 2 * j}:{45 + 2 * j}], 0"
        )
        A(f"  v_lshlrev_b32 v{vA + j}, 4, v{vA + j}")
    A("  s_lshl_b32 s14, s2, 2")
    A(f"  s_add_u32 s14, s14, {CTR_OFF}")
    A(f"  s_load_dword s13, {fl(R)}, s14")
    for q in range(4):  # part quarter bases
        A(f"  s_mul_i32 s15, s10, {q}")
        A(f"  s_add_u32 s{52 + 2 * q}, s6, s15")
        A(f"  s_addc_u32 s{53 + 2 * q}, s7, 0")
    A("  s_cmp_eq_u64 s[44:45], 0")
    A("  s_cbranch_scc1 .Lidle")  # idle CTA (small m)
    if stop == 0:
        A("  s_branch .Lidle")

    # my part, peers' quarters first
    for q in peers + [R]:
        for j in range(4):
            A(f"  s_mov_b64 exec, s[{44 + 2 * j}:{45 + 2 * j}]")
            A(
                f"  global_load_dwordx4 v[{pq(q, j)}:{pq(q, j) + 3}], v{vA + j}, s[{52 + 2 * q}:{53 + 2 * q}]"
            )
    if stop <= 1:
        A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
        A("  s_add_u32 s13, s13, 1")
        A("  s_branch .Ldone")
    for k, d in enumerate(peers):  # push bases: land[d] + R QMAX
        A(f"  s_add_u32 s{60 + 2 * k}, s{20 + 2 * d}, {R * QMAX}")
        A(f"  s_addc_u32 s{61 + 2 * k}, s{21 + 2 * d}, 0")
    A(
        "  s_waitcnt vmcnt(0) lgkmcnt(0)"
    )  # (exec = 0 loads: don't rely on partial counts)
    A("  s_add_u32 s13, s13, 1")  # this launch's seq
    for k, d in enumerate(peers):
        for j in range(4):
            A(f"  s_mov_b64 exec, s[{44 + 2 * j}:{45 + 2 * j}]")
            A(
                f"  global_store_dwordx4 v{vA + j}, v[{pq(d, j)}:{pq(d, j) + 3}], s[{60 + 2 * k}:{61 + 2 * k}] sc1"
            )

    def hop(region, tag):
        A("  s_waitcnt vmcnt(0)")
        A("  s_mov_b64 exec, 1")
        A("  s_lshl_b32 s16, s2, 2")
        A(f"  s_add_u32 s17, s16, {(region * 4 * G + R * G) * 4}")
        A(f"  v_mov_b32 v{FV}, s17")
        A(f"  v_mov_b32 v{FV + 1}, s13")
        for d in peers:
            A(f"  global_store_dword v{FV}, v{FV + 1}, {fl(d)} sc0 sc1")
        if "wb" in POLL:
            A("  buffer_wbl2 sc0 sc1")
        for k, s in enumerate(peers):
            A(f"  s_add_u32 s17, s16, {(region * 4 * G + s * G) * 4}")
            A(f"  v_mov_b32 v{FV + 2 + k}, s17")
        wg.start()
        A(f".Lw{tag}:")
        wg.check(tag, SITES[f"ar2.{tag}"])
        if "inv" in POLL:
            A("  buffer_inv sc1")
        for k in range(3):
            A(f"  global_load_dword v{FV + 5 + k}, v{FV + 2 + k}, {fl(R)} {POLL_LD}")
        A("  s_waitcnt vmcnt(0)")
        for k in range(3):
            A(f"  v_readfirstlane_b32 s{88 + k}, v{FV + 5 + k}")
        for k in range(3):
            A(f"  s_cmp_eq_u32 s{88 + k}, s13")
            A(f"  s_cbranch_scc0 .Lw{tag}")
        A("  s_mov_b64 exec, -1")

    if stop == 2:
        A("  s_waitcnt vmcnt(0)")
        A("  s_branch .Ldone")
    hop(0, "rs")
    if stop == 3:
        A("  s_branch .Ldone")
    for k, s in enumerate(peers):  # landing slots in my memory
        A(f"  s_add_u32 s{66 + 2 * k}, s{20 + 2 * R}, {s * QMAX}")
        A(f"  s_addc_u32 s{67 + 2 * k}, s{21 + 2 * R}, 0")
    for j in range(4):
        A(f"  s_mov_b64 exec, s[{44 + 2 * j}:{45 + 2 * j}]")
        for k in range(3):
            A(
                f"  global_load_dwordx4 v[{lq(k, j)}:{lq(k, j) + 3}], v{vA + j}, s[{66 + 2 * k}:{67 + 2 * k}] sc1"
            )
    A("  s_mov_b64 exec, -1")
    A("  s_waitcnt vmcnt(0)")
    for j in range(4):
        for w in range(4):
            srcs = [
                pq(R, j) + w if r == R else lq(peers.index(r), j) + w for r in range(4)
            ]
            for half, a in ((0, T), (1, T + 1)):
                for i, sv in enumerate(srcs):
                    t = a if i == 0 else T + 2
                    A(
                        f"  v_lshlrev_b32 v{t}, 16, v{sv}"
                        if half == 0
                        else f"  v_and_b32 v{t}, 0xffff0000, v{sv}"
                    )
                    if i:
                        A(f"  v_add_f32 v{a}, v{a}, v{t}")
            A(f"  v_cvt_pk_bf16_f32 v{Q + 4 * j + w}, v{T}, v{T + 1}")
    if stop == 4:
        A("  s_branch .Ldone")
    for d in range(4):  # out[d] + R * quarter bytes
        A(f"  s_mul_i32 s15, s10, {R}")
        A(f"  s_add_u32 s{72 + 2 * d}, s{36 + 2 * d}, s15")
        A(f"  s_addc_u32 s{73 + 2 * d}, s{37 + 2 * d}, 0")
    for j in range(4):
        A(f"  s_mov_b64 exec, s[{44 + 2 * j}:{45 + 2 * j}]")
        for d in peers:
            A(
                f"  global_store_dwordx4 v{vA + j}, v[{Q + 4 * j}:{Q + 4 * j + 3}], s[{72 + 2 * d}:{73 + 2 * d}] sc1"
            )
        A(
            f"  global_store_dwordx4 v{vA + j}, v[{Q + 4 * j}:{Q + 4 * j + 3}], s[{72 + 2 * R}:{73 + 2 * R}]"
        )
    if stop == 5:
        A("  s_waitcnt vmcnt(0)")
        A("  s_branch .Ldone")
    hop(1, "ag")
    A("  s_branch .Ldone")
    A(".Lidle:")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_add_u32 s13, s13, 1")
    A(".Ldone:")
    A("  s_mov_b64 exec, 1")
    A(f"  v_mov_b32 v{FV}, s14")
    A(f"  v_mov_b32 v{FV + 1}, s13")
    A(f"  global_store_dword v{FV}, v{FV + 1}, {fl(R)}")
    A("  s_endpgm")
    # a wait past its deadline: the err word (this launch's seq), then the guard's stop
    wg.tails(
        [
            "  s_lshl_b32 s16, s2, 2",
            f"  s_add_u32 s16, s16, {ERR_OFF}",
            f"  v_mov_b32 v{FV}, s16",
            f"  v_mov_b32 v{FV + 1}, s13",
            f"  global_store_dword v{FV}, v{FV + 1}, {fl(R)} sc1",
        ]
    )
    src = kernel_source(name, c, KA, lds=0, vgprs=FV + 8, agprs=0, sgprs=96, block=64)
    return dict(name=name, src=src, ka=KA)


# ---------------------------------------------------------------- fp8 wire (M ≥ 192)

KA8 = KArgs(
    *[(f"land{d}", "ptr") for d in range(4)],
    *[(f"land2_{d}", "ptr") for d in range(4)],
    *[(f"flags{d}", "ptr") for d in range(4)],
    *[(f"out{d}", "ptr") for d in range(4)],
    ("part", "ptr"),
    ("m", "u32"),
    ("guard", "ptr"),
)
SCALE_OFF = QMAX // 2  # scale bytes of a landing slot (codes take ≤ QMAX / 2)


def flag_bytes(g: int = G) -> int:
    """Flags buffer of a g-CTA kernel: [2 hops][4 sources][g] dwords, ctr [g], err [g]."""
    return (2 * 4 * g + 2 * g) * 4


def source_fp8(
    R: int,
    name: str | None = None,
    g: int = G,
    knock: frozenset = frozenset(),
    halt: bool = True,
):
    """``source`` with an MXFP8 wire (e4m3 codes, one E8M0 per 32 elements =
    4 granules = a lane quad, vLLM's ceil rule):
    - RS: the 3 peer quarters of my part go out as codes (8 B per granule) +
      scale bytes to land[d] slot R; the owner sums its own exact quarter and
      the 3 dequantised ones in fp32 in rank order;
    - AG: the owner quantises the fp32 sum once and pushes codes + scales to
      land2[d] slot R; every rank (the owner too) writes bf16(dequant) of each
      quarter into its own ``out``, so all ranks hold the same values.
    Landing slot layout: codes at 8·granule, scales at SCALE_OFF + granule / 4.
    Bytes on the wire: 0.52× the bf16 kernel's. g CTAs (flags: flag_bytes(g)); a
    CTA takes ≤ 4 steps of 64 granules per quarter. knock (timing only, wrong
    results): "scales" no scale-byte stores to peers. halt=False (tests only): a
    failed wait ends its wave instead of stopping the launch."""
    name = name or f"ar2f8_r{R}" + (f"_g{g}" if g != G else "") + "".join(
        f"_k{k}" for k in sorted(knock)
    ) + ("" if halt else "_nohalt")
    gl, ctr_off = g * 64, 2 * 4 * g * 4
    err_off = ctr_off + g * 4
    peers = [d for d in range(4) if d != R]
    vA, vC, vS = 2, 6, 10  # per j: 16 g (bf16), 8 g (codes), g / 4 (scale)
    P = 16  # part: P[q][j] quad
    W = 80  # RS send: [k][j] codes 2 (even) + scale 1, stride 4
    L = 128  # receive: [k][j] codes 2 + scale 1, stride 4
    SM = 176  # reduced f32 [j] 8
    AGC = 208  # AG codes [j] 2; AG scales AG_SC + j
    AG_SC = 216
    O = 220  # noqa: E741  own bf16 out quarter [j] 4
    T = 236  # temps (12)
    FV = 248
    c = []
    A = c.append
    # free SGPRs (t0, now, gptr) or free in the loop (words s16:17, polls s18); land 0..3 dead on a failure
    wg = WaitGuard(
        A, R, KA8.off["guard"], polls=18, t0=92, now=94, words=16, gptr=14, table=20,
        vp=FV + 6, halt=halt,
    )  # fmt: skip

    def pq(q, j):
        return P + 4 * (4 * q + j)

    def wq(k, j):
        return W + 4 * (4 * k + j)

    def lq(k, j):
        return L + 4 * (4 * k + j)

    def fl(d):
        return f"s[{36 + 2 * d}:{37 + 2 * d}]"

    def ex(j):
        return f"s[{52 + 2 * j}:{53 + 2 * j}]"

    def quant(f, codes, scale):
        """8 f32 in v[f:f+8] (clobbered) → codes v[codes:codes+2], E8M0 byte in v[scale]; amax over the quad."""
        a, e, mt = T + 8, T + 9, T + 10
        A(f"  v_max_f32 v{a}, |v{f}|, |v{f + 1}|")
        for i in range(2, 8):
            A(f"  v_max_f32 v{a}, v{a}, |v{f + i}|")
        A("  s_nop 1")
        A(
            f"  v_max_f32_dpp v{a}, v{a}, v{a} quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        )
        A("  s_nop 1")
        A(
            f"  v_max_f32_dpp v{a}, v{a}, v{a} quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        )
        A(f"  v_max_f32 v{a}, 0x00800000, v{a}")  # FLT_MIN floor
        A(f"  v_frexp_exp_i32_f32 v{e}, v{a}")
        A(f"  v_frexp_mant_f32 v{mt}, v{a}")
        A(f"  v_cmp_lt_f32 vcc, 0x3f600000, v{mt}")
        A(f"  v_addc_co_u32 v{e}, vcc, -9, v{e}, vcc")
        A(f"  v_max_i32 v{e}, 0xffffff81, v{e}")
        A(f"  v_min_i32 v{e}, 0x7f, v{e}")
        A(f"  v_sub_u32 v{mt}, 0, v{e}")
        for i in range(8):
            A(f"  v_ldexp_f32 v{f + i}, v{f + i}, v{mt}")
        for h in range(2):
            A(f"  v_cvt_pk_fp8_f32 v{codes + h}, v{f + 4 * h}, v{f + 4 * h + 1}")
            A(
                f"  v_cvt_pk_fp8_f32 v{codes + h}, v{f + 4 * h + 2}, v{f + 4 * h + 3} op_sel:[0,0,1]"
            )
        A(f"  v_add_u32 v{scale}, 0x7f, v{e}")

    def dequant(codes, scale, f):
        """Codes v[codes:codes+2] + E8M0 byte v[scale] → 8 f32 v[f:f+8] (exact)."""
        sc = T + 11
        A(f"  v_lshlrev_b32 v{sc}, 23, v{scale}")
        for h in range(2):
            A(
                f"  v_cvt_scalef32_pk_f32_fp8 v[{f + 4 * h}:{f + 4 * h + 1}], v{codes + h}, v{sc}"
            )
            A(
                f"  v_cvt_scalef32_pk_f32_fp8 v[{f + 4 * h + 2}:{f + 4 * h + 3}], v{codes + h}, v{sc} op_sel:[1,0,0]"
            )

    def unpack(src, f):
        for k in range(4):
            A(f"  v_lshlrev_b32 v{f + 2 * k}, 16, v{src + k}")
            A(f"  v_and_b32 v{f + 2 * k + 1}, 0xffff0000, v{src + k}")

    def pack(f, dst):
        for k in range(4):
            A(f"  v_cvt_pk_bf16_f32 v{dst + k}, v{f + 2 * k}, v{f + 2 * k + 1}")

    def scale_store(j, sv, base_pair):
        """The step's 16 quad scales (byte in v[sv] of lanes 4b) → 4 dwords from lanes 0..3 (16 single-byte
        sc1 stores to a peer cost up to 2.8 µs per all-reduce at M 256)."""
        a, t = T, T + 1
        skip = f".Lss{len(c)}"
        A(f"  s_cmp_eq_u64 {ex(j)}, 0")  # steps past this CTA's range: their granules
        A(f"  s_cbranch_scc1 {skip}")  # belong to the next CTA
        A("  s_mov_b64 exec, -1")
        A(f"  v_lshlrev_b32 v{a}, 6, v1")  # 4 · 16 lane
        for i in range(4):
            A(f"  ds_bpermute_b32 v{t + i}, v{a}, v{sv} offset:{16 * i}")
        A(f"  s_add_u32 s91, s11, {64 * j}")
        A(f"  v_mad_u32_u24 v{a}, 16, v1, s91")  # granule of lane 16 k
        A(f"  v_cmp_gt_u32 vcc, s7, v{a}")
        A("  s_and_b64 exec, vcc, 0xf")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_and_b32 v{t}, 0xff, v{t}")
        for i in range(1, 4):
            A(f"  v_and_b32 v{t + i}, 0xff, v{t + i}")
            A(f"  v_lshl_or_b32 v{t}, v{t + i}, {8 * i}, v{t}")
        A(f"  v_lshlrev_b32 v{a}, 2, v1")
        A(f"  v_add_u32 v{a}, v{vS + j}, v{a}")  # lanes 0..3: g_step / 4 + 4 k
        A(f"  global_store_dword v{a}, v{t}, {base_pair} sc1")
        A(f"{skip}:")

    def bases(first, region_pairs, plus):
        """s[first + 2k : +1] = pair + plus (k-th entry of region_pairs = SGPR pair start)."""
        for k, sp in enumerate(region_pairs):
            A(f"  s_add_u32 s{first + 2 * k}, s{sp}, {plus}")
            A(f"  s_addc_u32 s{first + 2 * k + 1}, s{sp + 1}, 0")

    A("  s_load_dwordx16 s[20:35], s[0:1], 0")  # land 0..3, land2 0..3
    A("  s_load_dwordx16 s[36:51], s[0:1], 64")  # flags 0..3, out 0..3
    A(f"  s_load_dwordx2 s[4:5], s[0:1], {KA8.off['part']:#x}")
    A(f"  s_load_dword s6, s[0:1], {KA8.off['m']:#x}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_mul_i32 s7, s6, 160")  # n: granules per quarter
    A("  s_lshl_b32 s9, s7, 4")  # quarter bytes (bf16)
    A("  s_mov_b32 s10, 1")  # S
    for k in (1, 2, 3):
        A(f"  s_cmp_gt_u32 s7, {k * gl}")
        A("  s_addc_u32 s10, s10, 0")
    A("  s_mul_i32 s11, s2, s10")
    A("  s_lshl_b32 s11, s11, 6")  # first granule of this CTA
    A("  v_and_b32 v1, 63, v0")
    for j in range(4):
        A(f"  v_add_u32 v{vA + j}, s11, v1")
        A(f"  v_add_u32 v{vA + j}, {j * 64}, v{vA + j}")
        A(f"  v_cmp_gt_u32 {ex(j)}, s7, v{vA + j}")
        A(f"  s_cmp_lt_u32 {j}, s10")
        A(f"  s_cselect_b64 {ex(j)}, {ex(j)}, 0")
        A(f"  v_lshlrev_b32 v{vC + j}, 3, v{vA + j}")
        A(f"  v_lshrrev_b32 v{vS + j}, 2, v{vA + j}")
        A(f"  v_lshlrev_b32 v{vA + j}, 4, v{vA + j}")
    A("  s_lshl_b32 s12, s2, 2")
    A(f"  s_add_u32 s12, s12, {ctr_off}")
    A(f"  s_load_dword s13, {fl(R)}, s12")
    for q in range(4):  # part quarter bases
        A(f"  s_mul_i32 s91, s9, {q}")
        A(f"  s_add_u32 s{60 + 2 * q}, s4, s91")
        A(f"  s_addc_u32 s{61 + 2 * q}, s5, 0")
    A(f"  s_cmp_eq_u64 {ex(0)}, 0")
    A("  s_cbranch_scc1 .Lidle")
    for q in peers + [R]:
        for j in range(4):
            A(f"  s_mov_b64 exec, {ex(j)}")
            A(
                f"  global_load_dwordx4 v[{pq(q, j)}:{pq(q, j) + 3}], v{vA + j}, s[{60 + 2 * q}:{61 + 2 * q}]"
            )
    A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
    A("  s_add_u32 s13, s13, 1")  # this launch's seq
    # ---- RS: quantise the peers' quarters, push codes + scales to land[d] slot R
    bases(68, [20 + 2 * d for d in peers], R * QMAX)
    bases(74, [20 + 2 * d for d in peers], R * QMAX + SCALE_OFF)
    for k, d in enumerate(peers):
        for j in range(4):
            A(f"  s_mov_b64 exec, {ex(j)}")
            unpack(pq(d, j), T)
            quant(T, wq(k, j), wq(k, j) + 2)
            A(
                f"  global_store_dwordx2 v{vC + j}, v[{wq(k, j)}:{wq(k, j) + 1}], s[{68 + 2 * k}:{69 + 2 * k}] sc1"
            )
            if "scales" not in knock:
                scale_store(j, wq(k, j) + 2, f"s[{74 + 2 * k}:{75 + 2 * k}]")

    def hop(region, tag):
        A("  s_waitcnt vmcnt(0)")
        A("  s_mov_b64 exec, 1")
        A("  s_lshl_b32 s16, s2, 2")
        A(f"  s_add_u32 s17, s16, {(region * 4 * g + R * g) * 4}")
        A(f"  v_mov_b32 v{FV}, s17")
        A(f"  v_mov_b32 v{FV + 1}, s13")
        for d in peers:
            A(f"  global_store_dword v{FV}, v{FV + 1}, {fl(d)} sc0 sc1")
        if "wb" in POLL:
            A("  buffer_wbl2 sc0 sc1")
        for k, s_ in enumerate(peers):
            A(f"  s_add_u32 s17, s16, {(region * 4 * g + s_ * g) * 4}")
            A(f"  v_mov_b32 v{FV + 2 + k}, s17")
        wg.start()
        A(f".Lw{tag}:")
        wg.check(tag, SITES[f"ar2f8.{tag}"])
        if "inv" in POLL:
            A("  buffer_inv sc1")
        for k in range(3):
            A(f"  global_load_dword v{FV + 5 + k}, v{FV + 2 + k}, {fl(R)} {POLL_LD}")
        A("  s_waitcnt vmcnt(0)")
        for k in range(3):
            A(f"  v_readfirstlane_b32 s{88 + k}, v{FV + 5 + k}")
        for k in range(3):
            A(f"  s_cmp_eq_u32 s{88 + k}, s13")
            A(f"  s_cbranch_scc0 .Lw{tag}")
        A("  s_mov_b64 exec, -1")

    def receive(region_base):
        """land[R] / land2[R] slots of the 3 peers → L[k][j] (codes + scale)."""
        for k, s_ in enumerate(peers):
            A(f"  s_add_u32 s{68 + 2 * k}, s{region_base}, {s_ * QMAX}")
            A(f"  s_addc_u32 s{69 + 2 * k}, s{region_base + 1}, 0")
            A(f"  s_add_u32 s{74 + 2 * k}, s{region_base}, {s_ * QMAX + SCALE_OFF}")
            A(f"  s_addc_u32 s{75 + 2 * k}, s{region_base + 1}, 0")
        for j in range(4):
            A(f"  s_mov_b64 exec, {ex(j)}")
            for k in range(3):
                A(
                    f"  global_load_dwordx2 v[{lq(k, j)}:{lq(k, j) + 1}], v{vC + j}, s[{68 + 2 * k}:{69 + 2 * k}] sc1"
                )
                A(
                    f"  global_load_ubyte v{lq(k, j) + 2}, v{vS + j}, s[{74 + 2 * k}:{75 + 2 * k}] sc1"
                )
        A("  s_mov_b64 exec, -1")
        A("  s_waitcnt vmcnt(0)")

    hop(0, "rs")
    receive(20 + 2 * R)
    # ---- reduce in rank order (own quarter exact), quantise once, own out = bf16(dequant)
    for j in range(4):
        A(f"  s_mov_b64 exec, {ex(j)}")
        f = SM + 8 * j
        for i, r in enumerate(range(4)):
            dst = f if i == 0 else T
            if r == R:
                unpack(pq(R, j), dst)
            else:
                dequant(lq(peers.index(r), j), lq(peers.index(r), j) + 2, dst)
            if i:
                for e in range(8):
                    A(f"  v_add_f32 v{f + e}, v{f + e}, v{T + e}")
        quant(f, AGC + 2 * j, AG_SC + j)
        dequant(AGC + 2 * j, AG_SC + j, T)
        pack(T, O + 4 * j)
    # ---- AG: codes + scales to land2[d] slot R; own quarter into my out
    bases(68, [28 + 2 * d for d in peers], R * QMAX)
    bases(74, [28 + 2 * d for d in peers], R * QMAX + SCALE_OFF)
    A(f"  s_mul_i32 s91, s9, {R}")
    A(f"  s_add_u32 s80, s{44 + 2 * R}, s91")
    A(f"  s_addc_u32 s81, s{45 + 2 * R}, 0")
    for j in range(4):
        for k in range(3):
            A(f"  s_mov_b64 exec, {ex(j)}")
            A(
                f"  global_store_dwordx2 v{vC + j}, v[{AGC + 2 * j}:{AGC + 2 * j + 1}], s[{68 + 2 * k}:{69 + 2 * k}] sc1"
            )
            if "scales" not in knock:
                scale_store(j, AG_SC + j, f"s[{74 + 2 * k}:{75 + 2 * k}]")
        A(f"  s_mov_b64 exec, {ex(j)}")
        A(f"  global_store_dwordx4 v{vA + j}, v[{O + 4 * j}:{O + 4 * j + 3}], s[80:81]")
    hop(1, "ag")
    receive(28 + 2 * R)
    for k, s_ in enumerate(peers):  # the peers' quarters → my out
        A(f"  s_mul_i32 s91, s9, {s_}")
        A(f"  s_add_u32 s{82 + 2 * k}, s{44 + 2 * R}, s91")
        A(f"  s_addc_u32 s{83 + 2 * k}, s{45 + 2 * R}, 0")
    for j in range(4):
        A(f"  s_mov_b64 exec, {ex(j)}")
        for k in range(3):
            dequant(lq(k, j), lq(k, j) + 2, T)
            pack(T, pq(peers[k], j))
            A(
                f"  global_store_dwordx4 v{vA + j}, v[{pq(peers[k], j)}:{pq(peers[k], j) + 3}], s[{82 + 2 * k}:{83 + 2 * k}]"
            )
    A("  s_branch .Ldone")
    A(".Lidle:")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_add_u32 s13, s13, 1")
    A(".Ldone:")
    A("  s_mov_b64 exec, 1")
    A(f"  v_mov_b32 v{FV}, s12")
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
    src = kernel_source(name, c, KA8, lds=0, vgprs=FV + 8, agprs=0, sgprs=96, block=64)
    return dict(name=name, src=src, ka=KA8)
