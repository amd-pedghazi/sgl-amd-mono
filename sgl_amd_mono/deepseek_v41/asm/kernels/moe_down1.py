# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/moe_down1.py (changed: import paths)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Routed-MoE stage 2 (down), v1: expert-major, per-pick outputs.

v0 (moe_down.py) took units (16-row block, pick group) row-block-major: each
unit was a 4.5 KB w2 chunk of a different expert (scattered reads) and the
ring held ~2 units per wave: 4.1-4.6 TB/s, and removing the h gathers gained
only 2-9%. v1 streams each expert's w2 contiguously instead:

- Units (pick group g, chunk c of 16 row blocks = 256 output rows), group-
  major, contiguous equal ranges per CTA; wave w takes row blocks 4w..4w+3 of
  the chunk (18 KB of w2, contiguous apart from the K-padding half-tiles).
- Step = one row block: 5 K-tiles (the last half padding: lanes 0..31 loaded,
  the rest zeroed) into an 8-step AGPR ring; A scales (one dword per row-block
  pair and 2 K-tiles) on even steps; the odd block's scales are byte 1/3 of
  the same dword, so op_sel is static.
- The group's h (B fragments, 5 tiles × 8 VGPRs + 2 scale dwords + the pick's
  weight) stays in VGPRs; the next group's is prefetched only when the group
  changes (two unit-body variants: exact s_waitcnt counts).
- Output: y[p][row] = w_p · (W2 h_p)[row], fp32, one dwordx4 store per lane
  per step (lane t + 16 gq: rows 4 gq..4 gq+3 of pick t). Two accumulator sets
  alternate, so a step's result is read while the next step's MFMAs run.
  moe_combine.py sums each token's 6 rows into bf16.

Tables / tensors:
- units int32 [n, 20]: e, c, gid, 0, picks[16] (p = token*6 + k, or -1)
- h fp8 [P, 576]; hs uint8 [P, 4, 8]; wts fp32 [P]; y fp32 [P, 5120]
"""

from sgl_amd_mono.deepseek_v41.asm.kernels import gtable
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import KArgs, kernel_source, rsrc

H, E = 5120, 384
W2_E = H * 320
S2_E = H * 24
CH = 16  # row blocks per chunk
NCH = 320 // CH  # 20 chunks

KA = KArgs(
    ("w2", "ptr"),
    ("s2", "ptr"),
    ("h", "ptr"),
    ("hs", "ptr"),
    ("wts", "ptr"),
    ("groups", "ptr"),
    ("y", "ptr"),
    ("counts", "ptr"),
    ("h_bytes", "u32"),
    ("hs_bytes", "u32"),
    ("wts_bytes", "u32"),
    ("y_bytes", "u32"),
)


KA_COMB = KArgs(
    *KA.args,
    ("ids", "ptr"),
    ("shared", "ptr"),
    ("part", "ptr"),
    ("sync", "ptr"),
    ("m", "u32"),
)
PSTRIDE = 4  # sync: one progress word per CTA, packed (8 lines per aggregator poll)
NAGG = 8  # aggregator combiners (CTAs 0..7): global min, own line each
AGG_OFF = PSTRIDE * gtable.GRID
CTR_OFF = AGG_OFF + 128 * NAGG  # then per-CTA seq counters
TL_OFF = CTR_OFF + 4 * gtable.GRID  # timeline (s_memrealtime low dwords), 16 per CTA:
TL_WORDS = 16  # 0 start (wave 0), 1-4 stage-2 wave ends, 5-9 chunk go, 10 combiner end
SYNC_BYTES = TL_OFF + 4 * TL_WORDS * gtable.GRID


def source(
    split: str = "ceil",
    D: int = 4,
    grid: int = gtable.GRID,
    PER_MAX: int = 128,
    w_pol: str = "nt",
    name: str | None = None,
    early_w: int = 0,
    knock: str = "",
    y_pol: str = "",
    counts_vec: bool = False,
    comb: bool = False,
    csleep: int = 1,
    cknock: str = "",
    cg: int = 4,
    asleep: int = 1,
    y_bf16: bool = False,
    y_batch: bool = False,
    y_exec: bool = False,
    s_pol: str = "",
):
    """Comb (K2 v2b, split "rr" only): a 5th wave per CTA combines column chunks
    while stage 2 runs (see combiner()); kernargs KA_COMB.
    early_w: issue the first D-1 steps' weight loads before the h gather.
    knock (perf experiments only, wrong results): "s" no y stores, "m" no MFMAs,
    "a" no A-scale loads, "c" stores to one contiguous L2-resident 1 KB per wave,
    "E" no stores during the stream, then each wave writes y_bytes / 1024 contiguous
    bytes (512 B dwordx2 stores) after its last step,
    "h" 8 B (bf16-sized) stores at the real addresses, "C" bf16-sized stores
    (dwordx2, every lane) to one L2-resident 512 B per wave.
    y_bf16: y in bf16, [P, 5120] (half the bytes written and read
    back by the combine; the fp32 stores cost ~3 bytes of weight stream each):
    one dwordx2 per lane per step, the same instruction counts.
    y_batch (with y_bf16): a unit's 4 results go out together at the next
    unit's first step (steps 0-2 parked in 6 VGPRs). vmcnt retires in issue
    order and a store under load takes ~5-10k clk to complete (measured),
    so every weight load issued after a store waits for it: once per unit
    instead of once per step.
    y_exec (with y_bf16): the y store runs under EXEC = the lanes with a pick
    (vY != the out-of-range v0) instead of sending every lane to the TA.
    s_pol: cache policy of the A-scale loads (no reuse; plain by default)."""
    SPU = 4  # steps (row blocks) per unit per wave
    UST = grid if split == "rr" else 1  # unit stride of a CTA
    # D ≤ SPU keeps a prefetch's effect on the wait counts inside its own unit;
    # D | SPU keeps ring slots static (slot = step % D).
    assert D <= SPU and SPU % D == 0
    UPI = 1
    LPS = 5 + (
        0 if "a" in knock else 3
    )  # loads per step: W 5 + A scales 3 (every step: no slot race)
    XL = 10 + 2 + 1  # h prefetch: 10 B + 2 SB + 1 weight
    R = 80
    TBL = (PER_MAX * R + 255) // 256 * 256
    lds = (
        TBL if split != "rr" else (E + 6 * 256 // 16) * R
    )  # rr: the whole table (NG <= E + P / 16)
    PLDS = lds  # comb: 4 progress slots (one per stage-2 wave)
    if comb:
        lds += 16
    assert not comb or (split == "rr" and grid == gtable.GRID and D == 4), (
        "comb: rr split, grid 256, D 4"
    )
    name = (
        name
        or f"moe_down1{'c' + str(csleep) + 'a' + str(asleep) + cknock + '_g' + str(cg) if comb else ''}_{split}_d{D}_g{grid}{'_' + w_pol.replace(' ', '') if w_pol else ''}_e{early_w}{'_k' + knock if knock else ''}{'_y' + y_pol.replace(' ', '') if y_pol else ''}{'_yb' if y_bf16 else ''}{'_bt' if y_batch else ''}{'_ye' if y_exec else ''}{'_s' + s_pol.replace(' ', '') if s_pol else ''}"
    )
    pol = f" {w_pol}" if w_pol else ""

    # VGPRs: v0 OOB, v1 lane, v2 lane*16, v3 lane*16+4096, v4 lane*4, v5 t, v6 gq
    vB, vS, vT, vY, vP = (
        8,
        9,
        10,
        11,
        12,
    )  # current group: h off, hs off, wts off, y off, p
    nB, nS, nT, nY, nP = 13, 14, 15, 16, 17
    SA0 = 18  # SA ring: 3 per slot
    XC = SA0 + 3 * D + 1
    XC += XC % 2
    XN = XC + 44
    TMP = XN + 44
    t = [TMP + i for i in range(16)]
    vgprs = TMP + 16
    assert not y_batch or (
        y_bf16 and split != "rr" and not comb and D == SPU == 4 and not knock
    ), "y_batch"
    hold = (
        [vgprs + i for i in range(6)] if y_batch else []
    )  # bf16 y of a unit's steps 0-2
    vgprs += len(hold)
    assert vgprs <= 256, vgprs
    HB, HSB, HW = 0, 40, 42  # offsets inside an h buffer: B, SB (2), weight
    WR0 = 8  # AGPRs: acc a[0:3], a[4:7]; ring a[8 + 20 * slot]
    agprs = WR0 + 20 * D

    def sa(s):
        return SA0 + 3 * (s % D)

    c = []
    A = c.append
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")
    A("  s_load_dwordx4 s[20:23], s[0:1], 0x40")
    A("  v_readfirstlane_b32 s48, v0")
    A("  s_lshr_b32 s48, s48, 6")
    if comb:
        # wave 4 combines; the stage-2 waves publish progress words (sync + 4 (4 b + w)), seq from ctr[b]
        A("  s_cmp_eq_u32 s48, 4")
        A("  s_cbranch_scc1 .Lcomb")
        A(f"  s_load_dwordx2 s[88:89], s[0:1], {KA_COMB.off['sync']:#x}")
        A("  s_lshl_b32 s95, s2, 2")
        A(f"  s_add_u32 s95, s95, {CTR_OFF}")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_load_dword s94, s[88:89], s95")
        A("  s_lshl_b32 s95, s48, 2")
        A(f"  v_mov_b32 v{t[14]}, {PLDS}")
        A(f"  v_add_u32 v{t[14]}, s95, v{t[14]}")  # my LDS progress slot
        A("  s_memrealtime s[68:69]")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_add_u32 s94, s94, 1")
        A("  s_lshl_b32 s94, s94, 20")  # this launch's seq << 20
        A("  s_add_u32 s95, s94, s2")  # nothing done yet: next unit = my first (b)
        A(f"  v_mov_b32 v{t[15]}, s95")
        A(
            f"  ds_write_b32 v{t[14]}, v{t[15]}"
        )  # before range_and_copy's lgkmcnt(0) + s_barrier
    A("  v_and_b32 v1, 63, v0")
    A("  v_lshlrev_b32 v2, 4, v1")
    A("  v_add_u32 v3, 0x1000, v2")
    A("  v_lshlrev_b32 v4, 2, v1")
    A("  v_and_b32 v5, 15, v1")
    A("  v_lshrrev_b32 v6, 4, v1")
    A("  s_waitcnt lgkmcnt(0)")
    # kernargs: w2 s[4:5] s2 s[6:7] h s[8:9] hs s[10:11] wts s[12:13] units s[14:15] y s[16:17]
    #           nunits s18 per s19 h_bytes s20 hs_bytes s21 wts_bytes s22 y_bytes s23
    c += rsrc(24, 4, nbytes=-1)
    c += rsrc(28, 6, nbytes=-1)
    c += rsrc(32, 8, nbytes_sgpr=20)
    c += rsrc(36, 10, nbytes_sgpr=21)
    c += rsrc(40, 12, nbytes_sgpr=22)
    c += rsrc(44, 16, nbytes_sgpr=23)
    c += gtable.range_and_copy(
        20,
        counts=18,
        groups=14,
        tv=[t[0], t[1], t[2]],
        end_label=".Lend",
        grid=grid,
        split=split,
        counts_vec=counts_vec,
    )
    A("  s_mov_b32 s49, s52")
    A("  s_mov_b32 s73, 0xaaaaaaab")
    A("  v_mov_b32 v0, 0x7fff0000")

    # SGPRs per unit: sW (w2 offset of the wave's first block), sA (scale offset), sY (y row offset), gid
    sW, sA, sY, sG = 56, 57, 58, 59
    nW, nA, nY_, nG = 60, 61, 62, 63
    # s75: previous unit's gid; s64..s67: per-step W soffsets (computed per unit)

    def params(u_sgpr, W, Asc, Ys, Gs, B, S, T, Y, P):
        t0, t2 = t[0], t[2]
        o = (
            gtable.decode_rr(u_sgpr, Gs, 78)
            if split == "rr"
            else gtable.decode(20, u_sgpr, Gs, 78)
        )
        o += [
            f"  v_mov_b32 v{t0}, s76",
            f"  ds_read_b32 v{t2}, v{t0}",
            f"  v_lshlrev_b32 v{t0}, 2, v5",
            f"  v_add_u32 v{t0}, s76, v{t0}",
            f"  ds_read_b32 v{P}, v{t0} offset:16",
            "  s_waitcnt lgkmcnt(0)",
            f"  v_readfirstlane_b32 s77, v{t2}",
            # first row block of this wave: n0 = 16 c + 4 w
            "  s_lshl_b32 s79, s78, 4",
            "  s_lshl_b32 s74, s48, 2",
            "  s_add_u32 s79, s79, s74",
            f"  s_mul_i32 s{W}, s77, {W2_E}",
            f"  s_mul_i32 s74, s79, {H}",
            f"  s_add_u32 s{W}, s{W}, s74",
            f"  s_mul_i32 s{Asc}, s77, {S2_E}",
            "  s_lshr_b32 s74, s79, 1",
            "  s_mul_i32 s74, s74, 768",
            f"  s_add_u32 s{Asc}, s{Asc}, s74",
            f"  s_lshl_b32 s{Ys}, s79, {5 if y_bf16 else 6}",  # y row offset: n0 * 16 rows * 4 B (2 B)
            f"  v_cmp_le_i32 vcc, 0, v{P}",
            f"  v_mul_u32_u24 v{t0}, 0x240, v{P}",
            f"  v_lshl_add_u32 v{t0}, v6, 4, v{t0}",
            f"  v_cndmask_b32 v{B}, v0, v{t0}, vcc",
            f"  v_lshlrev_b32 v{t0}, 5, v{P}",
            f"  v_lshl_add_u32 v{t0}, v6, 3, v{t0}",
            f"  v_cndmask_b32 v{S}, v0, v{t0}, vcc",
            f"  v_lshlrev_b32 v{t0}, 2, v{P}",
            f"  v_cndmask_b32 v{T}, v0, v{t0}, vcc",
            f"  v_mul_u32_u24 v{t0}, {H * (2 if y_bf16 else 4)}, v{P}",
            f"  v_lshl_add_u32 v{t0}, v6, {3 if y_bf16 else 4}, v{t0}",
            f"  v_cndmask_b32 v{Y}, v0, v{t0}, vcc",
        ]
        return o

    def w_loads(s, rel, W, Asc):
        """Loads for global step s = row block `rel` (0..3) of a unit with offsets W/Asc."""
        slot = WR0 + 20 * (s % D)
        o = [f"  s_add_u32 s64, s{W}, {rel * H}"]
        if "r" in knock:  # each step from a different expert (row-block-major pattern)
            o += [
                "  s_mul_i32 s85, s48, 24",
                f"  s_add_u32 s85, s85, {rel * 96}",
                f"  s_mul_i32 s85, s85, {W2_E}",
                "  s_add_u32 s64, s64, s85",
                f"  s_cmp_ge_u32 s64, {E * W2_E}",
                f"  s_cselect_b32 s87, {E * W2_E}, 0",
                "  s_sub_u32 s64, s64, s87",
            ]
        o += [
            f"  buffer_load_dwordx4 a[{slot + 4 * k}:{slot + 4 * k + 3}], v2, s[24:27], s64 offen offset:{k * 1024}{pol}"
            for k in range(4)
        ]
        o += ["  s_mov_b32 exec_lo, 0", "  s_mov_b32 exec_hi, -1"]
        o += [f"  v_accvgpr_write_b32 a{slot + 16 + q}, 0" for q in range(4)]
        o += [
            "  s_mov_b32 exec_lo, -1",
            "  s_mov_b32 exec_hi, 0",
            f"  buffer_load_dwordx4 a[{slot + 16}:{slot + 19}], v3, s[24:27], s64 offen{pol}",
            "  s_mov_b64 exec, -1",
        ]
        if "a" not in knock:
            o += [
                f"  buffer_load_dword v{sa(s) + k1}, v4, s[28:31], s{Asc} offen offset:{(rel // 2) * 768 + k1 * 256}"
                + (f" {s_pol}" if s_pol else "")
                for k1 in range(3)
            ]
        return o

    def h_loads(buf, B, S, T):
        o = []
        for k0 in range(5):
            r = buf + HB + 8 * k0
            o += [
                f"  buffer_load_dwordx4 v[{r}:{r + 3}], v{B}, s[32:35], 0 offen offset:{k0 * 128}",
                f"  buffer_load_dwordx4 v[{r + 4}:{r + 7}], v{B}, s[32:35], 0 offen offset:{k0 * 128 + 64}",
            ]
        o += [
            f"  buffer_load_dword v{buf + HSB + q}, v{S}, s[36:39], 0 offen offset:{4 * q}"
            for q in range(2)
        ]
        o.append(f"  buffer_load_dword v{buf + HW}, v{T}, s[40:43], 0 offen")
        return o

    def mfmas(s):
        slot = WR0 + 20 * (s % D)
        acc = 4 * (s % 2)
        odd = s % 2  # odd row block of the pair: scale byte +1
        o = []
        if "m" in knock:
            return o
        for k0 in range(5):
            sel_a = (k0 % 2) * 2 + odd
            sel_b = k0 % 4
            cin = "0" if k0 == 0 else f"a[{acc}:{acc + 3}]"
            o.append(
                f"  v_mfma_scale_f32_16x16x128_f8f6f4 a[{acc}:{acc + 3}], a[{slot + 4 * k0}:{slot + 4 * k0 + 3}], "
                f"v[{XC + HB + 8 * k0}:{XC + HB + 8 * k0 + 7}], {cin}, v{sa(s) + k0 // 2}, v{XC + HSB + k0 // 4} "
                f"op_sel:[{sel_a & 1},{sel_b & 1},0] op_sel_hi:[{sel_a >> 1},{sel_b >> 1},0] cbsz:4 blgp:0"
            )
        return o

    def store_prev(s, rel_prev, Y, Ys, WT):
        """Y for step s-1 (accumulator set (s-1)%2): ×w, one dwordx4 store.
        y_batch: rel_prev 0-2 → parked in hold; 3 → the unit's 4 stores."""
        acc = 4 * ((s - 1) % 2)
        o = ["  s_nop 7", "  s_nop 7"]  # MFMA → accumulator read
        o += [f"  v_accvgpr_read_b32 v{t[8 + q]}, a{acc + q}" for q in range(4)]
        o += [f"  v_mul_f32 v{t[8 + q]}, v{t[8 + q]}, v{WT}" for q in range(4)]
        if y_batch:
            if rel_prev < 3:
                h0 = hold[2 * rel_prev]
                return o + [
                    f"  v_cvt_pk_bf16_f32 v{h0}, v{t[8]}, v{t[9]}",
                    f"  v_cvt_pk_bf16_f32 v{h0 + 1}, v{t[10]}, v{t[11]}",
                ]
            o += [
                f"  v_cvt_pk_bf16_f32 v{t[8]}, v{t[8]}, v{t[9]}",
                f"  v_cvt_pk_bf16_f32 v{t[9]}, v{t[10]}, v{t[11]}",
            ]
            for r in range(4):
                d = (
                    f"v[{hold[2 * r]}:{hold[2 * r] + 1}]"
                    if r < 3
                    else f"v[{t[8]}:{t[9]}]"
                )
                o.append(
                    f"  buffer_store_dwordx2 {d}, v{Y}, s[44:47], s{Ys} offen offset:{r * 32}"
                    + (f" {y_pol}" if y_pol else "")
                )
            return o
        if "s" in knock or "E" in knock:
            o.append("  buffer_store_dword v0, v0, s[44:47], 0 offen")
        elif (
            "C" in knock
        ):  # bf16-sized (dwordx2, all lanes) to one L2-resident 512 B per wave
            o += [
                f"  v_cvt_pk_bf16_f32 v{t[8]}, v{t[8]}, v{t[9]}",
                f"  v_cvt_pk_bf16_f32 v{t[9]}, v{t[10]}, v{t[11]}",
                "  s_lshl_b32 s83, s2, 2",
                "  s_add_u32 s83, s83, s48",
                "  s_lshl_b32 s83, s83, 9",
                f"  v_lshlrev_b32 v{t[13]}, 1, v4",
                f"  buffer_store_dwordx2 v[{t[8]}:{t[9]}], v{t[13]}, s[44:47], s83 offen",
            ]
        elif "c" in knock:  # same 1 KB per wave every time: contiguous, L2-resident
            o += [
                "  s_lshl_b32 s83, s2, 2",
                "  s_add_u32 s83, s83, s48",
                "  s_lshl_b32 s83, s83, 10",
                f"  buffer_store_dwordx4 v[{t[8]}:{t[11]}], v2, s[44:47], s83 offen",
            ]
        elif (
            "u" in knock
        ):  # valid lanes only, pick-major n*64 B contiguous, fresh lines
            o += [
                f"  v_cmp_ne_u32 vcc, v{Y}, v0",
                f"  v_lshlrev_b32 v{t[13]}, 6, v5",
                f"  v_lshl_add_u32 v{t[13]}, v6, 4, v{t[13]}",
                f"  v_cndmask_b32 v{t[13]}, v0, v{t[13]}, vcc",
                "  s_lshl_b32 s83, s49, 2",
                "  s_add_u32 s83, s83, s48",
                "  s_lshl_b32 s83, s83, 2",
                f"  s_add_u32 s83, s83, {rel_prev}",
                "  s_lshl_b32 s83, s83, 10",
                "  s_flbit_i32_b32 s84, s23",
                "  s_sub_u32 s84, 31, s84",
                "  s_lshl_b32 s84, 1, s84",
                "  s_sub_u32 s84, s84, 1",
                "  s_and_b32 s83, s83, s84",
                f"  buffer_store_dwordx4 v[{t[8]}:{t[11]}], v{t[13]}, s[44:47], s83 offen",
            ]
        elif "h" in knock:  # bf16-sized: 8 B per lane at the real (scattered) address
            o.append(
                f"  buffer_store_dwordx2 v[{t[8]}:{t[9]}], v{Y}, s[44:47], s{Ys} offen offset:{rel_prev * 64}"
            )
        elif y_bf16:
            o += [
                f"  v_cvt_pk_bf16_f32 v{t[8]}, v{t[8]}, v{t[9]}",
                f"  v_cvt_pk_bf16_f32 v{t[9]}, v{t[10]}, v{t[11]}",
            ]
            if y_exec:
                o += [
                    f"  v_cmp_ne_u32 s[84:85], v{Y}, v0",
                    "  s_and_saveexec_b64 s[86:87], s[84:85]",
                ]
            o += [
                f"  buffer_store_dwordx2 v[{t[8]}:{t[9]}], v{Y}, s[44:47], s{Ys} offen offset:{rel_prev * 32}"
                + (f" {y_pol}" if y_pol else "")
            ]
            if y_exec:
                o += ["  s_mov_b64 exec, s[86:87]"]
        else:
            o.append(
                f"  buffer_store_dwordx4 v[{t[8]}:{t[11]}], v{Y}, s[44:47], s{Ys} offen offset:{rel_prev * 64}"
                + (f" {y_pol}" if y_pol else "")
            )
        return o

    # ---- prologue
    c += params(52, sW, sA, sY, sG, vB, vS, vT, vY, vP)
    if not early_w:
        c += h_loads(XC, vB, vS, vT)
        A("  s_waitcnt vmcnt(0)")
    A(f"  s_mov_b32 s75, s{sG}")
    A(f"  s_add_u32 s53, s52, {UST}")
    A("  s_sub_u32 s54, s50, 1")
    A("  s_min_u32 s53, s53, s54")
    c += params(53, nW, nA, nY_, nG, nB, nS, nT, nY, nP)
    # the wait counts assume exactly one store per iteration after its loads:
    # out-of-range dummy stores (dropped) stand in where there is no result yet
    dummy = "  buffer_store_dword v0, v0, s[44:47], 0 offen"
    for s in range(D - 1):
        u, rel = divmod(s, SPU)
        assert u <= 1
        c += w_loads(s, rel, sW if u == 0 else nW, sA if u == 0 else nA)
        if not y_batch:
            A(dummy)
    if early_w:
        # the h gather overlaps the first weight loads; waiting for all of it
        # here only makes the loop's first counts conservative
        c += h_loads(XC, vB, vS, vT)
        A("  s_waitcnt vmcnt(0)")
    # the first step has no previous result: a dummy accumulator store would be
    # out of range; instead the stream starts with step 0 storing nothing
    A("  s_mov_b32 s70, 0")  # first step flag

    # unit loop: UPI units per iteration so ring slots are static. Within a unit
    # body the steps are 4u'..4u'+3 (u' = position in the iteration).
    def unit_body(up: int, pref: bool, tag: str):
        o = []
        for rel in range(SPU):
            s = up * SPU + rel
            nxt = s + D - 1  # global step to issue loads for (relative)
            nu, nrel = divmod(nxt, SPU)
            dist = nu - up  # 0: this unit, 1: next unit, ≥2: beyond (not tracked)
            assert dist <= 1, (
                "ring deeper than one unit ahead: needs a second 'next' set"
            )
            o += w_loads(nxt, nrel, sW if dist == 0 else nW, sA if dist == 0 else nA)
            # newer than step s's loads (issued D-1 iterations ago): the loads of
            # steps s+1..s+D-1, the D-1 stores issued since (one per iteration),
            # and this unit's h prefetch (issued at rel 0) when rel ≥ 1
            # (only steps whose loads were issued before it, i.e. at or before rel 0)
            xl = XL if (pref and 1 <= rel <= D - 1) else 0
            # stores younger than step s's loads: one per iteration; y_batch: 4 at rel 0, so 4 if
            # this unit's rel 0 is among the last D - 1 iterations, else 0
            nst = (4 if rel >= 1 else 0) if y_batch else D - 1
            wait = (D - 1) * LPS + nst + xl
            assert wait <= 63, wait
            o.append(f"  s_waitcnt vmcnt({wait})")
            if rel == 0:
                # previous step's result (its unit's params are still current here)
                o += ["  s_cmp_eq_u32 s70, 0", f"  s_cbranch_scc1 .Lnostore{tag}"]
                o += store_prev(s, SPU - 1, t[12], 71, XC + HW)
                o += (
                    [f"  s_branch .Lstored{tag}", f".Lnostore{tag}:"]
                    + [dummy] * (4 if y_batch else 1)
                    + [f".Lstored{tag}:"]
                )
                o += [f"  s_cmp_eq_u32 s{sG}, s75", f"  s_cbranch_scc1 .Lnocopy{tag}"]
                o += [f"  v_mov_b32 v{XC + i}, v{XN + i}" for i in range(44)]
                o += [f".Lnocopy{tag}:", "  s_nop 1"]
                if pref:
                    o += h_loads(XN, nB, nS, nT)
                if comb:
                    # every op issued before this step's loads is done (vmcnt retires in issue order): all
                    # y stores of the unit two back. Next unit not covered = the previous one (u - grid),
                    # or this one at the start. To the CTA's combiner through LDS.
                    o += [
                        f"  s_sub_u32 s95, s49, {grid}",
                        "  s_cselect_b32 s95, s49, s95",
                        "  s_add_u32 s95, s95, s94",
                        f"  v_mov_b32 v{t[15]}, s95",
                        f"  ds_write_b32 v{t[14]}, v{t[15]}",
                    ]
            else:
                o += store_prev(s, rel - 1, vY, sY, XC + HW)
            o += mfmas(s)
        return o

    A(".Lunit:")
    for up in range(UPI):
        A(f"  s_cmp_eq_u32 s{nG}, s{sG}")
        A(f"  s_cbranch_scc0 .Lpref{up}")
        c += unit_body(up, False, f"n{up}")
        A(f"  s_branch .Ladv{up}")
        A(f".Lpref{up}:")
        c += unit_body(up, True, f"p{up}")
        A(f".Ladv{up}:")
        # advance: remember this unit's y offset for the delayed store of its last step
        A(f"  s_mov_b32 s71, s{sY}")
        A("  s_mov_b32 s70, 1")
        A(f"  s_mov_b32 s75, s{sG}")
        for a_, b_ in ((sW, nW), (sA, nA), (sY, nY_), (sG, nG)):
            A(f"  s_mov_b32 s{a_}, s{b_}")
        A(f"  v_mov_b32 v{t[12]}, v{vY}")  # previous y offset for the delayed store
        for a_, b_ in ((vB, nB), (vS, nS), (vT, nT), (vY, nY), (vP, nP)):
            A(f"  v_mov_b32 v{a_}, v{b_}")
        A(f"  s_add_u32 s49, s49, {UST}")
        A("  s_cmp_lt_u32 s49, s50")
        A("  s_cbranch_scc0 .Ldone")
        A(f"  s_add_u32 s53, s49, {UST}")
        A("  s_sub_u32 s54, s50, 1")
        A("  s_min_u32 s53, s53, s54")
        c += params(53, nW, nA, nY_, nG, nB, nS, nT, nY, nP)
    A("  s_branch .Lunit")
    A(".Ldone:")
    # the last step's result
    A("  s_nop 7\n  s_nop 7")
    c += store_prev(SPU * UPI, SPU - 1, t[12], 71, XC + HW)
    if "E" in knock:  # the same y bytes as one burst per wave, after its stream
        A(
            "  s_lshr_b32 s84, s23, 19"
        )  # stores of 512 B per wave = y_bytes / 1024 / 512
        A("  s_lshl_b32 s83, s2, 2")
        A("  s_add_u32 s83, s83, s48")
        A("  s_mul_i32 s83, s83, s84")
        A("  s_lshl_b32 s83, s83, 9")  # wave's region
        A(f"  v_lshlrev_b32 v{t[13]}, 1, v4")
        A(".LEloop:")
        A("  s_cmp_eq_u32 s84, 0")
        A("  s_cbranch_scc1 .LEdone")
        A(f"  buffer_store_dwordx2 v[{t[8]}:{t[9]}], v{t[13]}, s[44:47], s83 offen")
        A("  s_add_u32 s83, s83, 512")
        A("  s_sub_u32 s84, s84, 1")
        A("  s_branch .LEloop")
        A(".LEdone:")
    A("  s_waitcnt vmcnt(0)")
    A(".Lend:")
    if comb:
        A("  s_or_b32 s95, s94, 0xfffff")  # finished
        A(f"  v_mov_b32 v{t[15]}, s95")
        A(f"  ds_write_b32 v{t[14]}, v{t[15]}")
        A("  s_memrealtime s[86:87]")
        A(f"  s_mul_i32 s95, s2, {4 * TL_WORDS}")
        A(f"  s_add_u32 s95, s95, {TL_OFF}")
        A("  s_lshl_b32 s51, s48, 2")
        A("  s_add_u32 s51, s51, s95")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_mov_b32 v{t[15]}, s86")
        A(f"  v_mov_b32 v{t[14]}, s51")
        A(f"  global_store_dword v{t[14]}, v{t[15]}, s[88:89] offset:4")
        A("  s_cmp_eq_u32 s48, 0")
        A("  s_cbranch_scc0 .Lnostart")
        A(f"  v_mov_b32 v{t[15]}, s68")
        A(f"  global_store_dword v{t[14]}, v{t[15]}, s[88:89]")
        A(".Lnostart:")
    A("  s_endpgm")
    if comb:
        c += combiner(t, csleep, cknock, cg, PLDS, asleep)
    src = kernel_source(
        name,
        c,
        KA_COMB if comb else KA,
        lds=lds,
        vgprs=vgprs,
        agprs=agprs,
        sgprs=96,
        block=320 if comb else 256,
    )
    return dict(
        name=name,
        src=src,
        ka=KA_COMB if comb else KA,
        grid=grid,
        PER_MAX=PER_MAX,
        lds=lds,
        vgprs=vgprs,
        block=320 if comb else 256,
    )


def combiner(
    t: list[int],
    csleep: int = 1,
    cknock: str = "",
    CG: int = 4,
    PLDS: int = 0,
    asleep: int = 1,
) -> list[str]:
    """Wave 4 of each CTA (comb=True): part[token][col] = bf16(bf16(sum_k y[token*6+k][col]) + shared), same
    arithmetic as moe_combine add=True. Granule = (token, 8 columns); chunk c = columns 256c..256c+255, M*32
    granules. The 256 combiner waves form CG groups: wave b is in group q = b % CG, which owns chunks q, q + CG,
    ...; inside a chunk the group's 256/CG waves split the granules evenly (per = ceil(M*32 CG / 256) <= 128,
    two 64-lane passes, loaded together).
    Progress: the CTA's 4 stage-2 waves write "seq<<20 | next unit not done" into LDS slots; the combiner
    publishes their min as its CTA's word (sync + 128 b, sc1) whenever it changed, in every poll, and after its
    own chunks keeps publishing until its stage-2 waves are finished (so no combiner waits on a CTA whose
    combiner left). Polling all 256 words from every combiner slowed stage 2 by 30-50% (sc1 traffic on the
    hot lines), so combiners 0..NAGG-1 aggregate: they read the 256 words and publish the global min (d = word
    - seq<<20, stale words from seq - 1 count as 0), the others read one aggregator word (b % NAGG).
    Aggregators drain until everything is finished. A chunk starts once d >= (c+1) NG. Then ctr[b] = seq."""
    assert CG & (CG - 1) == 0 and CG <= 8
    lg = CG.bit_length() - 1
    NWG = gtable.GRID // CG  # waves per group
    SUB = 2
    o = []
    A = o.append
    off = KA_COMB.off
    YO = [
        10 + 7 * s for s in range(SUB)
    ]  # per pass: 6 y offsets + 1 shared/part offset
    LD = [32 + 52 * s for s in range(SUB)]  # per pass: 12 y quads + shared quad
    MASK = [40 + 2 * s for s in range(SUB)]
    A(".Lcomb:")
    A(
        "  s_waitcnt lgkmcnt(0)"
    )  # the prologue's kernarg loads into s4..s23 (SMEM: out of order)
    A("  s_barrier")  # the stage-2 waves' one barrier (range_and_copy)
    if "x" in cknock:
        A("  s_endpgm")
    A(f"  s_load_dwordx2 s[4:5], s[0:1], {off['y']:#x}")
    A(f"  s_load_dword s6, s[0:1], {off['y_bytes']:#x}")
    A(f"  s_load_dwordx8 s[8:15], s[0:1], {off['ids']:#x}")  # ids, shared, part, sync
    A(f"  s_load_dword s16, s[0:1], {off['m']:#x}")
    A(f"  s_load_dwordx2 s[18:19], s[0:1], {off['counts']:#x}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_load_dword s17, s[18:19], 0x0")  # NG
    A("  s_lshl_b32 s29, s2, 2")
    A(f"  s_add_u32 s29, s29, {CTR_OFF}")
    A("  s_load_dword s24, s[14:15], s29")  # ctr[b]
    o += rsrc(20, 4, nbytes_sgpr=6)
    A("  v_and_b32 v1, 63, v0")
    A("  v_mov_b32 v5, 0x7fff0000")
    A("  s_lshl_b32 s35, s16, 5")  # granules per chunk
    A(f"  s_add_u32 s38, s35, {NWG - 1}")
    A(f"  s_lshr_b32 s38, s38, {NWG.bit_length() - 1}")  # per wave
    A(f"  s_lshr_b32 s39, s2, {lg}")  # index in the group
    A("  s_mul_i32 s39, s39, s38")  # first granule
    for p in range(SUB):
        y0 = YO[p]
        A(f"  v_add_u32 v2, {64 * p}, v1")
        A(f"  v_cmp_gt_u32 s[{MASK[p]}:{MASK[p] + 1}], s38, v2")
        A("  v_add_u32 v2, s39, v2")  # gi
        A("  v_cmp_gt_u32 s[46:47], s35, v2")
        A(
            f"  s_and_b64 s[{MASK[p]}:{MASK[p] + 1}], s[{MASK[p]}:{MASK[p] + 1}], s[46:47]"
        )
        A("  v_lshrrev_b32 v3, 5, v2")  # token
        A("  v_and_b32 v4, 31, v2")  # 8-column group in the chunk
        A(f"  s_mov_b64 exec, s[{MASK[p]}:{MASK[p] + 1}]")
        A(f"  s_cbranch_execz .Lcnoid{p}")
        A("  v_mul_u32_u24 v8, 24, v3")
        for k in range(6):
            A(f"  global_load_dword v{24 + k}, v8, s[8:9] offset:{4 * k}")
        A(f"  v_mul_u32_u24 v{y0 + 6}, {H * 2}, v3")
        A(
            f"  v_lshl_add_u32 v{y0 + 6}, v4, 4, v{y0 + 6}"
        )  # shared / part offset (no chunk)
        A("  s_waitcnt vmcnt(0)")
        for k in range(6):
            A("  v_mul_u32_u24 v8, 6, v3")
            A(f"  v_add_u32 v8, {k}, v8")
            A(f"  v_mul_u32_u24 v8, {H * 4}, v8")
            A("  v_lshl_add_u32 v8, v4, 5, v8")
            A(f"  v_cmp_le_i32 vcc, 0, v{24 + k}")
            A(
                f"  v_cndmask_b32 v{y0 + k}, v5, v8, vcc"
            )  # y offset of pick k (no chunk), or OOB
        A(f".Lcnoid{p}:")
        A("  s_mov_b64 exec, -1")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_add_u32 s24, s24, 1")
    A("  s_lshl_b32 s25, s24, 20")
    A("  s_mov_b32 s28, 0x100000")
    A("  s_and_b32 s24, s24, 0xfff")
    A(
        f"  v_lshlrev_b32 v6, {PSTRIDE.bit_length() - 1}, v1"
    )  # poll: CTA l + 64 k at PSTRIDE (l + 64 k)
    A(f"  s_lshl_b32 s52, s2, {PSTRIDE.bit_length() - 1}")
    A("  v_mov_b32 v9, s52")  # my CTA's word
    A(f"  v_mov_b32 v31, {PLDS}")
    A("  s_mov_b32 s53, -1")  # last published
    A("  s_or_b32 s55, s25, 0xfffff")  # "finished"
    pub_n = [0]

    def publish():
        pub_n[0] += 1
        n = pub_n[0]
        A("  ds_read_b128 v[56:59], v31")
        A("  s_waitcnt lgkmcnt(0)")
        A("  v_min3_u32 v60, v56, v57, v58")
        A("  v_min_u32 v60, v60, v59")
        A("  s_nop 1")
        A("  v_readfirstlane_b32 s54, v60")
        A("  s_cmp_eq_u32 s54, s53")
        A(f"  s_cbranch_scc1 .Lcpub{n}")
        A("  s_mov_b32 s53, s54")
        A("  global_store_dword v9, v60, s[14:15] sc1")
        A(f".Lcpub{n}:")

    agg_n = [0]

    def poll():
        """s56 = global min d (units done = all u < s56 on every CTA), 0xfffff = finished."""
        agg_n[0] += 1
        n = agg_n[0]
        publish()
        A(f"  s_cmp_lt_u32 s2, {NAGG}")
        A(f"  s_cbranch_scc0 .Lcfol{n}")
        for k in range(gtable.GRID // 64):
            A(f"  s_add_u32 s48, s14, {64 * PSTRIDE * k}")
            A("  s_addc_u32 s49, s15, 0")
            A(f"  global_load_dword v{32 + k}, v6, s[48:49] sc1")
        A("  s_waitcnt vmcnt(0)")
        for k in range(gtable.GRID // 64):
            A(f"  v_subrev_u32 v{32 + k}, s25, v{32 + k}")
            A(f"  v_cmp_gt_u32 vcc, s28, v{32 + k}")
            A(f"  v_cndmask_b32 v{32 + k}, 0, v{32 + k}, vcc")  # stale -> 0
        A("  v_min3_u32 v7, v32, v33, v34")
        A("  v_min_u32 v7, v7, v35")
        for r in (1, 2, 4, 8):  # min within each 16-lane row
            A("  s_nop 1")
            A(f"  v_min_u32_dpp v7, v7, v7 row_ror:{r} row_mask:0xf bank_mask:0xf")
        A("  s_nop 1")
        A("  v_readlane_b32 s56, v7, 0")
        for l in (16, 32, 48):  # noqa: E741
            A(f"  v_readlane_b32 s57, v7, {l}")
            A("  s_min_u32 s56, s56, s57")
        A("  s_add_u32 s57, s56, s25")
        A("  v_mov_b32 v7, s57")
        A("  s_lshl_b32 s57, s2, 7")
        A(f"  s_add_u32 s57, s57, {AGG_OFF}")
        A("  v_mov_b32 v8, s57")
        A("  global_store_dword v8, v7, s[14:15] sc1")
        if asleep > 1:
            A(f"  s_sleep {asleep}")
        A(f"  s_branch .Lcpd{n}")
        A(f".Lcfol{n}:")
        A(f"  s_and_b32 s57, s2, {NAGG - 1}")
        A("  s_lshl_b32 s57, s57, 7")
        A(f"  s_add_u32 s57, s57, {AGG_OFF}")
        A("  v_mov_b32 v8, s57")
        A("  global_load_dword v7, v8, s[14:15] sc1")
        A("  s_waitcnt vmcnt(0)")
        A("  v_readfirstlane_b32 s56, v7")
        A("  s_sub_u32 s56, s56, s25")
        A("  s_cmp_lt_u32 s56, s28")
        A("  s_cselect_b32 s56, s56, 0")  # stale -> 0
        A(f".Lcpd{n}:")

    A(f"  s_and_b32 s26, s2, {CG - 1}")  # first chunk of my group
    A(".Lcc:")
    A("  s_add_u32 s27, s26, 1")
    A("  s_mul_i32 s27, s27, s17")  # units of chunks 0..c
    A("  s_mov_b32 s34, 2000000")  # bounded: a lost word must not hang the GPU
    A(".Lcpoll:")
    poll()
    A("  s_cmp_ge_u32 s56, s27")
    A("  s_cbranch_scc1 .Lcgo")
    A(f"  s_sleep {csleep}")
    A("  s_sub_u32 s34, s34, 1")
    A("  s_cbranch_scc0 .Lcpoll")
    A("  s_or_b32 s24, s24, 0x80000000")  # timed out: ctr gets bit 31 (the host checks)
    A(".Lcgo:")
    A("  s_memrealtime s[48:49]")
    A(f"  s_lshr_b32 s50, s26, {lg}")  # my k-th chunk
    A("  s_lshl_b32 s50, s50, 2")
    A(f"  s_mul_i32 s51, s2, {4 * TL_WORDS}")
    A("  s_add_u32 s50, s50, s51")
    A(f"  s_add_u32 s50, s50, {TL_OFF + 20}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_mov_b32 v8, s50")
    A("  v_mov_b32 v7, s48")
    A("  s_mov_b64 exec, 1")
    A("  global_store_dword v8, v7, s[14:15]")
    A("  s_lshl_b32 s36, s26, 10")  # y column offset of the chunk
    A("  s_lshl_b32 s37, s26, 9")  # part / shared column offset
    if "w" in cknock:
        A("  s_branch .Lcskip_all")
    ysc = "" if "y" in cknock else " sc1"
    for p in range(SUB):
        y0, L = YO[p], LD[p]
        A(f"  s_mov_b64 exec, s[{MASK[p]}:{MASK[p] + 1}]")
        for k in range(6):
            A(
                f"  buffer_load_dwordx4 v[{L + 8 * k}:{L + 8 * k + 3}], v{y0 + k}, s[20:23], s36 offen{ysc}"
            )
            A(
                f"  buffer_load_dwordx4 v[{L + 8 * k + 4}:{L + 8 * k + 7}], v{y0 + k}, s[20:23], s36 offen offset:16{ysc}"
            )
        A(f"  v_add_u32 v8, s37, v{y0 + 6}")
        A(f"  global_load_dwordx4 v[{L + 48}:{L + 51}], v8, s[10:11]")
    A("  s_waitcnt vmcnt(0)")
    for p in range(SUB):
        y0, L = YO[p], LD[p]
        A(f"  s_mov_b64 exec, s[{MASK[p]}:{MASK[p] + 1}]")
        A(f"  s_cbranch_execz .Lcskip{p}")
        for k in range(1, 6):
            for i in range(8):
                A(f"  v_add_f32 v{L + i}, v{L + i}, v{L + 8 * k + i}")
        R, T = L + 8, L + 12  # pick 1's registers are free now
        for q in range(4):
            A(f"  v_cvt_pk_bf16_f32 v{R + q}, v{L + 2 * q}, v{L + 2 * q + 1}")
        for q in range(4):
            A(f"  v_lshlrev_b32 v{T}, 16, v{R + q}")
            A(f"  v_and_b32 v{T + 1}, 0xffff0000, v{R + q}")
            A(f"  v_lshlrev_b32 v{T + 2}, 16, v{L + 48 + q}")
            A(f"  v_and_b32 v{T + 3}, 0xffff0000, v{L + 48 + q}")
            A(f"  v_add_f32 v{T}, v{T}, v{T + 2}")
            A(f"  v_add_f32 v{T + 1}, v{T + 1}, v{T + 3}")
            A(f"  v_cvt_pk_bf16_f32 v{R + q}, v{T}, v{T + 1}")
        A(f"  v_add_u32 v8, s37, v{y0 + 6}")
        A(f"  global_store_dwordx4 v8, v[{R}:{R + 3}], s[12:13]")
        A(f".Lcskip{p}:")
    A(".Lcskip_all:")
    A("  s_mov_b64 exec, -1")
    A(f"  s_add_u32 s26, s26, {CG}")
    A(f"  s_cmp_lt_u32 s26, {NCH}")
    A("  s_cbranch_scc1 .Lcc")
    A("  s_mov_b32 s34, 2000000")
    A(".Lcdrain:")  # until my CTA's stage 2 is finished (aggregators: everyone's)
    A(f"  s_cmp_lt_u32 s2, {NAGG}")
    A("  s_cbranch_scc1 .Lcdagg")
    publish()
    A("  s_cmp_eq_u32 s54, s55")
    A("  s_cbranch_scc1 .Lcdone")
    A("  s_branch .Lcdsl")
    A(".Lcdagg:")
    poll()
    A("  s_cmp_eq_u32 s56, 0xfffff")
    A("  s_cbranch_scc1 .Lcdone")
    A(".Lcdsl:")
    A(f"  s_sleep {csleep}")
    A("  s_sub_u32 s34, s34, 1")
    A("  s_cbranch_scc0 .Lcdrain")
    A("  s_or_b32 s24, s24, 0x80000000")
    A(".Lcdone:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_mov_b64 exec, 1")
    A("  v_mov_b32 v8, s29")
    A("  v_mov_b32 v7, s24")
    A("  global_store_dword v8, v7, s[14:15]")
    A("  s_memrealtime s[48:49]")
    A(f"  s_mul_i32 s51, s2, {4 * TL_WORDS}")
    A(f"  s_add_u32 s51, s51, {TL_OFF + 40}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_mov_b32 v8, s51")
    A("  v_mov_b32 v7, s48")
    A("  global_store_dword v8, v7, s[14:15]")
    A("  s_endpgm")
    return o
