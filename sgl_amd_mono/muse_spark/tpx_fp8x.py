# SPDX-License-Identifier: Apache-2.0
"""FP8 TP-expert GEMVs for the bf16-expert layers (0 and 61), FlyDSL: ``tpx_bf16._dn`` / ``_ug`` with the FP8
slices (``fp8_rows``), bit-identical to them, without their LDS-staged pipeline.

A wave works one (pick, 16-row tile) job (Triton's program (pick, column tile) at BN = 16; a job whose pick is not
its expert's first does nothing). The B operand (16 weight rows x 32 K a MFMA) is loaded straight into the MFMA's
B layout (lane: row l % 16, K 8 (l / 16) ..) and converted by v_cvt_scalef32_pk_bf16_fp8 (exact: E4M3 codes are
bf16 values); the A rows (the tokens that picked the expert; the other rows zero) come from L2. A 128-column scale
group is ``d = 0; d = mfma(a_j, b_j, d), j = 0..3; acc = fma(d, s, acc)``, Triton's ``acc += tl.dot(a,
w.to(bf16)) * s`` (v_mfma_f32_16x16x32_bf16, K ascending, the zero-started group dot, one fused multiply-add with
the power-of-two scale), so the outputs are bitwise the Triton kernels'. The groups are unrolled with ``DEPTH``
groups of weights in flight a wave (no carried-register copy to drain them)."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common.ops import (
    CM_NT,
    CM_SYS,
    ballot,
    clock,
    fp8_to_bf16x8,
    hw_exp2,
    lane_gather,
    load_ptr64,
    rsrc,
    uniform,
)
from sgl_amd_mono.common.plan import kernel_sources, key_tuple
from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.muse_spark.gfx950 import compat
from sgl_amd_mono.muse_spark.gfx950.device import step_tag

SOURCES = kernel_sources(__name__)

BRIDGE = compat.install()
_STREAM = fx.Stream(None)
V8B = None  # set at first build (fx.Vector.make_type needs the context)
GROUP = 128  # the columns a scale covers (tpx_bf16.FP8_GROUP)
WPB = 4  # waves a CTA (a job a wave)
DEPTH = 4  # groups of weights in flight a wave (ug; dn runs with 2: tpx_bf16.tpx_moe)
WCM = CM_NT  # the weights stream non-temporal (allocating them evicts other dirty lines: write-back traffic)
UG_LDS = True  # ug's A rows from an LDS copy of x (staged once a CTA); else from L2 with the weights


def _mfma(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def _a_off(ug, kk, t, pick, K, RS, NP):
    """Dword offset of a lane's 8 A values at K index ``kk`` (x [M, K] for ug; the gathered mids [TP, NP, RS])."""
    if ug:
        return (t * K + kk) // 2
    return ((kk // RS) * (NP * RS) + pick * RS + kk % RS) // 2


def _ll_load(r_m, offs):
    """The tagged chunks at dword offsets ``offs`` (uncached: peers store them), one vector (a plain helper)."""
    words = []
    for o in offs:
        w = fx.Vector(
            bo.buffer_load(r_m, o, vec_width=4, dtype=T.i32, cache_modifier=CM_SYS)
        )
        words += [w[k] for k in range(4)]
    return fx.Vector.from_elements(words, fx.Int32)


def _ll_pending(v, hs, tag):
    """Any live chunk whose two tags are not this launch's (a plain helper)."""
    bad = hs[0] & ((v[1] != tag) | (v[3] != tag))
    for i in range(1, len(hs)):
        bad = bad | (hs[i] & ((v[4 * i + 1] != tag) | (v[4 * i + 3] != tag)))
    return bad


def _a_load(lds, ug, g, t, kq, M, K, RS, NP, xl, r_a, has, pick, dead, V8B):
    """This lane's A operands of group g (a plain helper: no traced branches): ug from the LDS copy of x or
    from L2 (rows t < M; the others zero), dn from the gathered mids [TP, NP, RS] (rows ``has``)."""
    if lds:
        live = (t < M) if ug else has
        return [
            live.select(
                fx.Vector(
                    fx.ptr_load(
                        xl + (fx.min(t, M - 1) * K + g * GROUP + 32 * j + 8 * kq),
                        result_type=V8B,
                    )
                ).bitcast(fx.Int32),
                fx.Vector.filled(4, 0, fx.Int32),
            ).bitcast(fx.BFloat16)
            for j in range(4)
        ]
    if ug:
        vo = (t < M).select((fx.min(t, M - 1) * K + 8 * kq) // 2, dead)
        return [
            fx.Vector(
                bo.buffer_load(
                    r_a,
                    vo,
                    vec_width=4,
                    dtype=T.i32,
                    soffset_bytes=(g * GROUP + 32 * j) * 2,
                )
            ).bitcast(fx.BFloat16)
            for j in range(4)
        ]
    vo = has.select(
        (pick * RS + 8 * kq) // 2, dead
    )  # (the lane part; the group part is uniform)
    return [
        fx.Vector(
            bo.buffer_load(
                r_a,
                vo,
                vec_width=4,
                dtype=T.i32,
                soffset_bytes=(
                    ((g * GROUP + 32 * j) // RS) * (NP * RS) + (g * GROUP + 32 * j) % RS
                )
                * 2,
            )
        ).bitcast(fx.BFloat16)
        for j in range(4)
    ]


def _epilogue(ug, accs, i):
    """Element i of the output: dn the accumulator; ug Triton's silu(g) u, the same fp32 operations
    (v_exp_f32 of -log2(e) g, + 1, IEEE division, * u)."""
    if not ug:
        return accs[0][i]
    g_, u_ = accs[0][i], accs[1][i]
    return g_ / (fx.Float32(1.0) + hw_exp2(fx.Float32(-1.4426950408889634) * g_)) * u_


def _job_fn(kind, M, TOPK, K, ROWS, RS, NP, depth, W):
    """The fused launch's job body (``build``'s math, bitwise): A rows from LDS (ug: x; dn: the CTA's staged mid
    rows), the output rows stored to every rank's gathered region as tagged pairs."""
    NT = ROWS // 16
    G = K // GROUP
    ug = kind == "ug"
    NH = 2 if ug else 1
    W_ROWS = NH * ROWS

    @flyc.jit
    def job_body_f(ids_p, w, s, table, out_off, e, nt, lane, xl, tag):
        V8B = fx.Vector.make_type(8, fx.BFloat16)
        r_ids = rsrc(ids_p)
        t = lane % 16
        kq = lane // 16
        has = fx.Int32(0) != fx.Int32(0)
        slot = fx.Int32(0)
        for sl in range_constexpr(TOPK):
            v = fx.Int32(
                bo.buffer_load(
                    r_ids, fx.min(t, M - 1) * TOPK + sl, vec_width=1, dtype=T.i32
                )
            )
            hit = (v == e) & (t < M)
            slot = hit.select(fx.Int32(sl), slot)
            has = has | hit
        pick = t * TOPK + slot
        n = lane % 16
        row = nt * 16 + n
        # the expert's slices as their own buffers (offsets in dwords stay small)
        r_w = rsrc(w + fx.Int64(e) * (W_ROWS * K))
        r_s = rsrc(s + fx.Int64(e) * (W_ROWS * G * 4))

        def a_load(g):
            return _a_load(
                True, ug, g, t, kq, M, K, RS, NP, xl, None, has, pick, None, V8B
            )

        def b_load(g):
            """Group g's B operands, [half][j] as two dwords: the shuffled slices (``shuffle``) give a
            lane its four MFMAs' 8 B as two 16 B loads, a wave instruction 1 KB contiguous."""
            out_ = []
            for h in range_constexpr(NH):
                tile = nt + h * (
                    ROWS // 16
                )  # (uniform: the group part goes in the scalar offset, one
                hw = []  # lane offset register serves every load)
                for hh in range_constexpr(2):
                    v = fx.Vector(
                        bo.buffer_load(
                            r_w,
                            lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=WCM,
                            soffset_bytes=((tile * G + g) * 2 + hh) * 1024,
                        )
                    )
                    hw += [[v[0], v[1]], [v[2], v[3]]]
                out_.append(hw)
            return out_

        def s_load(g4):
            return [
                fx.Vector(
                    bo.buffer_load(
                        r_s,
                        n * G,
                        vec_width=4,
                        dtype=T.f32,
                        soffset_bytes=((nt * 16 + h * ROWS) * G + 4 * g4) * 4,
                    )
                )
                for h in range(NH)
            ]

        accs = [[fx.Float32(0.0) for _ in range(4)] for _ in range(NH)]
        wq = []
        for g in range_constexpr(min(depth, G)):
            wq.append(b_load(g))
        sq = [s_load(g4) for g4 in range(min(2, G // 4))]
        rocdl.sched_barrier(
            0
        )  # (the prefetch stays where it is issued: no sinking of loads to their uses)
        for g in range_constexpr(G):
            if const_expr(g + depth < G):
                wq.append(b_load(g + depth))
            if const_expr(g % 4 == 0 and g // 4 + 2 < G // 4):
                sq.append(s_load(g // 4 + 2))
            rocdl.sched_barrier(0)
            avs = a_load(g)
            for h in range_constexpr(NH):
                d = fx.Vector.filled(4, 0.0, fx.Float32)
                for j in range_constexpr(4):
                    bw = wq[g][h][j]
                    d = _mfma(
                        avs[j], fp8_to_bf16x8(fx.Int32(bw[0]), fx.Int32(bw[1])), d
                    )
                sc = fx.Float32(sq[g // 4][h][g % 4])
                accs[h] = [
                    fmath.fma(fx.Float32(d[i]), sc, accs[h][i]) for i in range(4)
                ]
        # out: this lane holds column n, rows (tokens) 4 kq + i
        # every rank's copy of the gathered output [TP, NP, ROWS] as tagged pairs (this rank's block from pair
        # ``out_off``): a pair (data word, launch tag) is its own arrival flag
        ros = [rsrc(load_ptr64(table, q_)) for q_ in range(W)]
        hv = has.select(fx.Int32(1), fx.Int32(0))
        for i in range_constexpr(4):
            tok = 4 * kq + i
            h_i = lane_gather(hv, tok)
            p_i = lane_gather(pick, tok)
            val = _epilogue(ug, accs, i)
            # rows n, n + 1 (this lane and the next) as one bf16 pair from the even lane
            nb = lane_gather(val.bitcast(fx.Int32), lane + 1).bitcast(fx.Float32)
            word = (
                fx.Vector.from_elements([val, nb], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)
            )
            if (h_i != 0) & (n % 2 == 0):
                for q_ in range_constexpr(W):
                    bo.buffer_store(
                        fx.Vector.from_elements([fx.Int32(word[0]), tag], fx.Int32),
                        ros[q_],
                        2 * (out_off + (p_i * ROWS + row) // 2),
                        cache_modifier=CM_SYS,
                    )

    return job_body_f


@functools.lru_cache(maxsize=None)
def build_fused(
    M: int,
    TOPK: int,
    H: int,
    RS: int,
    CS: int,
    W: int,
    dug: int = 4,
    ddn: int = 2,
    stamps: bool = False,
):
    """ug + the mids all-gather + dn + the y all-gather in ONE launch over a peer-symmetric buffer (``fused_layout``),
    with no barrier across ranks or CTAs' arrivals: every hand-off is a tagged pair (a bf16 pair's data word next to
    the launch tag, 8 B in one store; the mono Mailbox's form), so the data is its own flag.
      ug: 256 co-resident CTAs of WPB waves loop over the (pick, 16-row tile) jobs (``build``'s math, bitwise; A = x
          staged in LDS once a CTA) and store their mid rows to every rank's gathered mids.
      dn: a CTA's WPB waves share their dn pick (RS // 16 tiles a pick, WPB | that): the CTA polls that expert's mid
          rows from every rank into LDS [M, I] (``stage_a``), then its waves run their dn jobs with A from there and
          store y rows to every rank's gathered y.
      y:  every CTA polls its share of the gathered y and writes it plain (torch.all_gather's layout, at
          ``lay["yout"]``) for the next launch (Triton's _fin); the launch's end is the barrier.
    Reuse across launches is safe: a peer's next-launch stores into this rank's mids / y need this rank's next launch
    started (its dn polls this rank's next mids), and this launch ends only after its last read."""
    keyed = key_tuple(locals(), SOURCES)
    NP = M * TOPK
    lay = fused_layout(M, TOPK, RS, CS, W)
    BLOCKS = 256
    NWAVES = BLOCKS * WPB
    JU = NP * (RS // 16)
    JD = NP * (CS // 16)
    assert RS * W == H  # (K of dn = I: the gathered intermediate)
    job_ug = _job_fn("ug", M, TOPK, H, RS, RS, NP, dug, W)
    job_dn = _job_fn("dn", M, TOPK, RS * W, CS, RS, NP, ddn, W)
    CH = RS // 4  # 16 B tagged chunks (4 bf16) a (rank, pick) row of the gathered mids
    TB = 8  # tokens a staging batch (4 dwordx4 a token a thread at RS 512: 128 VGPRs a batch)
    YCH = W * NP * CS // 4  # 16 B tagged chunks of the gathered y
    assert (
        (W * CH) % (WPB * 64) == 0 and (CS // 16) % WPB == 0 and M * H * 2 <= 160 * 1024
    )

    @fx.struct
    class Lds:
        a: fx.Array[
            fx.BFloat16, M * H, 16
        ]  # x [M, H] for the ug jobs, then the dn job's A rows [M, I] (I = H)

    def first_pick(r_ids, p, e, lane):
        earlier = fx.Int64(0)
        for half in range_constexpr(-(-NP // 64)):
            q_ = lane + 64 * half
            mine = fx.Int32(
                bo.buffer_load(r_ids, fx.min(q_, NP - 1), vec_width=1, dtype=T.i32)
            )
            earlier = earlier | ballot((mine == e) & (q_ < p))
        return earlier == 0

    def st(dbg, bid, tid, idx):
        if stamps:
            _stamp(dbg, bid, tid, idx)

    @flyc.kernel(
        name=f"muse_tpx_fp8_ffn_{M}_{H}_{RS}_{CS}_{W}_{dug}_{ddn}_{int(stamps)}",
        known_block_size=[WPB * 64, 1, 1],
    )
    def k(
        x: Int64,
        ids_p: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        sym: Int64,
        table: Int64,
        rank: Int32,
        ep: Int64,
        layer: Int32,
        dbg: Int64,
    ):
        tid = fx.thread_idx.x
        tag = step_tag(
            ep, layer
        )  # the mono kernels' tag: the step (epoch word 0, advanced by the host) and the layer
        lane = tid % 64
        wave = uniform(tid // 64)
        bid = fx.block_idx.x
        gw = bid * WPB + wave
        r_ids = rsrc(ids_p)
        # x [M, H] into LDS once a CTA
        xl = fx.SharedAllocator().allocate(Lds).peek().a.ptr
        r_x = rsrc(x, M * H * 2)
        NW = M * H // 8
        for q in range_constexpr(-(-NW // (WPB * 64))):
            u = fx.min(tid + q * WPB * 64, NW - 1)
            fx.ptr_store(
                fx.Vector(bo.buffer_load(r_x, 4 * u, vec_width=4, dtype=T.i32)).bitcast(
                    fx.BFloat16
                ),
                xl + 8 * u,
            )
        gpu.barrier()
        st(dbg, bid, tid, 0)
        # ---- ug jobs
        for jj in range_constexpr(-(-JU // NWAVES)):
            job = gw + jj * NWAVES
            p = fx.min(job // (RS // 16), NP - 1)
            e = uniform(fx.Int32(bo.buffer_load(r_ids, p, vec_width=1, dtype=T.i32)))
            if first_pick(r_ids, p, e, lane) & (job < JU):
                job_ug(
                    ids_p,
                    w13,
                    s13,
                    table,
                    lay["mids"] // 8 + rank * (NP * RS // 2),
                    e,
                    job % (RS // 16),
                    lane,
                    xl,
                    tag,
                )
        st(dbg, bid, tid, 1)
        # ---- dn jobs (the CTA's shared pick: its A rows polled into LDS, then the waves' jobs)
        for jj in range_constexpr(-(-JD // NWAVES)):
            gpu.barrier()  # (the LDS: x, then the previous job's A rows, are dead)
            job = gw + jj * NWAVES
            p = fx.min(job // (CS // 16), NP - 1)
            e = uniform(fx.Int32(bo.buffer_load(r_ids, p, vec_width=1, dtype=T.i32)))
            go = first_pick(r_ids, p, e, lane) & (job < JD)
            if go:
                stage_a(sym, ids_p, e, tag, xl, tid)
            gpu.barrier()
            st(dbg, bid, tid, 2 + min(jj, 1))
            if go:
                job_dn(
                    ids_p,
                    w2,
                    s2,
                    table,
                    lay["ys"] // 8 + rank * (NP * CS // 2),
                    e,
                    job % (CS // 16),
                    lane,
                    xl,
                    tag,
                )
        st(dbg, bid, tid, 4)
        # ---- the gathered y, plain, for _fin
        unpack_y(sym, tag, bid * (WPB * 64) + tid)
        st(dbg, bid, tid, 5)

    @flyc.jit
    def stage_a(sym, ids_p, e, tag, xl, tid):
        """The A rows of expert ``e``'s dn jobs, [M, I] (row t: token t's pick of e, from every rank's mids;
        rows of tokens without e are not written: the jobs mask them), polled until their tags are this launch's."""
        r_m = rsrc(sym + fx.Int64(lay["mids"]), W * NP * RS * 4)
        r_i = rsrc(ids_p)
        DEAD = 0x1FFFFFF0  # dwords: past the buffer (no traffic, zeros)
        for t0 in range_constexpr(0, M, TB):
            offs, hs, dst = [], [], []
            for t in range_constexpr(t0, min(M, t0 + TB)):
                slot = fx.Int32(0)
                has = fx.Int32(0) != fx.Int32(0)
                for sl in range_constexpr(TOPK):
                    v = uniform(
                        fx.Int32(
                            bo.buffer_load(r_i, t * TOPK + sl, vec_width=1, dtype=T.i32)
                        )
                    )
                    hit = v == e
                    slot = hit.select(fx.Int32(sl), slot)
                    has = has | hit
                pick = t * TOPK + slot
                for q in range_constexpr(W * CH // (WPB * 64)):
                    u = tid + q * WPB * 64
                    r = u // CH
                    c = u % CH
                    offs.append(
                        has.select(
                            ((r * NP + pick) * (RS // 2) + 2 * c) * 2, fx.Int32(DEAD)
                        )
                    )
                    hs.append(has)
                    dst.append(t * (W * RS) + r * RS + 4 * c)
            v = _spin(lambda: _ll_load(r_m, offs), lambda v_: _ll_pending(v_, hs, tag))
            for i in range_constexpr(len(offs)):
                fx.ptr_store(
                    fx.Vector.from_elements(
                        [fx.Int32(v[4 * i]), fx.Int32(v[4 * i + 2])], fx.Int32
                    ).bitcast(fx.BFloat16),
                    xl + dst[i],
                )

    @flyc.jit
    def unpack_y(sym, tag, gt):
        """Thread ``gt``'s share of the gathered y: tagged chunks polled, data words stored plain."""
        r_y = rsrc(sym + fx.Int64(lay["ys"]), W * NP * CS * 4)
        r_c = rsrc(sym + fx.Int64(lay["yout"]), W * NP * CS * 2)
        PER = -(-YCH // (BLOCKS * WPB * 64))
        offs, hs = [], []
        for q in range_constexpr(PER):
            u = gt + q * (BLOCKS * WPB * 64)
            live = u < YCH
            offs.append(live.select(u * 4, fx.Int32(0x1FFFFFF0)))
            hs.append(live)
        v = _spin(lambda: _ll_load(r_y, offs), lambda v_: _ll_pending(v_, hs, tag))
        for q in range_constexpr(PER):
            u = gt + q * (BLOCKS * WPB * 64)
            if hs[q]:
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [fx.Int32(v[4 * q]), fx.Int32(v[4 * q + 2])], fx.Int32
                    ),
                    r_c,
                    2 * u,
                )

    @flyc.jit
    def launch(
        x: Int64,
        ids_p: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        sym: Int64,
        table: Int64,
        rank: Int32,
        ep: Int64,
        layer: Int32,
        dbg: Int64 = 0,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k(x, ids_p, w13, s13, w2, s2, sym, table, rank, ep, layer, dbg).launch(
            grid=(BLOCKS,), block=(WPB * 64,), stream=stream
        )

    return launch, lay


@flyc.jit
def _stamp(dbg, bid, tid, idx):
    if tid == 0:
        t = clock()
        bo.buffer_store(
            fx.Vector.from_elements(
                [fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32
            ),
            rsrc(dbg),
            (bid * 8 + idx) * 2,
        )


def fused_layout(M: int, TOPK: int, RS: int, CS: int, W: int) -> dict:
    """Byte offsets in the peer-symmetric buffer of one fused launch: the gathered mids [W, NP, RS] and y [W, NP, CS]
    as tagged pairs (8 B a bf16 pair), then the plain gathered y (torch.all_gather's layout) the launch leaves."""
    np_ = M * TOPK
    mids = 0
    ys = mids + W * np_ * RS * 4
    yout = ys + W * np_ * CS * 4
    return {"mids": mids, "ys": ys, "yout": yout, "bytes": yout + W * np_ * CS * 2}


@functools.lru_cache(maxsize=None)
def build(
    kind: str,
    M: int,
    TOPK: int,
    K: int,
    ROWS: int,
    RS: int,
    NP: int,
    depth: int = DEPTH,
):
    """``kind`` "dn": y[pick, n] = bf16(sum_k mids[pick, k] w2[e, n, k]), K = I, the mids gathered [TP, NP, RS];
    "ug": mid[pick, n] = bf16(silu(g) u), g / u over w13's gate row n / up row ROWS + n, K = H, A = x [M, H].
    ROWS = the output rows a rank (CS for dn, RS for ug)."""
    keyed = key_tuple(locals(), SOURCES)
    NT = ROWS // 16
    JOBS = NP * NT
    BLOCKS = -(-JOBS // WPB)
    G = K // GROUP
    ug = kind == "ug"
    NH = 2 if ug else 1
    W_ROWS = NH * ROWS

    @flyc.jit
    def job_body(xa, ids_p, w, s, out, xa_bytes, e, nt, lane, xl):
        V8B = fx.Vector.make_type(8, fx.BFloat16)
        r_ids = rsrc(ids_p)
        t = lane % 16
        kq = lane // 16
        has = fx.Int32(0) != fx.Int32(0)
        slot = fx.Int32(0)
        for sl in range_constexpr(TOPK):
            v = fx.Int32(
                bo.buffer_load(
                    r_ids, fx.min(t, M - 1) * TOPK + sl, vec_width=1, dtype=T.i32
                )
            )
            hit = (v == e) & (t < M)
            slot = hit.select(fx.Int32(sl), slot)
            has = has | hit
        pick = t * TOPK + slot
        n = lane % 16
        row = nt * 16 + n
        # the expert's slices as their own buffers (offsets in dwords stay small)
        r_w = rsrc(w + fx.Int64(e) * (W_ROWS * K))
        r_s = rsrc(s + fx.Int64(e) * (W_ROWS * G * 4))
        r_a = rsrc(xa, xa_bytes)
        dead = fx.Int32(0x1FFFFFF0)  # dwords: past every A buffer (zeros, no traffic)

        def a_dw(g, j):
            return has.select(
                _a_off(ug, g * GROUP + 32 * j + 8 * kq, t, pick, K, RS, NP), dead
            )

        def a_load(g):
            return _a_load(
                ug and UG_LDS, ug, g, t, kq, M, K, RS, NP, xl, r_a, has, pick, dead, V8B
            )

        def b_load(g):
            """Group g's B operands, [half][j] as two dwords: the shuffled slices (``shuffle``) give a
            lane its four MFMAs' 8 B as two 16 B loads, a wave instruction 1 KB contiguous."""
            out_ = []
            for h in range_constexpr(NH):
                tile = nt + h * (
                    ROWS // 16
                )  # (uniform: the group part goes in the scalar offset, one
                hw = []  # lane offset register serves every load)
                for hh in range_constexpr(2):
                    v = fx.Vector(
                        bo.buffer_load(
                            r_w,
                            lane * 4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=WCM,
                            soffset_bytes=((tile * G + g) * 2 + hh) * 1024,
                        )
                    )
                    hw += [[v[0], v[1]], [v[2], v[3]]]
                out_.append(hw)
            return out_

        def s_load(g4):
            return [
                fx.Vector(
                    bo.buffer_load(
                        r_s,
                        n * G,
                        vec_width=4,
                        dtype=T.f32,
                        soffset_bytes=((nt * 16 + h * ROWS) * G + 4 * g4) * 4,
                    )
                )
                for h in range(NH)
            ]

        accs = [[fx.Float32(0.0) for _ in range(4)] for _ in range(NH)]
        wq = []
        aq = []
        for g in range_constexpr(min(depth, G)):
            wq.append(b_load(g))
            aq.append(None if (ug and UG_LDS) else a_load(g))
        sq = [s_load(g4) for g4 in range(min(2, G // 4))]
        rocdl.sched_barrier(
            0
        )  # (the prefetch stays where it is issued: no sinking of loads to their uses)
        for g in range_constexpr(G):
            if const_expr(g + depth < G):
                wq.append(b_load(g + depth))
                aq.append(None if (ug and UG_LDS) else a_load(g + depth))
            if const_expr(g % 4 == 0 and g // 4 + 2 < G // 4):
                sq.append(s_load(g // 4 + 2))
            rocdl.sched_barrier(0)
            avs = a_load(g) if (ug and UG_LDS) else aq[g]
            for h in range_constexpr(NH):
                d = fx.Vector.filled(4, 0.0, fx.Float32)
                for j in range_constexpr(4):
                    bw = wq[g][h][j]
                    d = _mfma(
                        avs[j], fp8_to_bf16x8(fx.Int32(bw[0]), fx.Int32(bw[1])), d
                    )
                sc = fx.Float32(sq[g // 4][h][g % 4])
                accs[h] = [
                    fmath.fma(fx.Float32(d[i]), sc, accs[h][i]) for i in range(4)
                ]
        # out: this lane holds column n, rows (tokens) 4 kq + i
        ro = rsrc(out)
        hv = has.select(fx.Int32(1), fx.Int32(0))
        for i in range_constexpr(4):
            tok = 4 * kq + i
            h_i = lane_gather(hv, tok)
            p_i = lane_gather(pick, tok)
            val = _epilogue(ug, accs, i)
            # rows n, n + 1 (this lane and the next) as one bf16 pair from the even lane
            nb = lane_gather(val.bitcast(fx.Int32), lane + 1).bitcast(fx.Float32)
            word = (
                fx.Vector.from_elements([val, nb], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)
            )
            if (h_i != 0) & (n % 2 == 0):
                bo.buffer_store(fx.Int32(word[0]), ro, (p_i * ROWS + row) // 2)

    @fx.struct
    class Lds:
        x: fx.Array[fx.BFloat16, M * K if (ug and UG_LDS) else 8, 16]

    @flyc.kernel(
        name=f"muse_tpx_fp8x_{kind}_{M}_{K}_{ROWS}_{depth}",
        known_block_size=[WPB * 64, 1, 1],
    )
    def k(xa: Int64, ids_p: Int64, w: Int64, s: Int64, out: Int64, xa_bytes: Int32):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = uniform(tid // 64)
        job = fx.block_idx.x * WPB + wave
        p = fx.min(job // NT, NP - 1)
        nt = job % NT
        r_ids = rsrc(ids_p)
        e = uniform(fx.Int32(bo.buffer_load(r_ids, p, vec_width=1, dtype=T.i32)))
        earlier = fx.Int64(0)
        for half in range_constexpr(-(-NP // 64)):
            # picks 64 half + lane before p that chose e (the first pick works the expert)
            q_ = lane + 64 * half
            mine = fx.Int32(
                bo.buffer_load(r_ids, fx.min(q_, NP - 1), vec_width=1, dtype=T.i32)
            )
            earlier = earlier | ballot((mine == e) & (q_ < p))
        xl = None
        if const_expr(ug and UG_LDS):
            # x [M, K] into LDS once a CTA (every job's A rows; ds reads do not queue behind weights)
            xl = fx.SharedAllocator().allocate(Lds).peek().x.ptr
            r_x = rsrc(xa, xa_bytes)
            NW = M * K // 8  # 16 B units
            for q in range_constexpr(-(-NW // (WPB * 64))):
                u = fx.min(tid + q * WPB * 64, NW - 1)
                v = fx.Vector(bo.buffer_load(r_x, 4 * u, vec_width=4, dtype=T.i32))
                fx.ptr_store(v.bitcast(fx.BFloat16), xl + 8 * u)
            gpu.barrier()
        if (earlier == 0) & (job < JOBS):
            job_body(xa, ids_p, w, s, out, xa_bytes, e, nt, lane, xl)

    @flyc.jit
    def launch(
        xa: Int64,
        ids_p: Int64,
        w: Int64,
        s: Int64,
        out: Int64,
        xa_bytes: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k(xa, ids_p, w, s, out, xa_bytes).launch(
            grid=(BLOCKS,), block=(WPB * 64,), stream=stream
        )

    return launch


def shuffle(q):
    """FP8 slices ``q`` [E, R, K] (R % 16 == 0) re-laid out for the kernels' B loads: per 16-row tile and
    128-column group, [half 2][lane 64][16 B] with lane = 16 (k / 8 % 4) + row % 16 and its 16 B = the 8 B of
    MFMAs 2 half and 2 half + 1 (k = 128 g + 32 j + 8 (lane / 16) ..). A permutation: the same codes."""
    e, r, k = q.shape
    v = q.view(
        e, r // 16, 16, k // GROUP, 2, 2, 4, 8
    )  # (E, tile, n, g, half, jj, kq, b)
    return v.permute(0, 1, 3, 4, 6, 2, 5, 7).contiguous().view(e, r, k)


def dn(mids, ids, w2, s2, y, depth: int = DEPTH):
    """``tpx_bf16._dn`` (FP8) into ``y`` [NP, CS]: ``mids`` the gathered [TP * NP, RS] bf16."""
    import torch

    np_, cs = y.shape
    rs = mids.shape[1]
    m, topk = ids.shape
    k_ = w2.shape[2]
    L = build("dn", m, topk, k_, cs, rs, np_, depth)
    L(
        mids.data_ptr(),
        ids.data_ptr(),
        w2.data_ptr(),
        s2.data_ptr(),
        y.data_ptr(),
        mids.numel() * 2,
        stream=torch.cuda.current_stream(),
    )


def ug(x, ids, w13, s13, mid, depth: int = DEPTH):
    """``tpx_bf16._ug`` (FP8) into ``mid`` [NP, RS]: ``x`` [M, H] bf16."""
    import torch

    np_, rs = mid.shape
    m, topk = ids.shape
    k_ = x.shape[1]
    L = build("ug", m, topk, k_, rs, rs, np_, depth)
    L(
        x.data_ptr(),
        ids.data_ptr(),
        w13.data_ptr(),
        s13.data_ptr(),
        mid.data_ptr(),
        x.numel() * 2,
        stream=torch.cuda.current_stream(),
    )
