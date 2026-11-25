# SPDX-License-Identifier: Apache-2.0
"""K1: one Muse Spark layer's attention front, the normed input through the KV
cache write, for S = ``tokens`` rows (a token each; 8 consecutive rows a request)::

    x (bf16: the layer's attention input, the previous boundary's pre-norm)
      -> q | k | v | gate GEMV on the bf16 weight (fp32 MFMA accumulate, one
         bf16 rounding, as the split-K GEMM of ``qkv_proj_qk_norm_rope_cache``)
      -> q, k: per-head RMSNorm without weight (eps 1e-5, bf16); q then scaled by
         qk_scale_factor and rounded; then interleaved RoPE in the layers that
         have one, with bf16 cos / sin and each product rounded, as the fused
         kernel of the ROCm model does
      -> q, k, v, gate out (bf16); K / V -> the layer's page-16 NHD fp8 cache
         (unit KV scale, checked at bind; the sliding layers' slots through the
         SWA pool's full-to-SWA table)

Stage 1 is the GEMV tasks: 16 row groups of 144 rows (nine MFMA tiles) x 16 K
parts = 256 tasks, exactly one a CTA; a task streams 147 KB of weights (held in
registers over the token tiles, one token tile at a time) and S x 1 KB of x (its K
sixteenth of every token, straight from memory), its fp32 partials go to slot
``part`` of the ``k1_part`` region (a zero-sentinel word a (part, token, row), from
the waves that run no unit: ``put_quads``). Stage 2 is the
units, a (q / k / v head or gate row group, token) pair each, spread over every CTA
a wave each (``stage_units``): a dim a lane, the row's sixteen slots summed in order
and rounded once. Dependencies only point from stage 2 to stage 1 and all CTAs are
co-resident, so every wait makes progress.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
    update_dpp_i32,
    xshfl,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.common.sync import MASK_OOB, _spin_masked, sent_word
from sgl_amd_mono.muse_spark.gfx950.config import (
    G_OFF,
    HEAD_DIM,
    HIDDEN,
    KV_K,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
)
from sgl_amd_mono.muse_spark.gfx950.device import (
    poll_flags,
    stamp,
    udiv,
    umod,
)
from sgl_amd_mono.muse_spark.gfx950.layout import (
    ELEMS,
    G_ROWS,
    G_TILES,
    GATE_TASKS,
    GEMV_TASKS,
    HEAD_TASKS,
    K_PART,
    K_PARTS,
    K_PER_WAVE,
    K_STEP,
    K_STEPS,
    PARTS,
    QT0,
    RED_FLOATS,
    ROWS,
    SECOND_TASKS,
    TILE,
    token_tiles,
)

# A GEMV task's (row tile, token tile) accumulators in LDS, skewed against bank
# conflicts: a lane's 4 floats at wave RED_WAVE + 4 lane + 4 (lane / 16) of the tile's
# RED_PITCH floats (RED_PITCH = 16 mod 64). The partials' row sums read 64 consecutive
# rows of a token a wave: unskewed those hit 4 banks (16-way conflicts, ~0.4 us a
# round of 8 reads at 512 threads)
RED_WAVE = 64 * 4 + 16  # a wave's 64 lanes x 4 floats, + the skew's 12
RED_PITCH = WAVES * RED_WAVE + 16
assert RED_PITCH % 64 == 16


def red_at(it, wave, lane):
    """Float offset of lane ``lane`` of wave ``wave``'s accumulators of row tile ``it``."""
    return it * RED_PITCH + wave * RED_WAVE + lane * 4 + 4 * (lane // 16)


def gemv_k(lane, wave, kpart, kc):
    """The k of step ``kc`` this lane streams, in K part ``kpart`` (each wave a
    contiguous K_PER_WAVE range of the part)."""
    return kpart * K_PART + wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)


def gemv_w_loads(c, rg, kpart, w):
    """GEMV task (``rg``, ``kpart``)'s weight loads, this lane's: row tile ``it``'s
    row l16, K_STEPS steps of 16 B, in flight (72 VGPRs). Row-major bf16, read once
    a layer (nontemporal). ``[tile][step]``. A row group past the last reads past
    the buffer: zeros, no traffic (a CTA without the task)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(w, QKVG_ROWS * HIDDEN * 2)
    return [
        [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (
                        (rg * G_ROWS + it * ROWS + lane % 16) * HIDDEN
                        + gemv_k(lane, wave, kpart, kc)
                    )
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
            for kc in range(K_STEPS)
        ]
        for it in range(G_TILES)
    ]


@flyc.jit
def gemv_w_loads_if(c, task, w, live):
    """``gemv_w_loads`` of GEMV task ``task`` on the waves where ``live`` holds; zeros,
    and no loads at all, elsewhere (loads past the buffer still occupy the wave's
    in-order return queue: a later poll on the wave would wait for them)."""
    got = [
        [fx.Vector.filled(4, 0, fx.Int32) for _ in range(K_STEPS)]
        for _ in range(G_TILES)
    ]
    if live:
        got = gemv_w_loads(c, task // K_PARTS, task % K_PARTS, w)
    return got


# The partials go out from the waves that run no head / gate unit (waves
# unit_waves(S) ..): a unit wave's partial polls would return behind its own
# partial stores' acks (a wave's memory operations complete in order). A lane sums
# four consecutive rows of a token (16-byte LDS reads, in wave order) and stores
# them as one 16-byte word quad (``put_store``); the unit waves start polling once
# the CTA's partials are issued (polls issued while the partials are still being
# stored slow the hand-off down by ~0.7 us).


# The K parts' partials are zero-sentinel words, not tagged pairs: a partial's fp32
# bits stored complemented (``zs_enc``: never 0), so an unwritten word reads 0; the
# scratch starts zeroed and each word's one reader (a unit lane) re-arms it (0) in
# the launch that wrote it. Half a tagged pair's bytes for the puts and the polls.
# Word (part, token, row) at (part MAX_TOKENS + token) QKVG_ROWS + row (the first
# half of the ``k1_part`` region).
PUT_ROWS = 4  # rows a lane sums and stores (one 16-byte store)
QUADS = G_ROWS // PUT_ROWS  # row sets a token


def zs_enc(v):
    """The zero-sentinel word of fp32 ``v`` (all ones -> another NaN first)."""
    return sent_word(v.bitcast(fx.Int32)) ^ fx.Int32(-1)


def part_word(kpart, tk, row):
    return (kpart * MAX_TOKENS + tk) * QKVG_ROWS + row


def part_put(c, kpart, tk, row, v):
    """One partial's zero-sentinel word (``stage_gemv_deferred``)."""
    bo.buffer_store(
        zs_enc(v),
        rsrc(c["part"].value),
        part_word(kpart, tk, row),
        cache_modifier=CM_DEV,
    )


def put_quad(c, rg, kpart, t, qi):
    """Row set ``qi`` (set r4 of PUT_ROWS rows, token tl of tile ``t``): rows
    PUT_ROWS r4 .. of token tl summed over the waves in order (in wave order)
    -> the parts' words (one 16-byte store)."""
    red, part = c["red"], c["part"]
    VF = fx.Vector.make_type(PUT_ROWS, fx.Float32)
    r4 = umod(qi, QUADS)
    tl = udiv(qi, QUADS)
    r0 = r4 * PUT_ROWS
    # red_at(r0 / 16, w, 16 ((r0 % 16) / 4) + tl) + r0 % 4: the set's floats
    base = (
        udiv(r0, ROWS) * RED_PITCH + udiv(umod(r0, ROWS), 4) * 68 + tl * 4 + umod(r0, 4)
    )
    tot = [fx.Float32(0.0)] * PUT_ROWS
    for w in range(WAVES):
        v = fx.Vector(fx.ptr_load(red + (base + w * RED_WAVE), result_type=VF))
        tot = [tot[i] + fx.Float32(v[i]) for i in range(PUT_ROWS)]
    pi = part_word(kpart, t * TILE + tl, rg * G_ROWS + r0)
    return pi, tot


def put_store(c, pi, tot):
    """A row set's partials (``put_quad``): four zero-sentinel words at word ``pi``
    (past the buffer: dropped)."""
    bo.buffer_store(
        fx.Vector.from_elements([zs_enc(v) for v in tot], fx.Int32),
        rsrc(c["part"].value, MASK_OOB),
        pi,
        cache_modifier=CM_DEV,
    )


@flyc.jit
def put_quads(c, rg, kpart, t, nt):
    """Token tile ``t`` (``nt`` tokens)'s partials of GEMV task (``rg``, ``kpart``)
    from the put waves (``put_quad`` a lane a round; straight-line: a partial round's
    surplus lanes re-read the last set and store past the buffer, dropped)."""
    tid, wave = c["tid"], c["wave"]
    uw = unit_waves(c["S"])
    uw = (
        uw if uw <= WAVES - 2 else 0
    )  # (units on (nearly) every wave: puts on every wave)
    P = THREADS - 64 * uw
    n_q = QUADS * nt
    rounds = -(-n_q // P)
    if wave >= uw:
        got = []
        for p in range_constexpr(rounds):
            qi = tid - 64 * uw + p * P
            if const_expr((p + 1) * P > n_q):
                pi, tot = put_quad(c, rg, kpart, t, fx.min(qi, fx.Int32(n_q - 1)))
                pi = (qi < n_q).select(pi, fx.Int32(MASK_OOB // 4))
            else:
                pi, tot = put_quad(c, rg, kpart, t, qi)
            got.append((pi, tot))
        for p in range_constexpr(rounds):
            put_store(c, got[p][0], got[p][1])
        stamp(
            c, 6, 64 * uw
        )  # (trace) the partials issued (a wave whose lanes all store)


@flyc.jit
def stage_gemv(c, rg, kpart, first=None):
    """GEMV task: K part ``kpart`` of row group ``rg`` (64 rows, four MFMA tiles) for
    every token tile. Lane l of wave w holds row l % 16 of each tile and, per
    k-step, k = 1024 kpart + 128 w + 32 kc + 8 (l / 16) .. + 8 of the weight (A) and
    of token 16 t + l % 16 (B, 16 B straight from memory); the waves' accumulators
    of each (row tile, token tile) are summed in wave order through LDS and the
    part's fp32 partials go to slot ``kpart`` of the mailbox. ``first``: the task's
    weights, already in flight (``gemv_w_loads``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    x_cm = c["x_cm"]
    if const_expr(first is not None):
        w = first
    else:
        w = gemv_w_loads(c, rg, kpart, a["w"])
    if const_expr(c.get("defer", False)):
        stage_gemv_deferred(c, rg, kpart, w)
        return
    # a token tile at a time: the weights stay in registers, the accumulators are a
    # tile's nine; the waves' accumulators sum through a reduction row set a row tile
    for t in range_constexpr(tiles):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        xb = [
            fx.Vector(
                bo.buffer_load(
                    r_x,
                    (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=x_cm,
                )  # fmt: skip
            )
            for kc in range(K_STEPS)
        ]
        for it in range_constexpr(G_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(
                    w[it][kc].bitcast(fx.BFloat16), xb[kc].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + red_at(it, wave, lane))
        gpu.barrier()
        if const_expr(t == 0):
            stamp(c, 5)  # (trace) the first tile's MFMAs done (x landed)
        nt = min(TILE, S - t * TILE)
        put_quads(c, rg, kpart, t, nt)
        gpu.barrier()  # ``red`` is reused; the unit waves poll past the partials


@flyc.jit
def stage_gemv_deferred(c, rg, kpart, w):
    """The GEMV task of a merged launch's K1 with the boundary's norms deferred: x =
    bf16(rstd2 npw (alpha st + beta bf16(pp rstd))), pp = bf16(u pfw), is rstd2 (c1 +
    rstd c2) with c1 = bf16(npw alpha st) (written by K2's boundary: ``x``) and c2 =
    bf16(npw beta pp) from the unpooled rows u of this part's columns (one rank's,
    polled from K3's ``unp`` under K3's tag); the c1 half's MFMAs and row sums run
    first (while u lands), then c2's; the token's two scalars (K3's boundary,
    ``nscal``) fold the halves into the partial: so the K1 waits for neither the
    boundary's x nor its flag."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    d_poll, d_unp, d_ns = c["d_poll"], c["d_unp"], c["d_nscal"]
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    r_fw, r_be, r_nw = rsrc(a["pfw"]), rsrc(a["beta2"]), rsrc(a["npw"])
    V8F = fx.Vector.make_type(8, fx.Float32)
    # this lane's per-k constants (8 values a step)
    fwv, bev, nwv = [], [], []
    for kc in range_constexpr(K_STEPS):
        k = gemv_k(lane, wave, kpart, kc)
        fwv.append(
            fx.Vector(bo.buffer_load(r_fw, k // 2, vec_width=4, dtype=T.i32))
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        bev.append(
            [
                fx.Vector(bo.buffer_load(r_be, k + 4 * h, vec_width=4, dtype=T.f32))
                for h in range(2)
            ]
        )
        nwv.append(
            fx.Vector(bo.buffer_load(r_nw, k // 2, vec_width=4, dtype=T.i32))
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
    for t in range_constexpr(tiles):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        nt = min(TILE, S - t * TILE)
        n_out = G_ROWS * nt
        rounds = -(-n_out // THREADS)
        # ---- c1: the first half's MFMAs and row sums
        xb = [
            fx.Vector(
                bo.buffer_load(
                    r_x,
                    (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )  # fmt: skip
            for kc in range(K_STEPS)
        ]
        for it in range_constexpr(G_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(
                    w[it][kc].bitcast(fx.BFloat16), xb[kc].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        r1 = []
        for p in range_constexpr(rounds):
            idx = fx.min(tid + p * THREADS, n_out - 1)
            r1.append(
                row_sum(
                    red + ((idx % G_ROWS) // ROWS) * RED_FLOATS,
                    (idx % G_ROWS) % ROWS,
                    idx // G_ROWS,
                )
            )
        gpu.barrier()
        # ---- c2 from the unpooled rows of this lane's token and steps
        # (two pairs a spec: a poll load is at most four dwords)
        got = d_poll(
            [
                (d_unp, (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2 + 2 * hf, 2)
                for kc in range(K_STEPS)
                for hf in range(2)
            ]
        )
        ub = []
        for kc in range_constexpr(K_STEPS):
            words = []
            for q in range_constexpr(4):
                lo, hi = bf2_f32(got[2 * kc + q // 2][q % 2])
                vals = []
                for h, u in ((0, lo), (1, hi)):
                    j = 2 * q + h
                    pp = bf16_round(u * fx.Float32(fwv[kc][j]))
                    vals.append(
                        fx.Float32(nwv[kc][j]) * fx.Float32(bev[kc][j // 4][j % 4]) * pp
                    )
                words.append(bf16_pair(vals[0], vals[1]).bitcast(fx.Int32))
            ub.append(fx.Vector.from_elements(words, fx.Int32))
        for it in range_constexpr(G_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(
                    w[it][kc].bitcast(fx.BFloat16), ub[kc].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        # ---- the token's scalars, the fold, the partials
        for p in range_constexpr(rounds):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % G_ROWS
                tk = t * TILE + idx // G_ROWS
                r2 = row_sum(red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // G_ROWS)
                sc = d_poll([(d_ns, 2 * tk, 2)])
                rstd, rstd2 = sc[0][0].bitcast(fx.Float32), sc[0][1].bitcast(fx.Float32)
                part_put(c, kpart, tk, rg * G_ROWS + r, rstd2 * (r1[p] + rstd * r2))
        gpu.barrier()  # ``red`` is reused by the next token tile / task


# The head / gate stage, spread over every CTA. A unit is one (task, token) pair:
# task j < HEAD_TASKS a q, k or v head, the rest the gate's 64-row groups (task j's
# rows are j HEAD_DIM .. of the projection: q | k | v | gate). Each unit polls its
# rows' K-part partials (16 slots x 64 rows x 4 B = 4 KB a token): on 36 CTAs alone a
# CTA polled S x 8 KB (128 KB at S = 16) through one CU every round, which took longer
# than the hop itself. A unit is a wave, lane d its row's dim d; its re-polls load only
# the words still pending (``poll_part_zs``), and it re-arms them. Unit u runs on wave u / BLOCKS
# (mod WAVES: a pass) of CTA (u + UNIT_ROT) % BLOCKS.
UNIT_TASKS = HEAD_TASKS + GATE_TASKS  # 36
UNIT_ROT = 64  # the first CTA of the units' wrap (a second unit skips the first CTAs)
assert HEAD_TASKS - LOCAL_Q_HEADS == 2 * LOCAL_KV_HEADS
assert G_OFF == HEAD_TASKS * HEAD_DIM and HEAD_DIM == 64 and ELEMS == 4


def unit_waves(s: int) -> int:
    """Waves (units) a CTA runs at width ``s``, over its passes."""
    return -(-(UNIT_TASKS * s) // BLOCKS)


def unit_of(c, ps):
    """This wave's unit of pass ``ps``: (live, task, token)."""
    S, bid, wave = c["S"], c["bid"], c["wave"]
    u = (ps * WAVES + wave) * BLOCKS + (bid + (BLOCKS - UNIT_ROT)) % BLOCKS
    live = ((ps * WAVES + wave) < unit_waves(S)) & (u < UNIT_TASKS * S)
    return live, u % UNIT_TASKS, fx.min(u // UNIT_TASKS, S - 1)


@flyc.jit
def units_pre_if(c, live):
    """Every pass's ``unit_pre`` words on the waves where ``live`` holds (zeros, no
    loads, elsewhere)."""
    got = [(fx.Int32(0), fx.Int32(0)) for _ in range(-(-unit_waves(c["S"]) // WAVES))]
    if live:
        got = [
            unit_pre(c, unit_of(c, ps)[2])
            for ps in range(-(-unit_waves(c["S"]) // WAVES))
        ]
    return got


def unit_pre(c, tk):
    """Unit token ``tk``'s position and cache slot, loaded at K1's entry so the RoPE
    loads that need them issue at once later."""
    a = c["args"]
    pos = fx.Int32(
        bo.buffer_load(rsrc(a["positions"]), 2 * tk, vec_width=1, dtype=T.i32)
    )
    slot = fx.Int32(
        bo.buffer_load(
            rsrc(a["slots"]),
            (a["slot64"] != 0).select(2 * tk, tk),
            vec_width=1,
            dtype=T.i32,
        )
    )
    return pos, slot


def poll_part_zs(base, offs, gap=1):
    """Zero-sentinel words ``offs`` of the parts (device scope) once none is 0,
    re-polling only the words still pending; their fp32 bits (decoded)."""
    r = rsrc(base, MASK_OOB)
    N = len(offs)

    def load(prev):
        words = []
        for e in range_constexpr(N):
            off = fx.Int32(offs[e])
            if const_expr(prev is not None):
                d = prev[e] != fx.Int32(0)
                off = d.select(fx.Int32(MASK_OOB // 4), off)
            w = fx.Int32(
                bo.buffer_load(r, off, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)
            )
            words.append(w if prev is None else d.select(prev[e], w))
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[0] == fx.Int32(0)
        for e in range_constexpr(1, N):
            bad = bad | (v[e] == fx.Int32(0))
        return bad

    v = _spin_masked(load, pending, gap)
    return [v[e] ^ fx.Int32(-1) for e in range(N)]


@flyc.jit
def stage_unit(c, j, tk, pos, slot):
    """Unit (task ``j``, token ``tk``) on this wave, lane d = dim d: the token's row j
    HEAD_DIM + d from the parts' fp32 partials, summed in slot order and rounded once;
    q and k: the per-head RMSNorm without weight (q then scaled by qk_scale_factor and
    rounded) and interleaved RoPE in the layers that have one; k and v: the fp8 cache
    write; the gate rows as they are. The math of the former 16-lane head / gate tasks
    (4 dims a lane), a dim a lane: the norm's sum of squares runs over each 4 dims in
    order, then the same butterfly over the 16 groups; a RoPE pair is two neighbour
    lanes."""
    d = c["lane"]
    part = c["part"]
    a = c["args"]
    is_q = j < LOCAL_Q_HEADS
    is_k = (j >= LOCAL_Q_HEADS) & (j < LOCAL_Q_HEADS + LOCAL_KV_HEADS)
    is_g = j >= HEAD_TASKS
    kv_head = is_k.select(j - LOCAL_Q_HEADS, j - LOCAL_Q_HEADS - LOCAL_KV_HEADS)
    normed = is_q | is_k
    use_rope, use_swa = a["use_rope"], a["use_swa"]
    r_cs = rsrc(a["cos_sin"])
    KV_ROWS, SWA_ROWS = (
        c["kv_rows"],
        c.get("swa_rows", 0),
    )  # head-major pools' rows (0: NHD)
    if (use_swa != 0) & (slot >= 0):
        slot = fx.Int32(
            bo.buffer_load(
                rsrc(a["swa_map"]),
                (a["swa64"] != 0).select(2 * slot, slot),
                vec_width=1,
                dtype=T.i32,
            )
        )
    # this lane's RoPE pair d / 2: cos, sin
    cs = fx.Float32(
        bo.buffer_load(r_cs, pos * HEAD_DIM + d // 2, vec_width=1, dtype=T.f32)
    )
    sn = fx.Float32(
        bo.buffer_load(
            r_cs, pos * HEAD_DIM + HEAD_DIM // 2 + d // 2, vec_width=1, dtype=T.f32
        )
    )
    offs = [part_word(hf, tk, j * HEAD_DIM + d) for hf in range(PARTS)]
    gq = poll_part_zs(part.value, offs)
    v = gq[0].bitcast(fx.Float32)
    for hf in range_constexpr(1, PARTS):
        v = v + gq[hf].bitcast(fx.Float32)
    e = bf16_round(v)
    # the per-head norm: the sum of squares of dims 4 (d / 4) .. + 4 in order, then the
    # groups' butterfly
    # lane d - d % ELEMS + k of the quad by a DPP quad broadcast (no LDS round trip)
    ei = e.bitcast(fx.Int32)
    eq = [
        fx.Int32(update_dpp_i32(ei, ei, 0x55 * k, 0xF, 0xF, False)).bitcast(fx.Float32)
        for k in range(ELEMS)
    ]
    ss = fx.Float32(0.0)
    for k in range_constexpr(ELEMS):
        ss = ss + eq[k] * eq[k]
    ss = butterfly(ss, (8 * ELEMS, 4 * ELEMS, 2 * ELEMS, ELEMS))
    rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
    n = bf16_round(e * rstd)
    # the query scale after the norm, rounded
    n = is_q.select(bf16_round(n * QK_SCALE), n)
    # interleaved RoPE on (x, y) = dims (2i, 2i + 1), bf16 cos / sin, each product
    # rounded, then the sum rounded (the fused kernel's order)
    c0, s0 = bf16_round(cs), bf16_round(sn)
    npart = xshfl(n, 1)
    odd = (d % 2) == 1
    rot = odd.select(
        bf16_round(n * c0) + bf16_round(npart * s0),
        bf16_round(n * c0) - bf16_round(npart * s0),
    )
    roped = normed & (use_rope != 0)
    out = normed.select(roped.select(bf16_round(rot), n), e)
    # out (bf16): a lane pair a word; the K / V cache: a lane quad a word of fp8
    o1 = xshfl(out, 1)
    pair = fx.Vector.from_elements([out, o1], fx.Float32).to(fx.BFloat16)
    if (d % 2) == 0:
        if is_q:
            bo.buffer_store(pair, rsrc(a["q_out"]), tk * Q_K + j * HEAD_DIM + d)
        if is_g:
            bo.buffer_store(
                pair, rsrc(a["g_out"]), tk * Q_K + (j - HEAD_TASKS) * HEAD_DIM + d
            )
        if (j >= LOCAL_Q_HEADS) & (j < HEAD_TASKS):
            if is_k:
                bo.buffer_store(
                    pair, rsrc(a["k_out"]), tk * KV_K + kv_head * HEAD_DIM + d
                )
            else:
                bo.buffer_store(
                    pair, rsrc(a["v_out"]), tk * KV_K + kv_head * HEAD_DIM + d
                )
    o2 = xshfl(out, 2)
    o3 = xshfl(o1, 2)
    if ((d % ELEMS) == 0) & (j >= LOCAL_Q_HEADS) & (j < HEAD_TASKS):
        if slot >= 0:
            packed = fp8_pack4(
                clamp_fp8(out), clamp_fp8(o1), clamp_fp8(o2), clamp_fp8(o3)
            )
            # NHD page-16 fp8: slot s, head hh, dim d at byte (s KV_HEADS + hh) HEAD_DIM + d;
            # a full layer's head-major pool (``kv_rows``): byte (hh rows + s) HEAD_DIM + d
            idx = (slot * LOCAL_KV_HEADS + kv_head) * (HEAD_DIM // 4) + d // 4
            # a sliding layer's head-major SWA pool (``swa_rows``): byte (hh swa_rows + s) HEAD_DIM + d
            idx_swa = idx
            if const_expr(SWA_ROWS > 0):
                idx_swa = (kv_head * SWA_ROWS + slot) * (HEAD_DIM // 4) + d // 4
            idx_full = idx
            if const_expr(KV_ROWS > 0):
                idx_full = (kv_head * KV_ROWS + slot) * (HEAD_DIM // 4) + d // 4
            if const_expr(KV_ROWS > 0 or SWA_ROWS > 0):
                idx = (use_swa != 0).select(idx_swa, idx_full)
            if is_k:
                bo.buffer_store(packed, rsrc(a["k_cache"]), idx)
            else:
                bo.buffer_store(packed, rsrc(a["v_cache"]), idx)
    # re-arm the unit's words (after its outputs)
    r_p = rsrc(part.value)
    for o in offs:
        bo.buffer_store(fx.Int32(0), r_p, o, cache_modifier=CM_DEV)


@flyc.jit
def stage_units(c, pre):
    """This CTA's units (``unit_of``), a wave each a pass; ``pre``: each pass's
    (position, slot) from ``unit_pre``."""
    for ps in range_constexpr(-(-unit_waves(c["S"]) // WAVES)):
        live, j, tk = unit_of(c, ps)
        if live:
            stage_unit(c, j, tk, pre[ps][0], pre[ps][1])


@flyc.jit
def run_k1(c, bid, x_flag=None):
    """K1's stages on this CTA: its GEMV task, then its head / gate units. ``x_flag`` (a mailbox region): x was written earlier in this
    launch (the previous layer's boundary), flagged a token each; the first task's
    weights go out before the wait. In that launch the boundary CTAs (0 .. S) reach
    K1 last (their weights would land ~7 us after everyone's x): their GEMV tasks
    run on the last S CTAs, idle since K3's experts, which issue both tasks' weights
    on arrival."""
    S = c["S"]
    stamp(c, 0)
    first = None
    extra = None
    pre_w = c.get("k1_w")
    if const_expr(x_flag is not None and pre_w is not None):
        # every CTA runs its own task: the waves K3 kept longest issued its weights
        # and their units' token words there (``k1_early``, before their fin / row
        # waits), the others now; a wave issues no loads for the other half (a load
        # past the buffer still returns in order, ahead of the x-flag poll)
        late = ~c["k1_early"]
        pre = c["k1_units"]
        now_pre = units_pre_if(c, late)
        pre = [
            (
                late.select(now_pre[ps][0], pre[ps][0]),
                late.select(now_pre[ps][1], pre[ps][1]),
            )
            for ps in range(len(pre))
        ]
        now_w = gemv_w_loads_if(c, bid, c["args"]["w"], late)
        rocdl.sched_barrier(0)
        stamp(c, 4)  # (trace) the x-flag poll issued
        poll_flags(c, [(x_flag, S)])
        stamp(c, 1)  # x arrived
        first = [
            [pre_w[it][kc] | now_w[it][kc] for kc in range(K_STEPS)]
            for it in range(G_TILES)
        ]
        stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
    elif const_expr(x_flag is not None):
        # the units' token words now, ahead of everything they would wait behind
        pre = [
            unit_pre(c, unit_of(c, ps)[2]) for ps in range(-(-unit_waves(S) // WAVES))
        ]
        moved = bid < S  # a boundary CTA: its task runs on CTA BLOCKS - S + bid
        spare = bid >= BLOCKS - S
        own = moved.select(fx.Int32(GEMV_TASKS), bid)  # past the last task: zero loads
        t2 = spare.select(bid - (BLOCKS - S), fx.Int32(GEMV_TASKS))
        first = gemv_w_loads(c, own // K_PARTS, own % K_PARTS, c["args"]["w"])
        extra = gemv_w_loads(c, t2 // K_PARTS, t2 % K_PARTS, c["args"]["w"])
        rocdl.sched_barrier(0)
        if const_expr(not c.get("defer", False)):
            poll_flags(
                c, [(x_flag, S)]
            )  # (deferred: the GEMV polls its unpooled rows itself)
        stamp(c, 1)  # x arrived (a merged launch)
        if bid >= S:
            stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
        if bid >= BLOCKS - S:
            t2 = bid - (BLOCKS - S)
            stage_gemv(c, t2 // K_PARTS, t2 % K_PARTS, extra)
    else:
        # the units' token words now, ahead of everything they would wait behind
        pre = [
            unit_pre(c, unit_of(c, ps)[2]) for ps in range(-(-unit_waves(S) // WAVES))
        ]
        stamp(c, 1)  # entry
        stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
    if const_expr(SECOND_TASKS > 0):
        if (bid >= QT0) & (bid < QT0 + SECOND_TASKS):
            q = BLOCKS + bid - QT0  # the tasks past one a CTA, on idle CTAs
            stage_gemv(c, q // K_PARTS, q % K_PARTS)
    stamp(c, 2)  # the GEMV task(s) done
    stage_units(c, pre)
    stamp(c, 3)  # the units done (end)


def k1_lds(tokens: int):
    """K1's LDS: a reduction row set a row tile (a token tile at a time)."""

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, G_TILES * RED_PITCH, 16]

    return Lds
