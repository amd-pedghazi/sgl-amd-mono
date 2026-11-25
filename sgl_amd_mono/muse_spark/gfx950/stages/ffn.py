# SPDX-License-Identifier: Apache-2.0
"""K3: one Muse Spark layer from the experts' input to the next layer's attention
input, for S = ``tokens`` rows on a rank of the TP 8 / EP 8 group. Per token and
each of its picks this rank holds, the compact-A4 chain of the ROCm model
(``mxfp4_compact_a4_moe`` + the FlyDSL finalize) on the Quark MXFP4 experts as
stored::

    x (bf16: K2's pre-expert-normed pooled input)
      -> per-32 MXFP4 (aiter's _mxfp4_quant_op rule) -> gate / up on the scaled
         E2M1 x E2M1 MFMA (fp32) -> bf16(gate sigmoid(gate) up) -> MXFP4 again
      -> down on the scaled MFMA -> y (bf16, a row a (token, slot))
      -> the finalize: v = bf16(y pen_w), out += score v / sqrt(mean v^2 + 1e-8)
         over the slots in order -> bf16 partial rows
      -> pushed to every rank, summed in rank order (the one-stage all-reduce) -> bf16
      -> the unpooler (rank r's rows r HIDDEN / 8 .., bf16 after the fp32 sum),
         gathered into every rank
      -> the FFN boundary: post_input = bf16(branch pfw), nb = bf16(post_input
         rstd) (eps 1e-8), stream = alpha stream + beta nb (fp32, in place), the
         next layer's pre-attention norm (eps 1e-5, weight npw) -> bf16; on the
         last layer the stream is rounded to bf16 before that norm, whose weight
         is then the final norm's

TP experts (``K3Build.tpx``, ATOM's TP-sharded MoE): every rank holds every
expert's 1/TP of the intermediate width (gate / up rows) and 1/TP of the down
projection's output columns (``weights.tp_expert_copies``), and works every pick
of the step: no rank waits on another's expert load (EP's slowest rank set the
layer's time). The per-(token, slot) post norm needs the slot's whole down row,
so the mids go to every rank (``midx``: ug -> dn), and each dn item pushes its
finalized values v = bf16(y pen_w) and their sum of squares to every rank
(``vx`` / ``ssx``): every rank then finalizes the whole row itself (no
all-reduce) and unpools its rows as before.
The experts' work is grouped by expert. Every CTA builds the step's plan in LDS
(wave 0: a ballot a local expert over rounds of 64 picks; a pick's column is its
rank among its expert's picks), then the picked experts' columns are cut into
column tiles of 16 (an MFMA B operand): an expert picked by every token of a
64-row step has four tiles. The ug and dn items are tile-major, (tile, 32-row
block), spread as contiguous ranges over the CTAs, so a CTA re-stages a tile's
quantized x (or mids) only when its range crosses a tile; the next item's weights
are issued before this item's MFMAs.

Stages (CTA map in ``k3_bases``; every wait points to an earlier stage, all CTAs
co-resident)::

    plan    every CTA, wave 0
    ug      n_tiles x 128 items of (gate pair, up pair): mids -> ``k3_mid`` blocks
    dn      n_tiles x 128 items: y rows -> ``k3_y``
    fin     S x AR_PARTS tasks on CTAs S ..: a part's columns of the token's slots,
            the finalize (slot sums of squares from the dn items' ``k3_ss``), the push to
            every rank's ``ffn`` region, the rank-ordered sum -> ``k3_fbuf`` + a flag
    unpool  the rank's 64 row tiles on the CTAs past the fin tasks (weights issued
            before the fin flags' wait): rows pushed to every rank's ``unp`` region
    bnd     S tasks on CTAs 0 ..: the whole row's boundary -> stream, attention input
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.mx import FP4, mfma_scaled
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_to_bf16x8,
    fresh,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    popcount,
    read_lane_i32,
    row_sum,
    rsrc,
    sfresh64,
    uniform,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, cdiv, pair_layout
from sgl_amd_mono.common.sync import MASK_OOB, _spin_masked, sent_word, shift
from sgl_amd_mono.muse_spark.gfx950.config import (
    EXPERTS,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    MXFP4_GROUP,
    POST_EPS,
    RMS_EPS,
    TOP_K,
)
from sgl_amd_mono.muse_spark.gfx950.device import (
    acquire_loads,
    global_atomic,
    poll_flags,
    release_stores,
    stamp,
)
from sgl_amd_mono.muse_spark.gfx950.layout import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    X_FLAG_COPIES,
)
from sgl_amd_mono.muse_spark.gfx950.stages.mxfp4 import (
    FP4_LOAD_LAYOUT,
    FP4_LOAD_LAYOUTS,
    e2m1_code,
    fp4_load_offsets,
    fp4_unswizzle,
    lanes_before,
    mxfp4_scale,
)
from sgl_amd_mono.muse_spark.weights import TILE_ORDER

POOL_PARTS = 8  # the pooler's K parts (post_attn.P_PARTS)
# the TP experts' tile-shuffled slices store a wave's slab in the load layout's order
assert (
    tuple(
        FP4_LOAD_LAYOUTS[FP4_LOAD_LAYOUT][0][p]
        for p in ("j1", "j0", "q1", "q0", "l3", "l2", "l1", "l0")
    )
    == TILE_ORDER
)

# ---------------------------------------------------------------- the plan
COLS = MAX_TOKENS  # picking columns an expert can have: a token picks it at most once
TILE_COLS = 16  # columns an expert item takes: the MFMA's N
TILES_PER_EXPERT = COLS // TILE_COLS  # 4
MAX_TILES = LOCAL_EXPERTS * TILES_PER_EXPERT  # 128
# ---------------------------------------------------------------- the experts
CHUNK = (
    128  # k of one scaled MFMA: a lane's 16 B of E2M1 (32 values) x the 4 lane groups
)
UG_CHUNKS = MOE_HIDDEN // CHUNK // WAVES  # 4 chunks a wave: the waves split K
DN_CHUNKS = INTER // CHUNK // WAVES  # 4
XS_LDS_MAX = 8  # tokens an unpool pass stages in LDS
XQW = MOE_HIDDEN // 8  # packed E2M1 words a staged column (eight codes a word)
XSW = MOE_HIDDEN // MXFP4_GROUP  # its E8M0 scales, an Int32 each
MID_W = 5  # mailbox pairs a mid block: four words of eight codes, the scale
UG_T = (
    INTER // 32
)  # ug items a column tile: 32 mid rows each (a gate pair + an up pair)
DN_T = MOE_HIDDEN // 32  # dn items a column tile: 32 y rows each (one tile pair)
MID_PAIRS = UG_T * MID_W  # a (token, slot)'s mid blocks
W13_ROW_BYTES = MOE_HIDDEN // 2  # a packed gate / up row of the Quark u8 layout
W13_SCALE_COLS = MOE_HIDDEN // MXFP4_GROUP
W2_ROW_BYTES = INTER // 2
W2_SCALE_COLS = INTER // MXFP4_GROUP
RED_SLOT_FLOATS = WAVES * 64 * 4  # one row set of MFMA accumulators
RED_SLOTS = 8  # two parities x (gate pair + up pair)
# ---------------------------------------------------------------- fin / unpool / bnd
AR_PARTS_MAX = 8
# the fin flags in F_FLAG_COPIES copies (FF_STRIDE pairs apart): the unpool CTAs poll
# every (token, part) flag, and a hundred CTAs polling one copy's lines queue at their
# memory channel (a poll round trip of ~1 us); an unpool CTA polls copy bid % 64 (the
# fin's flag stores: a lane a copy, wave 4 of a split fin task; 64 copies measured
# ~-0.37 us a target case over 16 in the paired TP8 A/B, 7 rounds)
F_FLAG_COPIES = 64
FF_STRIDE = MAX_TOKENS * AR_PARTS_MAX
# the all-reduce's result rows (``k3_fbuf``) in FB_COPIES copies (FB_WORDS words apart):
# every unpool CTA stages every row of its pass; an unpool CTA reads copy bid % fb_copies(S)
FB_COPIES = 8
FB_WORDS = MAX_TOKENS * MOE_HIDDEN // 2


def fb_copies(tokens: int) -> int:
    """The ``k3_fbuf`` copies written at this width (the wide step's fin stores are on
    the heaviest rank's critical path: one copy)."""
    return FB_COPIES if tokens <= 8 else 1


UNP_ROWS = 16  # unpooler rows an unpool task: the MFMA's M
UNP_TASKS = HIDDEN // UNP_ROWS  # 512 row tiles, 1 / npes of them a rank
UNP_STEPS = MOE_HIDDEN // WAVES // 32  # 16 bf16 MFMA steps a wave
BND_ELEMS = HIDDEN // THREADS  # 16 elements a thread of a token's row
assert XQW == THREADS and INTER == MOE_HIDDEN and UG_CHUNKS == 4 and DN_CHUNKS == 4
assert (
    TILE_COLS * (XQW + XSW) <= XS_LDS_MAX * MOE_HIDDEN // 2
)  # a staged tile fits ``xcol``
assert MOE_HIDDEN == 8 * THREADS and HIDDEN == 16 * THREADS and TOP_K == 8
assert LOCAL_EXPERTS <= 64 and MAX_TILES <= 256 and UNP_TASKS == 2 * BLOCKS
MAX_PICKS = (
    MAX_TOKENS * TOP_K
)  # 512: a TP-experts plan's picks (its sorted list) and at most its tiles


def tpx_dims(npes: int) -> dict:
    """A TP-experts build's per-rank slices: ug items a tile (32 intermediate rows of
    the rank's INTER / TP), dn items a tile (32 output columns of its MOE_HIDDEN /
    TP), the rows an expert's gate | up and down slices hold."""
    assert INTER % (32 * npes) == 0 and MOE_HIDDEN % (32 * npes) == 0, npes
    return {
        "ug_t": UG_T // npes,
        "dn_t": DN_T // npes,
        "up_row0": INTER // npes,
        "w13_rows": 2 * INTER // npes,
        "w2_rows": MOE_HIDDEN // npes,
    }


assert MAX_TOKENS % TILE_COLS == 0 and BND_ELEMS == 16


def tile_cols(tokens: int) -> int:
    """Columns a tile can hold at this width: a tile's rounds of x / mid staging."""
    return min(TILE_COLS, tokens)


def ar_parts(tokens: int) -> int:
    """fin / AR tasks a token (``MOE_HIDDEN / parts`` columns each): 8 ranks' partial
    rows are too many for one CTA's poll, and S x parts CTAs spread the pushes; fewer
    parts at the wide steps leave the unpool tasks their own CTAs."""
    return 8 if tokens <= 8 else (4 if tokens <= 32 else 2)


def fin_ppt(parts: int) -> int:
    """Column pairs a fin thread (1 or 2: a poll spec of at most two pairs)."""
    ppt = cdiv(MOE_HIDDEN // parts // 2, THREADS)
    assert ppt in (1, 2), ppt
    return ppt


def k3_bases(tokens: int, npes: int) -> dict:
    """K3's CTA map: the bnd tasks on CTAs 0 .. S, the fin tasks (``ar_parts`` a
    token) from ``fin0``, the unpool tasks from ``unp0`` with ``unp_stride`` CTAs
    taking ``ku`` tasks each (one at TP 8). Disjoint, so a fin CTA's polls wait
    behind no unpool weights and a bnd CTA carries nothing it waits on."""
    parts = ar_parts(tokens)
    fin0 = tokens
    unp0 = fin0 + tokens * parts
    stride = BLOCKS - unp0
    tasks = UNP_TASKS // npes
    passes = cdiv(tokens, XS_LDS_MAX)  # token passes of XS_LDS_MAX: a job each
    assert UNP_TASKS % npes == 0 and stride > 0, (tokens, npes)
    return {
        "ar_parts": parts,
        "fin0": fin0,
        "unp0": unp0,
        "unp_stride": stride,
        "unp_tasks": tasks,
        "unp_passes": passes,
        "unp_jobs": tasks
        * passes,  # (row tile, token pass) pairs, spread over the unpool CTAs
        "ku": cdiv(tasks * passes, stride),
    }


def k3_scratch_layout() -> dict:
    """K3's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS: the
    mid blocks and y rows a (token, slot), a flag a (token, AR part), the dn items'
    sums of squares of the finalized values a (token, slot, item), and the
    all-reduce's result as plain bf16 words (``k3_fbuf``: 4 elements a "pair", read
    by LDS DMA, not polled)."""
    return pair_layout(
        [
            ("k3_mid", MAX_TOKENS * TOP_K * MID_PAIRS),
            ("k3_y", MAX_TOKENS * TOP_K * MOE_HIDDEN // 2),
            ("k3_fflag", F_FLAG_COPIES * MAX_TOKENS * AR_PARTS_MAX),
            ("k3_ss", MAX_TOKENS * TOP_K * DN_T),
            ("k3_fbuf", FB_COPIES * MAX_TOKENS * MOE_HIDDEN // 4),
            (
                "k3_nscal",
                MAX_TOKENS * 2,
            ),  # a token's (post rstd, pre rstd) for the next K1 (deferred)
            (
                "k3_ugq",
                2,
            ),  # the launch's ug queue: a word (zeroed at the merged launch's entry)
            ("k3_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def k3_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric regions, name -> (byte offset, bytes): ``attn`` and
    ``ffn`` [source rank][token][width / 2] bf16 pairs, ``unp`` [token][HIDDEN / 2]
    (each rank's unpooled rows from that rank). Must equal post_attn.sym_layout
    (the same names, order and formula: attn = npes MAX_TOKENS HIDDEN / 2 pairs
    at 0, ffn = npes MAX_TOKENS MOE_HIDDEN / 2 after it, unp = MAX_TOKENS HIDDEN /
    2 after that); the two are to be unified in one place."""
    return pair_layout(
        [
            ("attn", npes * MAX_TOKENS * HIDDEN // 2),
            ("ffn", npes * MAX_TOKENS * MOE_HIDDEN // 2),
            ("unp", MAX_TOKENS * HIDDEN // 2),
            # TP experts: every rank's mid blocks a pick, the finalized values and the dn
            # items' sums of squares a pick (each written by the rank owning that part)
            ("midx", MAX_TOKENS * TOP_K * MID_PAIRS),
            ("vx", MAX_TOKENS * TOP_K * MOE_HIDDEN // 2),
            ("ssx", MAX_TOKENS * TOP_K * DN_T),
            # the sharded pooler: every rank's partial pooled rows of its row slice, [K part][token][row]
            ("poolx", POOL_PARTS * MAX_TOKENS * MOE_HIDDEN // 2),
        ],
        align=256,
    )


def k3_lds_floats() -> dict:
    """K3's LDS, name -> floats (ints for the plan), independent of the width."""
    return {
        "red": RED_SLOTS * RED_SLOT_FLOATS,
        "xcol": XS_LDS_MAX
        * MOE_HIDDEN
        // 2,  # a tile's MXFP4 columns; the unpool passes' rows
        "misc": max(
            UNP_ROWS * XS_LDS_MAX, 2 * WAVES * TILE_COLS
        ),  # an unpool pass's bf16 row sums; dn's wave partials
        "sums": 2 * WAVES,  # ``block_sum``'s wave sums
        # EP: expert e's column j: its pick (token TOP_K + slot); TP experts: the picks sorted
        # by expert (expert e's column j at start_of[e] + j)
        "pick_of": max(LOCAL_EXPERTS * COLS, MAX_PICKS),
        "cnt_of": EXPERTS,  # expert e's picks
        "start_of": EXPERTS,  # TP experts: expert e's first sorted pick
        "tile_of": max(
            MAX_TILES, MAX_PICKS
        ),  # column tile i: e COLS + its first column
        "nt": 4,
    }


def k3_lds_bytes(tokens: int) -> int:
    return 4 * sum(k3_lds_floats().values())


def k3_lds(tokens: int):
    """K3's LDS: the tiles' accumulators, a tile's staged columns (also the
    unpool passes' rows), the reductions, the step's plan."""
    f = k3_lds_floats()
    n_red, n_xcol, n_misc, n_sums, n_pick_of, n_cnt_of, n_start_of, n_tile_of, n_nt = (
        f[k]
        for k in (
            "red",
            "xcol",
            "misc",
            "sums",
            "pick_of",
            "cnt_of",
            "start_of",
            "tile_of",
            "nt",
        )
    )

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, n_red, 16]
        xcol: fx.Array[fx.Float32, n_xcol, 16]
        misc: fx.Array[fx.Float32, n_misc, 16]
        sums: fx.Array[fx.Float32, n_sums, 16]
        pick_of: fx.Array[fx.Int32, n_pick_of, 16]
        cnt_of: fx.Array[fx.Int32, n_cnt_of, 16]
        start_of: fx.Array[fx.Int32, n_start_of, 16]
        tile_of: fx.Array[fx.Int32, n_tile_of, 16]
        nt: fx.Array[fx.Int32, n_nt, 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def bor(x, y):
    return x | y


def lane_prefix(v, lane):
    """The wave's inclusive prefix sum of v, lanes ascending (integers: any order is
    the same sum): DPP row shifts 1, 2, 4, 8 within each 16-lane row (an out-of-row
    source reads 0), then row_bcast15 (rows 1, 3 add their lower row's last lane) and
    row_bcast31 (rows 2, 3 add lane 31): six VALU steps, no LDS round trips."""
    from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

    incl = fx.Int32(v)
    for d in (1, 2, 4, 8):
        incl = incl + fx.Int32(
            update_dpp_i32(fx.Int32(0), incl, 0x110 + d, 0xF, 0xF, True)
        )
    incl = incl + fx.Int32(update_dpp_i32(fx.Int32(0), incl, 0x142, 0xA, 0xF, False))
    incl = incl + fx.Int32(update_dpp_i32(fx.Int32(0), incl, 0x143, 0xC, 0xF, False))
    return incl


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` through ``sums`` row ``slot`` (one barrier)."""
    lane, wave, sums = c["lane"], c["wave"], c["sums"]
    ws = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ws, sums + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(sums + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(sums + (slot * WAVES + w))
    return tot


def pick(c, t, slot):
    """Token t's pick in ``slot`` as a local expert, or -1 (another rank's); TP experts:
    the expert (every rank works every pick)."""
    a = c["args"]
    e = fx.Int32(
        bo.buffer_load(
            rsrc(a["ids"]),
            t * TOP_K + slot,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    if const_expr(c["tpx"]):
        return fx.max(fx.min(e, fx.Int32(EXPERTS - 1)), fx.Int32(0))
    e = e - a["local_start"]
    return ((e >= 0) & (e < LOCAL_EXPERTS)).select(e, fx.Int32(-1))


def pick_at(c, e, col):
    """Expert e's column ``col``'s pick: token TOP_K + slot."""
    if const_expr(c["tpx"]):
        return fx.ptr_load(c["pick_of"] + (fx.ptr_load(c["start_of"] + e) + col))
    return fx.ptr_load(c["pick_of"] + (e * COLS + col))


def tile_info(c, ti, n_tiles):
    """Column tile ``ti``'s (local expert, first column, columns), clamped to valid
    values past the plan (a CTA with no item still issues loads: expert 0's)."""
    tc = fx.max(fx.min(ti, n_tiles - 1), fx.Int32(0))
    packed = uniform(fx.ptr_load(c["tile_of"] + tc))
    packed = fx.max(
        fx.min(packed, fx.Int32((EXPERTS if c["tpx"] else LOCAL_EXPERTS) * COLS - 1)),
        fx.Int32(0),
    )
    e = packed // COLS
    col0 = packed % COLS
    cnt = uniform(fx.ptr_load(c["cnt_of"] + e))
    n = fx.max(fx.min(cnt - col0, fx.Int32(TILE_COLS)), fx.Int32(1))
    return e, col0, n


def w13_rsrc(c, e):
    """Expert e's packed gate | up rows and their scales (the Quark layout as stored;
    TP experts: the rank's slice, gate rows then up rows, ``w13_rows`` an expert)."""
    a = c["args"]
    rows = c["w13_rows"]
    return (
        rsrc(a["w13"] + fx.Int64(e) * (rows * W13_ROW_BYTES)),
        rsrc(a["s13"] + fx.Int64(e) * (rows * W13_SCALE_COLS)),
    )


def w2_rsrc(c, e, live=None):
    """Expert e's down rows (TP experts: the rank's ``w2_rows`` output rows); with
    ``live`` false, empty buffers: the loads return zeros and move no bytes."""
    a = c["args"]
    rows = c["w2_rows"]
    if live is None:
        return (
            rsrc(a["w2"] + fx.Int64(e) * (rows * W2_ROW_BYTES)),
            rsrc(a["s2"] + fx.Int64(e) * (rows * W2_SCALE_COLS)),
        )
    return (
        rsrc(
            a["w2"] + fx.Int64(e) * (rows * W2_ROW_BYTES),
            live.select(fx.Int64(rows * W2_ROW_BYTES), fx.Int64(0)),
        ),
        rsrc(
            a["s2"] + fx.Int64(e) * (rows * W2_SCALE_COLS),
            live.select(fx.Int64(rows * W2_SCALE_COLS), fx.Int64(0)),
        ),
    )


def fp4_tile_loads(c, r_w, r_s, row, chunks):
    """This lane's loads of one 16-row tile (``row``: its row, the tile's first + lane
    % 16) over the wave's ``chunks`` k chunks: (weight words, scale bytes) in flight.
    The MFMA wants a lane's 16 B = 32 E2M1 of the row at byte row K / 2 + chunk 64 +
    lane group 16; the words are loaded in the ``mxfp4.FP4_LOAD_LAYOUT`` lane map (a
    lane group reads whole sectors) and brought to that layout at the MFMAs
    (``tile_mfmas_a4``); the wave's chunks' 16 scale bytes of the row are one load (byte
    4 kc + g4: chunk kc's, lane group g4's). TP experts' slices are tile-shuffled
    (``weights.tile_shuffle``): each register's 16 B a lane in the same lane map, a
    wave's register 1 KB contiguous."""
    lane, wave = c["lane"], c["wave"]
    assert chunks == 4
    row_b = chunks * WAVES * CHUNK // 2
    lane_off, reg_off = fp4_load_offsets(lane, row_b)
    base = (row - lane % 16) * row_b + wave * (chunks * CHUNK // 2) + lane_off
    if c["tpx"]:
        # the tile-shuffled slices (``weights.tile_shuffle``): the same registers, each
        # a wave's 1 KB contiguous at register j x 1 KB of its 4 KB slab
        base = (
            (row // 16) * (16 * row_b) + wave * (16 * chunks * CHUNK // 2) + lane * 16
        )
        reg_off = [j * 1024 for j in range(chunks)]
    wts = []
    for kc in range(chunks):
        wts.append(
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (base + reg_off[kc]) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,  # read once a layer
                )
            )
        )
    sc16 = fx.Vector(
        bo.buffer_load(
            r_s,
            (row * (chunks * WAVES * CHUNK // 32) + wave * 16) // 4,
            vec_width=4,
            dtype=T.i32,
        )
    )
    # the scale bytes are taken at the MFMAs (``tile_mfmas_a4``): taken here, a loop
    # that carries the next tile's loads would wait for them at its back edge
    return wts, [sc16]


def tile_mfmas_a4(wts, scs, xb):
    """A tile's fp32 accumulator over the wave's chunks: the scaled E2M1 x E2M1
    MFMA on the weight words against the staged column blocks ``xb[kc]`` =
    (words, scale)."""
    if len(scs) == 1:
        # the raw scale words (``fp4_tile_loads``): lane group g4's byte of chunk kc
        g4 = (fx.thread_idx.x % 64) // 16
        sc16 = scs[0]
        scs = [(fx.Int32(sc16[kc]) >> (8 * g4)) & 0xFF for kc in range(len(wts))]
    wts = fp4_unswizzle(wts, fx.thread_idx.x % 64)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for kc in range(len(wts)):
        acc = mfma_scaled(
            wts[kc],
            xb[kc][0],
            acc,
            fx.Int32(scs[kc]),
            fx.Int32(xb[kc][1]),
            a_fmt=FP4,
            b_fmt=FP4,
        )
    return acc


def tile_store(c, slot, acc):
    fx.ptr_store(acc, c["red"] + ((slot * WAVES + c["wave"]) * 64 + c["lane"]) * 4)


def xq_loads(c, e, col0, n):
    """The bf16 x of expert e's columns col0 .. col0 + n, every round's load at
    once (in flight): round i is column i, eight values a thread. A column past n
    reads the last one's (its store is skipped); the token is clamped to the
    step's rows, as a rank no pick reached stages its (unused) first tile from a
    plan it never wrote."""
    S, tid = c["S"], c["tid"]
    r_xin = rsrc(c["args"]["xin"])

    def x_unit(i):
        v = tid + THREADS * i  # eight values: column v / XQW, values 8 (v % XQW) ..
        tk = pick_at(c, e, col0 + fx.min(v // XQW, n - 1)) // TOP_K
        tk = fx.max(fx.min(tk, fx.Int32(S - 1)), fx.Int32(0))
        return fx.Vector(
            bo.buffer_load(
                r_xin,
                tk * (MOE_HIDDEN // 2) + (v % XQW) * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )

    return [x_unit(i) for i in range(tile_cols(S))]


def xq_all_loads(c):
    """Every token's MXFP4 blocks (K2's ``xinq``: XQW code words then XSW scale words
    a token), 16 B a thread a round, in flight (the staged-column layout with the
    column = the token; S <= TILE_COLS)."""
    S, tid = c["S"], c["tid"]
    W_TOK = XQW + XSW
    r = rsrc(c["args"]["xinq"])
    n4 = S * W_TOK // 4
    return [bo.buffer_load(r, 4 * fx.min(tid + THREADS * i, n4 - 1), vec_width=4, dtype=T.i32, cache_modifier=CM_DEV) for i in range(cdiv(n4, THREADS))]  # fmt: skip


@flyc.jit
def stage_xq_all(c, xv):
    """The blocks (``xq_all_loads``) into LDS: token t's words at xcol[t XQW ..], its
    scales at xcol[TILE_COLS XQW + t XSW ..]."""
    S, tid, xcol = c["S"], c["tid"], c["xcol"]
    W_TOK = XQW + XSW
    n4 = S * W_TOK // 4
    for i in range_constexpr(len(xv)):
        v = tid + THREADS * i
        if v < n4:
            w4 = fx.Vector(xv[i])
            for j in range_constexpr(4):
                d = 4 * v + j
                tk, o = d // W_TOK, d % W_TOK
                at = (o < XQW).select(
                    tk * XQW + o, TILE_COLS * XQW + tk * XSW + (o - XQW)
                )
                fx.ptr_store(fx.Int32(w4[j]).bitcast(fx.Float32), xcol + at)
    gpu.barrier()


def xq_dma_ok(S) -> bool:
    """``xq_dma`` covers the step: whole 1 KB DMA rows of code and of scale words."""
    return (S * XQW * 4) % 1024 == 0 and (S * XSW * 4) % 1024 == 0


def xq_dma(c):
    """``stage_xq_all``'s LDS image of every token's blocks, DMA'd straight from K2's
    ``xinq`` by this wave alone (16 B a lane an instruction, ``sc1``: other CTAs stored
    them in this launch): the codes then the scales, each a contiguous run of tokens in
    ``xcol``. The caller waits (``s_waitcnt vmcnt(0)``) and holds the barrier. On the
    poll wave, behind nothing but its own share of the first weights: issued on every
    wave, the loads returned behind each wave's weights (loads complete in order)."""
    S, lane, xcol = c["S"], c["lane"], c["xcol"]
    W_TOK = XQW + XSW
    r = rsrc(c["args"]["xinq"])
    for i in range_constexpr(S * XQW * 4 // 1024):
        byte = i * 1024 + lane * 16
        tk = byte // (XQW * 4)
        lds_dma_sc1(r, xcol + i * 256, tk * (W_TOK * 4) + byte % (XQW * 4))
    for j in range_constexpr(S * XSW * 4 // 1024):
        byte = j * 1024 + lane * 16
        tk = byte // (XSW * 4)
        lds_dma_sc1(
            r,
            xcol + (TILE_COLS * XQW + j * 256),
            tk * (W_TOK * 4) + XQW * 4 + byte % (XSW * 4),
        )


def xq_b_tok(c, tok, chunks, kc):
    """This lane's B operand of chunk ``kc`` from token ``tok``'s staged blocks
    (``stage_xq_all``): its 32 codes of k block g4 and their scale."""
    lane, wave, xcol = c["lane"], c["wave"], c["xcol"]
    V4F = fx.Vector.make_type(4, fx.Float32)
    kb = (wave * chunks + kc) * (CHUNK // MXFP4_GROUP) + lane // 16
    q = fx.Vector(fx.ptr_load(xcol + (tok * XQW + kb * 4), result_type=V4F)).bitcast(
        fx.Int32
    )
    sc = fx.ptr_load(xcol + (TILE_COLS * XQW + tok * XSW + kb)).bitcast(fx.Int32)
    return q, sc


def lane_token(c, e, col0, n):
    """The token of this lane's column (lane % 16) of a tile (the last column's past n)."""
    tk = pick_at(c, e, col0 + fx.min(c["lane"] % 16, n - 1)) // TOP_K
    return fx.max(fx.min(tk, fx.Int32(c["S"] - 1)), fx.Int32(0))


def xq_b(c, n, chunks, kc, cb=0):
    """This lane's B operand of chunk ``kc``: its column's 32 codes of k block g4
    and their scale, from the staged blocks (a column past n: the last; ``cb``: the
    tile's first LDS column)."""
    lane, wave, xcol = c["lane"], c["wave"], c["xcol"]
    V4F = fx.Vector.make_type(4, fx.Float32)
    col = cb + fx.min(lane % 16, n - 1)
    kb = (wave * chunks + kc) * (CHUNK // MXFP4_GROUP) + lane // 16
    q = fx.Vector(fx.ptr_load(xcol + (col * XQW + kb * 4), result_type=V4F)).bitcast(
        fx.Int32
    )
    sc = fx.ptr_load(xcol + (TILE_COLS * XQW + col * XSW + kb)).bitcast(fx.Int32)
    return q, sc


def ug_pair_loads(c, q, pr, n_tiles):
    """ug item q's pair ``pr`` (0: gate rows, 1: up rows) of two 16-row tiles."""
    l16 = c["lane"] % 16
    e, _, _ = tile_info(c, q // c["ug_t"], n_tiles)
    r_w, r_s = w13_rsrc(c, e)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c,
                r_w,
                r_s,
                pr * c["up_row0"] + (q % c["ug_t"]) * 32 + t * 16 + l16,
                UG_CHUNKS,
            )
        ]
        for t in range(2)
    ]


def ug_or_dn_pair_loads(c, q, pr, dq, n_tiles, use_dn):
    """ug item ``q``'s pair ``pr``, or (``use_dn``) dn item ``dq``'s tile pair: the
    same load shape (gate / up and down rows are both MOE_HIDDEN = INTER wide), the
    buffer and rows selected, so the ug loop's last prefetch fetches the dn stage's
    first weights."""
    assert (
        MOE_HIDDEN == INTER
        and W13_ROW_BYTES == W2_ROW_BYTES
        and W13_SCALE_COLS == W2_SCALE_COLS
    )
    l16 = c["lane"] % 16
    a = c["args"]
    e_u, _, _ = tile_info(c, q // c["ug_t"], n_tiles)
    e_d, _, _ = tile_info(c, dq // c["dn_t"], n_tiles)
    UGT, DNT, R13, R2, UP0 = (
        c["ug_t"],
        c["dn_t"],
        c["w13_rows"],
        c["w2_rows"],
        c["up_row0"],
    )
    w_addr = use_dn.select(a["w2"] + fx.Int64(e_d) * (R2 * W2_ROW_BYTES), a["w13"] + fx.Int64(e_u) * (R13 * W13_ROW_BYTES))  # fmt: skip
    s_addr = use_dn.select(a["s2"] + fx.Int64(e_d) * (R2 * W2_SCALE_COLS), a["s13"] + fx.Int64(e_u) * (R13 * W13_SCALE_COLS))  # fmt: skip
    r_w, r_s = rsrc(w_addr), rsrc(s_addr)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c,
                r_w,
                r_s,
                use_dn.select(
                    (dq % DNT) * 32 + t * 16 + l16,
                    pr * UP0 + (q % UGT) * 32 + t * 16 + l16,
                ),
                UG_CHUNKS,
            )
        ]  # fmt: skip
        for t in range(2)
    ]


def pair_loads_e(c, q, pr, dq, e_u, e_d, use_dn, a=None):
    """``ug_or_dn_pair_loads`` with the tiles' experts given (``first_pair_mask``);
    ``a``: the weights' bases (default: the arguments)."""
    l16 = c["lane"] % 16
    a = c["args"] if a is None else a
    UGT, DNT, R13, R2, UP0 = (
        c["ug_t"],
        c["dn_t"],
        c["w13_rows"],
        c["w2_rows"],
        c["up_row0"],
    )
    w_addr = use_dn.select(a["w2"] + fx.Int64(e_d) * (R2 * W2_ROW_BYTES), a["w13"] + fx.Int64(e_u) * (R13 * W13_ROW_BYTES))  # fmt: skip
    s_addr = use_dn.select(a["s2"] + fx.Int64(e_d) * (R2 * W2_SCALE_COLS), a["s13"] + fx.Int64(e_u) * (R13 * W13_SCALE_COLS))  # fmt: skip
    r_w, r_s = rsrc(w_addr), rsrc(s_addr)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c,
                r_w,
                r_s,
                use_dn.select(
                    (dq % DNT) * 32 + t * 16 + l16,
                    pr * UP0 + (q % UGT) * 32 + t * 16 + l16,
                ),
                UG_CHUNKS,
            )
        ]  # fmt: skip
        for t in range(2)
    ]


def nth_expert(mask, n_tiles, t):
    """The local expert of column tile ``t`` (0 or 1) of a plan whose experts (one tile
    each, S <= TILE_COLS) are ``mask``'s set bits in order, clamped as ``tile_info``
    clamps it (an empty plan: expert 0)."""
    tc = fx.max(fx.min(t, n_tiles - 1), fx.Int32(0))
    m1 = (tc == 1).select(mask & (mask - 1), mask)  # the tile's bit is the lowest left
    low = m1 & (fx.Int32(0) - m1)
    e = popcount(low - 1)
    return (m1 == 0).select(fx.Int32(0), fx.min(e, fx.Int32(LOCAL_EXPERTS - 1)))


def first_pair_mask(c, mask):
    """This CTA's first pair's weights (``stage_ug``'s ``cur``) from the picked local
    experts' mask alone, before the plan: at S <= TILE_COLS an expert holds one column
    tile and the tiles list the picked experts in order, so the plan's tiles are the
    mask's set bits."""
    bid, UGT, DNT = c["bid"], c["ug_t"], c["dn_t"]
    n_tiles = popcount(mask)
    q_of, n_ug = ug_count(c, n_tiles)
    q0 = q_of(0)
    dq0, _ = dn_range(c, n_tiles)
    use_dn0 = n_ug == 0
    e_u = nth_expert(mask, n_tiles, fx.min(q0 // UGT, fx.Int32(1)))
    e_d = nth_expert(mask, n_tiles, fx.min(dq0 // DNT, fx.Int32(1)))
    return pair_loads_e(c, q0, 0, dq0, e_u, e_d, use_dn0)


def first_pair_table(c):
    """``first_pair_mask``'s plan arithmetic for every plan size, before the routing is
    known (``run_k3``: while it is polled): lane i holds n_tiles = i's first dn item
    and whether the CTA has no ug item, as dq0 | use_dn << 16 (``ug_count`` /
    ``dn_range``'s integer ops, lane-parallel; at S <= TILE_COLS n_tiles <= the local
    experts), and the expert weights' bases pinned in SGPRs (the routing's critical
    path then reloads no kernel argument and runs no runtime division)."""
    a = c["args"]
    ntv = fx.min(c["lane"], fx.Int32(LOCAL_EXPERTS))
    _, n_ug = ug_count(c, ntv)
    dq0, _ = dn_range(c, ntv)
    # (opaque moves: LLVM neither sinks the arithmetic past the routing's wait nor
    # rematerializes the bases from the kernarg segment after it)
    tab = fresh(fx.Int32(dq0) | (n_ug == 0).select(fx.Int32(1 << 16), fx.Int32(0)))
    return tab, {k: sfresh64(a[k]) for k in ("w13", "s13", "w2", "s2")}


def first_pair_fast(c, mask, tab, ka):
    """``first_pair_mask`` (the same loads) from ``first_pair_table``'s lane for this
    plan size and its pinned bases."""
    bid, UGT, DNT = c["bid"], c["ug_t"], c["dn_t"]
    n_tiles = popcount(mask)
    w = read_lane_i32(tab, n_tiles)
    dq0 = w & 0xFFFF
    use_dn0 = ((w >> 16) & 1) != 0
    q0 = bid  # (``ug_balanced``: the CTA's first ug item)
    e_u = nth_expert(mask, n_tiles, fx.min(q0 // UGT, fx.Int32(1)))
    e_d = nth_expert(mask, n_tiles, fx.min(dq0 // DNT, fx.Int32(1)))
    return pair_loads_e(c, q0, 0, dq0, e_u, e_d, use_dn0, dict(c["args"], **ka))


def dn_pair_loads(c, q, n_tiles, live=None, null=None):
    """dn item q's tile pair: 32 rows of the expert's down weight. ``live`` false /
    ``null`` true: no pair (an empty buffer: zeros, no traffic)."""
    l16 = c["lane"] % 16
    e, _, _ = tile_info(c, q // c["dn_t"], n_tiles)
    r_w, r_s = w2_rsrc(c, e, live)
    if null is not None:
        nb = null.select(fx.Int32(0), fx.Int32(0x7FFFFFFF))
        a = c["args"]
        rows = c["w2_rows"]
        r_w = rsrc(a["w2"] + fx.Int64(e) * (rows * W2_ROW_BYTES), nb)
        r_s = rsrc(a["s2"] + fx.Int64(e) * (rows * W2_SCALE_COLS), nb)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c, r_w, r_s, (q % c["dn_t"]) * 32 + t * 16 + l16, DN_CHUNKS
            )
        ]
        for t in range(2)
    ]


UNP_CLAIM_WORD = (
    2  # the claim counter's word in ``k3_ugq`` (``ug_dyn``'s queue is word 0)
)


@flyc.jit
def unp_claim(c, n_pool: int):
    """This pool CTA's ticket among the launch's pool CTAs (past the fin
    tasks) in the order their dn stage ended: the first ``unp_jobs`` take the unpool
    jobs, so a CTA whose dn ends last carries no unpool or early K1 weights into the
    fin hops (its CU's loads delayed the fin flags' poll). One atomic add a CTA on a
    counter that starts the launch at 0: the launch's last claim (every pool CTA
    claims once) zeroes it again, the next launch's claims coming after this one."""
    tid, nt = c["tid"], c["nt"]
    addr = c["ugq"].value + fx.Int64(4 * UNP_CLAIM_WORD)
    if tid == 0:
        ticket = global_atomic(addr, 1)
        if ticket == n_pool - 1:
            global_atomic(addr, 0, xchg=True)
        fx.ptr_store(ticket, nt + 2)
    gpu.barrier()
    return uniform(fx.ptr_load(nt + 2))


def unp_gtask(c, k):
    """CTA ``bid``'s k-th unpool job: (its index among the rank's jobs, its row tile of
    the unpooler, its token pass; outside the rank's jobs: the tile past the buffer, no
    traffic). A job is one row tile over one pass of XS_LDS_MAX tokens, so a wide
    step's passes spread over the unpool CTAs instead of queuing on a tile's. With
    ``unp_job`` (``unp_claim``) the job is the CTA's claimed ticket, not its place."""
    b = c["bases"]
    if const_expr(c.get("unp_job") is not None):
        assert k == 0
        job = c["unp_job"]  # (``unp_claim``)
    else:
        job = c["bid"] - b["unp0"] + b["unp_stride"] * k
    live = (job >= 0) & (job < b["unp_jobs"])
    tile = job // b["unp_passes"]
    ps = job % b["unp_passes"]
    return job, live.select(c["rank"] * b["unp_tasks"] + tile, fx.Int32(UNP_TASKS)), ps


def unp_w_loads(c, gtask):
    """A lane's loads of unpooler row tile ``gtask`` (16 rows x MOE_HIDDEN, row-major
    bf16, or E4M3 codes with ``c["fp8_unpool"]``): row l16, this wave's K eighth, 8
    values a step; 64 VGPRs (32 FP8), in flight."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    fp8 = c["fp8_unpool"]
    # a tile past the last reads past the buffer: zeros, no traffic
    r_u = rsrc(c["args"]["w_unpool"], HIDDEN * MOE_HIDDEN * (1 if fp8 else 2))
    row = gtask * UNP_ROWS + l16
    return [
        fx.Vector(
            bo.buffer_load(
                r_u,
                (row * MOE_HIDDEN + wave * (MOE_HIDDEN // WAVES) + kc * 32 + 8 * g4)
                // (4 if fp8 else 2),
                vec_width=2 if fp8 else 4,
                dtype=T.i32,
            )
        )
        for kc in range(UNP_STEPS)
    ]


@flyc.jit
def unp_w_loads_if(c, gtask, live):
    """``unp_w_loads`` on a CTA with an unpool job (``live``); zeros, and no loads at
    all, on the others (a load past the buffer still holds the wave's in-order
    returns: the fin CTAs' polls came back behind them)."""
    got = [
        fx.Vector.filled(2 if c["fp8_unpool"] else 4, 0, fx.Int32)
        for _ in range(UNP_STEPS)
    ]
    if live:
        got = unp_w_loads(c, gtask)
    return got


def unp_a(c, w):
    """An unpooler weight step as the MFMA's bf16 A operand (FP8 codes decoded)."""
    if c["fp8_unpool"]:
        return fp8_to_bf16x8(w[0], w[1])
    return w.bitcast(fx.BFloat16)


def unp_scale(c, row):
    """The unpooler row's FP8 scale (fp32, after the codes), or 1."""
    if not c["fp8_unpool"]:
        return fx.Float32(1.0)
    return fx.Float32(
        bo.buffer_load(
            rsrc(c["args"]["w_unpool"]),
            (HIDDEN * MOE_HIDDEN) // 4 + row,
            vec_width=1,
            dtype=T.f32,
        )
    )


# ---------------------------------------------------------------- 0. the plan
@flyc.jit
def stage_plan(c, ids_lds=None):
    """The step's plan in LDS, the same on every CTA: wave w owns local experts
    4 w .. 4 w + 4 and walks every pick (token p / TOP_K, slot p % TOP_K) in rounds
    of 64; a pick's column is its rank among its expert's picks in pick order (a
    ballot an expert a round, the count a uniform scalar); then wave 0 lists the
    experts' column tiles of 16 in expert order. Returns the tiles. (One wave over
    all 32 experts took 11 us at 64 rows; LDS atomics broke the multi-tile experts.)"""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    pick_of, cnt_of, tile_of, nt = c["pick_of"], c["cnt_of"], c["tile_of"], c["nt"]
    a = c["args"]
    local_start = a["local_start"]
    EPW = LOCAL_EXPERTS // WAVES  # experts a wave: 4
    r_ids = rsrc(a["ids"])
    PICKS = S * TOP_K
    ROUNDS = cdiv(PICKS, 64)
    # every round's ids, in flight at once (every wave loads them: 8 x 64 ids a round),
    # or (``ids_lds``, a merged launch) from the words ``run_k3`` polled into LDS
    if const_expr(ids_lds is not None):
        p_ids = [
            fx.ptr_load(ids_lds + fx.min(rr * 64 + lane, PICKS - 1)).bitcast(fx.Int32)
            for rr in range(ROUNDS)
        ]
    else:
        p_ids = [
            fx.Int32(
                bo.buffer_load(
                    r_ids,
                    fx.min(rr * 64 + lane, PICKS - 1),
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )  # fmt: skip
            for rr in range(ROUNDS)
        ]
    cnts = [
        fx.Int32(0) for _ in range(EPW)
    ]  # this wave's experts' picks so far (uniform)
    for rr in range_constexpr(ROUNDS):
        p = rr * 64 + lane
        le = p_ids[rr] - local_start
        ok = (p < PICKS) & (le >= 0) & (le < LOCAL_EXPERTS)
        col = fx.Int32(0)
        mine = fx.Int32(0) == fx.Int32(1)
        for k in range_constexpr(EPW):
            hit = ok & (le == wave * EPW + k)
            m = ballot(hit)
            col = hit.select(cnts[k] + lanes_before(m, lane), col)
            cnts[k] = cnts[k] + uniform(popcount(m))
            mine = mine | hit
        if mine & (col < COLS):
            fx.ptr_store(p, pick_of + (le * COLS + col))
    for k in range_constexpr(EPW):
        if lane == 0:
            fx.ptr_store(cnts[k], cnt_of + (wave * EPW + k))
    gpu.barrier()
    if wave == 0:
        # lane e < LOCAL_EXPERTS: expert e's count, its tiles and their first index
        mine0 = lane < LOCAL_EXPERTS
        cnt = mine0.select(
            fx.ptr_load(cnt_of + fx.min(lane, LOCAL_EXPERTS - 1)), fx.Int32(0)
        )
        tiles = mine0.select(
            (fx.min(cnt, COLS) + TILE_COLS - 1) // TILE_COLS, fx.Int32(0)
        )
        incl = lane_prefix(tiles, lane)
        tile0 = incl - tiles
        n_tiles = read_lane_i32(incl, 63)
        if lane == 0:
            # an empty plan's first tile is expert 0, column 0: a CTA with no item
            # still issues its first item's loads, from the list's head, not the
            # LDS's stale word (a later store of a real tile 0 wins)
            fx.ptr_store(fx.Int32(0), tile_of)
            fx.ptr_store(n_tiles, nt)
        for j in range_constexpr(TILES_PER_EXPERT):
            if j < tiles:
                fx.ptr_store(lane * COLS + j * TILE_COLS, tile_of + (tile0 + j))
    gpu.barrier()
    return uniform(fx.ptr_load(nt))


POLL_WAVE = (
    WAVES - 1
)  # a merged launch's K3 prologue polls on the last wave (``run_k3``)


def ug_first_loads(c, q, dq, n_tiles, use_dn, live):
    """``ug_or_dn_pair_loads(c, q, 0, dq, n_tiles, use_dn)`` with the buffers emptied
    where ``live`` is off (zeros, no traffic): the first pair split over the waves'
    issue times (``stage_ug``: the poll wave issues its share after the expert input's
    wait, the others at the plan); the two shares are OR-ed (``or_pair``)."""
    l16 = c["lane"] % 16
    a = c["args"]
    e_u, _, _ = tile_info(c, q // c["ug_t"], n_tiles)
    e_d, _, _ = tile_info(c, dq // c["dn_t"], n_tiles)
    UGT, DNT, R13, R2 = c["ug_t"], c["dn_t"], c["w13_rows"], c["w2_rows"]
    w_addr = use_dn.select(a["w2"] + fx.Int64(e_d) * (R2 * W2_ROW_BYTES), a["w13"] + fx.Int64(e_u) * (R13 * W13_ROW_BYTES))  # fmt: skip
    s_addr = use_dn.select(a["s2"] + fx.Int64(e_d) * (R2 * W2_SCALE_COLS), a["s13"] + fx.Int64(e_u) * (R13 * W13_SCALE_COLS))  # fmt: skip
    nb = live.select(fx.Int32(0x7FFFFFFF), fx.Int32(0))
    r_w, r_s = rsrc(w_addr, nb), rsrc(s_addr, nb)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c,
                r_w,
                r_s,
                use_dn.select(
                    (dq % DNT) * 32 + t * 16 + l16, (q % UGT) * 32 + t * 16 + l16
                ),
                UG_CHUNKS,
            )
        ]  # fmt: skip
        for t in range(2)
    ]


def or_pair(x, y):
    """Two ``ug_first_loads`` shares (one of them zeros) as one pair's loads."""
    return [
        [
            [
                fx.Vector.from_elements(
                    [fx.Int32(u[i]) | fx.Int32(v[i]) for i in range(4)], fx.Int32
                )
                for u, v in zip(px, py)
            ]
            for px, py in zip(tx, ty)
        ]  # fmt: skip
        for tx, ty in zip(x, y)
    ]


@flyc.jit
def stage_plan_tpx(c, ids_mb=None, flags=None):
    """TP experts' plan (every rank alike): wave 0 walks the picks in rounds of 64, a
    pick's column its rank among its expert's picks in pick order (the round's
    same-expert lanes from 8 ballots, the earlier rounds' count in LDS), then lays out
    the experts' starts in the sorted pick list (``pick_of``) and their column tiles
    of 16 in expert order. Returns the tiles. ``ids_mb`` (a mailbox region): the
    picks polled from the route tasks' tagged copy (no flag wait first). ``flags``
    (``poll_flags`` regions): polled meanwhile by wave 1 .., waited for at the plan's
    closing barrier."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    sorted_, cnt_of, start_of, tile_of, nt = (
        c["pick_of"],
        c["cnt_of"],
        c["start_of"],
        c["tile_of"],
        c["nt"],
    )
    a = c["args"]
    r_ids = rsrc(a["ids"])
    PICKS = S * TOP_K
    ROUNDS = cdiv(PICKS, 64)
    EPL = EXPERTS // 64  # experts a lane: 4
    if tid < EXPERTS:
        fx.ptr_store(fx.Int32(0), cnt_of + tid)
    gpu.barrier()
    if const_expr(flags is not None):
        poll_flags(c, flags, first=64, sync=False)
    if wave == 0:
        if const_expr(ids_mb is not None):
            got = c["poll"](
                [(ids_mb, fx.min(rr * 64 + lane, PICKS - 1), 1) for rr in range(ROUNDS)]
            )
            raw = [fx.Int32(got[rr][0]) for rr in range(ROUNDS)]
        else:
            raw = [
                fx.Int32(
                    bo.buffer_load(
                        r_ids,
                        fx.min(rr * 64 + lane, PICKS - 1),
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )  # fmt: skip
                for rr in range(ROUNDS)
            ]
        ids = [
            fx.max(fx.min(raw[rr], fx.Int32(EXPERTS - 1)), fx.Int32(0))
            for rr in range(ROUNDS)
        ]
        if const_expr(ROUNDS == 1 and S <= TILE_COLS):
            # one round, one tile an expert (S <= TILE_COLS picks of it at most): every value of the
            # plan from 8 ballots and popcounts, no LDS round trip. Over the expert id's bits, MSB
            # first: ``eq`` the lanes with my expert, ``less`` those with a smaller one; my column =
            # my same-expert lanes below me (pick order), my expert's start = the picks of smaller
            # experts, its tile = the smaller experts picked (their first lanes) - the values the
            # general path below computes through ``cnt_of`` / the prefix sums / ``start_of``
            e = ids[0]
            valid = lane < PICKS
            eq = ballot(valid)
            less = fx.Int64(0)
            for bit in range_constexpr(7, -1, -1):
                on = fx.Int64((e >> bit) & 1)
                sel = ballot(((e >> bit) & 1) == 1) ^ (
                    on - fx.Int64(1)
                )  # the lanes with my bit
                less = less | (
                    eq & (sel ^ fx.Int64(-1)) & (fx.Int64(0) - on)
                )  # mine 1, theirs 0
                eq = eq & sel
            below = lanes_before(eq, lane)
            start = popcount(less)
            lead = valid & (below == 0)
            leads = ballot(lead)
            if lane == 0:
                fx.ptr_store(popcount(leads), nt)
            if lead:
                fx.ptr_store(popcount(eq), cnt_of + e)
                fx.ptr_store(start, start_of + e)
                fx.ptr_store(e * COLS, tile_of + popcount(less & leads))
            if valid:
                fx.ptr_store(lane, sorted_ + (start + below))
        else:
            cols = []
            for rr in range_constexpr(ROUNDS):
                # a lane's same-expert lanes of the round by the expert id's 8 bits (a
                # ballot a bit: a fixed cost, where a turn a distinct expert took 6.8 us
                # at 64 picks); its column = the expert's count of the earlier rounds +
                # its same-expert lanes below it (pick order, alike on every rank); the
                # round's lowest lane of an expert moves the expert's count
                e = ids[rr]
                valid = rr * 64 + lane < PICKS
                m = ballot(valid)
                for bit in range_constexpr(8):
                    on = (e >> bit) & 1
                    m = m & (
                        ballot(on == 1) ^ (fx.Int64(on) - fx.Int64(1))
                    )  # its own bit's lanes
                below = lanes_before(m, lane)
                # (the first round's counts are the zeros stored above: no LDS round trip)
                base = fx.Int32(0) if rr == 0 else fx.ptr_load(cnt_of + e)
                cols.append(base + below)
                if valid & (below == 0):
                    fx.ptr_store(base + popcount(m), cnt_of + e)
            # the experts' starts and tiles: EPL experts a lane, an exclusive prefix over the lanes
            cn = [fx.ptr_load(cnt_of + (lane * EPL + k)) for k in range(EPL)]
            tl = [
                (fx.min(cn[k], fx.Int32(COLS)) + TILE_COLS - 1) // TILE_COLS
                for k in range(EPL)
            ]
            tot, tsum = cn[0], tl[0]
            for k in range_constexpr(1, EPL):
                tot = tot + cn[k]
                tsum = tsum + tl[k]
            incl = lane_prefix(tot, lane)
            tincl = lane_prefix(tsum, lane)
            n_tiles = read_lane_i32(tincl, 63)
            if lane == 0:
                # an empty plan's first tile is expert 0, column 0 (a later real tile 0 wins)
                fx.ptr_store(fx.Int32(0), tile_of)
                fx.ptr_store(n_tiles, nt)
            run, tile_run = incl - tot, tincl - tsum
            for k in range_constexpr(EPL):
                ek = lane * EPL + k
                fx.ptr_store(run, start_of + ek)
                for j in range_constexpr(TILES_PER_EXPERT):
                    if j < tl[k]:
                        fx.ptr_store(
                            ek * COLS + j * TILE_COLS, tile_of + (tile_run + j)
                        )
                run = run + cn[k]
                tile_run = tile_run + tl[k]
            # the sorted picks (a wave's LDS operations run in order: the starts are in)
            for rr in range_constexpr(ROUNDS):
                p = rr * 64 + lane
                if p < PICKS:
                    fx.ptr_store(
                        p, sorted_ + (fx.ptr_load(start_of + ids[rr]) + cols[rr])
                    )
    gpu.barrier()
    if const_expr(flags is not None):
        acquire_loads()
    return uniform(fx.ptr_load(nt))


# ---------------------------------------------------------------- 1 + 2. the experts
@flyc.jit
def stage_xq_cols(c, e, col0, n, xv=None):
    """Expert e's columns col0 .. col0 + n as MXFP4 blocks into LDS: column j's
    words at xcol[j XQW ..], its scales at xcol[TILE_COLS XQW + j XSW ..]; a
    32-value block is four lanes' (``xv``: the loads, already in flight)."""
    S, tid, xcol = c["S"], c["tid"], c["xcol"]
    xq_l = xcol
    xs_l = xcol + TILE_COLS * XQW
    if const_expr(xv is None):
        xv = xq_loads(c, e, col0, n)
    for i in range_constexpr(tile_cols(S)):
        v = tid + THREADS * i
        vb = xv[i].bitcast(fx.BFloat16).to(fx.Float32)
        vals = [fx.Float32(vb[k]) for k in range(8)]
        amax = abs(vals[0])
        for k in range_constexpr(1, 8):
            amax = fx.max(amax, abs(vals[k]))
        amax = butterfly(amax, (1, 2), fx.max)
        byte, qs = mxfp4_scale(amax)
        word = e2m1_code(vals[0], qs)
        for k in range_constexpr(1, 8):
            word = word | (e2m1_code(vals[k], qs) << (4 * k))
        if v < n * XQW:
            fx.ptr_store(word.bitcast(fx.Float32), xq_l + v)
            if (v % 4) == 0:
                fx.ptr_store(byte.bitcast(fx.Float32), xs_l + v // 4)
    gpu.barrier()


@flyc.jit
def stage_mid_q(c, e, col0, n, more=()):
    """Expert e's columns col0 .. col0 + n's mid blocks (MXFP4, from the ug items)
    into LDS in ``stage_xq_cols``' layout; every round polled in one spin (a
    column past n polls the last one's, its store skipped). ``more``: further tiles
    (e, col0, n, first LDS column) staged after it."""
    S, tid, xcol, poll, mid = c["S"], c["tid"], c["xcol"], c["poll"], c["mid"]
    if const_expr(c["tpx"]):
        mid = c[
            "own_midx"
        ]  # every rank's blocks of the pick, pushed here (UG_T in all)
    xq_l = xcol
    xs_l = xcol + TILE_COLS * XQW
    XR = cdiv(tile_cols(S) * UG_T, THREADS)
    tiles = [(e, col0, n, 0)] + list(more)

    def specs(e, col0, n, i):
        v = tid + THREADS * i  # block v % UG_T of column v / UG_T
        j = fx.min(v // UG_T, n - 1)
        base = (pick_at(c, e, col0 + j) * UG_T + v % UG_T) * MID_W
        return [(mid, base, 2), (mid, base + 2, 2), (mid, base + 4, 1)]

    got = poll([sp for (e, col0, n, _) in tiles for i in range(XR) for sp in specs(e, col0, n, i)], batch=3 * XR)  # a spin a tile  # fmt: skip
    for ti, (e, col0, n, cb) in enumerate(tiles):
        for i in range_constexpr(XR):
            v = tid + THREADS * i
            g = 3 * (ti * XR + i)
            if v < n * UG_T:
                words = [got[g][0], got[g][1], got[g + 1][0], got[g + 1][1]]
                fx.ptr_store(
                    fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Float32),
                    xq_l + ((cb + v // UG_T) * XQW + (v % UG_T) * 4),
                )
                fx.ptr_store(got[g + 2][0].bitcast(fx.Float32), xs_l + (cb * XSW + v))
    gpu.barrier()


@flyc.jit
def put_mid_block(c, blk, e, col0, n, mid_of):
    """Each column's 32 mids of block ``blk`` as one MXFP4 block to ``k3_mid``:
    thread (column tid / 32, row tid % 32), the block's lanes one half wave; the
    SwiGLU value rounded to bf16 first (the compact-A4 epilogue)."""
    tid, put, mid = c["tid"], c["put"], c["mid"]
    j = tid // 32
    r = tid % 32
    act = bf16_round(mid_of(r, fx.min(j, n - 1)))
    amax = butterfly(abs(act), (1, 2, 4, 8, 16), fx.max)
    byte, qs = mxfp4_scale(amax)
    word = butterfly(e2m1_code(act, qs) << (4 * (r % 8)), (1, 2, 4), bor)
    if const_expr(c["tpx"]):
        # the rank's block of the whole intermediate row, to every rank
        if (j < n) & ((r % 8) == 0):
            base = (
                pick_at(c, e, col0 + j) * UG_T + c["rank"] * c["ug_t"] + blk
            ) * MID_W
            for pe in range_constexpr(c["W"]):
                put(c["peer_midx"][pe], base + r // 8, word)
                if r == 0:
                    put(c["peer_midx"][pe], base + 4, byte)
    else:
        if (j < n) & ((r % 8) == 0):
            base = (pick_at(c, e, col0 + j) * UG_T + blk) * MID_W
            put(mid, base + r // 8, word)
            if r == 0:
                put(mid, base + 4, byte)


UG_COST, DN_COST = 5, 3  # a ug item's and a dn item's relative time (``dn_caps``)


def ug_balanced(c):
    """The ug items round-robin over the CTAs (item ``bid + BLOCKS k``) and the dn
    ranges weighted by each CTA's ug load (``dn_range``): with the x of every token
    staged once (``xq_pre``) a tile change costs nothing, the tiles' mids complete in
    rounds, and the CTAs that run one ug item fewer take more dn items, of the tiles
    whose mids are already in."""
    return bool(c["xq_pre"])


def ug_count(c, n_tiles):
    """This CTA's ug items: (item of the CTA's k-th, count)."""
    bid, UGT = c["bid"], c["ug_t"]
    TU = n_tiles * UGT
    if ug_balanced(c):
        n = (fx.max(TU - bid, fx.Int32(0)) + (BLOCKS - 1)) // BLOCKS
        return (lambda k: bid + BLOCKS * k), n
    q0 = (bid * TU) // BLOCKS
    return (lambda k: q0 + k), ((bid + 1) * TU) // BLOCKS - q0


def dn_caps(n_tiles, ug_t, dn_t):
    """The balanced dn shares (``ug_balanced``) of a plan of ``n_tiles``: (light
    CTAs, a light CTA's dn items, a heavy CTA's). The ``h`` heavy CTAs run one ug
    item more (``ug_count``); a CTA ends after its ug items (``UG_COST`` each) and
    its dn items (``DN_COST``), so the shares are the whole items that end every CTA
    by the earliest time E with enough items: E sits on a light CTA's dn step
    (family 1: a heavy CTA takes ceil(U / D) fewer) or a heavy CTA's (family 2: a
    light CTA takes floor(U / D) more). Plain Python ints or traced Int32."""
    U, D = UG_COST, DN_COST
    fu, cu = U // D, -(-U // D)
    TU, TD = n_tiles * ug_t, n_tiles * dn_t
    R, h = TU // BLOCKS, TU % BLOCKS
    na = BLOCKS - h
    if isinstance(TD, int):
        mx = max
    else:

        def mx(x, y):
            return fx.max(x, fx.Int32(y))

    def sel(p, x, y):
        return (x if p else y) if isinstance(p, bool) else p.select(x, y)

    # family 1: (m, m - cu) items, E = U R + D m
    m1 = sel(na * cu >= TD, (TD + na - 1) // na, (TD + h * cu + BLOCKS - 1) // BLOCKS)
    e1 = U * R + D * m1
    # family 2: (m + fu, m) items, E = U (R + 1) + D m
    m2 = (mx(TD - na * fu, 0) + BLOCKS - 1) // BLOCKS
    e2 = U * (R + 1) + D * m2
    one = e1 <= e2
    cap_a = sel(one, m1, m2 + fu)
    cap_b = sel(one, mx(m1 - cu, 0), m2)
    return na, cap_a, cap_b, h


def dn_range(c, n_tiles):
    """This CTA's dn items: a contiguous tile-major range (first, count). Balanced
    (``ug_balanced``): the light CTAs first (bid h .., one ug item fewer: they take
    the leading tiles, whose ug items ran in the full rounds), then the heavy ones
    (bid 0 .. h), each taking its share (``dn_caps``) until the items run out."""
    bid, UGT, DNT = c["bid"], c["ug_t"], c["dn_t"]
    TD = n_tiles * DNT
    if not ug_balanced(c):
        q0 = (bid * TD) // BLOCKS
        return q0, ((bid + 1) * TD) // BLOCKS - q0
    na, cap_a, cap_b, h = dn_caps(n_tiles, UGT, DNT)

    def start(j):
        return fx.min(fx.min(j, na) * cap_a + fx.max(j - na, fx.Int32(0)) * cap_b, TD)

    j = (bid >= h).select(bid - h, bid + na)
    d0 = start(j)
    return d0, start(j + 1) - d0


@flyc.jit
def stage_ug(c, n_tiles, xin_wait=None, first=None):
    """This CTA's ug items (tile, 32-row block; ``ug_count``), each two pairs (gate
    rows, then up rows): the tile's x restaged at a tile boundary, the next pair's
    weights issued before this pair's MFMAs, and after the up pair the SwiGLU mids
    of the 32 rows leave as MXFP4 blocks. The reduction rows alternate between
    items: one barrier an item. ``xin_wait``: the expert input's flags, waited for
    first (``run_k3`` polls them before the plan)."""
    S, bid, red = c["S"], c["bid"], c["red"]
    UGT, DNT = c["ug_t"], c["dn_t"]  # items a tile (TP experts: the rank's slices)
    q_of, n_ug = ug_count(c, n_tiles)
    q0 = q_of(0)
    n_items = 2 * n_ug
    t0 = q0 // UGT
    e0, c0, n0 = tile_info(c, t0, n_tiles)
    pre = c["xq_pre"]
    # the dn stage's first item: its weights go out as the loop's last prefetch (a
    # duplicate of the last pair went out there before; a CTA without ug items
    # issues them first thing), so they land while the dn stage waits for its mids
    dq0, _ = dn_range(c, n_tiles)
    # (merged launch, ``early_w``) the first pair's weights go out at the plan, before
    # the expert input's wait, on every wave but the one polling the expert input's
    # flags (a poll returns in order behind its wave's loads); that wave issues its share
    # once the flags are in. They stream while xn finishes; the x blocks follow them
    # (needed only for the first MFMAs, which wait for the weights anyway). Otherwise
    # the first tile's x goes out before the first weights: loads complete in order,
    # so x lands (and is quantized) while the weights stream
    early_w = xin_wait is not None and pre
    use_dn0 = n_items == 0
    # (``xq_dma``) every wave's share at the plan, the poll wave's too: its xin poll then
    # returns once that share landed, and it DMAs the staged blocks to LDS after the
    # flags, behind nothing else; the other waves stage no x (their loads would return
    # behind their weights)
    dma = early_w and xq_dma_ok(S)
    cur = None
    if const_expr(first is not None):
        cur = first  # (issued before the plan: ``first_pair_mask``)
    elif const_expr(dma):
        cur = ug_or_dn_pair_loads(c, q0, 0, dq0, n_tiles, use_dn0)
    elif const_expr(early_w):
        cur = ug_first_loads(c, q0, dq0, n_tiles, use_dn0, c["wave"] != POLL_WAVE)
    if const_expr(xin_wait is not None):
        rocdl.sched_barrier(0)  # (the first pair's loads stay ahead of the wait)
        xin_wait(dma)
        stamp(c, 14)  # the expert input arrived
    if const_expr(not dma):
        if const_expr(early_w):
            late = ug_first_loads(c, q0, dq0, n_tiles, use_dn0, c["wave"] == POLL_WAVE)
        xv0 = xq_all_loads(c) if pre else xq_loads(c, e0, c0, n0)
        if const_expr(early_w):
            cur = or_pair(cur, late)
        else:
            cur = ug_or_dn_pair_loads(c, q0, 0, dq0, n_tiles, use_dn0)
        if const_expr(pre):
            stage_xq_all(
                c, xv0
            )  # every token's blocks: a tile change is only the lanes' tokens
        else:
            stage_xq_cols(c, e0, c0, n0, xv0)
    cur_t = t0
    for i in range(0, n_items):
        q = q_of(i // 2)
        pr = i % 2
        ti = q // UGT
        e, col0, n = tile_info(c, ti, n_tiles)
        if const_expr(not pre):
            if ti != cur_t:
                gpu.barrier()  # every thread is done with the previous tile's columns
                stage_xq_cols(c, e, col0, n)
                cur_t = ti
        i1 = fx.min(i + 1, n_items - 1)
        nxt = ug_or_dn_pair_loads(
            c, q_of(i1 // 2), i1 % 2, dq0, n_tiles, i + 1 >= n_items
        )
        if const_expr(pre):
            tok = lane_token(c, e, col0, n)
            xb = [xq_b_tok(c, tok, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        else:
            xb = [xq_b(c, n, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        par = (i // 2) % 2
        pbase = par * (4 * RED_SLOT_FLOATS)
        for t in range_constexpr(2):
            tile_store(c, 4 * par + 2 * pr + t, tile_mfmas_a4(cur[t][0], cur[t][1], xb))
        if pr == 1:
            gpu.barrier()

            def mid_of(row, j):
                gate = row_sum(red + pbase + (row // 16) * RED_SLOT_FLOATS, row % 16, j)
                up = row_sum(
                    red + pbase + (2 + row // 16) * RED_SLOT_FLOATS, row % 16, j
                )
                return gate * sigmoid(gate) * up

            put_mid_block(c, q % UGT, e, col0, n, mid_of)
        cur = nxt
    return cur  # the dn stage's first item's weights, in flight


@flyc.jit
def stage_ug_dyn(c, n_tiles, xin_wait=None):
    """The ug items from a queue (merged launch, pre-quantized x): CTA bid's first
    item is ``bid``, every later one ``BLOCKS`` + a ticket of the launch's queue
    word (``k3_ugq``, zeroed at the launch's entry by CTA 0 before its xn task's
    flag, which every CTA polls first), so a CTA that finishes early takes more
    (a fixed range left the CTAs ~11 us apart at S = 16). A ticket goes out at an
    item's head, before its up pair's weights (it returns ahead of them), and
    reaches the CTA through LDS (``misc``, a slot an item parity) behind the
    item's barrier. The first item's gate pair goes out before the expert input's
    wait (``xin_wait``: the plan ran on the routing alone), so its weights stream
    while xn finishes. Returns the dn stage's first item's weights, in flight."""
    S, bid, red, tid, misc = c["S"], c["bid"], c["red"], c["tid"], c["misc"]
    UGT, DNT = c["ug_t"], c["dn_t"]
    TU = n_tiles * UGT
    TD = n_tiles * DNT
    dq0 = (bid * TD) // BLOCKS
    ctr = c["ugq"].value
    no = fx.Int32(0) == fx.Int32(1)
    q = bid
    cur = ug_or_dn_pair_loads(
        c, q, 0, dq0, n_tiles, q >= TU
    )  # (a CTA without an item: dn's first)
    rocdl.sched_barrier(0)  # (the first pair's loads stay ahead of the wait)
    if const_expr(xin_wait is not None):
        xin_wait()
        stamp(c, 14)  # the expert input arrived
    tk = fx.Int32(0)
    if tid == 0:
        # the second item's (after the expert input's wait: the queue word is zeroed
        # before the flags)
        tk = global_atomic(ctr, 1)
    xv = xq_all_loads(c)
    stage_xq_all(c, xv)
    if tid == 0:
        fx.ptr_store(tk.bitcast(fx.Float32), misc + 2)
    gpu.barrier()
    qn = fx.Int32(BLOCKS) + fx.ptr_load(misc + 2).bitcast(fx.Int32)
    k = fx.Int32(0)
    while q < TU:
        par = k % 2
        t2 = fx.Int32(0)
        if tid == 0:
            t2 = global_atomic(ctr, 1)  # the item after next
        e, col0, n = tile_info(c, q // UGT, n_tiles)
        tok = lane_token(c, e, col0, n)
        xb = [xq_b_tok(c, tok, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        up = ug_or_dn_pair_loads(c, q, 1, dq0, n_tiles, no)
        pbase = par * (4 * RED_SLOT_FLOATS)
        for t in range_constexpr(2):
            tile_store(c, 4 * par + t, tile_mfmas_a4(cur[t][0], cur[t][1], xb))
        if tid == 0:
            fx.ptr_store(t2.bitcast(fx.Float32), misc + par)
        nxt = ug_or_dn_pair_loads(
            c, qn, 0, dq0, n_tiles, qn >= TU
        )  # the next item's gate pair, or dn's first
        for t in range_constexpr(2):
            tile_store(c, 4 * par + 2 + t, tile_mfmas_a4(up[t][0], up[t][1], xb))
        gpu.barrier()
        q_after = fx.Int32(BLOCKS) + fx.ptr_load(misc + par).bitcast(fx.Int32)

        def mid_of(row, j):
            gate = row_sum(red + pbase + (row // 16) * RED_SLOT_FLOATS, row % 16, j)
            upv = row_sum(red + pbase + (2 + row // 16) * RED_SLOT_FLOATS, row % 16, j)
            return gate * sigmoid(gate) * upv

        put_mid_block(c, q % UGT, e, col0, n, mid_of)
        q = qn
        qn = q_after
        cur = nxt
        k = k + 1
    return cur


@flyc.jit
def stage_dn(c, n_tiles, first=None):
    """This CTA's contiguous range of the dn items (tile, 32-row block), streamed
    like the ug pairs: the next item's weights issued before this item's MFMAs, the
    tile's mid blocks restaged at a tile boundary, and the items taken in groups of
    two of one tile (the range's last item, or the last of a tile, alone) that
    share one barrier and one epilogue (waves 0 .. 3 the first item's rows, 4 .. 7
    the second's), each item's 32 y rows of its columns to ``k3_y``. ``first``: the
    first item's weights, in flight (the ug stage's last prefetch). Each item's sum
    of squares is its four waves' partials in wave order, published as before."""
    S, bid, red = c["S"], c["bid"], c["red"]
    DNT = c["dn_t"]
    if const_expr(c["ug_dyn"]):
        TD = n_tiles * DNT
        q0 = (bid * TD) // BLOCKS
        n_items = ((bid + 1) * TD) // BLOCKS - q0
    else:
        q0, n_items = dn_range(c, n_tiles)
    cur = (
        first if first is not None else dn_pair_loads(c, q0, n_tiles)
    )  # (a trace-time choice)
    cur_t = fx.Int32(-1)
    # the finalize's sum of squares (v = bf16(bf16(y) pen_w), the fin stage's values)
    # of each item's 32 rows of each column: a wave partial a column in LDS (two
    # parities), summed and published a group later, behind that group's barrier
    tid, lane, wave, misc, put = c["tid"], c["lane"], c["wave"], c["misc"], c["put"]
    r_pw = rsrc(c["args"]["pen_w"])
    j16, rp = (
        tid % TILE_COLS,
        tid // TILE_COLS,
    )  # column, row pair of the group (0 .. 31)
    hp = tid // TILE_COLS  # a publishing thread's item of its group (threads 0 .. 31)
    HW = WAVES // 2  # waves an item's rows
    prev_q, prev_e, prev_c0, prev_n, prev_k = (
        fx.Int32(-1),
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(0),
    )

    def publish_ss(par, pq, pe, pc0, pn, pk):
        # thread (item hp < pk of the group, column j16 < pn): the item's four waves'
        # partials in wave order
        if (pq >= 0) & (hp < pk) & (j16 < pn):
            w0 = hp * HW
            tot = fx.ptr_load(misc + (par * (WAVES * TILE_COLS) + w0 * TILE_COLS + j16))
            for w in range_constexpr(1, HW):
                tot = tot + fx.ptr_load(
                    misc + (par * (WAVES * TILE_COLS) + (w0 + w) * TILE_COLS + j16)
                )
            qq = pq + hp
            if const_expr(c["tpx"]):
                # the item's part of the pick's sum, to every rank (DN_T parts in all)
                at = pick_at(c, pe, pc0 + j16) * DN_T + c["rank"] * DNT + qq % DNT
                for pr_ in range_constexpr(c["W"]):
                    put(c["peer_ssx"][pr_], at, tot)
            else:
                put(c["ss"], pick_at(c, pe, pc0 + j16) * DN_T + qq % DN_T, tot)

    g = fx.Int32(0)  # this item's place in its group
    gk = fx.Int32(1)  # the group's items
    gq = q0  # the group's first item
    gpar = fx.Int32(0)  # the group's reduction rows' parity
    pw_word = fx.Int32(0)
    for i in range(0, n_items):
        q = q0 + i
        ti = q // DNT
        e, col0, n = tile_info(c, ti, n_tiles)
        if g == 0:
            # a group: this item and the next one of the same tile
            gk = ((i + 1 < n_items) & ((q + 1) // DNT == ti)).select(
                fx.Int32(2), fx.Int32(1)
            )
            gq = q
            # the post norm's weights of the group's rows go out ahead of the next
            # item's weights: behind them the group's epilogue would wait for those
            c0 = (q % DNT) * 32 + 2 * fx.min(rp, 16 * gk - 1)
            if const_expr(c["tpx"]):
                c0 = c0 + c["rank"] * c["w2_rows"]  # the rank's output columns
            pw_word = fx.Int32(bo.buffer_load(r_pw, c0 // 2, vec_width=1, dtype=T.i32))
        # the next item's weights stream while a tile's mids are polled (after the
        # last item: none, no traffic ahead of the fin / unpool loads)
        nxt = dn_pair_loads(
            c, q0 + fx.min(i + 1, n_items - 1), n_tiles, null=i + 1 >= n_items
        )
        if ti != cur_t:
            gpu.barrier()  # every thread is done with the previous tile's columns
            stage_mid_q(c, e, col0, n)
            # the poll's loads are in (behind the weights in flight): said here, the
            # compiler drains no load for the poll's registers on the path without a
            # tile change (it drained the next item's before this item's MFMAs)
            rocdl.s_waitcnt(vmcnt=0)
            stamp(c, 12)  # a tile's mids arrived (the CTA's last tile's survives)
            cur_t = ti
        xb = [xq_b(c, n, DN_CHUNKS, kc) for kc in range(DN_CHUNKS)]
        pbase = gpar * (4 * RED_SLOT_FLOATS)
        for t in range_constexpr(2):
            tile_store(c, 4 * gpar + 2 * g + t, tile_mfmas_a4(cur[t][0], cur[t][1], xb))
        if g == gk - 1:
            gpu.barrier()
            publish_ss(
                1 - gpar, prev_q, prev_e, prev_c0, prev_n, prev_k
            )  # the previous group's

            def y_of(row, j):
                return row_sum(red + pbase + (row // 16) * RED_SLOT_FLOATS, row % 16, j)

            rr = 2 * fx.min(
                rp, 16 * gk - 1
            )  # (row pairs past the group: its last, no store)
            ya, yb = y_of(rr, j16), y_of(rr + 1, j16)
            live = (j16 < n) & (rp < 16 * gk)
            if const_expr(not c["tpx"]):
                if live:
                    c["put_bf"](
                        c["y"],
                        pick_at(c, e, col0 + j16) * MOE_HIDDEN
                        + (gq % DN_T) * 32
                        + 2 * rp,
                        [ya, yb],
                    )
            c0 = (gq % DNT) * 32 + rr
            if const_expr(c["tpx"]):
                c0 = c0 + c["rank"] * c["w2_rows"]
            pw0, pw1 = bf2_f32(pw_word)
            v0 = bf16_round(bf16_round(ya) * pw0)
            v1 = bf16_round(bf16_round(yb) * pw1)
            if const_expr(c["tpx"]):
                # the finalized values v = bf16(y pen_w) to every rank: each finalizes the whole row
                if live:
                    at = pick_at(c, e, col0 + j16) * (MOE_HIDDEN // 2) + c0 // 2
                    for pr_ in range_constexpr(c["W"]):
                        c["put_bf"](c["peer_vx"][pr_], 2 * at, [v0, v1])
            ss = live.select(v0 * v0 + v1 * v1, fx.Float32(0.0))
            ss = butterfly(ss, (16, 32))  # the wave's four row pairs of column j16
            if lane < TILE_COLS:
                fx.ptr_store(
                    ss, misc + (gpar * (WAVES * TILE_COLS) + wave * TILE_COLS + lane)
                )
            prev_q, prev_e, prev_c0, prev_n, prev_k = gq, e, col0, n, gk
            gpar = 1 - gpar
            g = fx.Int32(0)
        else:
            g = fx.Int32(1)
        cur = nxt
    gpu.barrier()  # ``red`` is reused by the fin and unpool stages; the last group's partials are in
    publish_ss(1 - gpar, prev_q, prev_e, prev_c0, prev_n, prev_k)  # the last group's


# ---------------------------------------------------------------- 3. fin + all-reduce
# The all-reduce's partials (``ffn``) are zero-sentinel words, not tagged pairs: a
# (source rank, token, column pair)'s bf16 pair as one word, stored complemented
# (``zs_enc``: never 0, a word that was all ones becomes ``sent_word``'s NaN pair
# first), so an unwritten word reads 0 -- the peer buffer starts zeroed and every
# word is read and re-armed (0) by its one reader in the launch that pushed it, the
# next push to it coming from a later launch. Half a tagged pair's bytes: the
# heaviest rank's push (S x MOE_HIDDEN x 2 B to every peer) is link-bound.
def zs_enc(w):
    """The zero-sentinel word of bf16-pair word ``w``."""
    return sent_word(w) ^ fx.Int32(-1)


def zs_word(src, t, col):
    """The word of (source rank, token t, column pair at ``col``) in ``ffn``."""
    return ((src * MAX_TOKENS + t) * MOE_HIDDEN + col) // 2


def poll_zs(base, offs, n, gap=1):
    """Words ``off`` .. + ``n`` (n <= 4) of the zero-sentinel region at ``base``
    (system scope) for every ``off`` in ``offs``, once none is 0, re-polling only the
    specs still pending; [[n words decoded], ...] (the caller re-arms them)."""
    r = rsrc(base, MASK_OOB)
    N = len(offs)

    def load(prev):
        words = []
        for e in range_constexpr(N):
            off = fx.Int32(offs[e])
            if const_expr(prev is not None):
                d = prev[n * e] != fx.Int32(0)
                for q in range_constexpr(1, n):
                    d = d & (prev[n * e + q] != fx.Int32(0))
                off = d.select(fx.Int32(MASK_OOB // 4), off)
            w = bo.buffer_load(r, off, vec_width=n, dtype=T.i32, cache_modifier=CM_SYS)
            w = (
                [fx.Int32(w)]
                if n == 1
                else [fx.Int32(fx.Vector(w)[q]) for q in range(n)]
            )
            for q in range_constexpr(n):
                if const_expr(prev is None):
                    words.append(w[q])
                else:
                    words.append(d.select(prev[n * e + q], w[q]))
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[0] == fx.Int32(0)
        for e in range_constexpr(1, n * N):
            bad = bad | (v[e] == fx.Int32(0))
        return bad

    v = _spin_masked(load, pending, gap)
    return [[v[n * e + q] ^ fx.Int32(-1) for q in range(n)] for e in range(N)]


def zs_store(base, off, words):
    """Words ``off`` .. of a zero-sentinel region := ``words`` (one store, system scope)."""
    if len(words) == 1:
        bo.buffer_store(fx.Int32(words[0]), rsrc(base), off, cache_modifier=CM_SYS)
    else:
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(base),
            off,
            cache_modifier=CM_SYS,
        )


@flyc.jit
def fin_held(c, t):
    """Token t's slots this rank holds, as ``pick`` gives them (>= 0: held, -1: not),
    from the plan still in LDS (``pick_of`` / ``cnt_of``): a load of the ids here
    returned ~1 us after the CTA's last dn item, and the fin poll's addresses wait
    on it. Thread i scans local expert i / 16's columns 4 (i % 16) .. + 4; the
    slots found are OR-ed over the CTA (``sums``, free outside the boundary)."""
    tid, lane, wave, sums = c["tid"], c["lane"], c["wave"], c["sums"]
    assert LOCAL_EXPERTS * COLS == 4 * THREADS
    e = tid // 16
    j0 = (tid % 16) * 4
    cnt = fx.ptr_load(c["cnt_of"] + e)
    m = fx.Int32(0)
    for q in range_constexpr(4):
        p = fx.ptr_load(c["pick_of"] + (e * COLS + j0 + q))
        hit = (j0 + q < cnt) & ((p >> 3) == t)
        m = m | hit.select(fx.Int32(1) << (p & (TOP_K - 1)), fx.Int32(0))
    m = butterfly(m, (32, 16, 8, 4, 2, 1), bor)
    if lane == 0:
        fx.ptr_store(m.bitcast(fx.Float32), sums + wave)
    gpu.barrier()
    tot = fx.ptr_load(sums).bitcast(fx.Int32)
    for w in range_constexpr(1, WAVES):
        tot = tot | fx.ptr_load(sums + w).bitcast(fx.Int32)
    return [
        (((tot >> sl) & 1) != 0).select(fx.Int32(0), fx.Int32(-1))
        for sl in range(TOP_K)
    ]


@flyc.jit
def stage_fin(c, t, part):
    """fin task (token t, part): the part's columns of the token's slots (two or
    four elements a thread), the finalize (v = bf16(y pen_w), each slot's sum of
    squares over its whole row summed from the dn items' publishes in ``k3_ss``,
    score / rms weighted sum in slot order), pushed as zero-sentinel bf16-pair words
    (``zs_enc``) to every rank's ``ffn`` region, then the same columns of every rank's
    partial summed in rank order -> ``k3_fbuf`` (plain bf16) and the (token, part)
    flag. A token none of whose picks this rank holds pushes zeros and polls nothing.
    The token's held slots come from the plan in LDS (``fin_held``); the push and the
    partials' poll run on separate waves (``split``)."""
    S, W, tid, lane, wave, red = (
        c["S"],
        c["W"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    put, put_bf, poll = c["put"], c["put_bf"], c["poll"]
    y_mb, ss_mb, fbuf, own_ffn = c["y"], c["ss"], c["fbuf"], c["own_ffn"]
    a = c["args"]
    rank = c["rank"]
    AP = c["ar_parts"]
    AR_COLS = MOE_HIDDEN // AP
    # (``split``) the first half of the threads computes and pushes the partial, the
    # second half polls the ranks' partials of the same columns: a poll on a wave that
    # pushed returns behind the pushes' acks (remote, ~1.6-1.9 us under the AR's egress)
    HALF = THREADS // 2
    split = cdiv(AR_COLS // 2, HALF) <= 2
    PPT = cdiv(AR_COLS // 2, HALF) if split else fin_ppt(AP)  # column pairs a thread
    NV = 2 * PPT
    me = tid % HALF if split else tid
    act = NV * me < AR_COLS
    pusher = act & (tid < HALF) if split else act
    poller = act & (tid >= HALF) if split else act
    col = part * AR_COLS + NV * fx.min(
        me, AR_COLS // NV - 1
    )  # an idle thread loads the last columns
    # the post norm's weight of these columns (bf16 pairs)
    pwv = bo.buffer_load(rsrc(a["pen_w"]), col // 2, vec_width=PPT, dtype=T.i32)
    if const_expr(PPT == 1):
        pw_words = [fx.Int32(pwv)]
    else:
        pwv = fx.Vector(pwv)
        pw_words = [pwv[q] for q in range(PPT)]
    pw = [bf2_f32(w)[h] for w in pw_words for h in range(2)]
    es = fin_held(c, t)
    scs = [
        fx.Float32(
            bo.buffer_load(
                rsrc(a["scores"]),
                t * TOP_K + sl,
                vec_width=1,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        for sl in range(TOP_K)
    ]
    # the first slot this rank holds (a slot it does not hold polls that one's row)
    fb = fx.Int32(-1)
    for sl in range_constexpr(TOP_K):
        fb = ((fb < 0) & (es[sl] >= 0)).select(fx.Int32(sl), fb)
    a0 = fx.Float32(0.0)
    a1 = fx.Float32(0.0)
    a2 = fx.Float32(0.0)
    a3 = fx.Float32(0.0)
    # the slot this wave sums the squares of (a slot this rank does not hold: a held one's, taken as 0)
    psl = fx.Int32(fb)
    held_w = fx.Int32(0) == fx.Int32(1)
    for s_ in range_constexpr(TOP_K):
        hit = (wave == s_) & (es[s_] >= 0)
        psl = hit.select(fx.Int32(s_), psl)
        held_w = held_w | hit
    stamp(c, 9)  # (trace) the fin poll's addresses known
    if fb >= 0:
        # the slots' y columns and this wave's slot's sum of squares in one poll round
        # (the ss publish trails its item's y by one item: polled apart it was a
        # second round trip)
        got = poll(
            [
                (y_mb, (t * TOP_K + (es[sl] >= 0).select(fx.Int32(sl), fb)) * (MOE_HIDDEN // 2) + col // 2, PPT)
                for sl in range(TOP_K)
            ]
            + [(ss_mb, (t * TOP_K + psl) * DN_T + 2 * lane, 2)]
        )  # fmt: skip
        stamp(c, 13)  # the token's y columns and sums of squares arrived (fin CTAs)
        got_ss = got[TOP_K]
        vs = []
        for sl in range_constexpr(TOP_K):
            held = es[sl] >= 0
            v = []
            for q in range_constexpr(PPT):
                lo, hi = bf2_f32(got[sl][q])
                # the post norm with input scale: y x weight -> bf16
                v.append(held.select(bf16_round(lo * pw[2 * q]), fx.Float32(0.0)))
                v.append(held.select(bf16_round(hi * pw[2 * q + 1]), fx.Float32(0.0)))
            vs.append(v)
        # every slot's sum of squares over its whole row: the dn items' DN_T publishes,
        # two a lane, a wave a slot; 1 / rms a slot to LDS
        tot = held_w.select(
            got_ss[0].bitcast(fx.Float32) + got_ss[1].bitcast(fx.Float32),
            fx.Float32(0.0),
        )
        tot = butterfly(tot, (1, 2, 4, 8, 16, 32))
        if lane == 0:
            rinv = fx.Float32(1.0) / fmath.sqrt(tot / float(MOE_HIDDEN) + POST_EPS)
            fx.ptr_store(rinv, red + (TOP_K * WAVES + wave))
        gpu.barrier()
        for sl in range_constexpr(TOP_K):
            sc = scs[sl] * fx.ptr_load(red + (TOP_K * WAVES + sl))
            a0 = a0 + sc * vs[sl][0]
            a1 = a1 + sc * vs[sl][1]
            if const_expr(NV == 4):
                a2 = a2 + sc * vs[sl][2]
                a3 = a3 + sc * vs[sl][3]
    if const_expr(NV == 2):
        vals = [a0, a1]
    else:
        vals = [a0, a1, a2, a3]
    if pusher:
        # this rank's partial of these columns to every rank (itself too), zero-sentinel words
        enc = [
            zs_enc(bf16_pair(vals[2 * q], vals[2 * q + 1]).bitcast(fx.Int32))
            for q in range(PPT)
        ]
        for pe in range_constexpr(W):
            zs_store(c["peer_ffn"][pe].value, zs_word(rank, t, col), enc)
    stamp(c, 10)  # (trace) the partial pushed
    if poller:
        # the ranks' partials of the same columns, summed in rank order, one rounding
        # (a spec a rank: its PPT words)
        ws = poll_zs(own_ffn.value, [zs_word(src, t, col) for src in range(W)], PPT)
        words = []
        for q in range_constexpr(PPT):
            lo, hi = bf2_f32(ws[0][q])
            for src in range_constexpr(1, W):
                lo2, hi2 = bf2_f32(ws[src][q])
                lo = lo + lo2
                hi = hi + hi2
            words.append(bf16_pair(lo, hi).bitcast(fx.Int32))
        stamp(c, 15, HALF if split else 0)  # every rank's partials arrived (fin CTAs)
        r_f = rsrc(fbuf.value)
        for k_ in range_constexpr(fb_copies(c["S"])):  # every copy (``fb_copies``)
            if const_expr(PPT == 1):
                bo.buffer_store(
                    words[0],
                    r_f,
                    k_ * FB_WORDS + (t * MOE_HIDDEN + col) // 2,
                    cache_modifier=CM_DEV,
                )
            else:
                bo.buffer_store(
                    fx.Vector.from_elements(words, fx.Int32), r_f, k_ * FB_WORDS + (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV
                )  # fmt: skip
    if const_expr(split):
        # only the fbuf stores precede the flag: the pushers' remote stores need no
        # ack here (their readers check the words), and the flag goes from a poller
        if tid >= HALF:
            release_stores()
        gpu.barrier()
        fk = tid - HALF  # every copy of the flag, a lane each
        if (fk >= 0) & (fk < F_FLAG_COPIES):
            put(c["fflag"], fk * FF_STRIDE + t * AP + part, fx.Int32(1))
    else:
        release_stores()
        gpu.barrier()
        if tid < F_FLAG_COPIES:
            put(c["fflag"], tid * FF_STRIDE + t * AP + part, fx.Int32(1))
    if poller:
        # re-arm the words read (the next push to them is a later launch's)
        for src in range_constexpr(W):
            zs_store(own_ffn.value, zs_word(src, t, col), [fx.Int32(0)] * PPT)


@flyc.jit
def stage_fin_tpx(c, t, part):
    """TP experts' fin task (token t, part): the part's columns of the token's eight
    slots' finalized values v (``vx``, pushed by the rank that owns those columns)
    and each slot's sum of squares (``ssx``: DN_T parts from every rank, a wave a
    slot) in one poll round; score / rms weighted sum in slot order, one bf16
    rounding -> ``k3_fbuf`` and the (token, part) flag. No all-reduce: every rank
    finalizes the whole row from the same values, in the same order."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    put, poll = c["put"], c["poll"]
    vx, ssx, fbuf = c["own_vx"], c["own_ssx"], c["fbuf"]
    a = c["args"]
    AP = c["ar_parts"]
    AR_COLS = MOE_HIDDEN // AP
    PPT = fin_ppt(AP)  # column pairs a thread
    NV = 2 * PPT
    act = NV * tid < AR_COLS
    col = part * AR_COLS + NV * fx.min(
        tid, AR_COLS // NV - 1
    )  # an idle thread loads the last columns
    scs = [
        fx.Float32(
            bo.buffer_load(
                rsrc(a["scores"]),
                t * TOP_K + sl,
                vec_width=1,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        for sl in range(TOP_K)
    ]
    assert WAVES == TOP_K  # wave w sums slot w's squares
    got = poll(
        [(vx, (t * TOP_K + sl) * (MOE_HIDDEN // 2) + col // 2, PPT) for sl in range(TOP_K)]
        + [(ssx, (t * TOP_K + wave) * DN_T + 2 * lane, 2)]
    )  # fmt: skip
    stamp(c, 13)  # the token's v and sums of squares arrived
    tot = got[TOP_K][0].bitcast(fx.Float32) + got[TOP_K][1].bitcast(fx.Float32)
    tot = butterfly(tot, (1, 2, 4, 8, 16, 32))
    if lane == 0:
        fx.ptr_store(
            fx.Float32(1.0) / fmath.sqrt(tot / float(MOE_HIDDEN) + POST_EPS),
            red + (TOP_K * WAVES + wave),
        )
    gpu.barrier()
    acc = [fx.Float32(0.0) for _ in range(NV)]
    for sl in range_constexpr(TOP_K):
        sc = scs[sl] * fx.ptr_load(red + (TOP_K * WAVES + sl))
        for q in range_constexpr(PPT):
            lo, hi = bf2_f32(got[sl][q])
            acc[2 * q] = acc[2 * q] + sc * lo
            acc[2 * q + 1] = acc[2 * q + 1] + sc * hi
    if act:
        words = [
            bf16_pair(acc[2 * q], acc[2 * q + 1]).bitcast(fx.Int32) for q in range(PPT)
        ]
        r_f = rsrc(fbuf.value)
        for k_ in range_constexpr(fb_copies(c["S"])):  # every copy (``fb_copies``)
            if const_expr(PPT == 1):
                bo.buffer_store(
                    words[0],
                    r_f,
                    k_ * FB_WORDS + (t * MOE_HIDDEN + col) // 2,
                    cache_modifier=CM_DEV,
                )
            else:
                bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), r_f, k_ * FB_WORDS + (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV)  # fmt: skip
    release_stores()
    gpu.barrier()
    if tid < F_FLAG_COPIES:
        put(c["fflag"], tid * FF_STRIDE + t * AP + part, fx.Int32(1))


# ---------------------------------------------------------------- 4. unpool
# (k1_early_map) the unpool jobs are claimed by the pool CTAs in dn-end order
# (``unp_claim``), and the fin tasks too, after the unpool jobs, by every CTA past the
# bnd tasks: a fin task on a CTA whose dn item ended last (a fixed fin CTA) began its
# prologue (~0.9 us) and its y poll after that item, on the critical path


def k1_early_map(c) -> bool:
    """A merged launch's next K1 GEMV weights go out in K3 (``run_k3``): one unpool job
    a CTA, bf16 unpooler (no scale loads behind them)."""
    return c.get("k1_pre") is not None and c["bases"]["ku"] == 1 and not c["fp8_unpool"]


def lds_dma_sc1(rsrc_v, lds_ptr, voffset):
    """``device.lds_dma_16`` with the device-scope policy (``sc1``, before ``lds``:
    data another CTA stored in this launch): 16 B a lane into LDS at ``lds_ptr`` + 16
    lane, hidden from the compiler's wait counting (the caller waits)."""
    r = as_ir_value(rsrc_v)
    if "ptr<8>" in str(r.type):
        r = bo._ptr8_to_v4i32(r)
    rv = fx.Vector(r, (4,), fx.Int32)
    rs = fx.Vector.from_elements([uniform(fx.Int32(rv[i])) for i in range(4)], fx.Int32)
    base = uniform(fx.Int32(llvm.ptrtoint(T.i32, lds_ptr.llvm_ptr)))
    llvm.inline_asm(
        None,
        [
            as_ir_value(rs),
            as_ir_value(base),
            as_ir_value(fx.Int32(voffset)),
            as_ir_value(uniform(fx.Int32(0))),
        ],
        "s_mov_b32 m0, $1\n\ts_nop 0\n\tbuffer_load_dwordx4 $2, $0, $3 offen sc1 lds",  # codespell:ignore offen
        "s,s,v,s,~{m0},~{memory}",
        has_side_effects=True,
    )


def unp_slot(blk, row):
    """The LDS slot of 16-byte block ``blk`` (of 64 in a 1 KB DMA chunk) of staged row
    ``row``: XOR-swizzled by the row, so the MFMA B reads (a lane a row, the same block)
    hit eight distinct 16-byte bank groups instead of one (rows are 8 KB apart)."""
    return blk ^ row


def unp_rows_dma(c, r_f, t0, nt):
    """A pass's rows (tokens t0 .., ``k3_fbuf``) into ``xcol`` by LDS DMA: 16 B a lane a
    round at M0 + 16 lane, round i = row i, the lane's source block swizzled
    (``unp_slot``) (the caller waits for them)."""
    tid, wave, lane, xcol = c["tid"], c["wave"], c["lane"], c["xcol"]
    assert nt * (MOE_HIDDEN // 8) // THREADS == nt and nt <= 8  # a round a row
    for i in range_constexpr(nt * (MOE_HIDDEN // 8) // THREADS):
        v = wave * 64 + unp_slot(lane, i) + THREADS * i
        rocdl.raw_ptr_buffer_load_lds(
            r_f,
            (xcol + (i * THREADS + wave * 64) * 4).llvm_ptr,
            16,
            t0 * (MOE_HIDDEN * 2) + v * 16,
            fx.Int32(0),
            0,
            CM_DEV,  # fmt: skip
        )


# (k1_early_map) An unpool CTA's waves ``unp_k1_claim`` .. issue their next-K1 GEMV
# weights at the claim (behind the first tile's, as wave 0 .. before), the waves before
# it once the CTA's rows are pushed. A CU's memory pipeline under the tail's load
# streams ~27 GB/s (a task's 147 KB ~5.4 us; a wave stalls at issue while it is full),
# and the fin-flag poll queues behind the bytes in flight: on the heaviest EP rank,
# whose unpool CTAs claim at its dn end, ~1 us at t8. Fewer claim-time bytes return the
# flags sooner; the deferred waves' weights land later in K1 (K1 needs ~5 us of lead):
# t8 nets a gain with waves 1..3 deferred (5 / 8 lose), t16 (no heavy-rank lag) none.
UNP_K1_CLAIM_T8 = 4
UNP_K1_CLAIM_T16 = 1  # 1: wave 0's in K1, the others at the claim (unchanged)


def unp_k1_claim(c) -> int:
    return UNP_K1_CLAIM_T8 if c["S"] <= 8 else UNP_K1_CLAIM_T16


def unp_k1_late(c) -> bool:
    return c.get("k1_pre") is not None and unp_k1_claim(c) > 1


def unp_k1_rest(c, kw, ku):
    """The next K1's loads of the waves before ``unp_k1_claim`` (the others' in
    flight since the claim: ``kw`` / ``ku``, zeros on these waves), merged in."""
    rest = c["wave"] < unp_k1_claim(c)
    rocdl.sched_barrier(0)
    kl = c["k1_pre"](rest)
    ul = c["k1_units_pre"](rest)
    rocdl.sched_barrier(0)
    kw = [[kw[i][j] | kl[i][j] for j in range(len(kw[i]))] for i in range(len(kw))]
    ku = [
        (rest.select(ul[p][0], ku[p][0]), rest.select(ul[p][1], ku[p][1]))
        for p in range(len(ku))
    ]
    return kw, ku


UNP_RED_WAVE = 64 * 4 + 64  # a wave's accumulators in ``red``, + the skew's 48


def unp_red_at(wave, lane):
    """Float offset of a lane's 4 unpool accumulators in ``red``, skewed by 16 floats a
    16 lanes: the row sums (row r, token t a thread: lanes of 16 rows x 4 tokens) read
    64 distinct banks instead of 16."""
    return wave * UNP_RED_WAVE + lane * 4 + 16 * (lane // 16)


@flyc.jit
def unp_job(c, cur, gtask, t0, nt, live_k, first):
    """An unpool job on rows already in ``xcol`` (``cur``: its weights): the MFMAs,
    the wave-ordered row sums -> bf16, pushed to every rank's ``unp`` region
    (``live_k``) as zero-sentinel words (``zs_enc``; the boundary polls and re-arms
    them), or tagged pairs for a deferred K1. ``first``: the CTA's first job (its
    trace stamps)."""
    W, tid, lane, wave = c["W"], c["tid"], c["lane"], c["wave"]
    red, misc, xcol, put_bf = c["red"], c["misc"], c["xcol"], c["put_bf"]
    V4F = fx.Vector.make_type(4, fx.Float32)
    l16, g4 = lane % 16, lane // 16
    tl = fx.min(l16, nt - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for kc in range_constexpr(UNP_STEPS):
        # k = 512 wave + 32 kc + 8 g4: 16-byte block 4 kc + g4 of the row's chunk ``wave``
        blk = unp_slot(kc * 4 + g4, tl)
        fbv = fx.Vector(
            fx.ptr_load(
                xcol + (tl * (MOE_HIDDEN // 2) + wave * (MOE_HIDDEN // WAVES // 2) + blk * 4), result_type=V4F
            )
        )  # fmt: skip
        acc = mfma_bf16(unp_a(c, cur[kc]), fbv.bitcast(fx.BFloat16), acc)
    fx.ptr_store(acc, red + unp_red_at(wave, lane))
    gpu.barrier()
    if const_expr(first):
        stamp(c, 10)  # the first job's MFMAs done
    # branch = bf16(the fp32 sum over the waves, in wave order, x the row's FP8 scale)
    if tid < UNP_ROWS * nt:
        r, tk = tid % UNP_ROWS, tid // UNP_ROWS
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (unp_red_at(w, 16 * (r // 4) + tk) + r % 4))
        fx.ptr_store(bf16_round(tot * unp_scale(c, gtask * UNP_ROWS + r)), misc + tid)
    gpu.barrier()
    if const_expr(c["defer_k1"]):
        # (the deferred K1 polls these rows as tagged pairs)
        if live_k & (tid < (UNP_ROWS // 2) * nt):
            tk = tid // (UNP_ROWS // 2)
            rr = (tid % (UNP_ROWS // 2)) * 2
            vals = [fx.ptr_load(misc + (tk * UNP_ROWS + rr)), fx.ptr_load(misc + (tk * UNP_ROWS + rr + 1))]  # fmt: skip
            for pe in range_constexpr(W):
                put_bf(
                    c["peer_unp"][pe],
                    (t0 + tk) * HIDDEN + gtask * UNP_ROWS + rr,
                    vals,
                )
    else:
        # zero-sentinel words (``zs_enc``), half a token's 16 rows (four words) a thread
        if live_k & (tid < 2 * nt):
            tk = tid // 2
            r0 = (tid % 2) * (UNP_ROWS // 2)
            enc = [
                zs_enc(bf16_pair(fx.ptr_load(misc + (tk * UNP_ROWS + r0 + 2 * j)), fx.ptr_load(misc + (tk * UNP_ROWS + r0 + 2 * j + 1))).bitcast(fx.Int32))
                for j in range(UNP_ROWS // 4)
            ]  # fmt: skip
            for pe in range_constexpr(W):
                zs_store(
                    c["peer_unp"][pe].value,
                    ((t0 + tk) * HIDDEN + gtask * UNP_ROWS + r0) // 2,
                    enc,
                )
    gpu.barrier()
    if const_expr(first):
        stamp(c, 11)  # the first job's rows pushed


@flyc.jit
def stage_unpool(c, first_w):
    """This CTA's unpool tasks (one at TP 8; ``first_w``: the first's weights,
    issued before the fin flags' wait, which streams them): the summed rows are
    staged in LDS by DMA in passes of XS_LDS_MAX tokens, each pass's 16 x tokens
    bf16 sums pushed to every rank's ``unp`` region (zero-sentinel words, ``unp_job``).
    A further task's weights go out before the previous task's passes."""
    S, tid = c["S"], c["tid"]
    b = c["bases"]
    AP, KU = c["ar_parts"], b["ku"]
    # this CTA's copy of the rows (``FB_COPIES``)
    r_f = rsrc(
        c["fbuf"].value + fx.Int64(fx.Int32(c["bid"] % fb_copies(S)) * (FB_WORDS * 4))
    )
    assert S % XS_LDS_MAX == 0, S  # a job is one whole pass of XS_LDS_MAX tokens
    nt = XS_LDS_MAX  # every served width is whole passes
    task0, gtask0, ps0 = unp_gtask(c, 0)
    live = (task0 >= 0) & (task0 < b["unp_jobs"])
    if live:
        rocdl.sched_barrier(0)
        stamp(c, 13)  # (trace) the fin flags' poll issued (unpool CTAs)
        ff = shift(
            c["fflag"], fx.Int64(fx.Int32(c["bid"] % F_FLAG_COPIES) * (FF_STRIDE * 8))
        )  # this CTA's copy
        poll_flags(c, [(ff, S * AP)])
    stamp(c, 8)  # the fin flags arrived (unpool CTAs)
    if const_expr(c.get("k1_w") is not None):
        # (``k1_early_map``: one job, waves 1 .. hold the next K1's weights in flight)
        # the rows are staged by DMA the compiler's wait counting does not see (it
        # would wait for every earlier load before the rows' LDS reads), each wave
        # waits for its own; the barrier hands them to the waves
        tp0 = fx.min(ps0, b["unp_passes"] - 1) * XS_LDS_MAX
        late = unp_k1_late(c)
        kw, ku = c["k1_w"], c["k1_units"]
        if live:
            # (the claimed jobs' CTAs ended dn first: their K1 weights are in) every
            # wave stages its own K eighth of the rows
            w = c["wave"]
            for i in range_constexpr(nt * (MOE_HIDDEN // 8) // THREADS):
                vw = unp_slot(c["lane"], i) + 64 * w + THREADS * i
                lds_dma_sc1(r_f, c["xcol"] + (i * THREADS + w * 64) * 4, tp0 * (MOE_HIDDEN * 2) + vw * 16)  # fmt: skip
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            stamp(c, 9)  # the first job's rows landed in LDS
            unp_job(c, first_w, gtask0, tp0, nt, live, True)
            if const_expr(late):
                # (``unp_k1_claim``) the deferred waves' next-K1 loads, the rows pushed
                kw, ku = unp_k1_rest(c, kw, ku)
        if const_expr(late):
            c["k1_w"], c["k1_units"] = kw, ku
    else:
        if live:
            cur = first_w
            for k in range_constexpr(KU):
                task, gtask, job_ps = unp_gtask(c, k)
                live_k = (task >= 0) & (
                    task < b["unp_jobs"]
                )  # a dead job computes on a clamped pass, publishes nothing
                t0 = fx.min(job_ps, b["unp_passes"] - 1) * XS_LDS_MAX
                unp_rows_dma(c, r_f, t0, nt)
                # the next job's weights go out behind this job's rows: loads complete in
                # order, so the wait below counts them out (UNP_STEPS a lane still in flight)
                if const_expr(k + 1 < KU):
                    nxt = unp_w_loads(c, unp_gtask(c, k + 1)[1])
                    rocdl.s_waitcnt(vmcnt=UNP_STEPS)
                else:
                    rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if const_expr(k == 0):
                    stamp(c, 9)  # the first job's rows landed in LDS
                unp_job(c, cur, gtask, t0, nt, live_k, k == 0)
                if const_expr(k + 1 < KU):
                    cur = nxt


# ---------------------------------------------------------------- 5. the boundary
@flyc.jit
def stage_bnd(c, t, x_flag=None, x_put=None):
    """bnd task t: the token's whole unpooled row (16 elements a thread, from every
    rank's pushes) through the post norm with input scale (eps 1e-8), the gated
    residual (the fp32 stream in place) and the next layer's pre-attention norm
    (eps 1e-5) -> ``attn_out``. On the last layer the stream is rounded to bf16
    before that norm (the model's final-norm input dtype), whose weight is then the
    final norm's: ``c["last"]`` compiles the rule in, else the launch's ``last``
    argument selects it. The operands go out (and their row-independent VALU runs)
    before the wait on the row, zero-sentinel words re-armed once read
    (a deferred K1's build: tagged pairs). ``x_flag``
    (with ``x_put``, the next layer's mailbox): the next layer's K1 runs later in
    this launch; the token's flag."""
    tid, poll, own_unp = c["tid"], c["poll"], c["own_unp"]
    a = c["args"]
    e0 = tid * BND_ELEMS
    fw_w = [fx.Vector(bo.buffer_load(rsrc(a["pfw"]), e0 // 2 + 4 * i, vec_width=4, dtype=T.i32)) for i in range(2)]  # fmt: skip
    al = [fx.Vector(bo.buffer_load(rsrc(a["alpha"]), e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    be = [fx.Vector(bo.buffer_load(rsrc(a["beta"]), e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    st = [fx.Vector(bo.buffer_load(rsrc(a["res"]), t * HIDDEN + e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    nw_w = [fx.Vector(bo.buffer_load(rsrc(a["npw"]), e0 // 2 + 4 * i, vec_width=4, dtype=T.i32)) for i in range(2)]  # fmt: skip
    # the row-independent operands' VALU (weights unpacked, alpha st) before the
    # wait: the row lands microseconds after these loads
    fw = [
        bf2_f32(fx.Int32(fw_w[k // 8][(k % 8) // 2]))[k % 2] for k in range(BND_ELEMS)
    ]
    nw = [
        bf2_f32(fx.Int32(nw_w[k // 8][(k % 8) // 2]))[k % 2] for k in range(BND_ELEMS)
    ]
    ast = [
        fx.Float32(al[k // 4][k % 4]) * fx.Float32(st[k // 4][k % 4])
        for k in range(BND_ELEMS)
    ]
    rocdl.sched_barrier(0)
    if const_expr(c["defer_k1"]):
        got = poll([(own_unp, (t * HIDDEN + e0) // 2 + 2 * j, 2) for j in range(4)])
        u = [
            bf2_f32(got[j][q])[h] for j in range(4) for q in range(2) for h in range(2)
        ]
    else:
        # the row's zero-sentinel words (``unp_job``), re-armed once read
        got = poll_zs(
            own_unp.value, [(t * HIDDEN + e0) // 2 + 4 * j for j in range(2)], 4
        )
        zs_store(own_unp.value, (t * HIDDEN + e0) // 2, [fx.Int32(0)] * 4)
        zs_store(own_unp.value, (t * HIDDEN + e0) // 2 + 4, [fx.Int32(0)] * 4)
        u = [
            bf2_f32(got[j][q])[h] for j in range(2) for q in range(4) for h in range(2)
        ]
    stamp(c, 15)  # the token's unpooled row arrived (bnd CTAs)
    # post_input = bf16(branch x weight), nb = bf16(post_input x rstd)
    pp = [bf16_round(u[k] * fw[k]) for k in range(BND_ELEMS)]
    ssp = pp[0] * pp[0]
    for k in range_constexpr(1, BND_ELEMS):
        ssp = ssp + pp[k] * pp[k]
    rstd = hw_rsq(block_sum(c, ssp, 0) / float(HIDDEN) + POST_EPS)
    stamp(c, 9)  # (trace) the first norm's sum (every wave's row in)
    nb = [bf16_round(pp[k] * rstd) for k in range(BND_ELEMS)]
    new = [ast[k] + fx.Float32(be[k // 4][k % 4]) * nb[k] for k in range(BND_ELEMS)]
    for i in range_constexpr(4):
        bo.buffer_store(fx.Vector.from_elements(new[4 * i : 4 * i + 4], fx.Float32), rsrc(a["res"]), t * HIDDEN + e0 + 4 * i)  # fmt: skip
    if const_expr(c["last"]):
        pre = [bf16_round(v) for v in new]
    else:
        # (a uniform branch on the launch's argument: no rounding work on other layers)
        pre = new
        if a["last"] != 0:
            pre = [bf16_round(v) for v in new]
    ss2 = pre[0] * pre[0]
    for k in range_constexpr(1, BND_ELEMS):
        ss2 = ss2 + pre[k] * pre[k]
    rstd2 = hw_rsq(block_sum(c, ss2, 1) / float(HIDDEN) + RMS_EPS)
    stamp(c, 10)  # (trace) the second norm's sum
    if const_expr(c["defer_k1"]):
        # the next K1 folds its two operand halves with these (``pre_attn.stage_gemv``)
        if tid < 2:
            c["put"](c["nscal"], 2 * t + tid, (tid == 0).select(rstd, rstd2))
    for i in range_constexpr(2):
        words = [
            bf16_pair(pre[k] * rstd2 * nw[k], pre[k + 1] * rstd2 * nw[k + 1]).bitcast(
                fx.Int32
            )
            for k in range(8 * i, 8 * i + 8, 2)
        ]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(a["attn_out"]), (t * HIDDEN + e0) // 2 + 4 * i, cache_modifier=CM_DEV)  # fmt: skip
    if const_expr(x_flag is not None):
        # the next layer's K1 runs in this launch: the token's flag, under that
        # layer's tag (``x_put``)
        release_stores()
        gpu.barrier()
        stamp(c, 11)  # (trace) the stores drained
        # from the last wave: the next K1's x-flag poll (wave 0, ``poll_flags``) would
        # return behind this store's ack on the wave that issued it; every copy of the
        # flag (``layout.X_FLAG_COPIES``, MAX_TOKENS pairs apart), a lane each
        k = tid - (THREADS - 64)
        if (k >= 0) & (k < X_FLAG_COPIES):
            x_put(x_flag, k * MAX_TOKENS + t, fx.Int32(1))


# ---------------------------------------------------------------- the launch's order
@flyc.jit
def run_k3(c, route_flag=None, xin_flag=None, x_flag=None, x_put=None, res_flag=None):
    """K3's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    the routing and the expert input were written earlier in this launch, flagged a
    token each. ``x_flag`` / ``x_put``: see ``stage_bnd``."""
    S, bid = c["S"], c["bid"]
    b = c["bases"]
    AP = c["ar_parts"]
    stamp(c, 0)
    xin_wait = None
    ids_lds = None
    first = None
    if const_expr(route_flag is not None):
        # the routing first: the plan needs only the ids, polled straight from the
        # route's tagged words (``route_flag``: token t's at t TOP_K .., the ids /
        # scores stored before them) by the last wave alone -- every pushing wave of
        # K2 (pool reduce, xn, boundary flags) is wave 0, and a poll returns in order
        # behind its own wave's stores -- into LDS for every wave's plan. The expert
        # input's flags (xn's norm + quantization, a little after the routing) are
        # polled once now, a lane a token, and waited for after the plan
        # (``stage_ug``) on the same wave, which issues no weights before that wait.
        # (``res_flag`` = (region, count): K2's boundary parts flag their stream columns
        # late, after the CTA's pooler job; only the bnd CTAs read them, polled there
        # after the dn stage, so the plan does not wait for them)
        lane = c["lane"]
        assert S <= 64 and TOP_K == 8
        tl = fx.min(lane, fx.Int32(S - 1))  # this lane's token
        xin_spec = [(xin_flag, tl, 1)]
        xin_first = c["poll_issue"](xin_spec)
        ids_lds = c["xcol"]  # (free until the ug stage stages x)
        # the first pair from the picked experts' mask, before the plan; with a
        # balanced plan, ``first_pair_table`` while the routing is polled
        early_first = (
            not c["tpx"]
            and not c["ug_dyn"]
            and c["xq_pre"]
            and S <= TILE_COLS
            and xq_dma_ok(S)
        )
        fp_table = early_first and ug_balanced(c)
        if const_expr(fp_table):
            fp_tab, fp_ka = first_pair_table(c)
        if c["wave"] == POLL_WAVE:
            got = c["poll"]([(route_flag, 2 * tl, 2)])
            m = fx.Int32(0)
            for sl in range_constexpr(TOP_K):
                e = (fx.Int32(got[0][sl // 4]) >> (8 * (sl % 4))) & 255
                fx.ptr_store(e.bitcast(fx.Float32), ids_lds + (tl * TOP_K + sl))
                if const_expr(early_first):
                    le = e - c["args"]["local_start"]
                    ok = (le >= 0) & (le < LOCAL_EXPERTS)
                    m = m | ok.select(fx.Int32(1) << fx.max(fx.min(le, LOCAL_EXPERTS - 1), 0), fx.Int32(0))  # fmt: skip
            if const_expr(early_first):
                # the picked local experts of every token (lanes past S repeat S - 1)
                m = butterfly(m, (32, 16, 8, 4, 2, 1), bor)
                if lane == 0:
                    fx.ptr_store(m, c["nt"] + 1)
        gpu.barrier()
        acquire_loads()
        if const_expr(early_first):
            # the first pair's weights now, before the plan (every wave; the poll wave's
            # expert-input poll returns behind its share, ``stage_ug``)
            mask = uniform(fx.ptr_load(c["nt"] + 1))
            if const_expr(fp_table):
                first = first_pair_fast(c, mask, fp_tab, fp_ka)
            else:
                first = first_pair_mask(c, mask)

        def xin_wait(dma=False):
            if c["wave"] == POLL_WAVE:
                c["poll_wait"](xin_spec, xin_first)
                if const_expr(dma):
                    acquire_loads()
                    xq_dma(c)  # the staged blocks, from this wave alone
                    rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            acquire_loads()

    stamp(c, 1)  # the routing arrived (a merged launch) / entry
    if const_expr(c["tpx"]):
        n_tiles = stage_plan_tpx(c)
    else:
        n_tiles = stage_plan(c, ids_lds)
    stamp(c, 2)  # the plan built
    if const_expr(c["ug_dyn"]):
        dn_first = stage_ug_dyn(c, n_tiles, xin_wait)
    else:
        dn_first = stage_ug(c, n_tiles, xin_wait, first)
    stamp(c, 3)  # this CTA's ug items done
    stage_dn(c, n_tiles, dn_first)
    stamp(c, 4)  # this CTA's dn items done
    if const_expr(res_flag is not None):
        # the bnd CTAs' K2 stream columns (set long ago: one round trip, ahead of the
        # K1 weights below, hidden by the fin / unpool hops the bnd waits on)
        if bid < S:
            poll_flags(c, [res_flag])
    # the unpool CTAs' first task's weights go out now, behind nothing, so the fin
    # hops stream them; the other CTAs issue none
    fin_task = None
    if const_expr(k1_early_map(c)):
        # every CTA past the bnd tasks claims in dn-end order: the unpool jobs, then
        # (tickets past them) the fin tasks
        tk_ = fx.Int32(-1)
        if bid >= S:
            tk_ = unp_claim(c, BLOCKS - S)
        c["unp_job"] = tk_  # (a ticket past the jobs is no unpool job: ``unp_gtask``)
        ft_ = tk_ - b["unp_jobs"]
        fin_task = ((ft_ >= 0) & (ft_ < S * AP)).select(ft_, fx.Int32(-1))
    job0, gtask0, _ = unp_gtask(c, 0)
    unp_live0 = (job0 >= 0) & (job0 < b["unp_jobs"])
    first_w = unp_w_loads_if(c, gtask0, unp_live0)  # the first job's tile
    if const_expr(k1_early_map(c)):
        # the next layer's K1 GEMV weights of the CTAs K3 keeps longest go out now, so
        # they land during the tail's waits instead of after them (``pre_attn.run_k1``):
        # every wave of a boundary CTA (they wait for the unpooled rows) and waves 1 ..
        # of an unpool CTA; its wave 0 polls the fin flags, which would return behind
        # them (loads complete in order), so it issues its slice at K1. The other
        # waves issue none here (``pre_attn.gemv_w_loads_if``)
        if const_expr(unp_k1_late(c)):
            # (``unp_k1_claim``) an unpool CTA's waves before it issue theirs once its
            # rows are pushed (``stage_unpool``), not in K1
            now = (bid < S) | (unp_live0 & (c["wave"] >= unp_k1_claim(c)))
            c["k1_early"] = (bid < S) | unp_live0
            c["k1_units"] = c["k1_units_pre"](now)
            c["k1_w"] = c["k1_pre"](now)
        else:
            early = (bid < S) | (unp_live0 & (c["wave"] != 0))
            c["k1_early"] = early
            c["k1_units"] = c["k1_units_pre"](
                early
            )  # (the units' token words, a pair a pass)
            c["k1_w"] = c["k1_pre"](early)
    rocdl.sched_barrier(0)
    if const_expr(fin_task is not None):
        if fin_task >= 0:
            if const_expr(c["tpx"]):
                stage_fin_tpx(c, fin_task // AP, fin_task % AP)
            else:
                stage_fin(c, fin_task // AP, fin_task % AP)
    elif (bid >= b["fin0"]) & (bid < b["fin0"] + S * AP):
        if const_expr(c["tpx"]):
            stage_fin_tpx(c, (bid - b["fin0"]) // AP, (bid - b["fin0"]) % AP)
        else:
            stage_fin(c, (bid - b["fin0"]) // AP, (bid - b["fin0"]) % AP)
    stamp(c, 5)  # the fin task done (fin CTAs)
    stage_unpool(c, first_w)
    stamp(c, 6)  # the unpool task(s) done (unpool CTAs)
    if bid < S:
        stage_bnd(c, bid, x_flag, x_put)
    stamp(c, 7)  # the boundary done (end)
