# SPDX-License-Identifier: Apache-2.0
"""K-attn: one Muse Spark layer's attention over the fp8 KV cache, for S =
``tokens`` rows of R = S / QL requests, ``QL`` consecutive rows a request (a
DFlash verify's 8 draft tokens, or one for decode). Per request and KV head, the
rank's 8 query heads of each of its QL rows attend the request's keys, as the
ROCm model's verify kernels (``swa_verify_splitkv``, ``gqa_verify_splitkv_2kv``)
and aiter's unified attention compute them::

    S = scale log2(e) (q . k)   q bf16 (K1 already applied the query scale),
                                K E4M3 upcast to bf16, fp32
    masked: key < context + qpos + 1 (causal), context + qpos - key < window
    online softmax in the exp2 domain, P -> bf16, O += P V (V upcast)
    O / l -> bf16

Tasks. An attention task is one (request, KV head, ``nt`` row tiles, partition
of ``part`` keys; the partitions a sliding window masks whole are skipped): a
chunk's K / V loads and its V^T staging serve the task's tiles (QL = 8 is 64
rows = four tiles; two a task fit the registers, four spill). Each is a loop over 256-key chunks
with the waves splitting the keys: QK on wave-local key tiles, the row
statistics, P through LDS, V transposed through LDS, PV; the waves' key splits
merge in LDS as a tree (8 -> 4 -> 2 -> 1); the partial (m, l, O) a row goes out
as plain fp32 words with one flag a task. The partitions are sized to at most
``DIRECT_MAX`` a request from the longest request's span (``part_in`` overrides;
with QL = 8 the chunk work sizing keeps n_parts within it anyway), so one 95k-key
request's full layer is 64 tasks a tile pair and KV head and a sliding layer eight
tasks of one; the partition grows with the step's total chunk work so a step stays
near two equal tasks a CTA whatever the request mix.

Merge. One job a (request, KV head, row tile), on one CTA whatever the partition
count (single level: a level-1 hop of plain words, flags and a second poll cost
~3-4 us at 8k keys). The partial rows of a (request, head, partition, row tile)
are one block (``part_blk``: O in four 1 KB [16 rows][16 dims] chunks, then the
rows' (m, l)), so a merge wave's load reads whole contiguous chunks. The fold
order is the old two-level merge's, bit for bit: with at most ``DIRECT_FOLD``
partitions, partition lane pl folds pl, pl + 4, .. and the four lanes (a quad,
DPP broadcasts) fold in lane order; with more, ``MERGE_GROUPS`` contiguous groups
fold that way (a group a wave, the four lane states in each thread), the group
records meet in LDS and fold as the old level-2 job did. The merge jobs sit on
the last CTAs (the task loop's extra tasks fall on the first). In a merged launch
of at most ``OG_TAGGED_MAX`` rows a job also pushes its o rows to K2 as tagged
pairs, which K2 polls directly (``og``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    ballot,
    fp8_to_bf16x8,
    hw_exp2,
    mfma_bf16,
    permlane_swap,
    read_lane_i32,
    rsrc,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES
from sgl_amd_mono.muse_spark.gfx950.config import HEAD_DIM, LOCAL_KV_HEADS, LOG2E, Q_K
from sgl_amd_mono.muse_spark.gfx950.device import (
    lds_dma_16,
    poll_flags,
    release_stores,
    stamp,
    udiv,
    umod,
)
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_CHUNK as CHUNK
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_IDS_CHUNKS as IDS_CHUNKS
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_KEY_SPLITS as KEY_SPLITS
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_KEY_TILES as KEY_TILES
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_KEYS_PER_WAVE as KEYS_PER_WAVE
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_MAX_PARTS as MAX_PARTS
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_MERGE_GROUPS as GROUPS
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_PART_ROW as PART_ROW
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_Q_PITCH as Q_PITCH
from sgl_amd_mono.muse_spark.gfx950.layout import ATTN_V_STEPS as V_STEPS
from sgl_amd_mono.muse_spark.gfx950.layout import GQA
from sgl_amd_mono.muse_spark.gfx950.layout import KATTN_RING_DEPTH as RING_DEPTH

TILE = 16  # keys an MFMA tile
NEG = -1.0e30  # a masked score, and "no key yet" of a row's running max
NO_MAX = -1.0e29  # a running max below this is "no key yet": exp2 then takes 0
O_OFF = 4  # a partial row: m, l, pad, pad, O[64] (O 16 B aligned)
COMB_ROW = PART_ROW  # a tile row's (m, l, pad, pad, O[64]) in the LDS tree
SPEC_R = 2  # speculative page ids for R <= this
SPEC_CHUNKS = 64  # chunks a wave's speculative ids cover (a lane each)
REG_CHUNKS = 64  # a task of at most this many chunks loads its page ids to registers
OG_TAGGED_MAX = 16  # widths whose merge pushes its o rows to K2 as tagged pairs
MERGE_LANES = 4  # partitions folded side by side a (row, 8 dims): 16 x 8 x 4 = THREADS
PAGE_WORDS = (
    16 * HEAD_DIM // 4
)  # a 16-key fp8 page of one head in LDS: 1 KB, chunk-major (dims 16 c .., key)
V_SKEW = 16  # words between the two V pages beyond a page: the transposed reads of both hit distinct banks
SLOT_WORDS = (
    2 * KEY_TILES * PAGE_WORDS + V_SKEW
)  # a ring slot: a wave's chunk, its K pages then its V pages
TAU = 8.0  # a row's reference max moves only when a chunk's max exceeds it by this (log2 units): P <= 256
DIRECT_MAX = 64  # partitions a merge job folds (the sizing keeps n_parts within)
DIRECT_FOLD = (
    16  # partitions the merge folds in one level (four a lane); more in groups
)
BLK = 16 * HEAD_DIM + 32  # floats a (request, head, partition, row tile) partial block
BLK_ML = 16 * HEAD_DIM  # the block's (m, l) pairs, after its O
G_O = 8 * 16 * 2  # the grouped merge's LDS: the groups' (m, l) pairs, then their O rows
G_PITCH = HEAD_DIM + 4  # floats a group O row in LDS (bank spread)


def comb_floats(nt: int) -> int:
    """The key-split merge's LDS floats: every wave's state of ``nt`` tiles."""
    return WAVES * nt * 16 * COMB_ROW


def kattn_lds(nt: int):
    """K-attn's LDS as two aliasing views (a union of them, built by the launch): a
    task's Q tiles and the merge jobs' reduction, then either each wave's P tiles
    and V^T (the chunk loop) or the key splits' merge tree (after it)."""

    @fx.struct
    class Loop:
        qs: fx.Array[
            fx.BFloat16, nt * 16 * Q_PITCH, 16
        ]  # the task's Q rows (its row tiles)
        red: fx.Array[fx.Float32, THREADS * 10, 16]  # the merge jobs' lane combine
        ring: fx.Array[
            fx.Int32, WAVES * RING_DEPTH * SLOT_WORDS, 16
        ]  # each wave's K / V chunks (DMA ring)
        ids: fx.Array[
            fx.Int32, IDS_CHUNKS * 16, 16
        ]  # the task's page ids, a block of chunks

    @fx.struct
    class Tree:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]
        red: fx.Array[fx.Float32, THREADS * 10, 16]
        comb: fx.Array[fx.Float32, comb_floats(nt), 16]

    return Loop, Tree


def lds_views(loop, tree):
    """``c`` entries of the two ``kattn_lds`` views (``.peek()`` of union members)."""
    return {"qs": loop.qs.ptr, "red": loop.red.ptr, "ring": loop.ring.ptr, "ids": loop.ids.ptr, "comb": tree.comb.ptr}  # fmt: skip


@flyc.jit
def run_kattn(c, o_flag=None, og=None):
    """K-attn's work on this CTA for ``c["S"]`` rows, ``c["QL"]`` a request.
    ``o_flag`` (a mailbox region): each level-2 merge raises its (request, KV
    head, row tile)'s flag once its o rows are out (a launch that goes on to K2).
    ``og`` (K2's ``k2_og`` region, S <= OG_TAGGED_MAX): the merge also pushes its o
    rows there as tagged pairs, and its flag goes up as they are stored, a hint
    (``post_attn.og_round_poll`` polls the rows themselves)."""
    S, QL = c["S"], c["QL"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    put, poll = c["put"], c["poll"]
    part_w, pflag = c["part"], c["pflag"]
    a = c["args"]
    R = S // QL
    ROWS = QL * GQA  # rows a (request, KV head): 64 or 8
    ROW_TILES = (ROWS + 15) // 16  # 4 or 1
    NT = c["NT"]  # row tiles a task (the build's; 4 spills registers, 2 fits)
    # (#142) a layer kind's own build (``c["KIND"]``, QL 8): "sliding" (at most 7 requests:
    # the window is on) or "full" (window == 0): the other kind's KV pool reads and the
    # window's chunk-range select are compiled out. The same values: bit-exact to the
    # any-layer build on a layer of that kind.
    SL = c.get("KIND", "") == "sliding" and QL * GQA == 64 and S // QL <= 7
    FL = c.get("KIND", "") == "full" and QL * GQA == 64
    KV_ROWS = c.get(
        "kv_rows", 0
    )  # a head-major full pool's rows (``kv_rows``), else NHD
    SWA_ROWS = c.get("swa_rows", 0)  # a head-major sliding (SWA) pool's rows, else NHD
    TASK_TILES = ROW_TILES // NT
    l16 = lane % 16
    g4 = lane // 16
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    V2I = fx.Vector.make_type(2, fx.Int32)
    V4F = fx.Vector.make_type(4, fx.Float32)
    qs, ring, ids, comb = c["qs"], c["ring"], c["ids"], c["comb"]
    wring = ring + wave * (RING_DEPTH * SLOT_WORDS)  # this wave's K / V ring
    table_stride, extra, window, part_in = (
        a["table_stride"],
        a["extra"],
        a["window"],
        a["part_in"],
    )
    lens64 = a["lens64"]
    qk_scale = a["scale_bits"].bitcast(fx.Float32) * fx.Float32(LOG2E)
    r_k = rsrc(a["k_cache"])
    r_v = rsrc(a["v_cache"])
    r_pt = rsrc(a["page_table"], a["table_bytes"])
    r_q = rsrc(a["q"])
    r_part = rsrc(part_w.value)
    stamp(c, 0)

    # the requests' key counts, loaded once at entry (a dynamic request's count is
    # a select of them: no second dependent load in front of a task)
    seqs = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["seq_lens"]),
                (lens64 != 0).select(fx.Int32(2 * rr), fx.Int32(rr)),
                vec_width=1,
                dtype=T.i32,
            )
        )
        + extra
        for rr in range(R)
    ]
    # the page ids of the requests' first SPEC_CHUNKS chunks, loaded beside the
    # key counts (speculatively: a task whose chunks they cover takes its ids from
    # these registers instead of a dependent page-table load (and an LDS staging
    # with two barriers) once its chunk range is known): lane cc holds this
    # wave's two ids (key tiles kt) of chunk cc
    spv = []
    if const_expr(R <= SPEC_R):
        for rr in range_constexpr(R):
            spv.append(
                [
                    fx.Int32(
                        bo.buffer_load(
                            r_pt,
                            rr * table_stride + lane * 16 + 2 * wave + kt,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    for kt in range(2)
                ]
            )

    def seq_len_of(r):
        """Request r's keys (int32, or the low word of its int64), the step's rows included."""
        v = seqs[0]
        for rr in range_constexpr(1, R):
            v = (r == rr).select(seqs[rr], v)
        return v

    def chunk_range(keys, qpos_lo):
        """The chunks [lo, hi) of a request with ``keys`` keys a window from its
        first row's position leaves unmasked (every chunk of a full one)."""
        hi = udiv(keys + CHUNK - 1, CHUNK)
        ctx = keys - QL
        first = fx.max(ctx + qpos_lo - window + 1, fx.Int32(0))
        lo = (
            udiv(first, CHUNK)
            if SL
            else (
                fx.Int32(0)
                if FL
                else (window > 0).select(udiv(first, CHUNK), fx.Int32(0))
            )
        )
        return lo, hi

    # the partitions: a request's keys from its first unmasked chunk (a sliding
    # window's start; 0 for a full layer), the longest such span over at most
    # MAX_PARTS of them
    def span_of(r):
        lo_r, hi_r = chunk_range(seq_len_of(r), 0)
        return hi_r - lo_r

    max_span = span_of(fx.Int32(0))
    total_span = max_span
    for rr in range_constexpr(1, R):
        sp = span_of(fx.Int32(rr))
        max_span = fx.max(max_span, sp)
        total_span = total_span + sp

    # the partition (chunks a task) from the step's total chunk work, so that the
    # step has one task a CTA whatever the request mix (a task costs ~12 us beside
    # its chunks of ~1.8 us: its Q and page ids, the first chunk's exposed
    # latency, the key-split tree and the publish; eight 95k requests at 128
    # partitions each were 15 tasks a CTA, 212 us), the per-request rounding
    # checked so that no CTA gets a second task, and at least what keeps the
    # longest request within MAX_PARTS
    def tasks_of(pc_):
        n = fx.Int32(0)
        for rr in range_constexpr(R):
            lo_r, hi_r = chunk_range(seq_len_of(fx.Int32(rr)), 0)
            n = n + udiv((hi_r - lo_r) + pc_ - 1, pc_)
        return n * (LOCAL_KV_HEADS * TASK_TILES)

    per_cta = udiv(total_span * (LOCAL_KV_HEADS * TASK_TILES) + BLOCKS - 1, BLOCKS)
    floor_ = udiv(max_span + DIRECT_MAX - 1, DIRECT_MAX)
    pc = fx.max(fx.max(per_cta, floor_), fx.Int32(1))
    # (one request: pc >= floor_ = ceil(span / DIRECT_MAX) keeps its tasks within
    # LOCAL_KV_HEADS TASK_TILES DIRECT_MAX <= BLOCKS, so no rounding check moves pc:
    # the check's three dependent divisions are skipped)
    if const_expr(R > 1 or LOCAL_KV_HEADS * TASK_TILES * DIRECT_MAX > BLOCKS):
        # the check pc += tasks_of(pc) > BLOCKS three times, as one lane-parallel
        # probe: lane k tests pc + k (one vector division a request instead of
        # three dependent scalar ones); pc moves by the probes failing in a row
        # from k = 0 (at most 3, the third unchecked: the loop's result exactly)
        pc_l = pc + lane
        fails = ballot(tasks_of(pc_l) > BLOCKS)
        f0 = fx.Int32(fails & 1)
        f1 = fx.Int32((fails >> 1) & 1)
        f2 = fx.Int32((fails >> 2) & 1)
        pc = pc + f0 + f0 * f1 + f0 * f1 * f2
    part = (part_in > 0).select(fx.max(part_in, floor_ * CHUNK), pc * CHUNK)

    def n_parts_of(r):
        """Request r's partitions (its span's, the rest of the task slots idle)."""
        lo, hi = chunk_range(seq_len_of(r), 0)
        return udiv((hi - lo) * CHUNK + part - 1, part)

    # every request's partition count at once: lane rr divides request rr's
    # (one vector division; read back a lane each)
    np_lane = n_parts_of(fx.min(lane, fx.Int32(R - 1)))
    np_req = [read_lane_i32(np_lane, rr) for rr in range(R)]

    def n_parts_sel(r):
        """``n_parts_of(r)`` from the per-request counts."""
        v = np_req[0]
        for rr in range_constexpr(1, R):
            v = (r == rr).select(np_req[rr], v)
        return v

    def pflag_at(r, h, rt, p):
        """Pair index of the flag of the task holding row tile ``rt`` of (request r,
        head h) over partition p: a task's own (its first tile's), not the
        partition's: the tasks of a partition's row tiles finish apart (a merge
        that took the first one's flag read the other tiles' rows of the last layer)."""
        return ((r * LOCAL_KV_HEADS + h) * ROW_TILES + (rt // NT) * NT) * MAX_PARTS + p

    def part_blk(r, h, p, rt):
        """Float index of (request r, head h, partition p, row tile rt)'s block: O as
        four 16-dim chunks of [16 rows][16 dims] (1 KB each, a merge wave's load
        reads one whole), then the rows' (m, l) pairs; 4224 B (33 lines) a block, at
        most DIRECT_MAX partitions a (request, head) (within the region's bytes)."""
        return (((r * LOCAL_KV_HEADS + h) * DIRECT_MAX + p) * ROW_TILES + rt) * BLK

    # ============================================ attention tasks
    def attn_task(r, h, rt0, p, job):
        """Row tiles rt0 .. rt0 + NT of (request r, KV head h) over partition p: one
        K / V load and one V^T staging a chunk serve every tile of the task.
        ``job``: the task's index (the CTA's first is ``bid``)."""
        keys = seq_len_of(r)
        ctx = keys - QL
        ks = wave  # this wave's key split
        # Q: rows 16 (rt0 + it) + i = 8 qpos + hg of (request r, head h), 16 B a thread,
        # loaded at a block's start and stored to LDS after its first chunks' K / V
        # loads are issued (Q lands beside them instead of in front of the page ids;
        # loaded inside the block loop: a load before a loop that uses it makes the
        # compiler wait for it at the loop's preheader)
        q_thr = tid < NT * 16 * HEAD_DIM // 8
        i_t = udiv(tid, 16 * HEAD_DIM // 8)
        q_row = udiv(umod(tid, 16 * HEAD_DIM // 8), HEAD_DIM // 8)
        q_rowg = (rt0 + i_t) * 16 + q_row
        q_d0 = umod(tid, HEAD_DIM // 8) * 8
        q_t = r * QL + udiv(fx.min(q_rowg, ROWS - 1), GQA)
        q_off = udiv(
            q_t * Q_K
            + (h * GQA + umod(fx.min(q_rowg, ROWS - 1), GQA)) * HEAD_DIM
            + q_d0,
            2,
        )
        # S^T tiles: K is the MFMA's A operand (key l16, dims 8 g4 ..), Q^T its B
        # (row l16): a lane holds row l16 of each row tile and keys 4 g4 .. 4 g4 + 3 of
        # each key tile, so a row's running statistics are one lane's (four lanes
        # share a row), and its P values are packed in registers straight into the
        # PV MFMA's A operand (keys of tile 0 then tile 1; the V^T reads use the
        # same key order). The reference max of a row moves only when a chunk's max
        # exceeds it by TAU (lazy rescale, every row of the wave at once).
        r_lane = [(rt0 + it) * 16 + l16 for it in range(NT)]
        qpos = [
            fx.min(udiv(fx.min(r_lane[it], ROWS - 1), GQA), QL - 1) for it in range(NT)
        ]
        row_live = [r_lane[it] < ROWS for it in range(NT)]
        m_ref = [
            fx.Float32(NEG) for _ in range(NT)
        ]  # the row's reference max (its running max, lazily)
        l_lane = [
            fx.Float32(0.0) for _ in range(NT)
        ]  # this lane's keys' share of the row sum
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)] for _ in range(NT)
        ]
        lo, hi = chunk_range(keys, 0)
        c_lo = fx.max(lo, lo + p * udiv(part, CHUNK))
        c_hi = fx.min(hi, lo + (p + 1) * udiv(part, CHUNK))

        # The chunk loop: a wave's 32 keys of a chunk are two K pages and two V pages
        # (1 KB each, one head), landed in LDS by DMA (16 B a lane; chunk-major: dims
        # 16 c .. of key k at bytes 256 c + 16 k, conflict-free for both the K reads
        # and the transposed V reads) in the wave's ring slot; RING_DEPTH - 1 chunks
        # are in flight while one computes. (A VGPR-destination software pipeline was
        # drained by the compiler at every loop head: 6 us a chunk at 300k keys.)
        # The page ids of a block of IDS_CHUNKS chunks are staged in LDS first (a
        # CTA's first task within the first SPEC_CHUNKS chunks takes them from the
        # speculative registers loaded at entry instead, and a task of at most
        # REG_CHUNKS chunks from its own registers).
        assert KEY_TILES == 2 and V_STEPS == 1 and RING_DEPTH == 3  # (the ring waits)
        c_last = fx.max(c_hi - 1, c_lo)
        l16d, c16d = (
            lane % 16,
            lane // 16,
        )  # the DMA lane: key l16d, dims chunk c16d of a page
        # the transposed V read: this lane addresses key 4 g4 + (l16 / 2) % 4 of page (l16 / 2) / 4
        v_page = (l16 // 2) // 4
        v_key = 4 * g4 + (l16 // 2) % 4

        # the speculative ids serve the CTA's first task when they cover its chunks
        # (the later tasks' LDS ids and tree overwrote nothing of them, but the
        # sizing keeps a step at one task a CTA anyway)
        if const_expr(len(spv) > 0):
            use_spec = (job == bid) & (c_hi <= SPEC_CHUNKS)
            stage_ids = (job != bid) | (c_hi > SPEC_CHUNKS)
        else:
            use_spec = job < 0
            stage_ids = job >= 0
        # the task's own page ids when the speculative ones do not serve it and its
        # chunks fit a wave's lanes: lane cc holds this wave's two ids of chunk
        # c_lo + cc (one dependent load a wave instead of the LDS staging and its
        # two barriers)
        use_reg = stage_ids & (c_hi - c_lo <= REG_CHUNKS)
        stage_ids = stage_ids & (c_hi - c_lo > REG_CHUNKS)
        cc_r = fx.min(c_lo + lane, c_last)
        rgv = [
            fx.Int32(
                bo.buffer_load(
                    r_pt,
                    r * table_stride + cc_r * 16 + 2 * ks + kt,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            for kt in range(KEY_TILES)
        ]

        def issue(cc, sl, b_lo):
            """Chunk cc's four pages -> ring slot sl (four loads)."""
            slot = wring + sl * SLOT_WORDS
            for kt in range_constexpr(KEY_TILES):
                pg = fx.ptr_load(ids + ((cc - b_lo) * 16 + 2 * ks + kt))  # uniform
                if const_expr(len(spv) > 0):
                    sv = spv[0][kt]
                    for rr in range_constexpr(1, R):
                        sv = (r == rr).select(spv[rr][kt], sv)
                    pg_s = read_lane_i32(sv, fx.min(cc, fx.Int32(SPEC_CHUNKS - 1)))
                    pg = use_spec.select(pg_s, pg)
                # (the register ids' wait only on the path that reads them)
                if use_reg:
                    pg_r = read_lane_i32(rgv[kt], fx.min(cc - b_lo, fx.Int32(63)))
                    dma_pages(slot, kt, pg_r)
                else:
                    dma_pages(slot, kt, pg)

        def dma_pages(slot, kt, pg):
            """Page ``pg``'s K and V of key tile kt -> ring slot ``slot`` (two DMAs)."""
            voff = ((pg * 16 + l16d) * LOCAL_KV_HEADS + h) * HEAD_DIM + c16d * 16
            if const_expr(KV_ROWS > 0 and not SL):
                # a full layer reads the head-major pool (``kv_rows``): the head's 16 keys of a page are
                # 1 KB contiguous. The full-only build takes it whole; the any-layer build by window == 0
                hm_off = (h * KV_ROWS + pg * 16 + l16d) * HEAD_DIM + c16d * 16
                voff = hm_off if FL else (window == 0).select(hm_off, voff)
            if const_expr(SWA_ROWS > 0 and not FL):
                # a sliding layer reads the head-major SWA pool (``swa_rows``) the same way: the
                # sliding-only build whole, the any-layer build by window > 0
                hs_off = (h * SWA_ROWS + pg * 16 + l16d) * HEAD_DIM + c16d * 16
                voff = hs_off if SL else (window > 0).select(hs_off, voff)
            lds_dma_16(r_k, slot + kt * PAGE_WORDS, voff)
            lds_dma_16(r_v, slot + ((KEY_TILES + kt) * PAGE_WORDS + kt * V_SKEW), voff)

        def quad_max(v):
            """The max over the four lanes sharing a row (l16, l16 + 16, + 32, + 48)."""
            for off in (32, 16):
                x, y = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
                v = fx.max(x.bitcast(fx.Float32), y.bitcast(fx.Float32))
            return v

        def quad_sum(v):
            for off in (32, 16):
                x, y = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
                v = x.bitcast(fx.Float32) + y.bitcast(fx.Float32)
            return v

        n_blk = (c_hi - c_lo + IDS_CHUNKS - 1) // IDS_CHUNKS

        def stage_block(b_lo):
            gpu.barrier()  # the previous block's ids are consumed
            for i in range_constexpr(IDS_CHUNKS * 16 // THREADS):
                idx = tid + THREADS * i
                cc_i = fx.min(b_lo + idx // 16, c_last)
                pg = fx.Int32(
                    bo.buffer_load(
                        r_pt,
                        r * table_stride + cc_i * 16 + idx % 16,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                fx.ptr_store(pg, ids + idx)
            gpu.barrier()

        def issue_head(b_lo, b_last):
            # only real chunks are loaded (no repeated tail loads to wait for)
            issue(b_lo, 0, b_lo)
            for k in range_constexpr(1, RING_DEPTH):
                if b_lo + k <= b_last:
                    issue(b_lo + k, k, b_lo)

        # the first block's ids and first chunks before the block loop: the
        # loop's set-up (its accumulators' zeros, ...) follows the DMAs instead of
        # preceding them
        if n_blk > 0:
            if stage_ids:
                stage_block(c_lo)
            stamp(c, 3)  # a task's chunk loop begins (the last task's stamps survive)
            issue_head(c_lo, fx.min(c_lo + IDS_CHUNKS, c_hi) - 1)
        for b in range(0, n_blk):
            b_lo = c_lo + b * IDS_CHUNKS
            b_hi = fx.min(b_lo + IDS_CHUNKS, c_hi)
            b_last = b_hi - 1
            qv = fx.Vector(bo.buffer_load(r_q, q_off, vec_width=4, dtype=T.i32))
            if b > 0:
                stage_block(b_lo)
                issue_head(b_lo, b_last)
            # Q -> LDS (the compiler's wait for its load also covers the K / V DMAs
            # just issued, which the first chunk waits for anyway)
            if q_thr:
                fx.ptr_store(
                    qv.bitcast(fx.BFloat16), qs + ((i_t * 16 + q_row) * Q_PITCH + q_d0)
                )
            gpu.barrier()
            # each tile's A operands of this lane: row l16, dims 8 g4 + 32 step
            qa = [
                [
                    fx.Vector(
                        fx.ptr_load(
                            qs + ((it * 16 + l16) * Q_PITCH + 32 * step + 8 * g4),
                            result_type=V8B,
                        )
                    )  # fmt: skip
                    for step in range(2)
                ]
                for it in range(NT)
            ]
            for cc in range(b_lo, b_hi):
                sl = umod(cc - b_lo, RING_DEPTH)
                slot = wring + sl * SLOT_WORDS
                key0 = cc * CHUNK + ks * KEYS_PER_WAVE
                # chunk cc's pages landed (the later chunks' four DMAs each in flight,
                # when there are any)
                if cc + 2 <= b_last:
                    rocdl.s_waitcnt(vmcnt=8)
                elif cc < b_last:
                    rocdl.s_waitcnt(vmcnt=4)
                else:
                    rocdl.s_waitcnt(vmcnt=0)
                # the chunk's K operands (tile kt, step: key l16, dims 32 step + 8 g4)
                kb = []
                for kt in range_constexpr(KEY_TILES):
                    for step in range_constexpr(2):
                        widx = (
                            slot
                            + kt * PAGE_WORDS
                            + ((2 * step + g4 // 2) * 256 + l16 * 16 + (g4 % 2) * 8)
                            // 4
                        )
                        w = fx.Vector(fx.ptr_load(widx, result_type=V2I))
                        kb.append(fp8_to_bf16x8(w[0], w[1]))
                # ---- S^T: tile (it, kt): row l16, keys 4 g4 + e of key tile kt; scaled and masked
                inner = (cc != c_last) & (cc != lo)  # a chunk with no masked key
                # the mask  row_live & (inner | (kidx < ctx + qpos + 1 & (window == 0 |
                # ctx + qpos - kidx < window)))  as one unsigned range test a key:
                # lo <= kidx < hi  <=>  u32(kidx - lo) < u32(hi - lo) (wrapping int32,
                # exact for lo <= hi), lo = INT_MIN / hi = INT_MAX where a bound is
                # off, an empty range for a dead row (the same bits, a third of the
                # instructions and no scalar mask algebra)
                kb0 = key0 + 4 * g4
                rng = []
                for it in range_constexpr(NT):
                    lo_k = inner.select(
                        fx.Int32(-(2**31)),
                        (window == 0).select(
                            fx.Int32(-(2**31)), ctx + qpos[it] - window + 1
                        ),
                    )
                    hi_k = inner.select(fx.Int32(2**31 - 1), ctx + qpos[it] + 1)
                    rng.append(
                        (kb0 - lo_k, row_live[it].select(hi_k - lo_k, fx.Int32(0)))
                    )
                sc = []  # [it][kt * 4 + e]
                mx = []
                for it in range_constexpr(NT):
                    vals = []
                    for kt in range_constexpr(KEY_TILES):
                        st = fx.Vector.filled(4, 0.0, fx.Float32)
                        for step in range_constexpr(2):
                            st = mfma_bf16(kb[kt * 2 + step], qa[it][step], st)
                        for e in range_constexpr(4):
                            d0, wd = rng[it]
                            ok = fx.Uint32(d0 + (kt * TILE + e)) < fx.Uint32(wd)
                            vals.append(
                                ok.select(fx.Float32(st[e]) * qk_scale, fx.Float32(NEG))
                            )
                    sc.append(vals)
                    m8 = vals[0]
                    for j in range_constexpr(1, 2 * KEY_TILES * 4 // 2):
                        m8 = fx.max(m8, vals[j])
                    mx.append(quad_max(m8))
                # ---- the lazy rescale: when any row's chunk max exceeds its reference by TAU,
                # every row takes its new max and its sum and O are rescaled (selects:
                # the step is wave-uniform, the values are loop-carried)
                need = mx[0] > m_ref[0] + TAU
                for it in range_constexpr(1, NT):
                    need = need | (mx[it] > m_ref[it] + TAU)
                move = ballot(need) != 0
                alphas_acc = []
                for it in range_constexpr(NT):
                    m_new = move.select(fx.max(m_ref[it], mx[it]), m_ref[it])
                    m_g = (m_new > fx.Float32(NO_MAX)).select(m_new, fx.Float32(0.0))
                    alpha = hw_exp2(
                        m_ref[it] - m_g
                    )  # 0 from NEG (a first chunk), 1 when nothing moved
                    l_lane[it] = l_lane[it] * alpha
                    m_ref[it] = m_new
                    # O rows 4 g4 + e take the alpha of row 4 g4 + e: from lane 4 g4 + e (its g4 = 0 copy)
                    a_rows = [
                        fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32, (4 * g4 + e) * 4, alpha.bitcast(fx.Int32)
                            )
                        ).bitcast(fx.Float32)
                        for e in range(4)
                    ]
                    al = fx.Vector.from_elements(a_rows, fx.Float32)
                    accs[it] = [accs[it][dt] * al for dt in range(4)]
                # ---- P = exp2(S - m) as the PV A operand (eight bf16 a lane), the lane's row sums
                p_ops = []
                for it in range_constexpr(NT):
                    m_g = (m_ref[it] > fx.Float32(NO_MAX)).select(
                        m_ref[it], fx.Float32(0.0)
                    )
                    ps = [
                        hw_exp2(sc[it][j] - m_g) for j in range(2 * KEY_TILES * 4 // 2)
                    ]
                    acc_l = ps[0]
                    for j in range_constexpr(1, len(ps)):
                        acc_l = acc_l + ps[j]
                    l_lane[it] = l_lane[it] + acc_l
                    p_ops.append(
                        fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16)
                    )
                # ---- PV: V^T operands by transposed LDS reads (dim tile dt: lane (g4, l16) gets
                # dim 16 dt + l16 at keys 4 g4 .. of page 0 then of page 1)
                b_v = []
                for dt in range_constexpr(4):
                    widx = slot + ((KEY_TILES + v_page) * PAGE_WORDS + v_page * V_SKEW) + (dt * 256 + v_key * 16 + (l16 % 2) * 8) // 4  # fmt: skip
                    w = fx.Vector(
                        rocdl_raw.ds_read_tr8_b64(
                            T.vec(2, T.i32), widx.llvm_ptr
                        ).result,
                        (2,),
                        fx.Int32,
                    )  # gfx950 ds_read_b64_tr_b8
                    b_v.append(fp8_to_bf16x8(w[0], w[1]))
                accs = [
                    [mfma_bf16(p_ops[it], b_v[dt], accs[it][dt]) for dt in range(4)]
                    for it in range(NT)
                ]
                rocdl.s_waitcnt(lgkmcnt=0)  # every read of the slot is done
                if cc == b_lo:
                    stamp(c, 4)  # the first chunk done
                if cc + RING_DEPTH <= b_last:
                    # chunk cc + RING_DEPTH into this slot
                    issue(cc + RING_DEPTH, sl, b_lo)
            rocdl.s_waitcnt(
                vmcnt=0
            )  # (nothing in flight: the block's last chunk waited)
        stamp(c, 5)  # the chunk loop done
        gpu.barrier()  # every wave is done with the ring: the merge tree reuses the LDS

        # ---- the key splits merge in LDS in one round: every wave writes its
        # (m, l, O) rows (a (split, tile, row) record is COMB_ROW floats, O from
        # O_OFF); then each wave owns 2 NT of the task's rows, folds the eight splits
        # of them (a lane: one row, DPL dims) and publishes them with vector stores
        def comb_at(slot, it, row):
            return comb + ((slot * NT + it) * 16 + row) * COMB_ROW

        for it in range_constexpr(NT):
            l_row = quad_sum(l_lane[it])
            if g4 == 0:
                base = comb_at(ks, it, l16)
                fx.ptr_store(m_ref[it], base)
                fx.ptr_store(l_row, base + 1)
            for e in range_constexpr(4):
                base = comb_at(ks, it, 4 * g4 + e)
                for dt in range_constexpr(4):
                    fx.ptr_store(
                        fx.Float32(accs[it][dt][e]), base + (O_OFF + dt * 16 + l16)
                    )
        gpu.barrier()
        RPW = 2 * NT  # rows a wave: NT 16 / 8
        LPR = 64 // RPW  # lanes a row: 16 or 32
        DPL = HEAD_DIM // LPR  # dims a lane: 4 or 2
        VD = fx.Vector.make_type(DPL, fx.Float32)
        trow = wave * RPW + lane // LPR  # the task row (0 .. NT 16)
        it_f, row_f = trow // 16, trow % 16
        dd = (lane % LPR) * DPL
        m_f = fx.Float32(NEG)
        l_f = fx.Float32(0.0)
        o_f = fx.Vector.filled(DPL, 0.0, fx.Float32)
        for src in range_constexpr(KEY_SPLITS):
            base = comb_at(src, it_f, row_f)
            m2 = fx.ptr_load(base)
            l2 = fx.ptr_load(base + 1)
            v2 = fx.Vector(fx.ptr_load(base + (O_OFF + dd), result_type=VD))
            mn = fx.max(m_f, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m_f - mg)
            a2 = hw_exp2(m2 - mg)
            m_f, l_f, o_f = mn, l_f * a1 + l2 * a2, o_f * a1 + v2 * a2
        # ---- the task's partial rows: plain fp32 words (``part_blk`` layout), then the flag
        row = rt0 * 16 + trow
        if row < ROWS:
            blk = part_blk(r, h, p, rt0 + it_f)
            if lane % LPR == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([m_f, l_f], fx.Float32),
                    r_part,
                    blk + BLK_ML + row_f * 2,
                    cache_modifier=CM_DEV,
                )
            bo.buffer_store(
                o_f,
                r_part,
                blk + ((dd // 16) * 16 + row_f) * 16 + dd % 16,
                cache_modifier=CM_DEV,
            )
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(pflag, pflag_at(r, h, rt0, p), fx.Int32(1))

    # the tasks, request-major over the requests' own partition counts (a short
    # request has few): job -> (request, head, tile pair, partition) through the
    # requests' cumulative task counts, so a CTA iterates real tasks only
    counts = [np_req[rr] * (LOCAL_KV_HEADS * TASK_TILES) for rr in range(R)]
    starts = [fx.Int32(0)]
    for rr in range_constexpr(1, R):
        starts.append(starts[-1] + counts[rr - 1])
    total = starts[-1] + counts[-1]
    for job in range(bid, total, BLOCKS):
        r = fx.Int32(0)
        for rr in range_constexpr(1, R):
            r = (job >= starts[rr]).select(fx.Int32(rr), r)
        base = starts[0]
        for rr in range_constexpr(1, R):
            base = (r == rr).select(starts[rr], base)
        np_r = n_parts_sel(r)
        local = job - base
        # a partition's tasks of both KV heads and every row tile group (GC of them)
        # sit 8 jobs apart, on CTAs of one XCD (CTAs go to the XCDs in turn): they
        # stream the same K / V lines at about the same time (a key's two heads share
        # its 128 B lines), so each line comes from HBM once into that XCD's L2
        # (before, the tasks were np_r apart, on different XCDs, each reading the pages
        # from HBM). Groups of 8 partitions x GC jobs; the last np_r % 8 partitions in
        # a tail group.
        # (unsigned divisions: every operand is nonnegative)
        GC = LOCAL_KV_HEADS * TASK_TILES
        full = udiv(np_r, 8) * 8
        g, w = udiv(local, 8 * GC), umod(local, 8 * GC)
        jt = local - GC * full
        tail = fx.max(np_r - full, fx.Int32(1))
        in_full = local < GC * full
        jt_c = fx.max(jt, fx.Int32(0))  # (jt < 0 only in the unselected branch)
        jtq = udiv(jt_c, tail)
        p = in_full.select(g * 8 + umod(w, 8), full + (jt_c - jtq * tail))
        combo = in_full.select(udiv(w, 8), jtq)
        h = umod(combo, LOCAL_KV_HEADS)
        rt0 = udiv(combo, LOCAL_KV_HEADS) * NT
        attn_task(r, h, rt0, p, job)
    stamp(c, 1)  # the attention tasks done

    # ============================================ merge jobs
    # One job a (request, KV head, row tile) on one CTA, whatever the partition
    # count (at most DIRECT_MAX: the partition sizing above keeps n_parts there), in
    # the fold order of the two-level merge it replaces (bit for bit): with at most
    # DIRECT_FOLD partitions, partition lane pl folds partitions pl, pl + 4, .. and the
    # four lanes fold in lane order; with more, each of GROUPS contiguous groups of
    # ppg partitions folds that way into a group record, lane pl folds groups pl and
    # pl + 4 and the four lanes fold in lane order. A thread: partition lane
    # pl = tid % 4 (the lanes of a quad: DPP broadcasts combine them), dims
    # 16 (2 dh + j) + 4 dq .. + 4 of block chunk j = 0, 1 (dq = (tid / 4) % 4,
    # dh = tid / 256), row (tid / 16) % 16: a wave's load reads 256 B runs of four
    # blocks. A missing partition reads past the resource's end (zeros, no traffic).
    pl = tid % 4
    dq = (tid // 4) % 4
    row16 = (tid // 16) % 16
    dh = tid // 256
    mbid = (
        BLOCKS - 1 - bid
    )  # the last CTAs: the task loop's extra tasks sit on the first
    n_l2 = R * LOCAL_KV_HEADS * ROW_TILES
    part_floats = R * LOCAL_KV_HEADS * MAX_PARTS * ROWS * PART_ROW
    assert R * LOCAL_KV_HEADS * DIRECT_MAX * ROW_TILES * BLK <= part_floats
    assert (DIRECT_MAX + GROUPS - 1) // GROUPS <= 2 * MERGE_LANES

    # (a merged launch of S <= OG_TAGGED_MAX) the o rows also go out as tagged pairs
    # (a bf16 pair word, the launch tag) into K2's og region, o element i's pair at
    # dword i: K2's og rows poll the rows themselves, so no release, barrier, flag
    # poll and o load sit between the merge's stores and the o_proj
    og_tag = o_flag is not None and S <= OG_TAGGED_MAX
    if const_expr(og_tag):
        assert og is not None, "K2 polls the o rows as tagged pairs (og_round_poll)"
        og_v, o_tag = og.value, c["tag"]

    def og_put(words, at):
        """(``og_tag``) bf16 pair words ``words`` of o elements ``at`` .. as tagged pairs."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), o_tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(og_v),
            at,
            cache_modifier=CM_DEV,
        )

    def load_rec(r_src, blk, row_c):
        """A block's (m, l) of row ``row_c`` and this thread's 2 x 4 dims of O."""
        ml = fx.Vector(
            bo.buffer_load(
                r_src,
                blk + BLK_ML + row_c * 2,
                vec_width=2,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        o0 = fx.Vector(
            bo.buffer_load(
                r_src,
                blk + ((2 * dh) * 16 + row16) * 16 + dq * 4,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        o1 = fx.Vector(
            bo.buffer_load(
                r_src,
                blk + ((2 * dh + 1) * 16 + row16) * 16 + dq * 4,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        return ml, o0, o1

    def fold_rec(m, l, o0, o1, rec, ok):
        ml, p0, p1 = rec
        m2, l2 = fx.Float32(ml[0]), fx.Float32(ml[1])
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2

    def fold_all(srcs):
        """(m, l, O) from nothing with the states ``srcs`` folded in order."""
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        for m2, l2, p0, p1 in srcs:
            mn = fx.max(m, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m - mg)
            a2 = hw_exp2(m2 - mg)
            m, l, o0, o1 = mn, l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2
        return m, l, o0, o1

    def quad_bcast(v, src):
        """Lane ``src`` of this lane's quad's value of the f32 ``v`` (DPP quad_perm)."""
        x = v.bitcast(fx.Int32)
        return fx.Int32(update_dpp_i32(x, x, 0x55 * src, 0xF, 0xF, False)).bitcast(
            fx.Float32
        )

    def combine_lanes(m, l, o0, o1):
        """The four partition lanes of a quad fold in lane order (every lane the same)."""
        srcs = []
        for src in range_constexpr(MERGE_LANES):
            srcs.append(
                (
                    quad_bcast(m, src),
                    quad_bcast(l, src),
                    fx.Vector.from_elements([quad_bcast(fx.Float32(o0[j]), src) for j in range(4)], fx.Float32),
                    fx.Vector.from_elements([quad_bcast(fx.Float32(o1[j]), src) for j in range(4)], fx.Float32),
                )
            )  # fmt: skip
        return fold_all(srcs)

    def lane_fold(r_src, r, h, rt, row_c, p_lo, cnt, n_k):
        """Lane pl's partitions p_lo + 4 k + pl (k < n_k, those below p_lo + cnt)."""
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        ps = [p_lo + k * MERGE_LANES + pl for k in range(n_k)]
        oks = [pk < p_lo + cnt for pk in ps]
        recs = [
            load_rec(
                r_src, ok.select(part_blk(r, h, pk, rt), fx.Int32(part_floats)), row_c
            )
            for pk, ok in zip(ps, oks)
        ]
        for k in range_constexpr(n_k):
            m, l, o0, o1 = fold_rec(m, l, o0, o1, recs[k], oks[k])
        return m, l, o0, o1

    def merge_job(r, h, rt):
        """The partitions of (request r, head h) for row tile rt -> its o rows."""
        np_r = n_parts_sel(r)
        poll_flags(c, [(pflag, pflag_at(r, h, rt, 0), np_r)])

        def og_hint():
            # (``og_tag``) the job's o flag as its rows are stored, a hint: K2's og rows
            # poll the tagged rows once it is up, so their spin covers about the
            # stores' flight only
            if const_expr(og_tag):
                if tid == 0:
                    put(o_flag, (r * LOCAL_KV_HEADS + h) * ROW_TILES + rt, fx.Int32(1))

        r_src = rsrc(part_w.value, part_floats * 4)
        row_c = fx.min(rt * 16 + row16, ROWS - 1) - rt * 16
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)

        def store_o(m, l, o0, o1):
            og_hint()
            if (pl == 0) & (rt * 16 + row16 < ROWS):
                inv = fx.Float32(1.0) / l
                t = r * QL + (rt * 16 + row16) // GQA
                ob = t * Q_K + (h * GQA + (rt * 16 + row16) % GQA) * HEAD_DIM + dq * 4
                r_o = rsrc(a["o"])
                b0 = (o0 * inv).to(fx.BFloat16)
                b1 = (o1 * inv).to(fx.BFloat16)
                if const_expr(og_tag):
                    w0 = b0.bitcast(fx.Int32)
                    w1 = b1.bitcast(fx.Int32)
                    og_put([w0[0], w0[1]], ob + (2 * dh) * 16)
                    og_put([w1[0], w1[1]], ob + (2 * dh + 1) * 16)
                bo.buffer_store(b0, r_o, ob + (2 * dh) * 16, cache_modifier=CM_DEV)
                bo.buffer_store(b1, r_o, ob + (2 * dh + 1) * 16, cache_modifier=CM_DEV)

        def direct(n_k):
            return combine_lanes(
                *lane_fold(r_src, r, h, rt, row_c, fx.Int32(0), np_r, n_k)
            )

        def grouped(n_rec):
            """The level-1 groups then the level-2 fold, in their order, on one CTA:
            wave g holds group g (lane: row lane / 4, dims 16 q + 4 (lane % 4) .. + 4 of
            each chunk q, so a load reads a block's whole chunk), each thread folds the
            four partition lanes' states itself, the group records meet in LDS."""
            g = wave
            grow = lane // 4
            gq = lane % 4
            grow_c = fx.min(rt * 16 + grow, ROWS - 1) - rt * 16
            ppg = (np_r + GROUPS - 1) // GROUPS
            p_lo = g * ppg
            cnt = fx.max(fx.min(np_r - p_lo, ppg), fx.Int32(0))
            recs = []
            oks = []
            for i in range_constexpr(n_rec):
                ok = cnt > i
                blk = ok.select(part_blk(r, h, p_lo + i, rt), fx.Int32(part_floats))
                ml = fx.Vector(
                    bo.buffer_load(
                        r_src,
                        blk + BLK_ML + grow_c * 2,
                        vec_width=2,
                        dtype=T.f32,
                        cache_modifier=CM_DEV,
                    )
                )
                os_ = [
                    fx.Vector(bo.buffer_load(r_src, blk + (q * 16 + grow) * 16 + gq * 4, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
                    for q in range(4)
                ]  # fmt: skip
                recs.append((ml, os_))
                oks.append(ok)
            # partition lane s: partitions p_lo + s, p_lo + s + 4 (ppg <= 8)
            states = []
            for s_ in range_constexpr(MERGE_LANES):
                m = fx.Float32(NEG)
                l = fx.Float32(0.0)
                ov = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
                for i in range_constexpr(s_, n_rec, MERGE_LANES):
                    ml, os_ = recs[i]
                    ok = oks[i]
                    m2, l2 = fx.Float32(ml[0]), fx.Float32(ml[1])
                    mn = fx.max(m, m2)
                    mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
                    a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
                    a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
                    m, l = ok.select(mn, m), l * a1 + l2 * a2
                    ov = [ov[q] * a1 + os_[q] * a2 for q in range(4)]
                states.append((m, l, ov))
            m = fx.Float32(NEG)
            l = fx.Float32(0.0)
            ov = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
            for m2, l2, pv in states:
                mn = fx.max(m, m2)
                mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
                a1 = hw_exp2(m - mg)
                a2 = hw_exp2(m2 - mg)
                m, l = mn, l * a1 + l2 * a2
                ov = [ov[q] * a1 + pv[q] * a2 for q in range(4)]
            # the group record -> LDS: (m, l) pairs, then O rows of pitch G_PITCH
            if gq == 0:
                fx.ptr_store(m, comb + (g * 16 + grow) * 2)
                fx.ptr_store(l, comb + ((g * 16 + grow) * 2 + 1))
            for q in range_constexpr(4):
                fx.ptr_store(
                    ov[q], comb + (G_O + (g * 16 + grow) * G_PITCH + q * 16 + gq * 4)
                )
            gpu.barrier()
            # level 2: thread (row tid / 32, dims 2 (tid % 32) ..): lane pl's groups pl,
            # pl + 4, then the lanes in order
            frow = tid // 32
            fd = (tid % 32) * 2
            V2F = fx.Vector.make_type(2, fx.Float32)
            gs = []
            for gg in range_constexpr(GROUPS):
                gs.append(
                    (
                        fx.ptr_load(comb + (gg * 16 + frow) * 2),
                        fx.ptr_load(comb + ((gg * 16 + frow) * 2 + 1)),
                        fx.Vector(fx.ptr_load(comb + (G_O + (gg * 16 + frow) * G_PITCH + fd), result_type=V2F)),
                    )
                )  # fmt: skip

            def fold2(srcs):
                m = fx.Float32(NEG)
                l = fx.Float32(0.0)
                ov = fx.Vector.filled(2, 0.0, fx.Float32)
                for m2, l2, pv in srcs:
                    mn = fx.max(m, m2)
                    mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
                    a1 = hw_exp2(m - mg)
                    a2 = hw_exp2(m2 - mg)
                    m, l, ov = mn, l * a1 + l2 * a2, ov * a1 + pv * a2
                return m, l, ov

            lanes = [fold2([gs[s_], gs[s_ + MERGE_LANES]]) for s_ in range(MERGE_LANES)]
            m, l, ov = fold2(lanes)
            og_hint()
            if rt * 16 + frow < ROWS:
                inv = fx.Float32(1.0) / l
                t = r * QL + (rt * 16 + frow) // GQA
                bv = (ov * inv).to(fx.BFloat16)
                o_at = t * Q_K + (h * GQA + (rt * 16 + frow) % GQA) * HEAD_DIM + fd
                if const_expr(og_tag):
                    og_put([bv.bitcast(fx.Int32)[0]], o_at)
                bo.buffer_store(bv, rsrc(a["o"]), o_at, cache_modifier=CM_DEV)

        # (a lane's partitions past the count are no-ops of the fold: the loads a
        # count needs, the result the same bits)
        if np_r <= 2 * MERGE_LANES:
            m, l, o0, o1 = direct(2)
            store_o(m, l, o0, o1)
        elif np_r <= DIRECT_FOLD:
            m, l, o0, o1 = direct(DIRECT_FOLD // MERGE_LANES)
            store_o(m, l, o0, o1)
        elif np_r <= GROUPS * 4:
            grouped(4)
        elif np_r <= GROUPS * 6:
            grouped(6)
        else:
            grouped(8)
        if const_expr(o_flag is not None and not og_tag):
            # the (request, head, row tile)'s o rows are out: its flag, for the og tasks
            release_stores()
            gpu.barrier()
            if tid == 0:
                put(o_flag, (r * LOCAL_KV_HEADS + h) * ROW_TILES + rt, fx.Int32(1))

    if mbid < n_l2:
        rt = mbid % ROW_TILES
        h = (mbid // ROW_TILES) % LOCAL_KV_HEADS
        r = mbid // (ROW_TILES * LOCAL_KV_HEADS)
        merge_job(r, h, rt)
    stamp(c, 2)  # the merge done (end)
