# SPDX-License-Identifier: Apache-2.0
"""K2: one Muse Spark layer from the attention output to the experts' input,
for S = ``tokens`` rows; per token::

    o (bf16, the attention output), gate (K1's o_gate)
      -> per-head RMSNorm without weight (eps RMS_EPS) x sigmoid(gate), one bf16
         rounding (the fused norm-gate)
      -> o_proj GEMV (bf16 MFMA, fp32 accumulate, one rounding) -> the bf16
         partial rows pushed to every rank's ``attn`` region -> rank-ordered fp32
         sum, one rounding (the aiter one-stage all-reduce's numerics)
      -> attention boundary (the fused post-norm / residual / pre-norm with a
         weightless post norm): nb = bf16(attn_out rstd) (eps POST_EPS);
         stream = alpha stream + beta nb in fp32, in place; x = bf16(stream rstd2
         w_pre) (eps RMS_EPS), the FFN input
      -> router: the fp32 weight as bf16 hi + lo (the joint router / pooler's
         split): x @ hi + x @ lo in fp32, x ROUTER_MULT, sigmoid, + expert bias,
         top-8 by one integer max a pick (a tie to the lower expert), the picks'
         sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE -> ids (int32), scores (f32)
      -> pooler GEMV (bf16, one rounding) -> pre-expert norm (eps RMS_EPS,
         weight) -> expert input (bf16 [S, MOE_HIDDEN])

The work is task lists over the CTAs. Loads complete in order, so a CTA's
mailbox poll waits behind its own weight loads in flight: the last
``MAX_TOKENS`` CTAs (the chain CTAs, from ``CHAIN0``) issue their pooler rows'
weights at once, landed before their first poll, and run the per-token tasks
unqueued; the other CTAs (the stream CTAs) issue the pooler's weights after
their o tasks and wait on the tokens' flags behind them, as they must anyway::

    og      S tasks, stream CTA OG0 + t: token t's normed, gated o -> ``og``
    o       512 tasks (16 rows each), two a CTA on every CTA: o_proj partial
            rows -> every rank's ``attn`` region
    a_part  a_parts(S) tasks a token, chain CTAs AP0 ..: the all-reduce sum of
            the part's columns (the ranks' pairs polled in one round), the boundary
            with the row's sums of squares exchanged through ``ss``, x -> ``xbuf``
            (plain bf16 words), the stream in place, part 0 the token's ``xflag``
    router  128 tasks (16 expert tiles x an eighth of K), stream CTAs R0 ..:
            partial logits -> ``logits`` [part][token][expert]
    pool    256 tasks (128 rows x an eighth of K), one a CTA, the same eighth as
            the CTA's router task: partial pooled rows -> ``pool`` [part][token][row]
    route   S tasks, chain CTA CHAIN0 + t: top-8 -> ids / scores
    xn      S tasks, stream CTA t (beside the route): the pre-expert norm -> expert input

Width scaling (8 .. 64 rows): the o GEMV holds every token's og row in LDS
(S x 2 KB) and runs a token tile of 16 an MFMA accumulator, each tile's rows
summed through one reduction row set in turn; the router / pooler GEMVs load
each token tile's x of their K eighth straight into the B operands (16 B a lane a
step, no staging: a task reads S x 2 KB of x, not S x 16 KB) and sum their row
tiles through ``red_sets(S)`` row sets a round; the boundary runs ``a_parts(S)``
= 8 / 4 / 2 / 1 CTAs a token on the 64 chain CTAs.

The geometry constants, ``k2_scratch_layout`` and ``sym_layout`` belong in
``layout.py`` and live here until the K2 / K3 ports merge: ``sym_layout`` is the
rank's peer buffer for the whole layer (K3 pushes the FFN partials to ``ffn``
and the unpooled rows to ``unp``), so the K3 port imports it from here.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_to_bf16x8,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    permlane_swap,
    row_sum,
    rsrc,
    uniform,
    xshfl,
)
from sgl_amd_mono.common.plan import BLOCKS, PAIR_BYTES, THREADS, WAVES, pair_layout
from sgl_amd_mono.common.sync import MASK_OOB, SENT, _spin_masked, preg, sent_word
from sgl_amd_mono.muse_spark.gfx950.config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    TOP_K,
)
from sgl_amd_mono.muse_spark.gfx950.device import (
    acquire_loads,
    poll_flags,
    release_stores,
    stamp,
)
from sgl_amd_mono.muse_spark.gfx950.layout import (
    GQA,
    K_STEP,
    RED_FLOATS,
    ROWS,
    STEP_BATCH,
    TILE,
    TRACE_LAYERS,
    TRACE_SLOTS,
    scratch_bytes,
    token_tiles,
)
from sgl_amd_mono.muse_spark.gfx950.stages.attention import OG_TAGGED_MAX

# ---------------------------------------------------------------- K2 geometry (for layout.py)
O_TASKS = HIDDEN // ROWS  # 512 o_proj row groups
O_K_PER_WAVE = Q_K // WAVES  # 128: the o GEMV's K over the waves
O_STEPS = O_K_PER_WAVE // K_STEP  # 4 MFMA steps a wave a token tile
OG_THREADS = Q_K // 4  # 256: the og task's threads, 4 o elements each
OG_POLL = 8  # tokens an og-row poll on a stream CTA (16 B a thread a token)
K_PER_WAVE = HIDDEN // WAVES  # 1024: the router / pooler GEMVs' K over the waves
K_STEPS = K_PER_WAVE // K_STEP  # 32
R_TILES = EXPERTS // ROWS  # 16 expert tiles
R_PARTS = WAVES  # the router's K split: a task takes one wave's share of K
R_TASKS = R_TILES * R_PARTS  # 128: partial logits, summed by the route tasks
P_ROWS = 128  # pooler rows a task: eight MFMA row tiles
P_SUB = (
    P_ROWS // ROWS
)  # 8: the largest pooler job (row tiles); the LDS reduction is sized for it
P_TILES = MOE_HIDDEN // P_ROWS  # 32 pooler row tiles
# The pooler's 256 row tiles a K eighth over the 32 CTAs of that eighth (bid % 8): the
# 24 stream CTAs take NS tiles each, the 8 chain CTAs NC. A chain CTA issues its
# weights only after its boundary task (loads complete in order: issued before, they
# would queue the boundary's polls), so its job is small and lands at once; the
# stream CTAs' bigger jobs stream during the o tasks and the boundary.
P_NS = 10  # 24 x 10 = 240 row tiles
P_NC = 2  # 8 x 2 = 16
assert 24 * P_NS + 8 * P_NC == MOE_HIDDEN // ROWS and P_NS <= P_SUB + 2
P_PARTS = R_PARTS  # the pooler's K split is the router's: a CTA's two tasks share a K eighth of x
from sgl_amd_mono.muse_spark.gfx950.stages.ffn import POOL_PARTS  # noqa: E402

assert POOL_PARTS == P_PARTS
P_STEPS = (
    K_STEPS // WAVES
)  # 4 MFMA steps a wave a (row tile, token tile): a part's K over the waves
A_PARTS_MAX = 8  # CTAs a token's attention boundary at most
A_PAIRS_MAX = HIDDEN // (
    2 * THREADS
)  # 8: a thread's column pairs with the whole row on one CTA
CHAIN0 = BLOCKS - MAX_TOKENS  # 192: the chain CTAs, token bid - CHAIN0's route and xn
AP0 = CHAIN0  # the boundary tasks: the chain CTAs, before their pooler weights go out
A_TASKS_MAX = BLOCKS - CHAIN0  # 64 boundary tasks at most
R0 = 0  # the router tasks
ROUTE0 = (
    MAX_TOKENS  # the route tasks: stream CTA ROUTE0 + t, past the xn tasks (0 .. S)
)
assert O_TASKS == 2 * BLOCKS and R0 + R_TASKS <= CHAIN0  # two o tasks a CTA, every CTA
assert (
    MAX_TOKENS <= CHAIN0 and MAX_TOKENS <= A_TASKS_MAX and ROUTE0 + MAX_TOKENS <= CHAIN0
)
assert K_STEPS % STEP_BATCH == 0 and K_STEPS % WAVES == 0 and O_K_PER_WAVE % K_STEP == 0
assert (
    EXPERTS == 4 * 64 and TOP_K <= 64
)  # a lane holds 4 routing keys; the picks on one wave
assert HIDDEN == 16 * THREADS and MOE_HIDDEN == 8 * THREADS and Q_K == 2 * THREADS
assert (
    P_TILES * P_PARTS == BLOCKS
    and MAX_TOKENS % TILE == 0
    and OG_THREADS <= THREADS
    and (BLOCKS - CHAIN0) == 8 * 8
    and CHAIN0 == 24 * 8
)


# The o_proj all-reduce at TP 8 as a reduce-scatter + all-gather (``o_rs``): rank r owns
# the HIDDEN / 8 columns r HIDDEN / 8 .. (o tasks 64 r ..); every rank pushes an o task's
# bf16 partial rows only to its owner (``attn`` [src][token][col], as before); the owner's
# reduce CTAs (RED_CTA0 + j: stream CTAs without a router task) sum the 8 partials in rank
# order in fp32 from rank 0's, one bf16 rounding (the one-shot sum, bit for bit), and push
# the bf16 sums to every rank's ``attn`` [0][AG_T0 + token][col] (rank 0's slot rows past
# the step's tokens, unused by the reduce-scatter); the boundary reads the sums there.
RS_TP = 8  # the ranks of the reduce-scatter path (other TP sizes: the one-shot push)
RS_MAX_TOKENS = 16  # widths up to this (the all-gather rows sit at token AG_T0 ..)
AG_T0 = MAX_TOKENS // 2
assert RS_MAX_TOKENS <= AG_T0
RS_TASKS = O_TASKS // RS_TP  # 64 o tasks an owner
RED_CTA0 = R0 + R_TASKS  # 128: the reduce CTAs, past the router tasks' CTAs
RED_T0 = 4 * 64  # the reduce threads: from wave 4 (the o pushes run on waves 0, 1)
RED_PW = 2  # pairs a reduce thread (4 columns: 16 B a spec)
assert RED_CTA0 + RS_TASKS <= CHAIN0 and (ROWS // 2) * RS_MAX_TOKENS <= RED_T0


def o_rs(c) -> bool:
    """The o_proj all-reduce runs as reduce-scatter + all-gather (TP 8, S <= 16)."""
    return c["W"] == RS_TP and c["S"] <= RS_MAX_TOKENS


# The owner-side boundary statistics (``ob``, TP 8, S = 8 / 16): boundary part q of
# token t's four norm sums (``stage_a_part``'s) are formed on the rank that receives the
# part's o partials (``ob_owner``: part q's rank at 8 parts; at 4, one of its two column
# slices' ranks by the token's parity), straight from the reduce-scatter's partial rows,
# by one chain CTA a (token, part) (``stage_ob_stats``: ``stage_a_part``'s columns a
# thread, the same o sums, terms and ``block_sum4``), and pushed to every rank's
# statistics row (``XST0``: ``wait_x``'s ``ss``). The x operands' first half b1 =
# bf16(pre_w alpha st) needs no o sums: every CTA forms it for its own K part into LDS
# (``b1_store``); the second half b2 = bf16(pre_w beta v) comes from the all-gathered o
# sums v, now sentinel words (``ag_word``), polled with the statistics in one spin
# (``x_loads_ob``). The boundary CTAs' hop (all-gather -> x halves and sums through
# ``xbuf`` / ``ss`` -> every CTA) is gone from the chain; the stream update stays on the
# chain CTAs (``stage_a_part_res``, from the all-gathered sums and the rstd every CTA
# formed). Same arithmetic, bit for bit.
XST_T0 = AG_T0 + RS_MAX_TOKENS  # attn rows [0][48 ..]: the statistics (``ob``)
XST0 = (
    XST_T0 * HIDDEN // 2
)  # their first pair: (token t, part q)'s four at 4 (t A_PARTS_MAX + q)
OB_PB = 3  # ``red`` row set: the K part's pre_w beta products (``b1_store``)
OB_TASKS = 8  # statistics tasks a rank (chain CTAs AP0 ..)
assert (MAX_TOKENS - XST_T0) * HIDDEN // 2 >= 4 * RS_MAX_TOKENS * A_PARTS_MAX
# (``ob``) token t's xn task on stream CTA XN0_OB + t (reduce CTAs, no router task), not
# CTA t: the xn CTAs enter K3 ~2 us after the others, and the low CTAs carry the extra
# ug item of a plan's partial round (``ffn.ug_count``) -- on the heavy EP rank the xn
# CTAs ended the ug stage last
XN0_OB = CHAIN0 - RS_MAX_TOKENS
assert (
    XN0_OB >= RED_CTA0
    and XN0_OB + RS_MAX_TOKENS <= CHAIN0
    and XN0_OB >= ROUTE0 + MAX_TOKENS
)


def xn_cta0(c):
    """The first xn CTA (token t's: ``xn_cta0`` + t)."""
    return XN0_OB if ob(c) else 0


def ob(c) -> bool:
    """The owner-side boundary statistics (sharded pooler, reduce-scatter, a whole
    number of 8-token groups)."""
    return o_rs(c) and c["pshard"] and c["S"] % 8 == 0


def ob_g(s: int) -> int:
    """Ranks a boundary part's columns span (``RS_TP / a_parts``)."""
    return RS_TP // a_parts(s)


def ob_owner(s: int, q, t):
    """The rank of token ``t``'s boundary part ``q`` (``ob``)."""
    g = ob_g(s)
    return q * g + t % g if g > 1 else q


def ob_task(s: int, rank, j):
    """Statistics task ``j`` (0 .. OB_TASKS) on ``rank``: (token, part)."""
    g = ob_g(s)
    assert s * a_parts(s) == RS_TP * OB_TASKS
    if g > 1:
        return j * g + rank % g, rank // g
    return j, rank


# (``ob``) the all-gather rows are sentinel words, not tagged pairs: a (token, column
# pair)'s bf16 sums as one word (``sent_word``) at word ``ag_word``; the o task's CTA
# re-arms its tasks' words (SENT) before its partials go out, so a word is pushed only
# after its re-arm landed (the push follows the owner's reduce of those partials), and
# a reader takes any word off SENT as this launch's
def ag_word(t, col):
    """The all-gather word of token ``t``'s columns ``col``, ``col`` + 1 (``ob``)."""
    return ((AG_T0 + t) * HIDDEN + col) // 2


@flyc.jit
def ag_rearm(c):
    """(``ob``) this CTA's o tasks' all-gather words (this rank's rows) := SENT, at the
    CTA's K2 entry; ``release_stores`` + a barrier before its partials go out (the
    owner reduces them and only then pushes the sums here)."""
    S, tid, bid = c["S"], c["tid"], c["bid"]
    n = (ROWS // 2) * S  # a task's words
    assert 2 * n <= THREADS
    if tid < 2 * n:
        task = bid + (tid // n) * BLOCKS
        i = tid % n
        bo.buffer_store(
            fx.Int32(SENT), rsrc(c["own_attn"].value),
            ag_word(i // (ROWS // 2), task * ROWS + (i % (ROWS // 2)) * 2), cache_modifier=CM_SYS,
        )  # fmt: skip


def ag_put(c, dst, t, col, sums):
    """(``ob``) ``sums`` (4: columns ``col`` ..) of token ``t`` as two sentinel words to
    rank ``dst``'s all-gather row (system scope)."""
    words = [
        sent_word(bf16_pair(sums[2 * k], sums[2 * k + 1]).bitcast(fx.Int32))
        for k in range(2)
    ]
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32), rsrc(dst.value), ag_word(t, col), cache_modifier=CM_SYS
    )  # fmt: skip


def poll_sent_vec(base, offs, n, cm, gap=1):
    """Words ``off`` .. + ``n`` (n <= 4, one load each) of the sentinel buffer at
    ``base`` for every ``off`` in ``offs`` once none is SENT, re-polling only the
    specs still pending (no re-arm: their writer re-arms them); [[n words], ...]."""
    r = rsrc(base, MASK_OOB)
    N = len(offs)

    def done_of(prev, e):
        d = prev[n * e] != fx.Int32(SENT)
        for q in range_constexpr(1, n):
            d = d & (prev[n * e + q] != fx.Int32(SENT))
        return d

    def load(prev):
        words = []
        for e in range_constexpr(N):
            off = fx.Int32(offs[e])
            if const_expr(prev is not None):
                d = done_of(prev, e)
                off = d.select(fx.Int32(MASK_OOB // 4), off)
            w = bo.buffer_load(r, off, vec_width=n, dtype=T.i32, cache_modifier=cm)
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
        bad = v[0] == fx.Int32(SENT)
        for e in range_constexpr(1, n * N):
            bad = bad | (v[e] == fx.Int32(SENT))
        return bad

    v = _spin_masked(load, pending, gap)
    return [[v[n * e + q] for q in range(n)] for e in range(N)]


def pick_base(c, r):
    """Rank ``r``'s (traced) peer buffer base, picked from the bases the CTA holds."""
    W, bases = c["W"], c["bases"]
    b = bases[W - 1]
    for p in range(W - 2, -1, -1):
        b = (r == p).select(bases[p], b)
    return b


def a_parts(s: int) -> int:
    """CTAs a token's attention boundary at width ``s``: the largest power of two
    (at most A_PARTS_MAX) whose tasks fit the A_TASKS_MAX chain CTAs. (One task a
    token pulls 8 ranks x 16 KB of partials into one CTA: measured no better on one
    GPU and worse in the TP 8 server than eight parts with their three hops.)"""
    p = A_PARTS_MAX
    while p > 1 and p * s > A_TASKS_MAX:
        p //= 2
    assert p * s <= A_TASKS_MAX and A_PAIRS_MAX % p == 0, (s, p)
    return p


def k2_scratch_layout() -> dict:
    """K2's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS:
    the tokens' normed gated o (bf16 pairs), a flag a token for its x, the
    boundary parts' norm sums (the flags of their x halves), the
    router parts' partial logits (fp32), the pooled rows (bf16 pairs), x as plain
    bf16 words in two halves (``k2_xbuf``, ``k2_xbuf2``, bytes given as pairs), a
    flag a boundary part once its stream columns are out, and the merged launch's
    hand-off flags: a token's routing / expert input out, a (request, KV head,
    row tile)'s o rows out."""
    return pair_layout(
        [
            ("k2_og", MAX_TOKENS * Q_K // 2),
            ("k2_xflag", MAX_TOKENS),
            ("k2_ss", 4 * MAX_TOKENS * A_PARTS_MAX),
            ("k2_logits", R_PARTS * MAX_TOKENS * EXPERTS),
            ("k2_pool", P_PARTS * MAX_TOKENS * MOE_HIDDEN // 2),
            ("k2_xbuf", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_xbuf2", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_c1", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            # the expert input's MXFP4 blocks a token: MOE_HIDDEN / 8 code words, then MOE_HIDDEN / 32
            # scale words (K3's staged-column layout, ``ffn.stage_xq_all``)
            (
                "k2_xinq",
                MAX_TOKENS * (MOE_HIDDEN // 8 + MOE_HIDDEN // 32) * 4 // PAIR_BYTES,
            ),
            ("k2_resflag", MAX_TOKENS * A_PARTS_MAX),
            # a merged launch's routing hand-off: token t's eight ids as bytes of two
            # tagged words at pairs 2 t, 2 t + 1 (the words are the flag; the ids /
            # scores outputs are stored before them)
            ("k2_rflag", MAX_TOKENS * 2),
            ("k2_xinflag", MAX_TOKENS),
            ("k2_oflag", MAX_TOKENS * LOCAL_KV_HEADS * 4),
            ("k2_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric buffer, name -> (byte offset, bytes): the attention
    and FFN partial rows, [source rank][token][width] as bf16 pairs, and the
    unpooled rows [token][HIDDEN / 2] (each rank's rows from that rank). Shared
    by the layer: K2 pushes ``attn``, K3 ``ffn`` and ``unp``. The guard's peer
    flags follow at ``sym_bytes(npes)``."""
    from sgl_amd_mono.muse_spark.gfx950.stages.ffn import k3_sym_layout

    return k3_sym_layout(
        npes
    )  # one layout: K3's (attn, ffn, unp, then the TP-experts regions)


def sym_bytes(npes: int) -> int:
    """The peer buffer's data bytes: the guard flags' offset."""
    return scratch_bytes(sym_layout(npes))


def k2_xs_floats(tokens: int) -> int:
    """The ``xs`` LDS region: every token's og row (the o GEMV's B rows)."""
    return tokens * Q_K // 2


def red_sets(tokens: int) -> int:
    """Reduction row sets of the router / pooler passes (a row tile each a round):
    what the LDS leaves beside the og rows."""
    return (
        P_NS + 1 if tokens <= 16 else (4 if tokens <= 32 else 2)
    )  # one round of 9 tiles up to 16 rows


def k2_lds_bytes(tokens: int) -> int:
    return 4 * (
        k2_xs_floats(tokens)
        + red_sets(tokens) * RED_FLOATS
        + ROWS * MAX_TOKENS
        + EXPERTS
    )


def k2_lds(tokens: int):
    """K2's LDS for a step of ``tokens`` rows: the o GEMV's B rows, then the
    router / pooler GEMVs' staged x (the same region), one reduction row set, a
    task's output rows for every token, a token's sigmoids."""

    @fx.struct
    class Lds:
        xs: fx.Array[fx.Float32, k2_xs_floats(tokens), 16]
        red: fx.Array[fx.Float32, red_sets(tokens) * RED_FLOATS, 16]
        misc: fx.Array[fx.Float32, ROWS * MAX_TOKENS, 16]
        sig: fx.Array[fx.Float32, EXPERTS, 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def _cs(x, y):
    """A compare-exchange, descending."""
    return fx.max(x, y), fx.min(x, y)


def _sort8_bitonic(v):
    """A bitonic list of 8 keys sorted descending (three compare-exchange levels)."""
    v = list(v)
    for st in (4, 2, 1):
        for i in range(8):
            if i & st == 0:
                v[i], v[i + st] = _cs(v[i], v[i + st])
    return v


def top8_sorted(keys):
    """The wave's 8 largest of its 4 keys a lane (distinct), descending, on every lane:
    each lane sorts its 4, then six butterfly steps (xor 1 .. 32) each keep the top 8
    of its list and its partner's (a bitonic merge of two descending lists: the same
    set on both lanes). The picks are the 8 rounds of wave maxima's, in their order;
    ~45 independent VALU ops a step instead of 8 dependent rounds of six shuffles."""
    assert len(keys) == 4 and TOP_K == 8
    a, b = _cs(keys[0], keys[1])
    c, d = _cs(keys[2], keys[3])
    a, c = _cs(a, c)
    b, d = _cs(b, d)
    b, c = _cs(b, c)
    own = [a, b, c, d]
    # xor 1: two sorted 4s -> the sorted 8 of both (own descending, the partner's ascending: bitonic)
    oth = [xshfl(v, 1) for v in own]
    top = _sort8_bitonic(own + oth[::-1])
    for off in (2, 4, 8, 16, 32):
        if off >= 16:
            pairs = [
                permlane_swap(off, v, v) for v in top
            ]  # (the lower lane's, the upper lane's)
            lo, hi = [p[0] for p in pairs], [p[1] for p in pairs]
        else:
            lo, hi = top, [xshfl(v, off) for v in top]
        top = _sort8_bitonic([fx.max(lo[i], hi[7 - i]) for i in range(8)])
    return top


def route_key(v, e):
    """Signed-orderable i32 of score ``v`` with its low 8 bits replaced by 255 -
    expert ``e``: a pick is one integer max, a tie goes to the lower expert."""
    b = v.bitcast(fx.Int32)
    b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
    return (b & fx.Int32(-256)) | (255 - e)


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` over its threads, in wave order; ``slot`` picks the
    WAVES words of ``red`` (consecutive calls alternate slots)."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ws, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(red + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


@flyc.jit
def block_sum4(c, v0, v1, v2, v3):
    """Four CTA sums at once (one barrier): each wave's four sums to ``red`` slots 0
    .. 3, every thread adds the waves in wave order."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = [butterfly(v, (32, 16, 8, 4, 2, 1)) for v in (v0, v1, v2, v3)]
    if lane == 0:
        for k in range_constexpr(4):
            fx.ptr_store(ws[k], red + (k * WAVES + wave))
    gpu.barrier()
    outs = []
    for k in range_constexpr(4):
        tot = fx.ptr_load(red + k * WAVES)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + (k * WAVES + w))
        outs.append(tot)
    return outs


def ld_bf(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) bf16 at element ``e`` (even) of a bf16 buffer, as
    f32; at most 4 dwords a load."""
    out = []
    for e0 in range(0, n, 8):
        m = min(8, n - e0)
        w = bo.buffer_load(
            rsrc(ptr), (e + e0) // 2, vec_width=m // 2, dtype=T.i32, cache_modifier=cm
        )
        if m == 2:
            lo, hi = bf2_f32(fx.Int32(w))
            out += [lo, hi]
        else:
            v = fx.Vector(w).bitcast(fx.BFloat16).to(fx.Float32)
            out += [fx.Float32(v[j]) for j in range(m)]
    return out


def ld_f32(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) f32 at element ``e`` of an f32 buffer; at most 4 a load."""
    out = []
    for e0 in range(0, n, 4):
        m = min(4, n - e0)
        w = bo.buffer_load(
            rsrc(ptr), e + e0, vec_width=m, dtype=T.f32, cache_modifier=cm
        )
        v = fx.Vector(w)
        out += [fx.Float32(v[j]) for j in range(m)]
    return out


def st_f32(ptr, e, vals, cm=0):
    """``vals`` (2, 4, 8 or 16 f32) at element ``e`` of an f32 buffer; at most 4 a store."""
    for e0 in range(0, len(vals), 4):
        chunk = vals[e0 : e0 + 4]
        bo.buffer_store(
            fx.Vector.from_elements(chunk, fx.Float32),
            rsrc(ptr),
            e + e0,
            cache_modifier=cm,
        )


def st_bf(ptr, e, vals, cm=0):
    """``vals`` (4 or 8 f32) as bf16 at element ``e`` of a bf16 buffer."""
    for e0 in range(0, len(vals), 4):
        bo.buffer_store(
            fx.Vector.from_elements(vals[e0 : e0 + 4], fx.Float32).to(fx.BFloat16),
            rsrc(ptr),
            e + e0,
            cache_modifier=cm,
        )


def st_words(ptr, d, words, cm=0):
    """``words`` (1 .. 8 i32) at dword ``d`` of a buffer; at most 4 a store."""
    for d0 in range(0, len(words), 4):
        chunk = words[d0 : d0 + 4]
        if len(chunk) == 1:
            bo.buffer_store(chunk[0], rsrc(ptr), d + d0, cache_modifier=cm)
        else:
            bo.buffer_store(
                fx.Vector.from_elements(chunk, fx.Int32),
                rsrc(ptr),
                d + d0,
                cache_modifier=cm,
            )


# ---------------------------------------------------------------- og tasks
# ---------------------------------------------------------------- o tasks
def o_w_loads(c, task):
    """The o_proj weight words of o task ``task``'s rows, this lane's: in flight
    (a task past the last reads past the buffer: zeros, no traffic). Row-major
    bf16, 16 B a lane a k-step, read once a layer (nontemporal)."""
    lane, wave = c["lane"], c["wave"]
    row = task * ROWS + lane % 16
    r_wo = rsrc(c["args"]["w_o"], HIDDEN * Q_K * 2)
    return [
        fx.Vector(
            bo.buffer_load(
                r_wo,
                (row * Q_K + wave * O_K_PER_WAVE + s * K_STEP + 8 * (lane // 16)) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for s in range(O_STEPS)
    ]


@flyc.jit
def stage_og_rows(c, o_flag=None, QL=8, g_pre=None, swz=False):
    """Every token's normed gated o into LDS, computed here from o and the gate (16
    heads of 64: per-head RMSNorm without weight x sigmoid(gate), one bf16
    rounding; thread tid holds head (tid / 16) % 16, dims 4 (tid % 16) .. of
    token 2 round + tid / 256): the o tasks' B rows. It used to be an og task a
    token and a mailbox hop. ``o_flag`` (a merged launch): the requests' (KV head,
    row tile) flags first, then device-scope loads; at S <= ``OG_TAGGED_MAX`` the
    flags are a hint and the o words are polled as the merge's tagged pairs
    (``og_round_poll``). ``g_pre`` (``og_gate_pre``): the gate words, issued before
    the o weights: the sigmoids (most of the work) are computed before the flags'
    wait."""
    S, tid, lane, xs = c["S"], c["tid"], c["lane"], c["xs"]
    TPR = THREADS // OG_THREADS  # tokens a round: 2
    NR = -(-S // TPR)  # rounds
    e = og_elem(c)
    sig_pre = None
    if const_expr(g_pre is not None):
        # the sigmoids now, parked in this thread's words of ``red`` (free until the o
        # tasks): kept in registers, the compiler sinks them past the flags' wait
        assert NR * THREADS * 4 <= red_sets(S) * RED_FLOATS
        for r in range_constexpr(NR):
            sg = [sigmoid(x) for x in bf4_f32(g_pre[r])]
            fx.ptr_store(
                fx.Vector.from_elements(sg, fx.Float32),
                c["red"] + (r * THREADS + tid) * 4,
            )
        sig_pre = True
    o_cm = 0  # o from memory the host wrote: the default cache policy
    # (``attention.run_kattn``'s ``og``) the merge's o rows as tagged pairs: polled
    # straight after the flags (raised as the rows are stored), no acquire and no
    # second round trip for the o words
    tagged = o_flag is not None and S <= OG_TAGGED_MAX
    assert not tagged or sig_pre is not None  # (one og batch: ``og_gate_pre``)
    if const_expr(o_flag is not None):
        o_cm = CM_DEV  # the attention's device-scope stores, earlier in this launch
        rocdl.sched_barrier(0)
        RT = (QL * GQA + 15) // 16  # the attention's row tiles a (request, KV head)
        poll_flags(c, [(o_flag, (S // QL) * LOCAL_KV_HEADS * RT)])
        if const_expr(not tagged):
            acquire_loads()
    if const_expr(tagged):
        rocdl.sched_barrier(0)
    # every round's o (and gate) words in flight at once (OG_BATCH rounds a batch), not
    # one round trip a round (the LDS stores between rounds kept each round's loads
    # after them)
    for b in range_constexpr(-(-NR // OG_BATCH)):
        r0, r1 = b * OG_BATCH, min(NR, (b + 1) * OG_BATCH)
        if const_expr(tagged):
            batch = og_round_poll(c, r0, r1, e, OG_POLL_GAP)
        else:
            batch = og_round_loads(c, r0, r1, e, o_cm, gate=sig_pre is None)
        rocdl.sched_barrier(0)  # the batch's loads issued together, not one at each use
        for k in range_constexpr(len(batch)):
            t0, t, ow, gw = batch[k]
            ov = bf4_f32(ow)
            if const_expr(sig_pre is not None):
                sv = fx.Vector(
                    fx.ptr_load(
                        c["red"] + ((r0 + k) * THREADS + tid) * 4,
                        result_type=fx.Vector.make_type(4, fx.Float32),
                    )
                )
                sg = [fx.Float32(sv[j]) for j in range(4)]
            else:
                sg = [sigmoid(x) for x in bf4_f32(gw)]
            ss = fx.Float32(0.0)
            for j in range_constexpr(4):
                ss = ss + ov[j] * ov[j]
            ss = butterfly(ss, (8, 4, 2, 1))
            rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
            vals = [ov[j] * rstd * sg[j] for j in range(4)]
            if t0 + tid // OG_THREADS < S:
                fx.ptr_store(bf16_pair(vals[0], vals[1]), xs + xs_word(t, e // 2, swz))
                fx.ptr_store(
                    bf16_pair(vals[2], vals[3]), xs + xs_word(t, e // 2 + 1, swz)
                )
    gpu.barrier()


def og_elem(c):
    """This thread's first o element of a token row (head (tid / 16) % 16, 4 dims)."""
    return ((c["tid"] % OG_THREADS) // 16) * HEAD_DIM + (c["lane"] % 16) * 4


def og_gate_pre(c):
    """The gate words of every og round (S <= 2 OG_BATCH), issued now: before the o
    weights, which the og rows' o loads wait behind anyway (loads complete in order);
    None for wider steps."""
    S, tid, a = c["S"], c["tid"], c["args"]
    TPR = THREADS // OG_THREADS
    NR = -(-S // TPR)
    if NR > OG_BATCH:
        return None
    e = og_elem(c)
    out = []
    for r in range(NR):
        t = fx.min(r * TPR + tid // OG_THREADS, S - 1)
        out.append(
            bo.buffer_load(rsrc(a["g"]), (t * Q_K + e) // 2, vec_width=2, dtype=T.i32)
        )
    return out


OG_BATCH = 8  # og rounds whose o / gate words are in flight together (16 VGPRs a round)


def og_round_loads(c, r0, r1, e, o_cm, gate=True):
    """The o (and ``gate``: gate) words of og rounds ``r0`` .. ``r1`` (this thread's 4
    bf16 a round, two dwords), issued together: [(t0, t, o words, gate words)]."""
    S, tid, a = c["S"], c["tid"], c["args"]
    TPR = THREADS // OG_THREADS
    srcs = [(a["o"], o_cm)] + ([(a["g"], 0)] if gate else [])
    out = []
    for r in range(r0, r1):
        t0 = r * TPR
        t = fx.min(t0 + tid // OG_THREADS, S - 1)
        words = [
            bo.buffer_load(
                rsrc(ptr),
                (t * Q_K + e) // 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=cm,
            )
            for ptr, cm in srcs
        ]
        out.append((t0, t, words[0], words[1] if gate else None))
    return out


OG_POLL_GAP = 2  # (``og_round_poll``) s_sleep between the tagged rows' re-polls


def og_round_poll(c, r0, r1, e, gap=1):
    """``og_round_loads``' o words of og rounds ``r0`` .. ``r1`` from the merge's tagged
    pairs (``k2_og``: o element i's pair word at dword i, the tag after it), polled
    device-scope until every pair carries this launch's tag, re-polling only the
    rounds still pending: [(t0, t, o words, None)]."""
    S, tid, tag = c["S"], c["tid"], c["tag"]
    TPR = THREADS // OG_THREADS
    r = rsrc(c["og"].value, MASK_OOB)
    OOB = fx.Int32(MASK_OOB // 4)
    rounds = list(range(r0, r1))
    offs = [fx.min(rr * TPR + tid // OG_THREADS, S - 1) * Q_K + e for rr in rounds]

    def done(v, k):
        return (v[4 * k + 1] == tag) & (v[4 * k + 3] == tag)

    def load(prev):
        # every round's load issued before any select (the selects into the loop's
        # registers otherwise each waited for their load: a round trip a round)
        ws, dks = [], []
        for k in range_constexpr(len(rounds)):
            off = offs[k]
            if const_expr(prev is not None):
                dk = done(prev, k)
                off = dk.select(OOB, off)
                dks.append(dk)
            ws.append(
                fx.Vector(
                    bo.buffer_load(
                        r, off, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
                    )
                )
            )
        rocdl.sched_barrier(0)
        words = []
        for k in range_constexpr(len(rounds)):
            for q in range_constexpr(4):
                w = fx.Int32(ws[k][q])
                words.append(w if prev is None else dks[k].select(prev[4 * k + q], w))
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = ~done(v, 0)
        for k in range_constexpr(1, len(rounds)):
            bad = bad | ~done(v, k)
        return bad

    v = _spin_masked(load, pending, gap)
    out = []
    for k in range_constexpr(len(rounds)):
        rr = rounds[k]
        t0 = rr * TPR
        t = fx.min(t0 + tid // OG_THREADS, S - 1)
        out.append(
            (t0, t, fx.Vector.from_elements([v[4 * k], v[4 * k + 2]], fx.Int32), None)
        )
    return out


def bf4_f32(w):
    """Two dwords of 4 bf16 (``og_round_loads``) as 4 f32, as ``ld_bf`` converts them."""
    v = fx.Vector(w).bitcast(fx.BFloat16).to(fx.Float32)
    return [fx.Float32(v[j]) for j in range(4)]


# LDS layouts without bank conflicts (same values, same sums):
# - (the fused o pass) token t's og row in ``xs`` with its 16 B chunks
#   XOR-swizzled by t % 16: the o MFMAs' B loads (16 tokens, a lane each, 2 KB apart)
#   were 16-way conflicts;
# - a wave's MFMA accumulators in ``red`` with each 16-lane group rotated
#   by 4 lanes a group: ``row_sum``'s readers (row r, token t: lane 16 (r / 4) + t,
#   element r % 4) were 4-way conflicts.


def xs_word(t, w, swz):
    """Word ``w`` of token ``t``'s og row in ``xs`` (``swz``: chunks XOR t % 16)."""
    if not swz:
        return t * (Q_K // 2) + w
    return t * (Q_K // 2) + (((w >> 2) ^ (t & 15)) << 2) + (w & 3)


def acc_slot(lane):
    """A wave's accumulator slot (16 B) of lane ``lane``."""
    g = lane >> 4
    return (g << 4) + (((lane & 15) + (g << 2)) & 15)


def acc_store(red, wave, lane, acc):
    """This lane's MFMA accumulators into the row set ``red`` (``acc_row_sum`` reads)."""
    fx.ptr_store(acc, red + (wave * 64 + acc_slot(lane)) * 4)


def acc_row_sum(red, r, t):
    """``ops.row_sum`` of a row set ``acc_store`` filled: row r, token t, wave order."""
    g = r >> 2
    slot = (g << 4) + ((t + (g << 2)) & 15)
    tot = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + slot) * 4 + (r & 3)))
    return tot


@flyc.jit
def stage_o(c, task, wts):
    """o task ``task``: rows ``16 task ..`` of o_proj for every token tile (lane l
    of wave w holds row l % 16 and k = 128 w + 32 s + 8 (l / 16) .. of token
    16 t + l % 16, from LDS), the waves' accumulators summed in wave order, one
    bf16 rounding, the partial rows pushed to every rank's ``attn`` region.
    ``wts``: the task's weights, in flight (``o_w_loads``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    xs, red, misc, put_bf = c["xs"], c["red"], c["misc"], c["put_bf"]
    W, bases, rank, sy_attn = c["W"], c["bases"], c["rank"], c["sy_attn"]
    l16, g4 = lane % 16, lane // 16
    V4F = fx.Vector.make_type(4, fx.Float32)
    tiles = token_tiles(S)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tiles)]
    for s in range_constexpr(O_STEPS):
        k0 = wave * O_K_PER_WAVE + s * K_STEP + 8 * g4
        wa = wts[s].bitcast(fx.BFloat16)
        for t in range_constexpr(tiles):
            tok = fx.min(t * TILE + l16, S - 1)
            xb = fx.Vector(
                fx.ptr_load(xs + (tok * (Q_K // 2) + k0 // 2), result_type=V4F)
            )
            accs[t] = mfma_bf16(wa, xb.bitcast(fx.BFloat16), accs[t])
    # each tile's accumulators through the one reduction row set in turn (LDS)
    for t in range_constexpr(tiles):
        fx.ptr_store(accs[t], red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < ROWS * TILE:
            r = tid % ROWS
            tk = t * TILE + tid // ROWS
            if tk < S:
                fx.ptr_store(
                    bf16_round(row_sum(red, r, tid // ROWS)), misc + (tk * ROWS + r)
                )
        gpu.barrier()
    # pair i: token i / 8, rows 2 (i % 8), + 1; to every rank (itself too), or (``o_rs``)
    # to the rows' owner only
    rs = o_rs(c)
    g2 = ob(c) and ob_g(S) > 1
    if const_expr(rs and not g2):
        dst_rs = preg(pick_base(c, task // RS_TASKS), sy_attn, "attn")
    n_pairs = (ROWS // 2) * S
    if const_expr(g2):
        # (``ob``, two ranks a part) a token's owner by its parity: a wave pushes one
        # parity (its 8 tokens' row pairs), so the owner is uniform in the wave
        assert n_pairs == 2 * 64 and ob_g(S) == 2
        q = task * ROWS // (HIDDEN // a_parts(S))
        dst_rs = preg(pick_base(c, q * 2 + uniform(c["wave"] % 2)), sy_attn, "attn")
    for rnd in range_constexpr(-(-n_pairs // THREADS)):
        i = tid + rnd * THREADS
        if i < n_pairs:
            tk = i // (ROWS // 2)
            rr = (i % (ROWS // 2)) * 2
            if const_expr(g2):
                tk = 2 * ((i % 64) // (ROWS // 2)) + i // 64
            v0 = fx.ptr_load(misc + (tk * ROWS + rr))
            v1 = fx.ptr_load(misc + (tk * ROWS + rr + 1))
            if const_expr(rs):
                put_bf(
                    dst_rs,
                    (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr,
                    [v0, v1],
                )
            else:
                for p in range_constexpr(W):
                    put_bf(
                        preg(bases[p], sy_attn, "attn"),
                        (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr,
                        [v0, v1],
                    )
    gpu.barrier()  # ``misc`` is reused by the CTA's next task


# ---------------------------------------------------------------- the o tasks, fused (S <= 16)
# The level-2 merge CTAs (the attention's last ``o_merge_ctas`` CTAs) enter K2 last
# (~8 us), so their o weights went out last and their og rows' o loads queued behind
# them: their o tasks' partials were the reduce-scatter's last by ~0.7 us. In a merged
# launch their tasks move to helper CTAs (a third task each, ``o_help0`` ..), and a
# CTA's tasks run as one pass: one MFMA loop sharing the B rows, one LDS reduction,
# the pushes on disjoint waves (the same sums, wave order and rounding as ``stage_o``).


def o_fused(c) -> bool:
    """The fused o pass: one token tile (S <= 16)."""
    return token_tiles(c["S"]) == 1


def o_merge_ctas(c, o_flag, QL) -> int:
    """The level-2 merge CTAs whose o tasks move to helpers (0: none moved): the
    attention's ``R LOCAL_KV_HEADS ROW_TILES`` last CTAs (``attention.run_kattn``'s
    ``mbid``), in a merged launch."""
    if o_flag is None or not o_fused(c):
        return 0
    n = (c["S"] // QL) * LOCAL_KV_HEADS * ((QL * GQA + 15) // 16)
    h0 = o_help0(c)
    if h0 + 2 * n > BLOCKS - n or (h0 < CHAIN0 and h0 + 2 * n > CHAIN0):
        return 0
    return n


def o_help0(c) -> int:
    """The first helper CTA (a merge CTA's two tasks on two helpers): the chain CTAs past
    the statistics tasks (``ob``: no boundary task there), else the stream CTAs past
    the router tasks."""
    return AP0 + OB_TASKS if ob(c) else RED_CTA0


def o_extra_task(c, nm):
    """(task, live) of this CTA's third o task (a helper's: a merge CTA's), else an
    index past the last task (its weight loads read past the buffer: zeros, no traffic)."""
    h = fx.Int32(c["bid"]) - fx.Int32(o_help0(c))
    live = (h >= fx.Int32(0)) & (h < fx.Int32(2 * nm))
    hh = live.select(h, fx.Int32(0))
    task = (BLOCKS - nm) + hh % nm + (hh // nm) * BLOCKS
    return live.select(task, fx.Int32(O_TASKS)), live


@flyc.jit
def ag_rearm_tasks(c, tasks, lives):
    """``ag_rearm`` of the o tasks ``tasks`` (``lives``: a task's predicate or None)."""
    S, tid = c["S"], c["tid"]
    n = (ROWS // 2) * S  # a task's words
    assert len(tasks) * n <= THREADS
    for k in range_constexpr(len(tasks)):
        i = tid - k * n
        ok = (i >= 0) & (i < n)
        if const_expr(lives[k] is not None):
            ok = ok & lives[k]
        if ok:
            bo.buffer_store(
                fx.Int32(SENT), rsrc(c["own_attn"].value),
                ag_word(i // (ROWS // 2), tasks[k] * ROWS + (i % (ROWS // 2)) * 2), cache_modifier=CM_SYS,
            )  # fmt: skip


@flyc.jit
def stage_o_fused(c, tasks, wts, lives):
    """``stage_o`` of the o tasks ``tasks`` at once (one token tile): the MFMAs of every
    task on each B row load, the accumulators through ``red`` row set k a task (one
    barrier), task k's partial rows pushed by threads k n_pairs .. (``lives``: a task's
    predicate or None)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    xs, red, misc, put_bf = c["xs"], c["red"], c["misc"], c["put_bf"]
    W, bases, rank, sy_attn = c["W"], c["bases"], c["rank"], c["sy_attn"]
    NT_ = len(tasks)
    assert token_tiles(S) == 1 and NT_ * RED_FLOATS <= red_sets(S) * RED_FLOATS
    assert NT_ * ROWS * S <= ROWS * MAX_TOKENS
    l16, g4 = lane % 16, lane // 16
    V4F = fx.Vector.make_type(4, fx.Float32)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(NT_)]
    tok = fx.min(l16, S - 1)
    for s in range_constexpr(O_STEPS):
        k0 = wave * O_K_PER_WAVE + s * K_STEP + 8 * g4
        xb = fx.Vector(fx.ptr_load(xs + xs_word(tok, k0 // 2, True), result_type=V4F))
        for k in range_constexpr(NT_):
            accs[k] = mfma_bf16(
                wts[k][s].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), accs[k]
            )
    for k in range_constexpr(NT_):
        acc_store(red + k * RED_FLOATS, wave, lane, accs[k])
    gpu.barrier()
    if tid < ROWS * TILE:
        r = tid % ROWS
        tk = tid // ROWS
        if tk < S:
            for k in range_constexpr(NT_):
                fx.ptr_store(
                    bf16_round(acc_row_sum(red + k * RED_FLOATS, r, tk)), misc + (k * ROWS * S + tk * ROWS + r)
                )  # fmt: skip
    gpu.barrier()
    rs = o_rs(c)
    g2 = ob(c) and ob_g(S) > 1
    n_pairs = (ROWS // 2) * S
    assert NT_ * n_pairs <= THREADS
    for k in range_constexpr(NT_):
        task = tasks[k]
        if const_expr(rs and not g2):
            dst_rs = preg(pick_base(c, task // RS_TASKS), sy_attn, "attn")
        if const_expr(g2):
            assert n_pairs == 2 * 64 and ob_g(S) == 2
            q = task * ROWS // (HIDDEN // a_parts(S))
            dst_rs = preg(pick_base(c, q * 2 + uniform(c["wave"] % 2)), sy_attn, "attn")
        i = tid - k * n_pairs
        ok = (i >= 0) & (i < n_pairs)
        if const_expr(lives[k] is not None):
            ok = ok & lives[k]
        if ok:
            tk = i // (ROWS // 2)
            rr = (i % (ROWS // 2)) * 2
            if const_expr(g2):
                tk = 2 * ((i % 64) // (ROWS // 2)) + i // 64
            v0 = fx.ptr_load(misc + (k * ROWS * S + tk * ROWS + rr))
            v1 = fx.ptr_load(misc + (k * ROWS * S + tk * ROWS + rr + 1))
            if const_expr(rs):
                put_bf(
                    dst_rs,
                    (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr,
                    [v0, v1],
                )
            else:
                for p in range_constexpr(W):
                    put_bf(
                        preg(bases[p], sy_attn, "attn"),
                        (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr,
                        [v0, v1],
                    )
    gpu.barrier()  # ``misc`` is reused by the CTA's next stage


@flyc.jit
def run_o_split(c, o_flag, QL):
    """The og rows and the CTA's two o tasks as two ``stage_o`` passes (S > 16)."""
    bid = c["bid"]
    if const_expr(ob(c)):
        ag_rearm(c)
    g_pre = og_gate_pre(
        c
    )  # the gate words first: the sigmoids need not wait for the weights
    rocdl.sched_barrier(0)
    w_first = o_w_loads(c, bid)
    w_second = o_w_loads(c, bid + BLOCKS)
    stage_og_rows(c, o_flag, QL, g_pre)
    if const_expr(ob(c)):
        release_stores()  # the re-arm (long landed); stage_o's barriers precede its pushes
    stamp(c, 12)  # the og rows in LDS
    stage_o(c, bid, w_first)
    stage_o(c, bid + BLOCKS, w_second)


@flyc.jit
def run_o_fused(c, o_flag, QL, tasks, wtasks, lives):
    """The og rows and the o tasks ``tasks`` (weights of ``wtasks``) in one fused pass."""
    if const_expr(ob(c)):
        ag_rearm_tasks(c, tasks, lives)
    g_pre = og_gate_pre(
        c
    )  # the gate words first: the sigmoids need not wait for the weights
    rocdl.sched_barrier(0)
    wts = [o_w_loads(c, t) for t in wtasks]
    stage_og_rows(c, o_flag, QL, g_pre, swz=True)
    if const_expr(ob(c)):
        release_stores()  # the re-arm (long landed); the fused pass's barriers precede its pushes
    stamp(c, 12)  # the og rows in LDS
    stage_o_fused(c, tasks, wts, lives)


@flyc.jit
def run_o(c, o_flag, QL):
    """The CTA's og rows and o tasks: ``bid``, ``bid + BLOCKS`` and a helper's third
    (``o_merge_ctas``: none on a merge CTA in a merged launch), in one fused pass; two
    ``stage_o`` passes at S > 16."""
    bid = c["bid"]
    if const_expr(not o_fused(c)):
        run_o_split(c, o_flag, QL)
    else:
        nm = o_merge_ctas(c, o_flag, QL)
        b32 = fx.Int32(bid)
        if const_expr(nm > 0):
            # a merge CTA (the last nm) has no o task and forms no og rows: its x polls
            # read the all-gather words only past the statistics (``x_poll_ob``), so its
            # early start needs no re-arm landed before it by time
            xt, live = o_extra_task(c, nm)
            if b32 < fx.Int32(BLOCKS - nm):
                run_o_fused(
                    c,
                    o_flag,
                    QL,
                    [b32, b32 + BLOCKS, xt],
                    [b32, b32 + BLOCKS, xt],
                    [None, None, live],
                )
        else:
            run_o_fused(
                c, o_flag, QL, [b32, b32 + BLOCKS], [b32, b32 + BLOCKS], [None, None]
            )


def rs_partial_sums(c, ws, NP, PW):
    """The all-reduced o values of ``NP`` column pairs from the W ranks' polled partial
    pairs ``ws`` (spec (j / PW) W + src: rank src's pairs PW (j / PW) ..): each pair
    summed in rank order in fp32 from rank 0's, one bf16 rounding -- the one-shot
    push's sums, bit for bit. The reduce-scatter owner forms them here (its o reduce
    and the boundary statistics) and all-gathers them; 2 NP values (f32 of bf16)."""
    W = c["W"]
    A_PARTS = a_parts(c["S"])
    v = []
    for j in range_constexpr(NP):
        acc0, acc1 = bf2_f32(ws[(j // PW) * W][j % PW])
        for src in range_constexpr(1, W):
            lo, hi = bf2_f32(ws[(j // PW) * W + src][j % PW])
            acc0 = acc0 + lo
            acc1 = acc1 + hi
        v += [bf16_round(acc0), bf16_round(acc1)]
    if const_expr(A_PARTS > 1):
        # (a token's boundary in several parts) the pairs lie in one part's columns
        assert 2 * NP <= HIDDEN // A_PARTS
    return v


@flyc.jit
def stage_o_reduce(c, j):
    """The owner's reduce of its o task ``j`` (0 .. RS_TASKS; o task 64 rank + j: 16
    columns x S tokens), on threads RED_T0 .. (waves without o pushes: a poll waits
    behind its own wave's stores): a thread 4 columns of a token, the 8 ranks' bf16
    partials summed in fp32 in rank order from rank 0's, one bf16 rounding (what the
    boundary formed from the one-shot push, bit for bit), pushed to every rank's
    all-gather rows (``attn`` [0][AG_T0 + token]) in a rotated rank order."""
    S, tid, W, rank = c["S"], c["tid"], c["W"], c["rank"]
    own, poll, put_bf = c["own_attn"], c["poll"], c["put_bf"]
    NQ = ROWS // (2 * RED_PW)  # threads a token: 4
    i = tid - RED_T0
    if (tid >= RED_T0) & (tid < RED_T0 + NQ * S):
        tk = i // NQ
        col = (rank * RS_TASKS + j) * ROWS + (i % NQ) * (2 * RED_PW)
        ws = poll(
            [
                (own, ((src * MAX_TOKENS + tk) * HIDDEN + col) // 2, RED_PW)
                for src in range(W)
            ],
            batch=W,
        )
        stamp(c, 14, RED_T0)  # the ranks' partials of the reduce arrived
        sums = rs_partial_sums(c, ws, RED_PW, RED_PW)  # (bf16 values: pushed as bf16)
        for k in range_constexpr(W):
            dst = preg(pick_base(c, (rank + 1 + k) % W), c["sy_attn"], "attn")
            if const_expr(ob(c)):
                ag_put(c, dst, tk, col, sums)
            else:
                put_bf(dst, (AG_T0 + tk) * HIDDEN + col, sums)
        stamp(c, 15, RED_T0)  # the sums pushed


def ob_poll(c, t, col, NC):
    """The 8 ranks' o partial pairs of token ``t``'s columns ``col`` .. + NC (pushed here
    by ``stage_o``: this rank owns the token's part), re-polling only the late ones."""
    return c["poll_masked"](
        [(c["own_attn"], ((src * MAX_TOKENS + t) * HIDDEN + col) // 2, NC // 2) for src in range(c["W"])],
    )  # fmt: skip


def ob_sums(c, NC, ws):
    """The o sums of ``NC`` columns from the 8 ranks' polled partial pairs ``ws``, in rank
    order in fp32 from rank 0's: (fp32 sums, their bf16 values as f32) -- the
    all-gather's sums and the boundary's ``o_sum`` of them, bit for bit."""
    # (the bf16 values push as the same bf16 pairs as the fp32 sums: one rounding)
    v = rs_partial_sums(c, ws, NC // 2, NC // 2)
    return v, v


@flyc.jit
def stage_o_reduce_ob(c, j):
    """``stage_o_reduce`` at two ranks a boundary part (``ob``, ``ob_g`` = 2): this rank
    holds the partials of part q = rank / 2's columns for the tokens of its parity
    (``ob_owner``); reduce CTA ``j`` (0 .. 64) takes 32 of the columns of those S / 2
    tokens, a thread 4 columns of a token on the waves from RED_T0, the 8 ranks'
    partials summed in rank order in fp32 from rank 0's, one bf16 rounding, pushed to
    every rank's all-gather rows in a rotated rank order."""
    S, tid, W, rank = c["S"], c["tid"], c["W"], c["rank"]
    A_PARTS = a_parts(S)
    g = ob_g(S)
    NQ = 4 * g  # threads a token
    i = tid - RED_T0
    assert NQ * (S // g) <= THREADS - RED_T0
    if (i >= 0) & (i < NQ * (S // g)):
        t = (i // NQ) * g + rank % g
        col = (rank // g) * (HIDDEN // A_PARTS) + j * (16 * g) + (i % NQ) * 4
        ws = c["poll"](
            [(c["own_attn"], ((src * MAX_TOKENS + t) * HIDDEN + col) // 2, RED_PW) for src in range(W)],
            batch=W,
        )  # fmt: skip
        stamp(c, 14, RED_T0)  # the ranks' partials of the reduce arrived
        sums, _ = ob_sums(c, 4, ws)
        for k in range_constexpr(W):
            dst = preg(pick_base(c, (rank + 1 + k) % W), c["sy_attn"], "attn")
            ag_put(c, dst, t, col, sums)
        stamp(c, 15, RED_T0)  # the sums pushed


@flyc.jit
def stage_ob_stats(c, j):
    """Boundary part ``ob_task(j)``'s norm statistics on its owner (``ob``), a chain CTA:
    ``stage_a_part``'s columns a thread, the o sums from the 8 ranks' partials as the
    all-gather forms them, ``stage_a_part``'s four terms, ``block_sum4``; the four sums
    to every rank's statistics entry (token, part) in a rotated rank order."""
    S, tid, W, rank = c["S"], c["tid"], c["W"], c["rank"]
    a = c["args"]
    A_PARTS = a_parts(S)
    NC = 2 * (A_PAIRS_MAX // A_PARTS)
    t, q = ob_task(S, rank, j)
    col = q * (HIDDEN // A_PARTS) + tid * NC
    # the operands first: issued after the poll, each would be a round trip
    al = ld_f32(a["alpha"], col, NC)
    be = ld_f32(a["beta"], col, NC)
    st = ld_f32(a["res"], t * HIDDEN + col, NC)
    rocdl.sched_barrier(0)
    ws = ob_poll(c, t, col, NC)
    stamp(c, 8)  # the part's partials arrived (thread 0's)
    _, v = ob_sums(c, NC, ws)
    ssp, A_, B_, C_ = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
    for k in range_constexpr(NC):
        ssp = ssp + v[k] * v[k]
        A_ = A_ + (al[k] * st[k]) * (al[k] * st[k])
        B_ = B_ + al[k] * be[k] * st[k] * v[k]
        C_ = C_ + (be[k] * v[k]) * (be[k] * v[k])
    ssp, A_, B_, C_ = block_sum4(c, ssp, A_, B_, C_)  # one barrier for the four
    if tid < 4:
        val = (tid == 0).select(ssp, (tid == 1).select(A_, (tid == 2).select(B_, C_)))
        for k in range_constexpr(W):
            dst = preg(pick_base(c, (rank + 1 + k) % W), c["sy_attn"], "attn")
            c["put"](dst, XST0 + (t * A_PARTS_MAX + q) * 4 + tid, val)
    stamp(c, 9)  # the statistics pushed


def b1_loads(c, part):
    """(``ob``) the operands of this CTA's K part's share of the x operands that needs no
    o sums (``b1_store``), this thread's column pair: in flight."""
    S, tid = c["S"], c["tid"]
    a = c["args"]
    col = part * K_PER_WAVE + 2 * tid
    prw = ld_bf(a["pre_w"], col, 2)
    al = ld_f32(a["alpha"], col, 2)
    be = ld_f32(a["beta"], col, 2)
    sts = [ld_f32(a["res"], t * HIDDEN + col, 2) for t in range(S)]
    return prw, al, be, sts


@flyc.jit
def b1_store(c, ops):
    """(``ob``) b1 = bf16(pre_w alpha st) of this CTA's K part (columns K_PER_WAVE part
    ..) for every token into LDS (``xs``, [token][K_PER_WAVE / 2] pair words: the og
    rows', free past the o tasks) and the columns' pre_w beta (fp32, ``red`` row set
    OB_PB): a thread a column pair (``b1_loads``). Then this CTA's flag (``ss`` [bid]):
    the stream update overwrites ``res`` only once every CTA has read it here."""
    S, tid = c["S"], c["tid"]
    prw, al, be, sts = ops
    for t in range_constexpr(S):
        st = sts[t]
        w1 = bf16_pair(prw[0] * al[0] * st[0], prw[1] * al[1] * st[1])
        fx.ptr_store(w1, c["xs"] + (t * (K_PER_WAVE // 2) + tid))
    fx.ptr_store(
        fx.Vector.from_elements([prw[0] * be[0], prw[1] * be[1]], fx.Float32),
        c["red"] + (OB_PB * RED_FLOATS + 2 * tid),
    )
    gpu.barrier()
    if tid == 0:
        c["put"](c["ss"], c["bid"], fx.Int32(1))  # (its loads are in: no release)


@flyc.jit
def x_rstd_ob(c, sw, mid=None):
    """(``ob``) ``wait_x``'s rstd / rstd2 into ``misc`` from the statistics words ``sw``
    this thread polled (``x_poll_ob``; threads 0 .. 2 S A_PARTS: two words each).
    ``mid``: issues loads once every thread's statistics are in (its result returned)."""
    S, tid, misc = c["S"], c["tid"], c["misc"]
    A_PARTS = a_parts(S)
    NP = S * A_PARTS
    if tid < 2 * NP:
        fx.ptr_store(sw[0].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid))
        fx.ptr_store(sw[1].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid + 1))
    gpu.barrier()
    acquire_loads()
    first = mid() if const_expr(mid is not None) else None
    if tid < S:
        tot, At, Bt, Ct = (
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
        )
        for q in range_constexpr(A_PARTS):
            b = misc + (SUMS0 + (tid * A_PARTS + q) * 4)
            tot = tot + fx.ptr_load(b)
            At = At + fx.ptr_load(b + 1)
            Bt = Bt + fx.ptr_load(b + 2)
            Ct = Ct + fx.ptr_load(b + 3)
        rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
        tot2 = At + fx.Float32(2.0) * rstd * Bt + rstd * rstd * Ct
        fx.ptr_store(rstd, misc + (SCAL0 + 2 * tid))
        fx.ptr_store(
            hw_rsq(tot2 / float(HIDDEN) + RMS_EPS), misc + (SCAL0 + 2 * tid + 1)
        )
    gpu.barrier()
    return first


def x_poll_ob(c, v_offs, gap=1, pre=None):
    """(``ob``) on threads 0 .. 2 S A_PARTS, ``wait_x``'s two statistics words (a tagged
    spec) first, then ``x_rstd_ob``, then this lane's all-gathered sum words (``v_offs``:
    4 sentinel words each), re-polling only what is still pending. Returns the sum words.

    Two rounds, not one spin over both: every CTA polling its K part's sum words from
    the start re-read ~8 MB a round from memory (32 CTAs a part), which delayed the
    words' arrival by ~2 us; the statistics are 128 words a CTA, and once they are in,
    the sums mostly are too (both follow the same partials). And a part's statistics
    are pushed only after this rank's partials of its columns arrived at the owner, each
    pushed after its CTA's re-arm of the columns' sum words (``ag_rearm``): the sum
    words read past the statistics are this launch's (re-armed or pushed)."""
    S, tid = c["S"], c["tid"]
    A_PARTS = a_parts(S)
    NP = S * A_PARTS
    assert 2 * NP <= THREADS and SUMS0 + 4 * NP <= ROWS * MAX_TOKENS
    tag = c["tag"]
    r = rsrc(c["own_attn"].value, MASK_OOB)
    OOB = fx.Int32(MASK_OOB // 4)
    live_s = tid < 2 * NP
    tq = tid // 2
    s_off = live_s.select(
        2 * (XST0 + ((tq // A_PARTS) * A_PARTS_MAX + tq % A_PARTS) * 4 + 2 * (tid % 2)), OOB
    )  # fmt: skip
    N = len(v_offs)
    NV = 4 * N

    def s_done(v):
        return (~live_s) | ((v[1] == tag) & (v[3] == tag))

    def load_s(prev):
        off = s_off
        if const_expr(prev is not None):
            ds = s_done(prev)
            off = ds.select(OOB, off)
        w = fx.Vector(
            bo.buffer_load(r, off, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
        )
        words = []
        for q in range_constexpr(4):
            words.append(
                fx.Int32(w[q]) if prev is None else ds.select(prev[q], fx.Int32(w[q]))
            )
        return fx.Vector.from_elements(words, fx.Int32)

    if const_expr(pre is None):
        sv = _spin_masked(load_s, lambda v: ~s_done(v), gap)
        pre_out = None
    else:
        # the statistics' first load out, then ``pre`` (the weights' transposes: they wait
        # for the weights alone, issued before it) while it is in flight
        s_first = load_s(None)
        pre_out = pre()
        sv = _spin_masked(load_s, lambda v: ~s_done(v), gap, first=s_first)

    def v_done(v, e):
        d = v[4 * e] != fx.Int32(SENT)
        for q in range_constexpr(1, 4):
            d = d & (v[4 * e + q] != fx.Int32(SENT))
        return d

    def load(prev):
        # every load issued before any select: the selects into the loop's registers
        # otherwise each waited for their load (a round trip a spec)
        ws, ds = [], []
        for e in range_constexpr(N):
            off = fx.Int32(v_offs[e])
            if const_expr(prev is not None):
                d = v_done(prev, e)
                off = d.select(OOB, off)
                ds.append(d)
            ws.append(
                fx.Vector(
                    bo.buffer_load(
                        r, off, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
                    )
                )
            )
        if const_expr(prev is not None):
            rocdl.sched_barrier(0)
        words = []
        for e in range_constexpr(N):
            for q in range_constexpr(4):
                w = fx.Int32(ws[e][q])
                words.append(w if prev is None else ds[e].select(prev[4 * e + q], w))
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[0] == fx.Int32(SENT)
        for e in range_constexpr(1, NV):
            bad = bad | (v[e] == fx.Int32(SENT))
        return bad

    # the sum words' first loads go out once every thread's statistics are in (past the
    # barrier), before the rstd math; then ``pre`` (the weights' transposes: landed, ahead
    # of the statistics) while they are in flight
    first = x_rstd_ob(c, [sv[0], sv[2]], lambda: load(None))
    v = _spin_masked(load, pending, gap, first=first)
    return [[v[4 * e + q] for q in range(4)] for e in range(N)], pre_out


def x_loads_ob(c, part, t0, pre=None):
    """``wait_x`` + ``x_loads`` for the owner-side statistics (``ob``): x = bf16(rstd2 (b1 +
    rstd b2)) with b1 from LDS (``b1_store``) and b2 = bf16(pre_w beta v) from the
    all-gathered o sums v (sentinel words, polled device-scope with the statistics,
    ``x_poll_ob``) and ``b1_store``'s products."""
    S, lane, wave, misc = c["S"], c["lane"], c["wave"], c["misc"]
    l16, g4 = lane % 16, lane // 16
    assert t0 == 0 and S <= TILE
    tl = fx.min(t0 + l16, S - 1)
    xs, pbr = c["xs"], c["red"] + OB_PB * RED_FLOATS
    V4F = fx.Vector.make_type(4, fx.Float32)
    cls = [(wave * P_STEPS + kc) * K_STEP + 8 * g4 for kc in range(P_STEPS)]
    v_offs = [ag_word(tl, part * K_PER_WAVE + cl) for cl in cls]
    got4, pre_out = x_poll_ob(c, v_offs, pre=pre)
    rstd = fx.ptr_load(misc + (SCAL0 + 2 * tl))
    rstd2 = fx.ptr_load(misc + (SCAL0 + 2 * tl + 1))
    got = [
        [got4[e // 2][2 * (e % 2) + q] for q in range(2)] for e in range(2 * P_STEPS)
    ]
    out = []
    for kc in range_constexpr(P_STEPS):
        cl = cls[kc]
        b1w = fx.Vector(
            fx.ptr_load(xs + (tl * (K_PER_WAVE // 2) + cl // 2), result_type=V4F)
        ).bitcast(fx.Int32)
        pa = fx.Vector(fx.ptr_load(pbr + cl, result_type=V4F))
        pc = fx.Vector(fx.ptr_load(pbr + (cl + 4), result_type=V4F))
        pb = [fx.Float32(pa[k]) for k in range(4)] + [
            fx.Float32(pc[k]) for k in range(4)
        ]
        words = []
        for k in range_constexpr(4):
            lo, hi = bf2_f32(got[2 * kc + k // 2][k % 2])
            h1a, h1b = bf2_f32(fx.Int32(b1w[k]))
            h2a, h2b = bf2_f32(bf16_pair(pb[2 * k] * lo, pb[2 * k + 1] * hi).bitcast(fx.Int32))  # fmt: skip
            words.append(bf16_pair(rstd2 * (h1a + rstd * h2a), rstd2 * (h1b + rstd * h2b)).bitcast(fx.Int32))  # fmt: skip
        out.append(fx.Vector.from_elements(words, fx.Int32))
    if const_expr(pre is not None):
        return out, pre_out
    return out


# ---------------------------------------------------------------- boundary tasks
def o_sum(c, t, e0, PAIRS):
    """Token ``t``'s all-reduced o columns ``e0`` .. + 2 PAIRS (f32 of bf16): the
    ranks' partial pairs polled in one round and summed in rank order, one bf16
    rounding; or (``o_rs``) the owners' sums, polled from the all-gather rows."""
    W, poll, own = c["W"], c["poll"], c["own_attn"]
    PW = 2 if PAIRS % 2 == 0 else 1  # pairs a spec
    v = []
    if const_expr(o_rs(c)):
        pb = (AG_T0 + t) * (HIDDEN // 2) + e0 // 2
        ws = poll(
            [(own, pb + PW * q, PW) for q in range(PAIRS // PW)],
            batch=max(8, PAIRS // PW),
        )
        for j in range_constexpr(PAIRS):
            lo, hi = bf2_f32(ws[j // PW][j % PW])
            v += [lo, hi]
        return v
    pb = t * (HIDDEN // 2) + e0 // 2
    ws = poll(
        [
            (own, src * (MAX_TOKENS * HIDDEN // 2) + pb + PW * q, PW)
            for q in range(PAIRS // PW)
            for src in range(W)
        ],
        batch=max(8, (PAIRS // PW) * W),
    )
    return rs_partial_sums(c, ws, PAIRS, PW)


def keep_sums(s: int) -> bool:
    """The boundary part's operands fit ``xs``: a thread's o sums, alpha, beta and stream
    columns (2 A_PAIRS_MAX / a_parts words each)."""
    return 4 * THREADS * (2 * A_PAIRS_MAX // a_parts(s)) <= k2_xs_floats(s)


@flyc.jit
def stage_a_part(c, t, part):
    """Part ``part`` of token ``t``'s attention boundary (columns HIDDEN / A_PARTS
    part .., 2 PAIRS a thread): the ranks' partial o rows summed in rank order
    (``sum_partials``), one bf16 rounding; then, before any exchange, the pooler /
    router operands x = rstd2 (b1 + rstd b2) as its two halves b1 = bf16(pre_w alpha
    st), b2 = bf16(pre_w beta v) (plain bf16 words) and the part's four norm sums
    (``ss``, after the halves: the consumers' flag), so the GEMVs start without
    the sums' exchange, each consumer folding the scalars in itself. The stream's
    update is ``stage_a_part_res``, after the CTA's pooler job."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, xbuf, xbuf2 = c["own_attn"], c["ss"], c["xbuf"], c["xbuf2"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS  # columns a thread
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    # the operands first: after the poll each would be a dependent round trip
    prw = ld_bf(a["pre_w"], e0, NC)
    al = ld_f32(a["alpha"], e0, NC)
    be = ld_f32(a["beta"], e0, NC)
    st = ld_f32(a["res"], t * HIDDEN + e0, NC)
    rocdl.sched_barrier(0)
    # the all-reduce: this thread's PAIRS pairs from every rank in one poll round (two
    # adjacent pairs a spec), each pair summed in rank order, one bf16 rounding
    v = o_sum(c, t, e0, PAIRS)
    stamp(c, 8)  # every rank's partial o rows arrived (thread 0's columns)
    # the sums and operands kept for ``stage_a_part_res`` in this thread's words of
    # ``xs`` (the og rows', free past the o tasks and untouched by the pooler job): it
    # re-read them (the sums from the peer buffer, which a peer's next launch may then
    # be overwriting), a round trip behind the CTA's pushes
    if const_expr(keep_sums(S)):
        for j, vals in enumerate((v, al, be, st)):
            for k in range_constexpr(NC):
                fx.ptr_store(vals[k], c["xs"] + ((j * THREADS + tid) * NC + k))
    # the operands' two halves: x = bf16(new rstd2 pre_w) with new = alpha st + beta
    # bf16(v rstd) is rstd2 (b1 + rstd b2) up to the roundings
    w1 = [bf16_pair(prw[2 * k] * al[2 * k] * st[2 * k], prw[2 * k + 1] * al[2 * k + 1] * st[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
    w2 = [bf16_pair(prw[2 * k] * be[2 * k] * v[2 * k], prw[2 * k + 1] * be[2 * k + 1] * v[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
    st_words(xbuf, (t * HIDDEN + e0) // 2, w1, CM_DEV)
    st_words(xbuf2, (t * HIDDEN + e0) // 2, w2, CM_DEV)
    # the two norms' statistics: the row's sum of squares of v and, for the pre-FFN
    # norm of new = alpha st + beta v rstd, the sums A = (alpha st)^2, B = alpha beta
    # st v, C = (beta v)^2 (its sum of squares is A + 2 rstd B + rstd^2 C up to the
    # bf16 rounding of v rstd: a 1e-5 relative difference)
    ssp, A_, B_, C_ = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
    for k in range_constexpr(NC):
        ssp = ssp + v[k] * v[k]
        A_ = A_ + (al[k] * st[k]) * (al[k] * st[k])
        B_ = B_ + al[k] * be[k] * st[k] * v[k]
        C_ = C_ + (be[k] * v[k]) * (be[k] * v[k])
    ssp, A_, B_, C_ = block_sum4(c, ssp, A_, B_, C_)  # one barrier for the four
    release_stores()  # this wave's halves out ...
    gpu.barrier()  # ... every wave's, before the sums that flag them
    if tid < 4:
        put(
            ss_mb,
            (t * A_PARTS_MAX + part) * 4 + tid,
            (tid == 0).select(ssp, (tid == 1).select(A_, (tid == 2).select(B_, C_))),
        )


def a_part_rstd(c, t, v):
    """Token ``t``'s post-norm rstd from the parts' sums of squares (``ss``), summed in
    part order (one part: this CTA's own, through ``block_sum4``)."""
    A_PARTS = a_parts(c["S"])
    tot = fx.Float32(0.0)
    if A_PARTS > 1:
        got = c["poll"](
            [(c["ss"], (t * A_PARTS_MAX + q) * 4, 2) for q in range(A_PARTS)],
            batch=A_PARTS,
        )
        stamp(c, 9)  # the parts' norm sums arrived
        for q in range(A_PARTS):
            tot = tot + got[q][0].bitcast(fx.Float32)
    else:
        for k in range(len(v)):
            tot = tot + v[k] * v[k]
        tot = block_sum4(c, tot, tot, tot, tot)[0]
    return hw_rsq(tot / float(HIDDEN) + POST_EPS)


@flyc.jit
def stage_a_part_res(c, t, part):
    """The rest of boundary part ``part`` of token ``t``, after the CTA's pooler job
    (nothing reads it before K3's boundary): the ranks' partial o rows again (still in
    the mailbox, this layer's tag), the parts' sums of squares (``ss``), the post norm
    without weight, the stream updated in place (fp32), the next K1's first half
    (deferred K1), the (token, part) flag (``resflag``)."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, resflag = c["own_attn"], c["ss"], c["resflag"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    if const_expr(c["defer_k1"]):
        npw1 = ld_bf(a["npw"], e0, NC)
        al2 = ld_f32(a["alpha2"], e0, NC)
    if const_expr(ob(c)):
        # (``ob``: no ``stage_a_part`` ran) the all-gathered sums again and the token's
        # rstd ``x_rstd_ob`` formed (``misc``); ``res`` is overwritten below only once
        # every CTA's ``b1_loads`` have read it (a flag a CTA, one round)
        al = ld_f32(a["alpha"], e0, NC)
        be = ld_f32(a["beta"], e0, NC)
        st = ld_f32(a["res"], t * HIDDEN + e0, NC)
        ws = poll_sent_vec(c["own_attn"].value, [ag_word(t, e0)], PAIRS, CM_SYS)[0]
        v = []
        for k in range_constexpr(PAIRS):
            lo, hi = bf2_f32(ws[k])
            v += [lo, hi]
        rstd = fx.ptr_load(c["misc"] + (SCAL0 + 2 * t))
        if tid < BLOCKS:
            poll([(c["ss"], tid, 1)])
        gpu.barrier()
    elif const_expr(keep_sums(S)):
        # the o sums and operands ``stage_a_part`` kept, and the token's post-norm rstd
        # that ``wait_x`` formed from the same parts' sums in the same order (``misc``)
        v, al, be, st = [
            [fx.ptr_load(c["xs"] + ((j * THREADS + tid) * NC + k)) for k in range(NC)]
            for j in range(4)
        ]
        rstd = fx.ptr_load(c["misc"] + (SCAL0 + 2 * t))
    else:
        al = ld_f32(a["alpha"], e0, NC)
        be = ld_f32(a["beta"], e0, NC)
        st = ld_f32(a["res"], t * HIDDEN + e0, NC)
        v = o_sum(c, t, e0, PAIRS)
        rstd = a_part_rstd(c, t, v)
    n = [bf16_round(v[k] * rstd) for k in range(NC)]
    new = [al[k] * st[k] + be[k] * n[k] for k in range(NC)]
    st_f32(a["res"], t * HIDDEN + e0, new, CM_DEV)
    if const_expr(c["defer_k1"]):
        c1w = [bf16_pair(npw1[2 * k] * al2[2 * k] * new[2 * k], npw1[2 * k + 1] * al2[2 * k + 1] * new[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
        st_words(c["c1"], (t * HIDDEN + e0) // 2, c1w, CM_DEV)
    release_stores()
    gpu.barrier()
    if tid == 0:
        put(
            resflag, t * A_PARTS + part, fx.Int32(1)
        )  # the stream's part (and c1) is out


# ---------------------------------------------------------------- router and pooler GEMVs
# (``wcontig``, with ``ob``) the pooler / router weights streamed in whole row segments: a
# wave's share of a row tile is 16 rows x 256 B (its K slice), and the MFMA A layout
# (lane 16 g4 + l16: row l16, step kc's 16 B at 64 kc + 16 g4) made each dwordx4 load
# touch 16 rows x 64 B (16 DRAM pages a quarter-wave): the 16 MB pool / router burst ran
# at ~2.7 TB/s and the router CTAs' x polls queued behind it. Load j now reads rows 4 j ..
# 4 j + 3 whole (lane L: row 4 j + L / 16, 16 B chunk L % 16), and the wave moves the four
# loads through its own 4 KB of LDS (``red`` row sets WT_RED0 .., the 16 B chunks
# XOR-swizzled by row: conflict-free) into the MFMA layout: the same A registers, bit
# for bit, once the weights landed (while the x words are in flight).
WT_RED0 = 4  # the transposes' ``red`` row sets (4 KB a wave)
assert WAVES * 1024 <= 4 * RED_FLOATS


def wcontig(c) -> bool:
    """The pooler / router weights in whole row segments (``ob``, the BF16 pooler)."""
    return ob(c) and not c["fp8_pool"]


def w_contig_loads(r_w, row0, k0, row_len, lane):
    """A wave's 16 rows x 128 bf16 (rows ``row0`` .., elements ``k0`` ..) as 4 dwordx4 loads
    of whole 256 B row segments (load j, lane L: row row0 + 4 j + L / 16, elements k0 + 8
    (L % 16) ..), nontemporal; ``w_transpose`` puts them in the MFMA A layout."""
    return [
        fx.Vector(
            bo.buffer_load(
                r_w,
                ((row0 + 4 * j + lane // 16) * row_len + k0 + 8 * (lane % 16)) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for j in range(4)
    ]


def w_transpose(c, ws):
    """``w_contig_loads``' 4 loads in the MFMA A layout (step kc: lane 16 g4 + l16 holds row
    l16's 16 B chunk 4 kc + g4) through this wave's 4 KB of ``red`` (16 B chunk ch of row r
    at r 16 + (ch ^ r)): 4 stores, 4 loads, no barrier (the wave's own region, in order)."""
    lane, wave = c["lane"], c["wave"]
    base = c["red"] + (WT_RED0 * RED_FLOATS + wave * 1024)
    V4F = fx.Vector.make_type(4, fx.Float32)
    for j in range_constexpr(4):
        r = 4 * j + lane // 16
        fx.ptr_store(
            fx.Vector(ws[j]).bitcast(fx.Float32),
            base + (r * 16 + ((lane % 16) ^ r)) * 4,
        )
    l16, g4 = lane % 16, lane // 16
    out = []
    for kc in range_constexpr(4):
        v = fx.ptr_load(base + (l16 * 16 + ((4 * kc + g4) ^ l16)) * 4, result_type=V4F)
        out.append(fx.Vector(v).bitcast(fx.Int32))
    return out


def w_transpose_all(c, pool_w, router_w):
    """``w_transpose`` of every row tile's loads (``wcontig``): (pool_w, router_w)."""
    pw = [w_transpose(c, ws) for ws in pool_w]
    rw = None
    if router_w is not None:
        rw = (w_transpose(c, router_w[0]), w_transpose(c, router_w[1]))
    return pw, rw


def pool_w_loads(c, rt0, part, nsub):
    """The pooler weights of the job (row tiles ``rt0`` .. + ``nsub``, K eighth
    ``part``) over this wave's P_STEPS k-steps, this lane's, in flight (16 VGPRs a
    row tile bf16, 8 FP8). Row-major, 8 values a lane a step (16 B bf16, 8 B of E4M3
    codes with ``c["fp8_pool"]``), read once a layer (nontemporal)."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    r_w = rsrc(c["args"]["w_pool"])
    fp8 = c["fp8_pool"]
    if wcontig(c):
        return [
            w_contig_loads(
                r_w, c.get("pool_row0", 0) + (rt0 + sub) * ROWS,
                part * K_PER_WAVE + wave * (P_STEPS * K_STEP), HIDDEN, lane,
            )
            for sub in range(nsub)
        ]  # fmt: skip
    out = []
    for sub in range(nsub):
        row = c.get("pool_row0", 0) + (rt0 + sub) * ROWS + l16
        if fp8:
            # the swizzled copy (``pack_fp8_rows_pooler``): a step pair's 16 codes of this
            # lane group sit together, one 16 B load a pair
            assert P_STEPS % 2 == 0
            steps = []
            for pr in range(P_STEPS // 2):
                v = fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (
                            row * HIDDEN
                            + part * K_PER_WAVE
                            + wave * (P_STEPS * K_STEP)
                            + pr * (2 * K_STEP)
                            + 16 * g4
                        )
                        // 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )  # fmt: skip
                )
                steps.append(fx.Vector.from_elements([v[0], v[1]], fx.Int32))
                steps.append(fx.Vector.from_elements([v[2], v[3]], fx.Int32))
            out.append(steps)
            continue
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (
                            row * HIDDEN
                            + part * K_PER_WAVE
                            + (wave * P_STEPS + kc) * K_STEP
                            + 8 * g4
                        )
                        // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )  # fmt: skip
                )
                for kc in range(P_STEPS)
            ]
        )
    return out


def pool_a(c, w):
    """A pooler weight step as the MFMA's bf16 A operand (FP8 codes decoded)."""
    if c["fp8_pool"]:
        return fp8_to_bf16x8(w[0], w[1])
    return w.bitcast(fx.BFloat16)


def pool_scale(c, row):
    """The pooler row's FP8 scale (fp32, after the codes), or 1."""
    if not c["fp8_pool"]:
        return fx.Float32(1.0)
    return fx.Float32(
        bo.buffer_load(
            rsrc(c["args"]["w_pool"]),
            (MOE_HIDDEN * HIDDEN) // 4 + row,
            vec_width=1,
            dtype=T.f32,
        )
    )


def router_w_loads(c, tile, part, live=None):
    """The split router's hi and lo rows of expert tile ``tile`` over this wave's
    steps of K eighth ``part``, this lane's: ([hi steps], [lo steps]). ``live`` off:
    zeros, no traffic (a stream CTA without a router task)."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    row = tile * ROWS + l16
    a = c["args"]
    nb = None if live is None else live.select(fx.Int32(0x7FFFFFFF), fx.Int32(0))

    def loads(w):
        r_w = rsrc(w, nb)
        if wcontig(c):
            return w_contig_loads(
                r_w,
                tile * ROWS,
                part * K_PER_WAVE + wave * (P_STEPS * K_STEP),
                HIDDEN,
                lane,
            )
        return [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (
                        row * HIDDEN
                        + part * K_PER_WAVE
                        + (wave * P_STEPS + kc) * K_STEP
                        + 8 * g4
                    )
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )  # fmt: skip
            )
            for kc in range(P_STEPS)
        ]

    return loads(a["w_hi"]), loads(a["w_lo"])


def x_loads(c, part, t0):
    """Token tile t0 .. t0 + 16's x over K eighth ``part``, this wave's steps: lane
    l16's token (clamped), 8 values a step, x = bf16(rstd2 (b1 + rstd b2)) from the
    boundary's two halves (device-scope: other CTAs wrote them) and the token's
    scalars (``wait_x``)."""
    S, lane, wave, misc = c["S"], c["lane"], c["wave"], c["misc"]
    l16, g4 = lane % 16, lane // 16
    tl = fx.min(t0 + l16, S - 1)
    r1, r2 = rsrc(c["xbuf"]), rsrc(c["xbuf2"])
    rstd = fx.ptr_load(misc + (SCAL0 + 2 * tl))
    rstd2 = fx.ptr_load(misc + (SCAL0 + 2 * tl + 1))
    out = []
    for kc in range_constexpr(P_STEPS):
        at = (
            tl * HIDDEN + part * K_PER_WAVE + (wave * P_STEPS + kc) * K_STEP + 8 * g4
        ) // 2
        h1 = (
            fx.Vector(
                bo.buffer_load(r1, at, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
            )
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        h2 = (
            fx.Vector(
                bo.buffer_load(r2, at, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
            )
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        words = [bf16_pair(rstd2 * (fx.Float32(h1[2 * j]) + rstd * fx.Float32(h2[2 * j])), rstd2 * (fx.Float32(h1[2 * j + 1]) + rstd * fx.Float32(h2[2 * j + 1]))).bitcast(fx.Int32) for j in range(4)]  # fmt: skip
        out.append(fx.Vector.from_elements(words, fx.Int32))
    return out


@flyc.jit
def stage_passes(c, rt0, part, pool_w, router_w, r_tile, is_router, nsub):
    """The pooler job (row tiles ``rt0`` .., ``part``) and, with ``router_w`` (a
    stream CTA), router task (``r_tile``, ``part``) over every token tile: x loaded
    straight into the B operands (16 B a lane a step, a token tile of 16), the
    waves' accumulators of each row tile summed through ``red`` (``red_sets`` row
    tiles a round) and published: pooled rows as bf16 pairs to ``pool``
    [part][token][row], logits as fp32 to ``logits`` [part][token][expert]."""
    S, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    put, put_bf, pool, logits = c["put"], c["put_bf"], c["pool"], c["logits"]
    RS = red_sets(S)
    NT_ = nsub + (
        1 if router_w is not None else 0
    )  # row tiles a pass: the job's and the router's
    for ps in range_constexpr(token_tiles(S)):
        t0 = ps * TILE
        nt = min(TILE, S - t0)
        if const_expr(wcontig(c)):
            # the weights' transposes while the x words are in flight (one token tile)
            assert token_tiles(S) == 1
            xb, (pool_w, router_w) = x_loads_ob(
                c, part, t0, pre=lambda: w_transpose_all(c, pool_w, router_w)
            )
        else:
            xb = x_loads_ob(c, part, t0) if ob(c) else x_loads(c, part, t0)
        stamp(c, 1)  # x arrived (thread 0's)
        tiles = []
        for sub in range_constexpr(nsub):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(P_STEPS):
                acc = mfma_bf16(
                    pool_a(c, pool_w[sub][kc]), xb[kc].bitcast(fx.BFloat16), acc
                )
            tiles.append(acc)
        if const_expr(router_w is not None):
            racc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(P_STEPS):
                racc = mfma_bf16(
                    router_w[0][kc].bitcast(fx.BFloat16),
                    xb[kc].bitcast(fx.BFloat16),
                    racc,
                )
                racc = mfma_bf16(
                    router_w[1][kc].bitcast(fx.BFloat16),
                    xb[kc].bitcast(fx.BFloat16),
                    racc,
                )
            tiles.append(racc)
        # the row sums over the waves (wave order), RS row tiles a round
        for r0 in range_constexpr(0, NT_, RS):
            n = min(RS, NT_ - r0)
            for j in range_constexpr(n):
                acc_store(red + j * RED_FLOATS, wave, lane, tiles[r0 + j])
            gpu.barrier()
            # (sharded) the router tile's logits first: the stores and the loads after
            # them complete in issue order, so the pooled rows' stores go after
            order = list(range(n))
            if const_expr(c["pshard"]):
                order = [j for j in range(n) if r0 + j >= nsub] + [
                    j for j in range(n) if r0 + j < nsub
                ]
            for j in order:
                ti = r0 + j
                rb = red + j * RED_FLOATS
                if const_expr(ti < nsub):
                    # pooled rows: thread (token tk, row pair rp) of the tile, bf16 pairs
                    if tid < (ROWS // 2) * TILE:
                        tk = tid // (ROWS // 2)
                        rp = (tid % (ROWS // 2)) * 2
                        if tk < nt:
                            prow = (rt0 + ti) * ROWS + rp
                            if const_expr(c["pshard"]):
                                # the rank's rows: the K part's partial to this GPU's
                                # ``pool`` (rows of the slice); ``stage_pool_reduce``
                                # sums the parts and pushes the sums to every rank
                                put_bf(
                                    pool,
                                    (part * MAX_TOKENS + t0 + tk) * MOE_HIDDEN + prow,
                                    [
                                        acc_row_sum(rb, rp, tk),
                                        acc_row_sum(rb, rp + 1, tk),
                                    ],
                                )
                            else:
                                put_bf(
                                    pool,
                                    (part * MAX_TOKENS + t0 + tk) * MOE_HIDDEN + prow,
                                    [
                                        acc_row_sum(rb, rp, tk) * pool_scale(c, prow),
                                        acc_row_sum(rb, rp + 1, tk)
                                        * pool_scale(c, prow + 1),
                                    ],
                                )
                else:
                    # the router tile's partial logits: thread (token tk, expert row r)
                    if is_router & (tid < ROWS * TILE):
                        tk = tid // ROWS
                        r = tid % ROWS
                        if tk < nt:
                            put(
                                logits,
                                (part * MAX_TOKENS + t0 + tk) * EXPERTS
                                + r_tile * ROWS
                                + r,
                                acc_row_sum(rb, r, tk),
                            )
            gpu.barrier()  # ``red`` is reused by the next round


PR_WAVE = WAVES - 2  # the pool reduce's wave (the route takes the last)


@flyc.jit
def stage_pool_reduce(c, rt, part):
    """The sharded pooler's K parts summed on this GPU (a row tile's eight part CTAs
    share its tokens: tokens ``part + 8 k``): a thread a (token, row pair) polls the
    parts' bf16 partials (``pool``), sums them in part order in fp32 from 0 -- the
    sum ``stage_xn`` formed from the eight partials, bit for bit -- and pushes the two
    fp32 sums to every rank's ``poolx`` [token][row] (a pair a row: a quarter of the
    eight parts' bytes on the links, which bound the push)."""
    S, tid, poll, pool = c["S"], c["tid"], c["poll"], c["pool"]
    put_words = c["put_words"]
    NK = -(-S // P_PARTS)  # tokens a share
    # on wave PR_WAVE: the passes' stores went out on waves 0 .. 3 (a poll returns in
    # order behind its own wave's stores)
    i = tid - PR_WAVE * 64
    assert NK * (ROWS // 2) <= 64
    if (i >= 0) & (i < NK * (ROWS // 2)):
        k = i // (ROWS // 2)
        rp = (i % (ROWS // 2)) * 2
        t = part + P_PARTS * k
        if t < S:
            prow = rt * ROWS + rp
            got = poll(
                [
                    (pool, ((p * MAX_TOKENS + t) * MOE_HIDDEN + prow) // 2, 1)
                    for p in range(P_PARTS)
                ],
                batch=P_PARTS,
            )
            v0, v1 = fx.Float32(0.0), fx.Float32(0.0)
            for p in range_constexpr(P_PARTS):
                lo, hi = bf2_f32(got[p][0])
                v0 = v0 + lo
                v1 = v1 + hi
            grow = c["rank"] * (MOE_HIDDEN // c["W"]) + prow
            words = [v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)]
            for dst in rot_poolx(c):
                put_words(dst, t * MOE_HIDDEN + grow, words)


def rot_poolx(c):
    """Every rank's ``poolx`` in a rotated order (rank + 1, .., rank): at each step
    the ranks push to distinct peers, not all to the same one. Picked from the bases
    the CTA holds (selects, no load on the push's path)."""
    W, rank, bases = c["W"], c["rank"], c["bases"]
    out = []
    for k in range(W):
        r = (rank + 1 + k) % W
        b = bases[W - 1]
        for p in range(W - 2, -1, -1):
            b = (r == p).select(bases[p], b)
        out.append(preg(b, c["sy_poolx"], "poolx"))
    return out


SCAL0 = 0  # misc floats: a token's (rstd, rstd2) at 2 t
SUMS0 = 2 * MAX_TOKENS  # then the (token, part) norm sums, 4 each


@flyc.jit
def wait_x(c):
    """Every (token, part)'s four norm sums (stored after its x halves: their flag;
    one round, two words a thread), then each token's rstd (the post norm) and rstd2
    (the pre-FFN norm) into LDS (``misc``) for ``x_loads``, then device-scope loads."""
    S, tid, ss_mb, misc, poll = c["S"], c["tid"], c["ss"], c["misc"], c["poll"]
    A_PARTS = a_parts(S)
    NP = S * A_PARTS
    assert 2 * NP <= THREADS and SUMS0 + 4 * NP <= ROWS * MAX_TOKENS
    rocdl.sched_barrier(0)
    if tid < 2 * NP:
        tq = tid // 2
        got = poll(
            [
                (
                    ss_mb,
                    ((tq // A_PARTS) * A_PARTS_MAX + tq % A_PARTS) * 4 + 2 * (tid % 2),
                    2,
                )
            ]
        )
        fx.ptr_store(got[0][0].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid))
        fx.ptr_store(got[0][1].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid + 1))
    gpu.barrier()
    acquire_loads()
    if tid < S:
        tot, At, Bt, Ct = (
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
        )
        for q in range_constexpr(A_PARTS):
            b = misc + (SUMS0 + (tid * A_PARTS + q) * 4)
            tot = tot + fx.ptr_load(b)
            At = At + fx.ptr_load(b + 1)
            Bt = Bt + fx.ptr_load(b + 2)
            Ct = Ct + fx.ptr_load(b + 3)
        rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
        tot2 = At + fx.Float32(2.0) * rstd * Bt + rstd * rstd * Ct
        fx.ptr_store(rstd, misc + (SCAL0 + 2 * tid))
        fx.ptr_store(
            hw_rsq(tot2 / float(HIDDEN) + RMS_EPS), misc + (SCAL0 + 2 * tid + 1)
        )
    gpu.barrier()


# ---------------------------------------------------------------- route and xn tasks
@flyc.jit
def stage_route(c, t, route_flag=None):
    """route task: token ``t``'s top-8 on wave 0 (lane l holds experts l, l + 64,
    ..): the parts' partial logits summed in part order, sigmoid(logit x
    ROUTER_MULT), + bias, a pick an integer max over ``route_key`` (a tie to the
    lower expert), the picks' sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE."""
    tid, lane, wave, sig = c["tid"], c["lane"], c["wave"], c["sig"]
    put, poll, logits = c["put"], c["poll"], c["logits"]
    a = c["args"]
    pre = c["pshard"]  # (the replicated pooler's route keeps its order)
    # (sharded) the route on the last wave: wave 0 pushed the CTA's pool-reduce share to
    # every rank just before, and a poll returns in order behind its own wave's stores
    # (the remote acks); the flag goes out from this wave alone, after its own stores
    # drained (no barrier: the other waves' pushes are nothing the flag publishes)
    RW = WAVES - 1 if pre else 0
    if wave == RW:
        if const_expr(pre):
            # the bias first: issued after the poll, its round trip followed the logits'
            bias = [
                fx.Float32(
                    bo.buffer_load(
                        rsrc(a["bias"]), lane + 64 * h, vec_width=1, dtype=T.f32
                    )
                )
                for h in range(4)
            ]
        got = poll(
            [
                (logits, (p * MAX_TOKENS + t) * EXPERTS + lane + 64 * h, 1)
                for p in range(R_PARTS)
                for h in range(4)
            ],
            batch=4 * R_PARTS,  # one round trip for the parts' logits
        )
        stamp(c, 10, RW * 64)  # the parts' logits arrived (route CTAs)
        keys = []
        for h in range_constexpr(4):
            e = lane + 64 * h
            logit = got[h][0].bitcast(fx.Float32)
            for p in range_constexpr(1, R_PARTS):
                logit = logit + got[p * 4 + h][0].bitcast(fx.Float32)
            sc = sigmoid(logit * fx.Float32(ROUTER_MULT))
            fx.ptr_store(sc, sig + e)
            if const_expr(pre):
                b = bias[h]
            else:
                b = fx.Float32(
                    bo.buffer_load(rsrc(a["bias"]), e, vec_width=1, dtype=T.f32)
                )
            keys.append(route_key(sc + b, e))
        # (#142) the picks: the wave's top 8 keys, sorted (``top8_sorted``); the same
        # picks in the same order as 8 rounds of wave maxima
        top = top8_sorted(keys)
        pid = fx.Int32(0)
        packed = [fx.Int32(0), fx.Int32(0)]  # the picks as bytes (``route_flag``)
        for p in range_constexpr(TOP_K):
            pk = 255 - (top[p] & 255)
            pid = (lane == p).select(pk, pid)
            packed[p // 4] = packed[p // 4] | (pk << (8 * (p % 4)))
        if lane < TOP_K:
            w = fx.ptr_load(sig + pid)
            tot = butterfly(w, (1, 2, 4))
            bo.buffer_store(
                pid, rsrc(a["ids"]), t * TOP_K + lane, cache_modifier=CM_DEV
            )
            bo.buffer_store(
                (w / (tot + fx.Float32(ROUTE_EPS)) * fx.Float32(ROUTE_SCALE)).bitcast(
                    fx.Int32
                ),
                rsrc(a["scores"]),
                t * TOP_K + lane,
                cache_modifier=CM_DEV,
            )
        if const_expr(route_flag is not None):
            # the token's routing for the experts' plan: its ids as tagged words (data
            # and flag in one; K3 polls them), after this wave's ids / scores stores
            # drained, so a CTA that saw the words reads those too
            release_stores()
            if lane == 0:
                c["put_words"](route_flag, 2 * t, packed)


@flyc.jit
def stage_xn(c, t, xin_flag=None):
    """xn task: token ``t``'s pre-expert norm of its pooled row (8 elements a
    thread, the K parts' partial rows summed), weight ``xn_w``, one bf16 rounding
    -> the expert input."""
    tid, poll, put, pool = c["tid"], c["poll"], c["put"], c["pool"]
    if const_expr(c["pshard"]):
        pool = c["own_poolx"]  # every rank's row slice, pushed here
    a = c["args"]
    e0 = tid * 8
    if const_expr(c["pshard"]):
        xw = ld_bf(
            a["xn_w"], e0, 8
        )  # before the poll: its round trip overlaps the wait
        # the ranks' summed rows (``stage_pool_reduce``): a pair a row, fp32
        got = poll([(pool, t * MOE_HIDDEN + e0 + 2 * j, 2) for j in range(4)], batch=4)
        stamp(c, 11)  # the pooled rows arrived (xn CTAs)
        v = [got[j // 2][j % 2].bitcast(fx.Float32) for j in range(8)]
    else:
        # the P_PARTS K-eighths' pooled rows (bf16 pairs), summed in part order, one round
        got = poll(
            [
                (pool, ((p * MAX_TOKENS + t) * MOE_HIDDEN + e0) // 2 + 2 * j, 2)
                for p in range(P_PARTS)
                for j in range(2)
            ],
            batch=2 * P_PARTS,
        )
        stamp(c, 11)  # the parts' pooled rows arrived (xn CTAs)
        v = [fx.Float32(0.0) for _ in range(8)]
        for p in range_constexpr(P_PARTS):
            for j in range_constexpr(2):
                for q in range_constexpr(2):
                    lo, hi = bf2_f32(got[p * 2 + j][q])
                    v[4 * j + 2 * q] = v[4 * j + 2 * q] + lo
                    v[4 * j + 2 * q + 1] = v[4 * j + 2 * q + 1] + hi
    ss = fx.Float32(0.0)
    for j in range_constexpr(8):
        ss = ss + v[j] * v[j]
    rstd = hw_rsq(block_sum(c, ss, 0) / float(MOE_HIDDEN) + RMS_EPS)
    if const_expr(not c["pshard"]):
        xw = ld_bf(a["xn_w"], e0, 8)
    out = [v[j] * rstd * xw[j] for j in range(8)]
    st_bf(a["xin"], t * MOE_HIDDEN + e0, out, CM_DEV)
    if const_expr(c["xq_pre"]):
        # the same blocks K3 would quantize from the bf16 row (``ffn.stage_xq_cols``): a
        # thread's eight values one code word, four threads a 32-value block
        from sgl_amd_mono.muse_spark.gfx950.stages.mxfp4 import mxfp4_scale

        vals = [bf16_round(x) for x in out]
        amax = abs(vals[0])
        for k in range_constexpr(1, 8):
            amax = fx.max(amax, abs(vals[k]))
        amax = butterfly(amax, (1, 2), fx.max)
        byte, qs = mxfp4_scale(amax)
        # ``e2m1_code`` of the same products bit for bit by the gfx950 conversion
        # (scale 1: half to even, saturating at 6, the sign kept even at 0), two
        # values a byte: 4 instructions for the thread's 8 codes
        word = fx.Int32(0)
        for j in range_constexpr(4):
            word = fx.Int32(
                rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    as_ir_value(word),
                    as_ir_value(vals[2 * j] * qs),
                    as_ir_value(vals[2 * j + 1] * qs),
                    as_ir_value(fx.Float32(1.0)),
                    j,
                )
            )
        XQW_, XSW_ = MOE_HIDDEN // 8, MOE_HIDDEN // 32
        base = t * (XQW_ + XSW_)
        bo.buffer_store(word, rsrc(c["xinq"]), base + tid, cache_modifier=CM_DEV)
        if tid % 4 == 0:
            bo.buffer_store(
                byte, rsrc(c["xinq"]), base + XQW_ + tid // 4, cache_modifier=CM_DEV
            )
    if const_expr(xin_flag is not None):
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(xin_flag, t, fx.Int32(1))


# ---------------------------------------------------------------- the CTAs' roles
@flyc.jit
def run_chain(c, route_flag=None, xin_flag=None, o_flag=None, QL=8):
    """A chain CTA (``bid >= CHAIN0``): its two o tasks (``bid`` and ``bid +
    BLOCKS``), its boundary part (the 64 boundary tasks sit here), then its pooler
    weights, the tokens' flags, the pooler passes, and token ``bid - CHAIN0``'s
    route and xn."""
    S, bid = c["S"], c["bid"]
    a = c["args"]
    A_PARTS = a_parts(S)
    tok = bid - CHAIN0
    # the o tasks and the boundary first, the pooler weights after: issued before
    # them, the weights (~15 us to land) delay the o tasks, and in the TP 8 server
    # every rank's boundary then waits for the slowest rank's o partials (measured
    # +1.1 ms a step)
    run_o(c, o_flag, QL)  # the og rows and the o tasks
    stamp(c, 2)  # the o tasks done
    obd = ob(c)
    # the small pooler job (P_NC row tiles past the stream CTAs' 24 P_NS), K eighth bid % 8
    part = bid % P_PARTS
    if const_expr(c["pshard"]):
        # the sharded pooler: a rank's 512 rows are 32 tiles a K part, one a CTA (24 stream, 8 chain)
        nsub = 1
        rt0 = 24 + (bid - CHAIN0) // P_PARTS
    else:
        nsub = P_NC
        rt0 = 24 * P_NS + P_NC * ((bid - CHAIN0) // P_PARTS)
    if const_expr(obd):
        # (``ob``) the tile's weights first (32 KB: landed before the reduce-scatter's
        # partials, which the statistics CTAs poll behind them), then (CTAs AP0 ..) a
        # boundary part's statistics of this rank's partials
        pool_w = pool_w_loads(c, rt0, part, nsub)
        rocdl.sched_barrier(0)
        if bid < AP0 + OB_TASKS:
            stage_ob_stats(c, bid - AP0)
    else:
        if (bid >= AP0) & (bid < AP0 + A_PARTS * S):
            stage_a_part(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS)
    stamp(c, 3)  # the boundary part done (boundary CTAs)
    if const_expr(not obd):
        pool_w = pool_w_loads(c, rt0, part, nsub)
    rocdl.sched_barrier(0)
    if const_expr(obd):
        b1_store(c, b1_loads(c, part))  # (``x_loads_ob`` waits for x)
    else:
        wait_x(c)
    stamp(c, 4)  # every token's x arrived
    stage_passes(
        c, rt0, part, pool_w, None, fx.Int32(0), fx.Int32(0) == fx.Int32(1), nsub
    )
    stamp(c, 5)  # the pooler passes done
    if const_expr(c["pshard"]):
        stage_pool_reduce(c, rt0, part)
        stamp(c, 13, PR_WAVE * 64)  # the reduce share pushed
    if (bid >= AP0) & (bid < AP0 + A_PARTS * S):
        stage_a_part_res(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS)
    stamp(c, 6)  # the boundary's stream update done (boundary CTAs)
    if const_expr(not c["pshard"]):
        if tok < S:
            stage_route(
                c, tok, route_flag
            )  # the xn task runs on stream CTA tok meanwhile


@flyc.jit
def run_stream(c, o_flag=None, QL=8, xin_flag=None, route_flag=None):
    """A stream CTA (``bid < CHAIN0``): its og task (CTAs 0 .. S), its two o tasks
    (``bid`` and ``bid + BLOCKS``), then its pooler weights, the tokens' flags, and
    the router / pooler passes (a router task's rows on CTAs R0 ..: every stream CTA
    loads a tile's rows, the index clamped; they stream while the first pass is
    staged)."""
    S, bid = c["S"], c["bid"]
    run_o(c, o_flag, QL)  # the og rows and the o tasks
    stamp(c, 2)  # the o tasks done
    obd = ob(c)
    if const_expr(o_rs(c)):
        if (bid >= RED_CTA0) & (bid < RED_CTA0 + RS_TASKS):
            if const_expr(obd and ob_g(S) > 1):
                stage_o_reduce_ob(
                    c, bid - RED_CTA0
                )  # the parity's tokens, to every rank
            else:
                stage_o_reduce(
                    c, bid - RED_CTA0
                )  # this rank's o columns, to every rank
    stamp(c, 3)  # (the boundary parts run on the chain CTAs)
    # the pooler job: P_NS row tiles of K eighth bid % 8 (the 24 stream CTAs of an eighth)
    part = bid % P_PARTS
    nsub = 1 if c["pshard"] else P_NS
    rt0 = (bid // P_PARTS) * nsub
    pool_w = pool_w_loads(c, rt0, part, nsub)
    # a router task on CTAs R0 .. (every stream CTA loads a tile's rows, the index
    # clamped): the same K eighth
    r_tile = fx.min((bid - R0) // R_PARTS, fx.Int32(R_TILES - 1))
    if const_expr(c["pshard"]):
        # one pooler tile in flight: the router's rows go out behind it, before the x
        # polls, and land while the boundary runs (after the polls, their round trip
        # sat between x and the first pass)
        router_w = router_w_loads(
            c, r_tile, part, ((bid >= R0) & (bid < R0 + R_TASKS)) if obd else None
        )
        rocdl.sched_barrier(0)
        if const_expr(obd):
            b1_store(c, b1_loads(c, part))  # (``x_loads_ob`` waits for x)
        else:
            wait_x(c)
    else:
        # the replicated pooler's big job in flight: the x polls wait behind it
        # already; the router's rows stream while the first token tile's x lands
        rocdl.sched_barrier(0)
        wait_x(c)
        router_w = router_w_loads(c, r_tile, part)
    stamp(c, 4)  # every token's x arrived
    stage_passes(
        c, rt0, part, pool_w, router_w, r_tile, (bid >= R0) & (bid < R0 + R_TASKS), nsub
    )
    stamp(c, 5)  # the router / pooler passes done
    if const_expr(c["pshard"]):
        stage_pool_reduce(c, rt0, part)
        stamp(c, 6, PR_WAVE * 64)  # the reduce share pushed
    else:
        stamp(c, 6)  # the pooled rows out (with the passes)
    X0 = xn_cta0(c)
    if (bid >= X0) & (bid < X0 + S):
        stage_xn(c, bid - X0, xin_flag)  # token bid - X0's expert input
    if const_expr(c["pshard"]):
        if (bid >= ROUTE0) & (bid < ROUTE0 + S):
            # token bid - ROUTE0's routing: on a stream CTA past the xn tasks, right
            # after its passes (on a chain CTA it waited behind the boundary's stream
            # update, itself behind the CTA's pushes)
            stage_route(c, bid - ROUTE0, route_flag)


@flyc.jit
def run_k2(c, route_flag=None, xin_flag=None, o_flag=None, QL=8):
    """K2's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    in a launch that goes on to the experts, the route and xn tasks raise a
    token's flag once its routing / expert input is out. ``o_flag``: the
    attention ran earlier in this launch, a flag a (request of ``QL`` rows, KV
    head, row tile) once its o rows are out."""
    bid = c["bid"]
    stamp(c, 0)
    if bid >= CHAIN0:
        run_chain(c, route_flag, xin_flag, o_flag, QL)
    else:
        run_stream(c, o_flag, QL, xin_flag, route_flag)
    if const_expr(c["pshard"] and xin_flag is not None and route_flag is not None):
        # a CTA without a route / xn task waits for the last token's flags here, one
        # thread each, before K3's poll of every token's flags (a thread a flag): the
        # ~200 CTAs done early would spin on them all, slowing the hand-offs still due
        S, tid = c["S"], c["tid"]
        X0 = xn_cta0(c)
        if ((bid < X0) | (bid >= X0 + S)) & ((bid < ROUTE0) | (bid >= ROUTE0 + S)):
            # (the route flag only: K3 starts its plan on the routing, before xin)
            if tid == 64:
                c["poll"]([(route_flag, 2 * (S - 1), 1)])
            gpu.barrier()
    stamp(c, 7)  # end
