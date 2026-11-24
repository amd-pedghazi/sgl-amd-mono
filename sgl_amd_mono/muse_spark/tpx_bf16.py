# SPDX-License-Identifier: Apache-2.0
"""TP experts for the bf16-expert layers (0 and 61): every rank reads its 1 / TP of
every picked expert, so no rank waits on another's expert load (EP's slowest rank
holds ~10-12 of a step's 64 picks at 8 rows, ~1.7x the mean).

A rank holds, for every expert, the gate and up rows of its INTER / TP intermediate
columns (``w13`` [EXPERTS, 2 RS, MOE_HIDDEN]) and the down rows of its MOE_HIDDEN / TP
output columns (``w2`` [EXPERTS, CS, INTER]), as the MXFP4 layers' TP experts
(``weights.tp_expert_copies``). A step::

    ug   a program a (picked expert, column tile): bf16(silu(gate) up) of every token
         that picked it -> mids [pick, RS]          (pick = token * TOP_K + slot)
    all-gather the mids -> [TP, pick, RS]
    dn   a program a (picked expert, column tile): bf16(mids @ w2.T) -> y [pick, CS]
    all-gather y -> [TP, pick, CS]
    fin  a program a token: v = bf16(y pen_w), out = sum over the slots of
         score v / sqrt(mean v^2 + eps), fp32 -> bf16

The rounding points are those of SGLang's bf16 fused MoE (``muse_moe_flydsl``); its
finalize sums each EP rank's slots to bf16 and the all-reduce sums those, here every
rank sums all the slots once in fp32 (no all-reduce: the result is whole on every rank).

FP8 slices (``fp8_rows``, a numerics change): E4M3 weights with a power-of-two scale a
(row, 128 columns), so every dequantized weight is a bf16 value and the GEMVs are the
bf16 ones on those weights (each group's dot scaled after the MFMA); half the bytes.
"""

import torch
import triton
import triton.language as tl

_BM = 16  # the token rows of an MFMA tile (TOP_K picks a token, a token picks an expert once)
# (weight rows, K step, stages) a program: measured on MI355X at 8 rows, ~50 picked experts (ug 5.5 TB/s, dn 4.1)
_UG = (64, 256, 2)
_DN = (32, 128, 3)
FP8_GROUP = 128  # the columns a scale covers; the FP8 launches step K by one group (tiles: MI355X, 55 experts)
_UG8 = (64, 3)
_DN8 = (64, 3)


def fp8_rows(w: torch.Tensor, chunk: int = 16):
    """``w [E, N, K]`` bf16 as (E4M3 codes [E, N, K], fp32 power-of-two scales
    [E, N, K / FP8_GROUP]): the smallest power of two that puts a group's max at or
    below 448, the codes rounded to nearest."""
    e, n, k = w.shape
    q = torch.empty((e, n, k), dtype=torch.float8_e4m3fn, device=w.device)
    sc = torch.empty((e, n, k // FP8_GROUP), dtype=torch.float32, device=w.device)
    for i in range(0, e, chunk):
        g = w[i : i + chunk].float().view(-1, n, k // FP8_GROUP, FP8_GROUP)
        amax = g.abs().amax(-1).clamp_min(1e-30)
        s_ = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        sc[i : i + chunk] = s_
        q[i : i + chunk] = (g / s_[..., None]).view(-1, n, k).to(torch.float8_e4m3fn)
    from sgl_amd_mono.muse_spark import tpx_fp8x

    return tpx_fp8x.shuffle(
        q
    ), sc  # (the FlyDSL GEMVs' layout: a permutation of the same codes)


@triton.jit
def _leader(ids_ptr, p, NP: tl.constexpr):
    """Pick p's expert, and whether p is that expert's first pick (the one program that works it)."""
    q = tl.arange(0, NP)
    ids = tl.load(ids_ptr + q)
    e = tl.load(ids_ptr + p)
    first = tl.sum(((ids == e) & (q < p)).to(tl.int32), axis=0) == 0
    return e, first


@triton.jit
def _token_picks(ids_ptr, e, M, TOPK: tl.constexpr, BM: tl.constexpr):
    """Per token row: the pick (token * TOPK + slot) that chose expert e, and whether one did."""
    t = tl.arange(0, BM)
    s = tl.arange(0, TOPK)
    ids = tl.load(
        ids_ptr + t[:, None] * TOPK + s[None, :], mask=t[:, None] < M, other=-1
    )
    hit = ids == e
    has = tl.sum(hit.to(tl.int32), axis=1) > 0
    slot = tl.sum(tl.where(hit, s[None, :], 0), axis=1)
    return t, t * TOPK + slot, has


@triton.jit
def _mac(acc, a, w, s_ptr, FP8: tl.constexpr):
    """acc + a @ w; FP8: the codes (exact in bf16) times one group, then that group's
    power-of-two column scales (exact), so only the fp32 sum order differs from bf16."""
    if FP8:
        acc += tl.dot(a, w.to(tl.bfloat16)) * tl.load(s_ptr)[None, :]
    else:
        acc = tl.dot(a, w, acc)
    return acc


@triton.jit
def _ug(
    X,
    IDS,
    W13,
    S13,
    MID,
    M,
    H: tl.constexpr,
    RS: tl.constexpr,
    NP: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FP8: tl.constexpr,
):
    e, first = _leader(IDS, tl.program_id(0), NP)
    if first:
        t, pick, has = _token_picks(IDS, e, M, TOPK, BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        wg = W13 + e.to(tl.int64) * (2 * RS * H) + n[None, :] * H + kk[:, None]
        wu = wg + RS * H
        sg = (
            S13 + e.to(tl.int64) * (2 * RS * (H // BK)) + n * (H // BK)
        )  # FP8: BK is one scale group
        su = sg + RS * (H // BK)
        xp = X + t[:, None] * H + kk[None, :]
        g = tl.zeros((BM, BN), tl.float32)
        u = tl.zeros((BM, BN), tl.float32)
        for _ in range(H // BK):
            a = tl.load(xp, mask=has[:, None], other=0.0)
            g = _mac(g, a, tl.load(wg, cache_modifier=".cg"), sg, FP8)
            u = _mac(u, a, tl.load(wu, cache_modifier=".cg"), su, FP8)
            xp += BK
            wg += BK
            wu += BK
            sg += 1
            su += 1
        mid = g / (1.0 + tl.exp2(-1.4426950408889634 * g)) * u
        tl.store(
            MID + pick[:, None] * RS + n[None, :],
            mid.to(tl.bfloat16),
            mask=has[:, None],
        )


@triton.jit
def _dn(
    MIDS,
    IDS,
    W2,
    S2,
    Y,
    M,
    I: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    NP: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FP8: tl.constexpr,
):
    e, first = _leader(IDS, tl.program_id(0), NP)
    if first:
        t, pick, has = _token_picks(IDS, e, M, TOPK, BM)
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        wp = W2 + e.to(tl.int64) * (CS * I) + n[None, :] * I + kk[:, None]
        sp = (
            S2 + e.to(tl.int64) * (CS * (I // BK)) + n * (I // BK)
        )  # FP8: BK is one scale group
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, I, BK):
            # the gathered mids [TP, pick, RS]: column k0 + j is rank k0 // RS's column k0 % RS + j
            ap = (
                MIDS
                + (k0 // RS) * (NP * RS)
                + pick[:, None] * RS
                + (k0 % RS)
                + kk[None, :]
            )
            a = tl.load(ap, mask=has[:, None], other=0.0)
            acc = _mac(acc, a, tl.load(wp, cache_modifier=".cg"), sp, FP8)
            wp += BK
            sp += 1
        tl.store(
            Y + pick[:, None] * CS + n[None, :], acc.to(tl.bfloat16), mask=has[:, None]
        )


@triton.jit
def _fin(
    YS,
    SCORES,
    PEN,
    OUT,
    eps,
    H: tl.constexpr,
    CS: tl.constexpr,
    NP: tl.constexpr,
    TOPK: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.arange(0, H)
    s = tl.arange(0, TOPK)
    w = tl.load(PEN + h).to(tl.float32)
    # the gathered y [TP, pick, CS]: column h is rank h // CS's column h % CS
    col = (h // CS) * (NP * CS) + (h % CS)
    pick = t * TOPK + s
    y = tl.load(YS + pick[:, None] * CS + col[None, :]).to(tl.float32)
    v = (y * w[None, :]).to(tl.bfloat16).to(tl.float32)
    r = tl.load(SCORES + pick) / tl.sqrt(tl.sum(v * v, axis=1) / H + eps)
    tl.store(OUT + t * H + h, tl.sum(r[:, None] * v, axis=0).to(tl.bfloat16))


def tpx_moe(x, scores, ids, w13, w2, pen_w, eps, all_gather, s13=None, s2=None):
    """The bf16 TP-experts FFN of ``x`` [M, MOE_HIDDEN] bf16 under the routing ``scores`` /
    ``ids`` [M, TOP_K] (fp32 / int32, global expert ids): the post-expert-normed weighted
    expert sum [M, MOE_HIDDEN] bf16, whole on every rank. ``w13`` / ``w2`` are the rank's
    slices (bf16, or FP8 codes with their ``s13`` / ``s2`` scales from ``fp8_rows``);
    ``all_gather(t)`` concatenates every rank's ``t`` along dim 0 in rank order."""
    fp8 = s13 is not None
    m, h = x.shape
    topk = ids.shape[1]
    rs, cs, inter = w13.shape[1] // 2, w2.shape[1], w2.shape[2]
    np_ = m * topk
    assert (
        m <= _BM and np_ & (np_ - 1) == 0 and x.is_contiguous() and ids.is_contiguous()
    )
    mid = torch.empty((np_, rs), dtype=torch.bfloat16, device=x.device)
    if fp8:
        # the FlyDSL GEMVs (tpx_fp8x: bitwise these Triton kernels' results, on the shuffled slices)
        from sgl_amd_mono.muse_spark import tpx_fp8x

        tpx_fp8x.ug(x, ids, w13, s13, mid, 4)
        mids = all_gather(mid)
        y = torch.empty((np_, cs), dtype=torch.bfloat16, device=x.device)
        tpx_fp8x.dn(mids, ids, w2, s2, y, 2)
        ys = all_gather(y)
        out = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
        _fin[(m,)](ys, scores, pen_w, out, eps, h, cs, np_, topk, num_warps=8)
        return out
    bn, bk, st = (*_UG8[:1], FP8_GROUP, _UG8[1]) if fp8 else _UG
    _ug[(np_, rs // bn)](
        x,
        ids,
        w13,
        s13 if fp8 else w13,
        mid,
        m,
        h,
        rs,
        np_,
        topk,
        _BM,
        bn,
        bk,
        fp8,
        num_warps=4,
        num_stages=st,
        matrix_instr_nonkdim=16,
    )
    mids = all_gather(mid)
    y = torch.empty((np_, cs), dtype=torch.bfloat16, device=x.device)
    bn, bk, st = (*_DN8[:1], FP8_GROUP, _DN8[1]) if fp8 else _DN
    _dn[(np_, cs // bn)](
        mids,
        ids,
        w2,
        s2 if fp8 else w2,
        y,
        m,
        inter,
        rs,
        cs,
        np_,
        topk,
        _BM,
        bn,
        bk,
        fp8,
        num_warps=4,
        num_stages=st,
        matrix_instr_nonkdim=16,
    )
    ys = all_gather(y)
    out = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    _fin[(m,)](ys, scores, pen_w, out, eps, h, cs, np_, topk, num_warps=8)
    return out
