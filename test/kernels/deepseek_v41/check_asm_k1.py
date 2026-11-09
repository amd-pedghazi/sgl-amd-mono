"""Check the asm K1 (``asm/k1.py``) against SGLang's decode chain on gfx950 (torchrun, TP4: the
asm kernels' shapes; replicated weights alike on every rank, wq_b / wo_a / wo_b and the sinks
this rank's shard), on SGLang's SWA cache layout and lists.

Stage ``front`` (seam, then front; ``--chain v3``: seam_v3 + front_v3, ``v1``: seam + front).
The front rounds differently from SGLang by design: v3 quantizes the attention norm's input
without the row's rstd (folded in downstream), and both chains' record writer quantizes the bf16
of the normed kv (SGLang's from f32). So besides SGLang's outputs, each check also has an fp64
reference without activation quantization, and each launch is checked on its own inputs:

- seam: the residual (as check_ffn.py bounds it); the collapse against fp64 on the pre mix it
  reads (the input ``pre``, the previous launch's), in ulps of the largest term, at most SGLang's
  error + 0.5 ulp; the
  mixes against SGLang's boundary kernel with K2's asm bound (``ASM_MIX_RTOL``, ``ASM_MIX_ATOL``);
  v1: the fin's MXFP8 of the normed input bit for bit against SGLang's norm on the rows whose
  collapse is SGLang's;
- wqkv: the kv rows (v3: and the q latent's sums of squares) against fp64 on its own MXFP8 x;
- wq_b: q (q RoPE included) against fp64 on its own MXFP8 x;
- the SWA records: against SGLang's writer on the front's own kv rows (E8M0 scales within 1,
  e4m3 codes of equal-scale blocks within one step, the error against the fp64 of that input at
  most ERR_RATIO times SGLang's writer's own), and both chains' records against the fp64
  reference (asm at most ERR_RATIO times SGLang's); nothing written outside the step's records,
  a pad slot (-1) skipped;
- q (v3: with q_norm's rrms, as the attention folds it in) against SGLang's q
  (check_k1.query_ref) and both against the fp64 reference;
- a graph replay bit for bit against eager.
Variants: ``fold`` (a pending FFN post), ``settled`` (no post owed: after an Engram write)
and ``index`` (fold on an index source): the index step's inputs (normed, qr) on the
scratch they are made from (within 1 bf16 step), and against SGLang's norm output and q norm
and both against fp64 (``INDEX_RATIO_V3`` on v3, ``ERR_RATIO`` on v1); on v3, x1's slot holds
y' = bf16(bf16(collapse) · attn_norm.w), checked as the collapse (SGLang's analog: its
collapse times attn_norm.w).

Stage ``attend``: the attention alone on SGLang's window and top-k lists (``check_attend``).
Stage ``back``: wo_a, then wo_b with all-reduce #1, both wires (``check_back``); wo_b against
its own wo_a output (bf16 wire ``k2.wo_b.rel``, fp8 wire ``FP8_WIRE_REL``).
Stage ``call``: ``AsmK1.__call__`` end to end, the hand-offs between the pieces (``check_call``),
on a consumer layer, an index source (the index step's place) and a window-only layer
(ratio 0: layers 0 and 1)."""

import argparse
import math
import os
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F
from check_ffn import (
    HC,
    HC_EPS,
    HIDDEN,
    RMS_EPS,
    finish,
    hc_post_fp64,
    seam_inputs,
    term_ulps,
)
from check_k1 import NOPE, Q_RANK, RECORD, attn_weights, query_ref, rope_table

from sgl_amd_mono.deepseek_v41.spec import NUMERICS

HEAD, KV = 512, 512
BLOCK, PAGES = 256, 64  # SGLang's SWA pages hold 256 rows (kv_cache_configurator)
SWA_PAGES, COMP_PAGES, COMP_BLOCK = 12, 32, 128  # the attend stage's random caches
SWA_WIDTH, TOPK = 128, 512
# the mixes: K2's asm bound (check_asm_k2.py's post / comb), |asm - SGLang| <=
# ASM_MIX_ATOL + ASM_MIX_RTOL · |SGLang|: vLLM's seam sums and iterates in its own order
ASM_MIX_RTOL, ASM_MIX_ATOL = 2e-3, 1e-5
ATTN_COS = NUMERICS.bound("k2.attn_out.cosine")
# the fp8 wire's MXFP8 of the partials (about 0.035 seen), as check_asm_k2.py allows its fp8
# wire; the bf16 wire: k2.wo_b.rel
FP8_WIRE_REL = 0.05
# each launch against fp64 on its own inputs: its bf16 output rounding and f32 sums
STAGE_REL, STAGE_COS = 4e-3, 0.99999
# the record and q bound: the front's error against fp64 at most this times
# SGLang's; bit for bit is not reachable (the writers quantize bf16 and f32 sources)
ERR_RATIO = 1.25
# an index source's normed input and qr against fp64, at
# most this times SGLang's on v3: its normed is rounded twice (y', then y' · rstd), its qr
# carries wq_b's MXFP8 rounding (the q vLLM's v3 indexer reads), so about √2 is expected;
# v1 at most ERR_RATIO
INDEX_RATIO_V3 = 1.5
# "index": "fold" on an index source (the front also writes the index step's inputs)
VARIANTS = ("fold", "settled")
CALLS = ("consumer", "index", "window")


def cosine(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def rel(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


def collapse_ulps(got, pre, res64, w=None):
    """A collapse (sum_h pre[h] · res[h], times ``w`` for y') against fp64 on its own pre
    mixes, in bf16 ulps of the largest term (a near-cancelling sum has no ulp of its own)."""
    terms = pre.double().reshape(-1, HC, 1) * res64
    if w is not None:
        terms = terms * w.double()
    return term_ulps(got, terms.sum(1), terms.abs().amax(1))


def rank_sum(part):
    """The ranks' bf16 partials summed in rank order in f32, then bf16 (the bf16 wire's rule)."""
    parts = [torch.empty_like(part) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, part)
    ref = parts[0].float()
    for p in parts[1:]:
        ref = ref + p.float()
    return ref.bfloat16()


def deq_w(w, codes):
    """E4m3 [N, K] with E8M0 [N / 32, K / 32] codes -> fp64."""
    s = torch.exp2(codes.double() - 127)
    return w.double() * s.repeat_interleave(32, 0).repeat_interleave(32, 1)


def unfrag_x(xf, sf, kx):
    """gemv8.frag_x's inverse: (e4m3 codes [256, kx], E8M0 [256, kx / 32])."""
    x = xf.view(16, kx // 128, 2, 4, 16, 16).permute(0, 4, 1, 2, 3, 5).reshape(256, kx)
    s = sf.view(16, kx // 128, 4, 16).permute(0, 3, 1, 2).reshape(256, kx // 32)
    return x, s


def deq_x(x, s):
    """MXFP8 codes [n, k] with E8M0 per 32 columns -> fp64."""
    return x.view(torch.float8_e4m3fn).double() * torch.exp2(
        s.double() - 127
    ).repeat_interleave(32, 1)


def rope64(v, cs, pos, first):
    """GPT-J RoPE in fp64 on dims first.. of each head of v [M, heads, D]."""
    c = cs[pos].double()[:, None, :32]
    s = cs[pos].double()[:, None, 32:]
    v = v.clone()
    e, o = v[..., first::2].clone(), v[..., first + 1 :: 2].clone()
    v[..., first::2], v[..., first + 1 :: 2] = e * c - o * s, e * s + o * c
    return v


def records(buf, slot):
    """The 584 bytes of slots ``slot`` (>= 0): 576 data bytes at row·576 of the page, then 8
    scale bytes at BLOCK·576 + row·8."""
    page, off = slot // BLOCK, slot % BLOCK
    ar = torch.arange(576, device=buf.device)
    col = torch.cat(
        [off[:, None] * 576 + ar, BLOCK * 576 + off[:, None] * 8 + ar[:8]], 1
    )
    return buf[page[:, None].expand_as(col), col]


def record_mask(buf, slot):
    m = torch.zeros_like(buf, dtype=torch.bool)
    page, off = slot // BLOCK, slot % BLOCK
    ar = torch.arange(576, device=buf.device)
    col = torch.cat(
        [off[:, None] * 576 + ar, BLOCK * 576 + off[:, None] * 8 + ar[:8]], 1
    )
    m[page[:, None].expand_as(col), col] = True
    return m


def decode(rec):
    """Records [n, 584] -> fp64 [n, 512]: NoPE e4m3 times its 64 dims' 2^(scale - 127), RoPE bf16."""
    nope = rec[:, :NOPE].contiguous().view(torch.float8_e4m3fn).double()
    nope = nope * torch.exp2(rec[:, 576:583].double() - 127).repeat_interleave(64, 1)
    rope = rec[:, NOPE:576].contiguous().view(torch.bfloat16).double()
    return torch.cat([nope, rope], 1)


def e4m3_steps(a, b):
    """Max distance of two e4m3 code tensors in representable steps (same sign)."""

    def key(c):
        c = c.long()
        mag = c & 0x7F
        return torch.where(c & 0x80 != 0, -mag, mag)

    return (key(a) - key(b)).abs().max().item() if a.numel() else 0


def bf16_ulps(a, b):
    """Max distance of two bf16 tensors in representable steps."""

    def key(t):
        i = t.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (key(a) - key(b)).abs().max().item() if a.numel() else 0


def sglang_lists(M, lists, dev, junk):
    """check_k2.key_lists' keys in SGLang's metadata form: swa_page_indices [M, 128] (the
    window, then -1) with swa_topk_lengths, and sparse_page_indices [M, 512] (the top-k,
    then ``junk`` slots the fixed-stride mode must not read) with sparse_topk_lengths."""
    swa_idx = torch.full((M, SWA_WIDTH), -1, dtype=torch.int32)
    swa_lens = torch.zeros(M, dtype=torch.int32)
    tk_idx = junk[: M * TOPK].view(M, TOPK).cpu().clone()
    tk_lens = torch.zeros(M, dtype=torch.int32)
    for t, (swa, comp) in enumerate(lists):
        swa_idx[t, : len(swa)] = torch.tensor(swa, dtype=torch.int32)
        tk_idx[t, : len(comp)] = torch.tensor(comp, dtype=torch.int32)
        swa_lens[t], tk_lens[t] = len(swa), len(comp)
    return tuple(t.to(dev) for t in (swa_idx, swa_lens, tk_idx, tk_lens))


def check_attend(k1, w, M, args, sink, cos_sin, dev, rank):
    """The attention alone on SGLang's lists: random V4 records (256-row SWA pages, 128-row
    compressed pages), random q (v3: rr = 1 through rstd = 1 and Σ = 1280), against
    check_k2.attention_xo then MXFP8 as the kernel's output; with and without compressed
    keys. Returns the failures."""
    from check_ffn import mxfp8_qdq
    from check_k2 import attention_xo, key_lists, make_cache

    from sgl_amd_mono.deepseek_v41.asm.k1_runner import MAX_M

    run, fails = k1.run, []
    gk = torch.Generator().manual_seed(4000 + M)
    gd = torch.Generator(device=dev).manual_seed(4100 + M)
    swa_raw, swa_keys = make_cache(SWA_PAGES, BLOCK, gd, dev)
    comp_raw, comp_keys = make_cache(COMP_PAGES, COMP_BLOCK, gd, dev)
    pad = (M - 1,)
    _, _, lists = key_lists(M, gk, dev, pad_rows=pad)
    # key_lists draws slots below 48 x 64 (SWA) and 32 x 128 (compressed): both caches hold them
    junk = torch.randint(
        0, COMP_PAGES * COMP_BLOCK, (MAX_M * TOPK,), generator=gk, dtype=torch.int32
    )
    keys = sglang_lists(M, lists, dev, junk)
    pos = torch.randint(0, cos_sin.shape[0], (M,), generator=gk).to(dev)
    q = (torch.randn(M, 16, HEAD, generator=gd, device=dev) * 0.05).bfloat16()
    if k1.v3:
        run.s["rstd"][:M] = 1.0
        run.s["ss"].view(MAX_M, 40)[:M] = 32.0
    live = torch.tensor([t not in pad for t in range(M)], device=dev)
    for ratio in (1, 0):
        L = types.SimpleNamespace(ratio=ratio)
        run.s["o"].fill_(0x7F)  # e4m3 NaN: a row the kernel skips shows up
        k1.attend(w, L, q, pos, (swa_raw, BLOCK, comp_raw, COMP_BLOCK), keys)
        torch.cuda.synchronize()
        ox, os_ = unfrag_x(run.s["o"], run.s["os"], 16 * HEAD)
        got = deq_x(ox[:M], os_[:M])
        use = lists if ratio else [(s, []) for s, _ in lists]
        ref = attention_xo(q, swa_keys, comp_keys, use, sink, pos, cos_sin).double()
        ref_mx = mxfp8_qdq(ref.float(), "vllm").double()
        c_mx, c_ex = cosine(got[live], ref_mx[live]), cosine(got[live], ref[live])
        row = F.cosine_similarity(got[live], ref_mx[live], dim=1).min().item()
        checks = {
            "attention": c_mx >= ATTN_COS and row > 0.999,
            "pad row reads nothing": bool((got[~live] == 0).all()),
            "finite": bool(torch.isfinite(got).all()),
        }
        bad = [k for k, v in checks.items() if not v]
        if rank == 0:
            print(
                f"M={M:3d} {args.chain} attend ratio {ratio} {'PASS' if not bad else 'FAIL ' + ','.join(bad)}"
                f" | cos vs MXFP8 of the reference {c_mx:.6f} (min row {row:.5f}), vs the "
                f"reference {c_ex:.6f}",
                flush=True,
            )
        fails += [f"M={M} attend ratio {ratio} {b}" for b in bad]
    return fails


def mxfp8(x):
    """MXFP8 of x [n, k] with the kernels' ceil rule: (e4m3 codes as uint8, E8M0 per 32 columns)."""
    from check_ffn import code_ceil

    g = x.float().view(x.shape[0], -1, 32)
    code = code_ceil(g.abs().amax(-1).clamp_min(2.0**-126) / 448.0).clamp(0, 254)
    q = (g * torch.exp2(127.0 - code.float())[..., None]).to(torch.float8_e4m3fn)
    return q.view(x.shape).view(torch.uint8), code.to(torch.uint8)


def check_back(k1, w, M, args, dev, rank, tp, cpu):
    """wo_a, then wo_b with all-reduce #1 (``back_ar``) on an attention output in the
    attention's MXFP8 frag, with this rank's own wo_a / wo_b shard, both wires: wo_a's MXFP8
    output against fp64, the all-reduced output against the rank-order sum of the ranks'
    bf16 partials (the bf16 wire's rule) of wo_b on the kernel's own wo_a output, equal on
    every rank, no wait timeouts."""
    from check_ffn import mxfp8_qdq

    from sgl_amd_mono.deepseek_v41.asm.ar_runner import WoBAllReduce
    from sgl_amd_mono.deepseek_v41.asm.kernels import gemv8

    run, fails = k1.run, []
    if k1.wob is None:  # collective: every rank builds it at its first back stage
        k1.wob = WoBAllReduce(rank, cpu, tp)
    gb = torch.Generator(device=dev).manual_seed(5000 + 97 * rank + M)

    def fp8(n, k):
        wt = (torch.randn(n, k, generator=gb, device=dev) * 0.5).to(torch.float8_e4m3fn)
        return wt, torch.randint(
            118, 122, (n // 32, k // 32), generator=gb, device=dev
        ).to(torch.uint8)

    woa, woa_s = fp8(2048, 4096)
    wob, wob_s = fp8(HIDDEN, 2048)
    wb = dict(w)
    wb["wo_a"] = gemv8.frag_w(woa.view(torch.uint8), woa_s, gemv8.WO_A)
    wb["wo_b"] = gemv8.frag_w(wob.view(torch.uint8), wob_s, gemv8.WO_B1)
    oq, os_ = mxfp8(torch.randn(M, 16 * HEAD, generator=gb, device=dev) * 0.1)
    xf, sf = gemv8.frag_x(oq, os_)
    run.s["o"].copy_(xf)
    run.s["os"].copy_(sf)
    pos = torch.zeros(M, dtype=torch.int64, device=dev)
    od = deq_x(oq, os_)
    ya = torch.einsum(
        "tgk,grk->tgr", od.view(M, 2, 4096), deq_w(woa, woa_s).view(2, 1024, 4096)
    ).reshape(M, 2048)
    z_ref = mxfp8_qdq(ya.bfloat16().float(), "vllm").double()
    wob64 = deq_w(wob, wob_s)
    ref_req = rank_sum(
        (z_ref @ wob64.T).bfloat16()
    )  # on the reference's own wo_a output
    for wire in ("bf16", "fp8"):
        out = run.back_ar(wb, M, pos, k1.wob, wire)
        torch.cuda.synchronize()
        zx, zs = unfrag_x(run.s["z"], run.s["zs"], 2048)
        z_got = deq_x(zx[:M], zs[:M])
        z_rel = rel(z_got, z_ref)
        o_rel = rel(out, rank_sum((z_got @ wob64.T).bfloat16()))
        o_req = rel(out, ref_req)
        same = out.clone()
        dist.broadcast(same, 0)
        errors = k1.wob.errors()
        bound = NUMERICS.bound("k2.wo_b.rel") if wire == "bf16" else FP8_WIRE_REL
        checks = {
            "wo_a": z_rel <= NUMERICS.bound("k2.wo_a.rel"),
            "all-reduced output": o_rel <= bound,
            "ranks agree": torch.equal(same, out),
            "no wait timeouts": errors == 0,
        }
        bad = [k for k, v in checks.items() if not v]
        if rank == 0:
            print(
                f"M={M:3d} {args.chain} back {wire} {'PASS' if not bad else 'FAIL ' + ','.join(bad)}"
                f" | wo_a MXFP8 rel {z_rel:.2e} (bound {NUMERICS.bound('k2.wo_a.rel'):.0e}), "
                f"output rel {o_rel:.2e} on its own wo_a output (bound {bound:.0e}; "
                f"{o_req:.2e} on the reference's), timeouts {errors}",
                flush=True,
            )
        fails += [f"M={M} back {wire} {b}" for b in bad]
    return fails


def check_call(k1, w, M, args, inputs, sink, cos_sin, back64, dev, rank, tp, cpu, kind):
    """``AsmK1.__call__`` end to end on a layer of ``kind`` (CALLS): the step's records go
    into a cache of random records, each token's window holds its own new record and other
    rows, its top-k random compressed rows. The hand-offs are checked against references
    built from the pieces' own outputs: the attention over the cache as the front left it,
    with the front's q (and rr), against attention_xo; the output against wo_a / wo_b on the
    attention's own MXFP8 output, summed over the ranks in rank order (the bf16 wire).
    ``back64``: this rank's wo_a [2, 1024, 4096] and wo_b [5120, 2048] in fp64. "index": the
    top-k slots hold other slots until the index step writes the step's own, so the
    attention matches only if the step ran once, after the front and before it. "window":
    no top-k, and the settled residual of the first layer (x None) comes back unchanged."""
    from check_ffn import mxfp8_qdq
    from check_k2 import attention_xo, make_cache

    from sgl_amd_mono.deepseek_v41.asm.ar_runner import WoBAllReduce
    from sgl_amd_mono.deepseek_v41.asm.k1_runner import MAX_M

    run = k1.run
    if k1.wob is None:  # collective: every rank builds it at its first stage needing it
        k1.wob = WoBAllReduce(rank, cpu, tp)
    gk = torch.Generator().manual_seed(6000 + M)
    gd = torch.Generator(device=dev).manual_seed(6100 + M)
    res, x, post, comb, pre, pos, slot = inputs(M, 7)
    ratio = 0 if kind == "window" else 1
    swa, _ = make_cache(PAGES, BLOCK, gd, dev)
    comp, comp_keys = make_cache(COMP_PAGES, COMP_BLOCK, gd, dev)
    lists = []
    for t in range(M):
        s = int(slot[t])
        if s < 0:
            lists.append(([], []))
            continue
        n = int(torch.randint(1, SWA_WIDTH + 1, (1,), generator=gk))
        others = torch.randperm(PAGES * BLOCK, generator=gk)
        others = others[others != s][: n - 1].tolist()
        ntopk = int(torch.randint(0, TOPK + 1, (1,), generator=gk)) if ratio else 0
        cl = torch.randperm(COMP_PAGES * COMP_BLOCK, generator=gk)[:ntopk].tolist()
        if ntopk > 2:
            cl[ntopk // 2] = -1
        lists.append(([s] + others, cl))
    junk = torch.randint(
        0, COMP_PAGES * COMP_BLOCK, (MAX_M * TOPK,), generator=gk, dtype=torch.int32
    )
    keys = sglang_lists(M, lists, dev, junk)
    index, calls = None, []
    if kind == "index":
        own = keys[2].clone()
        keys[2].copy_(junk.flip(0)[: M * TOPK].view(M, TOPK))

        def index(normed, qr):
            calls.append((tuple(normed.shape), tuple(qr.shape), normed.dtype, qr.dtype))
            keys[2].copy_(own)

    settled = kind == "window"
    L = types.SimpleNamespace(i=2, ratio=ratio)
    out, r, post_a, comb_a, pre_a = k1(
        w, L, None if settled else x, res, None if settled else post,
        None if settled else comb, pre, pos, slot, (swa, BLOCK, comp, COMP_BLOCK), keys,
        index, "bf16",
    )  # fmt: skip
    torch.cuda.synchronize()
    live = slot >= 0
    # the attention: the front's q and rr, the cache with the step's own records
    q = run.s["q"][:M].view(M, 16, HEAD).double()
    if k1.v3:
        rd = run.s["rstd"][:M].double()
        ss = run.s["ss"].view(MAX_M, 40)[:M].double().sum(1)
        q = q * (rd * torch.rsqrt(rd * rd * ss / Q_RANK + RMS_EPS))[:, None, None]
    all_slots = torch.arange(PAGES * BLOCK, device=dev)
    swa_keys = decode(records(swa, all_slots)).float()
    xo = attention_xo(
        q.float(), swa_keys, comp_keys, lists, sink, pos, cos_sin
    ).double()
    ox, os_ = unfrag_x(run.s["o"], run.s["os"], 16 * HEAD)
    o_got = deq_x(ox[:M], os_[:M])
    o_ref = mxfp8_qdq(xo.float(), "vllm").double()
    a_cos = cosine(o_got[live], o_ref[live])
    a_row = F.cosine_similarity(o_got[live], o_ref[live], dim=1).min().item()
    # wo_a on the attention's own output; wo_b on wo_a's own output, the partials in rank order
    woa_f, wob_f = back64
    ya = torch.einsum("tgk,grk->tgr", o_got.view(M, 2, 4096), woa_f).reshape(M, 2048)
    z = mxfp8_qdq(ya.bfloat16().float(), "vllm").double()
    zx, zs = unfrag_x(run.s["z"], run.s["zs"], 2048)
    z_got = deq_x(zx[:M], zs[:M])
    z_rel = rel(z_got[live], z[live])
    o_rel = rel(out[live], rank_sum((z_got @ wob_f.T).bfloat16())[live])
    same = out.clone()
    dist.broadcast(same, 0)
    errors = k1.wob.errors()
    bf = torch.bfloat16
    checks = {
        "attention on the front's q and records": a_cos >= ATTN_COS and a_row > 0.999,
        "wo_a": z_rel <= NUMERICS.bound("k2.wo_a.rel"),
        "output": o_rel <= NUMERICS.bound("k2.wo_b.rel"),
        "ranks agree": torch.equal(same, out),
        "no wait timeouts": errors == 0,
        "finite": all(
            bool(torch.isfinite(t).all()) for t in (out, r, post_a, comb_a, pre_a)
        ),
        "index step once, on the index rows": calls
        == ([((M, HIDDEN), (M, Q_RANK), bf, bf)] if kind == "index" else []),
    }
    if settled:  # the identity post-mix: equal values (a -0 may come back +0)
        checks["settled residual unchanged"] = torch.equal(r.float(), res.float())
    bad = [k for k, v in checks.items() if not v]
    if rank == 0:
        print(
            f"M={M:3d} {args.chain} call {kind:8s} "
            f"{'PASS' if not bad else 'FAIL ' + ','.join(bad)} | attention cos "
            f"{a_cos:.6f} (min row {a_row:.5f}), wo_a MXFP8 rel {z_rel:.2e}, output rel "
            f"{o_rel:.2e} on its own wo_a output, timeouts {errors}, index step calls "
            f"{len(calls)}",
            flush=True,
        )
    return [f"M={M} call {kind} {b}" for b in bad]


def fake_layer(k1w, k2w, shuffled):
    """The spec's _Layer as AsmK1.weights reads it: SGLang's module attributes and the
    monokernel's K1 / K2 weights."""
    lin = types.SimpleNamespace(mxfp8_aiter_preshuffled=shuffled)
    attn = types.SimpleNamespace(
        wqkv_a=lin, wq_b=lin, wo_b=lin, indexer=None, eps=RMS_EPS,
        softmax_scale=HEAD**-0.5,
    )  # fmt: skip
    module = types.SimpleNamespace(
        self_attn=attn,
        input_layernorm=types.SimpleNamespace(variance_epsilon=RMS_EPS),
        rms_norm_eps=RMS_EPS,
        hc_eps=HC_EPS,
        hc_sinkhorn_iters=20,
    )
    return types.SimpleNamespace(
        i=2, module=module, k1=k1w, k2=k2w, ratio=1, index=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="64")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument(
        "--shuffled", action="store_true", help="SGLang's aiter (16, 16) preshuffle"
    )
    ap.add_argument("--chain", choices=("v3", "v1"), default="v3", help="AsmK1's chain")
    ap.add_argument(
        "--stages", default="front", help="front, attend, back, call (comma-separated)"
    )
    ap.add_argument("--calls", default=",".join(CALLS), help="the call stage's layers")
    args = ap.parse_args()
    v3 = args.chain == "v3"
    stages = set(args.stages.split(","))
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")  # WoBAllReduce's IPC handle exchange
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_k_norm_rope_flashmla
    from sglang.kernels.ops.gemm.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        prepare_mxfp8_native_weight,
    )
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        hc_boundary_fused_deferred,
        rmsnorm_with_sinkhorn,
    )

    from sgl_amd_mono.deepseek_v41.asm.k1 import AsmK1
    from sgl_amd_mono.deepseek_v41.asm.k1_runner import MAX_M, K1Runner
    from sgl_amd_mono.deepseek_v41.asm.kernels.build import build_lock
    from sgl_amd_mono.deepseek_v41.gfx950.runner import K1Weights, K2Weights

    assert tp == 4, "the asm kernels are TP4's (16 local heads, wo_a in 2 groups)"
    heads = 64 // tp
    # replicated weights alike on every rank; wq_b, wo_a, wo_b and the sinks this rank's shard
    g = torch.Generator(device=dev).manual_seed(5)
    gr = torch.Generator(device=dev).manual_seed(500 + rank)
    a = attn_weights(heads, dev, g)
    shard = attn_weights(heads, dev, gr)
    a["wqb"], a["wqb_s"] = shard["wqb"], shard["wqb_s"]
    del shard
    a["q_norm"] = (1 + 0.2 * torch.randn(Q_RANK, generator=g, device=dev)).bfloat16()
    hc_fn = torch.randn(24, HC * HIDDEN, generator=g, device=dev) * 0.01
    hc_scale = (torch.randn(3, generator=g, device=dev) * 0.5).abs() + 0.5
    hc_base = torch.randn(24, generator=g, device=dev) * 0.1
    norm = (1 + torch.randn(HIDDEN, generator=g, device=dev) * 0.1).bfloat16()
    cos_sin = rope_table(dev)
    freqs = torch.complex(cos_sin[:, :32], cos_sin[:, 32:])  # SGLang's freqs_cis

    def fp8(n, k):
        w = (torch.randn(n, k, generator=gr, device=dev) * 0.5).to(torch.float8_e4m3fn)
        return w, torch.randint(
            118, 122, (n // 32, k // 32), generator=gr, device=dev
        ).to(torch.uint8)

    woa, woa_s = fp8(2048, 4096)
    wob, wob_s = fp8(HIDDEN, 2048)
    sink = torch.randn(heads, generator=gr, device=dev)
    if args.shuffled:
        from aiter.ops.shuffle import shuffle_weight

        shuf = lambda t: shuffle_weight(t, (16, 16))  # noqa: E731
    else:
        shuf = lambda t: t  # noqa: E731
    k1w = K1Weights(
        hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base, norm=norm,
        wqkv=shuf(a["wqkv"]), wqkv_s=a["wqkv_s"], q_norm=a["q_norm"], kv_norm=a["kv_norm"],
        wqb=shuf(a["wqb"]), wqb_s=a["wqb_s"], cos_sin=cos_sin,
    )  # fmt: skip
    k2w = K2Weights(
        woa=woa, woa_s=woa_s, wob=shuf(wob), wob_s=wob_s, sink=sink, cos_sin=cos_sin
    )
    L = fake_layer(k1w, k2w, args.shuffled)
    with build_lock():
        run = K1Runner(dev, v3=True)
    k1 = AsmK1(run, chain=args.chain)
    w = k1.weights(L)
    # frag copies built from the row-major weights, whatever SGLang's layout
    from sgl_amd_mono.deepseek_v41.asm.kernels import gemv8

    for name, cfg, (wt, ws) in (
        ("wqkv", gemv8.WQKV32N if v3 else gemv8.WQKV16, (a["wqkv"], a["wqkv_s"])),
        ("wq_b", gemv8.WQ_B, (a["wqb"], a["wqb_s"])),
        ("wo_b", gemv8.WO_B1, (wob, wob_s)),
    ):
        want = gemv8.frag_w(wt.view(torch.uint8), ws, cfg)
        assert torch.equal(w[name][0], want[0]) and torch.equal(w[name][1], want[1]), (
            name
        )
    w_sh, w_su = prepare_mxfp8_native_weight(
        a["wqkv"], torch.exp2(a["wqkv_s"].float() - 127), (32, 32)
    )
    wqkv64, wqb64 = deq_w(a["wqkv"], a["wqkv_s"]), deq_w(a["wqb"], a["wqb_s"])
    stride = math.ceil(BLOCK * RECORD / 576) * 576
    edges = torch.tensor([BLOCK - 1, BLOCK, PAGES * BLOCK - 1], device=dev)
    positions = cos_sin.shape[0]

    def inputs(M, trial):
        _, res, post, comb, pre = seam_inputs(M, tp, 0, dev)
        gi = torch.Generator(device=dev).manual_seed(77 + M + 1000 * trial)
        res = (res * (torch.randn(HIDDEN, generator=gi, device=dev) * 1.5).exp()).to(
            res.dtype
        )
        x = (torch.randn(M, HIDDEN, generator=gi, device=dev) * 0.1).to(torch.bfloat16)
        pos = torch.randint(0, positions, (M,), generator=gi, device=dev)
        pos[: min(M, 2)] = torch.tensor([positions - 1, 65536], device=dev)[: min(M, 2)]
        perm = torch.randperm(PAGES * BLOCK, generator=gi, device=dev)
        slot = torch.cat([edges, perm[~torch.isin(perm, edges)]])[:M].clone()
        slot[-1] = -1  # a pad row: no record
        return res, x, post, comb, pre, pos, slot

    fails = []
    widths = [int(v) for v in args.widths.split(",")]
    for M in widths:
        assert 1 <= M <= MAX_M
        if "attend" in stages:
            fails += check_attend(k1, w, M, args, sink, cos_sin, dev, rank)
        if "back" in stages:
            fails += check_back(k1, w, M, args, dev, rank, tp, cpu)
        if "call" in stages:
            back64 = (deq_w(woa, woa_s).view(2, 1024, 4096), deq_w(wob, wob_s))
            for kind in args.calls.split(","):
                fails += check_call(
                    k1, w, M, args, inputs, sink, cos_sin, back64, dev, rank, tp, cpu,
                    kind,
                )  # fmt: skip
        for vname in args.variants.split(",") if "front" in stages else ():
            fold, index = vname in ("fold", "index"), vname == "index"
            static = [t.clone() for t in inputs(M, 0)]
            res, x, post, comb, pre, pos, slot = static
            swa = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)

            def launch():
                return k1.front(
                    w, x if fold else None, res, post if fold else None,
                    comb if fold else None, pre, pos, slot, swa, BLOCK, index=index,
                )  # fmt: skip

            launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = launch()
            rows_g = k1.index_rows if index else ()
            st = dict(replay=0, outside=0, x1=0, res=0, res_err=0.0, res_sgl=0.0, mix=0.0,
                      kv=1.0, ss=0.0, qx=0.0, q=1.0, w_nope=0, w_steps=0, w_rope=0, w_scale=0,
                      w_dec=0.0, rec_asm=0.0, rec_sgl=0.0, rec_cos=1.0, q_cos=1.0, q_rel=0.0,
                      q_asm=0.0, q_sgl=0.0, rstd=0.0, xq=0, w_sdiff=0, w_rel=0.0,
                      w_own=0.0, w_asm=0.0, i_own=0, n_asm=0.0, n_sgl=0.0,
                      r_asm=0.0, r_sgl=0.0, n_cos=1.0, r_cos=1.0, x1_asm=0.0, x1_sgl=0.0,
                      mix_use=0.0, xq_rows=0)  # fmt: skip
            for trial in range(args.trials):
                for t, v in zip(static, inputs(M, trial + 1)):
                    t.copy_(v)
                pre_in = pre.clone()
                swa.fill_(255)
                q, r, post_a, comb_a, pre_a = launch()
                torch.cuda.synchronize()
                valid = slot >= 0
                vs = slot[valid]
                eager = [t.clone() for t in (q, r, post_a, comb_a, pre_a, swa)]
                rows_e = [t.clone() for t in k1.index_rows] if index else []
                for t in (*outs, *rows_g):
                    t.fill_(float("nan"))
                swa.fill_(255)
                graph.replay()
                torch.cuda.synchronize()
                st["replay"] += sum(
                    int((u.view(torch.uint8) != v.view(torch.uint8)).sum())
                    for u, v in zip(eager + rows_e, [*outs, swa, *rows_g])
                )
                q, r, post_a, comb_a, pre_a, swa_e = eager
                st["outside"] += int((swa_e[~record_mask(swa_e, vs)] != 255).sum())

                # SGLang's chain: boundary, norm (+ the mixes), wqkv_a, the SWA writer
                ro, y, co = hc_boundary_fused_deferred(
                    x if fold else None, res, post if fold else None, comb if fold else None,
                    pre, hc_fn, hc_scale, hc_base, HC, 20, RMS_EPS, HC_EPS,
                )  # fmt: skip
                ro = res if ro is None else ro
                xq, normed_s = rmsnorm_with_sinkhorn(
                    y, norm, RMS_EPS, co, emit_fp8=True
                )
                pre_r, post_r, comb_r = co.tensors()
                qkv = mxfp8_native_blockscaled_linear(
                    xq.q[:M], w_sh, w_su, input_scale=xq.scale[:M]
                )
                ref = torch.full_like(swa_e, 255)
                fused_k_norm_rope_flashmla(
                    qkv[:, Q_RANK:].contiguous(), a["kv_norm"], RMS_EPS, freqs, pos,
                    slot.int(), ref, BLOCK,
                )  # fmt: skip

                # seam: residual, collapse, mixes
                if fold:
                    ref64, big = hc_post_fp64(x, res, post, comb)
                    st["res_err"] = max(st["res_err"], term_ulps(r, ref64, big))
                    st["res_sgl"] = max(st["res_sgl"], term_ulps(ro, ref64, big))
                    st["res"] += int(
                        (r.view(torch.int16) != ro.view(torch.int16)).sum()
                    )
                else:  # the identity post-mix: equal values (a -0 may come back +0)
                    st["res"] += int((r.float() != res.float()).sum())
                # the collapse (v3 index: x1's slot holds y' = bf16(bf16(collapse) · attn_norm.w),
                # SGLang's analog from its collapse) against fp64 on the pre mix both chains read
                # (the input ``pre``, the previous launch's; the seam's mixes are outputs)
                res64 = ref64 if fold else res.double()
                if index and v3:
                    got_s, w_y = (y.float() * norm.float()).bfloat16(), norm
                else:
                    got_s, w_y = y, None
                    st["x1"] += int(
                        (run.s["x1"][:M].view(torch.int16) != y.view(torch.int16)).sum()
                    )
                st["x1_asm"] = max(
                    st["x1_asm"], collapse_ulps(run.s["x1"][:M], pre_in, res64, w_y)
                )
                st["x1_sgl"] = max(
                    st["x1_sgl"], collapse_ulps(got_s, pre_in, res64, w_y)
                )
                for k1_t, ref_t in ((pre_a, pre_r), (post_a, post_r), (comb_a, comb_r)):
                    d = (k1_t.reshape(ref_t.shape) - ref_t).abs()
                    st["mix"] = max(st["mix"], d.max().item())
                    st["mix_use"] = max(st["mix_use"], (
                        d / (ASM_MIX_ATOL + ASM_MIX_RTOL * ref_t.abs())
                    ).max().item())  # fmt: skip

                # fp64 references: the attention norm without activation quantization
                x1 = y.double()
                rstd = torch.rsqrt(x1.pow(2).mean(1) + RMS_EPS)
                if (
                    v3
                ):  # the mixes waves' rstd, which the kv wave and the attention apply
                    st["rstd"] = max(st["rstd"], rel(run.s["rstd"][:M], rstd))
                qkv_e = (x1 * rstd[:, None] * norm.double()) @ wqkv64.T
                ql = qkv_e[:, :Q_RANK]
                qr_e = (
                    ql
                    * torch.rsqrt(ql.pow(2).mean(1, keepdim=True) + RMS_EPS)
                    * a["q_norm"].double()
                )
                q_e = rope64((qr_e @ wqb64.T).view(M, heads, HEAD), cos_sin, pos, NOPE)
                kv = qkv_e[:, Q_RANK:]
                kv_e = (
                    kv
                    * torch.rsqrt(kv.pow(2).mean(1, keepdim=True) + RMS_EPS)
                    * a["kv_norm"].double()
                )
                kv_e = rope64(kv_e[:, None, :], cos_sin, pos, NOPE)[:, 0]

                # wqkv on its own MXFP8 x: v3 its kv rows (bf16) and the q latent's 40 sums of
                # squares; v1 every row (bf16), its x being the fin's MXFP8 of the normed input
                yx, ysc = unfrag_x(run.s["yq"], run.s["ys"], HIDDEN)
                qkv_o = deq_x(yx[:M], ysc[:M]) @ wqkv64.T
                if v3:
                    kv_got = run.s["qr_kv"][:M, Q_RANK:]
                    st["kv"] = min(st["kv"], cosine(kv_got, qkv_o[:, Q_RANK:]))
                    st["ss"] = max(st["ss"], rel(
                        run.s["ss"].view(MAX_M, 40)[:M].sum(1),
                        qkv_o[:, :Q_RANK].bfloat16().double().pow(2).sum(1),
                    ))  # fmt: skip
                else:
                    st["kv"] = min(st["kv"], cosine(run.s["qr_kv"][:M], qkv_o))
                    # bit for bit on the rows whose collapse is SGLang's (a row's rstd reads
                    # every value of the row); the other rows' collapse is checked above
                    eq = (run.s["x1"][:M].view(torch.int16) == y.view(torch.int16)).all(
                        1
                    )
                    st["xq_rows"] += int(eq.sum())
                    st["xq"] += int(
                        (
                            yx[:M].view(torch.uint8)[eq]
                            != xq.q[:M].view(torch.uint8)[eq]
                        ).sum()
                    )
                    st["xq"] += int(
                        (
                            ysc[:M].view(torch.uint8)[eq]
                            != xq.scale[:M].view(torch.uint8)[eq]
                        ).sum()
                    )
                # wq_b + q RoPE on its own MXFP8 x
                qx, qxs = unfrag_x(run.s["qrq"], run.s["qrs"], Q_RANK)
                q_o = rope64(
                    (deq_x(qx[:M], qxs[:M]) @ wqb64.T).view(M, heads, HEAD),
                    cos_sin,
                    pos,
                    NOPE,
                )
                st["qx"] = max(st["qx"], rel(q, q_o))
                st["q"] = min(st["q"], cosine(q, q_o))

                # the writer rules: SGLang's writer on the front's own kv rows (v3: x rstd, bf16)
                kv_in = run.s["qr_kv"][:M, Q_RANK:]
                if v3:
                    kv_in = (kv_in.float() * run.s["rstd"][:M, None]).bfloat16()
                ref_w = torch.full_like(swa_e, 255)
                fused_k_norm_rope_flashmla(
                    kv_in.contiguous(),
                    a["kv_norm"],
                    RMS_EPS,
                    freqs,
                    pos,
                    slot.int(),
                    ref_w,
                    BLOCK,
                )
                got, want_w, want = (
                    records(swa_e, vs),
                    records(ref_w, vs),
                    records(ref, vs),
                )
                st["w_nope"] += int((got[:, :NOPE] != want_w[:, :NOPE]).sum())
                # a block's scale moves by one where the bf16 of its max crosses a power of two
                gs, ws_ = got[:, 576:583], want_w[:, 576:583]
                st["w_scale"] += int((gs != ws_).sum())
                st["w_sdiff"] = max(
                    st["w_sdiff"], int((gs.int() - ws_.int()).abs().max())
                )
                same = (gs == ws_).repeat_interleave(64, 1)
                st["w_steps"] = max(
                    st["w_steps"],
                    e4m3_steps(got[:, :NOPE][same], want_w[:, :NOPE][same]),
                )
                gr_, wr_ = (
                    t[:, NOPE:576].contiguous().view(torch.bfloat16)
                    for t in (got, want_w)
                )
                st["w_rope"] += int((gr_ != wr_).sum())
                dg, dw = decode(got), decode(want_w)
                st["w_dec"] = max(
                    st["w_dec"], ((dg - dw).abs().max() / dw.abs().max()).item()
                )
                # SGLang's writer's own rounding on that input: fp64 norm and RoPE of kv_in
                kn = kv_in.double()
                kn = kn * torch.rsqrt(kn.pow(2).mean(1, keepdim=True) + RMS_EPS)
                kn = rope64((kn * a["kv_norm"].double())[:, None], cos_sin, pos, NOPE)[
                    :, 0
                ]
                st["w_rel"] = max(st["w_rel"], rel(dg, dw))
                st["w_own"] = max(st["w_own"], rel(dw, kn[valid]))
                st["w_asm"] = max(st["w_asm"], rel(dg, kn[valid]))
                # both chains' records against fp64
                ke = kv_e[valid]
                st["rec_asm"] = max(st["rec_asm"], rel(dg, ke))
                st["rec_sgl"] = max(st["rec_sgl"], rel(decode(want), ke))
                st["rec_cos"] = min(st["rec_cos"], cosine(dg, decode(want)))

                # v3: q with q_norm's rrms (the attention folds it into its scores)
                q_asm = q.double()
                if v3:
                    rd = run.s["rstd"][:M].double()
                    ss = run.s["ss"].view(MAX_M, 40)[:M].double().sum(1)
                    q_asm = (
                        q_asm
                        * (rd * torch.rsqrt(rd * rd * ss / Q_RANK + RMS_EPS))[
                            :, None, None
                        ]
                    )
                q_sgl = query_ref(qkv, a, cos_sin, pos)
                st["q_cos"] = min(st["q_cos"], cosine(q_asm, q_sgl))
                st["q_rel"] = max(st["q_rel"], rel(q_asm, q_sgl))
                st["q_asm"] = max(st["q_asm"], rel(q_asm, q_e))
                st["q_sgl"] = max(st["q_sgl"], rel(q_sgl, q_e))

                if index:
                    normed_a, qr_a = rows_e
                    # the index rows on the scratch they were made from
                    if v3:
                        rd = run.s["rstd"][:M]
                        own_n = (run.s["x1"][:M].float() * rd[:, None]).bfloat16()
                        qx_c, qx_s = unfrag_x(run.s["qrq"], run.s["qrs"], Q_RANK)
                        sq = run.s["ss"].view(MAX_M, 40)[:M].sum(1)
                        rr = rd / torch.sqrt(rd * rd * (sq / Q_RANK) + RMS_EPS)
                        own_q = deq_x(qx_c[:M], qx_s[:M]).float() * rr[:, None]
                    else:
                        own_n, ql_a = normed_a, run.s["qr_kv"][:M, :Q_RANK].float()
                        own_q = (
                            ql_a
                            * torch.rsqrt(ql_a.pow(2).mean(1, keepdim=True) + RMS_EPS)
                            * a["q_norm"].float()
                        )
                    st["i_own"] = max(
                        st["i_own"],
                        bf16_ulps(normed_a, own_n),
                        bf16_ulps(qr_a, own_q.bfloat16()),
                    )
                    # against SGLang's (its norm's output; the q norm of its own q latent)
                    # and both against fp64
                    ql_s = qkv[:, :Q_RANK].float()
                    qr_s = (
                        ql_s
                        * torch.rsqrt(ql_s.pow(2).mean(1, keepdim=True) + RMS_EPS)
                        * a["q_norm"].float()
                    ).bfloat16()
                    normed_e = x1 * rstd[:, None] * norm.double()
                    st["n_asm"] = max(st["n_asm"], rel(normed_a, normed_e))
                    st["n_sgl"] = max(st["n_sgl"], rel(normed_s, normed_e))
                    st["r_asm"] = max(st["r_asm"], rel(qr_a, qr_e))
                    st["r_sgl"] = max(st["r_sgl"], rel(qr_s, qr_e))
                    st["n_cos"] = min(st["n_cos"], cosine(normed_a, normed_s))
                    st["r_cos"] = min(st["r_cos"], cosine(qr_a, qr_s))
            del graph
            n_codes = args.trials * int((slot >= 0).sum()) * NOPE
            yp = index and v3  # x1's slot holds y'
            what = "y'" if yp else "collapse"
            checks = {
                "replay": st["replay"] == 0,
                "no write outside the records": st["outside"] == 0,
                # as the residual: against fp64, at most SGLang's error + 0.5 ulp
                f"{what} vs fp64": st["x1_asm"] <= st["x1_sgl"] + 0.5,
                # check_ffn.py's residual bound: SGLang's error + 0.5 ulp, <= 0.01 % differ
                "residual": (
                    st["res_err"] <= st["res_sgl"] + 0.5
                    and st["res"] <= 1e-4 * args.trials * M * HC * HIDDEN
                )
                if fold
                else st["res"] == 0,
                "mixes": st["mix_use"] <= 1,
                "rstd": st["rstd"] <= 1e-6,
                # v1: the fin's MXFP8 of the normed input is SGLang's norm's (same collapse)
                "normed MXFP8": st["xq"] == 0,
                "wqkv": st["kv"] >= STAGE_COS and st["ss"] <= STAGE_REL,
                "wq_b": st["q"] >= STAGE_COS and st["qx"] <= STAGE_REL,
                "writer": st["w_sdiff"] <= 1
                and st["w_steps"] <= 1
                and st["w_asm"] <= ERR_RATIO * st["w_own"],
                "records vs fp64": st["rec_asm"] <= ERR_RATIO * st["rec_sgl"],
                "q vs fp64": st["q_asm"] <= ERR_RATIO * st["q_sgl"],
            }
            index_note = ""
            if index:
                ratio_i = INDEX_RATIO_V3 if v3 else ERR_RATIO
                checks.update({
                    "index rows on their own inputs": st["i_own"] <= 1,
                    "normed vs fp64": st["n_asm"] <= ratio_i * st["n_sgl"],
                    "qr vs fp64": st["r_asm"] <= ratio_i * st["r_sgl"],
                })  # fmt: skip
                index_note = (
                    f" | index rows: own inputs max {st['i_own']} ulp; normed vs fp64 asm "
                    f"{st['n_asm']:.2e}, SGLang {st['n_sgl']:.2e}, "
                    f"cos(asm, SGLang) {st['n_cos']:.6f}; qr vs fp64 asm {st['r_asm']:.2e}, "
                    f"SGLang {st['r_sgl']:.2e}, cos(asm, SGLang) {st['r_cos']:.6f} (bound "
                    f"{ratio_i}x)"
                )
            bad = [k for k, v in checks.items() if not v]
            xq_note = "" if v3 else (
                f"normed MXFP8 diff {st['xq']} on the {st['xq_rows']} rows with SGLang's "
                "collapse, "
            )  # fmt: skip
            x1_note = "" if yp else f", {st['x1']} values differ from SGLang's"
            if rank == 0:
                print(
                    f"M={M:3d} {args.chain} {vname:7s} {'PASS' if not bad else 'FAIL ' + ','.join(bad)}"
                    f"{' (shuffled)' if args.shuffled else ''} | seam: {what} vs fp64 on the input "
                    f"pre asm {st['x1_asm']:.2f} / SGLang {st['x1_sgl']:.2f} ulp{x1_note}, "
                    f"{xq_note}"
                    f"residual diff {st['res']} ({st['res_err']:.2f} / SGLang {st['res_sgl']:.2f} ulp), "
                    f"mixes max|d| {st['mix']:.2e} ({st['mix_use']:.2f} of K2's asm bound), "
                    f"rstd rel {st['rstd']:.1e} | "
                    f"wqkv: kv cos {st['kv']:.7f}, ss rel {st['ss']:.1e} | wq_b: cos {st['q']:.7f} "
                    f"rel {st['qx']:.1e} | writer vs SGLang's on the same kv: NoPE codes differ "
                    f"{st['w_nope']}/{n_codes} ({100 * st['w_nope'] / n_codes:.2f} %, max {st['w_steps']} "
                    f"step in equal-scale blocks), RoPE values differ {st['w_rope']}, scales differ "
                    f"{st['w_scale']}/{n_codes // 64} (max {st['w_sdiff']}), decoded max|d|/max "
                    f"{st['w_dec']:.1e}, rel {st['w_rel']:.2e}; vs fp64 asm {st['w_asm']:.2e}, "
                    f"SGLang {st['w_own']:.2e} | records vs fp64: asm {st['rec_asm']:.2e}, SGLang "
                    f"{st['rec_sgl']:.2e}, cos(asm, SGLang) {st['rec_cos']:.6f} | q: cos(asm, SGLang) "
                    f"{st['q_cos']:.6f} rel {st['q_rel']:.2e}; vs fp64 asm {st['q_asm']:.2e}, SGLang "
                    f"{st['q_sgl']:.2e} | replay diff {st['replay']}, outside writes {st['outside']}"
                    f"{index_note}",
                    flush=True,
                )
            fails += [f"M={M} {vname} {b}" for b in bad]
    finish(fails, dev)


if __name__ == "__main__":
    main()
