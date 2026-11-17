"""Check the asm K2 (``asm/k2.py``: vllm#65421's FFN half, then the combine with
all-reduce #2) on SGLang's loaded weight layout, TP4 on gfx950:

    torchrun --nproc-per-node 4 check_asm_k2.py [--chain v3|v1] [--wire bf16|fp8] [--parts]

Against check_ffn.py's references (SGLang's gfx950 boundary kernel and norm, the golden
MoE). The asm chains round at other points than SGLang's ops, so the MoE is bounded on
the kernel's own picks against a golden with the chain's own rules:
- the residual: SGLang's error against an fp64 hc_post + 0.5 ulp, at most 0.01 % of the
  elements differ (check_ffn.py's bound);
- the pre mix and the normed row (v3: y' scaled by rstd): check_ffn.py's bounds; the post
  and comb mixes within ASM_MIX_RTOL;
- the MoE on the kernel's picks against the golden with the chain's rules (v3: MXFP8 of y'
  before rstd, the gate / up sums scaled by rstd; v1: MXFP8 of the normed row) within
  MOE_REL, every token's cosine above TOKEN_COS;
- the routing: the kernel's picks against the golden's on the kernel's row, and the
  golden's picks on the kernel's row against those on SGLang's, for at least ROUTING_SAME
  of the tokens (a top-6 near tie may flip);
- ``--parts``: this rank's routed rows per pick and its shared expert against goldens with
  the chain's rules and no bf16 rounding of the SwiGLU output, within PART_REL;
- no timed-out wait, every rank's output equal, and every width in one graph replayed bit
  for bit against eager.
The MoE against check_ffn.py's golden (SGLang's rounding points) is printed, not bounded:
check_asm_layer.py and the model-level accuracy gates bound its effect.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from check_ffn import (
    HC,
    HIDDEN,
    LIMIT,
    MIX_RTOL,
    TOPK,
    bf,
    bf16_ulps,
    cos,
    deq_fp4,
    deq_fp8_block,
    finish,
    golden_partial,
    hc_post_fp64,
    make_weights,
    mono_weights,
    mxfp8_qdq,
    reduced,
    rel,
    routing,
    seam_inputs,
    sglang_ffn_seam,
    term_ulps,
)

from sgl_amd_mono.deepseek_v41.spec import NUMERICS

WIDTHS = (17, 24, 32, 40, 48, 64, 96, 128, 192, 256)
# the asm mixes sum in another order with hardware exp / rcp; comb's grows with the rows
# (up to 6.8e-4 at 256 rows seen)
ASM_MIX_RTOL = 2e-3
# the fp8 wire's MXFP8 of the partial sums adds about 0.02 (bf16 wire: about 0.017 seen)
MOE_REL = {"bf16": 0.025, "fp8": 0.05}
TOKEN_COS = {"bf16": 0.999, "fp8": 0.998}
PART_REL = 0.005
ROUTING_SAME = 0.9


def same_sets(a, b):
    return (a.sort(1).values == b.sort(1).values).all(1).float().mean().item()


def max_rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).abs() / b.abs().clamp_min(1e-6)).max().item()


def all_reduced(part):
    g = part.float()
    dist.all_reduce(g)
    return g.to(torch.bfloat16)


def hot_bias(w, hot):
    """Every token's top 6 from ``hot`` experts (+4 router bias, the same on every rank):
    many rows an expert, where check_ffn.py's router spreads the rows almost uniformly."""
    g = torch.Generator(device=w["bias"].device).manual_seed(777)
    idx = torch.randperm(w["bias"].shape[0], generator=g, device=w["bias"].device)[:hot]
    b = w["bias"].float()
    b[idx] += 4.0
    w["bias"] = b.to(w["bias"].dtype)


def golden_picks(xq, scale, ids, wts, w, inter, padded, parts=False, mid_bf16=True):
    """check_ffn.golden_partial on given picks (the kernel's: no routing flips) and an
    MXFP8-rounded input ``xq`` for the routed and the shared expert, the gate / up sums of
    both scaled by the row factor ``scale`` (v3) or not (None). ``parts``: the per-pick
    rows [M, TOPK, H] and the shared expert's output instead of their sum."""
    M = xq.shape[0]
    r = None if scale is None else scale.float()[:, None]
    per_k = torch.zeros(M, TOPK, HIDDEN, device=xq.device)
    for e in ids.unique().tolist():
        rows, ks = (ids == e).nonzero(as_tuple=True)
        h = xq[rows] @ deq_fp4(w["w13_raw"][e], w["s13_raw"][e]).T
        if r is not None:
            h = h * r[rows]
        gate, up = h[:, :padded], h[:, padded:]
        mid = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
        if mid_bf16:  # check_ffn.py's rule
            mid = bf(mid)
        y = mxfp8_qdq(mid, "aiter") @ deq_fp4(w["w2_raw"][e], w["s2_raw"][e]).T
        per_k[rows, ks] = bf(y * wts[rows, ks, None])
    hs = xq @ deq_fp8_block(w["sgu"], w["sgu_s"]).T
    if r is not None:
        hs = hs * r
    smid = bf(
        F.silu(bf(hs[:, :inter].clamp(max=LIMIT)))
        * bf(hs[:, inter:].clamp(-LIMIT, LIMIT))
    )
    shared = bf(mxfp8_qdq(smid, "vllm") @ deq_fp8_block(w["sw2"], w["sw2_s"]).T)
    if parts:
        return per_k, shared
    routed = torch.zeros(M, HIDDEN, device=xq.device)
    for k in range(TOPK):
        routed = bf(routed + per_k[:, k])
    return bf(routed + shared)


def check_width(k2, wt, w, inter, padded, M, args, rank, tp, dev):
    """One width: (the record, the failed checks, the replay step's inputs)."""
    part, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
    x = reduced(part)  # the attention all-reduce's output, as SGLang's seam gets it
    out, res_o, post_o, comb_o, pre_o = k2(wt, x, res, post, comb, pre, args.wire)
    torch.cuda.synchronize()
    res2, normed_ref, post2, comb2, pre2 = sglang_ffn_seam(
        part, res, post, comb, pre, w
    )
    s, v3 = k2.run.s, k2.chain == "v3"
    # v3: the seam writes y' = bf16(x1 w) without the row's rstd and p3 writes rstd; v1:
    # the fin launch writes the normed row
    xin = s["y"][:M]
    rstd = s["rstd"][:M] if v3 else None
    normed = (xin.float() * rstd[:, None]).to(torch.bfloat16) if v3 else xin.clone()
    ids_k, wts_k = s["ids"][:M].long(), s["wts"][:M].float()
    ids, gpart = golden_partial(normed, w, inter, padded)
    gold = all_reduced(gpart)
    ids_ref, _ = routing(normed_ref, w)
    own = all_reduced(
        golden_picks(mxfp8_qdq(xin, "aiter"), rstd, ids_k, wts_k, w, inter, padded)
    )
    tok_own = F.cosine_similarity(out.float(), own.float(), dim=1)
    tok = F.cosine_similarity(out.float(), gold.float(), dim=1)
    _, res_share = bf16_ulps(res_o, res2)
    ref64, big = hc_post_fp64(x, res, post, comb)
    err, err_sgl = term_ulps(res_o, ref64, big), term_ulps(res2, ref64, big)
    row = dict(
        M=M, chain=k2.chain, wire=args.wire, hot=args.hot, res_err_ulp=err, res_err_sglang_ulp=err_sgl,
        res_differ_pct=100 * res_share, normed_cos=cos(normed, normed_ref),
        normed_rel=rel(normed, normed_ref), post_max_rel=max_rel(post_o.view(M, HC), post2),
        comb_max_rel=max_rel(comb_o, comb2), pre_max_rel=max_rel(pre_o, pre2),
        moe_own_rel=rel(out, own), moe_own_cos=cos(out, own),
        min_token_own=tok_own.min().item(), picks_same=same_sets(ids_k, ids),
        routing_same=same_sets(ids, ids_ref), moe_sglang_rules_rel=rel(out, gold),
        moe_sglang_rules_cos=cos(out, gold), min_token_sglang_rules=tok.min().item(),
        timeouts=int(k2.errors()),
    )  # fmt: skip
    checks = {
        "residual err <= SGLang's + 0.5 ulp, <= 0.01% differ": err <= err_sgl + 0.5
        and res_share <= 1e-4,
        "pre": torch.allclose(pre_o, pre2, rtol=MIX_RTOL, atol=1e-5),
        "post": torch.allclose(post_o.view(M, HC), post2, rtol=ASM_MIX_RTOL, atol=1e-5),
        "comb": torch.allclose(comb_o, comb2, rtol=ASM_MIX_RTOL, atol=1e-5),
        "normed": row["normed_cos"] > NUMERICS.bound("ffn.normed.cosine")
        and row["normed_rel"] < NUMERICS.bound("ffn.normed.rel"),
        "moe, chain's rules": row["moe_own_rel"] < MOE_REL[args.wire]
        and row["min_token_own"] > TOKEN_COS[args.wire],
        "kernel picks = golden picks": row["picks_same"] >= ROUTING_SAME,
        "routing same": row["routing_same"] >= ROUTING_SAME,
        "timeouts 0": row["timeouts"] == 0,
    }
    if args.parts:
        P = M * TOPK
        if v3:
            y_k = s["y2"].view(torch.bfloat16).view(-1)[: P * HIDDEN].view(P, HIDDEN)
        else:
            y_k = s["y2"][:P]
        pk, sh = golden_picks(
            mxfp8_qdq(xin, "aiter"), rstd, ids_k, wts_k, w, inter, padded,
            parts=True, mid_bf16=False,
        )  # fmt: skip
        row["routed_part_rel"] = rel(y_k.float().view(M, TOPK, HIDDEN), pk)
        row["shared_part_rel"] = rel(s["shared"][:M], sh)
        checks["parts, chain's rules"] = (
            row["routed_part_rel"] < PART_REL and row["shared_part_rel"] < PART_REL
        )
    ref0 = out.clone()
    dist.broadcast(ref0, 0)
    checks["ranks agree"] = torch.equal(ref0, out)
    bad = [k for k, v in checks.items() if not v]
    row["pass"], row["failed"] = not bad, bad
    if rank == 0:
        parts = (
            f", parts rel routed {row['routed_part_rel']:.4f} shared "
            f"{row['shared_part_rel']:.4f}"
            if args.parts
            else ""
        )
        print(
            f"M={M:3d} {k2.chain} {args.wire}{f' hot {args.hot}' if args.hot else ''} "
            f"{'PASS' if not bad else 'FAIL ' + ','.join(bad)}: "
            f"residual vs fp64 {err:.2f} / SGLang {err_sgl:.2f} ulp, {row['res_differ_pct']:.4f}% "
            f"differ; normed rel {row['normed_rel']:.4f}; mixes max rel post "
            f"{row['post_max_rel']:.1e} comb {row['comb_max_rel']:.1e} pre "
            f"{row['pre_max_rel']:.1e}; moe (chain's rules) rel {row['moe_own_rel']:.4f} "
            f"min-token {row['min_token_own']:.5f}; picks {row['picks_same']:.3f}, routing "
            f"{row['routing_same']:.3f}{parts}; moe (SGLang's rules, not bounded) rel "
            f"{row['moe_sglang_rules_rel']:.4f} min-token {row['min_token_sglang_rules']:.4f}; "
            f"timeouts {row['timeouts']}",
            flush=True,
        )  # fmt: skip
    return row, [f"M={M} {k}" for k in bad], (M, res, post, comb, pre, x, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=("v3", "v1"), default="v3")
    ap.add_argument("--wire", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--widths", default=",".join(map(str, WIDTHS)))
    ap.add_argument("--replays", type=int, default=500)
    ap.add_argument("--parts", action="store_true", help="bound this rank's parts too")
    ap.add_argument(
        "--hot", type=int, default=0, help="route every token to these many experts"
    )
    ap.add_argument(
        "--deadline", type=float, default=10.0, help="the waits' deadline (s)"
    )
    ap.add_argument("--out", default="", help="write the records here (JSON, rank 0)")
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert tp == 4, "the asm K2 is written for TP4"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.asm.guard import AsmGuard
    from sgl_amd_mono.deepseek_v41.asm.k2 import AsmK2

    w, inter, padded = make_weights(tp, rank, dev)
    if args.hot:
        hot_bias(w, args.hot)
    wt = AsmK2.weights(mono_weights(w))
    guard = AsmGuard(rank, cpu, tp, deadline_s=args.deadline)
    k2 = AsmK2(rank, cpu, guard, chain=args.chain, halt=False)
    rows, fails, steps = [], [], []
    for M in (int(v) for v in args.widths.split(",")):
        row, bad, step = check_width(k2, wt, w, inter, padded, M, args, rank, tp, dev)
        rows.append(row)
        fails += bad
        steps.append(step)

    # every width in one graph, back to back, bit for bit against eager
    def run_all():
        return [
            k2(wt, x, res, post, comb, pre, args.wire)[0]
            for _, res, post, comb, pre, x, _ in steps
        ]

    ref = [s[6].clone() for s in steps]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run_all()
    bad_replays = 0
    for i in range(args.replays):
        graph.replay()
        if i % 50 == 49 or i == args.replays - 1:
            torch.cuda.synchronize()
            bad_replays += sum(not torch.equal(a, b) for a, b in zip(outs, ref))
    timeouts, words = k2.errors(), guard.errors()
    if rank == 0:
        print(
            f"replay check: {args.replays} replays of widths {[s[0] for s in steps]} in one "
            f"graph, outputs sampled every 50 replays: {bad_replays} mismatching checks; "
            f"timeouts {timeouts}; guard words {words or 'clean'}",
            flush=True,
        )
    if bad_replays or timeouts or words:
        fails.append("replay")
    if args.out and rank == 0:
        with open(args.out, "w") as f:
            json.dump(
                dict(chain=args.chain, wire=args.wire, hot=args.hot, rows=rows, replays=args.replays,
                     replay_mismatches=bad_replays, timeouts=timeouts, guard=words,
                     fails=fails),
                f, indent=1,
            )  # fmt: skip
    finish(fails, dev)


if __name__ == "__main__":
    main()
