# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/k2_runner.py (changed: import paths)
"""K2, the FFN half of a decoder layer (from the attention all-reduce's output to the
MoE all-reduce's input), as one callable.

    run = K2Runner("chain", v3=True)
    wt = run.weights(raw_from_layer(layer))
    part, r_new, post, comb, pre = run(wt, x, residual, post_mix, res_mix, attn_pre)

``part`` = bf16(routed + shared), this rank's partial (vLLM all-reduces it).
- Kernels: k2.phases(layout) gives the bodies, each launched as its own
  kernel (the chain; P4 = the prep's sort CTA alone). The layout puts the
  shared down before the combine, which adds it ("part").
- v3 (the FFN front v3): no fin launch (seam2 yq="ug" writes MXFP8 + bf16 y'
  without rstd; p3 v3 applies rstd and runs fin's mixes on a 5th wave; stage 1
  scales g / u by rstd), prep + shared down and router + shared gate_up as one
  launch each (kernels/multi.py), stage 2 writes y in bf16 (``comb_in``'s y is
  bf16 then).
- Scratch is persistent per runner, sized for M = 256 and initialised once
  (partial's unwritten columns 0, xs pad 0, stage 2's hs pad 0x7F), so a call
  launches no fills. Layers run one after another on a stream, so they share
  it. Outputs are fresh per call.
"""

import struct

import torch

from sgl_amd_mono.deepseek_v41.asm.kernels import ffn_route as fr
from sgl_amd_mono.deepseek_v41.asm.kernels import (
    ffn_seam,
    k2,
    moe_combine,
    moe_down1,
    moe_prep,
    moe_ug1,
    multi,
)
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import Kernel

H, MAX_M, TOPK, I_REAL = 5120, 256, 6, 576
BF = torch.bfloat16
LAYOUT = "sdn_prep+cost+part+pst"


def f32bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def raw_from_layer(layer) -> dict:
    """A vLLM DeepseekV4DecoderLayer's FFN-half tensors under the capture's
    names (capture/capture_moe.py)."""
    ts = {
        n: t
        for n, t in [*layer.ffn.named_parameters(), *layer.ffn.named_buffers()]
        if t is not None and "routed_experts" not in n
    }
    for n in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        ts[f"seam.{n}"] = getattr(layer, n)
    ts["seam.ffn_norm.weight"] = layer.ffn_norm.weight
    for n, v in (
        ("rms_norm_eps", layer.rms_norm_eps),
        ("hc_eps", layer.hc_eps),
        ("hc_post_alpha", layer.hc_post_alpha),
        ("hc_sinkhorn_iters", layer.hc_sinkhorn_iters),
        ("norm_eps", layer.ffn_norm.variance_epsilon),
    ):
        ts[f"seam.{n}"] = torch.tensor(float(v), dtype=torch.float64)
    e = layer.ffn.experts.routed_experts
    for n in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        ts[f"routed.{n}"] = getattr(e, n)
    return ts


def _find(ws: dict, key: str) -> torch.Tensor:
    hits = [n for n in ws if key in n]
    assert len(hits) == 1, (key, hits)
    return ws[hits[0]]


class K2Runner:
    def __init__(
        self, mode: str = "chain", layout: str = LAYOUT, device=None, v3: bool = True
    ):
        assert mode == "chain", mode
        assert "part" in layout.split("+"), (
            "the runner returns part: the layout needs the adding combine"
        )
        self.mode, self.layout, self.v3 = mode, layout, v3
        self.block: dict
        self.merged: dict
        self.block, self.merged = (
            {},
            {},
        )  # block size per tag; first tag → (kernel, grid, block, tags)
        self.dev = torch.device(device or "cuda")
        self.phases = k2.phases(layout)
        self.ks = {
            tag: (Kernel(kk["name"], kk["src"], kk["ka"]), grid)
            for ph in self.phases
            for tag, kk, grid in ph
        }
        # P4: the prep's sort CTA alone (the seam's 2nd pass wrote xq / xs)
        self.ks["prep"] = (self.ks["prep"][0], 1)
        # the seam's main pass v2 (bitwise v1, faster)
        kk = ffn_seam.source_v2()
        self.ks["seam"] = (Kernel(kk["name"], kk["src"], kk["ka"]), "seam2")
        if v3:
            self._v3()
        self._scratch()

    def _v3(self):
        mk = lambda k: Kernel(k["name"], k["src"], k["ka"])
        src = {tag: k for ph in self.phases for tag, k, _ in ph}
        del self.ks["fin"]
        self.ks["seam"] = (
            mk(ffn_seam.source_v2(yq="ug", name="ffn_seam2_ug")),
            "seam2",
        )
        p3 = fr.source_p3(v3=True)
        self.ks["p3"] = (mk(p3), "m")
        self.block["p3"] = p3["block"]
        assert self.ks["ug"][0].name.startswith("moe_ug1_ceil"), self.ks["ug"][0].name
        self.ks["ug"] = (
            mk(moe_ug1.source(counts_vec=True, rstd=True)),
            self.ks["ug"][1],
        )
        assert (
            self.ks["dn"][0].name.startswith("moe_down1_table")
            and self.ks["comb"][0].name == "moe_combine_add"
        )
        self.ks["dn"] = (
            mk(moe_down1.source(split="table", counts_vec=True, y_bf16=True)),
            self.ks["dn"][1],
        )
        self.ks["comb"] = (
            mk(moe_combine.source(add=True, y_bf16=True)),
            self.ks["comb"][1],
        )
        g = self.ks["sdn"][1]
        assert (
            (g[0] if isinstance(g, tuple) else g) == fr.D_GRID
            and self.ks["rt"][1] == fr.R_GRID
            and self.ks["sgu"][1] == fr.S_GRID
        ), (g, self.ks["rt"][1], self.ks["sgu"][1])
        for first, k in (
            (
                "prep",
                multi.source(
                    "ffn_prep_sdn",
                    [("prep", src["prep"], 1), ("sdn", src["sdn"], fr.D_GRID)],
                ),
            ),
            (
                "rt",
                multi.source(
                    "ffn_rt_sgu",
                    [
                        ("rt", fr.source_router(w_agpr=True), fr.R_GRID),
                        ("sgu", fr.source_shared(tpr=2), fr.S_GRID),
                    ],
                ),
            ),
        ):
            self.merged[first] = (
                mk(k),
                k["grid"],
                k["block"],
                (first, "sdn" if first == "prep" else "sgu"),
            )

    def _scratch(self):
        d, P = self.dev, MAX_M * TOPK
        f = lambda *s, dt=torch.float32: torch.empty(*s, dtype=dt, device=d)
        self.s = dict(
            partial=torch.zeros(MAX_M, ffn_seam.SLICES, 32, device=d),
            x1=f(MAX_M, H, dt=BF),
            y=f(MAX_M, H, dt=BF),
            xq=f(MAX_M, H, dt=torch.uint8),
            xs=torch.zeros(MAX_M, 192, dtype=torch.uint8, device=d),
            rpart=f(MAX_M, fr.RS, fr.E),
            gpart=f(MAX_M, fr.SS, fr.NS),
            logits=f(MAX_M, fr.E),
            ids=f(MAX_M, TOPK, dt=torch.int32),
            wts=f(MAX_M, TOPK),
            hq=f(MAX_M, I_REAL, dt=torch.uint8),
            hs_sh=f(MAX_M, I_REAL // 32, dt=torch.uint8),
            shared=f(MAX_M, H, dt=BF),
            groups=f(P, 20, dt=torch.int32),
            counts=torch.zeros(moe_prep.COUNTS_DSPLIT, dtype=torch.int32, device=d),
            h=torch.zeros(P, I_REAL, dtype=torch.uint8, device=d),
            hs=torch.full((P, 32), 0x7F, dtype=torch.uint8, device=d),
            y2=f(P, H),
            rstd=torch.ones(MAX_M, device=d),
        )

    def weights(self, raw: dict) -> dict:
        """Kernel-ready weights of one layer from capture-named tensors."""
        g = lambda n: raw[f"seam.{n}"]
        assert int(g("hc_sinkhorn_iters").item()) == ffn_seam.SINKHORN
        u8 = lambda t: t.view(torch.uint8)
        return dict(
            fn=g("hc_ffn_fn").float().contiguous(),
            fn_hl=ffn_seam.fn_hl(g("hc_ffn_fn").float()),
            scale=g("hc_ffn_scale").float().contiguous(),
            base=g("hc_ffn_base").float().contiguous(),
            norm_w=g("ffn_norm.weight").to(BF).contiguous(),
            rms_eps=f32bits(g("rms_norm_eps").item()),
            hc_eps=f32bits(g("hc_eps").item()),
            post_alpha=f32bits(g("hc_post_alpha").item()),
            norm_eps=f32bits(g("norm_eps").item()),
            gate=_find(raw, "gate.weight").to(BF).contiguous(),
            bias=_find(raw, "e_score_correction_bias").float().contiguous(),
            sw=raw["shared_experts.gate_up_proj.weight"].contiguous(),
            ss=raw["shared_experts.gate_up_proj.weight_scale"].contiguous(),
            dw=raw["shared_experts.down_proj.weight"].contiguous(),
            ds=raw["shared_experts.down_proj.weight_scale"].contiguous(),
            w13=u8(raw["routed.w13_weight"]),
            s13=raw["routed.w13_weight_scale"],
            w2=u8(raw["routed.w2_weight"]),
            s2=raw["routed.w2_weight_scale"],
        )

    def outputs(self, residual: torch.Tensor) -> dict:
        m, d = residual.shape[0], residual.device
        return dict(
            r_out=torch.empty_like(residual),
            post=torch.empty(m, 4, 1, device=d),
            comb=torch.empty(m, 4, 4, device=d),
            pre=torch.empty(m, 4, device=d),
            part=torch.empty(m, H, dtype=BF, device=d),
        )

    def args(self, w: dict, ins: tuple, o: dict) -> dict:
        """Per-body kernargs (each body's own KArgs names)."""
        residual, x, post, comb, pre = ins
        m = x.shape[0]
        P = m * TOPK
        s = {
            n: t[:P] if n in ("groups", "h", "hs", "y2") else t[:m]
            for n, t in self.s.items()
            if n != "counts"
        }
        ids, wts = s["ids"].reshape(-1), s["wts"].reshape(-1)
        a = dict(
            seam=dict(
                residual=residual,
                x=x,
                post=post,
                comb=comb,
                pre=pre,
                r_out=o["r_out"],
                x1=s["x1"],
                partial=s["partial"],
                m=m,
                fn_hl=w["fn_hl"],
                tg=ffn_seam.tg_for(m),
            ),
            fin=dict(
                partial=s["partial"],
                x1=s["x1"],
                norm_w=w["norm_w"],
                scale=w["scale"],
                base=w["base"],
                post_out=o["post"],
                comb_out=o["comb"],
                pre_out=o["pre"],
                y=s["y"],
                xq=s["xq"],
                xs=s["xs"],
                m=m,
                rms_eps=w["rms_eps"],
                hc_eps=w["hc_eps"],
                post_alpha=w["post_alpha"],
                norm_eps=w["norm_eps"],
            ),
            rt=dict(gate=w["gate"], y=s["y"], part=s["rpart"], m=m),
            sgu=dict(
                w=w["sw"], ws=w["ss"], xq=s["xq"], xs=s["xs"], part=s["gpart"], m=m
            ),
            p3=dict(
                rpart=s["rpart"],
                bias=w["bias"],
                logits=s["logits"],
                ids=s["ids"],
                wts=s["wts"],
                gpart=s["gpart"],
                hq=s["hq"],
                hs=s["hs_sh"],
                m=m,
            ),
            prep=dict(
                x=s["y"],
                ids=ids,
                xq=s["xq"],
                xs=s["xs"],
                groups=s["groups"],
                counts=self.s["counts"],
                m=m,
                x_bytes=m * H * 2,
                ids_bytes=P * 4,
                xq_bytes=m * H,
                xs_bytes=m * 192,
                groups_bytes=P * 80,
            ),
            ug=dict(
                w13=w["w13"],
                s13=w["s13"],
                xq=s["xq"],
                xs=s["xs"],
                groups=s["groups"],
                h=s["h"],
                hs=s["hs"],
                counts=self.s["counts"],
                xq_bytes=m * H,
                xs_bytes=m * 192,
                h_bytes=P * I_REAL,
                hs_bytes=P * 32,
            ),
            dn=dict(
                w2=w["w2"],
                s2=w["s2"],
                h=s["h"],
                hs=s["hs"],
                wts=wts,
                groups=s["groups"],
                y=s["y2"],
                counts=self.s["counts"],
                h_bytes=P * I_REAL,
                hs_bytes=P * 32,
                wts_bytes=P * 4,
                y_bytes=P * H * 4,
            ),
            sdn=dict(
                w=w["dw"], ws=w["ds"], hq=s["hq"], hs=s["hs_sh"], out=s["shared"], m=m
            ),
            comb=dict(
                y=s["y2"],
                ids=ids,
                out=o["part"],
                m=m,
                y_bytes=P * H * 4,
                ids_bytes=P * 4,
                out_bytes=m * H * 2,
                add=s["shared"],
                add_bytes=m * H * 2,
            ),
        )
        if self.v3:  # y2's first P·5120·2 bytes hold the bf16 y (in the f32 buffer)
            yb = self.s["y2"].view(BF).view(-1)[: P * H].view(P, H)
            a["seam"].update(x1=s["y"], norm_w=w["norm_w"], xq=s["xq"], xs=s["xs"])
            a["p3"].update(
                mx_partial=s["partial"],
                mx_scale=w["scale"],
                mx_base=w["base"],
                mx_post_out=o["post"],
                mx_comb_out=o["comb"],
                mx_pre_out=o["pre"],
                mx_rstd=s["rstd"],
                mx_m=m,
                mx_rms_eps=w["rms_eps"],
                mx_hc_eps=w["hc_eps"],
                mx_post_alpha=w["post_alpha"],
                mx_norm_eps=w["norm_eps"],
                mx_dbg=s["rstd"],
            )
            a["ug"].update(rstd=s["rstd"])
            a["dn"].update(y=yb, y_bytes=P * H * 2)
            a["comb"].update(y=yb, y_bytes=P * H * 2)
        return a

    def launch(self, a: dict, m: int, skip=()) -> None:
        done = set()
        for ph in self.phases:
            for tag, _, _ in ph:
                if tag in skip or tag in done or tag not in self.ks:
                    continue
                if tag in self.merged:
                    kern, g, blk, tags = self.merged[tag]
                    kern(
                        g, blk, **{f"{t}.{n}": v for t in tags for n, v in a[t].items()}
                    )
                    done |= set(tags)
                    continue
                kern, grid = self.ks[tag]
                g = (
                    (m * 640 + 255) // 256
                    if grid == "comb"
                    else m
                    if grid == "m"
                    else 40 * ffn_seam.tg_for(m)
                    if grid == "seam2"
                    else grid[0]
                    if isinstance(grid, tuple)
                    else grid
                )
                kern(g, self.block.get(tag, 256), **a[tag])

    def __call__(
        self, w: dict, x, residual, post_mix, res_mix, attn_pre, fold: bool = False
    ):
        """X [M, 5120] bf16 (the attention's reduced output), residual [M, 4,
        5120] bf16, post_mix [M, 4, 1], res_mix [M, 4, 4], attn_pre [M, 4] f32
        -> part [M, 5120] bf16, R_new, post_next, comb_next, ffn_pre.
        fold (chain mode): no combine launch; part is left unwritten and
        ``self.comb_in`` = (y, ids, add) for ar_runner.CombAllReduce (the
        combine with all-reduce #2 folded in; y bf16 with v3)."""
        m = x.shape[0]
        assert 1 <= m <= MAX_M, m
        ins = (residual, x, post_mix, res_mix, attn_pre)
        o = self.outputs(residual)
        a = self.args(w, ins, o)
        self.launch(a, m, skip=("comb",) if fold else ())
        if fold:
            c = a["comb"]
            self.comb_in = (c["y"], c["ids"], c["add"])
        return o["part"], o["r_out"], o["post"], o["comb"], o["pre"]
