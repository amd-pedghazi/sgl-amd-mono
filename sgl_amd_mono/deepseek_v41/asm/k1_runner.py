# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/k1_runner.py (changed: import paths, Triton import, weights() from plain tensors, no vLLM top-k fallback in idx_wide)
"""K1, the attention half of a decoder layer (from the attention seam's
inputs to the all-reduced attention output), in the pieces vLLM's breakable
CUDA graph needs:

    run = K1Runner(v3=True)
    w = run.weights(layer, v3=True)
    # captured: the seam (mHC pre-mix + norm) → (r_new, post, comb, y, pre)
    outs = run.seam_v3(w, residual, sub, post_mix, res_mix, pre_mix)
    # captured: wqkv, norms + SWA insert, wq_b
    q = run.front_v3(w, m, positions, slots, swa_cache, swa_bs)
    # eager break: sparse attention → o (frag)
    run.attend(w, q, ..., v3=True)
    # ... or vLLM's MXFP8 attention output → o (mixed steps)
    run.from_attn_out(attn_out, m)
    # captured: wo_a, wo_b with all-reduce #1 folded in
    out = run.back_ar(w, m, pos, wob, wire)

The v1 pieces (``seam``, ``front``, ``idx_q``, ``back``) serve the Engram
seam and the v3 pieces' internals.

After an index source's top-k, ``run.translate(...)`` maps it through the
block table into the scratch's fixed-stride lists (token t at t·512, counts in
``tk_lens``), which ``attend(.., fixed=True)`` reads instead of vLLM's ragged
pack (4 launches). ``run.idx_logits(...)`` (kernels/idx_logits.py) computes
the indexer's MXFP4 logits and writes the same lists from its top-512 in one
launch, in place of vLLM's logits, top-k and translate (``idx_wide`` /
``idx_gather``: wide rows and the candidate source / consumers).

v3 (``K1Runner(v3=True)``):
``seam_v3`` runs the seam main only (y' = MXFP8(x1 · attn_norm.w) into wqkv's
x, no fin; post / comb / pre come back empty and are filled by ``front_v3``),
``front_v3`` runs wqkv32n with the mixes waves (fin's mixes, rstd[t]) and wq_b
with the kv wave (rstd applied, record + SWA insert); q stays without the q
norm's rrms, which ``attend(.., v3=True)`` folds into its score scale (and
``q_rr`` applies for vLLM's attention on mixed steps). The y it returns is a
placeholder: on consumer layers vLLM only passes it on to our attention.
Index layers on v3 (``seam_v3(.., idx=True)``): y is y' = bf16(x1 ·
attn_norm.w) without rstd; the indexer's weights_proj reads it and its wq_b
reads front_v3's q x (no rrms), both off by a positive per-token factor that
the top-k selection doesn't see; comp_tail(rs_part=) scales the compressor
GEMM's rows by rstd itself. ``idx_q`` runs after ``front_v3`` as after ``front``.

Activations between the pieces stay in the runner's scratch in gemv8's frag
layout (sized for M = 256, allocated once; layers run one after another on a
stream, so they share it). Weights are frag-shuffled copies (gemv8.frag_w),
~40 MB per layer.
"""

import struct

import torch
import triton
import triton.language as tl

from sgl_amd_mono.deepseek_v41.asm.kernels import (
    attn_dec1,
    ffn_seam,
    gemv8,
    idx_logits,
    idx_topk,
    qkv_norm,
)
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import Kernel

H, MAX_M, HEADS, D = 5120, 256, 16, 512
BF = torch.bfloat16
ATTRS = {"wqkv": "fused_wqa_wkv", "wq_b": "wq_b", "wo_a": "wo_a", "wo_b": "wo_b"}
GEMVS = {n: (gemv8.K1[n], a) for n, a in ATTRS.items()}
GEMVS["idx_wq_b"] = (gemv8.IDX_WQ_B, "indexer.wq_b")


def f32bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


gemv_mode = gemv8.k1_mode


@triton.jit
def _translate_kernel(
    out, lens, topk, topk_stride, req_of, valid, bt, bt_stride, bs, TOPK: tl.constexpr
):
    """Token t's top-k (local positions, -1 = none) → global slots at
    out[t·TOPK ..] (-1 stays -1) and lens[t] = its non-negative count (0 for
    an invalid token), as _compute_global_topk_ragged_indices_and_indptr_old."""
    t = tl.program_id(0)
    j = tl.arange(0, TOPK)
    loc = tl.load(topk + t * topk_stride + j)
    ok = loc >= 0
    req = tl.load(req_of + t)
    blk = tl.load(bt + req * bt_stride + loc // bs, mask=ok, other=0)
    tl.store(out + t * TOPK + j, tl.where(ok, blk * bs + loc % bs, -1))
    tl.store(lens + t, tl.where(tl.load(valid + t) != 0, tl.sum(ok.to(tl.int32), 0), 0))


def attn_slog(m: int) -> int:
    return 2 if m <= 64 else 1 if m <= 128 else 0


class K1Runner:
    def __init__(self, device=None, v3: bool = False):
        self.dev = torch.device(device or "cuda")
        self.v3 = v3
        mk = lambda d, n=None: Kernel(
            n or d["name"],
            d["src"] if n is None else d["src"].replace(d["name"], n),
            d["ka"],
        )
        self.k_seam = mk(ffn_seam.source_v2("k1_seam2"))
        self.k_seam_eg = mk(ffn_seam.source_v2("k1_seam2_eg", post_stride=64))
        self.k_fin = mk(ffn_seam.source_fin("k1_seam_fin", frag=True))
        self.k_qkv = mk(qkv_norm.source())
        self.k_attn = mk(attn_dec1.source("k1_attn", frag_out=True))
        self.k_attn_fx = mk(
            attn_dec1.source("k1_attn_fx", frag_out=True, tk_fixed=True)
        )
        self.k_lg: dict = {}
        self.k_g = {}
        for name, (cfg, _) in GEMVS.items():
            for md in {gemv_mode(name, m) for m in (64, 128, 256)}:
                self.k_g[name, md] = mk(gemv8.source(cfg, md), f"{cfg.name}_{md}")
        if v3:
            self.k_seam_q = mk(ffn_seam.source_v2("k1_seam2q", yq=True))
            self.k_seam_eg_q = mk(
                ffn_seam.source_v2("k1_seam2q_eg", post_stride=64, yq=True)
            )
            # index layers: + bf16 y' in x1's slot (the compressor GEMM and the
            # indexer's weights_proj read it)
            self.k_seam_fy = mk(ffn_seam.source_v2("k1_seam2fy", yq="fy"))
            self.k_seam_eg_fy = mk(
                ffn_seam.source_v2("k1_seam2fy_eg", post_stride=64, yq="fy")
            )
            d = gemv8.source(gemv8.WQKV32N, "k", mix_wave=True)
            self.k_wqkv_mx, self.mx_block = mk(d), d["block"]
            d = gemv8.source(gemv8.WQ_B, "k", kv_wave="rstd", kv_bar=True)
            self.k_wqb_kv, self.kv_block = mk(d), d["block"]
            # index layers: + the indexer's wq_b with its q RoPE + MXFP4 as extra CTAs
            # (mode k)
            d = gemv8.source(
                gemv8.WQ_B,
                "k",
                kv_wave="rstd",
                kv_bar=True,
                extra=(gemv8.IDX_WQ_BQ, "k"),
            )
            self.k_wqb_kv_x2, self.x2_grid = mk(d), d["grid"]
            # A3 with rr / rstd and bf16 split partials; speculative slot ids at M ≤ 128
            # (attn_spec)
            self.k_attn3 = {
                (sp, fx): mk(
                    attn_dec1.source(
                        f"k1_attn3{'_sp' * sp}{'_fx' * fx}",
                        frag_out=True,
                        rr_scale="rstd",
                        spec=sp,
                        part_bf16=True,
                        tk_fixed=fx,
                    )
                )
                for sp in (False, True)
                for fx in (False, True)
            }
        self._scratch()

    def _scratch(self):
        d = self.dev
        u8 = lambda n: torch.zeros(n, dtype=torch.uint8, device=d)
        self.s = dict(
            x1=torch.empty(MAX_M, H, dtype=BF, device=d),
            partial=torch.zeros(MAX_M, ffn_seam.SLICES, 32, device=d),
            yq=u8(MAX_M * H),
            ys=u8(MAX_M * H // 32),
            qr_kv=torch.empty(MAX_M, 1792, dtype=BF, device=d),
            qrq=u8(MAX_M * 1280),
            qrs=u8(MAX_M * 40),
            q=torch.empty(MAX_M, HEADS * D, dtype=BF, device=d),
            iq=torch.empty(MAX_M, 4096, dtype=BF, device=d),
            o=u8(MAX_M * HEADS * D),
            os=u8(MAX_M * HEADS * D // 32),
            z=u8(MAX_M * 2048),
            zs=u8(MAX_M * 64),
            apart=torch.zeros(MAX_M, attn_dec1.PSTRIDE // 4, device=d),
            acnt=torch.zeros(MAX_M * 5, dtype=torch.int32, device=d),
            dummy=u8(64),
            tk=torch.empty(MAX_M * attn_dec1.TK_STRIDE, dtype=torch.int32, device=d),
            tk_lens=torch.empty(MAX_M, dtype=torch.int32, device=d),
            eye=torch.eye(4, device=d).expand(MAX_M, 4, 4).contiguous(),
        )
        if self.v3:
            self.s.update(
                iqc=u8(MAX_M * 2048).view(MAX_M, 32, 64),
                iqs=u8(MAX_M * 128).view(MAX_M, 128),
                ss=torch.zeros(MAX_M * 40, device=d),
                rstd=torch.zeros(MAX_M, device=d),
                y3=torch.empty(MAX_M, H, dtype=BF, device=d),
                part_wqkv32=torch.zeros(gemv8.WQKV32N.part_bytes() // 4, device=d),
                cnt_wqkv32=torch.zeros(gemv8.WQKV32N.NGP, dtype=torch.int32, device=d),
            )
            self._mix: tuple | None = None
        for name, (cfg, _) in GEMVS.items():
            self.s["part_" + name] = torch.zeros(
                max(cfg.part_bytes() // 4, 1), device=d
            )
            self.s["cnt_" + name] = torch.zeros(cfg.NGP, dtype=torch.int32, device=d)

    # ---------------------------------------------------------------- weights

    def weights(self, t: dict, seam_only: bool = False, v3: bool = False) -> dict:
        """Kernel-ready tensors of one DSv4.1 decoder layer's attention half
        (+ the indexer's wq_b on index layers) from ``t``, plain tensors of this
        rank (sgl_amd_mono.deepseek_v41.asm.k1 reads them off SGLang's layer);
        seam_only: the seam's alone; v3: wqkv for WQKV32N (instead of WQKV16's)
        and q_norm.weight in bf16. ``t``: hc_fn [24, 4H], hc_scale [3], hc_base
        [24] f32, norm [H] (attn_norm.weight), rms_eps, hc_eps, post_alpha,
        norm_eps, sinkhorn; q_norm [1280], kv_norm [512] bf16, sink [16] f32,
        cos_sin [positions, 64] f32 (32 cos, then 32 sin), scale (the softmax
        scale), eps (q / kv norms); wqkv, wq_b, wo_a, wo_b and on index layers
        idx_wq_b (with idx_cos_sin): (e4m3 [N, K] row-major, E8M0 [N/32, K/32])."""
        w = dict(
            fn_hl=ffn_seam.fn_hl(t["hc_fn"].float()),
            scale=t["hc_scale"].float().contiguous(),
            base=t["hc_base"].float().contiguous(),
            norm_w=t["norm"].to(BF).contiguous(),
            rms_eps=f32bits(float(t["rms_eps"])),
            hc_eps=f32bits(float(t["hc_eps"])),
            post_alpha=f32bits(float(t["post_alpha"])),
            norm_eps=f32bits(float(t["norm_eps"])),
        )
        assert int(t["sinkhorn"]) == ffn_seam.SINKHORN
        if seam_only:
            return w
        w.update(
            qw=t["q_norm"].contiguous(),
            kvw=t["kv_norm"].contiguous(),
            sink=t["sink"].float()[:HEADS].contiguous(),
            cs=t["cos_sin"].float().contiguous(),
            qk_scale=f32bits(float(t["scale"]) * 1.4426950408889634),
            eps=f32bits(float(t["eps"])),
        )
        index = t.get("idx_wq_b") is not None
        if v3:
            w["qw_bf"] = t["q_norm"].to(BF).contiguous()
            if index:
                wq, wqs = t["idx_wq_b"]
                w["idx_wq_bq"] = gemv8.frag_w(
                    wq.view(torch.uint8).contiguous(),
                    wqs.view(torch.uint8).contiguous(),
                    gemv8.IDX_WQ_BQ,
                )
                w["ics"] = t["idx_cos_sin"].float().contiguous()
        for name, (cfg, _) in GEMVS.items():
            if name == "idx_wq_b" and not index:
                continue
            if name == "wqkv" and v3:
                cfg = gemv8.WQKV32N
            wt, sc = t[name]
            assert tuple(wt.shape) == (cfg.N, cfg.K), (name, tuple(wt.shape))
            w[name] = gemv8.frag_w(
                wt.view(torch.uint8).contiguous(),
                sc.view(torch.uint8).contiguous(),
                cfg,
            )
        return w

    # ---------------------------------------------------------------- pieces

    def seam(self, w, residual, sub, post_mix, res_mix, pre_mix, kern=None):
        """The attention seam (mHC post of the FFN + pre of the attention):
        → R_new, post, comb, y (bf16, vLLM's attention input), pre; y also as
        MXFP8 frag in the scratch (wqkv's x)."""
        m, d = sub.shape[0], sub.device
        r_out = torch.empty_like(residual)
        post, comb = torch.empty(m, 4, 1, device=d), torch.empty(m, 4, 4, device=d)
        pre, y = torch.empty(m, 4, device=d), torch.empty(m, H, dtype=BF, device=d)
        s = self.s
        tg = ffn_seam.tg_for(m)
        (kern or self.k_seam)(
            40 * tg,
            256,
            residual=residual,
            x=sub,
            post=post_mix,
            comb=res_mix,
            pre=pre_mix,
            fn_hl=w["fn_hl"],
            r_out=r_out,
            x1=s["x1"],
            partial=s["partial"],
            m=m,
            tg=tg,
        )
        self.k_fin(
            m,
            256,
            partial=s["partial"],
            x1=s["x1"],
            norm_w=w["norm_w"],
            scale=w["scale"],
            base=w["base"],
            post_out=post,
            comb_out=comb,
            pre_out=pre,
            y=y,
            xq=s["yq"],
            xs=s["ys"],
            m=m,
            rms_eps=w["rms_eps"],
            hc_eps=w["hc_eps"],
            post_alpha=w["post_alpha"],
            norm_eps=w["norm_eps"],
        )
        return r_out, post, comb, y, pre

    def seam_eg(self, w, residual, value, gates, pre_mix):
        """The attention seam of an Engram layer: the
        Engram output r'' = r' + g_h · v is a post-mix of r' (``residual``,
        vLLM's mhc_post output) with x = v (value [M, 5120], unit stride rows),
        post = the gate all-gather's [M, 16] f32 (copy h's gate at 4 h) and
        comb = I, so the seam kernels produce r'' and the pre in one go."""
        m = residual.shape[0]
        assert (
            value.stride(1) == 1
            and value.stride(0) == H
            and gates.shape == (m, 16)
            and gates.is_contiguous()
        )
        return self.seam(
            w, residual, value, gates, self.s["eye"][:m], pre_mix, kern=self.k_seam_eg
        )

    def seam_v3(
        self, w, residual, sub, post_mix, res_mix, pre_mix, kern=None, idx=False
    ):
        """seam() for v3: the seam main writes wqkv's x (MXFP8 of x1 · attn_norm.w);
        post / comb / pre are allocated here and filled by front_v3 (the mixes waves); y
        is a placeholder (the scratch). idx (index layers): y = y' = bf16(x1 ·
        attn_norm.w) [M, 5120] without rstd (x1's slot of the scratch; the indexer's
        top-k doesn't change under a positive row scale, comp_tail applies rstd
        itself)."""
        m, d = sub.shape[0], sub.device
        r_out = torch.empty_like(residual)
        post, comb = torch.empty(m, 4, 1, device=d), torch.empty(m, 4, 4, device=d)
        pre = torch.empty(m, 4, device=d)
        s = self.s
        tg = ffn_seam.tg_for(m)
        if kern is None:
            kern = self.k_seam_fy if idx else self.k_seam_q
        kern(
            40 * tg,
            256,
            residual=residual,
            x=sub,
            post=post_mix,
            comb=res_mix,
            pre=pre_mix,
            fn_hl=w["fn_hl"],
            r_out=r_out,
            x1=s["x1"],
            partial=s["partial"],
            m=m,
            tg=tg,
            norm_w=w["norm_w"],
            xq=s["yq"],
            xs=s["ys"],
        )
        self._mix = (post, comb, pre)
        return r_out, post, comb, (s["x1"] if idx else s["y3"])[:m], pre

    def seam_eg_v3(self, w, residual, value, gates, pre_mix, idx=False):
        """seam_eg() for v3 (the Engram layer's seam as a post-mix, see seam_eg); idx as
        seam_v3."""
        m = residual.shape[0]
        assert (
            value.stride(1) == 1
            and value.stride(0) == H
            and gates.shape == (m, 16)
            and gates.is_contiguous()
        )
        return self.seam_v3(
            w,
            residual,
            value,
            gates,
            self.s["eye"][:m],
            pre_mix,
            kern=self.k_seam_eg_fy if idx else self.k_seam_eg_q,
            idx=idx,
        )

    def front_v3(self, w, m, pos, slots, swa_cache, swa_bs, idx=False):
        """v3's front: wqkv32n (q → wq_b's MXFP8 x unnormalised, Σx² partials; kv rows;
        the mixes waves fill seam_v3's post / comb / pre and rstd) → wq_b + q RoPE, the
        kv wave (rstd, kv norm, RoPE, record, SWA insert at slots) → q [M, 16, 512] bf16
        without rrms (a view of the scratch). idx (index layers): wq_b's launch also
        runs the indexer's wq_b + its q RoPE + MXFP4 (IDX_WQ_BQ, extra CTAs) on the same
        x → ``idx_q3``."""
        s = self.s
        assert self._mix is not None, "front_v3 follows seam_v3"
        post, comb, pre = self._mix
        assert post.shape[0] == m
        c32, cq = gemv8.WQKV32N, gemv8.WQ_B
        self.k_wqkv_mx(
            c32.grid,
            self.mx_block,
            w=w["wqkv"][0],
            ws=w["wqkv"][1],
            x=s["yq"],
            xs=s["ys"],
            out=s["qr_kv"],
            outs=s["qrq"],
            part=s["part_wqkv32"],
            cnt=s["cnt_wqkv32"],
            cs=s["qrs"],
            pos=s["ss"],
            dbg=w["qw_bf"],
            m=m,
            mx_partial=s["partial"],
            mx_scale=w["scale"],
            mx_base=w["base"],
            mx_post_out=post,
            mx_comb_out=comb,
            mx_pre_out=pre,
            mx_rstd=s["rstd"],
            mx_m=m,
            mx_rms_eps=w["rms_eps"],
            mx_hc_eps=w["hc_eps"],
            mx_post_alpha=w["post_alpha"],
            mx_norm_eps=w["norm_eps"],
            mx_dbg=s["dummy"],
        )
        cache = swa_cache.view(swa_cache.shape[0], -1).view(torch.uint8)
        d = s["dummy"]
        kq, grid, x2 = self.k_wqb_kv, cq.grid, {}
        if idx:
            kq, grid = self.k_wqb_kv_x2, self.x2_grid
            x2 = {
                "x2_w": w["idx_wq_bq"][0],
                "x2_ws": w["idx_wq_bq"][1],
                "x2_out": s["iqc"],
                "x2_outs": s["iqs"],
                "x2_part": s["part_idx_wq_b"],
                "x2_cnt": s["cnt_idx_wq_b"],
                "x2_cs": w["ics"],
            }
        kq(
            grid,
            self.kv_block,
            w=w["wq_b"][0],
            ws=w["wq_b"][1],
            x=s["qrq"],
            xs=s["qrs"],
            out=s["q"],
            outs=d,
            part=s["part_wq_b"],
            cnt=s["cnt_wq_b"],
            cs=w["cs"],
            pos=pos,
            dbg=d,
            m=m,
            kv_qr_kv=s["qr_kv"],
            kv_qw=w["qw"],
            kv_kvw=w["kvw"],
            kv_pos=pos,
            kv_cs=w["cs"],
            kv_slots=slots,
            kv_cache=cache,
            kv_xq=d,
            kv_xs=d,
            kv_m=m,
            kv_eps=w["eps"],
            kv_stride=cache.stride(0),
            kv_bsl=swa_bs.bit_length() - 1,
            kv_rstd=s["rstd"],
            **x2,
        )
        return s["q"][:m].view(m, HEADS, D)

    def idx_q3(self, m):
        """front_v3(.., idx=True)'s indexer q as vLLM's rocm_mxfp4_indexer_q_quant
        returns it: (codes [M, 32, 64] uint8, E8M0 scales as one int32 per head [M, 32])
        (views of the scratch)."""
        return self.s["iqc"][:m], self.s["iqs"][:m].view(torch.int32)

    def q_rr(self, q, eps):
        """v3's q · rrms (rr = rstd · rsq(rstd² · Σ / 1280 + eps), front_v3's partials)
        for vLLM's attention (mixed steps; eager), bf16 in place."""
        m = q.shape[0]
        rd = self.s["rstd"][:m]
        ss = self.s["ss"].view(MAX_M, 40)[:m].sum(-1)
        rr = rd * torch.rsqrt(rd * rd * ss / 1280.0 + eps)
        q.copy_((q.float() * rr[:, None, None]).to(q.dtype))
        return q

    def _gemv(self, name, w, x, xs, out, outs, m, pos):
        cfg = GEMVS[name][0]
        s = self.s
        self.k_g[name, gemv_mode(name, m)](
            cfg.grid,
            256,
            w=w[name][0],
            ws=w[name][1],
            x=x,
            xs=xs,
            out=out,
            outs=outs,
            part=s["part_" + name],
            cnt=s["cnt_" + name],
            cs=w["cs"],
            pos=pos,
            dbg=s["dummy"],
            m=m,
        )

    def front(self, w, m, pos, slots, swa_cache, swa_bs):
        """Wqkv → q/kv norm, q MXFP8, SWA insert at slots (int64, −1 = skip) →
        wq_b + q RoPE → q [M, 16, 512] bf16 (a view of the scratch)."""
        s = self.s
        self._gemv("wqkv", w, s["yq"], s["ys"], s["qr_kv"], s["dummy"], m, pos)
        cache = swa_cache.view(swa_cache.shape[0], -1).view(torch.uint8)
        self.k_qkv(
            (m + 3) // 4,
            256,
            qr_kv=s["qr_kv"],
            qw=w["qw"],
            kvw=w["kvw"],
            pos=pos,
            cs=w["cs"],
            slots=slots,
            cache=cache,
            xq=s["qrq"],
            xs=s["qrs"],
            m=m,
            eps=w["eps"],
            stride=cache.stride(0),
            bsl=swa_bs.bit_length() - 1,
        )
        self._gemv("wq_b", w, s["qrq"], s["qrs"], s["q"], s["dummy"], m, pos)
        return s["q"][:m].view(m, HEADS, D)

    def idx_q(self, w, m, pos):
        """The indexer's wq_b on front's MXFP8 qr → bf16 [M, 32, 128] (a view
        of the scratch)."""
        s = self.s
        self._gemv("idx_wq_b", w, s["qrq"], s["qrs"], s["iq"], s["dummy"], m, pos)
        return s["iq"][:m].view(m, 32, 128)

    def translate(self, topk, req, valid, bt, bs):
        """An index source's top-k [n, 512] → the scratch's fixed-stride lists
        (``attend(.., fixed=True)``'s tk_idx / tk_ptr)."""
        n = topk.shape[0]
        assert n <= MAX_M and topk.shape[1] == attn_dec1.TK_STRIDE
        _translate_kernel[(n,)](
            self.s["tk"],
            self.s["tk_lens"],
            topk,
            topk.stride(0),
            req,
            valid,
            bt,
            bt.stride(0),
            bs,
            TOPK=attn_dec1.TK_STRIDE,
            num_warps=4,
        )
        return self.s["tk"], self.s["tk_lens"]

    def idx_logits_prepare(self, width: int) -> None:
        """Builds the fused logits + top-512 kernel for rows of ``width`` columns, its
        logits scratch and the per-row hand-off counters (call outside graph
        capture)."""
        nl, lg_cap, lds_s1, _ = idx_logits.plan(width, MAX_M)
        key = (nl, lg_cap, lds_s1)
        if key not in self.k_lg:
            d = idx_logits.source(nl, lg_cap, lds_s1=lds_s1)
            self.k_lg[key] = (Kernel(d["name"], d["src"], d["ka"]), d)
        cols = -(-width // 32) * 32
        if f"lg{cols}" not in self.s:
            self.s[f"lg{cols}"] = torch.empty(MAX_M, cols, device=self.dev)
        if "lgcnt" not in self.s:
            self.s["lgcnt"] = torch.zeros(
                MAX_M * 32, dtype=torch.int32, device=self.dev
            )  # one per 128 B

    def idx_logits(
        self,
        width,
        q,
        qs,
        w,
        kv,
        page_stride,
        page_size,
        kbt,
        lens,
        valid,
        req,
        bt,
        bs,
        out_local,
    ):
        """The indexer's logits of the n = q.shape[0] decode rows (q [n, 32, 64] / qs
        [n, 32, 4] MXFP4, w [n, 32] f32, row r over kv pages kbt[r] (aiter's preshuffled
        pages, page_stride bytes apart), lens[r] columns) and their top-512, as topk()
        leaves it."""
        n = q.shape[0]
        nl, lg_cap, lds_s1, slog = idx_logits.plan(width, n)
        k, d = self.k_lg[(nl, lg_cap, lds_s1)]
        cols = -(-width // 32) * 32
        lg = self.s[f"lg{cols}"]
        assert (
            n <= MAX_M
            and bs & (bs - 1) == 0
            and page_size & (page_size - 1) == 0
            and page_size % 32 == 0
        )
        assert (
            q.dtype == qs.dtype == torch.uint8
            and q.is_contiguous()
            and qs.is_contiguous()
        )
        assert (
            q.shape[1:] == (32, 64)
            and qs.shape[1:] == (32, 4)
            and w.dtype == torch.float32
        )
        assert (
            w.stride() == (32, 1)
            and kbt.dtype == torch.int32
            and kbt.stride(1) == 1
            and kbt.shape[0] >= n
        )
        assert (
            lens.dtype == req.dtype == bt.dtype == torch.int32
            and valid.dtype == torch.bool
        )
        assert (
            lens.is_contiguous()
            and req.is_contiguous()
            and valid.is_contiguous()
            and page_stride < 2**32
        )
        assert -(-width // page_size) <= 512 and out_local.shape[1] == idx_topk.K
        k(
            8 * (1 << slog) * (-(-n // 8)),
            d["block"],
            logits=lg,
            lens=lens,
            valid=valid,
            req=req,
            bt=bt,
            out_local=out_local,
            tk=self.s["tk"],
            tk_lens=self.s["tk_lens"],
            lstride=cols * 4,
            bt_stride=bt.stride(0) * 4,
            local_stride=out_local.stride(0) * 4,
            bs_log2=bs.bit_length() - 1,
            q=q,
            qs=qs,
            w=w,
            kv=kv,
            kbt=kbt,
            cnt=self.s["lgcnt"],
            rows=n,
            kbt_stride=kbt.stride(0) * 4,
            page_stride=page_stride,
            e_log2=page_size.bit_length() - 1,
            slog=slog,
        )
        return self.s["tk"], self.s["tk_lens"]

    # wide rows: block-table entries per row the merge prefetches (≥ width / bs)
    WIDE_BT = 8192

    SEL_BLOCKS = (
        4 * 64 * 16 * 8
    )  # the source's top-2048 blocks in asm up to this many blocks (else vLLM's)

    def idx_wide_prepare(self, width: int, source: bool = False) -> None:
        """Builds the wide-row indexer kernels (rows past idx_logits' one-CTA top-k, the
        dense layers at long context) for rows of ``width`` columns: the main kernel
        (sized for the widest slices, at MAX_M) and a merge kernel per S, the candidate
        slots and the per-row tags (call outside capture). source (the candidate
        source): the main kernel also writes the block scores (8 positions), and the
        top-2048 blocks kernel (kernels/idx_topk.py at K 2048, 1024 threads per row)."""
        nl, lg_cap, _, slog0 = idx_logits.plan_wide(width, MAX_M)
        key = ("wide", nl, lg_cap) + (("bs",) if source else ())
        if key not in self.k_lg:
            d = idx_logits.source(
                nl,
                lg_cap,
                wide=True,
                split=True,
                nlf=1,
                bt_max=self.WIDE_BT,
                bsc=source,
            )
            self.k_lg[key] = (Kernel(d["name"], d["src"], d["ka"]), d)
        for slog in range(slog0, idx_logits.SLOG_MAX + 1):
            nlf = -(-(1 << slog) * idx_logits.CAPS // 2048)
            if ("merge", nlf) not in self.k_lg:
                d = idx_logits.merge_source(nlf, self.WIDE_BT)
                self.k_lg[("merge", nlf)] = (Kernel(d["name"], d["src"], d["ka"]), d)
        nlf_max = -(-(1 << idx_logits.SLOG_MAX) * idx_logits.CAPS // 2048)
        cols = idx_logits.row_slots(nlf_max, idx_logits.SLOG_MAX)
        if (
            "wslots" not in self.s or self.s["wslots"].shape[1] < 2 * cols
        ):  # (idx_gather_prepare's is smaller)
            self.s["wslots"] = torch.zeros(
                MAX_M, 2 * cols, dtype=torch.float32, device=self.dev
            )
        if "wtag" not in self.s:
            self.s["wtag"] = torch.zeros(
                MAX_M * 32, dtype=torch.int32, device=self.dev
            )  # one per 128 B
        if not source:
            self.wide_key = key
            return
        self.src_key = key
        nb = -(-width // 8)
        bstride = -(-width // 256) * 32  # rows of 128 B
        if self.s.get("wbs") is None or self.s["wbs"].shape[1] < bstride:
            self.s["wbs"] = torch.zeros(
                MAX_M, bstride, dtype=torch.float32, device=self.dev
            )
        self.k_sel = None
        if nb <= self.SEL_BLOCKS:
            d = idx_topk.source(
                16, -(-nb // (4 * 64 * 16)), k=2048, blk_out=True, lg_flag="nt"
            )
            self.k_sel = (Kernel(d["name"], d["src"], d["ka"]), d)

    def idx_wide(
        self,
        width,
        q,
        qs,
        w,
        kv,
        page_stride,
        page_size,
        kbt,
        lens,
        valid,
        req,
        bt,
        bs,
        out_local,
        cand=None,
        bends=None,
    ):
        """The wide-row path of idx_logits() (same arguments and results): S CTAs per
        row each keep a slice's logits and its ~515 best candidates, then a merge kernel
        per row takes the exact top-512. cand / bends (the candidate source, after
        idx_wide_prepare(source=True)): also the top-2048 blocks of 8 positions (newest
        block pinned) → cand [rows, 2048] (block ids, -1 padded; rows ≥ the decode rows
        (graph padding) get -1 only) for rows of bends [rows] blocks."""
        n = q.shape[0]
        nl, lg_cap, _, slog = idx_logits.plan_wide(width, n)
        k, d = self.k_lg[self.wide_key if cand is None else self.src_key]
        assert nl <= d["lg_cap"] // 64
        nlf = -(-(1 << slog) * idx_logits.CAPS // 2048)
        km, dm = self.k_lg[("merge", nlf)]
        slots = self.s["wslots"]
        assert (
            n <= MAX_M
            and bs & (bs - 1) == 0
            and page_size & (page_size - 1) == 0
            and page_size % 32 == 0
        )
        assert (
            q.dtype == qs.dtype == torch.uint8
            and q.is_contiguous()
            and qs.is_contiguous()
        )
        assert (
            q.shape[1:] == (32, 64)
            and qs.shape[1:] == (32, 4)
            and w.dtype == torch.float32
        )
        assert (
            w.stride() == (32, 1)
            and kbt.dtype == torch.int32
            and kbt.stride(1) == 1
            and kbt.shape[0] >= n
        )
        assert (
            lens.dtype == req.dtype == bt.dtype == torch.int32
            and valid.dtype == torch.bool
        )
        assert (
            lens.is_contiguous()
            and req.is_contiguous()
            and valid.is_contiguous()
            and page_stride < 2**32
        )
        assert -(-width // bs) <= self.WIDE_BT and out_local.shape[1] == idx_topk.K
        args = dict(
            logits=slots,
            lens=lens,
            valid=valid,
            req=req,
            bt=bt,
            out_local=out_local,
            tk=self.s["tk"],
            tk_lens=self.s["tk_lens"],
            lstride=slots.stride(0) * 4,
            bt_stride=bt.stride(0) * 4,
            local_stride=out_local.stride(0) * 4,
            bs_log2=bs.bit_length() - 1,
            q=q,
            qs=qs,
            w=w,
            kv=kv,
            kbt=kbt,
            cnt=self.s["wtag"],
            rows=n,
            kbt_stride=kbt.stride(0) * 4,
            page_stride=page_stride,
            e_log2=page_size.bit_length() - 1,
            slog=slog,
        )
        extra = {}
        if cand is not None:
            wbs = self.s["wbs"]
            rows = cand.shape[0]
            assert (
                cand.dtype == torch.int32
                and cand.shape[1] == 2048
                and cand.stride(1) == 1
                and rows <= MAX_M
            )
            assert (
                bends.dtype == torch.int32
                and bends.is_contiguous()
                and bends.shape[0] >= rows
            )
            assert -(-width // 8) <= wbs.shape[1]
            extra = dict(bs=wbs, bs_stride=wbs.stride(0) * 4)
        k(8 * (1 << slog) * (-(-n // 8)), d["block"], **args, **extra)
        km(n, dm["block"], **args)
        if cand is not None:
            if self.k_sel is not None:
                ks, ds = self.k_sel  # (valid / req / bt: unused by the block-id output)
                ks(
                    rows,
                    ds["block"],
                    logits=wbs,
                    lens=bends,
                    valid=bends,
                    req=bends,
                    bt=bends,
                    out_local=cand,
                    tk=self.s["tk"],
                    tk_lens=self.s["tk_lens"],
                    lstride=wbs.stride(0) * 4,
                    bt_stride=n,
                    local_stride=cand.stride(0) * 4,
                    bs_log2=0,
                )
            else:  # vLLM's _topk (rocm_paged_mxfp4_indexer) has no SGLang counterpart here
                raise NotImplementedError(
                    f"the candidate source's top-2048 blocks past {self.SEL_BLOCKS} blocks"
                )
        return self.s["tk"], self.s["tk_lens"]

    def idx_gather_prepare(self) -> None:
        """Builds the candidate consumers' kernels (idx_logits gather mode, plain K
        loads, S = 2 / 4 CTAs per row, the row's last CTA merges: ≤ 512 CTAs, so every
        split it polls is resident or done) and the candidate slots / tags (call outside
        capture)."""
        for slog in (1, 2):
            nl, lg_cap, nlf, _ = idx_logits.plan_gather(MAX_M, slog)
            key = ("gather", nl, lg_cap)
            if key not in self.k_lg:
                d = idx_logits.source(
                    nl,
                    lg_cap,
                    wide=True,
                    split=False,
                    nlf=nlf,
                    bt_max=self.WIDE_BT,
                    gather=True,
                    kshared="none",
                )
                self.k_lg[key] = (Kernel(d["name"], d["src"], d["ka"]), d)
        cols = idx_logits.row_slots(idx_logits.plan_gather(MAX_M, 2)[2], 2)
        if "wslots" not in self.s or self.s["wslots"].shape[1] < 2 * cols:
            self.s["wslots"] = torch.zeros(
                MAX_M, 2 * cols, dtype=torch.float32, device=self.dev
            )
        if "wtag" not in self.s:
            self.s["wtag"] = torch.zeros(MAX_M * 32, dtype=torch.int32, device=self.dev)

    def idx_gather(
        self,
        q,
        qs,
        w,
        kv,
        page_stride,
        page_size,
        kbt,
        lens,
        valid,
        req,
        bt,
        bs,
        out_local,
        cand,
    ):
        """A candidate consumer's indexer: logits over the row's candidate blocks (cand
        [n, 2048] block ids of 8 positions, any order, -1 = none) and their exact
        top-512 → out_local (positions) and the fixed-stride lists, as idx_logits()."""
        n = q.shape[0]
        nl, lg_cap, nlf, slog = idx_logits.plan_gather(n)
        k, d = self.k_lg[("gather", nl, lg_cap)]
        slots = self.s["wslots"]
        assert (
            n <= MAX_M
            and bs & (bs - 1) == 0
            and page_size & (page_size - 1) == 0
            and page_size % 32 == 0
        )
        assert (
            q.dtype == qs.dtype == torch.uint8
            and q.is_contiguous()
            and qs.is_contiguous()
        )
        assert (
            q.shape[1:] == (32, 64)
            and qs.shape[1:] == (32, 4)
            and w.dtype == torch.float32
        )
        assert (
            w.stride() == (32, 1)
            and kbt.dtype == torch.int32
            and kbt.stride(1) == 1
            and kbt.shape[0] >= n
        )
        assert (
            lens.dtype == req.dtype == bt.dtype == torch.int32
            and valid.dtype == torch.bool
        )
        assert (
            lens.is_contiguous()
            and req.is_contiguous()
            and valid.is_contiguous()
            and page_stride < 2**32
        )
        assert (
            out_local.shape[1] == idx_topk.K
            and cand.dtype == torch.int32
            and cand.shape[1] == idx_logits.NCAND
        )
        assert cand.stride(1) == 1 and cand.shape[0] >= n
        args = dict(
            logits=slots,
            lens=lens,
            valid=valid,
            req=req,
            bt=bt,
            out_local=out_local,
            tk=self.s["tk"],
            tk_lens=self.s["tk_lens"],
            lstride=slots.stride(0) * 4,
            bt_stride=bt.stride(0) * 4,
            local_stride=out_local.stride(0) * 4,
            bs_log2=bs.bit_length() - 1,
            q=q,
            qs=qs,
            w=w,
            kv=kv,
            kbt=kbt,
            cnt=self.s["wtag"],
            rows=n,
            kbt_stride=kbt.stride(0) * 4,
            page_stride=page_stride,
            e_log2=page_size.bit_length() - 1,
            slog=slog,
        )
        k(
            8 * (1 << slog) * (-(-n // 8)),
            d["block"],
            **args,
            cand=cand,
            cand_stride=cand.stride(0) * 4,
        )
        return self.s["tk"], self.s["tk_lens"]

    def attend(
        self,
        w,
        q,
        pos,
        swa_cache,
        swa_bs,
        sw_idx,
        sw_ptr,
        comp_cache,
        comp_bs,
        tk_idx,
        tk_ptr,
        fixed: bool = False,
        v3: bool = False,
    ):
        """Sparse attention of the decode rows q [M, 16, 512] → o (MXFP8 frag);
        fixed: tk_idx / tk_ptr are translate()'s lists and counts; v3: q from
        front_v3 (rrms folded into the score scale)."""
        m = q.shape[0]
        slog = attn_slog(m)
        s = self.s
        sc = swa_cache.view(swa_cache.shape[0], -1).view(torch.uint8)
        has_comp = comp_cache is not None and tk_idx is not None
        cc = (
            comp_cache.view(comp_cache.shape[0], -1).view(torch.uint8)
            if has_comp
            else sc
        )
        x3 = {}
        kern = self.k_attn_fx if fixed else self.k_attn
        if v3:
            kern = self.k_attn3[m <= 128, fixed and has_comp]
            x3 = dict(
                ss=s["ss"],
                eps=w["eps"],
                rstd=s["rstd"],
                sw_cap=sw_idx.numel(),
                tk_cap=(tk_idx.numel() if has_comp else sw_idx.numel()),
            )
        kern(
            8 * (1 << slog) * ((m + 7) // 8),
            256,
            q=q,
            swa=sc,
            comp=cc,
            swa_idx=sw_idx,
            swa_ptr=sw_ptr,
            tk_idx=tk_idx if has_comp else sw_idx,
            tk_ptr=tk_ptr if has_comp else sw_ptr,
            sink=w["sink"],
            pos=pos,
            cs=w["cs"],
            out=s["o"],
            outs=s["os"],
            part=s["apart"],
            cnt=s["acnt"],
            dbg=s["dummy"],
            m=m,
            swa_stride=sc.stride(0),
            swa_bsl=swa_bs.bit_length() - 1,
            comp_stride=cc.stride(0),
            comp_bsl=(comp_bs.bit_length() - 1) if has_comp else 0,
            has_comp=int(has_comp),
            qk_scale=w["qk_scale"],
            slog=slog,
            **x3,
        )

    def from_attn_out(self, data, scale, m):
        """VLLM's MXFP8 attention output (row-major, e.g. a mixed prefill +
        decode step) → o (frag)."""
        xf, sf = gemv8.frag_x(data[:m].view(torch.uint8), scale[:m].view(torch.uint8))
        self.s["o"].copy_(xf)
        self.s["os"].copy_(sf)

    def back(self, w, m, pos):
        """wo_a → z (MXFP8 frag) → wo_b → partial [M, 5120] bf16 (fresh)."""
        s = self.s
        out = torch.empty(m, H, dtype=BF, device=self.dev)
        self._gemv("wo_a", w, s["o"], s["os"], s["z"], s["zs"], m, pos)
        self._gemv("wo_b", w, s["z"], s["zs"], out, s["dummy"], m, pos)
        return out

    def back_ar(self, w, m, pos, wob, wire):
        """back() with all-reduce #1 folded into wo_b (ar_runner.WoBAllReduce,
        collective) → the all-reduced [M, 5120] bf16 (fresh)."""
        s = self.s
        self._gemv("wo_a", w, s["o"], s["os"], s["z"], s["zs"], m, pos)
        return wob(w["wo_b"][0], w["wo_b"][1], s["z"], s["zs"], m, wire)
