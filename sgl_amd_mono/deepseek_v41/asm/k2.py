# SPDX-License-Identifier: Apache-2.0
"""The FFN half of DSv4.1's decode layers above MONO_MAX rows at TP4 (gfx950):
vllm#65421's asm K2 chain (``k2_runner.K2Runner``) on the plugin's ``FFNWeights``, then
the combine with all-reduce #2 (``ar_runner.CombAllReduce``).

``AsmK2.__call__`` takes the attention half's (x_attn, residual, post, comb, pre),
x_attn all-reduced, and returns the layer's (x, residual, post, comb, pre) as the
monokernel's K2 does, as fresh tensors. One AsmK2 per process: the runner's scratch
(sized for MAX_M rows) is shared by the layers, which run one after another on one
stream, and every rank makes the same calls in the same order.

``SGLANG_AMD_MONO_ASM_K2``: "v1" (default: a fin launch normalizes the row first, as
SGLang's norm does) or "v3" (vLLM's chain: the seam quantizes the row to MXFP8 before
its rstd, which stage 1 applies to the gate / up sums). On the FFN half at 64-192 rows
with served-like routing, v1 takes 10.6-11.1 % less time than SGLang's and v3 17.0-18.7 %
less, but v3 missed the 1-point GSM8K rule against SGLang over two pooled runs, so v1 is
the default.
"""

import os

import torch

HC = 4
CHAINS = ("v1", "v3")
POST_ALPHA = 2.0  # mHC's post multiplier: SGLang's MHC_POST_MULT_VALUE


def chain_from_env() -> str:
    c = os.environ.get("SGLANG_AMD_MONO_ASM_K2", "v1")
    if c not in CHAINS:
        raise ValueError(f"SGLANG_AMD_MONO_ASM_K2={c!r}: v1 or v3")
    return c


class AsmK2:
    """K2 of every wide layer of one process (TP4)."""

    def __init__(
        self,
        rank: int,
        cpu_group,
        guard=None,
        *,
        chain: str | None = None,
        halt: bool = True,
    ):
        from sgl_amd_mono.deepseek_v41.asm.ar_runner import CombAllReduce
        from sgl_amd_mono.deepseek_v41.asm.k2_runner import K2Runner

        self.chain = chain or chain_from_env()
        if self.chain not in CHAINS:
            raise ValueError(f"chain {self.chain!r}: v1 or v3")
        self.run = K2Runner("chain", v3=self.chain == "v3")
        self.car = CombAllReduce(rank, cpu_group, 4, guard=guard, halt=halt)

    @staticmethod
    def weights(ffn) -> dict:
        """One layer's weights as K2Runner reads them, from the plugin's ``FFNWeights``.
        The routed experts (SGLang's MXFP4 in AITER's gate / up interleaved shuffle),
        the shared expert's FP8 copy, the router and the norm are read in place; new are
        fn_hl (the mHC fn as bf16 hi / lo fragments, 2 MB) and the scalar words."""
        from sgl_amd_mono.deepseek_v41 import HF_CONFIG
        from sgl_amd_mono.deepseek_v41.asm.k2_runner import I_REAL, H, f32bits
        from sgl_amd_mono.deepseek_v41.asm.kernels import ffn_seam, moe_down1, moe_ug1

        if HF_CONFIG["hc_sinkhorn_iters"] != ffn_seam.SINKHORN:
            raise ValueError(
                f"the asm seam runs {ffn_seam.SINKHORN} Sinkhorn iterations, the model "
                f"{HF_CONFIG['hc_sinkhorn_iters']}"
            )
        u8, fn = torch.uint8, ffn.hc_ffn_fn.float().contiguous()
        w = dict(
            fn=fn,
            fn_hl=ffn_seam.fn_hl(fn),
            scale=ffn.hc_ffn_scale.float().contiguous(),
            base=ffn.hc_ffn_base.float().contiguous(),
            norm_w=ffn.ffn_norm.to(torch.bfloat16).contiguous(),
            rms_eps=f32bits(HF_CONFIG["rms_norm_eps"]),
            hc_eps=f32bits(HF_CONFIG["hc_eps"]),
            post_alpha=f32bits(POST_ALPHA),
            norm_eps=f32bits(HF_CONFIG["rms_norm_eps"]),
            gate=ffn.gate_w.to(torch.bfloat16).contiguous(),
            bias=ffn.bias.float().contiguous(),
            sw=ffn.sgu.contiguous(),
            ss=ffn.sgu_s.view(u8).contiguous(),
            dw=ffn.sw2.contiguous(),
            ds=ffn.sw2_s.view(u8).contiguous(),
            w13=ffn.w13.view(u8),
            s13=ffn.w13_s.view(u8),
            w2=ffn.w2.view(u8),
            s2=ffn.w2_s.view(u8),
        )
        E = ffn.w13.shape[0]
        # the kernels stride through these buffers with fixed per-expert sizes (TP4)
        sizes = {
            "experts": (E, HF_CONFIG["n_routed_experts"]),
            "w13": (w["w13"].numel(), E * moe_ug1.W13_E),
            "s13": (w["s13"].numel(), E * moe_ug1.S13_E),
            "w2": (w["w2"].numel(), E * moe_down1.W2_E),
            "s2": (w["s2"].numel(), E * moe_down1.S2_E),
            "sw": (w["sw"].numel(), 2 * I_REAL * H),
            "ss": (w["ss"].numel(), 2 * I_REAL // 32 * (H // 32)),
            "dw": (w["dw"].numel(), H * I_REAL),
            "ds": (w["ds"].numel(), H // 32 * (I_REAL // 32)),
        }
        bad = {k: v for k, v in sizes.items() if v[0] != v[1]}
        if bad:
            raise ValueError(
                f"SGLang's layout differs from the asm K2's (got, want): {bad}"
            )
        return w

    def __call__(self, w, x_attn, residual, post_a, comb_a, pre_a, wire: str):
        m = x_attn.shape[0]
        _, res, post, comb, pre = self.run(
            w, x_attn, residual, post_a.view(m, HC, 1), comb_a, pre_a, fold=True
        )
        return self.car(*self.run.comb_in, m, wire), res, post, comb, pre

    def errors(self) -> int:
        """The combine's CTAs whose wait timed out (``halt=False`` kernels). Syncs."""
        return self.car.errors()
