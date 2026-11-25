# SPDX-License-Identifier: Apache-2.0
"""MXFP4 activation quantization on the device, as aiter's Triton ``_mxfp4_quant_op``
rounds it (the rule of ``mxfp4_quant_rows`` and of the compact-A4 experts' input and
SwiGLU re-quantization), plus the ballot lane count the expert plan needs.

A 32-value block's E8M0 scale comes from the block's max |x|: the max is rounded
at mantissa bit 21, the unbiased exponent is ``floor(log2) - 2`` clamped at -127,
and each value times ``2^-unbiased`` is rounded to E2M1 half to even, saturating at
6, with the 0 / 0.5 / 1 denormal step taken by the 2^22 addition. This is the
"even" scaling of aiter's ``_dynamic_mxfp4_quant_kernel``, not the ceil rule of
``common.mx.code_ceil`` (vLLM's MXFP8), so it lives here until ``common.mx``
takes it (M2 plan B.2).

``lanes_before`` counts by 32-bit halves: ``common.ops.lanes_below`` shifts a
64-bit one by the lane and miscounts past lane 31, which a 64-pick ballot round
reaches.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

from sgl_amd_mono.common.ops import permlane_swap, popcount

E2M1_MAX = 6.0


def e8m0_f32(byte):
    """An E8M0 byte (as loaded, any width) -> the f32 power of two it encodes."""
    return ((fx.Int32(byte) & 0xFF) << 23).bitcast(fx.Float32)


def mxfp4_scale(amax):
    """The MXFP4 block scale of a block whose max |x| is ``amax`` (fp32, >= 0):
    (the E8M0 code as an Int32, the quantizing factor ``2^-unbiased`` as fp32)."""
    bits = (amax.bitcast(fx.Int32) + fx.Int32(0x200000)) & fx.Int32(-0x800000)
    unb = fx.max(((bits >> 23) & 0xFF) - 129, fx.Int32(-127))
    return unb + 127, ((fx.Int32(127) - unb) << 23).bitcast(fx.Float32)


def e2m1_code(x, qscale):
    """The E2M1 code (an Int32 holding 4 bits, the sign in bit 3) of ``x * qscale``:
    half to even, saturating at 6, the denormal step through the 2^22 addition."""
    qb = (x * qscale).bitcast(fx.Int32)
    neg = qb < 0
    ab = qb & fx.Int32(0x7FFFFFFF)
    af = ab.bitcast(fx.Float32)
    den = ((af + fx.Float32(4194304.0)).bitcast(fx.Int32) - fx.Int32(149 << 23)) & 0xFF
    nrm = ((ab + fx.Int32((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = (af >= fx.Float32(E2M1_MAX)).select(
        fx.Int32(7), (af < fx.Float32(1.0)).select(den, nrm)
    )
    return code | neg.select(fx.Int32(8), fx.Int32(0))


def lanes_before(bal, lane):
    """How many lanes of ballot ``bal`` (Int64) lie below ``lane``, by 32-bit halves."""
    lo = fx.Int32(bal & fx.Int64(0xFFFFFFFF))
    hi = fx.Int32(bal >> fx.Int64(32))
    low_part = popcount(lo & ((fx.Int32(1) << fx.min(lane, 31)) - 1))
    high_part = popcount(lo) + popcount(
        hi & ((fx.Int32(1) << fx.max(lane - 32, 0)) - 1)
    )
    return (lane < 32).select(low_part, high_part)


# ---------------------------------------------------------------- expert weight loads
# A wave's view of one 16-row MXFP4 weight tile: 4 dwordx4 registers j (j0, j1) x 4
# lane groups q (q0, q1, a 16-lane row each) x 16 lanes l (l0 .. l3). The scaled MFMA
# wants register j = kc (j0 = k0: the 64 B half of the 128 B line, j1 = k1: the line
# of the wave's 256 B K slab), lane group q = g4 (the 16 B piece of the half: g0, g1),
# lane l = the row (r0 .. r3). Loaded that way, each 16-lane group reads 16 B from 16
# rows: 16 sectors a lane group (the half-line pattern, ~4.7-5.8 TB/s on a cold
# stream). A load layout maps the same position bits to other address bits so a lane
# group reads few sectors; the loaded registers are then brought back to the MFMA
# layout with bit swaps between a register bit and a lane bit (each swap moves bytes
# between lanes and registers only: every MFMA gets the same bytes in the same K
# order). Swap costs a register pair: q0 / q1 one v_permlane16 / 32_swap a dword
# pair, a lane bit a v_mov_b32_dpp + v_cndmask a dword (as compiled: the select is
# VOP3, so the DPP does not fold).
FP4_MFMA_LAYOUT = dict(
    j0="k0", j1="k1", q0="g0", q1="g1", l0="r0", l1="r1", l2="r2", l3="r3"
)
FP4_LOAD_LAYOUTS = {
    # (load layout, the swaps (register bit, position bit) back to the MFMA layout)
    "mfma": (FP4_MFMA_LAYOUT, ()),
    # lane pairs 32 B contiguous, a lane group 8 rows x 32 B, a register 8 rows x
    # 128 B (whole lines); 40 VALU a tile (8 permlane16_swap, 16 DPP, 16 cndmask)
    "a8f": (
        dict(j0="r0", j1="k1", q0="k0", q1="g1", l0="g0", l1="r1", l2="r2", l3="r3"),
        (("j0", "l0"), ("j0", "q0")),
    ),
    # lane quads 64 B contiguous, a lane group 4 rows x 64 B, a register 4 rows x
    # 256 B; 80 VALU a tile (TP8 ~0.5% behind a8f)
    "b4": (
        dict(j0="r0", j1="r1", q0="k0", q1="k1", l0="g0", l1="g1", l2="r2", l3="r3"),
        (("j0", "l0"), ("j0", "q0"), ("j1", "l1"), ("j1", "q1")),
    ),
}
# the expert loads' layout: with it, nontemporal (``CM_NT``) weight loads stream at
# ~5.1 / 5.8 / 6.5 TB/s (112 / 168 / 256 MB cold, MI355X) where the MFMA layout's ran
# 4.7 / 5.3 / 5.8 TB/s by default and 4.1 / 4.8 / 5.2 with nt; TP8 layer -3 .. -8 us
FP4_LOAD_LAYOUT = "a8f"
_FP4_BYTES = {"g0": 16, "g1": 32, "k0": 64, "k1": 128}
_FP4_ROWS = {"r0": 1, "r1": 2, "r2": 4, "r3": 8}
_POS_BIT = {"q0": 4, "q1": 5, "l0": 0, "l1": 1, "l2": 2, "l3": 3}


def _check_layouts():
    """Every load layout is a bijection whose swaps end in the MFMA layout."""
    for name, (lay, swaps) in FP4_LOAD_LAYOUTS.items():
        assert sorted(lay.values()) == sorted(FP4_MFMA_LAYOUT.values()), name
        cur = dict(lay)
        for jb, pb in swaps:
            cur[jb], cur[pb] = cur[pb], cur[jb]
        assert cur == FP4_MFMA_LAYOUT, name


_check_layouts()


def fp4_load_offsets(lane, row_bytes, layout=None):
    """The load layout's byte offsets: a lane's (its lane bits'), and each register's
    (python ints), relative to the tile's first row and the wave's K slab."""
    lay = FP4_LOAD_LAYOUTS[layout or FP4_LOAD_LAYOUT][0]
    lane_off = fx.Int32(0)
    for pos, bit in _POS_BIT.items():
        lab = lay[pos]
        w = _FP4_ROWS[lab] * row_bytes if lab in _FP4_ROWS else _FP4_BYTES[lab]
        lane_off = lane_off + ((lane >> bit) & 1) * w
    reg_off = []
    for j in range(4):
        o = 0
        for jb, v in (("j0", j & 1), ("j1", j >> 1)):
            lab = lay[jb]
            o += v * (
                _FP4_ROWS[lab] * row_bytes if lab in _FP4_ROWS else _FP4_BYTES[lab]
            )
        reg_off.append(o)
    return lane_off, reg_off


_DPP_FROM_BELOW = {
    3: 0x118,
    2: 0x114,
    1: 0x4E,
    0: 0xB1,
}  # lane l reads l - 2^b (its bit set)
_DPP_FROM_ABOVE = {
    3: 0x108,
    2: 0x104,
    1: 0x4E,
    0: 0xB1,
}  # lane l reads l + 2^b (bit clear)


def _swap_lane_bit(b, a_, b_, lane):
    """Swap position bits (register, lane bit b) of the register pair (a_: 0, b_: 1)."""
    on = (lane & (1 << b)) != 0
    off = (lane & (1 << b)) == 0
    from_b = fx.Int32(update_dpp_i32(b_, b_, _DPP_FROM_BELOW[b], 0xF, 0xF, True))
    from_a = fx.Int32(update_dpp_i32(a_, a_, _DPP_FROM_ABOVE[b], 0xF, 0xF, True))
    return off.select(a_, from_b), on.select(b_, from_a)


def fp4_unswizzle(wts, lane, layout=None):
    """The loaded registers (``fp4_load_offsets`` layout) in the MFMA layout."""
    swaps = FP4_LOAD_LAYOUTS[layout or FP4_LOAD_LAYOUT][1]
    if not swaps:
        return wts
    R = [[fx.Int32(w[d]) for d in range(4)] for w in wts]
    for jb, pb in swaps:
        m = 1 if jb == "j0" else 2
        for j in range(4):
            if j & m:
                continue
            j2 = j | m
            for d in range(4):
                x, y = R[j][d], R[j2][d]
                if pb == "q0":
                    x, y = permlane_swap(16, x, y)
                elif pb == "q1":
                    x, y = permlane_swap(32, x, y)
                else:
                    x, y = _swap_lane_bit(_POS_BIT[pb], x, y, lane)
                R[j][d], R[j2][d] = x, y
    return [fx.Vector.from_elements(R[j], fx.Int32) for j in range(4)]
