# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/moe_combine.py (changed: import paths)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Routed-MoE combine: out[token][r] = bf16(Σ_k y[token*6 + k][r]) over the
token's 6 picks (y already weighted, from moe_down1.py). A pick with id < 0
is skipped by pointing its loads out of range (they read 0).

One thread per (token, 8 rows): 6 id loads, 12 dwordx4 y loads, one 16 B store.
add=True (part, the MoE's output before its all-reduce): out =
bf16(bf16(Σ y) + add), add bf16 [M, 5120] (the shared expert), as vLLM's bf16
``routed + shared``.
"""

from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import KArgs, kernel_source, rsrc

H = 5120
KA = KArgs(
    ("y", "ptr"),
    ("ids", "ptr"),
    ("out", "ptr"),
    ("m", "u32"),
    ("y_bytes", "u32"),
    ("ids_bytes", "u32"),
    ("out_bytes", "u32"),
)
KA_ADD = KArgs(*KA.args, ("add", "ptr"), ("add_bytes", "u32"))


def source(name: str | None = None, add: bool = False, y_bf16: bool = False):
    """y_bf16: y bf16 [P, 5120] (moe_down1 y_bf16), one dwordx4
    per pick, widened to f32 and summed in the same order."""
    name = name or ("moe_combine_add" if add else "moe_combine") + (
        "_yb" if y_bf16 else ""
    )
    yb = 2 if y_bf16 else 4
    ka = KA_ADD if add else KA
    c = []
    A = c.append
    A("  s_load_dwordx8 s[4:11], s[0:1], 0x0")  # y, ids, out, m, y_bytes
    A("  s_load_dwordx2 s[12:13], s[0:1], 0x20")  # ids_bytes, out_bytes
    if add:
        A(f"  s_load_dwordx2 s[36:37], s[0:1], {ka.off['add']:#x}")
        A(f"  s_load_dword s38, s[0:1], {ka.off['add_bytes']:#x}")
    A("  s_waitcnt lgkmcnt(0)")
    c += rsrc(16, 4, nbytes_sgpr=11)
    c += rsrc(20, 6, nbytes_sgpr=12)
    c += rsrc(24, 8, nbytes_sgpr=13)
    if add:
        c += rsrc(40, 36, nbytes_sgpr=38)
    A("  s_lshl_b32 s28, s2, 8")
    A("  v_add_u32 v1, s28, v0")  # gid
    A(f"  s_mul_i32 s29, s10, {H // 8}")  # total threads
    A("  v_cmp_gt_u32 vcc, s29, v1")
    A("  s_and_saveexec_b64 s[30:31], vcc")
    A("  s_cbranch_execz .Lend")
    A("  s_mov_b32 s32, 0xcccccccd")
    A("  v_lshrrev_b32 v2, 7, v1")
    A("  v_mul_hi_u32 v2, v2, s32")
    A("  v_lshrrev_b32 v2, 2, v2")  # token = gid / 640
    A(f"  v_mul_u32_u24 v3, {H // 8}, v2")
    A("  v_sub_u32 v3, v1, v3")  # chunk
    A("  v_mul_u32_u24 v4, 24, v2")  # ids offset token*6*4
    for k in range(6):
        A(f"  buffer_load_dword v{10 + k}, v4, s[20:23], 0 offen offset:{4 * k}")
    A("  v_mov_b32 v5, 0x7fff0000")
    A(f"  v_mul_u32_u24 v6, {6 * H * yb}, v2")  # y row of pick token*6
    A(f"  v_lshl_add_u32 v6, v3, {4 if y_bf16 else 5}, v6")  # + chunk * 32 B (16 B)
    A("  s_waitcnt vmcnt(0)")
    for k in range(6):
        A(f"  v_add_u32 v7, {k * H * yb}, v6")
        A(f"  v_cmp_le_i32 vcc, 0, v{10 + k}")
        A(f"  v_cndmask_b32 v{16 + k}, v5, v7, vcc")
    A(f"  v_mul_u32_u24 v8, {H * 2}, v2")
    A("  v_lshl_add_u32 v8, v3, 4, v8")  # out / add offset
    if add:
        A("  buffer_load_dwordx4 v[76:79], v8, s[40:43], 0 offen")
    for k in range(6):
        b = 24 + 8 * k
        if y_bf16:
            A(f"  buffer_load_dwordx4 v[{b + 4}:{b + 7}], v{16 + k}, s[16:19], 0 offen")
        else:
            A(f"  buffer_load_dwordx4 v[{b}:{b + 3}], v{16 + k}, s[16:19], 0 offen")
            A(
                f"  buffer_load_dwordx4 v[{b + 4}:{b + 7}], v{16 + k}, s[16:19], 0 offen offset:16"
            )
    A("  s_waitcnt vmcnt(0)")
    if y_bf16:  # bf16 pairs → f32: element 2q = d << 16, 2q + 1 = d & 0xffff0000
        for k in range(6):
            b = 24 + 8 * k
            for q in range(4):
                A(f"  v_lshlrev_b32 v{b + 2 * q}, 16, v{b + 4 + q}")
                A(f"  v_and_b32 v{b + 2 * q + 1}, 0xffff0000, v{b + 4 + q}")
    for k in range(1, 6):
        for i in range(8):
            A(f"  v_add_f32 v{24 + i}, v{24 + i}, v{24 + 8 * k + i}")
    for q in range(4):
        A(f"  v_cvt_pk_bf16_f32 v{72 + q}, v{24 + 2 * q}, v{25 + 2 * q}")
    if add:  # bf16 pairs → f32 (lo = d << 16, hi = d & 0xffff0000), add, round again
        for q in range(4):
            A(f"  v_lshlrev_b32 v80, 16, v{72 + q}")
            A(f"  v_and_b32 v81, 0xffff0000, v{72 + q}")
            A(f"  v_lshlrev_b32 v82, 16, v{76 + q}")
            A(f"  v_and_b32 v83, 0xffff0000, v{76 + q}")
            A("  v_add_f32 v80, v80, v82")
            A("  v_add_f32 v81, v81, v83")
            A(f"  v_cvt_pk_bf16_f32 v{72 + q}, v80, v81")
    A("  buffer_store_dwordx4 v[72:75], v8, s[24:27], 0 offen")
    A(".Lend:")
    A("  s_endpgm")
    src = kernel_source(
        name,
        c,
        ka,
        lds=0,
        vgprs=88 if add else 80,
        agprs=0,
        sgprs=48 if add else 40,
        block=256,
    )
    return dict(name=name, src=src, ka=ka)
