# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm#65421 at 7ba343d8 (Apache-2.0):
# vllm/models/deepseek_v41/amd/asm/kernels/k2.py (changed: import paths)
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""K2 skeleton: the FFN half as one persistent kernel,
256 CTAs × 256 threads, the step-3 phase kernels as bodies separated by grid
barriers.

- **Bodies:** each phase kernel's code, unchanged except that labels get the
  phase tag, ``s_endpgm`` branches to the phase exit and every store gets
  ``sc1`` (a barrier doesn't publish plain stores across XCDs). Before a body
  the skeleton points s[0:1] at that phase's block of the combined kernargs
  (laid out exactly as the phase's own KArgs), sets s2 = CTA index and v0 =
  thread id, EXEC = all. Phases with fewer CTAs skip the rest; the combine
  (grid 2.5 M) loops over virtual CTAs b, b + 256, ...
- **Reserved:** s96 barrier index (leader wave), s97 CTA, s[98:99] kernarg
  base, s100-s101 virtual-CTA loop, v255 thread id. Bodies use ≤ s95, ≤ v247.
- **Grid barrier** (common/pk_barrier.py's hierarchical one): all waves wait
  for their stores and ``s_barrier``; wave 0 adds to its XCD group's counter
  (CTA b runs on XCD b % 8); the group's last arriver adds to the global
  counter; the last global arriver writes the barrier index to the 8 release
  lines (sc1); wave 0 polls its group's line, then ``s_barrier``.
- **Coherence** (``coh``): "plain" (default): plain loads, sc1
  stores, nothing invalidated. Coherent because (a) the dispatch acquire
  invalidates every XCD's L2 at kernel start (the chain relies on the same),
  (b) every scratch buffer is written in one phase and only read in later
  ones, so no XCD can hold a line of it from before the write, and (c) sc1
  stores write through to memory. Only the barrier words are read before
  they are written; they keep their sc1 polls. Bitwise equal to the chain on
  layers 3/5/22/30, M 64-256, 5 launches each; stage 1's slowest CTA 8 µs
  faster than with "loads" at M 256.
  "loads": loads of data written earlier
  in the launch (SC1_LOADS: per body, the buffer-resource / base SGPRs) get
  ``sc1``, and moe_ug1 / moe_down1 read ``counts`` with a vector sc1 load;
  nothing is invalidated, so code, weights and inputs keep their L2 lines.
  "inv": plain loads, ``buffer_inv sc1`` after each release (first version:
  ~1.5 µs per barrier). "sc0": timing experiment only.
  No ``s_dcache_inv``: it made every phase's kernarg s_loads miss (~2 µs per
  phase). The only scalar load of data written in the kernel is moe_ug1 /
  moe_down1 reading ``counts``, which nothing reads before P4 writes it, and
  the kernel-start acquire clears the scalar cache of earlier launches.
- **No per-launch memset:** counters and release lines only grow. At start
  wave 0 reads its release line, R0 = the last completed barrier; barrier k
  waits for counters (R0 + k)·32 / ·8 and release value R0 + k (equality and
  signed differences, so u32 wrap is fine). Every CTA reads R0 before it
  arrives anywhere, so nothing races it. The buffer is zeroed once at
  allocation (and must be re-zeroed after a timed-out launch).
- **Timeline:** wave 0 stores s_memrealtime (100 MHz, low dword) per CTA:
  slot 0 start, 2k-1 all waves done with phase k, 2k barrier k released,
  2 NB + 1 end; slot 31 = 0x80000000 | k if barrier k timed out.
- **Knock-outs:** bit i of ``knock`` skips body i (in BODIES order); the
  barriers still run.
"""

import regex as re

from sgl_amd_mono.deepseek_v41.asm.kernels import ffn_route as fr
from sgl_amd_mono.deepseek_v41.asm.kernels import (
    ffn_seam,
    moe_combine,
    moe_down1,
    moe_prep,
    moe_ug1,
)
from sgl_amd_mono.deepseek_v41.asm.kernels.kmod import KArgs, kernel_source

NCU, GRP = 256, 8
PER_GRP = NCU // GRP
SPIN = 1 << 22
TL_SLOTS = 32
BAR_BYTES = 8192  # global counter @0, group g's counter @256 + 256 g, release line @4352 + 256 g
SKEL = [("bar", "ptr"), ("tl", "ptr"), ("m", "u32"), ("knock", "u32")]
OFF_BAR, OFF_TL, OFF_M, OFF_KNOCK = 0, 8, 16, 20


# per body: base SGPRs of the buffer resources (or global_load SGPR pairs)
# that point at data written earlier in the same launch
SC1_LOADS = {
    "fin": {32, 36},
    "rt": {20},
    "sgu": {28, 32},
    "p3": {24, 44},
    "prep": {28},
    "ug": {32, 36, 78},
    "dn": {32, 36, 40, 78},
    "sdn": {28, 32},
    "comb": {16, 20, 40},
}


LAYOUTS = ("base", "sdn_prep", "split", "cost", "sdn_prep+cost")


# per body: stores of outputs nothing in the kernel reads (R_new, the mixes, the
# logits, part): the kernel-end release publishes them, so they skip sc1 ("pst")
PLAIN_STORES = {"seam": {56}, "fin": {48, 52, 56}, "p3": {32}, "comb": {24}}


def phases(layout: str = "base"):
    """Phases in order; a barrier follows each but the last. Bodies within a
    phase run back to back in each CTA. Grid: an int, "m" or "comb" (virtual
    CTAs over all 256), or (virtual grid, first CTA, CTA count): CTAs first ..
    first + count - 1 loop over virtual CTAs v, v + count, ...
    layout: "base" or options joined by "+":
    - "sdn_prep": the shared down in P4 on CTAs 1-160 (P4's sort uses CTA 0).
    - "split": router on CTAs 0-127, shared gate_up on 128-255 (slower).
    - "cost": P4 splits stage 2 by cost (moe_prep dsplit), stage 2 reads it.
    - "pst" (source()): plain stores for PLAIN_STORES.
    - "part": the combine adds the shared expert's output (moe_combine add):
      ``out`` = bf16(routed + shared), the MoE's output before its all-reduce.
    - "inv_ug" / "inv_dn" (source()): ``buffer_inv sc1`` after the barrier
      before stage 1 / 2 and plain loads in that stage (coh "loads" only).
    - "kpre" (source()): every wave loads the whole kernarg block once in the
      prologue, so each body's kernarg s_loads hit the scalar cache."""
    opts = set(layout.split("+")) - {"base"}
    assert opts <= {
        "sdn_prep",
        "split",
        "cost",
        "inv_ug",
        "inv_dn",
        "part",
        "pst",
        "kpre",
    }, layout
    sdn = ("sdn", fr.source_shared_down(), fr.D_GRID)
    p2 = (
        [
            ("rt", fr.source_router(), (fr.R_GRID, 0, 128)),
            ("sgu", fr.source_shared(), (fr.S_GRID, 128, 128)),
        ]
        if "split" in opts
        else [("rt", fr.source_router(), 256), ("sgu", fr.source_shared(), fr.S_GRID)]
    )
    cost = "cost" in opts
    return [
        [("seam", ffn_seam.source(), 256)],
        [("fin", ffn_seam.source_fin(), "m")],
        p2,
        [("p3", fr.source_p3(), "m")],
        [("prep", moe_prep.source(dsplit=cost), 1)]
        + ([(sdn[0], sdn[1], (fr.D_GRID, 1, fr.D_GRID))] if "sdn_prep" in opts else []),
        [("ug", moe_ug1.source(counts_vec=True), 256)],
        [
            (
                "dn",
                moe_down1.source(counts_vec=True, split="table" if cost else "ceil"),
                256,
            )
        ]
        + ([] if "sdn_prep" in opts else [sdn]),
        [("comb", moe_combine.source(add="part" in opts), "comb")],
    ]


BODIES = [
    tag for ph in phases() for tag, _, _ in ph
]  # "base" order (knock bits follow the layout)
NB = len(phases()) - 1


def body_of(
    k: dict, tag: str, sc1_bases: set = frozenset(), plain_stores: set = frozenset()
):
    """The kernel's code with tagged labels, s_endpgm → exit branch, sc1
    stores (except through the plain_stores base SGPRs), sc1 on loads through
    the given base SGPRs; and its VGPR / AGPR / LDS use."""
    src, name = k["src"], k["name"]
    body = src.split(f"\n{name}:\n", 1)[1].split("\n.Lfunc_end:", 1)[0]
    out = []
    for ln in body.split("\n"):
        ln = re.sub(r"\.L(\w+)", lambda mt: f".L{tag}_{mt.group(1)}", ln)
        if ln.strip() == "s_endpgm":
            ln = f"  s_branch .L{tag}_exit"
        elif re.match(r"\s*(buffer|global)_store", ln) and "sc1" not in ln:
            b = re.search(r"s\[(\d+):\d+\]", ln)
            if not (b and int(b.group(1)) in plain_stores):
                ln += " sc1"
        elif re.match(r"\s*(buffer|global)_load", ln) and "sc1" not in ln:
            b = re.search(r"s\[(\d+):\d+\]", ln)
            if b and int(b.group(1)) in sc1_bases:
                ln += " sc1"
        out.append(ln)
    acc = int(re.search(r"\.amdhsa_accum_offset (\d+)", src).group(1))
    tot = int(re.search(r"\.amdhsa_next_free_vgpr (\d+)", src).group(1))
    lds = int(re.search(r"\.amdhsa_group_segment_fixed_size (\d+)", src).group(1))
    sg = int(re.search(r"\.amdhsa_next_free_sgpr (\d+)", src).group(1))
    assert acc <= 255 and sg <= 96, (tag, acc, sg)
    return out, acc, tot - acc, lds


def stamp(slot: int) -> list[str]:
    """Wave 0, EXEC = lane 0: tl[CTA][slot] = s_memrealtime (low dword)."""
    return [
        f"  s_load_dwordx2 s[6:7], s[98:99], {OFF_TL:#x}",
        "  s_memrealtime s[8:9]",
        "  s_waitcnt lgkmcnt(0)",
        f"  s_mul_i32 s10, s97, {TL_SLOTS * 4}",
        f"  s_add_u32 s10, s10, {slot * 4}",
        "  v_mov_b32 v1, s10",
        "  v_mov_b32 v2, s8",
        "  global_store_dword v1, v2, s[6:7]",
    ]


def barrier(k: int, inv: str = "sc1") -> list[str]:
    t = f".Lbar{k}"
    L = [
        "  s_mov_b64 exec, -1",
        "  s_waitcnt vmcnt(0) lgkmcnt(0)",
        "  s_barrier",
        "  v_readfirstlane_b32 s4, v255",
        "  s_lshr_b32 s4, s4, 6",
        "  s_cmp_eq_u32 s4, 0",
        f"  s_cbranch_scc0 {t}_follow",
        "  s_mov_b64 exec, 1",
    ]
    L += stamp(2 * k - 1)
    L += [
        f"  s_load_dwordx2 s[4:5], s[98:99], {OFF_BAR:#x}",
        "  s_add_u32 s96, s96, 1",
        f"  s_and_b32 s8, s97, {GRP - 1}",
        "  s_lshl_b32 s8, s8, 8",
        "  s_waitcnt lgkmcnt(0)",
        "  s_add_u32 s9, s8, 256",
        "  v_mov_b32 v1, s9",
        "  v_mov_b32 v2, 1",
        "  global_atomic_add v3, v1, v2, s[4:5] sc0 sc1",
        "  s_waitcnt vmcnt(0)",
        "  v_readfirstlane_b32 s10, v3",
        "  s_add_u32 s10, s10, 1",
        f"  s_mul_i32 s11, s96, {PER_GRP}",
        "  s_cmp_eq_u32 s10, s11",
        f"  s_cbranch_scc0 {t}_wait",
        "  v_mov_b32 v1, 0",
        "  global_atomic_add v3, v1, v2, s[4:5] sc0 sc1",
        "  s_waitcnt vmcnt(0)",
        "  v_readfirstlane_b32 s10, v3",
        "  s_add_u32 s10, s10, 1",
        f"  s_mul_i32 s11, s96, {GRP}",
        "  s_cmp_eq_u32 s10, s11",
        f"  s_cbranch_scc0 {t}_wait",
        "  v_mov_b32 v4, s96",
    ]
    for g in range(GRP):
        L += [
            f"  v_mov_b32 v1, {4352 + 256 * g}",
            "  global_store_dword v1, v4, s[4:5] sc1",
        ]
    L += [
        "  s_waitcnt vmcnt(0)",
        f"{t}_wait:",
        "  s_add_u32 s9, s8, 4352",
        "  v_mov_b32 v1, s9",
        f"  s_mov_b32 s12, {SPIN}",
        f"{t}_poll:",
        "  global_load_dword v3, v1, s[4:5] sc1",
        "  s_waitcnt vmcnt(0)",
        "  v_readfirstlane_b32 s10, v3",
        "  s_sub_u32 s10, s10, s96",
        "  s_cmp_ge_i32 s10, 0",
        f"  s_cbranch_scc1 {t}_done",
        "  s_sub_u32 s12, s12, 1",
        "  s_cmp_eq_u32 s12, 0",
        f"  s_cbranch_scc0 {t}_poll",
        f"  s_load_dwordx2 s[6:7], s[98:99], {OFF_TL:#x}",
        f"  s_mul_i32 s10, s97, {TL_SLOTS * 4}",
        f"  s_add_u32 s10, s10, {(TL_SLOTS - 1) * 4}",
        "  v_mov_b32 v1, s10",
        f"  v_mov_b32 v2, {0x80000000 | k}",
        "  s_waitcnt lgkmcnt(0)",
        "  global_store_dword v1, v2, s[6:7]",
        f"{t}_done:",
    ]
    L += stamp(2 * k)
    L += ([f"  buffer_inv {inv}"] if inv else []) + [
        "  s_waitcnt vmcnt(0)",
        "  s_mov_b64 exec, -1",
        f"{t}_follow:",
        "  s_barrier",
    ]
    return L


def run_body(tag: str, body: list[str], grid, base: int, bit: int) -> list[str]:
    def enter(cta: str):
        return [
            f"  s_add_u32 s0, s98, {base}",
            "  s_addc_u32 s1, s99, 0",
            f"  s_mov_b32 s2, {cta}",
            "  v_mov_b32 v0, v255",
            "  s_mov_b64 exec, -1",
            "  s_nop 1",
        ]  # readfirstlane of v0 needs a wait state

    leave = [
        f".L{tag}_exit:",
        "  s_mov_b64 exec, -1",
        "  s_waitcnt vmcnt(0) lgkmcnt(0)",
    ]
    L = [
        f"  s_load_dwordx2 s[4:5], s[98:99], {OFF_M:#x}",
        "  s_waitcnt lgkmcnt(0)",  # s4 = m, s5 = knock
        f"  s_bitcmp1_b32 s5, {bit}",
        f"  s_cbranch_scc1 .L{tag}_skip",
    ]
    nv, first, cnt = grid if isinstance(grid, tuple) else (grid, 0, NCU)
    if nv == "comb":
        L += [
            "  s_mul_i32 s101, s4, 640",
            "  s_add_u32 s101, s101, 255",
            "  s_lshr_b32 s101, s101, 8",
        ]
    elif nv == "m":
        L += ["  s_mov_b32 s101, s4"]
    else:
        L += [f"  s_mov_b32 s101, {nv}"]
    if first or cnt < NCU:
        L += [
            f"  s_sub_u32 s100, s97, {first}",
            f"  s_cmp_ge_u32 s100, {cnt}",
            f"  s_cbranch_scc1 .L{tag}_skip",
        ]
    else:
        L += ["  s_mov_b32 s100, s97"]
    L += [
        f".L{tag}_virt:",
        "  s_cmp_ge_u32 s100, s101",
        f"  s_cbranch_scc1 .L{tag}_skip",
    ]
    L += (
        enter("s100")
        + body
        + leave
        + [f"  s_add_u32 s100, s100, {cnt}", f"  s_branch .L{tag}_virt"]
    )
    return L + [f".L{tag}_skip:"]


def source(name: str = "ffn_k2", coh: str = "plain", layout: str = "base"):
    """coh: "plain" | "loads" | "inv" | "sc0" (module docstring); layout: see phases()."""
    coh = {"none": "plain"}.get(coh, coh)
    inv = {"plain": "", "loads": "", "inv": "sc1", "sc0": "sc0"}[coh]
    phs = phases(layout)
    opts = set(layout.split("+"))
    inv_before = {tag for tag in ("ug", "dn") if f"inv_{tag}" in opts}
    args = list(SKEL)
    for ph in phs:
        for tag, k, _ in ph:
            args += [(f"{tag}.{n}", kind) for n, kind in k["ka"].args]
    ka = KArgs(*args)
    c = []
    A = c.append
    # prologue: keep kernarg base, CTA, thread id; wave 0 reads R0
    A("  s_mov_b64 s[98:99], s[0:1]")
    A("  s_mov_b32 s97, s2")
    A("  v_mov_b32 v255, v0")
    if "kpre" in opts:
        for off in range(0, ka.size, 64):
            A(f"  s_load_dwordx16 s[4:19], s[0:1], {off:#x}")
        A("  s_waitcnt lgkmcnt(0)")
    A("  v_readfirstlane_b32 s4, v0")
    A("  s_lshr_b32 s4, s4, 6")
    A("  s_cmp_eq_u32 s4, 0")
    A("  s_cbranch_scc0 .Lstarted")
    A("  s_mov_b64 exec, 1")
    A(f"  s_load_dwordx2 s[4:5], s[98:99], {OFF_BAR:#x}")
    A(f"  s_and_b32 s8, s97, {GRP - 1}")
    A("  s_lshl_b32 s8, s8, 8")
    A("  s_add_u32 s8, s8, 4352")
    A("  v_mov_b32 v1, s8")
    A("  s_waitcnt lgkmcnt(0)")
    A("  global_load_dword v3, v1, s[4:5] sc1")
    A("  s_waitcnt vmcnt(0)")
    A("  v_readfirstlane_b32 s96, v3")
    c += stamp(0)
    A("  s_mov_b64 exec, -1")
    A(".Lstarted:")
    vg = ag = lds = 0
    bit = 0
    for i, ph in enumerate(phs):
        for tag, k, grid in ph:
            sc1 = (
                SC1_LOADS.get(tag, set())
                if coh == "loads" and tag not in inv_before
                else set()
            )
            body, v, a, l = body_of(  # noqa: E741
                k, tag, sc1, PLAIN_STORES.get(tag, set()) if "pst" in opts else set()
            )
            vg, ag, lds = max(vg, v), max(ag, a), max(lds, l)
            base = ka.off[f"{tag}.{k['ka'].args[0][0]}"]
            assert all(
                ka.off[f"{tag}.{n}"] - base == k["ka"].off[n] for n, _ in k["ka"].args
            ), tag
            c += run_body(tag, body, grid, base, bit)
            bit += 1
        if i < len(phs) - 1:
            nxt = {tag for tag, _, _ in phs[i + 1]}
            c += barrier(i + 1, "sc1" if coh == "loads" and nxt & inv_before else inv)
    A("  s_mov_b64 exec, -1")
    A("  v_readfirstlane_b32 s4, v255")
    A("  s_lshr_b32 s4, s4, 6")
    A("  s_cmp_eq_u32 s4, 0")
    A("  s_cbranch_scc0 .Lend")
    A("  s_mov_b64 exec, 1")
    c += stamp(2 * NB + 1)
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    src = kernel_source(
        name, c, ka, lds=lds, vgprs=256, agprs=(ag + 3) // 4 * 4, sgprs=102, block=256
    )
    return dict(name=name, src=src, ka=ka, lds=lds, vgprs=vg, agprs=ag)
