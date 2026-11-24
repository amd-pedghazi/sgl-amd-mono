# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 4cd75a2
# (ROCm/ATOM#2559), MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v4/mono/op.py and atom/models/deepseek_v4/mono/runner.py
"""One rank's DeepSeek-V4 launches on gfx950: the packed weights, each width's kernels
and buffers, and a step's launches over SGLang's caches.

Every decoder layer is one launch (``kernels/layer.py``). The layers of one attention
variant (compress ratio) share a width's compiled kernel, scratch and peer buffer;
their tags keep them apart: word 0 of the epoch buffer is the step, which the host
moves on after a step's last launch, and a layer's tag is
``step * LAYER_SLOTS + layer + 1``. Words 1-5 are the guard block (``common/guard.py``).

The kernel reads and writes SGLang's caches in place, so a request can take either
path from one step to the next: the two-pool fp8 unified KV (the SWA ring rows, then
each 256-token page's compressed rows), the compressor and indexer states (a ring a
request slot, ``[kv | score]`` rows) and the FP4 index keys. The weights are packed
at construction, before SGLang sizes the KV cache; the caches bind at the first
forward (``bind``), which reads only pointers and shapes.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.deepseek_v4.gfx950.config import BLOCK_TOKENS
from sgl_amd_mono.deepseek_v4.gfx950.kernels.layer import (
    build_dsv4_kernel,
    build_scrub,
)
from sgl_amd_mono.deepseek_v4.gfx950.layout import layout

# The SGLang names the runner reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "get_unified_kv",
        "get_unified_kv_rope",
        "get_attention_compress_states",
        "get_indexer_compress_states",
        "get_index_k_fp4_payload_buffer",
        "get_index_k_fp4_scale_buffer",
        "unified_swa_ring_size",
        "unified_swa_pages",
        "page_size",
    ),
    "srt/mem_cache/deepseek_v4_compress_state.py": (
        "kv_score_buffer",
        "kv_score",
        "ring_size",
        "online",
    ),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "forward_metadata",
        "core_attn_metadata",
        "positions_casual",
        "page_table",
    ),
    "srt/model_executor/forward_batch_info.py": ("req_pool_indices",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
}


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds and binds run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"DeepSeek-V4 monokernels: {what} inside a graph capture")


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    fn = getattr(launch, "func", None) or launch._func
    names = [n for n in inspect.signature(fn).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def _ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def shape_dims(cfg) -> dict:
    """The shape arguments layout() and build_dsv4_kernel() must share identically."""
    return {
        "hidden": cfg.hidden,
        "q_lora": cfg.q_lora,
        "head_dim": cfg.head_dim,
        "o_groups": cfg.o_groups,
        "o_lora": cfg.o_lora,
        "hc_mult": cfg.hc_mult,
        "compress_ratio": cfg.compress_ratio,
        "n_keys": cfg.n_keys,
        "c_coff": cfg.c_coff,
        # 0: no indexer (only CSA has one)
        "index_head_dim": cfg.index_head_dim if cfg.indexed else 0,
        "index_heads": cfg.index_heads if cfg.indexed else 0,
        "max_seq": cfg.max_seq,
        "index_topk": cfg.index_topk if cfg.indexed else 0,
        "indexer_hadamard": cfg.indexer_hadamard,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "inter": cfg.inter,
    }


class Variant:
    """One attention variant's launch at one width: the compiled kernel, its scratch,
    its peer buffer (the data regions, then the guard's error flags) and the scrub."""

    def __init__(self, cfg, *, S, rank, npes, group, spec_steps, c_ring, fault, device):
        dims = shape_dims(cfg)
        self.scr, self.sym = layout(S, cfg.heads, npes, cfg.window, **dims)
        self.scratch = torch.zeros(self.scr["_bytes"], dtype=torch.uint8, device=device)
        # collective over ``group`` (a CPU group): every rank builds its variants alike
        self.peer = PeerBuffer(
            self.sym["_bytes"] + guard.FLAG_BYTES, group, rank, npes, device
        )
        # the tags start from 1: an unwritten pair must read tag 0
        self.peer.bytes.zero_()
        self.launch = build_dsv4_kernel(
            S,
            cfg.heads,
            npes,
            cfg.window,
            scale=cfg.softmax_scale,
            spec_steps=spec_steps,
            swiglu_limit=cfg.swiglu_limit,
            hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            window_rows=cfg.cache_rows,
            c_ring=c_ring,
            fault_site=fault[0],
            fault_after=fault[1],
            **dims,
        )
        # the step's scrub visits mailbox pairs: every data region is pairs
        self.scrub = build_scrub(self.scr["_bytes"] // 8, self.sym["_bytes"] // 8)


@dataclass
class Layer:
    """One layer's weights and bound caches. ``t``: the packed matrices and the small
    tensors the kernel reads as the checkpoint stores them."""

    i: int
    cfg: Any  # V4Config
    t: dict
    hc_sb: dict  # side -> [3 scales | hc_mix bases] f32
    cos: torch.Tensor  # f32 [positions, rope_dim / 2]
    sin: torch.Tensor
    use_hash: bool
    # bound at the first forward (``Dsv4MonoRunner.bind``): addresses and strides
    caches: dict = field(default_factory=dict)

    @property
    def ratio(self) -> int:
        return self.cfg.compress_ratio


def step_rows(pos, req, block_tables, *, ring, swa_pages, window, variants):
    """The unified KV rows of a step's rows, as SGLang's pool places them: the SWA row
    each row writes (``req * ring + pos % ring``) and, per compress ratio, the kernel's
    ``dest_rows`` and index list. ``variants``: ratio -> (n_keys, indexed).

    The index list is the window's rows, oldest first (the row's own token last), then
    for HCA every compressed entry so far (``swa_pages + page * K_PB + e % K_PB``, the
    kernel's ``comp_row``); -1 elsewhere. CSA's compressed half is the indexer's pick,
    which the kernel makes itself. ``pos``, ``req``: int64 [rows]; ``block_tables``:
    the 256-token pages [rows, pages]."""
    rows, dev = pos.numel(), pos.device
    swa_row = (req * ring + pos % ring).to(torch.int32)
    base = torch.full_like(swa_row, swa_pages)
    q = pos[:, None] - (window - 1) + torch.arange(window, device=dev)[None]
    win = torch.where(q >= 0, req[:, None] * ring + q.clamp(min=0) % ring, -1)
    dest, indices = {}, {}
    for r, (n_keys, indexed) in variants.items():
        dest[r] = torch.stack([swa_row, base])
        tail = n_keys - window
        if r and not indexed:
            k_pb = BLOCK_TOKENS // r
            e = torch.arange(tail, device=dev)
            cols = (e // k_pb).clamp(max=block_tables.shape[1] - 1)
            page = block_tables.gather(1, cols[None].expand(rows, -1))
            comp = swa_pages + page.to(torch.int64) * k_pb + e % k_pb
            rest = torch.where(e[None] < ((pos + 1) // r)[:, None], comp, -1)
        else:
            rest = torch.full((rows, tail), -1, dtype=torch.int64, device=dev)
        indices[r] = torch.cat([win, rest], 1).to(torch.int32).contiguous()
    return swa_row, dest, indices


@dataclass
class StepInputs:
    """A step's per-row inputs, shared by every layer of a compress ratio."""

    rows: int
    pos: torch.Tensor  # int32 [rows]
    tokens: torch.Tensor  # int32 [rows]
    slots: torch.Tensor  # int32 [rows]: the request slot (state ring, SWA ring)
    run: torch.Tensor  # int32 [rows]: the row's index among its request's rows
    block_tables: torch.Tensor  # int32 [rows, pages]: the 256-token pages
    dest: dict  # ratio -> int32 [2, rows]: (SWA row written, compressed rows' base)
    indices: dict  # ratio -> int32 [rows, n_keys]: the window's rows, then HCA's


class Dsv4MonoRunner:
    """The packed layers and, per launch width, the compiled kernels and buffers of one
    TP rank. Width builds are collective (the peer handshakes): every rank builds the
    same widths in the same order."""

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        cpu_group,
        device: torch.device,
        layers: list,
        spec_steps: int,
        c_rings: dict,
        tokens_per_request: int,
    ) -> None:
        _ensure_writable_flydsl_cache()
        self.tp, self.rank, self.group, self.device = tp, rank, cpu_group, device
        self.layers = layers
        self.spec_steps = spec_steps
        self.c_rings = c_rings  # compress ratio -> SGLang's state ring rows
        self.tokens_per_request = tokens_per_request
        self.fault = guard.fault_injection(rank) or (0, 0)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        self.widths: dict[int, dict[int, Variant]] = {}
        self.pool = None
        self.swa_ring = self.swa_pages = 0

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, as the guard block's clock ticks."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def build_width(self, S: int) -> None:
        """The S-row launch: a variant per compress ratio."""
        no_capture(f"the {S}-row launch built")
        variants: dict[int, Variant] = {}
        for L in self.layers:
            if L.ratio not in variants:
                variants[L.ratio] = Variant(
                    L.cfg,
                    S=S,
                    rank=self.rank,
                    npes=self.tp,
                    group=self.group,
                    spec_steps=self.spec_steps,
                    c_ring=self.c_rings[L.ratio],
                    fault=self.fault,
                    device=self.device,
                )
        # every peer buffer zeroed before any rank's first push lands on it
        torch.cuda.synchronize()
        self.widths[S] = variants

    def precompile(self, S: int) -> None:
        for v in self.widths[S].values():
            compile_only(v.launch)
            compile_only(v.scrub)

    # ------------------------------------------------------------------ caches
    def cache_reasons(self, pool) -> list[str]:
        """Reasons SGLang's caches are not the ones the kernel addresses (checked at
        the first forward, after SGLang sized them)."""
        why = []
        if getattr(pool, "unified_kv_pool", None) is None:
            return ["the KV pool is not the unified one (unified_kv_triton)"]
        if not getattr(pool, "_unified_kv_fp8", False):
            why.append("the unified KV pool is bf16, not the two-pool fp8 layout")
        if pool.page_size != BLOCK_TOKENS:
            why.append(f"page size {pool.page_size}, the kernel's is {BLOCK_TOKENS}")
        for L in self.layers:
            kv = pool.get_unified_kv(L.i)
            rope = pool.get_unified_kv_rope(L.i)
            if kv.shape[-1] != 512 or kv.element_size() != 1 or rope.shape[-1] != 64:
                why.append(
                    f"layer {L.i}: KV pools {tuple(kv.shape)} / {tuple(rope.shape)}"
                )
                break
            for kind, get, width in (
                ("compressor", pool.get_attention_compress_states, L.cfg.head_dim),
                ("indexer", pool.get_indexer_compress_states, L.cfg.index_head_dim),
            ):
                if kind == "indexer" and not L.cfg.indexed:
                    continue
                st = get(L.i)
                buf = st.kv_score_buffer.kv_score
                want = 2 * L.cfg.c_coff * width
                if st.online or buf.dtype != torch.float32 or buf.shape[-1] != want:
                    why.append(
                        f"layer {L.i}: the {kind} state is {buf.dtype} "
                        f"{tuple(buf.shape)}, the kernel's rows are f32 [kv | score] "
                        f"of {want}"
                    )
                if st.ring_size != self.c_rings[L.ratio]:
                    why.append(
                        f"layer {L.i}: a {st.ring_size}-row {kind} state ring, the "
                        f"kernel's is {self.c_rings[L.ratio]}"
                    )
            if L.cfg.indexed:
                pay = pool.get_index_k_fp4_payload_buffer(L.i)
                sc = pool.get_index_k_fp4_scale_buffer(L.i)
                if tuple(pay.shape[1:]) != (1, 4, 64, 16) or tuple(sc.shape[1:]) != (
                    1,
                    4,
                    64,
                ):
                    why.append(
                        f"layer {L.i}: FP4 index pages {tuple(pay.shape)} / "
                        f"{tuple(sc.shape)}, the kernel's hold 64 entries"
                    )
            if why:
                break
        return why

    def bind(self, pool) -> None:
        """Each layer's cache addresses and strides (pointers only: no GPU work)."""
        self.swa_ring = int(pool.unified_swa_ring_size)
        self.swa_pages = int(pool.unified_swa_pages)
        for L in self.layers:
            c = {
                "kv": _ptr(pool.get_unified_kv(L.i)),
                "kv_rope": _ptr(pool.get_unified_kv_rope(L.i)),
                "kv_state": 0,
                "score_state": 0,
                "st_kv": 0,
                "i_kv_state": 0,
                "i_score_state": 0,
                "st_i": 0,
                "i_cache": 0,
                "i_cache_s": 0,
                # a page's compressed rows
                "env_rows": BLOCK_TOKENS // L.ratio,
            }
            if L.ratio:
                st = pool.get_attention_compress_states(L.i)
                cw = L.cfg.c_coff * L.cfg.head_dim
                c["kv_state"] = _ptr(st.kv_score_buffer.kv_score)
                c["score_state"] = c["kv_state"] + cw * 4
                c["st_kv"] = st.ring_size * 2 * cw  # floats a request slot
            if L.cfg.indexed:
                st = pool.get_indexer_compress_states(L.i)
                iw = L.cfg.c_coff * L.cfg.index_head_dim
                c["i_kv_state"] = _ptr(st.kv_score_buffer.kv_score)
                c["i_score_state"] = c["i_kv_state"] + iw * 4
                c["st_i"] = st.ring_size * 2 * iw
                c["i_cache"] = _ptr(pool.get_index_k_fp4_payload_buffer(L.i))
                c["i_cache_s"] = _ptr(pool.get_index_k_fp4_scale_buffer(L.i))
            L.caches = c
        self.pool = pool

    # ------------------------------------------------------------------ a step
    def step_inputs(self, forward_batch, input_ids, rows: int) -> StepInputs:
        """The step's rows from SGLang's replay-fresh decode / verify metadata (inside
        the graph where there is one). A pad row is SGLang's: request slot 0, position
        0 (the reserved dummy slot, which the normal path writes too)."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        md = get_attn_backend().forward_metadata.core_attn_metadata
        dev = self.device
        pos = md.positions_casual[:rows].to(torch.int32)
        bt = md.page_table[:rows].to(torch.int32).contiguous()
        req = forward_batch.req_pool_indices.to(torch.int64)
        if req.numel() != rows:
            req = req.repeat_interleave(rows // req.numel())
        slots = req.to(torch.int32)
        tokens = input_ids[:rows].to(torch.int32)
        # a verify's rows are its request's tokens, contiguous and in order
        run = (
            torch.arange(rows, dtype=torch.int32, device=dev) % self.tokens_per_request
        )
        _, dest, indices = step_rows(
            pos.to(torch.int64),
            req,
            bt,
            ring=self.swa_ring,
            swa_pages=self.swa_pages,
            window=self.layers[0].cfg.window,
            variants={L.ratio: (L.cfg.n_keys, L.cfg.indexed) for L in self.layers},
        )
        return StepInputs(rows, pos, tokens, slots, run, bt, dest, indices)

    def launch(self, L: Layer, h: torch.Tensor, inp: StepInputs) -> torch.Tensor:
        """Layer ``L`` on ``h`` ([rows, hc_mult, hidden] bf16); the next residual."""
        v = self.widths[inp.rows][L.ratio]
        t, c = L.t, L.caches
        p = _ptr

        def q(k):
            return p(t[k]) if k in t else 0

        x_out = torch.empty_like(h)
        v.launch(
            p(h), p(x_out), p(inp.pos), c["kv"], c["kv_rope"],
            p(inp.dest[L.ratio]), p(inp.indices[L.ratio]), p(L.cos), p(L.sin),
            p(t["g_in"]), p(t["g_q"]), p(t["g_kv"]), p(t["g_post"]), p(t["attn_sink"]),
            q("ape"), q("g_ckv"), c["kv_state"], c["score_state"],
            q("i_ape"), q("g_ickv"), c["i_kv_state"], c["i_score_state"], c["i_cache"],
            q("hc_attn_fn"), p(L.hc_sb["attn"]), q("hc_ffn_fn"), p(L.hc_sb["ffn"]),
            p(t["w_qkv_a"]), p(t["s_qkv_a"]), q("w_qkv_c"),
            p(t["w_q_b"]), p(t["s_q_b"]), q("w_i_q_b"), q("s_i_q_b"), q("i_w"),
            p(t["w_o_a"]), p(t["s_o_a"]), p(t["w_o_b"]), p(t["s_o_b"]),
            p(t["w_r"]), p(t["bias"]),
            p(t["w_ug"]), p(t["s_ug"]), p(t["w_dn"]), p(t["s_dn"]),
            q("w_sug"), q("s_sug"), q("w_sdn"), q("s_sdn"),
            p(v.scratch), v.peer.local, p(v.peer.addresses), p(self.epoch),
            p(inp.slots), p(inp.slots),
            p(inp.tokens) if L.use_hash else 0, q("tid2eid"),
            p(inp.block_tables), c["i_cache_s"],
            self.rank, L.i, c["st_kv"], c["st_i"], int(L.use_hash),
            inp.block_tables.stride(0), c["env_rows"], p(inp.run),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return x_out

    def end_step(self, rows: int) -> None:
        """After the step's last launch: scrub stale pairs, then move the step on."""
        stream = torch.cuda.current_stream()
        for v in self.widths[rows].values():
            v.scrub(_ptr(self.epoch), _ptr(v.scratch), v.peer.local, stream=stream)
        self.epoch[:1].add_(1)
