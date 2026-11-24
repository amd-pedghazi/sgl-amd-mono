# SPDX-License-Identifier: Apache-2.0
"""One rank's MiniMax-M3 launches on gfx950: buffers, per-width kernels, per-layer launch.

A sparse MoE layer is one launch: K1 (the input add and norm, the fused qkv / index
projection, the q / k norms, RoPE, the K / V / index-K cache writes and the indexer
scores) fused in front of K4 (the top-k block selection, sparse attention, o_proj, the
router, the shared and routed MXFP4 experts and both all-reduces). A launch reads the
previous layer's (ar, h_mid) while it writes its own, so both alternate between two
buffers by layer parity, baked into each layer's K1 arguments. Everything but the
caches is allocated at construction, before SGLang sizes the KV cache; the caches and
the request table are bound at the first forward (``bind``). Word 0 of the epoch
buffer is the step, which the host moves on after a step's last launch so a layer's
mailbox tags differ from its tags one step earlier; words 1-5 are the guard block
(``common/guard.py``).
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.minimax_m3.gfx950.config import (
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    SPARSE_BLOCK,
)

FP8_CACHE = (torch.float8_e4m3fn, torch.uint8)

# the SGLang names the runner reads, by file under sglang/; test_contract.py pins them
SGLANG_READS = {
    "srt/mem_cache/memory_pool.py": (
        "get_key_buffer",
        "get_value_buffer",
        "get_index_k_buffer",
        "main_pool",
        "kv_cache_layout",
        "head_num",
        "page_size",
    ),
    "srt/layers/attention/minimax_sparse_backend.py": (
        "index_topk_freq",
        "req_to_token",
        "max_context_len",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "req_pool_indices",
        "seq_lens",
        "out_cache_loc",
    ),
}


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds, binds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"MiniMax-M3 monokernels: {what} inside a graph capture")


def compile_only(launch) -> None:
    """Build the launcher without dispatching its kernels."""
    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    _preload_compiled(launch, *([0] * len(names)), torch.cuda.current_stream())


def _ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


class M3MonoRunner:
    """The buffers, kernels and launches of one TP rank's MiniMax-M3 layers."""

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        cpu_group,
        device: torch.device,
        layers: list,
        k4_build: dict,
    ) -> None:
        from sgl_amd_mono.minimax_m3.gfx950.layout import (
            SCRATCH_BYTES as K4_SCRATCH_BYTES,
        )
        from sgl_amd_mono.minimax_m3.gfx950.layout import (
            sym_layout,
        )
        from sgl_amd_mono.minimax_m3.gfx950.pre_attn import (
            SCRATCH_BYTES as K1_SCRATCH_BYTES,
        )

        _ensure_writable_flydsl_cache()
        self.tp, self.rank, self.device = tp, rank, device
        self.layers = layers
        self.k4_build = k4_build
        self.fault_site, self.fault_after = guard.fault_injection(rank) or (0, 0)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=device)
        # the peers' error flags follow the data regions (Guard's flags_offset)
        sym = sym_layout(tp)["_bytes"]
        self.peers = PeerBuffer(sym + guard.FLAG_BYTES, cpu_group, rank, tp, device)
        self.peers.bytes.zero_()
        bf16 = torch.bfloat16
        # row k = token k; layer i reads ars[i % 2] and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=device) for _ in range(2)]  # fmt: skip
        self.h_mids = [torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=device) for _ in range(2)]  # fmt: skip
        self.h = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=device)
        self.q = torch.empty(MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=device)  # fmt: skip
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=device)
        self._kernels: dict = {}
        self.pool = None  # bound at the first forward

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, as the guard block's clock ticks."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def kernel(self, rows: int):
        if rows not in self._kernels:
            from sgl_amd_mono.minimax_m3.gfx950.post_attn import build_post_attn_kernel

            no_capture(f"the {rows}-row launch first reached")
            self._kernels[rows] = build_post_attn_kernel(
                tokens=rows,
                fault_site=self.fault_site,
                fault_after=self.fault_after,
                **self.k4_build,
            )
        return self._kernels[rows]

    def precompile(self, rows: int) -> None:
        compile_only(self.kernel(rows))

    def bind(self, backend, pool) -> None:
        """Bind the caches, the request table, the K1 arguments and the per-width tables.

        The caches: the sparse layers' page16 SHUFFLE K / V and NHD index-K, one KV head
        a rank. The K1 arguments: each layer's pointers that do not change per step. The
        row tables' width is the backend's context, known only now, after SGLang sized
        the KV cache: at most 4.5 MB (a 1M context)."""
        from sgl_amd_mono.minimax_m3.gfx950.pre_attn import K1_ARGS

        no_capture("the caches bound")
        self.k_ptrs, self.v_ptrs, self.k1_args = [], [], []
        for i, L in enumerate(self.layers):
            k = pool.get_key_buffer(L.layer_id)
            v = pool.get_value_buffer(L.layer_id)
            ik = pool.get_index_k_buffer(L.layer_id)
            self.k_ptrs.append(_ptr(k))
            self.v_ptrs.append(_ptr(v))
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(L.g_in),
                "w_qkv": _ptr(L.w_qkv), "s_qkv": _ptr(L.s_qkv),
                "g_q": _ptr(L.g_q), "g_k": _ptr(L.g_k),
                "g_iq": _ptr(L.g_iq), "g_ik": _ptr(L.g_ik),
                "cos_sin": _ptr(L.cos_sin), "iq_out": _ptr(self.iq),
                "index_cache": _ptr(ik), "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor(
                    [ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=self.device
                )  # fmt: skip
            )
        self.req_to_token = backend.req_to_token
        # not the allocator's size: EAGLE's extra lookahead page adds no context or score capacity
        self.max_context_len = int(backend.max_context_len)
        self.bt_width = (self.max_context_len + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        dev = self.device
        # per token count: the step's block table and sequence lengths
        self.tables = {
            n: (
                torch.zeros(n, self.bt_width, dtype=torch.int32, device=dev),
                torch.ones(n, dtype=torch.int32, device=dev),
            )
            for n in range(1, MAX_TOKENS + 1)
        }
        self.cols = torch.arange(
            0, self.bt_width * SPARSE_BLOCK, SPARSE_BLOCK, dtype=torch.int64, device=dev
        )  # fmt: skip
        self.pool = pool

    def cache_reasons(self, backend, pool) -> list[str]:
        """Reasons the bound caches are not what the kernels write (checked once,
        at the first forward, after SGLang sized them)."""
        why = []
        if backend.index_topk_freq != 1:
            why.append("the indexer must select blocks in every layer")
        # sgl-project/sglang#41998's pool flag, read with a fallback until it merges
        if not getattr(pool, "use_aiter_sparse_pa", False):
            why.append("the sparse layers' K / V are not page16 SHUFFLE")
        if pool.page_size != SPARSE_BLOCK:
            why.append(f"page size {pool.page_size}, the kernels index 128-token pages")
        main = pool.main_pool
        if main.kv_cache_layout != "nhd" or main.head_num != 1:
            why.append(f"KV views {main.kv_cache_layout} with {main.head_num} heads")
        ids = [L.layer_id for L in self.layers]
        k0, v0 = pool.get_key_buffer(ids[0]), pool.get_value_buffer(ids[0])
        i0 = pool.get_index_k_buffer(ids[0])
        if not (
            k0.dtype in FP8_CACHE and v0.dtype in FP8_CACHE and i0.dtype in FP8_CACHE
        ):
            why.append(f"caches {k0.dtype} / {v0.dtype} / {i0.dtype}, the kernels write e4m3")  # fmt: skip
        for lid in ids:
            k, v = pool.get_key_buffer(lid), pool.get_value_buffer(lid)
            ik = pool.get_index_k_buffer(lid)
            if not (k.is_contiguous() and v.is_contiguous() and ik.is_contiguous()
                    and k.shape == k0.shape and ik.shape == i0.shape
                    and k.shape[-1] == ik.shape[-1] == HEAD_DIM):  # fmt: skip
                why.append(
                    f"layer {lid}: cache views {tuple(k.shape)} / {tuple(ik.shape)}"
                )
        r2t = backend.req_to_token
        if r2t.dtype != torch.int32 or r2t.dim() != 2:
            why.append(f"req_to_token {r2t.dtype} {tuple(r2t.shape)}")
        blocks = (int(backend.max_context_len) + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        if not 0 < blocks <= MAX_INDEX_BLOCKS:
            why.append(f"{blocks} blocks a request; the scores hold {MAX_INDEX_BLOCKS}")
        return why

    def prepare_step(self, forward_batch, rows: int, query_len: int, is_verify: bool):
        """This step's causal block table and sequence lengths, from the replay-fresh
        decode / verify buffers (inside the graph where there is one)."""
        from sgl_amd_mono.minimax_m3.metadata import prepare_token_rows

        bt, seq_lens = self.tables[rows]
        return prepare_token_rows(
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.cols,
            bt,
            seq_lens,
            forward_batch.out_cache_loc[:rows],
            query_len,
            is_verify,
            self.max_context_len,
        )

    def launch(
        self,
        *,
        i: int,
        rows: int,
        query_len: int,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        tables: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse layer ``i`` (0: the first MoE layer) in a decoder layer's terms:
        ``(ar, res)`` in, the next layer's ``(ar, h_mid)`` out.

        ``ar`` has to be in ``ars[i % 2]``, where the previous launch left it; the copy
        is for the first sparse layer, which SGLang's modules hand it."""
        L = self.layers[i]
        ar = self.ars[i % 2]
        if hidden.data_ptr() != ar.data_ptr():
            ar[:rows].copy_(hidden.view(rows, HIDDEN))
        block_table, seq_lens = tables
        _run_compiled(
            self.kernel(rows),
            _ptr(self.h), _ptr(self.q), _ptr(block_table), _ptr(seq_lens),
            self.k_ptrs[i], self.v_ptrs[i], _ptr(L.k_scale), _ptr(L.v_scale),
            _ptr(L.w_o), _ptr(L.s_o), _ptr(L.g_post), _ptr(L.gate),
            _ptr(L.bias), _ptr(L.w13), _ptr(L.s13), _ptr(L.w2), _ptr(L.s2),
            _ptr(self.h_mids[(i + 1) % 2]), _ptr(self.ars[(i + 1) % 2]),
            _ptr(self.scratch4), self.peers.local, _ptr(self.peers.addresses),
            _ptr(self.epoch), self.rank, L.layer_id,
            block_table.shape[1], query_len,
            _ptr(self.k1_args[i]), _ptr(positions), _ptr(slot_mapping),
            _ptr(residual.view(rows, HIDDEN)),
            torch.cuda.current_stream(),
        )  # fmt: skip
        if i == len(self.layers) - 1:
            # the next step's epoch, enqueued after this step's last launch
            self.epoch[0].add_(1)
        return self.ars[(i + 1) % 2][:rows], self.h_mids[(i + 1) % 2][:rows]
