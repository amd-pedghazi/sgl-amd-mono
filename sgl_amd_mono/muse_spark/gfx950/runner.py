# SPDX-License-Identifier: Apache-2.0
"""One rank's Muse Spark launches on gfx950.

A decoder layer with MXFP4 experts runs as K1 (``stages/pre_attn``: the fused
projection, the q / k norms and query scale, RoPE, the fp8 cache write), K-attn
(``stages/attention``: the verify attention over the cache), K2
(``stages/post_attn``: o-norm-gate, o_proj with the attention all-reduce, the
attention boundary, router / pooler, top-8, the pre-expert norm) and K3
(``stages/ffn``: the experts, finalize with the EP all-reduce, the sharded
unpooler, the FFN boundary). The bf16-expert layers (0 and 61) run K1, K-attn and
K2, then their bf16 experts (``tpx_bf16`` under TP experts) and SGLang's fused
unpooler + FFN boundary (SGLANG_AMD_MONO_MUSE_BF16_K2=0: SGLang's own decoder
layer, as the stage-0 plugin did). A launch takes ``(stream, attn_input)``,
the fp32 residual stream (updated in place) and the layer's bf16 attention input,
and hands on the next pair.

Everything but the caches is allocated at construction, before SGLang sizes the
KV cache. The caches, the SWA slot table and the attention backend are bound at
the first forward (``bind``). Word 0 of the epoch buffer is the step: the host
moves it on after a step's last launch, which keeps a layer's mailbox tags
distinct from the same layer's one step earlier; words 1-5 are the guard block.
"""

from __future__ import annotations

import inspect
import os
import struct
from pathlib import Path

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.muse_spark.gfx950.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    KV_K,
    LAYER_SLOTS,
    LOCAL_KV_HEADS,
    MAX_TOKENS,
    MOE_HIDDEN,
    PAGE,
    Q_K,
    TOP_K,
    WINDOW,
)
from sgl_amd_mono.muse_spark.gfx950.layout import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    k1_scratch_layout,
    kattn_scratch_layout,
    scratch_bytes,
)

# The SGLang names the runner reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/attention/aiter_backend.py": (
        "AiterAttnBackend",
        "page_size",
        "kv_cache_dtype",
        "use_triton_unified_attention",
        "use_sliding_window_kv_pool",
        "token_to_kv_pool",
        "forward_metadata",
        "swa_page_table",
        "kv_indices",
        "k_scale",
        "v_scale",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "kernels/ops/gemm/unpool_gemm_norm_residual.py": ("unpool_gemm_norm_residual",),
    "srt/distributed/parallel_state.py": ("get_tp_group",),
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/mem_cache/swa_memory_pool.py": (
        "SWAKVPool",
        "layers_mapping",
        "get_kv_buffer",
        "full_to_swa_index_mapping",
    ),
    "srt/rocm_muse_spark/model.py": (
        "forward(attn_input)",
        "forward(ffn_boundary)",
        "forward(pre_norm_dtype)",
        "self_attn",
        "attn",
        "mlp",
        "local_expert_start",
    ),
    "srt/layers/radix_attention.py": ("sliding_window_size", "layer_id", "scaling"),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "seq_lens",
        "forward_mode",
    ),
    # ForwardMode and CaptureHiddenMode, in forward_batch_info.py before forward_mode.py
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "is_target_verify",
    ),
}

FP8_CACHE = torch.float8_e4m3fn
# SGLANG_AMD_MONO_MUSE_KERNELS=0: every layer on SGLang's modules (the stage-0 plugin)
USE_KERNELS = os.environ.get("SGLANG_AMD_MONO_MUSE_KERNELS", "1") == "1"
# an MXFP4 layer's attention, K2, K3 and the next layer's K1 as one launch (the
# default); SGLANG_AMD_MONO_MUSE_LAYER=0: the four launches
USE_LAYER = os.environ.get("SGLANG_AMD_MONO_MUSE_LAYER", "1") == "1"
# SGLANG_AMD_MONO_MUSE_SKIP=1: a timing diagnostic, the MXFP4 layers skipped (garbage
# output): the step's cost outside the kernel layers
SKIP_LAYERS = os.environ.get("SGLANG_AMD_MONO_MUSE_SKIP", "0") == "1"
# SGLANG_AMD_MONO_MUSE_TRACE=<dir>: the kernels stamp their stages (``device.stamp``, a slot
# set a layer) and a thread snapshots the stamps to <dir>/rank<r>.pt every few seconds
# (``test/kernels/report_muse_spark_trace.py``). A diagnostic: the stamps cost a little.
TRACE_DIR = os.environ.get("SGLANG_AMD_MONO_MUSE_TRACE", "")
# SGLANG_AMD_MONO_MUSE_TPX=1: TP experts (ATOM's TP-sharded MoE): every rank works every
# pick on its slices of every expert (weights.tp_expert_copies, ~0.8 GB a layer), so no
# rank waits on another's expert load
# TP experts (every rank its slices of every expert, ~0.8 GB a layer more): 600 s AgentX C1 P90 +8%, C2 +22%
# over EP (the EP skew); =0 keeps EP
TPX = os.environ.get("SGLANG_AMD_MONO_MUSE_TPX", "1") == "1"
# the merged launch's next K1 folds the FFN boundary's norms itself (no wait on the boundary's x)
DEFER_K1 = os.environ.get("SGLANG_AMD_MONO_MUSE_DEFER_K1", "0") == "1"
# K2 hands K3 the expert input pre-quantized, staged once a CTA (S <= 16); =0 restages a tile's columns
XQ_PRE = os.environ.get("SGLANG_AMD_MONO_MUSE_XQ_PRE", "1") == "1"
# K3's ug items past the first from a launch-wide queue (with XQ_PRE, at 16 rows: C2 -0.2 ms a step, C1 neutral)
UG_DYN = os.environ.get("SGLANG_AMD_MONO_MUSE_UG_DYN", "1") == "1"
# the pooler sharded by rows over the 8 ranks (Kimi-K3's latent projection): 8.4 MB a layer a rank instead of 67;
# the partial pooled rows go to every rank (=0: the replicated pooler)
PSHARD = os.environ.get("SGLANG_AMD_MONO_MUSE_PSHARD", "1") == "1"
# the bf16-expert layers (0 and 61) on K1 / the attention / K2, then SGLang's bf16 experts and FFN boundary
# (=0: SGLang's whole layer)
BF16_K2 = os.environ.get("SGLANG_AMD_MONO_MUSE_BF16_K2", "1") == "1"
# with BF16_K2 and TP experts, the bf16-expert layers' slices as FP8 with power-of-two group scales (a numerics
# change; =0: bf16 slices)
BF16_FP8 = os.environ.get("SGLANG_AMD_MONO_MUSE_BF16_FP8", "1") == "1"


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds, binds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"Muse Spark monokernels: {what} inside a graph capture")


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
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


def _f32_bits(value: float) -> int:
    return struct.unpack("i", struct.pack("f", value))[0]


# per-layer-kind builds: the sliding layers launch their own build (attention.run_kattn's SL: the full
# layers' paths compiled out); SGLANG_AMD_MONO_SPLIT_BUILDS=0 puts every layer on the one build
SPLIT_BUILDS = os.environ.get("SGLANG_AMD_MONO_SPLIT_BUILDS", "1") == "1"
# the merged launches warm L2 with the next layer's kernel code before their K1 (layer.prefetch_code): the sliding /
# full builds alternate, and a switch otherwise costs ~1-1.5 us of instruction fetch; SGLANG_AMD_MONO_CODE_PREFETCH=0: off
CODE_PREFETCH = os.environ.get("SGLANG_AMD_MONO_CODE_PREFETCH", "1") == "1"


def kv_head_major_rows(t) -> int:
    """A [rows, heads, dim] KV buffer stored head-major ([heads][rows][dim]: strides (dim, rows dim, 1)):
    its rows; 0 for any other layout."""
    if (
        t.dim() != 3
        or t.stride(2) != 1
        or t.stride(0) != t.shape[2]
        or t.stride(1) != t.shape[0] * t.shape[2]
    ):
        return 0
    return int(t.shape[0])


class MuseMonoRunner:
    """The buffers, kernels and launches of one TP rank's Muse Spark layers."""

    def __init__(
        self, *, views, modules, boundaries, rank: int, npes: int, cpu_group, device
    ):
        from sgl_amd_mono.muse_spark.gfx950.stages.ffn import k3_scratch_layout
        from sgl_amd_mono.muse_spark.gfx950.stages.post_attn import (
            k2_scratch_layout,
            sym_bytes,
        )

        _ensure_writable_flydsl_cache()
        assert len(views) == len(modules) == len(boundaries) <= LAYER_SLOTS
        self.views, self.modules, self.boundaries = (
            list(views),
            list(modules),
            list(boundaries),
        )
        self.tpx = TPX and USE_KERNELS
        self.tpx_w = None  # the TP-experts slices: built at bind, after SGLang sized the KV cache
        self.pool_rows = None  # the sharded pooler's row slices a layer: built at bind
        self.rank, self.npes, self.device = rank, npes, torch.device(device)
        self.use_kernels = USE_KERNELS
        self.use_layer = USE_LAYER
        # the pooler / unpooler as FP8 copies when every kernel layer carries them
        self.fp8_dense = all(
            v.w_pool_fp8 is not None and v.w_unpool_fp8 is not None
            for v in views
            if v.kind == "mxfp4"
        )
        self.pshard = PSHARD and npes == 8 and not self.fp8_dense
        self._k1_done = -1  # the layer whose K1 the previous merged launch ran
        self._trace_shape = None  # (rows, q_len) of the last launch, for the attention trace region's offset
        self.fault_site, self.fault_after = guard.fault_injection(rank) or (0, 0)
        dev, bf16 = self.device, torch.bfloat16
        self.deadline_s = None
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        from sgl_amd_mono.muse_spark.gfx950.layer import pf_table_bytes, pf_table_offset

        # K1's scratch, then the code prefetch's table (entry (rows, layer): the next layer's kernel code)
        self.scratch_k1 = torch.zeros(
            pf_table_offset() + pf_table_bytes(), dtype=torch.uint8, device=dev
        )
        self._pf_filled = set()  # the (rows, layer) entries written
        self._pf_code = {}  # kernel symbol -> (code address, bytes)
        self._pf_prev = None  # (rows, layer) of the last eager merged launch
        self.scratch_attn = torch.zeros(
            max(scratch_bytes(kattn_scratch_layout(MAX_TOKENS, q)) for q in (1, 8)),
            dtype=torch.uint8, device=dev,
        )  # fmt: skip
        self.scratch_k2 = torch.zeros(
            scratch_bytes(k2_scratch_layout()), dtype=torch.uint8, device=dev
        )
        self.scratch_k3 = torch.zeros(
            scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
        )
        # the peers' error flags follow the partial-row regions (Guard's flags_offset);
        # the handshake is collective: every rank gets here (attach agreed first)
        self.peers = PeerBuffer(
            sym_bytes(npes) + guard.FLAG_BYTES, cpu_group, rank, npes, dev
        )
        self.peers.bytes.zero_()
        # layers 0 / 61's FP8 TP-expert FFN as one launch (tpx_fp8x.build_fused): its gathered mids / y
        # (tagged pairs) and the plain y it leaves in their own peer buffer (sized for the largest step it
        # serves, 16 rows)
        self.tpx_fused = None
        if self.tpx and npes > 1:
            from sgl_amd_mono.muse_spark.tpx_bf16 import _BM
            from sgl_amd_mono.muse_spark.tpx_fp8x import fused_layout

            nb = fused_layout(_BM, TOP_K, INTER // npes, MOE_HIDDEN // npes, npes)[
                "bytes"
            ]
            self.tpx_fused = PeerBuffer(nb, cpu_group, rank, npes, dev)
            self.tpx_fused.bytes.zero_()
        # row k = token k of the step
        self.q = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=dev)
        self.k = torch.empty(MAX_TOKENS, KV_K, dtype=bf16, device=dev)
        self.v = torch.empty(MAX_TOKENS, KV_K, dtype=bf16, device=dev)
        self.g = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=dev)
        self.o = torch.empty(MAX_TOKENS, Q_K, dtype=bf16, device=dev)
        self.ids = torch.empty(MAX_TOKENS, TOP_K, dtype=torch.int32, device=dev)
        self.scores = torch.empty(MAX_TOKENS, TOP_K, dtype=torch.float32, device=dev)
        self.xin = torch.empty(MAX_TOKENS, MOE_HIDDEN, dtype=bf16, device=dev)
        self.attn_in = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        # K1 reads the cos / sin table of every token: a NoPE layer takes any table
        tables = [L.cos_sin for L in self.views if L.cos_sin is not None]
        if not tables:
            raise RuntimeError("Muse Spark monokernels: no rotary table")
        self.any_cos_sin = tables[0]
        self._k1: dict = {}
        self._kattn: dict = {}
        self._k2: dict = {}
        self._k3: dict = {}
        self._layer: dict = {}
        self.backend = None
        self.pool = None
        self.caches = None  # per layer: (k cache ptr, v cache ptr, is a sliding layer)
        self.kv_rows = (
            0  # the full layers' pool rows when it is head-major (``bind``), else 0
        )
        self.swa_rows = 0  # the sliding layers' (SWA) pool rows when it is head-major (``bind``), else 0
        self.swa_map = None
        self.windows = None

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, as the guard block's clock ticks."""
        self.deadline_s = float(seconds)
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    # ---------------------------------------------------------------- kernels
    def k1_kernel(self, rows: int):
        if (rows, self.kv_rows, self.swa_rows) not in self._k1:
            from sgl_amd_mono.muse_spark.gfx950.layer import K1Build, build_k1

            no_capture(f"the {rows}-row K1 first reached")
            self._k1[(rows, self.kv_rows, self.swa_rows)] = build_k1(K1Build(tokens=rows, fault_site=self.fault_site, fault_after=self.fault_after, trace=bool(TRACE_DIR), kv_rows=self.kv_rows, swa_rows=self.swa_rows))  # fmt: skip
        return self._k1[(rows, self.kv_rows, self.swa_rows)]

    def kattn_kernel(self, rows: int, q_len: int):
        if (rows, q_len, self.kv_rows, self.swa_rows) not in self._kattn:
            from sgl_amd_mono.muse_spark.gfx950.layer import KAttnBuild, build_kattn

            no_capture(f"the {rows}-row K-attn first reached")
            self._kattn[(rows, q_len, self.kv_rows, self.swa_rows)] = build_kattn(
                KAttnBuild(
                    tokens=rows,
                    q_len=q_len,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                    trace=bool(TRACE_DIR),
                    kv_rows=self.kv_rows,
                    swa_rows=self.swa_rows,
                )  # fmt: skip
            )
        return self._kattn[(rows, q_len, self.kv_rows, self.swa_rows)]

    def k2_kernel(self, rows: int):
        if rows not in self._k2:
            from sgl_amd_mono.muse_spark.gfx950.build_k2 import K2Build, build_k2

            no_capture(f"the {rows}-row K2 first reached")
            self._k2[rows] = build_k2(
                K2Build(
                    tokens=rows,
                    npes=self.npes,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                    trace=bool(TRACE_DIR),
                    fp8_pool=self.fp8_dense,
                    pshard=self._pshard(rows),
                )
            )
        return self._k2[rows]

    def k3_kernel(self, rows: int):
        if rows not in self._k3:
            from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3

            no_capture(f"the {rows}-row K3 first reached")
            self._k3[rows] = build_k3(
                K3Build(
                    tokens=rows,
                    npes=self.npes,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                    trace=bool(TRACE_DIR),
                    fp8_unpool=self.fp8_dense,
                    tpx=self.tpx,
                )
            )
        return self._k3[rows]

    def layer_kernel(self, rows: int, q_len: int, next_k1: bool, kind: str = ""):
        if (rows, q_len, next_k1, kind, self.kv_rows, self.swa_rows) not in self._layer:
            from sgl_amd_mono.muse_spark.gfx950.layer import LayerBuild, build_layer

            no_capture(f"the {rows}-row layer launch first reached")
            self._layer[(rows, q_len, next_k1, kind, self.kv_rows, self.swa_rows)] = (
                build_layer(
                    LayerBuild(
                        tokens=rows,
                        q_len=q_len,
                        npes=self.npes,
                        next_k1=next_k1,
                        fault_site=self.fault_site,
                        fault_after=self.fault_after,
                        trace=bool(TRACE_DIR),
                        fp8_dense=self.fp8_dense,
                        tpx=self.tpx,
                        defer_k1=DEFER_K1,
                        xq_pre=XQ_PRE,
                        ug_dyn=UG_DYN,
                        pshard=self._pshard(rows),
                        kind=kind,
                        kv_rows=self.kv_rows,
                        swa_rows=self.swa_rows,
                        pf=CODE_PREFETCH,
                    )  # fmt: skip
                )
            )
        return self._layer[(rows, q_len, next_k1, kind, self.kv_rows, self.swa_rows)]

    def _pf_note(self, i: int, rows: int, kernel) -> None:
        """After an EAGER merged launch of layer i (its kernel now loaded): if layer i - 1 was a merged launch, its
        table entry gets this kernel's code range (a host write once per (rows, layer), never under capture: the
        captured graphs read the table the warm-up runs wrote)."""
        if torch.cuda.is_current_stream_capturing():
            return
        prev, self._pf_prev = self._pf_prev, (rows, i)
        if prev != (rows, i - 1) or prev in self._pf_filled:
            return
        import re

        from sgl_amd_mono.common.code_addr import all_kernels
        from sgl_amd_mono.muse_spark.gfx950.layer import pf_entry

        arts = list(getattr(kernel, "_mem_cache", {}).values())
        if not arts:
            return
        m = re.search(r"(muse_spark_layer\w*)", arts[0]._ir_text)
        if m is None:
            return
        name = m.group(1)
        if name not in self._pf_code:
            hits = [v for k, v in all_kernels().items() if name in k]
            if len(hits) != 1 or len(hits[0]) != 1:
                self._pf_filled.add(
                    prev
                )  # not found or ambiguous: leave the entry 0 (no prefetch)
                return
            self._pf_code[name] = hits[0][0]
        addr, nbytes = self._pf_code[name]
        off = pf_entry(rows, i - 1)
        e = torch.tensor([addr, nbytes], dtype=torch.int64)
        self.scratch_k1[off : off + 16].copy_(e.view(torch.uint8), non_blocking=False)
        self._pf_filled.add(prev)

    def _kind(self, i: int, rows: int, q_len: int) -> str:
        """Layer i's build kind at 8 rows a request and at most 7 requests: "sliding" (its own build:
        the full layers' attention paths compiled out) or "full" (the sliding paths out, the job mapping
        in whole groups of 8 partitions); else "" (one build for every layer)."""
        if not SPLIT_BUILDS or q_len != 8 or rows // q_len > 7 or self.caches is None:
            return ""
        return "sliding" if self.caches[i][2] else "full"

    def _build_tpx(self) -> None:
        """The TP-experts slices (``weights.tp_expert_copies``, ~0.8 GB a layer): taken
        here, at the first bind (an eager forward every rank runs: the all-to-alls are
        collective), from the memory SGLang leaves outside its static fraction once the
        KV cache is sized -- the recipe's fraction has no room for them before."""
        import time

        from sglang.srt.runtime_context import get_parallel

        from sgl_amd_mono.muse_spark.weights import tp_expert_copies

        t0 = time.time()
        free0 = torch.cuda.mem_get_info(self.device)[0]
        group = get_parallel().tp_group.device_group if self.npes > 1 else None
        self.tpx_w = tp_expert_copies(
            self.views,
            self.modules,
            self.rank,
            self.npes,
            group,
            bf16=BF16_K2,
            bf16_fp8=BF16_FP8,
        )
        torch.cuda.synchronize(self.device)
        gb = sum(sum(t.numel() for t in w) for w in self.tpx_w if w is not None) / 2**30
        free1 = torch.cuda.mem_get_info(self.device)[0]
        print(f"[muse mono] rank {self.rank}: TP experts slices {gb:.1f} GiB in {time.time() - t0:.1f} s; device free {free0 / 2**30:.1f} -> {free1 / 2**30:.1f} GiB", flush=True)  # fmt: skip

    def _pshard(self, rows: int) -> bool:
        """The sharded pooler at this width: S <= 16 only (with the boundary split, a 64-row
        launch's waits never complete at TP 8 -- under investigation; 32 / 64 rows keep the
        replicated pooler)."""
        return self.pshard and rows <= 16

    def _pool_w(self, i: int, rows: int):
        """Layer i's pooler for K2: the rank's row slice (sharded), the FP8 copy, or the weight."""
        L = self.views[i]
        if self._pshard(rows):
            return self.pool_rows[i]
        return L.w_pool_fp8 if self.fp8_dense else L.w_pool

    def _router_w(self, i: int, rows: int) -> tuple:
        """Layer i's router (hi, lo) for K2."""
        L = self.views[i]
        return L.gate_hi, L.gate_lo

    def _experts(self, i: int) -> tuple:
        """Layer ``i``'s w13, s13, w2, s2 pointers: EP's whole local experts, or under TP
        experts the rank's slices of every expert."""
        if self.tpx:
            w = self.tpx_w[i] if self.tpx_w is not None else None
            if w is None:
                raise RuntimeError(
                    f"Muse Spark layer {i}: TP-experts launch without its slices"
                )
            return tuple(_ptr(t) for t in w)
        L = self.views[i]
        return (_ptr(L.w13), _ptr(L.s13), _ptr(L.w2), _ptr(L.s2))

    @staticmethod
    def _k2_layer(view) -> bool:
        """Whether a layer's K2 runs as a launch (MXFP4 layers, and the bf16-expert ones under BF16_K2)."""
        return view.kind == "mxfp4" or BF16_K2

    def merged(self, i: int, rows: int, q_len: int) -> bool:
        """Layer ``i`` runs as one launch (the attention, K2, K3 and, when the next
        layer takes the kernels too, its K1)."""
        return self.use_layer and self.views[i].kind == "mxfp4" and rows % q_len == 0

    def _next_k1(self, i: int) -> bool:
        return i + 1 < len(self.views) and self._k2_layer(self.views[i + 1])

    def precompile(self, rows: int, q_len: int = 8) -> None:
        """Compile every launch of a width (FlyDSL's disk cache then serves the
        other ranks and the next start)."""
        if not self.use_kernels:
            return
        compile_only(self.k1_kernel(rows))
        if rows % q_len == 0:
            compile_only(self.kattn_kernel(rows, q_len))
            if self.use_layer:
                for next_k1, kind in {
                    (self._next_k1(i), self._kind(i, rows, q_len))
                    for i in range(len(self.views))
                    if self.views[i].kind == "mxfp4"
                }:
                    compile_only(self.layer_kernel(rows, q_len, next_k1, kind))
        if not (self.use_layer and rows % q_len == 0):
            compile_only(self.k2_kernel(rows))
            compile_only(self.k3_kernel(rows))
        elif BF16_K2 and any(v.kind != "mxfp4" for v in self.views):
            compile_only(self.k2_kernel(rows))

    # ---------------------------------------------------------------- caches
    def cache_reasons(self, backend, pool) -> list[str]:
        """Why the kernels cannot read this backend's caches (empty: they can)."""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        if not isinstance(backend, AiterAttnBackend):
            return [
                f"attention backend {type(backend).__name__}, the kernels read the aiter backend's caches"
            ]
        why = []
        if backend.page_size != PAGE:
            why.append(f"page size {backend.page_size}, built for {PAGE}")
        if backend.kv_cache_dtype != FP8_CACHE:
            why.append(f"KV cache {backend.kv_cache_dtype}, built for {FP8_CACHE}")
        if not backend.use_triton_unified_attention:
            why.append("the kernels mirror aiter's unified attention")
        if not backend.use_sliding_window_kv_pool or type(pool).__name__ != "SWAKVPool":
            return why + [
                f"KV pool {type(pool).__name__}, the kernels read the sliding-window pool"
            ]
        m = pool.full_to_swa_index_mapping
        if (
            m is None
            or m.dtype not in (torch.int32, torch.int64)
            or not m.is_contiguous()
        ):
            why.append("no contiguous int32 / int64 full-to-SWA slot table")
        for name in ("k_scale", "v_scale"):
            s = getattr(backend, name, None)
            if s is not None and (
                not isinstance(s, torch.Tensor) or s.numel() != 1 or float(s) != 1.0
            ):
                why.append(f"backend {name} {s}: the kernels write fp8 at unit scale")
        for i, layer in enumerate(self.modules):
            attn = layer.self_attn.attn
            if attn.k_scale is not None or attn.v_scale is not None:
                why.append(f"layer {i}: a per-layer K / V scale")
            _, is_swa = pool.layers_mapping[attn.layer_id]
            if is_swa != (attn.sliding_window_size == WINDOW):
                why.append(
                    f"layer {i}: pool {'sliding' if is_swa else 'full'} vs window {attn.sliding_window_size}"
                )
            k, v = pool.get_kv_buffer(attn.layer_id)
            for t, what in ((k, "K"), (v, "V")):
                hm = (
                    t.dim() == 3 and kv_head_major_rows(t) > 0
                )  # (a full or a sliding pool head-major)
                if not (t.dtype == FP8_CACHE and (t.is_contiguous() or hm) and t.dim() == 3 and tuple(t.shape[-2:]) == (LOCAL_KV_HEADS, HEAD_DIM)):  # fmt: skip
                    why.append(f"layer {i} {what} cache: {t.dtype} {tuple(t.shape)} {t.stride()}, want contiguous (or a full layer's head-major) {FP8_CACHE} [*, {LOCAL_KV_HEADS}, {HEAD_DIM}]")  # fmt: skip
        return why

    def bind(self, backend, pool) -> None:
        """The layers' caches, windows and the SWA slot table, after SGLang sized them."""
        no_capture("the caches bound")
        hm_rows, hs_rows = set(), set()
        self.caches, self.windows = [], []
        for layer in self.modules:
            attn = layer.self_attn.attn
            _, is_swa = pool.layers_mapping[attn.layer_id]
            k, v = pool.get_kv_buffer(attn.layer_id)
            self.caches.append((_ptr(k), _ptr(v), bool(is_swa)))
            if not is_swa:
                hm_rows |= {kv_head_major_rows(k), kv_head_major_rows(v)}
            else:
                hs_rows |= {kv_head_major_rows(k), kv_head_major_rows(v)}
            w = int(attn.sliding_window_size or -1)
            self.windows.append(w if w > 0 else 0)
        # a head-major full pool ([head][row][dim] storage, seen as [row][head][dim]): its rows (the
        # kernels' plane), every full layer's alike; else NHD (0)
        if len(hm_rows) > 1:
            raise RuntimeError(
                f"Muse Spark: the full layers' KV buffers mix layouts / sizes {sorted(hm_rows)}"
            )
        self.kv_rows = hm_rows.pop() if hm_rows else 0
        if len(hs_rows) > 1:
            raise RuntimeError(
                f"Muse Spark: the sliding layers' KV buffers mix layouts / sizes {sorted(hs_rows)}"
            )
        self.swa_rows = hs_rows.pop() if hs_rows else 0
        self.swa_map = pool.full_to_swa_index_mapping
        self.backend, self.pool = backend, pool
        if self.tpx and self.tpx_w is None:
            self._build_tpx()
        if self.pshard and self.pool_rows is None:
            n = MOE_HIDDEN // self.npes
            self.pool_rows = [
                v.w_pool[self.rank * n : (self.rank + 1) * n].contiguous() if self._k2_layer(v) else None for v in self.views
            ]  # fmt: skip
        if TRACE_DIR:
            # the pinned host buffers before any capture (no host allocation inside
            # one); the save thread is CPU-only, so it may run through the captures
            regions = {
                "k1": self.scratch_k1,
                "attn": self.scratch_attn,
                "k2": self.scratch_k2,
                "k3": self.scratch_k3,
            }
            self._trace_host = {k: torch.empty(TRACE_LAYERS * BLOCKS * TRACE_SLOTS * 8, dtype=torch.uint8, pin_memory=True) for k in regions}  # fmt: skip
            self._trace_epoch_host = torch.empty_like(
                self.epoch, device="cpu", pin_memory=True
            )
            self._start_trace_thread()

    # ---------------------------------------------------------------- stage trace
    def _trace_regions(self):
        from sgl_amd_mono.muse_spark.gfx950.stages.ffn import k3_scratch_layout
        from sgl_amd_mono.muse_spark.gfx950.stages.post_attn import k2_scratch_layout

        rows, q_len = self._trace_shape

        def region(buf, layout, name):
            off, nbytes = layout[name]
            return buf[off : off + nbytes]

        return {
            "k1": region(self.scratch_k1, k1_scratch_layout(), "k1_trace"),
            "attn": region(
                self.scratch_attn, kattn_scratch_layout(rows, q_len), "attn_trace"
            ),
            "k2": region(self.scratch_k2, k2_scratch_layout(), "k2_trace"),
            "k3": region(self.scratch_k3, k3_scratch_layout(), "k3_trace"),
        }

    def _trace_copy_out(self) -> None:
        """Enqueue, in stream order after the step's last launch (inside the graph when
        capturing), the stamp regions' copies to pinned host buffers. A side thread's
        own device copies stalled the ranks' chains (blit kernels / the GIL) and
        skewed what they measured; these copies cost the step ~0.2 ms and nothing
        inside a launch."""
        if self._trace_shape is None:
            return
        for k, v in self._trace_regions().items():
            self._trace_host[k][: v.numel()].copy_(v, non_blocking=True)
        self._trace_epoch_host.copy_(self.epoch, non_blocking=True)

    def _start_trace_thread(self) -> None:
        """Save the host copies of the stamps to ``TRACE_DIR/rank<r>.pt`` every two
        seconds (a daemon thread that never touches the device)."""
        import threading
        import time

        os.makedirs(TRACE_DIR, exist_ok=True)

        def loop():
            path = os.path.join(TRACE_DIR, f"rank{self.rank}.pt")
            while True:
                time.sleep(2.0)
                host = getattr(self, "_trace_host", None)
                if host is None or self._trace_shape is None:
                    continue
                rows, q_len = self._trace_shape
                try:
                    snap = {k: v.clone() for k, v in host.items()}
                    snap["epoch"] = self._trace_epoch_host.clone()
                    snap.update(
                        rank=self.rank, npes=self.npes, time=time.time(), rows=rows, q_len=q_len,
                        windows=list(self.windows), swa=[bool(c[2]) for c in self.caches],
                        kinds=[v.kind for v in self.views], use_layer=self.use_layer,
                    )  # fmt: skip
                    torch.save(snap, path + ".tmp")
                    os.replace(path + ".tmp", path)
                except Exception as e:  # noqa: BLE001 - a diagnostic thread never takes the server down
                    print(
                        f"[muse mono trace] rank {self.rank}: snapshot failed: {e!r}",
                        flush=True,
                    )
                    time.sleep(10.0)

        threading.Thread(
            target=loop, name=f"muse-mono-trace-{self.rank}", daemon=True
        ).start()

    # ---------------------------------------------------------------- launches
    def _attn_args(self, i: int, rows: int, forward_batch, q_len: int) -> tuple:
        """K-attn's arguments after q and o for layer ``i``: the cache, the page
        table, the lengths, the window, the scale, the scratch."""
        meta = self.backend.forward_metadata
        k_cache, v_cache, is_swa = self.caches[i]
        table = meta.swa_page_table if is_swa else meta.kv_indices
        if (
            table is None
            or table.dtype != torch.int32
            or table.dim() != 2
            or not table.is_contiguous()
        ):
            raise RuntimeError(f"K-attn layer {i}: page table {None if table is None else (table.dtype, tuple(table.shape))}")  # fmt: skip
        lens = forward_batch.seq_lens
        if lens.dtype not in (torch.int32, torch.int64) or lens.numel() * q_len < rows:
            raise RuntimeError(f"K-attn: seq_lens {lens.dtype} {tuple(lens.shape)}")
        extra = q_len if forward_batch.forward_mode.is_target_verify() else 0
        scale = float(self.modules[i].self_attn.attn.scaling)
        return (
            k_cache, v_cache, _ptr(table), table.shape[1], table.numel() * 4, _ptr(lens),
            int(lens.dtype == torch.int64), extra, self.windows[i], 0, _f32_bits(scale),
            _ptr(self.scratch_attn),
        )  # fmt: skip

    def launch(
        self,
        *,
        i: int,
        rows: int,
        q_len: int,
        positions,
        forward_batch,
        stream,
        attn_input,
        pre_norm_dtype,
    ):
        """Layer ``i`` on ``rows`` rows (``q_len`` a request): (stream, next
        layer's attention input)."""
        L = self.views[i]
        kernels = self.use_kernels and rows % q_len == 0
        if not (kernels and (L.kind == "mxfp4" or BF16_K2)):
            out = self.modules[i](
                positions, stream, forward_batch, attn_input=attn_input,
                ffn_boundary=self.boundaries[i], pre_norm_dtype=pre_norm_dtype,
            )  # fmt: skip
            # newer SGLang layers also return an FP8 copy of the next attention input
            return out[0], out[1]
        if self.pool is None:
            raise RuntimeError("Muse Spark monokernels: launch before bind")
        if not (
            stream.dtype == torch.float32
            and stream.is_contiguous()
            and tuple(stream.shape) == (rows, HIDDEN)
        ):
            raise RuntimeError(
                f"Muse Spark launch: stream {tuple(stream.shape)} {stream.dtype}"
            )
        if not (attn_input.dtype == torch.bfloat16 and attn_input.is_contiguous() and tuple(attn_input.shape) == (rows, HIDDEN)):  # fmt: skip
            raise RuntimeError(
                f"Muse Spark launch: attention input {tuple(attn_input.shape)} {attn_input.dtype}"
            )
        slots = forward_batch.out_cache_loc
        if slots.dtype not in (torch.int32, torch.int64) or slots.numel() < rows:
            raise RuntimeError(
                f"Muse Spark launch: out_cache_loc {slots.dtype} {tuple(slots.shape)}"
            )
        if SKIP_LAYERS:
            return stream, attn_input
        if TRACE_DIR:
            self._trace_shape = (rows, q_len)
        k_cache, v_cache, is_swa = self.caches[i]
        m = self.swa_map
        cur = torch.cuda.current_stream()
        cos_sin = L.cos_sin if L.cos_sin is not None else self.any_cos_sin
        if self._k1_done != i:  # else the previous merged launch ran this layer's K1
            self.k1_kernel(rows)(
                _ptr(attn_input), _ptr(L.w_qkvg), _ptr(cos_sin), _ptr(positions), _ptr(slots), _ptr(m),
                k_cache, v_cache, _ptr(self.q), _ptr(self.k), _ptr(self.v), _ptr(self.g),
                _ptr(self.scratch_k1), _ptr(self.epoch), i, int(L.use_rope), int(is_swa),
                int(slots.dtype == torch.int64), int(m.dtype == torch.int64), stream=cur,
            )  # fmt: skip
        self._k1_done = -1
        last = i == len(self.modules) - 1
        if self.merged(i, rows, q_len):
            next_k1 = self._next_k1(i)
            L1 = self.views[i + 1] if next_k1 else L
            k_cache1, v_cache1, is_swa1 = (
                self.caches[i + 1] if next_k1 else self.caches[i]
            )
            cos_sin1 = L1.cos_sin if L1.cos_sin is not None else self.any_cos_sin
            self.layer_kernel(rows, q_len, next_k1, self._kind(i, rows, q_len))(
                _ptr(self.q), _ptr(self.o), *self._attn_args(i, rows, forward_batch, q_len),
                _ptr(self.g), _ptr(stream), _ptr(L.w_o), _ptr(L.alpha), _ptr(L.beta), _ptr(L.pre_w),
                *(_ptr(w) for w in self._router_w(i, rows)), _ptr(L.bias), _ptr(self._pool_w(i, rows)), _ptr(L.xn_w),
                _ptr(self.ids), _ptr(self.scores), _ptr(self.xin), _ptr(self.scratch_k2),
                int(self.modules[i].mlp.local_expert_start), *self._experts(i), _ptr(L.pen_w), _ptr(L.pfw), _ptr(L.alpha2), _ptr(L.beta2), _ptr(L.npw),
                _ptr(L.w_unpool_fp8 if self.fp8_dense else L.w_unpool), _ptr(self.attn_in), _ptr(self.scratch_k3),
                _ptr(L1.w_qkvg), _ptr(cos_sin1), _ptr(positions), _ptr(slots), _ptr(m), k_cache1,
                v_cache1, _ptr(self.k), _ptr(self.v), _ptr(self.scratch_k1), int(L1.use_rope),
                int(is_swa1), int(slots.dtype == torch.int64), int(m.dtype == torch.int64),
                self.peers.local, _ptr(self.peers.addresses), self.rank, _ptr(self.epoch), i,
                int(last), stream=cur,
            )  # fmt: skip
            if next_k1:
                self._k1_done = i + 1
            if CODE_PREFETCH:
                self._pf_note(
                    i,
                    rows,
                    self.layer_kernel(rows, q_len, next_k1, self._kind(i, rows, q_len)),
                )
            return stream, self.attn_in[:rows]
        self.kattn_kernel(rows, q_len)(
            _ptr(self.q), _ptr(self.o), *self._attn_args(i, rows, forward_batch, q_len),
            _ptr(self.epoch), i, stream=cur,
        )  # fmt: skip
        self.k2_kernel(rows)(
            _ptr(self.o), _ptr(self.g), _ptr(stream), _ptr(L.w_o), _ptr(L.alpha), _ptr(L.beta),
            _ptr(L.pre_w), *(_ptr(w) for w in self._router_w(i, rows)), _ptr(L.bias), _ptr(self._pool_w(i, rows)),
            _ptr(L.xn_w), _ptr(self.ids), _ptr(self.scores), _ptr(self.xin), _ptr(self.scratch_k2),
            self.peers.local, _ptr(self.peers.addresses), self.rank, _ptr(self.epoch), i, stream=cur,
        )  # fmt: skip
        if L.kind != "mxfp4":
            return self._bf16_experts(i, rows, stream, pre_norm_dtype)
        self.k3_kernel(rows)(
            _ptr(self.xin), _ptr(self.ids), _ptr(self.scores), int(self.modules[i].mlp.local_expert_start),
            *self._experts(i), _ptr(L.pen_w), _ptr(stream), _ptr(L.pfw),
            _ptr(L.alpha2), _ptr(L.beta2), _ptr(L.npw), _ptr(L.w_unpool_fp8 if self.fp8_dense else L.w_unpool), _ptr(self.attn_in),
            _ptr(self.scratch_k3), self.peers.local, _ptr(self.peers.addresses), self.rank,
            _ptr(self.epoch), i, int(last), stream=cur,
        )  # fmt: skip
        return stream, self.attn_in[:rows]

    def _bf16_experts(self, i: int, rows: int, stream, pre_norm_dtype):
        """A bf16-expert layer's FFN (layers 0 and 61, whose experts stay bf16) after its
        K2, on K2's routed, normed input: under TP experts ``tpx_bf16`` (whole on every
        rank), else SGLang's bf16 experts and the expert-parallel all-reduce; then the
        unpooler fused into the FFN boundary, as the layer's own forward ends."""
        from sglang.kernels.ops.gemm.unpool_gemm_norm_residual import (
            unpool_gemm_norm_residual,
        )
        from sglang.srt.distributed import (
            get_tp_group,
            tensor_model_parallel_all_reduce,
        )

        from sgl_amd_mono.muse_spark.tpx_bf16 import tpx_moe

        moe, boundary = self.modules[i].mlp, self.boundaries[i]
        if self.tpx:
            w = self.tpx_w[i]
            w13, s13, w2, s2 = w if len(w) == 4 else (w[0], None, w[1], None)
            group = get_tp_group()
            if s13 is not None and self.tpx_fused is not None and rows <= 16:
                # FP8 slices: ug + the mids gather + dn + the y gather in one launch over peer stores
                # (bitwise tpx_moe's: the same GEMVs, torch.all_gather's layouts), then the same _fin
                from sgl_amd_mono.muse_spark import tpx_fp8x
                from sgl_amd_mono.muse_spark.tpx_bf16 import _fin

                pb = self.tpx_fused
                rs, cs = w13.shape[1] // 2, w2.shape[1]
                launch, lay = tpx_fp8x.build_fused(
                    rows, TOP_K, MOE_HIDDEN, rs, cs, self.npes
                )
                launch(
                    _ptr(self.xin), _ptr(self.ids), _ptr(w13), _ptr(s13), _ptr(w2), _ptr(s2), pb.local,
                    _ptr(pb.addresses), self.rank, _ptr(self.epoch), i, stream=torch.cuda.current_stream(),
                )  # fmt: skip
                np_ = rows * TOP_K
                ys = (
                    pb.bytes[lay["yout"] : lay["yout"] + self.npes * np_ * cs * 2]
                    .view(torch.bfloat16)
                    .view(self.npes * np_, cs)
                )
                out = torch.empty(
                    (rows, MOE_HIDDEN), dtype=torch.bfloat16, device=self.device
                )
                _fin[(rows,)](
                    ys,
                    self.scores[:rows],
                    self.views[i].pen_w,
                    out,
                    float(moe.post_expert_norm.eps),
                    MOE_HIDDEN,
                    cs,
                    np_,
                    TOP_K,
                    num_warps=8,
                )
            else:
                out = tpx_moe(
                    self.xin[:rows], self.scores[:rows], self.ids[:rows], w13, w2, self.views[i].pen_w,
                    float(moe.post_expert_norm.eps), lambda t: group.all_gather(t, dim=0), s13, s2,
                )  # fmt: skip
        else:
            out = moe.experts(
                self.xin[:rows],
                self.scores[:rows],
                self.ids[:rows],
                moe.local_expert_start,
                moe.post_expert_norm,
            )
            out = tensor_model_parallel_all_reduce(out)
        boundary._ensure_cached_weights(torch.bfloat16)
        attn_input, stream = unpool_gemm_norm_residual(
            out, moe.post_expert_proj.weight, stream, boundary._cc_post_weight, boundary._cc_pre_weight,
            boundary._cc_alpha, boundary._cc_beta, boundary._post_norm.eps, boundary._pre_norm.eps,
            block_n=64, block_k=128, split_k=4, pre_bf16=pre_norm_dtype is torch.bfloat16, num_warps=16,
        )  # fmt: skip
        return stream, attn_input

    def end_step(self) -> None:
        """The next step's epoch, enqueued after this step's last launch."""
        if TRACE_DIR:
            self._trace_copy_out()  # captured into the graph with the step
        self.epoch[0].add_(1)
        self._k1_done = -1
