# SPDX-License-Identifier: Apache-2.0
"""D-draft: the DFlash2 draft's layers as one launch, for S = ``tokens`` rows on a
rank of a TP group, each layer D-pre -> D-attn -> D-post -> D-MLP
(``draft_attn_half.py``, ``draft_attn.py``, ``draft_mlp.py``), their arithmetic
unchanged.

The launch boundaries between them become hand-offs (``dl_flags``): each CTA
raises its head flag once its D-pre stages are done (the attention tasks poll
every CTA's: D-post reads the coefficients any CTA reduced, D-MLP rewrites the
rows every CTA's D-pre read), the level-2 merges put the o rows as tagged pairs
(the o GEMV polls them), each D-post unit raises its attn_out flag (a D-MLP CTA
polls the 16 of its K part), each D-MLP CTA its out flag under the next layer's
tag (the next D-pre CTA polls the 16 of its K part). So every
wait of a stage follows, through a chain of hand-offs, every CTA's earlier
stages. The data behind a flag is stored and loaded past the L2 (another XCD's
CTA wrote it). Each CTA's W_o rows go in flight after its attention tasks. With
TP > 1 each rank computes its share of the two kernel projections and pushes it
to every rank (``draft_mlp.coef_shard``), and the two TP sums of a layer are a
reduce-scatter and an all-gather (``draft_mlp.stage_out``). The LDS is the union of the stages' (a
barrier between them); their scratch and peer regions stay separate.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import msgspec
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import CM_DEV, fresh, load_ptr64, peer_bases
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, preg, sreg
from sgl_amd_mono.muse_spark.gfx950 import compat
from sgl_amd_mono.muse_spark.gfx950 import draft_attn as DA
from sgl_amd_mono.muse_spark.gfx950 import draft_attn_half as DH
from sgl_amd_mono.muse_spark.gfx950 import draft_mlp as D
from sgl_amd_mono.muse_spark.gfx950.config import MAX_TOKENS
from sgl_amd_mono.muse_spark.gfx950.device import step_tag

BRIDGE = compat.install()
SOURCES = kernel_sources(__name__)
_STREAM = fx.Stream(None)

Q_LEN = 8  # rows a request: the draft block
LAYERS = 5
# a layer's row of the weight table (int64 addresses)
WEIGHTS = (
    "w_in",
    "w_kpa",
    "base_a",
    "w_qkv",
    "q_norm",
    "k_norm",
    "w_o",
    "w_post",
    "w_kp",
    "base",
    "w_gu",
    "w_down",
)


def dl_flags_layout() -> dict:
    """The launch's flags (pairs): a head, an attn_out and an out flag a CTA; the o box:
    the attention output as tagged bf16 pairs."""
    return pair_layout(
        [
            ("dl_hflag", BLOCKS),
            ("dl_aflag", BLOCKS),
            ("dl_xflag", BLOCKS),
            ("dl_obox", MAX_TOKENS * DA.Q_K // 2),
        ],
        align=256,
    )


class DDraftBuild(msgspec.Struct, frozen=True):
    tokens: int
    npes: int = 8
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


def _lds_union(key: DDraftBuild):
    S = key.tokens
    ALoop, ATree = DA.dattn_lds(
        min(DA.DAttnBuild(S, Q_LEN).nt, (DA.attn_rows(Q_LEN) + 15) // 16)
    )
    PreLds, PostLds, MlpLds = DH.dp_lds(S), DH.do_lds(S), D.dm_lds(S)

    @fx.union
    class Lds:
        loop: ALoop  # type: ignore[valid-type]
        tree: ATree  # type: ignore[valid-type]
        pre: PreLds  # type: ignore[valid-type]
        post: PostLds  # type: ignore[valid-type]
        mlp: MlpLds  # type: ignore[valid-type]

    return Lds


def _own_lanes(c: dict) -> dict:
    """``c`` with its thread ids through an opaque move: the stage's lane math is
    computed in the stage, not hoisted over the launch (or out of D-draft's layer
    loop) into registers that spill."""
    tid = fresh(fx.thread_idx.x)
    c.update(tid=tid, lane=tid % 64, wave=tid // 64)
    return c


def run_layer(key: DDraftBuild, lds, a: dict, tag, layer):
    """Layer ``layer``'s four stages on this CTA (``a``: the launch's arguments with
    the layer's weights and caches)."""
    S, npes = key.tokens, key.npes
    fs, fa, tr = key.fault_site, key.fault_after, key.trace
    fl = dl_flags_layout()
    flags = a["flags"]
    hflag, aflag, xflag, obox = (
        sreg(flags, fl[r][0], r)
        for r in ("dl_hflag", "dl_aflag", "dl_xflag", "dl_obox")
    )
    # D-pre: q, the block's K / V, res' and the coefficients past the L2; the head flags
    pre = lds.pre.peek()
    p_args = {
        "x": a["x"], "res": a["res"], "first": a["first"], "w_in": a["w_in"], "w_kp": a["w_kpa"], "base": a["base_a"],
        "w_qkv": a["w_qkv"], "q_norm": a["q_norm"], "k_norm": a["k_norm"], "cos_sin": a["cos_sin"],
        "positions": a["positions"], "pos64": a["pos64"], "slots": a["slots"], "slot64": a["slot64"],
        "k_cache": a["k_cache"], "v_cache": a["v_cache"], "res_out": a["res_pre"], "q": a["q"],
    }  # fmt: skip
    post_peers = peer_bases(a["peers_post"], npes)
    dpc = DH.do_sym_layout(npes)["dpc"][0]
    if npes == 1:
        p_args["coef_out"] = a["coef"]  # (else D-post polls each rank's share)
    members = {"red": pre.red.ptr, "hbuf": pre.hbuf.ptr, "hc": pre.hc.ptr, "ssl": pre.ssl.ptr, "nf": pre.nf.ptr, "cf": pre.cf.ptr, "hrow": pre.hrow.ptr, "hss": pre.hss.ptr}  # fmt: skip
    extra = {
        "xcm": CM_DEV,
        "hflag": hflag,
        "xflag": xflag,
    }  # xflag: the previous layer's out (from the second layer)
    cp = DH._common(
        DH.DPreBuild(S, fs, fa, tr), members, p_args, tag, extra, layer=layer
    )
    for region, (off, _) in DH.dp_scratch_layout().items():
        cp[region[3:]] = sreg(a["scratch_pre"], off, region)
    if npes > 1:
        # the kernel projection: this rank's 1 / npes of the rows, pushed to every rank
        D.coef_shard(
            cp,
            npes,
            a["rank"],
            preg(a["sym_post"], dpc, "dpc"),
            [preg(b, dpc, "dpc") for b in post_peers],
        )
    else:
        D.coef_shard(cp, 1, 0)
    DH.run_dpre(_own_lanes(cp))
    gpu.barrier()
    # D-post (its context first: the attention prefetches its weights): the o box
    # polled, the attn_out flags
    post = lds.post.peek()
    do_off = DH.do_sym_layout(npes)["do"][0]
    extra = {
        "W": npes, "rank": a["rank"], "own_do": preg(a["sym_post"], do_off, "do"),
        "peer_do": [preg(b, do_off, "do") for b in post_peers],
        "xcm": CM_DEV, "o_box": obox, "aflag": aflag,
    }  # fmt: skip
    if npes > 1:
        dao = DH.do_sym_layout(npes)["dao"][0]
        extra.update(
            coef_box=preg(a["sym_post"], dpc, "dpc"), ao_box=preg(a["sym_post"], dao, "dao"),
            peer_ao=[preg(b, dao, "dao") for b in post_peers],
        )  # fmt: skip
    o_args = {
        "o": a["o"],
        "w_o": a["w_o"],
        "coef": a["coef"],
        "base": a["base_a"],
        "attn_out": a["attn_out"],
    }
    members = {"red": post.red.ptr, "dbuf": post.dbuf.ptr, "cf1": post.cf1.ptr}
    co = DH._common(
        DH.DPostBuild(S, npes, fs, fa, tr), members, o_args, tag, extra, layer=layer
    )
    for region, (off, _) in DH.do_scratch_layout().items():
        co[region[3:]] = sreg(a["scratch_post"], off, region)
    # D-MLP's context (the attention prefetches its weights too): its K part's attn_out
    # flags before the norm, its out flags
    m_args = {
        "res": a["res_pre"], "attn_out": a["attn_out"], "w_post": a["w_post"], "w_kp": a["w_kp"], "base": a["base"],
        "w_gu": a["w_gu"], "w_down": a["w_down"], "res_out": a["res_out"], "out": a["out"],
    }  # fmt: skip
    cm = D.dmlp_context(
        D.DMLPBuild(S, npes, fs, fa, tr), lds.mlp.peek(), m_args, a["scratch_mlp"], a["sym_mlp"], a["peers_mlp"],
        a["rank"], tag, D.dm_scratch_layout(), D.dm_sym_layout(npes), layer=layer,
    )  # fmt: skip
    cm.update(aflag=aflag, xcm=CM_DEV, xflag=xflag, put_next=Mailbox(tag + 1).put)
    # D-attn: the head flags before the tasks, the o rows into the o box
    attn_key = DA.DAttnBuild(S, Q_LEN, fault_site=fs, fault_after=fa, trace=tr)
    a_args = {
        "q": a["q"], "o": a["o"], "k_cache": a["k_cache"], "v_cache": a["v_cache"], "page_table": a["page_table"],
        "table_stride": a["table_stride"], "table_bytes": a["table_bytes"], "req_pool": a["req_pool"],
        "rpi64": a["rpi64"], "seq_lens": a["seq_lens"], "lens64": a["lens64"], "extra": fx.Int32(Q_LEN),
        "window": fx.Int32(0), "part_in": fx.Int32(0), "scale_bits": a["scale_bits"],
    }  # fmt: skip
    ca = DA.dattn_context(
        attn_key,
        lds,
        a_args,
        a["scratch_attn"],
        tag,
        DA.dattn_scratch_layout(S, Q_LEN),
        layer=layer,
    )
    ca.update(xcm=CM_DEV, hflag=hflag, h_flags=BLOCKS)
    # W_o of this CTA's D-post rows in flight once its attention tasks are done (and
    # before its merges: they wait the tasks longer than the loads take)
    # and the gate / up GEMV's first round into D-MLP's LDS (but on the merge CTAs, the
    # last ones: their merges would wait behind it; they load it at D-MLP's entry)
    early = fx.block_idx.x < BLOCKS - S // Q_LEN * (
        (DA.attn_rows(Q_LEN) + 15) // 16
    ) * (DA.HEAD_DIM // 64)  # (the merge CTAs: a job a dim pass)
    cm["gu_early"] = early.select(fx.Int32(1), fx.Int32(0))

    def prefetch():
        ws = DH.o_loads(_own_lanes(co), co["bid"])
        D.gu_dma(_own_lanes(cm), early)
        return ws

    ws = DA.run_dattn(_own_lanes(ca), o_box=obox, prefetch=prefetch)
    gpu.barrier()
    DH.run_dpost(_own_lanes(co), ws)
    gpu.barrier()
    D.run_dmlp(_own_lanes(cm))
    gpu.barrier()  # the next layer's stages reuse the LDS


def run_table_layer(key: DDraftBuild, lds, a: dict, epoch, layer):
    """Layer ``layer``: its weights from ``wtab``, its caches the layer's of ``kv``; its
    input the first layer's x or the previous layer's out (res: res_out)."""
    layer = fx.Int32(layer)
    first = layer == 0
    la = dict(a)
    for j, nm in enumerate(WEIGHTS):
        la[nm] = load_ptr64(a["wtab"], layer * len(WEIGHTS) + j)
    k, v = a["kv"][0]
    for i in range(1, LAYERS):
        k, v = (
            (layer == i).select(a["kv"][i][0], k),
            (layer == i).select(a["kv"][i][1], v),
        )
    la.update(k_cache=k, v_cache=v)
    la["first"] = first.select(fx.Int32(1), fx.Int32(0))
    la["x"] = first.select(a["x"], a["out"])
    la["res"] = first.select(a["x"], a["res_out"])
    run_layer(key, lds, la, step_tag(epoch, layer), layer)


def build_ddraft(key: DDraftBuild):
    """``@flyc.jit`` launcher of D-draft for ``key.tokens`` rows on a rank of a
    ``key.npes``-way TP group. Arguments: ``wtab`` (int64 [LAYERS, len(WEIGHTS)]: the
    layers' weights), ``k0`` / ``v0`` .. (the layers' fp8 caches), D-pre's other
    arguments (``build_dpre``; x the first layer's input), D-attn's (``build_dattn``,
    its q / o shared), D-post's (``build_dpost``, its coef D-pre's), D-MLP's
    (``build_dmlp``: its res D-pre's res_out, its attn_out D-post's; out / res_out
    the last layer's), the flags (``dl_flags_layout`` bytes), ``rank``, ``epoch``."""
    S, npes = key.tokens, key.npes
    assert S % Q_LEN == 0 and 1 <= S <= MAX_TOKENS and LAYERS == 5
    name = kernel_symbol(
        "muse_spark_ddraft", s=S, w=npes, f=key.fault_site, t=int(key.trace)
    )
    keyed = key_tuple(key, SOURCES)
    Lds = _lds_union(key)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def ddraft(
        wtab: Int64, k0: Int64, v0: Int64, k1: Int64, v1: Int64, k2: Int64, v2: Int64, k3: Int64, v3: Int64,
        k4: Int64, v4: Int64,
        x: Int64, cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, res_pre: Int64,
        q: Int64, coef: Int64, scratch_pre: Int64,
        o: Int64, page_table: Int64, table_stride: Int32, table_bytes: Int32, req_pool: Int64, rpi64: Int32,
        seq_lens: Int64, lens64: Int32, scale_bits: Int32, scratch_attn: Int64,
        attn_out: Int64, scratch_post: Int64, sym_post: Int64, peers_post: Int64,
        res_out: Int64, out: Int64, scratch_mlp: Int64, sym_mlp: Int64, peers_mlp: Int64,
        flags: Int64, rank: Int32, epoch: Int64,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(
                epoch, fault_site=key.fault_site or None, fault_after=key.fault_after
            )
        ):
            lds = fx.SharedAllocator().allocate(Lds)
            a = {
                "wtab": wtab, "kv": [(k0, v0), (k1, v1), (k2, v2), (k3, v3), (k4, v4)],
                "x": x, "cos_sin": cos_sin, "positions": positions, "pos64": pos64,
                "slots": slots, "slot64": slot64, "res_pre": res_pre, "q": q, "coef": coef, "scratch_pre": scratch_pre,
                "o": o, "page_table": page_table, "table_stride": table_stride, "table_bytes": table_bytes,
                "req_pool": req_pool, "rpi64": rpi64, "seq_lens": seq_lens, "lens64": lens64,
                "scale_bits": scale_bits, "scratch_attn": scratch_attn, "attn_out": attn_out,
                "scratch_post": scratch_post, "sym_post": sym_post, "peers_post": peers_post, "res_out": res_out,
                "out": out, "scratch_mlp": scratch_mlp, "sym_mlp": sym_mlp, "peers_mlp": peers_mlp, "flags": flags,
                "rank": rank,
            }  # fmt: skip
            for i in range(LAYERS):
                run_table_layer(key, lds, a, epoch, i)

    @flyc.jit
    def launch(
        wtab: Int64, k0: Int64, v0: Int64, k1: Int64, v1: Int64, k2: Int64, v2: Int64, k3: Int64, v3: Int64,
        k4: Int64, v4: Int64,
        x: Int64, cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, res_pre: Int64,
        q: Int64, coef: Int64, scratch_pre: Int64,
        o: Int64, page_table: Int64, table_stride: Int32, table_bytes: Int32, req_pool: Int64, rpi64: Int32,
        seq_lens: Int64, lens64: Int32, scale_bits: Int32, scratch_attn: Int64,
        attn_out: Int64, scratch_post: Int64, sym_post: Int64, peers_post: Int64,
        res_out: Int64, out: Int64, scratch_mlp: Int64, sym_mlp: Int64, peers_mlp: Int64,
        flags: Int64, rank: Int32, epoch: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        ddraft(
            wtab, k0, v0, k1, v1, k2, v2, k3, v3, k4, v4, x, cos_sin, positions, pos64, slots, slot64, res_pre, q,
            coef, scratch_pre, o, page_table, table_stride, table_bytes, req_pool, rpi64, seq_lens, lens64,
            scale_bits, scratch_attn, attn_out, scratch_post, sym_post, peers_post, res_out, out, scratch_mlp,
            sym_mlp, peers_mlp, flags, rank, epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
