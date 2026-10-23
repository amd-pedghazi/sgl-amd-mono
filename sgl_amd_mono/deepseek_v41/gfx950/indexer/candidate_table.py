# SPDX-License-Identifier: Apache-2.0
# ``_reindex_table_kernel`` from sgl-project/sglang#43984 at 134919476c
# (kernels/ops/mono/gfx950/deepseek_v41/indexer/candidate_table.py)
"""A REINDEX layer's compact columns: K2i's key table for them, and their selection mapped back.

Outputs go to caller-owned buffers, so a replayed graph reads the captured addresses."""

import torch
import triton
import triton.language as tl

from . import index_plan as ip

PLANE_ROWS = 64  # SGLang's FP4 index plane page (``compressed_page_size``)


@triton.jit
def _reindex_table_kernel(
    ids,
    block_tables,
    batch_ids,
    visible,
    ids_out,
    table,
    context,
    ids_stride,
    page_stride,
    topk_blocks,
    UNITS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """SGLang's candidate blocks (unordered, -1 after the kept ones) with the newest moved last.

    The newest block's rows past ``visible`` are not keys yet, so with it last ``context`` (the
    full blocks, then its visible rows) stops the scorer short of them."""
    token = tl.program_id(0)
    batch = tl.load(batch_ids + token)
    slots = tl.arange(0, BLOCK)
    mask = slots < topk_blocks
    cand = tl.load(ids + token * ids_stride + slots, mask=mask, other=-1)
    live = cand >= 0
    kept = tl.sum(live.to(tl.int32))
    seen = tl.load(visible + token).to(tl.int32)
    newest = (seen - 1) // ROWS
    at = tl.max(tl.where(live & (cand == newest), slots, -1))
    last = kept - 1
    moved = tl.sum(tl.where(slots == last, cand, 0))
    swap = at >= 0
    cand = tl.where(
        swap & (slots == at), moved, tl.where(swap & (slots == last), newest, cand)
    )
    tl.store(ids_out + token * ids_stride + slots, cand, mask=mask)
    page = tl.load(
        block_tables + batch * page_stride + cand // UNITS, mask=live, other=0
    )
    tl.store(
        table + token * ids_stride + slots,
        tl.where(live, page * UNITS + cand % UNITS, 0),
        mask=mask,
    )
    tail = tl.sum(tl.where(slots == last, cand, 0))
    last_span = tl.minimum(tl.maximum(seen - tail * ROWS, 0), ROWS)
    tl.store(context + token, tl.where(kept > 0, (kept - 1) * ROWS + last_span, 0))


def reindex_table(
    *,
    ids: torch.Tensor,
    page_table: torch.Tensor,
    owners: torch.Tensor,
    bounds: torch.Tensor,
    ids_out: torch.Tensor,
    table: torch.Tensor,
    context: torch.Tensor,
) -> None:
    """SGLang's ``CandidateBlocks.ids`` -> ``ids_out`` (newest block last), ``table``, ``context``.

    ``table`` holds their virtual 8-row pages, ``context`` [rows] the compact columns each row's
    scorer reads; ``owners[t]`` is row t's page-table row."""
    rows, topk_blocks = ids.shape
    assert ids.stride(1) == 1 and ids_out.stride(0) == table.stride(0) == ids.stride(0)
    _reindex_table_kernel[(rows,)](
        ids=ids,
        block_tables=page_table,
        batch_ids=owners,
        visible=bounds,
        ids_out=ids_out,
        table=table,
        context=context,
        ids_stride=ids.stride(0),
        page_stride=page_table.stride(0),
        topk_blocks=topk_blocks,
        UNITS=PLANE_ROWS // ip.BLOCK_ROWS,
        ROWS=ip.BLOCK_ROWS,
        BLOCK=triton.next_power_of_2(topk_blocks),
    )


@triton.jit
def _map_compact_kernel(
    compact_pos,
    ids,
    page_table,
    raw_out,
    page_out,
    ids_stride,
    table_stride,
    out_stride,
    TOPK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
):
    """One program a row: each selected compact column -> its position and pool slot; -1 stays."""
    row = tl.program_id(0)
    j = tl.arange(0, TOPK)
    c = tl.load(compact_pos + row * TOPK + j)
    valid = c >= 0
    col = tl.where(valid, c, 0)
    block = tl.load(ids + row * ids_stride + col // BLOCK_ROWS, mask=valid, other=0)
    pos = block * BLOCK_ROWS + col % BLOCK_ROWS
    page = tl.load(
        page_table + row * table_stride + pos // PAGE_ROWS, mask=valid, other=0
    )
    tl.store(raw_out + row * out_stride + j, tl.where(valid, pos, -1))
    tl.store(
        page_out + row * out_stride + j,
        tl.where(valid, page * PAGE_ROWS + pos % PAGE_ROWS, -1),
    )


def map_compact_selection(
    *,
    compact_pos: torch.Tensor,
    ids: torch.Tensor,
    page_table: torch.Tensor,
    raw_out: torch.Tensor,
    page_out: torch.Tensor,
) -> None:
    """A REINDEX layer's top-k of its compact columns -> SGLang's raw and page indices.

    ``compact_pos`` is -1 padded after the valid ones, ``ids`` is ``reindex_table``'s
    ``ids_out``, and ``raw_out`` / ``page_out`` are ``sparse_raw_indices`` /
    ``sparse_page_indices``."""
    rows = compact_pos.shape[0]
    assert compact_pos.stride(0) == ip.TOPK and raw_out.stride(0) == page_out.stride(0)
    _map_compact_kernel[(rows,)](
        compact_pos=compact_pos,
        ids=ids,
        page_table=page_table,
        raw_out=raw_out,
        page_out=page_out,
        ids_stride=ids.stride(0),
        table_stride=page_table.stride(0),
        out_stride=raw_out.stride(0),
        TOPK=ip.TOPK,
        BLOCK_ROWS=ip.BLOCK_ROWS,
        PAGE_ROWS=PLANE_ROWS,
    )
