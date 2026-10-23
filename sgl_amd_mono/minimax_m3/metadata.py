# SPDX-License-Identifier: Apache-2.0
"""Replay-fresh causal rows for SGLang decode and linear EAGLE verification."""

import torch


def prepare_token_rows(
    req_to_token,
    req_pool_indices,
    seq_lens,
    block_columns,
    out_block_table,
    out_seq_lens,
    slot_mapping,
    query_len: int,
    is_verify: bool,
    max_context_len: int,
):
    """Expand each request to ``query_len`` rows without exposing future tokens.

    SGLang's TARGET_VERIFY ``seq_lens`` hold the accepted prefix (ATOM's hold the total
    length), so query j sees prefix + j + 1 keys. It runs in the captured graph, fresh
    on every replay. A row whose slot is 0 (the KV allocator's graph padding) or
    negative (no cache write) is zeroed after the causal arithmetic, so it cannot join
    the wide MoE's expert union.
    """
    n = out_seq_lens.numel()
    token = torch.arange(n, dtype=torch.int64, device=seq_lens.device)
    request = torch.div(token, query_len, rounding_mode="floor")
    rows = req_pool_indices[request].to(torch.int64)
    gathered = req_to_token[rows[:, None], block_columns[None, :]]
    torch.floor_divide(gathered, 128, out=out_block_table)
    lengths = seq_lens[request]
    if is_verify:
        lengths = lengths + (token % query_len + 1).to(lengths.dtype)
        # the scheduler drops lookahead past the context limit: those rows need only stay in bounds
        lengths = lengths.clamp(max=max_context_len)
    out_seq_lens.copy_(lengths)
    padding = slot_mapping[:n] <= 0
    out_seq_lens.masked_fill_(padding, 0)
    out_block_table.masked_fill_(padding[:, None], 0)
    return out_block_table, out_seq_lens
