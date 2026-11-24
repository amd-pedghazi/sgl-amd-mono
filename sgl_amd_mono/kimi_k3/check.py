# SPDX-License-Identifier: Apache-2.0
"""SGLANG_AMD_MONO_CHECK=1 (eager only): every layer K1 takes also runs on
SGLang's own attention, on the caches as they were, and the launch is checked
against it: the attention output, the bank row it contributes, the recurrent
state it leaves, and the conv window it hands back. Each is scored against its
``spec.NUMERICS`` bound. Every check is logged; a miss is logged as a violation.

The launch reads every tensor as a raw address, so a wrong value is not a
crash. It is carried into the next step and into the draft's input, and the
server keeps serving. The kernel checks (``test/kernels/kimi_k3``) score the
launch on synthetic weights, one layer and one step at a time. This scores it
on the real weights, at TP8, with the state of one step carried into the next.

K1 hands the conv window back once a step, after every layer, while SGLang's
attention writes it inside the layer. That one output is therefore compared
after ``k1.finish()``, in ``compare_handoff``; everything else is compared in
``compare``, right after the launch.

This runs on eager steps only. A captured graph does not run Python on replay,
and the reductions here would be captured into the graph. Start the server with
--disable-cuda-graph to make every decode step eager.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
}


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return {
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
    }


def _kda_slices(built, packed):
    """The KDA pool entries SGLang's own attention writes for this layer.

    One layer and one slot each, so saving them costs a few MiB and not the
    whole pool.
    """
    st = built._k1_state
    mi = packed.k1.mamba_idx
    row = st["row"]
    pairs = [
        ("ssm", st.get("inter_ssm"), (mi, row)),
        ("window", st.get("inter_window"), (mi, row)),
    ]
    return [(n, t, idx) for n, t, idx in pairs if t is not None]


def reference(built, layer, packed, hidden, bank, forward_batch, zero_allocator):
    """SGLang's own attention on this layer's input, with the caches put back.

    Runs before K1 on the same input. The stage's prepare() returns a new
    stream and assigns it, so restoring the previous stream is an assignment.
    The bank row and count it adds, and the KDA pool entries it writes, are
    saved and written back, so K1 then runs on the caches as they were.
    """
    if torch.cuda.is_current_stream_capturing():
        return None
    if built._k1_state is None or not isinstance(hidden, torch.Tensor):
        return None
    saved_stream = forward_batch.residual_stream
    slices = _kda_slices(built, packed)
    saved = [(t, idx, t[idx].clone()) for _n, t, idx in slices]
    # SGLang's attention appends a bank row and counts it. K1 counts its own
    # row instead, so the reference's row and its count both have to go back.
    saved_nvb = bank.num_valid_blocks
    saved_bank = bank.block_residual.clone()
    try:
        # The stream checks that this is the tensor it recorded as written, so
        # it has to be hidden itself and not a copy. prepare() returns a new
        # tensor and does not write through this one.
        x = layer.attn_boundary.prepare(hidden, forward_batch, capture_gathered=None)
        core = built._pre_o_proj(layer, x, forward_batch, zero_allocator)
        row = None
        if bank.num_valid_blocks > saved_nvb:
            # DSpark's capture reads the bank, so a row that differs changes
            # the draft's input while leaving the layer's own output right.
            row = bank.block_residual[:, saved_nvb, :].clone()
        return {
            "core": core.detach().clone(),
            "row": row,
            "nvb": saved_nvb,
            "state": {n: t[idx].clone() for n, t, idx in slices},
            "slices": slices,
        }
    except Exception as exc:  # a check must not take the server down
        logger.warning(
            "monokernel check reference failed on layer %d: %r", packed.i, exc
        )
        return None
    finally:
        for t, idx, v in saved:
            t[idx] = v
        bank.block_residual.copy_(saved_bank)
        bank.num_valid_blocks = saved_nvb
        forward_batch.residual_stream = saved_stream


def compare(built, i: int, core: torch.Tensor, ref, bank) -> None:
    """The launch against the reference, for everything it writes in the layer.

    The conv window is left to compare_handoff: K1 has not written it yet.
    """
    if ref is None or torch.cuda.is_current_stream_capturing():
        return
    outputs = {}
    if ref["core"].shape == core.shape:
        outputs["core"] = (core.detach(), ref["core"])
    if ref["row"] is not None:
        outputs["bank"] = (bank.block_residual[:, ref["nvb"], :], ref["row"])
    for name, t, idx in ref["slices"]:
        if name == "window":
            continue
        got, want = t[idx], ref["state"][name]
        if got.shape == want.shape:
            outputs[name] = (got, want)
    _report(i, outputs)
    built._check_handoff.append((i, ref["slices"], ref["state"]))


def compare_handoff(built) -> None:
    """The conv window K1 hands back, after finish() has written it.

    Called once a step. Nothing reads the window before finish(), which is what
    makes this the output a layer's own comparison cannot cover.
    """
    pending = getattr(built, "_check_handoff", None)
    if not pending:
        return
    for i, slices, want in pending:
        for name, t, idx in slices:
            if name != "window":
                continue
            got = t[idx]
            if got.shape == want[name].shape:
                _report(i, {"window": (got, want[name])})
    built._check_handoff = []


def _profile(t: torch.Tensor, keep: int) -> str:
    """The norm along one axis, with every other axis summed out."""
    dims = [d for d in range(t.dim()) if d != keep]
    norms = torch.linalg.vector_norm(t.float(), dim=dims)
    return ",".join(f"{v:.6g}" for v in norms.tolist())


def _profile_lines(name: str, got: torch.Tensor, want: torch.Tensor) -> list:
    """Where inside the output the two differ. Built for a violation only.

    The metrics say an output is wrong. They do not say which part of it is,
    and the two are not the same question: a hand-off that writes the right
    values into the wrong entries scores the same as one that writes the wrong
    values. The norm by token names the rows that moved. For the conv window
    the norm by column also names the entries of each row, which is what
    separates a partial write from a wrong one -- a write that covers one
    column of the window leaves that column right and the others stale.
    """
    if got.dim() < 2 or got.shape != want.shape:
        return []
    # [1, tokens, ...] for a pool slice, [tokens, ...] for core and bank
    tok = 1 if got.dim() >= 4 else 0
    lines = [f"k1.{name} by_token got {_profile(got, tok)} want {_profile(want, tok)}"]
    if name == "window":
        lines.append(
            f"k1.{name} by_col got {_profile(got, tok + 1)} "
            f"want {_profile(want, tok + 1)}"
        )
    return lines


def _report(i: int, outputs: dict) -> None:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k3.spec import NUMERICS

    if not outputs or get_parallel().tp_rank != 0:
        return
    values = {
        f"k1.{out}.{metric}": v
        for out, (got, want) in outputs.items()
        for metric, v in _metrics(got, want).items()
    }
    logger.info(
        "monokernel check layer %d: %s",
        i,
        " ".join(f"{k}={v:.8g}" for k, v in values.items()),
    )
    missed = set()
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.8g, bound %g",
                i,
                name,
                v,
                bound,
            )
            missed.add(name.split(".")[1])
    # Once an output, however many of its metrics missed.
    for out in sorted(missed):
        for line in _profile_lines(out, *outputs[out]):
            logger.error("monokernel check violation: layer %d %s", i, line)
