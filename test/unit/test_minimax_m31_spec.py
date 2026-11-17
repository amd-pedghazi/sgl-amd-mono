"""The MiniMax-M3.1-Flash FFN monokernel spec on CPU: which checkpoint selects it, and
the shared-expert shard its load hook keeps for a TP rank."""

from types import SimpleNamespace
from unittest import mock

import torch

from sgl_amd_mono.minimax import spec as family
from sgl_amd_mono.minimax_m3 import spec as m3
from sgl_amd_mono.minimax_m31 import spec as m31

# MiniMax-M3.1-Flash's config.json: the quantization config sits at the top level
M31 = SimpleNamespace(
    architectures=["MiniMaxM3SparseForConditionalGeneration"],
    quantization_config={"quant_method": "mxfp8", "moe_quant_algo": "NVFP4"},
    text_config=SimpleNamespace(hidden_size=6144),
)
M3 = SimpleNamespace(
    architectures=["MiniMaxM3SparseForConditionalGeneration"],
    text_config=SimpleNamespace(quantization_config={"quant_method": "quark"}),
)


def test_each_checkpoint_selects_its_own_spec():
    assert family._model(M31) is m31
    assert family._model(M3) is m3


def test_the_load_hook_keeps_this_ranks_shard():
    """gate / up shard their rows and down its columns: a slice of the other dimension
    would hand the launch another rank's shared expert."""
    rows, cols, rank = 3072, 6144, 1
    ckpt = "language_model.model.layers.7.block_sparse_moe.shared_experts"
    full = {
        "gate": torch.arange(rows * cols, dtype=torch.int32).view(rows, cols),
        "down": torch.arange(cols * rows, dtype=torch.int32).view(cols, rows),
    }
    copies = {}
    with (
        mock.patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(tp_rank=rank, tp_size=4),
        ),
        mock.patch.object(torch.cuda, "current_device", return_value="cpu"),
    ):
        for proj, tensor in full.items():
            m31.load_tensor(copies, None, f"{ckpt}.{proj}_proj.weight", tensor)
        m31.load_tensor(copies, None, "model.layers.7.mlp.gate.weight", full["gate"])
    shard = rows // 4
    gate = copies["shared expert mxfp8", 7, "gate", "weight"]
    down = copies["shared expert mxfp8", 7, "down", "weight"]
    assert torch.equal(
        gate.view(torch.int32), full["gate"][rank * shard : (rank + 1) * shard]
    )
    assert torch.equal(
        down.view(torch.int32), full["down"][:, rank * shard : (rank + 1) * shard]
    )
    assert len(copies) == 2, "the hook kept a tensor that is not a shared expert's"


def test_both_models_serve_the_shared_specs_steps():
    """SGLang routes M3 and M3.1 by the shared spec's device limits, so a change to
    either model's widths or TP sizes must change the other's too."""
    assert m3.SPEC.devices == m31.SPEC.devices == family.SPEC.devices


def test_the_shared_spec_rechecks_every_requirement():
    """The shared spec re-checks each model's Requirements as SGLang's guard does,
    except dtypes, which SGLang checks on the shared spec alone; a field SGLang adds
    would go unchecked for both models."""
    from sglang.srt.monokernels.spec import Requirements

    rechecked = {"attention_backends", "server_args", "hf_config", "env"}
    assert set(Requirements.__struct_fields__) == rechecked | {"dtypes"}
    assert m3.SPEC.requires.dtypes == m31.SPEC.requires.dtypes
    assert family.SPEC.requires.dtypes == m3.SPEC.requires.dtypes


def test_what_the_shared_spec_takes_from_one_model_agrees():
    """SGLang reads the fields the shared spec does not hand to the checkpoint's own
    spec from the shared spec alone, so both models must agree on them."""
    from sglang.srt.monokernels.spec import MonoSpec

    dispatched = {
        "features",
        "requires",
        "model_disable_reasons",
        "build",
        "config_disable_reasons",
        "load_tensor",
    }
    shared = {"devices", "api_version", "tokens_per_request", "unread_metadata"}
    # only the kind: each model's check mode reads its own bounds
    assert set(MonoSpec.__struct_fields__) == dispatched | shared | {"name", "numerics"}
    for field in shared:
        assert getattr(m3.SPEC, field) == getattr(m31.SPEC, field), field
        assert getattr(family.SPEC, field) == getattr(m3.SPEC, field), field
    assert m3.SPEC.numerics.kind == m31.SPEC.numerics.kind == family.SPEC.numerics.kind
