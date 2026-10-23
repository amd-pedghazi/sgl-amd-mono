"""The GLM-5.3 gfx950 monokernel spec on CPU (RFC #43949): the layers its forward
covers on the GLM-5.3 layer structure, its row routing and TP4 row cap, the keys
each row of a decode or MTP verify step attends, its registry entry, its config and
model refusal reasons, and the layer check's top-k metric and cache poisoning.
SGLang checks its declared requirements, the same way for every spec."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from sglang.srt.layers.moe.utils import MoeRunnerBackend

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.glm53 import HF_CONFIG, check
from sgl_amd_mono.glm53 import spec as glm53
from sgl_amd_mono.glm53.gfx950.weights import rope_tables

# GLM-5.3's layer structure and quantization (its config.json)
GLM53 = SimpleNamespace(
    **HF_CONFIG,
    num_hidden_layers=78,
    first_k_dense_replace=3,
    moe_layer_freq=1,
    quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    rope_parameters={"rope_theta": 8000000, "rope_type": "default"},
    architectures=["GlmMoeDsaForCausalLM"],
)
# the server arguments the spec reads before load: no speculative decoding
CFG = SimpleNamespace(page_size=None, speculative_algorithm=None)
# MTP: SGLang's GLM-5.3 default draft chain (NEXTN resolves to EAGLE)
MTP = SimpleNamespace(
    page_size=None,
    speculative_algorithm="EAGLE",
    speculative_eagle_topk=1,
    speculative_num_steps=3,
    speculative_num_draft_tokens=4,
)
# the graph keys of a verify step of one request (4 rows) and a decode step of 4
VERIFY = registry.ShapeKey(
    forward_mode="target_verify", batch_size=1, tokens_per_request=4
)
DECODE = registry.ShapeKey(forward_mode="decode", batch_size=4, tokens_per_request=1)


def test_the_forward_covers_the_moe_layers():
    # the dense layers 0 to 2 run SGLang's path inside the forward
    assert glm53._covered_layers(GLM53) == list(range(3, 78))


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    arch = ["GlmMoeDsaForCausalLM"]
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx950"
    )
    assert found is glm53.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx942"
    )
    assert found is None and "gfx942" in why


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


def _built(lm, layers, runtimes, tp=8, context_len=202752):
    return glm53._Built(
        lm, layers, runtimes, None, None, tp, "cpu", context_len=context_len
    )


def test_a_step_runs_at_the_next_width_and_tp4_takes_four_rows():
    # a one-row launch never completes on gfx950: one row runs at width 2
    assert [glm53._width(r) for r in glm53.ROWS] == [2, 2, 4, 4, 8, 8, 8, 8]
    # TP4 builds no 8-row launch: a wider step runs the model's own forward
    runtimes = {(w, k): None for w in (2, 4) for k in ("index", "shared")}
    layers = {i: SimpleNamespace(i=i) for i in range(3, 78)}
    lm = _Lm()
    built = _built(lm, layers, runtimes, tp=4)
    key = registry.ShapeKey(forward_mode="decode", batch_size=5, tokens_per_request=1)
    out = built.forward(registry.Plan(key=key), "ids", "positions", "batch")
    assert out == "the model's forward"
    assert lm.ran == [("ids", "positions", "batch", None, None)]
    assert _built(lm, layers, runtimes, tp=8).max_rows == 8


def _step_rows(key, positions, slots, seq_lens, context_len=202752):
    """``_Built._step`` on CPU for a 4-row step: each row's position and its keys
    (indptr). ``seq_lens`` is what SGLang passes: one length per request."""
    W = 4
    runtimes = {(W, k): None for k in ("index", "shared")}
    built = _built(_Lm(), {3: None}, runtimes, context_len=context_len)
    batch = SimpleNamespace(
        out_cache_loc=torch.tensor(slots), seq_lens=torch.tensor(seq_lens)
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(real_page_table=torch.zeros(W, 1))
    )
    with mock.patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=backend,
    ):
        step = built._step(
            registry.Plan(key=key), torch.tensor(positions), batch, len(slots), W
        )
    return step.positions.tolist(), step.indptr.tolist()


@pytest.mark.parametrize(
    "key, positions, slots, seq_lens, indptr",
    [
        # one request of prefix 10 verifies 4 drafts at positions 10 to 13
        (VERIFY, [10, 11, 12, 13], [100, 101, 102, 103], [10], [0, 11, 23, 36, 50]),
        # one row a request (seq_lens = position + 1); slot 0 is a pad row
        (DECODE, [5, 7, 0, 0], [40, 41, 0, 0], [6, 8, 1, 1], [0, 6, 14, 14, 14]),
        # a row attends at most the indexer's 2048 keys
        (
            VERIFY,
            [4000, 4001, 4002, 4003],
            [7, 8, 9, 10],
            [4000],
            [0, 2048, 4096, 6144, 8192],
        ),
    ],
    ids=["verify step", "decode step and a pad row", "rows past 2048 keys"],
)
def test_each_row_attends_the_keys_up_to_its_position(
    key, positions, slots, seq_lens, indptr
):
    """A verify row attends its request's prefix and the drafts up to itself, not
    the prefix alone: in a verify step SGLang's seq_lens holds only the prefix."""
    assert _step_rows(key, positions, slots, seq_lens)[1] == indptr


def test_verify_lookahead_past_the_context_stays_in_the_tables():
    """Verify rows past the context keep positions inside the rotary table and the
    indexer's score rows (12 tokens here); the scheduler discards their tokens."""
    positions, indptr = _step_rows(
        VERIFY, [10, 11, 12, 13], [100, 101, 102, 103], [10], context_len=12
    )
    assert positions == [10, 11, 11, 11]
    assert indptr == [0, 11, 23, 35, 47]


@pytest.mark.parametrize(
    "changes, refused",
    [
        ({}, False),
        ({"speculative_eagle_topk": 2}, True),
        ({"speculative_num_draft_tokens": 3}, True),
    ],
    ids=["default chain", "tree draft", "draft tokens"],
)
def test_mtp_runs_one_draft_chain(changes, refused):
    cfg = SimpleNamespace(**{**vars(MTP), **changes})
    reasons = glm53._config_disable_reasons(cfg, GLM53)
    assert len(reasons) == int(refused)
    assert all("one draft chain" in r for r in reasons)


def test_the_layer_check_counts_the_own_chain_keys_a_row_drops():
    # row 0 drops SGLang's 11 and its own slot 100; row 1 drops SGLang's 21 for row
    # 0's slot 100, which is no drop of its own chain; row 2 is a pad row
    ref = torch.full((3, 2048), -1)
    ref[0, :3] = torch.tensor([10, 11, 100])
    ref[1, :2] = torch.tensor([20, 21])
    topk = torch.zeros(2 * 2048 + 1, dtype=torch.int32)
    topk[:5] = torch.tensor([10, 12, 13, 20, 100])
    step = SimpleNamespace(indptr=torch.tensor([0, 3, 5, 5]), topk=topk)
    out = check._topk_outputs(ref, step, torch.tensor([100, 101, 0]), 3)
    assert out == {"topk": {"rel": 2 / 3}, "topk_own": {"max_abs": 1.0}}


def test_the_layer_check_poisons_only_this_steps_index_scales():
    # page 1 token 5 and page 2 token 63 are this step's keys; slot 0 is a pad row.
    # Bytes below 120 keep every f32 scale and FP8 value finite
    pages = torch.randint(0, 120, (3, check.PAGE_TOKENS * 132), dtype=torch.uint8)
    before = pages.clone()
    check.poison_new_index_scales(pages, torch.tensor([0, 64 + 5, 128 + 63]))
    _, scales = check._index_keys(before)
    _, new_scales = check._index_keys(pages)
    poisoned = torch.zeros_like(scales, dtype=torch.bool)
    poisoned[1, 5] = poisoned[2, 63] = True
    assert (new_scales[poisoned] == check.POISON_SCALE).all()
    assert torch.equal(new_scales[~poisoned], scales[~poisoned])
    # the FP8 values stay, so only the scale of a key read from the cache changes
    values = slice(0, check.INDEX_VALUE_BYTES)
    assert torch.equal(pages[:, values], before[:, values])


def test_config_reasons_name_each_mismatch():
    assert glm53._config_disable_reasons(CFG, GLM53) == []
    other = SimpleNamespace(
        **{
            **vars(GLM53),
            "index_topk_freq": 8,
            "moe_layer_freq": 2,
            "quantization_config": {"quant_method": "quark"},
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
        }
    )
    cfg = SimpleNamespace(**{**vars(CFG), "page_size": 1})
    reasons = glm53._config_disable_reasons(cfg, other)
    text = " ".join(reasons)
    assert "index_topk_freq" in text
    assert "MoE" in text
    assert "--page-size 1" in text
    assert "FP8 block-128" in text
    assert "RoPE 'yarn'" in text


def _linear(shuffled=True):
    # SGLang marks only the block-FP8 linears it rewrote into AITER's layout
    if not shuffled:
        return SimpleNamespace(weight=SimpleNamespace())
    return SimpleNamespace(
        aiter_bpreshuffled=True, weight=SimpleNamespace(is_shuffled=True)
    )


def _layer(
    experts_dtype=torch.float8_e4m3fn,
    n_experts=257,
    indexer=True,
    shuffled=True,
    runner=MoeRunnerBackend.AITER,
    fused_indexer=True,
):
    w13 = torch.empty(n_experts, 1, 1, dtype=experts_dtype)
    quant = SimpleNamespace(runner=SimpleNamespace(runner_backend=runner))
    ind = SimpleNamespace(
        wq_b=_linear(), wk=_linear(), use_gfx950_fused_indexer=fused_indexer
    )
    attn = SimpleNamespace(
        fused_qkv_a_proj_with_mqa=_linear(shuffled),
        q_b_proj=_linear(),
        o_proj=_linear(),
        indexer=ind if indexer else None,
    )
    experts = SimpleNamespace(w13_weight=w13, quant_method=quant)
    return SimpleNamespace(mlp=SimpleNamespace(experts=experts), self_attn=attn)


def _model(layers):
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def _layers():
    # GLM-5.3's indexers: layers 0 to 2, then every fourth from layer 6
    return [_layer(indexer=i % 4 == 2 or i < 3) for i in range(78)]


def test_model_reasons():
    good = _layers()
    assert glm53._model_disable_reasons(_model(good), GLM53, {}) == []
    bad = list(good)
    bad[5] = _layer(experts_dtype=torch.bfloat16)
    bad[9] = _layer(n_experts=256)
    bad[2] = _layer(indexer=False)
    reasons = " ".join(glm53._model_disable_reasons(_model(bad), GLM53, {}))
    assert "layer 5: needs block-FP8 experts" in reasons
    assert "layer 9: needs the shared expert fused" in reasons
    assert "layer 3 reuses the top-k of layer 2" in reasons


def test_model_reasons_name_the_layouts_the_kernel_cannot_read():
    # These served wrong tokens without a refusal: the kernel read row-major FP8
    # projections (--fp8-gemm-backend triton) and experts off the AITER MoE
    # runner as (16, 16) tiles, and scored rotated index queries against keys that
    # an indexer other than SGLang's gfx950 fused one caches unrotated
    layers = _layers()
    layers[4] = _layer(indexer=False, shuffled=False)
    layers[5] = _layer(indexer=False, shuffled=False)
    layers[7] = _layer(indexer=False, runner=MoeRunnerBackend.TRITON)
    layers[10] = _layer(fused_indexer=False)
    reasons = glm53._model_disable_reasons(_model(layers), GLM53, {})
    assert [r.split(":")[0] for r in reasons] == [
        "layer 7",
        "fused_qkv_a_proj_with_mqa of 2 layers from 4",
        "the indexer of layer 10",
    ], reasons
    assert "AITER MoE runner" in reasons[0] and "(16, 16)" in reasons[1]
    assert "gfx950 fused DSA indexer" in reasons[2]


def test_every_indexer_rotary_must_equal_the_mla_one():
    # The first covered layer reuses layer 2's top-k and has no indexer, so an
    # indexer rotary unlike the MLA one went unchecked
    table = torch.randn(16, 64)

    def layer(indexer_table=None):
        rotary = SimpleNamespace(cos_sin_cache=table)
        indexer = None
        if indexer_table is not None:
            indexer = SimpleNamespace(
                rotary_emb=SimpleNamespace(cos_sin_cache=indexer_table)
            )
        return SimpleNamespace(
            self_attn=SimpleNamespace(rotary_emb=rotary, indexer=indexer)
        )

    cos, _ = rope_tables([layer(), layer(table.clone())], {})
    assert torch.equal(cos, table[:, :32].to(torch.bfloat16))
    with pytest.raises(ValueError, match="indexer's rotary"):
        rope_tables([layer(), layer(table + 1)], {})
