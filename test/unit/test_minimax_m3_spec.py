"""The MiniMax-M3 gfx950 monokernel spec on CPU (RFC #43949).

Checks its registry entries, the configurations it refuses before load, the layer views
it refuses after load (meta tensors of the real shapes) and the verify widths it serves
on the EAGLE3 recipe. SGLang checks the declared requirements, the same way for every
spec."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry
try:
    from sglang.srt.kernel_plugins.amd.binding import covered_capture_sizes
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.guards import covered_capture_sizes

import sgl_amd_mono
from sgl_amd_mono.minimax_m3 import spec as m3
from sgl_amd_mono.minimax_m3.gfx950.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
    ROTARY_DIM,
    qkv_rows,
)
from sgl_amd_mono.minimax_m3.weights import layer_reasons

# MiniMax-M3's text config (amd/MiniMax-M3-MXFP4 config.json, text_config)
TEXT = SimpleNamespace(
    hidden_size=6144,
    num_hidden_layers=60,
    num_local_experts=128,
    num_experts_per_tok=4,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    scoring_func="sigmoid",
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    rms_norm_eps=1e-6,
    sparse_attention_config={
        "sparse_topk_blocks": 16,
        "sparse_init_block": 0,
        "sparse_local_block": 1,
        "sparse_block_size": 128,
        "sparse_score_type": "max",
    },
)
VL = SimpleNamespace(
    text_config=TEXT, architectures=["MiniMaxM3SparseForConditionalGeneration"]
)
# the EAGLE3 recipe's server arguments the spec reads
CFG = SimpleNamespace(
    speculative_algorithm="EAGLE3",
    speculative_eagle_topk=1,
    speculative_num_steps=3,
    speculative_num_draft_tokens=4,
    enable_hierarchical_cache=True,
    hicache_io_backend="kernel",
    hicache_mem_layout="layer_first",
    enable_lmcache=False,
    enable_flexkv=False,
    disaggregation_decode_enable_offload_kvcache=False,
)


def _replace(ns, **changes):
    return SimpleNamespace(**{**vars(ns), **changes})


def _config_reasons(cfg=CFG, hf_config=VL):
    with (
        mock.patch.object(m3, "_has_flydsl", return_value=True),
        mock.patch.object(m3, "_pending_env_reasons", return_value=[]),
    ):
        return m3._config_disable_reasons(cfg, hf_config)


def test_the_plugin_registers_both_architectures():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    for arch in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ):
        found, why = registry.find_spec(
            backend=sgl_amd_mono.BACKEND, architectures=[arch], isa="gfx950"
        )
        assert found is m3.SPEC, why


def test_the_recipe_is_accepted():
    assert _config_reasons() == []
    assert _config_reasons(hf_config=TEXT) == []  # the text-only checkpoint
    plain = _replace(CFG, speculative_algorithm=None, enable_hierarchical_cache=False)
    assert _config_reasons(cfg=plain) == []


@pytest.mark.parametrize(
    "cfg, text",
    [
        (_replace(CFG, speculative_eagle_topk=2), TEXT),
        (_replace(CFG, speculative_num_draft_tokens=3), TEXT),
        (_replace(CFG, hicache_io_backend="direct"), TEXT),
        (_replace(CFG, hicache_mem_layout="page_first"), TEXT),
        (_replace(CFG, enable_lmcache=True), TEXT),
        (CFG, _replace(TEXT, hidden_size=5120)),
        (CFG, _replace(TEXT, swiglu_alpha=1.5)),
        (CFG, _replace(TEXT, num_hidden_layers=200)),
        (
            CFG,
            _replace(
                TEXT,
                sparse_attention_config={
                    **TEXT.sparse_attention_config,
                    "sparse_local_block": 2,
                },
            ),
        ),
    ],
    ids=[
        "tree draft",
        "draft tokens",
        "hicache io",
        "hicache layout",
        "lmcache",
        "hidden size",
        "swiglu alpha",
        "mailbox layers",
        "selection",
    ],
)
def test_each_unsupported_setting_is_one_reason(cfg, text):
    assert len(_config_reasons(cfg=cfg, hf_config=_replace(VL, text_config=text))) == 1


def test_an_sglang_without_the_shuffle_kv_is_refused_with_a_reason():
    # without sgl-project/sglang#41998 / #41708's switches: a refusal naming them, not a crash
    from sglang.srt.environ import envs

    present = {n: getattr(envs, n, None) for n in m3.PENDING_ENV}
    with mock.patch("sglang.srt.environ.envs", SimpleNamespace()):
        missing = m3._pending_env_reasons()
    assert len(missing) == len(m3.PENDING_ENV)
    assert all("sglang#41998" in r for r in missing)
    if all(v is not None for v in present.values()):
        # an SGLang with them: the defaults are what the kernels need
        assert m3._pending_env_reasons() == []


def test_the_recipe_routes_verify_batches_of_one_to_four():
    # 4 rows a request: batch sizes 1-4 (4 to 16 rows) join the configured ones (8: the normal path)
    assert m3.SPEC.widths("gfx950") == tuple(range(1, 17))
    assert covered_capture_sizes(
        widths=m3.SPEC.widths("gfx950"), tokens_per_request=4, sizes=[1, 2, 4, 8]
    ) == [1, 2, 3, 4, 8]


class GemmaRMSNorm(torch.nn.Module):
    """Named as SGLang's: the views check the norm's type by name."""

    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(n, dtype=torch.bfloat16, device="meta")
        )


def _meta(*shape, dtype=torch.bfloat16, **attrs):
    t = torch.empty(*shape, dtype=dtype, device="meta")
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


def _layer(**changes):
    """A sparse M3 decoder layer at TP4, its tensors on the meta device."""
    E = N_ROUTED + 1
    experts = SimpleNamespace(
        moe_ep_rank=0,
        num_local_experts=E,
        num_fused_shared_experts=1,
        top_k=5,
        intermediate_size_per_partition=INTER,
        moe_runner_config=SimpleNamespace(activation="silu"),
        w13_weight=_meta(
            E, 2 * INTER, HIDDEN // 2, dtype=torch.uint8, is_shuffled=True
        ),
        w2_weight=_meta(E, HIDDEN, INTER // 2, dtype=torch.uint8, is_shuffled=True),
        w13_weight_scale=_meta(E, 2 * INTER, HIDDEN // 32, dtype=torch.uint8),
        w2_weight_scale=_meta(E, HIDDEN, INTER // 32, dtype=torch.uint8),
    )
    attn = SimpleNamespace(
        is_sparse_attention_layer=True,
        disable_index_value=True,
        num_heads=LOCAL_Q_HEADS,
        num_kv_heads=1,
        num_idx_heads=1,
        head_dim=HEAD_DIM,
        idx_head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        use_gemma_norm=True,
        qk_norm_type="per_head",
        q_norm=GemmaRMSNorm(HEAD_DIM),
        k_norm=GemmaRMSNorm(HEAD_DIM),
        index_q_norm=GemmaRMSNorm(HEAD_DIM),
        index_k_norm=GemmaRMSNorm(HEAD_DIM),
        _fused_qkv_index=SimpleNamespace(weight=_meta(qkv_rows(1), HIDDEN)),
        o_proj=SimpleNamespace(weight=_meta(HIDDEN, O_K)),
        rotary_emb=SimpleNamespace(cos_sin_cache=_meta(1 << 20, ROTARY_DIM)),
        attn=SimpleNamespace(k_scale=None, v_scale=None),
    )
    mlp = SimpleNamespace(
        experts=experts,
        gate=SimpleNamespace(weight=_meta(N_ROUTED, HIDDEN, dtype=torch.float32)),
        e_score_correction_bias=_meta(N_ROUTED, dtype=torch.float32),
    )
    layer = SimpleNamespace(
        is_layer_sparse=True,
        self_attn=attn,
        mlp=mlp,
        input_layernorm=GemmaRMSNorm(HIDDEN),
        post_attention_layernorm=GemmaRMSNorm(HIDDEN),
    )
    for path, value in changes.items():
        owner, name = layer, path
        while "." in name:
            head, name = name.split(".", 1)
            owner = getattr(owner, head)
        setattr(owner, name, value)
    return layer


def test_the_views_accept_a_sparse_layer():
    assert layer_reasons(_layer(), 3) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"self_attn.disable_index_value": False},
        {"self_attn.num_idx_heads": 4},  # indexer context parallelism
        {
            "self_attn.o_proj": SimpleNamespace(
                weight=_meta(HIDDEN, O_K, dtype=torch.float8_e4m3fn)
            )
        },
        {
            "mlp.experts.w13_weight": _meta(
                N_ROUTED + 1, 2 * INTER, HIDDEN // 2, dtype=torch.uint8
            )
        },
        # SGLang interleaves a SiTU activation's gate and up rows
        {"mlp.experts.moe_runner_config": SimpleNamespace(activation="situ")},
        {"mlp.experts.num_local_experts": N_ROUTED},
        {"mlp.experts.moe_ep_rank": 1},
        {"mlp.e_score_correction_bias": None},
        {"input_layernorm": SimpleNamespace(weight=_meta(HIDDEN))},  # not GemmaRMSNorm
        {
            "self_attn.attn": SimpleNamespace(
                k_scale=_meta(4, dtype=torch.float32), v_scale=None
            )
        },
    ],
    ids=[
        "index values",
        "index heads",
        "fp8 o_proj",
        "unshuffled experts",
        "interleaved gate and up",
        "no fused shared expert",
        "expert parallel",
        "no routing bias",
        "norm type",
        "per-head k scale",
    ],
)
def test_each_unsupported_view_is_refused(changes):
    assert len(layer_reasons(_layer(**changes), 3)) == 1
