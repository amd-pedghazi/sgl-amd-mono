"""The DeepSeek-V4-Pro gfx950 monokernel spec on CPU (RFC #43949): its registry entry
next to V4.1's, the configurations it refuses, and the unified KV rows its launches
address. SGLang checks its declared requirements, the same way for every spec."""

from types import SimpleNamespace

import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.deepseek_v4 import spec as dsv4
from sgl_amd_mono.deepseek_v4.gfx950.config import V4_PRO_CONFIG, compress_ratios
from sgl_amd_mono.deepseek_v4.gfx950.runner import step_rows
from sgl_amd_mono.deepseek_v41 import spec as dsv41

# DeepSeek-V4-Pro-0813's config.json, as SGLang holds it
PRO = SimpleNamespace(
    **V4_PRO_CONFIG,
    compress_ratios=list(compress_ratios(61, n_mtp=3)),
    model_type="deepseek_v4",
    architectures=["DeepseekV4ForCausalLM"],
    context_len=1 << 20,
)


def server(**kw):
    return SimpleNamespace(
        **{
            "load_format": "auto",
            "disaggregation_mode": "null",
            "disaggregation_transfer_backend": "mooncake",
            "speculative_algorithm": None,
            "speculative_num_draft_tokens": None,
            **kw,
        }
    )


def test_each_deepseek_v4_model_finds_its_own_spec():
    sgl_amd_mono.register_specs()
    find = registry.find_spec
    for arch, model_type, spec in (
        ("DeepseekV4ForCausalLM", "deepseek_v4", dsv4.SPEC),
        ("DeepseekV4ForCausalLM", "deepseek_v41", dsv41.SPEC),
    ):
        found, why = find(
            backend=sgl_amd_mono.BACKEND,
            architectures=[arch],
            isa="gfx950",
            model_type=model_type,
        )
        assert found is spec, why
    # V4-Pro's DSpark draft keeps SGLang's forward: no spec, not a refusal
    assert not registry.spec_registered(
        backend=sgl_amd_mono.BACKEND,
        architectures=["DeepseekV4ForCausalLMDSpark"],
        model_type="deepseek_v4",
    )


def reasons(hf=PRO, **kw):
    return [
        r for r in dsv4._config_disable_reasons(server(**kw), hf) if "FlyDSL" not in r
    ]


def test_the_v4_pro_recipe_is_served(monkeypatch):
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    assert reasons() == []
    # DSpark at block size 6: one request's verify of 7 rows a launch
    assert reasons(speculative_algorithm="DSPARK", speculative_num_draft_tokens=7) == []


def test_what_the_kernels_do_not_build_in_is_refused(monkeypatch):
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "0")
    flash = SimpleNamespace(**{**vars(PRO), "hidden_size": 4096})
    assert any("DeepSeek-V4-Pro" in r for r in reasons(flash))
    schedule = SimpleNamespace(**{**vars(PRO), "compress_ratios": [4] * 64})
    assert any("compress_ratios" in r for r in reasons(schedule))
    # two requests' verifies would fit a launch, or one would not
    for k in (2, 4, 9):
        got = reasons(speculative_algorithm="DSPARK", speculative_num_draft_tokens=k)
        assert len(got) == 1 and f"verify of {k} tokens" in got[0]
    assert len(reasons(load_format="dummy")) == 1
    monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND")
    assert any("MXFP8 activations" in r for r in reasons())


def test_the_rows_are_where_sglangs_pool_puts_them():
    torch.manual_seed(0)
    ring, swa_pages, window, rows, pages = 134, 1000, 128, 7, 40
    pos = torch.tensor([0, 5, 127, 128, 300, 4095, 9000], dtype=torch.int64)
    req = torch.tensor([0, 3, 3, 1, 2, 5, 7], dtype=torch.int64)
    bt = torch.randint(0, 500, (rows, pages), dtype=torch.int32)
    variants = {128: (window + 128, False), 4: (window + 1024, True)}
    swa, dest, idx = step_rows(
        pos, req, bt, ring=ring, swa_pages=swa_pages, window=window, variants=variants
    )
    for s in range(rows):
        p, r = int(pos[s]), int(req[s])
        assert int(swa[s]) == r * ring + p % ring
        for ratio in variants:
            assert dest[ratio][:, s].tolist() == [r * ring + p % ring, swa_pages]
            # the window: positions p - 127 .. p, the row's own last
            win = [r * ring + q % ring if q >= 0 else -1 for q in range(p - 127, p + 1)]
            assert idx[ratio][s, :window].tolist() == win
        # HCA: every entry closed at or before p, in the page's two rows
        n = (p + 1) // 128
        hca = [
            swa_pages + int(bt[s, e // 2]) * 2 + e % 2 if e < n else -1
            for e in range(128)
        ]
        assert idx[128][s, window:].tolist() == hca
        assert (idx[4][s, window:] == -1).all()  # CSA: the kernel's own pick
