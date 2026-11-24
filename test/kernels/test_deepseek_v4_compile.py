"""The DeepSeek-V4-Pro layer launch compiles for every width the spec serves, on both
attention variants, with SGLang's compressor state rings (FlyDSL's compile-only mode,
one gfx950 GPU), with a fault-injection site too."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

# a TP8 rank of DeepSeek-V4-Pro: layer 0 is HCA (ratio 128), layer 2 CSA (ratio 4)
LAYERS = {128: 0, 4: 2}
# SGLang's state rings (resolve_compress_state_ring_size): plain decode, then DSpark
RINGS = {0: {4: 8, 128: 128}, 6: {4: 16, 128: 256}}


def build(rows, ratio, spec_steps, **kw):
    from sgl_amd_mono.deepseek_v4.gfx950.kernels.layer import build_dsv4_kernel
    from sgl_amd_mono.deepseek_v4.gfx950.runner import shape_dims
    from sgl_amd_mono.deepseek_v4.gfx950.weights import V4Config

    cfg = V4Config().for_layer(LAYERS[ratio])
    cfg.max_seq = 1 << 20
    cfg.validate()
    return build_dsv4_kernel(
        rows,
        cfg.heads,
        8,
        cfg.window,
        scale=cfg.softmax_scale,
        spec_steps=spec_steps,
        swiglu_limit=cfg.swiglu_limit,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        window_rows=cfg.cache_rows,
        c_ring=RINGS[spec_steps][ratio],
        **shape_dims(cfg),
        **kw,
    )


@pytest.mark.parametrize("ratio", LAYERS)
@pytest.mark.parametrize("rows", (1, 2, 4, 8))
def test_every_decode_width_compiles(rows, ratio):
    from sgl_amd_mono.deepseek_v4.gfx950.runner import compile_only
    from sgl_amd_mono.deepseek_v4.spec import SPEC

    assert rows in SPEC.widths("gfx950")
    compile_only(build(rows, ratio, spec_steps=0))


@pytest.mark.parametrize("ratio", LAYERS)
def test_a_dspark_verify_compiles(ratio):
    from sgl_amd_mono.deepseek_v4.gfx950.runner import compile_only

    compile_only(build(7, ratio, spec_steps=6))


def test_the_scrub_and_a_fault_site_compile():
    from sgl_amd_mono.deepseek_v4.gfx950.kernels.layer import build_scrub
    from sgl_amd_mono.deepseek_v4.gfx950.runner import compile_only

    compile_only(build_scrub(1 << 16, 1 << 12))
    compile_only(build(1, 4, spec_steps=0, fault_site=1, fault_after=0))
