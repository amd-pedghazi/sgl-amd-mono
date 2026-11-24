# SPDX-License-Identifier: Apache-2.0
"""The dispatch the runners rely on (AITER's ``_preload_compiled`` and ``_run_compiled``):
a preload never runs the kernel, the first real call runs it once, and later calls, in a
graph too, skip FlyDSL's JIT dispatch."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_first_call_runs_once_then_reuses_its_compiled_callable(monkeypatch):
    flyc = pytest.importorskip("flydsl.compiler")
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import T

    from sgl_amd_mono.glm53_flash.gfx950.runner import compile_only

    @flyc.kernel
    def increment(out: fx.Int64, value: fx.Int64):
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), out.ir_value()).result
        count = fx.Int32(llvm.LoadOp(T.i32, ptr).result)
        llvm.StoreOp((count + 1).ir_value(), ptr)
        ptr_value = llvm.IntToPtrOp(
            ir.Type.parse("!llvm.ptr<1>"), (out + 4).ir_value()
        ).result
        llvm.StoreOp(fx.Int32(value).ir_value(), ptr_value)

    @flyc.jit
    def launch(out: fx.Int64, value: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        increment(out, value).launch(grid=(1,), block=(1,), stream=stream)

    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    out1 = torch.zeros(2, dtype=torch.int32, device="cuda")
    out2 = torch.zeros_like(out1)
    # a real buffer first: compile_only's zero addresses fault if a preload dispatches
    _preload_compiled(launch, out1.data_ptr(), 99, torch.cuda.current_stream())
    torch.cuda.synchronize()
    assert out1.tolist() == [0, 0]
    compile_only(launch)
    torch.cuda.synchronize()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    s1.wait_stream(torch.cuda.current_stream())
    s2.wait_stream(torch.cuda.current_stream())
    _run_compiled(launch, out1.data_ptr(), 11, s1)
    s1.synchronize()
    assert out1.tolist() == [1, 11]

    def no_jit(*args, **kwargs):
        raise AssertionError("a warm launch re-entered JIT dispatch")

    monkeypatch.setattr(launch, "_build_full_cache_key", no_jit)
    _run_compiled(launch, out2.data_ptr(), 22, s2)
    s2.synchronize()
    assert out2.tolist() == [1, 22]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s1):
        _run_compiled(launch, out1.data_ptr(), 33, torch.cuda.current_stream())
    for n in (2, 3):
        graph.replay()
        torch.cuda.synchronize()
        assert out1.tolist() == [n, 33]
