import pytest
import torch

from laya_rocm import runtime
from laya_rocm.agent import _is_gpu_error


@pytest.mark.parametrize("arch,want", [
    ("gfx1151", torch.bfloat16), ("gfx1100", torch.bfloat16), ("gfx1201", torch.bfloat16),
    ("gfx942", torch.bfloat16), ("gfx90a", torch.bfloat16), ("gfx950", torch.bfloat16),
    ("gfx1030", torch.float16), ("gfx906", torch.float16), ("", torch.float16),
])
def test_preferred_amp_dtype(monkeypatch, arch, want):
    monkeypatch.setattr(runtime, "gfx_arch", lambda device=0: arch)
    assert runtime.preferred_amp_dtype(torch.bfloat16) == want
    assert runtime.preferred_amp_dtype(torch.float16) == torch.float16


@pytest.mark.parametrize("msg", [
    "HIP error: invalid device function", "CUDA out of memory", "hipErrorNoBinaryForGpu",
    "HSA_STATUS_ERROR_OUT_OF_RESOURCES", "no kernel image is available",
])
def test_hip_errors_trigger_fallback(msg):
    assert _is_gpu_error(RuntimeError(msg))


def test_shape_errors_do_not_trigger_fallback():
    assert not _is_gpu_error(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))


def test_configure_rejects_bad_values():
    if not runtime.is_rocm():
        pytest.skip("configure is a no-op off ROCm")
    with pytest.raises(ValueError):
        runtime.configure(blas="mkl")


def test_environment_is_serialisable():
    import json
    json.dumps(runtime.environment(), default=str)


def test_configure_rejects_bad_native_triton():
    if not runtime.is_rocm():
        pytest.skip("configure is a no-op off ROCm")
    with pytest.raises(ValueError):
        runtime.configure(native_triton="maybe")


def test_c_compiler_honours_cc(monkeypatch):
    monkeypatch.setenv("CC", "/opt/my-cc")
    assert runtime.c_compiler() == "/opt/my-cc"


@pytest.mark.gpu
def test_outer_product_bmm_runs_without_compiler(monkeypatch):
    """ModernBERT's RoPE hits torch._native's Triton outer-product bmm; it must work with no gcc."""
    monkeypatch.setattr(runtime, "c_compiler", lambda: None)
    runtime.configure()
    a, b = torch.randn(4, 64, 1, device="cuda"), torch.randn(4, 1, 32, device="cuda")
    torch.testing.assert_close(torch.bmm(a, b).cpu(), torch.bmm(a.cpu(), b.cpu()))
