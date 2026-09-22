"""ROCm detection and backend configuration.

Everything here is safe to import on CPU-only, CUDA and ROCm builds of PyTorch: on anything that
is not ROCm the configuration functions are no-ops and `environment()` still reports what it can.
"""
import os
import platform
import re
import shutil
import warnings
from typing import Any, Dict, Optional

import torch

# gfx families with native BF16 matrix instructions (MFMA on CDNA, WMMA on RDNA3+).
# Anything else (RDNA1/2, Vega, older) gets FP16, which those parts execute natively.
_NATIVE_BF16 = re.compile(r"^gfx(908|90a|94\d|95\d|11\d\d|12\d\d)")

BLAS_CHOICES = ("default", "hipblaslt", "hipblas", "rocblas")
FA_CHOICES = ("default", "aotriton", "ck")
TUNABLEOP_CHOICES = ("off", "use", "tune")
NATIVE_TRITON_CHOICES = ("auto", "on", "off")


def is_rocm() -> bool:
    """True when this PyTorch build targets HIP/ROCm (whether or not a GPU is visible)."""
    return bool(getattr(torch.version, "hip", None))


def is_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def gfx_arch(device: Any = 0) -> Optional[str]:
    """The AMD gfx target of `device` (e.g. 'gfx1151'), or None if not a ROCm GPU."""
    if not (is_rocm() and torch.cuda.is_available()):
        return None
    name = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "") or ""
    return name.split(":")[0] or None


def preferred_amp_dtype(checkpoint_dtype: torch.dtype, device: Any = 0) -> torch.dtype:
    """Autocast dtype for a ROCm GPU.

    Upstream Laya drops to FP16 when `get_device_capability()[0] < 8`, an NVIDIA SM test. On ROCm
    that tuple is the gfx version, so the test passes by accident on every AMD part. Decide from
    the architecture instead: keep the checkpoint's BF16 where the hardware does BF16 natively,
    otherwise use FP16.
    """
    if checkpoint_dtype != torch.bfloat16:
        return checkpoint_dtype
    arch = gfx_arch(device) or ""
    return torch.bfloat16 if _NATIVE_BF16.match(arch) else torch.float16


def c_compiler() -> Optional[str]:
    """The C compiler Triton would use to build its launcher stubs (same lookup order), or None."""
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")


_native_triton_disabled = False


def native_triton_enabled() -> bool:
    """False when PyTorch's Triton-backed `torch._native` op overrides are off (or absent)."""
    try:
        from torch._native import common_utils
    except Exception:
        return False
    return not (_native_triton_disabled or common_utils.check_native_jit_disabled())


def disable_native_triton(reason: str = "") -> bool:
    """Route `torch._native` Triton overrides (e.g. the outer-product `bmm` ModernBERT's RoPE hits)
    back to the stock hipBLAS/ATen kernels. Returns True if anything was deregistered."""
    global _native_triton_disabled
    if not native_triton_enabled():
        return False
    try:
        from torch._native import registry
        registry.deregister_op_overrides(disable_dsl_names="triton")
    except Exception as e:  # registry API is private; never let it break model loading
        warnings.warn("laya_rocm: could not disable torch._native Triton ops (%s)" % e, RuntimeWarning)
        return False
    _native_triton_disabled = True
    if reason:
        warnings.warn("laya_rocm: %s; using stock PyTorch kernels instead of torch._native Triton ops."
                      % reason, RuntimeWarning, stacklevel=2)
    return True


def configure(
    blas: Optional[str] = None,
    fa: Optional[str] = None,
    tunableop: Optional[str] = None,
    tunableop_file: Optional[str] = None,
    native_triton: Optional[str] = "auto",
) -> Dict[str, Any]:
    """Select ROCm GEMM / attention backends and TunableOp mode for this process.

    Every argument defaults to None, meaning "leave PyTorch's choice alone". Returns what is in
    effect afterwards. Call before the first forward pass: backend choices are process-global.

      blas       'default' | 'hipblaslt' | 'hipblas' | 'rocblas'
      fa         'default' | 'aotriton' | 'ck'    (Flash-Attention library behind SDPA)
      tunableop  'off' | 'use' (replay a tuned CSV) | 'tune' (tune new GEMM shapes and record)
      native_triton  'auto' (off when no C compiler: Triton JIT-builds a C stub on first use) |
                 'on' | 'off'   -- torch>=2.13 `torch._native` Triton op overrides
    """
    if not is_rocm():
        return effective_backends()

    if native_triton is not None:
        if native_triton not in NATIVE_TRITON_CHOICES:
            raise ValueError("native_triton must be one of %s, got %r" % (NATIVE_TRITON_CHOICES, native_triton))
        if native_triton == "off":
            disable_native_triton()
        elif native_triton == "auto" and c_compiler() is None:
            disable_native_triton("no C compiler found (install gcc or set CC)")

    if blas is not None:
        if blas not in BLAS_CHOICES:
            raise ValueError("blas must be one of %s, got %r" % (BLAS_CHOICES, blas))
        if blas != "default":
            torch.backends.cuda.preferred_blas_library("cublaslt" if blas == "hipblaslt" else "cublas")
            if blas == "rocblas":
                # rocBLAS is PyTorch's plain "cublas" path on ROCm; make sure hipBLASLt is not preferred.
                os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

    if fa is not None:
        if fa not in FA_CHOICES:
            raise ValueError("fa must be one of %s, got %r" % (FA_CHOICES, fa))
        if fa != "default" and hasattr(torch.backends.cuda, "preferred_rocm_fa_library"):
            torch.backends.cuda.preferred_rocm_fa_library(fa)

    if tunableop is not None:
        if tunableop not in TUNABLEOP_CHOICES:
            raise ValueError("tunableop must be one of %s, got %r" % (TUNABLEOP_CHOICES, tunableop))
        tun = torch.cuda.tunable
        if tunableop_file:
            tun.set_filename(tunableop_file)
        tun.enable(tunableop != "off")
        tun.tuning_enable(tunableop == "tune")
        if tunableop == "use" and tunableop_file and os.path.exists(tunableop_file):
            tun.read_file(tunableop_file)

    return effective_backends()


def effective_backends() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        lib = torch.backends.cuda.preferred_blas_library()
        out["blas"] = str(lib).split(".")[-1]
    except Exception:
        out["blas"] = None
    try:
        out["fa"] = str(torch.backends.cuda.preferred_rocm_fa_library()).split(".")[-1] if is_rocm() else None
    except Exception:
        out["fa"] = None
    try:
        out["ck_sdpa_available"] = bool(torch.backends.cuda.is_ck_sdpa_available()) if is_rocm() else False
    except Exception:
        out["ck_sdpa_available"] = None
    try:
        tun = torch.cuda.tunable
        out["tunableop"] = ("tune" if tun.tuning_is_enabled() else "use") if tun.is_enabled() else "off"
        out["tunableop_file"] = tun.get_filename() if tun.is_enabled() else None
    except Exception:
        out["tunableop"] = None
    out["native_triton"] = native_triton_enabled()
    out["c_compiler"] = c_compiler()
    return out


def environment() -> Dict[str, Any]:
    """The hardware/software tuple a benchmark result must be published with."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wsl": is_wsl(),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "gpu_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
    }
    for mod in ("transformers", "laya", "numpy", "tokenizers"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({
            "gpu": p.name,
            "gfx_arch": getattr(p, "gcnArchName", None),
            "compute_units": p.multi_processor_count,
            "vram_gib": round(p.total_memory / 2 ** 30, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    env["backends"] = effective_backends()
    env["env_vars"] = {k: v for k, v in os.environ.items()
                       if k.startswith(("PYTORCH_TUNABLEOP", "TORCH_BLAS", "HSA_", "HIP_", "ROCR_",
                                        "MIOPEN_", "TORCH_ROCM", "HIPBLASLT"))}
    return env
