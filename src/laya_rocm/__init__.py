"""laya-rocm: drop-in AMD ROCm runtime for Laya.

    import laya_rocm as laya        # instead of `import laya`

Everything `laya` exports is re-exported unchanged, except `Agent`/`RLAgent`/`load`/`Router`,
which are ROCm-aware subclasses with the same signatures.
"""
import laya as _laya
from laya import *  # noqa: F401,F403

from . import runtime
from .agent import Agent, RLAgent, load
from .router import Router
from .runtime import configure, environment, is_rocm

__version__ = "0.1.0"
laya_version = _laya.__version__

__all__ = list(_laya.__all__) + ["runtime", "configure", "environment", "is_rocm", "laya_version"]
