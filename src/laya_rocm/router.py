"""`laya.Router` that builds `laya_rocm.Agent` checkpoints."""
from typing import Any, Dict, Optional

import laya.router as _upstream
from laya.router import _split, normalise_name

from .agent import Agent


class Router(_upstream.Router):
    """Drop-in for `laya.Router`. Extra keyword arguments are passed to every `laya_rocm.Agent`."""

    def __init__(self, *args, rocm_options: Optional[Dict[str, Any]] = None, **kwargs):
        self.rocm_options = dict(rocm_options or {})
        super().__init__(*args, **kwargs)

    def load(self, name: str):
        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            repo, sub = _split(self.models[key])
            agent = Agent(repo, device=self.device, token=self.token, subfolder=sub, **self.rocm_options)
            self._agents[key] = agent
            self._order.append(key)
            self._evict()
            return agent
