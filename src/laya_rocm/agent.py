"""ROCm-aware drop-in replacement for `laya.Agent`.

Same constructor, same `predict`/`system_one` signature, same output dict. What changes:

* precision is chosen from the AMD gfx architecture rather than an NVIDIA compute-capability test;
* HIP runtime errors trigger the same CPU fallback that CUDA errors do upstream;
* the question head (instructions + options) is tokenized once and cached, and the state is
  tokenized once per call instead of once per question -- token ids are byte-identical to upstream;
* `predict_many` scores several states against one question set in a single forward pass.
"""
import functools
import threading
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import laya.agent as _upstream
from laya.common import (
    QTYPES,
    collate_items,
    confidence_from_probs,
    render_options,
    serialize_state,
    temp_bucket,
)

from . import runtime

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_GPU_ERROR_MARKERS = ("memory", "cuda", "hip", "rocm", "hsa", "device function", "no kernel image")


def _is_gpu_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(m in msg for m in _GPU_ERROR_MARKERS)


class Agent(_upstream.Agent):
    """`laya.Agent` with ROCm-correct precision, HIP-aware fallback and cached tokenization.

    Extra keyword arguments (all optional, defaults keep upstream behaviour on non-ROCm builds):

      dtype             'auto' | 'bf16' | 'fp16' | 'fp32'  autocast precision on GPU
      blas, fa,         forwarded to `laya_rocm.runtime.configure` before the model is placed
      tunableop, tunableop_file,
      native_triton
      head_cache_size   question heads kept tokenized (0 disables the cache)
    """

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        *,
        dtype: str = "auto",
        blas: Optional[str] = None,
        fa: Optional[str] = None,
        tunableop: Optional[str] = None,
        tunableop_file: Optional[str] = None,
        native_triton: Optional[str] = "auto",
        head_cache_size: int = 4096,
    ):
        if dtype not in ("auto",) + tuple(_DTYPES):
            raise ValueError("dtype must be 'auto', 'bf16', 'fp16' or 'fp32', got %r" % (dtype,))
        runtime.configure(blas=blas, fa=fa, tunableop=tunableop, tunableop_file=tunableop_file,
                          native_triton=native_triton)
        super().__init__(model_id_or_path, device=device, token=token, subfolder=subfolder)

        if self.device.type == "cuda":
            if dtype != "auto":
                self.dtype = _DTYPES[dtype]
            elif runtime.is_rocm():
                ckpt = _upstream.amp_dtype(self.cfg.get("amp_dtype", "fp16"))
                self.dtype = runtime.preferred_amp_dtype(ckpt, self.device)
        self._lock = threading.Lock()
        self._head = (functools.lru_cache(maxsize=head_cache_size)(self._build_head)
                      if head_cache_size > 0 else self._build_head)

    # ------------------------------------------------------------------ tokenization
    def _build_head(self, qtype: str, ins: str, opts: Tuple[str, ...]):
        """The state-independent prefix of `laya.common.build_sequence`, token for token."""
        tok = self.tok
        mask_tok = tok.mask_token
        head_max_len = self.cfg.get("head_max_len", 192)
        ins = str(ins).replace(mask_tok, " ")
        head_ids = tok("%s question: %s" % (qtype, ins), add_special_tokens=False)["input_ids"]
        opt_ids = [[tok.mask_token_id]
                   + tok(" " + o.replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
                   for o in opts]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]
        ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(tok.sep_token_id)
        return tuple(ids), tuple(markers)

    def _state_ids(self, state) -> List[int]:
        mask_tok = self.tok.mask_token
        return self.tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]

    def _sequence(self, q: Dict, state_ids: List[int]):
        max_len = self.cfg.get("max_len", 512)
        opts = tuple(render_options(q))
        head, markers = self._head(q["t"], q["ins"], opts)
        room = max(0, max_len - len(head) - 1)
        ids = list(head) + state_ids[:room] + [self.tok.sep_token_id]
        return ids[:max_len], [m for m in markers if m < max_len], len(opts)

    def _items(self, state, parsed: List[Tuple[str, Dict]]):
        st = self._state_ids(state)
        items = []
        for qid, q in parsed:
            seq, markers, n_opts = self._sequence(q, st)
            if len(markers) != n_opts:
                raise ValueError("question %r options exceed head_max_len=%d"
                                 % (qid, self.cfg.get("head_max_len", 192)))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        return items

    # ------------------------------------------------------------------ forward
    def _forward(self, b):
        use_amp = self.device.type == "cuda"
        args = [b[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        try:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
                return self.model(*[t.to(self.device, non_blocking=True) for t in args])
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type == "cpu" or not _is_gpu_error(e):
                raise
            warnings.warn("laya_rocm: GPU inference failed (%s); falling back to CPU FP32." % e,
                          RuntimeWarning, stacklevel=3)
            with self._lock:
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.model.to(self.device)
            return self.model(*args)

    def _decode(self, questions, parsed, items, logits, act, row0: int):
        answers = {}
        for r, (qid, q) in enumerate(parsed):
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[row0 + r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf_score = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[row0 + r, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf_score,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    "action": ext,
                }
        return answers

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Same contract as `laya.Agent.system_one`."""
        return self.predict_many([state], questions)[0]

    predict = system_one

    @torch.no_grad()
    def predict_many(self, states: Sequence[Union[str, dict, list]],
                     questions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Answer the same questions for several states in one forward pass.

        Returns one `system_one`-shaped dict per state, in order. Useful for micro-batching a
        request queue: the GPU sees len(states) * len(questions) sequences at once.
        """
        parsed = [(qid, self._to_internal(qdef)) for qid, qdef in questions.items()]
        groups = [self._items(s, parsed) for s in states]
        b = collate_items(groups, self.tok.pad_token_id)
        if b is None:
            return [{"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
                    for _ in states]
        logits, act = self._forward(b)
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        att = b["attention_mask"].sum(-1).tolist()

        out, row0 = [], 0
        for items in groups:
            n = len(items)
            out.append({
                "model": "laya-rl-agent",
                "answers": self._decode(questions, parsed, items, logits, act, row0),
                "usage": {"input_tokens": int(sum(att[row0:row0 + n])), "output_tokens": 0},
            })
            row0 += n
        return out


RLAgent = Agent


def load(model_id_or_path: str = "convaiinnovations/laya", device: Optional[str] = None,
         token: Optional[str] = None, subfolder: Optional[str] = None, **rocm_options) -> Agent:
    """Drop-in for `laya.load`; `rocm_options` are the extra keyword arguments of `Agent`."""
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, **rocm_options)
