"""laya_rocm must be indistinguishable from laya: same token ids, same API, same answers."""
import inspect

import pytest
import torch

import laya
import laya_rocm
from laya.common import build_sequence

from cases import QUESTIONS, STATES


def test_reexports_everything():
    missing = [n for n in laya.__all__ if not hasattr(laya_rocm, n)]
    assert not missing


def test_signatures_are_supersets():
    up = inspect.signature(laya.Agent.__init__).parameters
    ours = inspect.signature(laya_rocm.Agent.__init__).parameters
    assert list(up) == list(ours)[: len(up)]
    assert issubclass(laya_rocm.Agent, laya.Agent)
    assert issubclass(laya_rocm.Router, laya.Router)


@pytest.mark.parametrize("qid", list(QUESTIONS))
@pytest.mark.parametrize("si", range(len(STATES)))
def test_token_ids_identical(tokenizer_agent, qid, si):
    a = tokenizer_agent
    q = laya.Agent._to_internal(QUESTIONS[qid])
    state = STATES[si]
    max_len, hml = a.cfg.get("max_len", 512), a.cfg.get("head_max_len", 192)
    want_ids, want_markers = build_sequence(a.tok, state, q, max_len, hml)
    got_ids, got_markers, _ = a._sequence(q, a._state_ids(state))
    assert got_ids == want_ids
    assert got_markers == want_markers


@pytest.mark.model
def test_cpu_outputs_identical(cpu_pair):
    up, ours = cpu_pair
    assert ours.dtype == up.dtype == torch.float32
    for state in STATES:
        assert ours.predict(state, QUESTIONS) == up.predict(state, QUESTIONS)


@pytest.mark.model
def test_predict_many_matches_single(cpu_pair):
    _, ours = cpu_pair
    many = ours.predict_many(STATES[:4], QUESTIONS)
    for state, got in zip(STATES[:4], many):
        want = ours.predict(state, QUESTIONS)
        assert got["usage"] == want["usage"]
        for qid, ans in want["answers"].items():
            if "probabilities" in ans:
                for k, v in ans["probabilities"].items():
                    assert got["answers"][qid]["probabilities"][k] == pytest.approx(v, abs=2e-3)


@pytest.mark.model
@pytest.mark.gpu
def test_gpu_placement_and_dtype(gpu_agent):
    assert gpu_agent.device.type == "cuda"
    assert gpu_agent.dtype in (torch.bfloat16, torch.float16)
    r = gpu_agent.predict(STATES[1], QUESTIONS)
    assert set(r["answers"]) == set(QUESTIONS)
    assert gpu_agent.device.type == "cuda", "silently fell back to CPU"


@pytest.mark.model
@pytest.mark.gpu
def test_gpu_agrees_with_cpu_fp32(cpu_pair, gpu_agent):
    _, cpu = cpu_pair
    for state in STATES:
        want, got = cpu.predict(state, QUESTIONS), gpu_agent.predict(state, QUESTIONS)
        for qid, ans in want["answers"].items():
            if "probabilities" in ans:
                for k, v in ans["probabilities"].items():
                    assert got["answers"][qid]["probabilities"][k] == pytest.approx(v, abs=0.05), (qid, k)
