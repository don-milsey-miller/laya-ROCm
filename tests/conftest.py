import os

import pytest
import torch

MODEL = os.environ.get("LAYA_TEST_MODEL", "convaiinnovations/laya")


def pytest_collection_modifyitems(config, items):
    run_model = os.environ.get("LAYA_RUN_MODEL_TESTS") == "1"
    has_gpu = bool(getattr(torch.version, "hip", None)) and torch.cuda.is_available()
    for it in items:
        if "model" in it.keywords and not run_model:
            it.add_marker(pytest.mark.skip(reason="set LAYA_RUN_MODEL_TESTS=1 to download and run the checkpoint"))
        if "gpu" in it.keywords and not has_gpu:
            it.add_marker(pytest.mark.skip(reason="needs a ROCm GPU"))


@pytest.fixture(scope="session")
def tokenizer_agent():
    """A laya_rocm.Agent shell with only the tokenizer and config -- enough to test tokenization."""
    import json

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    import laya.agent as up
    from laya_rocm.agent import Agent

    path = snapshot_download(MODEL, allow_patterns=["rl_agent_config.json", "tokenizer/*"])
    up._fix_tokenizer_config(path)
    a = Agent.__new__(Agent)
    a.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    with open(os.path.join(path, "rl_agent_config.json")) as f:
        a.cfg = json.load(f)
    a._head = a._build_head
    return a


@pytest.fixture(scope="session")
def cpu_pair():
    import laya
    import laya_rocm

    return laya.load(MODEL, device="cpu"), laya_rocm.load(MODEL, device="cpu")


@pytest.fixture(scope="session")
def gpu_agent():
    import laya_rocm

    return laya_rocm.load(MODEL, device="cuda")
