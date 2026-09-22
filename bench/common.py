"""Shared benchmark plumbing: agent construction per configuration, datasets and metrics.

The metrics, `score_cases` packing and dataset builders are copied from upstream Laya's
`research/scripts/build_benchmark_nb.py` (Apache-2.0) so numbers are directly comparable with
`research/results/t4_colab_benchmark.json`. Only device handling was generalised.
"""
import json
import math
import os
import sys
import time
from typing import Any, Dict, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket  # noqa: E402

import laya_rocm  # noqa: E402
from laya_rocm import runtime  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def add_config_args(ap):
    """Arguments that define one benchmark configuration (one fresh process each)."""
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--impl", choices=("laya", "laya_rocm"), default="laya_rocm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("auto",) + tuple(DTYPES), default="auto")
    ap.add_argument("--blas", choices=runtime.BLAS_CHOICES, default=None)
    ap.add_argument("--fa", choices=runtime.FA_CHOICES, default=None)
    ap.add_argument("--tunableop", choices=runtime.TUNABLEOP_CHOICES, default=None)
    ap.add_argument("--tunableop-file", default=None)
    ap.add_argument("--native-triton", choices=runtime.NATIVE_TRITON_CHOICES, default="auto")
    ap.add_argument("--compile", choices=("off", "default", "reduce-overhead", "max-autotune"), default="off")
    ap.add_argument("--name", default=None, help="result file stem (default: derived from the config)")
    return ap


def config_name(a) -> str:
    if a.name:
        return a.name
    parts = [a.model.split("/")[-1], a.impl, a.device, a.dtype]
    for k in ("blas", "fa", "tunableop"):
        v = getattr(a, k)
        if v not in (None, "default"):
            parts.append("%s-%s" % (k, v))
    if a.compile != "off":
        parts.append("compile-" + a.compile)
    return "_".join(parts)


def make_agent(a):
    """Build the agent a configuration describes. Returns (agent, load_seconds, backends)."""
    # Backend selection is process-global and applies to upstream `laya` just the same. The
    # native-Triton switch is needed by upstream too: without a C compiler, stock laya on
    # torch>=2.13 ROCm fails its first forward pass.
    backends = runtime.configure(blas=a.blas, fa=a.fa, tunableop=a.tunableop,
                                 tunableop_file=a.tunableop_file, native_triton=a.native_triton)
    t = time.perf_counter()
    if a.impl == "laya":
        import laya
        agent = laya.load(a.model, device=a.device)
        if a.dtype != "auto" and agent.device.type == "cuda":
            agent.dtype = DTYPES[a.dtype]
    else:
        agent = laya_rocm.load(a.model, device=a.device, dtype=a.dtype)
    load_s = time.perf_counter() - t
    if a.compile != "off":
        agent.model = torch.compile(agent.model, mode=None if a.compile == "default" else a.compile)
    return agent, load_s, backends


def meta(a, agent, load_s: float, backends) -> Dict[str, Any]:
    return {
        "config": config_name(a),
        "args": vars(a),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(agent.device),
        "amp_dtype": str(agent.dtype).replace("torch.", ""),
        "load_s": round(load_s, 2),
        "backends": backends,
        "environment": runtime.environment(),
    }


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def write_json(obj, name: str, subdir: str) -> str:
    d = os.path.join(RESULTS, subdir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print("wrote %s" % os.path.relpath(path, REPO), flush=True)
    return path


# ------------------------------------------------------------------ scoring (upstream, generalised)
def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


@torch.no_grad()
def score_cases(agent, cases, max_tokens=16384, max_seqs=128, progress=""):
    """cases: [(state, {qid: qdef})] -> (raw logits per question, index, seconds, n_dropped)."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0
    for ci, (state, questions) in enumerate(cases):
        for qid, qdef in questions.items():
            q = to_internal(qdef)
            try:
                ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            except Exception:
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1
                continue
            if len(markers) != len(render_options(q)):
                index.append((ci, qid, QTYPES[q["t"]], 0)); items.append(None); dropped += 1
                continue
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            index.append((ci, qid, QTYPES[q["t"]], len(markers)))

    dev = agent.device
    use_amp = dev.type == "cuda"
    valid = [i for i, it in enumerate(items) if it is not None]
    order = sorted(valid, key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    t0, done, i = time.time(), 0, 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"])); j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], agent.tok.pad_token_id)
        args = [b[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
        with torch.autocast(dev.type, dtype=agent.dtype, enabled=use_amp):
            logits, _ = agent.model(*args)
        logits = logits.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = logits[r, :len(sel[r]["markers"])]
        done += j - i
        if progress:
            el = time.time() - t0
            sys.stdout.write("\r  [%s] %d/%d seq | %.0f seq/s | ETA %ds    " % (
                progress, done, len(order), done / max(el, 1e-9), int(el * (len(order) - done) / max(1, done))))
            sys.stdout.flush()
        i = j
    if progress:
        print("\r  [%s] %d sequences in %.1fs (%d dropped)%s" % (
            progress, len(order), time.time() - t0, dropped, " " * 20), flush=True)
    return out, index, time.time() - t0, dropped


def softmax_t(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def temp_for(agent, qt, k, calibrated=True):
    if not calibrated:
        return 1.0
    return float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))


# ------------------------------------------------------------------ metrics (upstream)
def ece_score(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def macro_f1(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    f1 = []
    for c in sorted(set(gold.tolist()) | set(pred.tolist())):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1))


def aurc(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    o = np.argsort(-conf)
    return float((np.cumsum(1 - correct[o]) / np.arange(1, len(o) + 1)).mean())


def hard_metrics(rows):
    """rows: [(gold_idx, probs)] -> single-label classification + calibration metrics."""
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return {"n": 0}
    gold = np.array([g for g, _ in rows])
    pred = np.array([int(np.argmax(p)) for _, p in rows])
    conf = np.array([float(np.max(p)) for _, p in rows])
    corr = (pred == gold).astype(float)
    m = {"n": len(rows),
         "accuracy": float(corr.mean()),
         "macro_f1": macro_f1(gold, pred),
         "ece": ece_score(conf, corr),
         "brier": float(np.mean([((np.asarray(p) - np.eye(len(p))[g]) ** 2).sum() for g, p in rows])),
         "nll": float(np.mean([-math.log(max(float(p[g]), 1e-12)) for g, p in rows])),
         "aurc": aurc(conf, corr),
         "mean_confidence": float(conf.mean())}
    for cov in (0.5, 0.8):
        k = max(1, int(len(conf) * cov))
        m["acc_at_%d_coverage" % int(cov * 100)] = float(corr[np.argsort(-conf)[:k]].mean())
    return m


# ------------------------------------------------------------------ suites (upstream)
def load_typed_decisions():
    from datasets import load_dataset
    td = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    cases, golds = [], []
    for r in td:
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        gmap = {}
        for qid, qdef in questions.items():
            g = gold[qid]
            if qdef["type"] == "choice":
                keys = list(qdef["criteria"].keys())
                gmap[qid] = {"idx": keys.index(str(g["label"]))}
            elif qdef["type"] == "noul":
                gmap[qid] = {"idx": 1 if str(g["label"]).lower() == "true" else 0}
            else:
                gmap[qid] = {"idx": int(g["label"])}
        cases.append((state, questions))
        golds.append(gmap)
    return cases, golds


def load_ag_news(n=600):
    from datasets import load_dataset
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    cases, golds = [], []
    for r in list(d)[:n]:
        cases.append(({"article": r["text"]},
                      {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                                 "criteria": dict(crit)}}))
        golds.append({"topic": {"idx": int(r["label"])}})
    return cases, golds


SUITES = {"typed_decisions": load_typed_decisions, "en.ag_news": load_ag_news}


def finish(a):
    """Persist TunableOp results now rather than relying on interpreter shutdown.

    `write_file()` only exists in some torch versions; where it does not, results are flushed at
    exit anyway, so a missing API is not an error.
    """
    if a.tunableop == "tune":
        try:
            torch.cuda.tunable.write_file()
        except AttributeError:
            pass


def cuda_peak_mib() -> Optional[float]:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return None
