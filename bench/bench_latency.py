"""Latency / throughput for one configuration, in a fresh process.

  python bench/bench_latency.py --impl laya_rocm --dtype bf16 [--quick]

Sections of the output JSON (results/latency/<config>.json):
  latency      upstream-shaped: 1/5/10/50 questions on upstream's LAT_STATE (compare with the T4 run)
  sweep        questions/call x state length, p50/p95/p99, q/s, forward-only GPU time, peak memory
  micro_batch  predict_many over N states vs N sequential predict calls (laya_rocm only)
  cold         model load time and first-call latency (kernel selection, TunableOp, compile)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402

# Upstream's latency workload (research/scripts/build_benchmark_nb.py, section 7).
LAT_STATE = {"ticket": {"subject": "Payout failing",
             "messages": [{"from": "customer",
                           "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}

FILLER = ("I was charged twice for my subscription this month and the second charge still shows as pending. "
          "Support told me it would clear in two days but it has been a week. ")


def qs(n):
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


def state_of_tokens(tok, n_tokens: int):
    """A ticket whose message text is ~n_tokens long under this tokenizer."""
    ids = tok(FILLER * (n_tokens // 20 + 2), add_special_tokens=False)["input_ids"][:n_tokens]
    text = tok.decode(ids, clean_up_tokenization_spaces=False)
    return {"ticket": {"subject": "Double charge", "messages": [{"from": "customer", "text": text}]}}


class ForwardTimer:
    """CUDA events around every model forward: GPU-timeline time of the encoder+head only."""

    def __init__(self, model, device):
        self.on = torch.device(device).type == "cuda"
        self.pairs = []
        if self.on:
            model.register_forward_pre_hook(self._pre)
            model.register_forward_hook(self._post)

    def _pre(self, *_):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.pairs.append([e, None])

    def _post(self, *_):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.pairs[-1][1] = e

    def reset(self):
        self.pairs = []

    def total_ms(self):
        return sum(a.elapsed_time(b) for a, b in self.pairs if b is not None) if self.on else None


def measure(fn, device, timer, warmup, reps, n_questions):
    for _ in range(warmup):
        fn()
    common.sync(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    for _ in range(reps):
        timer.reset()
        common.sync(device)
        t = time.perf_counter()
        fn()
        common.sync(device)
        wall.append((time.perf_counter() - t) * 1000)
        g = timer.total_ms()
        if g is not None:
            gpu.append(g)
    p50 = float(np.percentile(wall, 50))
    r = {"p50_ms": round(p50, 2),
         "p95_ms": round(float(np.percentile(wall, 95)), 2),
         "p99_ms": round(float(np.percentile(wall, 99)), 2),
         "mean_ms": round(float(np.mean(wall)), 2),
         "min_ms": round(float(np.min(wall)), 2),
         "ms_per_question": round(p50 / n_questions, 3),
         "questions_per_s": round(n_questions / (p50 / 1000), 1),
         "reps": reps}
    if gpu:
        g50 = float(np.percentile(gpu, 50))
        r["forward_p50_ms"] = round(g50, 2)
        r["host_share"] = round(max(0.0, 1 - g50 / p50), 3)
    r["peak_mib"] = common.cuda_peak_mib()
    return r


def main():
    ap = common.add_config_args(argparse.ArgumentParser())
    ap.add_argument("--quick", action="store_true", help="small sweep, few reps (smoke-test the harness)")
    a = ap.parse_args()

    agent, load_s, backends = common.make_agent(a)
    dev = agent.device
    timer = ForwardTimer(agent.model, dev)
    out = {"meta": common.meta(a, agent, load_s, backends)}
    print("%s: %s on %s (%s), load %.1fs" % (out["meta"]["config"], a.impl, dev, agent.dtype, load_s), flush=True)

    warm, reps = (2, 5) if a.quick else (5, 30)

    # cold: first call pays kernel selection / TunableOp tuning / compilation
    t = time.perf_counter()
    agent.system_one(LAT_STATE, qs(1))
    common.sync(dev)
    out["cold"] = {"load_s": round(load_s, 2), "first_call_ms": round((time.perf_counter() - t) * 1000, 1)}

    # upstream-shaped block (upstream used 3 warmup / 20 reps; we keep the same question mix)
    out["latency"] = {}
    for nq in (1, 5, 10, 50):
        q = qs(nq)
        r = measure(lambda: agent.system_one(LAT_STATE, q), dev, timer, warm, reps if not a.quick else 5, nq)
        out["latency"]["%d_questions" % nq] = r
        print("  latency %2dq  p50 %7.2f ms  p95 %7.2f ms  %7.1f q/s  fwd %s ms"
              % (nq, r["p50_ms"], r["p95_ms"], r["questions_per_s"], r.get("forward_p50_ms")), flush=True)

    # sweep: questions/call x state tokens
    nqs = (1, 4, 16, 50) if a.quick else (1, 2, 4, 8, 16, 32, 50)
    lens = (64, 512) if a.quick else (64, 128, 256, 512)
    out["sweep"] = []
    for L in lens:
        st = state_of_tokens(agent.tok, L)
        for nq in nqs:
            q = qs(nq)
            r = measure(lambda: agent.system_one(st, q), dev, timer, warm, reps, nq)
            r.update({"state_tokens": L, "questions": nq})
            out["sweep"].append(r)
            print("  sweep %4d tok %2dq  p50 %8.2f ms  %7.1f q/s  host %s  peak %s MiB"
                  % (L, nq, r["p50_ms"], r["questions_per_s"], r.get("host_share"), r["peak_mib"]), flush=True)

    # micro-batching: N requests with the same question set
    if hasattr(agent, "predict_many"):
        out["micro_batch"] = []
        st = state_of_tokens(agent.tok, 128)
        q = qs(4)
        for n in ((1, 8, 32) if a.quick else (1, 2, 4, 8, 16, 32)):
            states = [st] * n
            batched = measure(lambda: agent.predict_many(states, q), dev, timer, warm, reps, n * len(q))
            seq = measure(lambda: [agent.system_one(s, q) for s in states], dev, timer, 1, max(3, reps // 4), n * len(q))
            out["micro_batch"].append({"states": n, "questions_per_state": len(q), "state_tokens": 128,
                                       "batched": batched, "sequential": seq,
                                       "speedup": round(seq["p50_ms"] / batched["p50_ms"], 2)})
            print("  micro-batch %2d states  batched %8.2f ms  sequential %8.2f ms  x%.2f"
                  % (n, batched["p50_ms"], seq["p50_ms"], seq["p50_ms"] / batched["p50_ms"]), flush=True)

    common.finish(a)
    out["meta"]["backends_after"] = common.runtime.effective_backends()
    common.write_json(out, out["meta"]["config"] + ("_quick" if a.quick else ""), "latency")


if __name__ == "__main__":
    main()
