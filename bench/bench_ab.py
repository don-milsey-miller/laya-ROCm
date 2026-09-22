"""Paired A/B comparison of configurations, interleaved in time.

Wall-clock latency on this machine drifts by 10-20% over hours (iGPU clocks share a power and
thermal budget with the CPU), which is larger than the differences between backend settings. A
configuration measured this morning cannot be compared with one measured this afternoon.

This driver runs every configuration in *rounds*: A, B, C, A, B, C, ... each in a fresh process,
each round a short probe. Configurations are therefore exposed to the same drift, and a
comparison within a round is meaningful. Reported per configuration:

  median-of-rounds p50, and the per-round ratio against the baseline configuration (the first
  one), summarised as median and range -- the ratio is what survives drift.

  python bench/bench_ab.py --rounds 5 [--probe-only 1,10,50]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
TUNE_FILE = os.path.join(RESULTS, "tunableop", "tunableop_results0.csv")

# The comparisons worth making. The first entry is the baseline every ratio is against.
CONFIGS = [
    ("bf16", []),
    ("fp16", ["--dtype", "fp16"]),
    ("bf16_rocblas", ["--blas", "rocblas"]),
    ("fp16_rocblas", ["--dtype", "fp16", "--blas", "rocblas"]),
    ("bf16_tunableop", ["--tunableop", "use", "--tunableop-file", TUNE_FILE]),
    ("upstream_bf16", ["--impl", "laya"]),
]


def probe(extra, questions, reps):
    """One short measurement in a fresh process; returns {n_questions: p50_ms}."""
    code = (
        "import sys, time, json;sys.path.insert(0, %r);"
        "import argparse, numpy as np, common, bench_latency as B;"
        "ap = common.add_config_args(argparse.ArgumentParser());a = ap.parse_args();"
        "agent, _, _ = common.make_agent(a);dev = agent.device;t = B.ForwardTimer(agent.model, dev);"
        "out = {};"
        "res = [out.update({str(n): B.measure(lambda: agent.system_one(B.LAT_STATE, B.qs(n)), dev, t, 3, %d, n)['p50_ms']}) for n in %r];"
        "print('RESULT' + json.dumps(out))" % (HERE, reps, list(questions))
    )
    p = subprocess.run([sys.executable, "-c", code] + extra, cwd=REPO, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        raise RuntimeError("probe failed: %s" % (p.stderr[-2000:] or p.stdout[-2000:]))
    return {int(k): v for k, v in json.loads(line[-1][len("RESULT"):]).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--questions", default="1,10,50")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default="ab_paired")
    a = ap.parse_args()
    questions = [int(x) for x in a.questions.split(",")]

    raw = {name: {n: [] for n in questions} for name, _ in CONFIGS}
    for r in range(a.rounds):
        for name, extra in CONFIGS:
            t = time.time()
            got = probe(extra, questions, a.reps)
            for n, ms in got.items():
                raw[name][n].append(ms)
            print("round %d/%d  %-16s %s  (%.0fs)" % (
                r + 1, a.rounds, name,
                "  ".join("%dq %7.2f ms" % (n, got[n]) for n in questions), time.time() - t), flush=True)

    base = CONFIGS[0][0]
    out = {"meta": {"rounds": a.rounds, "reps_per_probe": a.reps, "baseline": base,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "each round runs every configuration back to back in fresh processes; "
                            "ratios are paired within a round, so they survive clock drift"},
           "configs": {}}
    print("\n%-16s %s" % ("config", "  ".join("%dq: median ms | ratio vs %s (min-max)" % (n, base) for n in questions)))
    for name, _ in CONFIGS:
        rec = {}
        for n in questions:
            vals = raw[name][n]
            ratios = [v / b for v, b in zip(vals, raw[base][n])]
            rec["%dq" % n] = {"p50_ms_per_round": [round(v, 2) for v in vals],
                              "median_ms": round(statistics.median(vals), 2),
                              "drift_range_pct": round(100 * (max(vals) - min(vals)) / min(vals), 1),
                              "ratio_vs_baseline_median": round(statistics.median(ratios), 4),
                              "ratio_min": round(min(ratios), 4), "ratio_max": round(max(ratios), 4)}
        out["configs"][name] = rec
        print("%-16s %s" % (name, "  ".join(
            "%7.2f  ×%.3f (%.3f-%.3f)" % (rec["%dq" % n]["median_ms"], rec["%dq" % n]["ratio_vs_baseline_median"],
                                          rec["%dq" % n]["ratio_min"], rec["%dq" % n]["ratio_max"])
            for n in questions)), flush=True)
    drift = max(rec["%dq" % n]["drift_range_pct"] for rec in out["configs"].values() for n in questions)
    out["meta"]["max_drift_range_pct"] = drift
    print("\nlargest spread for one unchanged configuration across rounds: %.1f%%" % drift)

    d = os.path.join(RESULTS, "ab")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, a.out + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/ab/%s.json" % a.out)


if __name__ == "__main__":
    main()
