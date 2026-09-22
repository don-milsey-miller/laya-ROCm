"""Accuracy, calibration and agreement with a CPU FP32 reference, for one configuration.

  python bench/bench_accuracy.py --device cpu --dtype fp32 --name ref_cpu_fp32   # once: the reference
  python bench/bench_accuracy.py --impl laya_rocm --dtype bf16                   # any GPU configuration

Writes results/accuracy/<config>.json (upstream-shaped metrics per suite) and <config>.npz (raw
logits). When results/accuracy/<--reference>.npz exists, each suite also gets an `agreement`
block: top-1 agreement, mean/p99 max-|dp| and confidence-threshold flips at 0.5 and 0.85.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common  # noqa: E402

THRESHOLDS = (0.5, 0.85)


def pack(logits):
    k = np.array([0 if z is None else len(z) for z in logits], dtype=np.int32)
    arr = np.full((len(logits), max(1, int(k.max()) if len(k) else 1)), np.nan, dtype=np.float32)
    for i, z in enumerate(logits):
        if z is not None:
            arr[i, :len(z)] = z
    return arr, k


def calibrated_probs(agent, arr, k, index):
    return [None if kk == 0 else common.softmax_t(arr[i, :kk], common.temp_for(agent, qt, kk, True))
            for i, (kk, (_, _, qt, _)) in enumerate(zip(k, index))]


def agreement(p_ref, p):
    both = [(a, b) for a, b in zip(p_ref, p) if a is not None and b is not None]
    if not both:
        return {"n": 0}
    dp = np.array([float(np.abs(a - b).max()) for a, b in both])
    top1 = np.array([int(a.argmax()) == int(b.argmax()) for a, b in both])
    ca = np.array([float(a.max()) for a, _ in both])
    cb = np.array([float(b.max()) for _, b in both])
    out = {"n": len(both),
           "top1_agreement": round(float(top1.mean()), 5),
           "mean_abs_dp": float(dp.mean()),
           "p99_abs_dp": float(np.percentile(dp, 99)),
           "max_abs_dp": float(dp.max())}
    for t in THRESHOLDS:
        out["flips_at_%s" % t] = int(((ca >= t) != (cb >= t)).sum())
        out["flip_rate_at_%s" % t] = round(float(((ca >= t) != (cb >= t)).mean()), 5)
    return out


def main():
    ap = common.add_config_args(argparse.ArgumentParser())
    ap.add_argument("--suites", default=",".join(common.SUITES))
    ap.add_argument("--reference", default="ref_cpu_fp32", help="config stem of the reference run")
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args()

    agent, load_s, backends = common.make_agent(a)
    name = common.config_name(a)
    out = {"meta": common.meta(a, agent, load_s, backends), "suites": {}}
    print("%s on %s (%s)" % (name, agent.device, agent.dtype), flush=True)

    ref_path = os.path.join(common.RESULTS, "accuracy", a.reference + ".npz")
    ref = np.load(ref_path) if os.path.exists(ref_path) and a.reference != name else None
    saved = {}
    for suite in a.suites.split(","):
        cases, golds = common.SUITES[suite]()
        logits, index, secs, dropped = common.score_cases(agent, cases, max_tokens=a.max_tokens,
                                                          progress="%s/%s" % (name, suite))
        arr, k = pack(logits)
        saved[suite + "/logits"], saved[suite + "/k"] = arr, k
        probs = calibrated_probs(agent, arr, k, index)
        rows_cal, rows_raw = [], []
        for i, (ci, qid, qt, kk) in enumerate(index):
            g = golds[ci].get(qid)
            if g is None:
                continue
            rows_cal.append((g["idx"], probs[i]))
            rows_raw.append((g["idx"], None if kk == 0 else common.softmax_t(arr[i, :kk], 1.0)))
        n_ok = sum(1 for p in probs if p is not None)
        r = {"calibrated": common.hard_metrics(rows_cal), "raw": common.hard_metrics(rows_raw),
             "seconds": round(secs, 2), "dropped_questions": dropped,
             "questions_per_second": round(n_ok / max(secs, 1e-9), 1)}
        if ref is not None and suite + "/logits" in ref.files:
            p_ref = calibrated_probs(agent, ref[suite + "/logits"], ref[suite + "/k"], index)
            r["agreement"] = agreement(p_ref, probs)
            r["agreement"]["reference"] = a.reference
        out["suites"][suite] = r
        c = r["calibrated"]
        print("  %-16s acc %.4f  ece %.4f  brier %.4f  %.1f q/s  %s" % (
            suite, c["accuracy"], c["ece"], c["brier"], r["questions_per_second"],
            ("agree %.4f flips@.85 %d" % (r["agreement"]["top1_agreement"], r["agreement"]["flips_at_0.85"]))
            if "agreement" in r else ""), flush=True)

    common.finish(a)
    out["meta"]["peak_mib"] = common.cuda_peak_mib()
    common.write_json(out, name, "accuracy")
    np.savez_compressed(os.path.join(common.RESULTS, "accuracy", name + ".npz"), **saved)


if __name__ == "__main__":
    main()
