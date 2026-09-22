"""Build BENCHMARKS.md (and results/plots/*.png) from results/{latency,accuracy}/*.json.

  python bench/make_report.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
PLOTS = os.path.join(RESULTS, "plots")

# Upstream research/results/t4_colab_benchmark.json -> latency (Tesla T4, torch 2.11+cu128).
T4 = {"laya": {1: 39.5, 5: 84.5, 10: 158.6, 50: 771.3},
      "laya-multilingual": {1: 32.8, 5: 40.1, 10: 72.3, 50: 337.4}}
T4_TYPED = {"accuracy": 0.362, "ece": 0.1741, "brier": 0.7496}  # laya, calibrated, n=2000
T4_AGNEWS = {"accuracy": 0.9467, "ece": 0.0394, "brier": 0.0870}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_ab():
    p = os.path.join(RESULTS, "ab", "ab_paired.json")
    return json.load(open(p)) if os.path.exists(p) else None


def repeats(_unused=None):
    """Runs of one unchanged configuration, used as the noise floor."""
    lat = load("latency", skip_repeats=False)
    names = [n for n in lat if n == "laya_rocm_bf16" or n.startswith("laya_rocm_bf16_repeat")]
    out = {}
    for nq in (1, 10, 50):
        v = [lat[n]["latency"]["%d_questions" % nq]["p50_ms"] for n in names]
        if len(v) > 1:
            out[nq] = {"values": sorted(v), "spread_pct": 100 * (max(v) - min(v)) / min(v)}
    return out


def load(kind, skip_repeats=True):
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, kind, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.endswith("_quick") or (skip_repeats and "_repeat" in name):
            continue
        out[name] = json.load(open(p))
    return out


def model_of(r):
    return r["meta"]["args"]["model"].split("/")[-1]


def f(x, nd=1):
    return "–" if x is None else ("%.*f" % (nd, x))


def env_table(r):
    e = r["meta"]["environment"]
    rows = [("GPU", "%s (%s, %s CUs)" % (e.get("gpu"), e.get("gfx_arch"), e.get("compute_units"))),
            ("GPU memory visible", "%s GiB (unified with system RAM)" % e.get("vram_gib")),
            ("Platform", "%s%s" % (e.get("platform"), " — WSL2" if e.get("wsl") else "")),
            ("PyTorch / HIP", "%s / %s" % (e.get("torch"), e.get("hip"))),
            ("transformers / laya", "%s / %s" % (e.get("transformers"), e.get("laya"))),
            ("Default backends", "BLAS %s, flash-attention %s, CK SDPA %s" % (
                {"cublaslt": "hipBLASLt", "cublas": "rocBLAS"}.get(str(e["backends"].get("blas")).lower(),
                                                                  e["backends"].get("blas")),
                e["backends"].get("fa"),
                "available" if e["backends"].get("ck_sdpa_available") else "unavailable")),
            ("C compiler", e["backends"].get("c_compiler") or "none (torch._native Triton ops off)")]
    return "| | |\n|---|---|\n" + "\n".join("| %s | %s |" % r for r in rows)


def headline(lat):
    lines = ["| Configuration | 1 q | 5 q | 10 q | 50 q | ms / question @50 |", "|---|---:|---:|---:|---:|---:|"]
    for m, t in T4.items():
        lines.append("| **Tesla T4** — upstream `%s` (published) | %s | %s | %s | %s | %s |" % (
            m, t[1], t[5], t[10], t[50], f(t[50] / 50, 2)))
    for name, r in lat.items():
        L = r["latency"]
        lines.append("| `%s` | %s | %s | %s | %s | %s |" % (
            name, *(f(L["%d_questions" % n]["p50_ms"]) for n in (1, 5, 10, 50)),
            f(L["50_questions"]["ms_per_question"], 2)))
    return "\n".join(lines)


def config_table(lat):
    lines = ["| Configuration | model | AMP | cold first call ms | peak q/s (sweep) | host share @1q | peak MiB |",
             "|---|---|---|---:|---:|---:|---:|"]
    for name, r in lat.items():
        sw = r.get("sweep", [])
        best = max(sw, key=lambda s: s["questions_per_s"]) if sw else None
        one = next((s for s in sw if s["questions"] == 1 and s["state_tokens"] == 128), None)
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            name, model_of(r), r["meta"]["amp_dtype"], f(r["cold"]["first_call_ms"], 0),
            ("%s (%dq × %d tok)" % (f(best["questions_per_s"]), best["questions"], best["state_tokens"])) if best else "–",
            ("%.0f%%" % (100 * one["host_share"])) if one and "host_share" in one else "–",
            f(max(s["peak_mib"] or 0 for s in sw), 0) if sw else "–"))
    return "\n".join(lines)


def sweep_table(r):
    sw = r["sweep"]
    lens = sorted({s["state_tokens"] for s in sw})
    nqs = sorted({s["questions"] for s in sw})
    cell = {(s["state_tokens"], s["questions"]): s for s in sw}
    lines = ["| questions / call | " + " | ".join("%d tok: p50 ms (q/s)" % L for L in lens) + " |",
             "|---:|" + "---:|" * len(lens)]
    for n in nqs:
        lines.append("| %d | " % n + " | ".join(
            "%s (%s)" % (f(cell[(L, n)]["p50_ms"]), f(cell[(L, n)]["questions_per_s"], 0)) for L in lens) + " |")
    return "\n".join(lines)


def micro_table(r):
    lines = ["| states / call | `predict_many` p50 ms | sequential `predict` p50 ms | speed-up |", "|---:|---:|---:|---:|"]
    for m in r.get("micro_batch", []):
        lines.append("| %d | %s | %s | ×%.2f |" % (m["states"], f(m["batched"]["p50_ms"]), f(m["sequential"]["p50_ms"]), m["speedup"]))
    return "\n".join(lines)


def accuracy_table(acc, suite, t4):
    lines = ["| Configuration | device / AMP | accuracy | ECE | Brier | top-1 agree vs CPU FP32 | mean \\|Δp\\| | p99 \\|Δp\\| | flips @0.5 | flips @0.85 | q/s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append("| **Tesla T4** upstream (published) | cuda / bf16 | %.4f | %.4f | %.4f | | | | | | |"
                 % (t4["accuracy"], t4["ece"], t4["brier"]))
    for name, r in acc.items():
        s = r["suites"].get(suite)
        if not s:
            continue
        c, ag = s["calibrated"], s.get("agreement", {})
        lines.append("| `%s` | %s / %s | %.4f | %.4f | %.4f | %s | %s | %s | %s | %s | %s |" % (
            name, r["meta"]["device"], r["meta"]["amp_dtype"].replace("bfloat16", "bf16").replace("float", "fp"),
            c["accuracy"], c["ece"], c["brier"],
            f(ag.get("top1_agreement"), 4) if ag else "ref", f(ag.get("mean_abs_dp"), 5) if ag else "",
            f(ag.get("p99_abs_dp"), 4) if ag else "",
            "%d / %d" % (ag["flips_at_0.5"], ag["n"]) if ag else "",
            "%d / %d" % (ag["flips_at_0.85"], ag["n"]) if ag else "", f(s["questions_per_second"])))
    return "\n".join(lines)


def plots(lat, default):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    os.makedirs(PLOTS, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    made = []

    r = lat[default]
    sw = r["sweep"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, L in enumerate(sorted({s["state_tokens"] for s in sw})):
        pts = sorted((s["questions"], s["questions_per_s"]) for s in sw if s["state_tokens"] == L)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label="%d-token state" % L)
        ax.annotate("%d tok" % L, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=MUTED, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 50]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "50"])
    ax.set_xlabel("questions per call"); ax.set_ylabel("questions / second")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Throughput vs batch — %s, Radeon 8060S" % default, loc="left", fontsize=11)
    fig.tight_layout()
    p = os.path.join(PLOTS, "throughput_sweep.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)

    names = [n for n in lat if model_of(lat[n]) == "laya"]
    vals = [lat[n]["latency"]["1_questions"]["p50_ms"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(range(len(names)), vals, height=0.6, color=SERIES[0])
    ax.axvline(T4["laya"][1], color=MUTED, lw=1, ls="--")
    ax.annotate("T4 (upstream) %.1f ms" % T4["laya"][1], (T4["laya"][1], len(names) - 0.5),
                xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=9)
    for i, v in enumerate(vals):
        ax.annotate("%.1f" % v, (v, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("p50 latency, 1 question (ms) — lower is better")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(PLOTS, "latency_1q_by_config.png"); fig.savefig(p, dpi=150); plt.close(fig)
    made.append(p)
    return made


def variance_section(lat, ab):
    rep = repeats()
    md = ["## How much of this is noise", "",
          "This GPU shares its power and thermal budget with the CPU, and its clocks move with the "
          "machine's state. The **same unchanged configuration**, run at different points across one "
          "afternoon, measured:", ""]
    if rep:
        md += ["| questions / call | p50 ms across identical runs | spread |", "|---:|---|---:|"]
        for nq, r in sorted(rep.items()):
            md.append("| %d | %s | **%.0f%%** |" % (nq, ", ".join("%.1f" % v for v in r["values"]), r["spread_pct"]))
        md += ["", "Single-question latency is the worst case: between calls the GPU goes idle and clocks "
                   "down, so the measurement reflects how fast it ramps back up. Sustained batches hold their "
                   "clocks and are steadier.", ""]
    md += ["**Consequence: any wall-clock difference below roughly 10-20% here is noise, not a result.** The "
           "absolute latency tables below were recorded across several hours and should be read with that in "
           "mind. Differences between configurations come from the paired runs instead.", ""]
    if ab:
        m = ab["meta"]
        md += ["### Paired comparison", "",
               "Each round runs every configuration back to back in fresh processes, so all of them meet the "
               "same machine state; the ratio within a round is what survives drift. %d rounds, %d timed "
               "repetitions per probe, baseline `%s`." % (m["rounds"], m["reps_per_probe"], m["baseline"]), "",
               "| Configuration | 1 q | 10 q | 50 q |", "|---|---|---|---|"]
        for name, rec in ab["configs"].items():
            cells = []
            for nq in (1, 10, 50):
                c = rec.get("%dq" % nq)
                cells.append("–" if not c else "%.1f ms — **×%.3f** (%.3f–%.3f)" % (
                    c["median_ms"], c["ratio_vs_baseline_median"], c["ratio_min"], c["ratio_max"]))
            md.append("| `%s` | %s |" % (name, " | ".join(cells)))
        md += ["", "Ratio below 1.000 is faster than the baseline. Where the min-max range straddles 1.000, "
                   "the configuration is indistinguishable from the baseline on this hardware.", ""]
    return md


def summary(lat, acc, log, ab):
    """The findings. Speed claims come from the paired runs; accuracy claims from the logits."""
    def ratio(cfg, nq):
        return (ab or {}).get("configs", {}).get(cfg, {}).get("%dq" % nq)

    def agree(cfg, suite="typed_decisions"):
        return acc.get(cfg, {}).get("suites", {}).get(suite, {}).get("agreement", {})

    def verdict(c):
        """A paired ratio only counts when its whole min-max range sits on one side of 1.0."""
        if not c:
            return None
        if c["ratio_min"] > 1.0:
            return "%.0f%% slower" % (100 * (c["ratio_vs_baseline_median"] - 1))
        if c["ratio_max"] < 1.0:
            return "%.0f%% faster" % (100 * (1 - c["ratio_vs_baseline_median"]))
        return "no measurable difference"

    def host_share(cfg):
        sw = lat.get(cfg, {}).get("sweep", [])
        v = next((s.get("host_share") for s in sw if s["questions"] == 1 and s["state_tokens"] == 128), None)
        return "-" if v is None else "%.0f%%" % (100 * v)

    out = ["## What the numbers say", ""]
    ref_acc = (acc.get("ref_cpu_fp32", {}).get("suites", {}).get("typed_decisions", {})
               .get("calibrated", {}).get("accuracy"))
    a32 = agree("laya_rocm_bf16")
    if a32 and ref_acc:
        out.append("- **Laya runs on ROCm and the answers are right.** GPU BF16 matches a CPU FP32 run of the "
                   "same checkpoint on %.2f%% of typed-decisions top-1 answers, and GPU FP32 reproduces it "
                   "exactly. The CPU reference itself reproduces upstream's published T4 accuracy (%.4f vs "
                   "%.4f), so this harness measures what upstream measured."
                   % (100 * a32["top1_agreement"], ref_acc, T4_TYPED["accuracy"]))

    sw = lat.get("laya_rocm_bf16", {}).get("sweep", [])
    if sw:
        best = max(sw, key=lambda s: s["questions_per_s"])
        out.append("- **The GPU, not Python, is the limit.** Only %s of a single-question call happens outside "
                   "the forward pass, and throughput saturates near %.0f questions/s (%dq x %d tok). Little is "
                   "left to win on the host side; GEMM efficiency is the lever."
                   % (host_share("laya_rocm_bf16"), best["questions_per_s"], best["questions"], best["state_tokens"]))

    a16 = agree("laya_rocm_fp16")
    if a16 and a32:
        out.append("- **FP16 is the accuracy choice, not the speed choice.** It tracks the FP32 reference far "
                   "better than BF16 does: top-1 agreement %.4f vs %.4f, %d vs %d flips at the 0.85 confidence "
                   "threshold, p99 |dp| %.4f vs %.4f. It is not faster -- paired, it is %s at 10 questions and "
                   "%s at 50. BF16's wider exponent buys nothing for this model while its shorter mantissa costs "
                   "precision, so FP16 is the better trade wherever a confidence threshold drives a decision. "
                   "It is **not** the default -- one GPU is not evidence for every GPU -- so pass `dtype=\"fp16\"`."
                   % (a16["top1_agreement"], a32["top1_agreement"], a16["flips_at_0.85"], a32["flips_at_0.85"],
                      a16["p99_abs_dp"], a32["p99_abs_dp"],
                      verdict(ratio("fp16", 10)) or "unclear", verdict(ratio("fp16", 50)) or "unclear"))

    tune_s = log.get("latency/laya_rocm_bf16_tunableop-tune", {}).get("seconds")
    vt = verdict(ratio("bf16_tunableop", 1))
    if vt:
        out.append("- **TunableOp: %s at one question, %s at fifty -- and it has to be paid for.** The first "
                   "tuning pass, with nothing cached, took about an hour: it benchmarks kernel candidates for "
                   "every new GEMM shape the workload produces.%s Tuning must be repeated per GPU and after "
                   "ROCm/PyTorch upgrades, and the CSV is not portable. Worth it for a long-lived service on "
                   "stable shapes, not for a short job."
                   % (vt, verdict(ratio("bf16_tunableop", 50)) or "unclear",
                      (" A later pass reusing that CSV took %.0f minutes." % (tune_s / 60)) if tune_s else ""))

    vr = verdict(ratio("bf16_rocblas", 50))
    if vr:
        out.append("- **rocBLAS instead of the default hipBLASLt: %s.** A single early run suggested a large "
                   "rocBLAS win on one workload; the paired runs do not support it. Measure your own shapes "
                   "before switching." % vr)

    vu = verdict(ratio("upstream_bf16", 50))
    if vu:
        out.append("- **Upstream `laya` on the same GPU measures %s than `laya_rocm`** (paired, and the same at "
                   "1 and 10 questions). Caching the tokenized question head removes host-side work, which is "
                   "real but small beside the forward pass -- only %s of a one-question call is host-side to "
                   "begin with. The package's value is that it *runs* and picks sane precision, not that it "
                   "makes the GPU faster." % (vu, host_share("laya_rocm_bf16")))

    ml, en = lat.get("laya-multilingual_rocm_bf16"), lat.get("laya_rocm_bf16")
    if ml and en:
        out.append("- **The multilingual checkpoint is about twice as fast as the English one** (%.1f vs %.1f ms "
                   "at one question) because it is a smaller model (322M vs 421M parameters). It also ships "
                   "uncalibrated, so its confidences deserve less trust -- upstream's finding, unrelated to ROCm."
                   % (ml["latency"]["1_questions"]["p50_ms"], en["latency"]["1_questions"]["p50_ms"]))

    mb = lat.get("laya_rocm_bf16", {}).get("micro_batch", [])
    if mb:
        out.append("- **`predict_many` wins only a little on this GPU** (up to x%.2f, and noisy) because a single "
                   "call already saturates it. It helps more on the smaller multilingual checkpoint, which does not."
                   % max(m["speedup"] for m in mb))

    out.append("- **Without a C compiler, stock `laya` cannot complete a GPU forward pass on torch 2.13 + ROCm "
               "at all** -- a Triton kernel in `torch._native` JIT-builds a C stub. `laya_rocm` detects this and "
               "routes those ops to the stock kernels. That is a correctness fix, not a speed one.")
    return out + [""]


def main():
    lat, acc = load("latency"), load("accuracy")
    log_path = os.path.join(RESULTS, "matrix_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    default = "laya_rocm_bf16" if "laya_rocm_bf16" in lat else next(iter(lat), None)
    if default is None:
        raise SystemExit("no latency results yet")
    pngs = plots(lat, default)

    md = ["# laya-rocm benchmarks", "",
          "Laya on an AMD Radeon 8060S (Ryzen AI Max+ 395, gfx1151) through ROCm on WSL2. Every configuration "
          "ran in a fresh process. Latency is wall-clock `predict()` time after warm-up, synchronised with the "
          "GPU. Generated by `bench/make_report.py` from `results/`.", "",
          "## Environment", "", env_table(lat[default]), ""]
    for p in pngs:
        md += ["![%s](%s)" % (os.path.basename(p), os.path.relpath(p, REPO).replace(os.sep, "/")), ""]
    ab = load_ab()
    md += summary(lat, acc, log, ab) + variance_section(lat, ab) + ["## Latency against upstream's T4 numbers", "",
           "p50 ms per `predict()` call on upstream's own latency workload (a ~100-token support ticket, "
           "alternating choice / yes-no questions). Recorded across several hours, so treat small "
           "differences between rows as noise -- see the paired table above.", "", headline(lat), "",
           "## Configurations", "",
           "*host share* is the fraction of wall time spent outside the model's forward pass (tokenization, "
           "collation, copies, decoding); forward time is measured with CUDA events.", "", config_table(lat), "",
           "## Sweep: `%s`" % default, "", sweep_table(lat[default]), "",
           "## Micro-batching with `predict_many` (4 questions, 128-token states)", "", micro_table(lat[default]), ""]
    if acc:
        md += ["## Accuracy and calibration", "",
               "Calibrated metrics, same formulas as upstream. Agreement compares every question's calibrated "
               "probabilities with a CPU FP32 run of the same checkpoint; a *flip* is a question whose top "
               "confidence lands on the other side of the threshold.", "",
               "### typed-decisions (2,000 decisions, zero-shot)", "", accuracy_table(acc, "typed_decisions", T4_TYPED), "",
               "### AG News (600, in training mix)", "", accuracy_table(acc, "en.ag_news", T4_AGNEWS), ""]
    bad = {k: v for k, v in log.items() if v["status"] in ("failed", "skipped")}
    if bad:
        md += ["## Configurations that did not run", "", "| configuration | status | reason |", "|---|---|---|"]
        for k, v in bad.items():
            reason = v.get("reason") or (v.get("stderr_tail", "").strip().splitlines() or [""])[-1][:160]
            md.append("| `%s` | %s | %s |" % (k, v["status"], reason.replace("|", "\\|")))
        md.append("")
    md += ["## Caveats", "",
           "- WSL2 runs are reported to reach roughly 70–80% of native Linux throughput; ROCm profilers are "
           "unavailable in WSL and pinned host memory is disabled.",
           "- The 8060S is an iGPU: its memory is carved out of system RAM, and CPU load competes for the same bandwidth.",
           "- Without a C compiler, torch 2.13's Triton-backed `torch._native` ops are switched off (stock kernels "
           "are used) and `torch.compile` is unavailable. This applies to upstream `laya` as well: on this stack it "
           "otherwise fails its first GPU forward pass.",
           "- The T4 rows are upstream's published numbers (torch 2.11 + CUDA 12.8), not re-measured here.", ""]
    out = os.path.join(REPO, "BENCHMARKS.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))
    print("wrote BENCHMARKS.md", *[os.path.relpath(p, REPO) for p in pngs])


if __name__ == "__main__":
    main()
