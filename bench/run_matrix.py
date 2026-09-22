"""Run every benchmark configuration, each in a fresh process (backend choices are process-global).

  python bench/run_matrix.py [--only latency|accuracy] [--filter substr] [--force] [--dry-run]

Existing result files are skipped unless --force. Failures are recorded (with the stderr tail) in
results/matrix_log.json rather than aborting the run -- a configuration that cannot run on this
stack (e.g. torch.compile without a C compiler) is itself a result.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
TUNE_FILE = os.path.join(RESULTS, "tunableop", "tunableop_results0.csv")

EN = "convaiinnovations/laya"
ML = "convaiinnovations/laya-multilingual"

# (name, extra args). Order matters: TunableOp "tune" must precede "use".
LATENCY = [
    ("laya_upstream_bf16", ["--impl", "laya"]),
    ("laya_rocm_bf16", []),
    ("laya_rocm_fp16", ["--dtype", "fp16"]),
    ("laya_rocm_fp32", ["--dtype", "fp32"]),
    ("laya_rocm_bf16_rocblas", ["--blas", "rocblas"]),
    ("laya_rocm_fp16_rocblas", ["--dtype", "fp16", "--blas", "rocblas"]),
    ("laya_rocm_bf16_tunableop-tune", ["--tunableop", "tune", "--tunableop-file", TUNE_FILE]),
    ("laya_rocm_bf16_tunableop-use", ["--tunableop", "use", "--tunableop-file", TUNE_FILE]),
    ("laya_rocm_bf16_native-triton", ["--native-triton", "on"]),
    ("laya_rocm_bf16_compile", ["--compile", "default"]),
    ("laya-multilingual_upstream_bf16", ["--impl", "laya", "--model", ML]),
    ("laya-multilingual_rocm_bf16", ["--model", ML]),
]

ACCURACY = [
    ("ref_cpu_fp32", ["--device", "cpu", "--dtype", "fp32"]),
    ("laya_upstream_bf16", ["--impl", "laya"]),
    ("laya_rocm_bf16", []),
    ("laya_rocm_fp16", ["--dtype", "fp16"]),
    ("laya_rocm_fp32", ["--dtype", "fp32"]),
    ("laya_rocm_bf16_rocblas", ["--blas", "rocblas"]),
    ("laya_rocm_fp16_rocblas", ["--dtype", "fp16", "--blas", "rocblas"]),
    ("laya_rocm_bf16_tunableop-use", ["--tunableop", "use", "--tunableop-file", TUNE_FILE]),
]

NEEDS_CC = ("native-triton", "compile")


def has_cc():
    return bool(os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang"))


def run(kind, name, extra, force, dry):
    out = os.path.join(RESULTS, kind, name + ".json")
    if os.path.exists(out) and not force:
        return {"status": "exists"}
    if any(k in name for k in NEEDS_CC) and not has_cc():
        return {"status": "skipped", "reason": "needs a C compiler (Triton/Inductor JIT); install gcc"}
    if "tunableop-use" in name and not os.path.exists(TUNE_FILE):
        return {"status": "skipped", "reason": "no tuned results yet; run the latency tunableop-tune config first"}
    cmd = [sys.executable, os.path.join(HERE, "bench_%s.py" % kind), "--name", name] + extra
    print("\n=== %s/%s\n    %s" % (kind, name, " ".join(cmd[1:])), flush=True)
    if dry:
        return {"status": "dry-run"}
    os.makedirs(os.path.dirname(TUNE_FILE), exist_ok=True)
    t = time.time()
    p = subprocess.run(cmd, cwd=REPO, stdout=None, stderr=subprocess.PIPE, text=True)
    rec = {"status": "ok" if p.returncode == 0 else "failed", "returncode": p.returncode,
           "seconds": round(time.time() - t, 1)}
    if p.returncode != 0:
        rec["stderr_tail"] = p.stderr[-3000:]
        print(p.stderr[-3000:], file=sys.stderr, flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("latency", "accuracy"))
    ap.add_argument("--filter", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    log_path = os.path.join(RESULTS, "matrix_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    plan = [("latency", n, e) for n, e in LATENCY] + [("accuracy", n, e) for n, e in ACCURACY]
    for kind, name, extra in plan:
        if (a.only and kind != a.only) or a.filter not in name:
            continue
        rec = run(kind, name, extra, a.force, a.dry_run)
        if rec["status"] not in ("exists", "dry-run"):
            log["%s/%s" % (kind, name)] = dict(rec, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            os.makedirs(RESULTS, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
        print("    -> %s" % rec["status"], flush=True)


if __name__ == "__main__":
    main()
