"""`python -m laya_rocm doctor` -- check that Laya will actually run on the AMD GPU."""
import argparse
import json
import sys
import time


def doctor(model: str, smoke: bool) -> int:
    import torch

    from . import runtime

    runtime.configure()  # apply the same defaults a load would, so the report is what you will get
    env = runtime.environment()
    print(json.dumps(env, indent=2, default=str))
    ok = True
    if not runtime.is_rocm():
        print("\n[!] This PyTorch build is not ROCm (torch.version.hip is None).")
        print("    Install a ROCm wheel: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html")
        ok = False
    elif not torch.cuda.is_available():
        print("\n[!] ROCm PyTorch is installed but no GPU is visible.")
        if runtime.is_wsl():
            print("    WSL: install librocdxg (https://github.com/ROCm/librocdxg) and, for ROCm < 7.13,"
                  " export HSA_ENABLE_DXG_DETECTION=1")
        ok = False
    elif runtime.c_compiler() is None:
        print("\n[i] No C compiler found. Triton needs one to JIT its launcher stubs, so laya_rocm turns off")
        print("    torch._native Triton ops and uses stock kernels. torch.compile will not work either.")
        print("    To enable both: sudo apt install gcc   (or point CC at a compiler)")
    if ok and smoke:
        from .agent import load
        t = time.perf_counter()
        agent = load(model, device="cuda")
        print("\nloaded %s in %.1fs on %s (%s)" % (model, time.perf_counter() - t, agent.device, agent.dtype))
        q = {"s": {"type": "choice", "instructions": "What is the sentiment?",
                   "criteria": {"positive": "positive", "negative": "negative", "neutral": "neutral"}}}
        r = agent.predict({"text": "This works surprisingly well."}, q)
        print(json.dumps(r["answers"], indent=2))
        if agent.device.type != "cuda":
            print("[!] Model ended up on %s, not the GPU." % agent.device)
            ok = False
    print("\nOK" if ok else "\nNOT OK")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="laya-rocm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="report the ROCm environment and run a GPU smoke test")
    d.add_argument("--model", default="convaiinnovations/laya")
    d.add_argument("--no-smoke", action="store_true", help="skip loading the model")
    a = ap.parse_args(argv)
    if a.cmd == "doctor":
        return doctor(a.model, not a.no_smoke)
    return 2


if __name__ == "__main__":
    sys.exit(main())
