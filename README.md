# laya-rocm

Drop-in AMD ROCm runtime for [Laya](https://github.com/NandhaKishorM/laya), the calibrated
"System 1" decision model.

```python
import laya_rocm as laya          # instead of `import laya`

agent = laya.load("convaiinnovations/laya", device="cuda")   # "cuda" is correct on ROCm
agent.predict({"text": "This works surprisingly well."},
              {"s": {"type": "choice", "instructions": "What is the sentiment?",
                     "criteria": {"positive": "positive", "negative": "negative", "neutral": "neutral"}}})
```

Everything `laya` exports is re-exported unchanged. `Agent`, `RLAgent`, `load` and `Router` are
subclasses with the same signatures and the same output format, plus:

| | upstream `laya` on ROCm | `laya_rocm` |
|---|---|---|
| Autocast precision | NVIDIA compute-capability test (`major < 8`), which AMD parts pass only by accident | chosen from the gfx architecture: BF16 on CDNA2+/RDNA3+, FP16 on older parts; `dtype=` to override |
| CPU fallback on GPU errors | only when the message contains "memory" or "cuda", so HIP errors are raised instead | also catches HIP / HSA / ROCm errors |
| torch ≥ 2.13 without a C compiler | first GPU forward fails (`Failed to find C compiler`, from a Triton op in `torch._native`) | switches those Triton ops off, uses the stock kernels, warns once |
| Tokenization | state tokenized once per question, question text on every call | question head cached, state tokenized once per call (token ids identical, tested) |
| Batching requests | one state per call | `predict_many(states, questions)`: one forward pass for many states |
| Backend selection | – | `blas=`, `fa=`, `tunableop=`, `native_triton=` (see `laya_rocm.configure`) |

Results are identical to upstream in FP32 on CPU, and GPU results match a CPU FP32 reference
(see [BENCHMARKS.md](BENCHMARKS.md) for agreement and threshold-flip counts).

## Install

1. Install a ROCm build of PyTorch for your GPU first
   ([AMD's instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)).
   `laya-rocm` deliberately does not pin `torch`, so pip will not replace your ROCm wheel.
2. `pip install laya-rocm` (pulls `laya>=0.3.5,<0.4`).
3. Check the setup: `laya-rocm doctor`. It prints the environment, loads the model on the GPU,
   and fails if the model ends up on the CPU.

### WSL2 (Radeon / Ryzen AI) — optional

On native Linux, follow AMD's install instructions and skip this section; the package needs
nothing else. For WSL2, `scripts/setup_wsl.sh` builds a complete environment under `~/.laya-rocm` **without sudo**: `uv`
Python, [librocdxg](https://github.com/ROCm/librocdxg) (the Windows GPU bridge), libgomp and the
ROCm PyTorch wheel. It needs Windows 11, AMD Adrenalin ≥ 26.2.2 and WSL2.

```bash
bash scripts/setup_wsl.sh gfx1151     # your gfx target
bash scripts/wsl.sh laya-rocm doctor  # run anything inside the env
```

Optional: `sudo apt install gcc` enables PyTorch's Triton-backed native ops and `torch.compile`.
Without it `laya_rocm` falls back to the stock kernels automatically.

## Options

```python
laya.load(model, device="cuda",
          dtype="auto",           # 'auto' | 'bf16' | 'fp16' | 'fp32'
          blas=None,              # 'hipblaslt' | 'rocblas' | 'default'
          fa=None,                # 'aotriton' | 'ck' | 'default'  (flash-attention behind SDPA)
          tunableop=None,         # 'tune' | 'use' | 'off'
          tunableop_file=None,
          native_triton="auto")   # 'auto' (off without a C compiler) | 'on' | 'off'
```

Backend choices are process-wide. `None` leaves PyTorch's default in place. Only use settings
that [BENCHMARKS.md](BENCHMARKS.md) shows winning on your kind of hardware — the wins measured
here are shape- and GPU-specific, which is why none of them is a default.

`tunableop="tune"` writes a CSV of the best kernel per GEMM shape; `"use"` replays it. That file
is tied to the GPU, ROCm and PyTorch versions that produced it, so tune on your own machine (this
repo ships no tuned CSV).

## Scope of testing

Validated on a **Radeon 8060S iGPU (Ryzen AI Max+ 395, gfx1151) under WSL2**, with torch 2.13 +
ROCm 10. Nothing in the package is specific to that machine — precision is chosen per gfx
architecture and all backend settings default to PyTorch's own choices — but other ROCm GPUs are
*expected* to work rather than proven. CI covers tokenization parity, the dtype table and API
compatibility on CPU only. Reports from other hardware are welcome.

Benchmark numbers, and any tuning conclusion drawn from them, come from that one machine. They
are not used to change defaults for other architectures.

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) — **measured on one machine** (Radeon 8060S / WSL2). To reproduce:

```bash
pip install -e ".[bench]"
python bench/run_matrix.py        # every configuration, each in a fresh process
python bench/make_report.py       # -> BENCHMARKS.md, results/plots/
```

## Tests

```bash
pytest -q tests                                  # fast: tokenization parity, dtype table, API
LAYA_RUN_MODEL_TESTS=1 pytest -q -m model tests  # downloads the checkpoint; GPU tests need ROCm
```

## Versioning

`laya-rocm` subclasses upstream internals, so it pins `laya<0.4` and is tested against each laya
minor release before the pin moves.

## License

Apache-2.0, same as Laya. `bench/common.py` adapts metric and dataset code from upstream's
`research/scripts/build_benchmark_nb.py`.
