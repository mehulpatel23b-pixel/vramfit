# vramfit

**Will this model actually fit on this GPU?**

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](requirements.txt)
[![Tests](https://img.shields.io/badge/tests-83%20passing-brightgreen.svg)](tests/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg)](#installation)

`vramfit` answers the question every local-LLM user hits within five minutes of
downloading a model: **can my hardware run this, at what quantisation, at what
speed, and with how much context?**

It is a single zero-dependency Python CLI that models VRAM the way inference
runtimes actually allocate it, then tells you the verdict in one screen.

```
$ vramfit best --model qwen2.5-32b --gpu rtx4090 --context 16384

Qwen2.5 32B  ->  RTX 4090 24GB
quant q3   context 16384   batch 1

  weights         13.13 GB
  kv cache         4.00 GB
  scratch          1.05 GB
  ------------------------
  total           18.18 GB
  usable          21.82 GB
  headroom         3.65 GB

  [###################.....]    83%  FITS
  est. decode    50.1 tok/s
  max context   16384 tokens

  - weights 13.13GB + kv 4.00GB + scratch 1.05GB
  - fits with 3.65GB spare (83% of usable VRAM used)
  - KV cache in f16 dominates. q8 KV would free ~2.0GB.

quantisation sweep (best quality first):
quant   total GB    tok/s   verdict
-----------------------------------
f16        69.98      0.3   NO
q8         36.99      1.0   NO
q6         31.06      1.9   NO
q5         26.68      6.5   PARTIAL
q4         22.56     24.7   PARTIAL
q3         18.18     50.1   FITS
q2         14.57     67.2   FITS
iq2        13.54     74.4   FITS
```

That output is the whole point. It does not just say "no": it shows exactly
which quantisation crosses the line, what the KV cache costs, and what speed
you trade away to get there.

---

## Why this exists

Most VRAM calculators do one multiplication: `params × bits ÷ 8`. That number
is wrong in both directions, and wrong by a lot:

- **It ignores the KV cache**, which at 32K context can be larger than the
  weights themselves and is the real reason people hit OOM.
- **It ignores runtime scratch**, the graph buffers and CUDA context that
  consume 0.5-3 GB before a single token is generated.
- **It treats advertised VRAM as usable VRAM.** A 24 GB card does not give
  you 24 GB; the driver and display hold back 1-2 GB.
- **It says nothing about speed**, so "it fits" and "it is usable" get
  conflated. A model that fits at 2 tok/s does not fit.

`vramfit` models all four, cites every number in the code, and keeps the whole
thing to under 1000 lines of standard-library Python so you can audit the maths
yourself.

---

## Installation

### From source (recommended while the project is young)

```bash
git clone https://github.com/mehulpatel23b-pixel/vramfit.git
cd vramfit
pip install -e .
```

### Without installing anything

The package is importable straight from `src/`:

```bash
git clone https://github.com/mehulpatel23b-pixel/vramfit.git
cd vramfit
PYTHONPATH=src python3 -m vramfit list
```

### Requirements

- Python 3.8 or newer
- **Nothing else.** No numpy, no torch, no requests.

`vramfit` runs on an air-gapped machine, inside a distroless container, or on a
CI runner in well under a second.

---

## Quick start

```bash
# What is the best quant my card can afford for this model?
vramfit best --model llama3-8b --gpu rtx4060ti16

# Exact plan at one quantisation, including headroom and tok/s
vramfit plan --model llama3-8b --gpu rtx4090 --quant q5

# I want to run 70B. What is the cheapest card that does it?
vramfit gpu --model llama3-70b --quant q4

# Should I buy a 3090 or a 4090 for this workload?
vramfit compare --model qwen2.5-32b --gpus 3090,4090 --context 32768

# Every quantisation side by side
vramfit sweep --model mistral-7b --gpu rtx4070

# Machine-readable, for scripts and CI gates
vramfit plan --model qwen2.5-7b --gpu rtx3090 --json
```

Models can be a preset name or a raw parameter count. `--model 13b` works for
anything without a preset.

```bash
vramfit plan --model 13b --gpu rtx4070 --quant q4 --context 16384
```

---

## What the verdicts mean

| Verdict | Meaning | What to do |
|---|---|---|
| `comfortable` | Under 85% of usable VRAM | Run it. Room for longer context. |
| `tight` | 85-100% of usable VRAM | It runs, but a browser tab or a longer prompt can OOM you. Drop a quant or halve the context. |
| `PARTIAL` | Weights overflow; some layers spill to CPU | Works, but decode speed collapses. Only acceptable for batch jobs. |
| `NO` | Does not fit at any useful speed | Smaller model, lower quant, or bigger card. |

Exit codes follow the verdict, so you can gate on it:

- `0` the model fits
- `1` the model does not fit
- `2` bad arguments (unknown GPU, unknown quant, unparseable model)

```bash
vramfit plan --model llama3-70b --gpu rtx3070 --json > plan.json || echo "does not fit"
```

---

## Architecture

Five modules, one direction of dependency, no cycles. Each layer is testable
on its own.

```
src/vramfit/
├── model.py       ModelSpec, parameter parsing, quant table, KV-cache maths
├── hardware.py    GPU catalogue: VRAM, bandwidth, per-device overhead
├── estimator.py   The core: weights + KV + scratch -> verdict, tok/s, max ctx
├── planner.py     Policy: sweep quants, pick the best fit, suggest devices
├── report.py      Rendering: tables and bars, no colour dependency
└── cli.py         Argument parsing, exit codes, JSON output
```

### The memory model

Total footprint is the sum of three allocations:

```
total = weights + kv_cache + activation_scratch
```

**1. Weights**

```
weights_bytes = params × bits_per_weight(quant) ÷ 8
```

`bits_per_weight` is the *effective* rate, not the nominal one. `q4_k_m` is
4.5 bits per weight, not 4, because K-quants keep some tensors at higher
precision. Using 4.0 makes the estimate optimistically wrong by ~12%, which is
exactly the error that causes an OOM at the last layer.

**2. KV cache**

```
per_token = n_layers × n_kv_heads × head_dim × 2 (K and V) × bytes_per_element
kv_bytes  = per_token × context × batch
```

This is why grouped-query attention matters: Llama 3 8B has 32 attention heads
but only 8 KV heads, so its cache is 4× smaller than a naive count predicts.
`vramfit` uses real KV-head counts for every preset. For unknown models it
infers the shape from `params ≈ 12 × layers × hidden²`, which is accurate to
within a layer or two for standard transformers.

**3. Activation scratch**

Graph buffers, the CUDA context and allocator fragmentation. Modelled as 8% of
weight bytes with a 0.5 GB floor, scaling with `sqrt(batch)`. This is the term
every calculator forgets and every runtime pays.

**4. Usable VRAM**

```
usable = (vram − overhead) × (1 − safety_margin)
```

Overhead is per-device and measured: ~1.5 GB on discrete NVIDIA cards, 6-16 GB
on Apple unified memory (where the OS reserves much more), and it is why a
24 GB 4090 reports ~21.8 GB usable here rather than 24.

### The speed model

Decode is memory-bandwidth-bound: generating one token requires reading the
entire set of active weights once. So the ceiling is:

```
tokens_per_second ≈ effective_bandwidth ÷ bytes_read_per_token
```

`effective_bandwidth` is 70% of theoretical peak, which is what llama.cpp
actually achieves on consumer cards. Partial offload is penalised by adding
PCIe transfer time for the CPU-resident layers, with an extra factor for the
serialisation cost. Unified-memory devices (Apple silicon, CPU) get an 0.85
multiplier for shared-memory contention.

**Calibration:** Llama 3 8B at Q4 on an RTX 4090 is measured at ~158 tok/s in
llama.cpp benchmarks. `vramfit` returns 156.2. The `test_matches_published_llamacpp_benchmark`
test pins this so a future refactor cannot silently drift.

### Why "best quant" is a policy, not a formula

The highest-quality quantisation that fits is what you want, but "fits" has
three tiers. `planner.recommend()` walks `f16 → q8 → q6 → q5 → q4 → q3 → q2 → iq2`
and takes the first *comfortable* fit, falling back to *tight*, then to the best
partial offload. Nothing below `q3` passes silently: it comes with an explicit
quality warning, because q2 and iq2 visibly damage reasoning and code output.

---

## Real-world use cases

### 1. Buying a GPU without guessing

The most expensive mistake in local AI is buying the wrong card. Feed in your
model and target context, get the cheapest card that actually holds it.

```bash
vramfit gpu --model llama3-70b --quant q4
```

```
device                         usable    total    tok/s   verdict
-----------------------------------------------------------------
RTX A6000 48GB                   45.1    42.44     13.5   TIGHT
L40S 48GB                        45.1    42.44     15.2   TIGHT
A100 80GB                        76.1    42.44     35.9   FITS
```

Note what this reveals: two 48 GB cards are *tight* for 70B at q4 while an
80 GB card has real headroom. A params-only calculator would have called all
three comfortable and you would have learned the truth after the invoice.

### 2. Deciding whether to download a 40 GB file

Model downloads are slow and disk is finite. Check first.

```bash
vramfit best --model mixtral-8x7b --gpu m4max128
```

On unified-memory Macs this is the difference between a working setup and an
afternoon of swapping.

### 3. Right-sizing a self-hosted inference server

Before you provision, confirm the model fits with the batch size and context
your API actually serves.

```bash
vramfit plan --model qwen2.5-32b --gpu a100-80 --context 32768 --batch 8 --json
```

### 4. CI gate for model upgrades

Treat "does the new model still fit our inference box?" as a test, not a
post-deploy surprise.

```yaml
- name: Model fits the deployment GPU
  run: vramfit plan --model qwen2.5-32b --gpu l40s --quant q4 --context 16384
```

Exit code `1` fails the build.

### 5. Choosing KV-cache quantisation

The KV cache is often the difference between 8K and 32K context. Check the
trade before committing:

```bash
vramfit plan --model llama3-8b --gpu rtx4070 --context 32768 --kv-quant q8 --json
```

### 6. Understanding *why* a model is slow

If a model fits but crawls, the `gpu_layers_pct` field tells you whether it is
fully resident or spilling to system RAM. Below 100%, the speed number explains
the collapse.

### 7. Scripting capacity planning

```python
from vramfit import resolve_model, find_gpu, PlanOptions, estimate

plan = estimate(
    resolve_model("llama3-8b"),
    find_gpu("rtx4090"),
    PlanOptions(quant="q4", context=8192),
)
if not plan.fits:
    raise SystemExit(f"won't fit: {plan.total_gb:.1f} GB needed")
print(f"{plan.tokens_per_second:.0f} tok/s, {plan.max_context} ctx")
```

---

## Usage reference

```
vramfit plan    --model M --gpu G     one model, one device
vramfit best    --model M --gpu G     highest-quality quant that fits
vramfit gpu     --model M             cheapest devices that can host it
vramfit compare --model M --gpus A,B  several devices side by side
vramfit sweep   --model M --gpu G     every quantisation for one device
vramfit list                          known GPUs and model presets
```

Shared options:

| Flag | Default | Description |
|---|---|---|
| `--model` | required | preset name (`llama3-8b`) or param count (`13b`) |
| `--gpu` | required for most | device key (`rtx4090`, `5090`, `m4max128`) |
| `--quant` | `q4` | `f16 f32 bf16 q8 q6 q5 q4 q3 q2 iq2` |
| `--context` | `8192` | context window in tokens |
| `--batch` | `1` | concurrent sequences |
| `--kv-quant` | `f16` | KV cache dtype, `f16` or `q8` |
| `--json` | off | machine-readable output |

Quant names accept the spellings people actually use: `q4_k_m`, `Q4-K-M`,
`int4` and `gguf-q4` all resolve to `q4`.

---

## Project structure

```
vramfit/
├── src/vramfit/
│   ├── __init__.py        public API surface
│   ├── __main__.py        python -m vramfit
│   ├── model.py           ModelSpec, quant table, KV-cache maths
│   ├── hardware.py        GPU catalogue with real overheads
│   ├── estimator.py       the memory + speed model
│   ├── planner.py         quant sweep and recommendation policy
│   ├── report.py          terminal rendering
│   └── cli.py             argument parsing and exit codes
├── tests/
│   ├── test_model.py      parsing, quantisation, KV-cache identities
│   ├── test_estimator.py  verdicts, monotonicity, benchmark calibration
│   ├── test_planner.py    sweep ordering, recommendation policy
│   └── test_cli.py        exit codes, JSON, error paths
├── examples/
│   └── plan_setup.py      runnable Python API walkthrough
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Testing

```bash
# standard library only
PYTHONPATH=src python3 -m unittest discover -s tests -v

# or with pytest
pip install pytest && pytest
```

83 tests. Beyond the obvious happy paths they pin the invariants that make the
tool trustworthy:

- **KV-cache identity** — Llama 3 8B at 8192 context must equal exactly 1 GiB.
  This catches any error in the head/layer/dtype chain.
- **Monotonicity** — lower quant must never use more memory; longer context and
  bigger batch must never use less.
- **Benchmark calibration** — 8B Q4 on a 4090 must land within 100-200 tok/s of
  the measured 158.
- **Arithmetic consistency** — `total` must equal the sum of its parts, and
  `headroom` must equal `usable − total`.
- **No-fit explains itself** — every failing plan must carry a human-readable
  reason.

---

## Accuracy and limits

`vramfit` is a planning tool, not a profiler. It gets you to the right decision
in under a second; it will not match `nvidia-smi` to the megabyte.

- **Speed estimates are ±20%.** Real throughput depends on your CUDA version,
  driver, kernel selection, prompt batching and the phase of the moon. Use them
  to compare options, not to quote SLAs.
- **Prefill is not modelled.** Prompt processing is compute-bound and far
  faster than decode; this tool reports decode throughput only.
- **Flash attention and paged KV** can let runtimes exceed the naive KV bound.
  `vramfit` deliberately stays conservative so its "fits" verdicts hold under a
  plain llama.cpp or vLLM default configuration.
- **MoE models** are handled explicitly: total parameters drive the VRAM
  estimate (all experts must be resident) while active parameters drive the
  speed estimate (only routed experts are read per token). Presets for Mixtral
  and DeepSeek V3 carry real active-parameter counts; pass `active_params=` on a
  custom `ModelSpec` for any other mixture-of-experts model.
- **Unknown architectures** fall back to inferred shape, accurate to roughly a
  layer or two. Pass a `ModelSpec` with explicit `n_layers`/`n_kv_heads`/
  `head_dim` when precision matters.

---

## Contributing

Adding a GPU is one dictionary entry in `hardware.py`: name, VRAM, memory
bandwidth, and the driver overhead you have actually observed. Adding a model
preset is one entry in `model.py`. Both come with tests; run them before
opening a PR.

If you have real benchmark numbers that disagree with the estimator, that is a
bug report worth filing. Include the model, quant, card, runtime and measured
tok/s so the calibration can be corrected.

---

## License

MIT. See [LICENSE](LICENSE).

Not affiliated with any inference runtime. `vramfit` plans; llama.cpp, vLLM,
Ollama and friends execute.
