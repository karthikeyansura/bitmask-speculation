<div align="center">
<img src="assets/banner.svg" alt="bitmask-speculation" height="100"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-custom%20kernels-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-fused%20kernel-orange.svg)](https://triton-lang.org/)
</div>

# bitmask-speculation

Grammar-constrained decoding masks every token a schema forbids; speculative
decoding lets a small draft model propose tokens that the target verifies.
Combined naively, the draft ignores the grammar, so the target rejects proposals
it would have masked out, and every masked step adds a vocabulary-wide reduction.

C3 applies the target's grammar to the draft as well, and ordered prefix
verification keeps output identical to constrained target-greedy decoding. C4
also sets each round's draft budget from the mask density of the committed
grammar state. Packed 32-bit token masks feed CUDA and Triton kernels for exact
valid-token counts and deterministic masked argmax.

## GPU kernels

Triton fuses the packed-mask count with masked argmax and reduces independent
rows in one batched launch. On one A10G, for six rows of vocabulary 151,936 with
about 1% valid bits, median microseconds per batch:

| Backend | Graph replay | Async wrapper |
|---|---:|---:|
| `torch` | 257.55 | 422.86 |
| `cuda`, per row | 1517.59 | 1530.46 |
| `triton`, batched | 6.21 | 116.44 |

Graph replay excludes Python dispatch and allocation; the asynchronous wrapper
includes them. The per-row CUDA path is slower than PyTorch on this workload.
Kernel latency is not decoding throughput. With Qwen3-4B / Qwen3-0.6B in BF16,
Triton selection and full-prefix drafting, C3 and C4 ranged from 0.62× to 1.23×
the constrained target across six schema regimes: enum-heavy and nested-record
schemas were faster, the other four slower. See the
[per-schema results](docs/benchmarks.md#a10g-measurements).

## GPU quickstart

Python 3.12 or later on a Linux NVIDIA host. From the checkout:

```bash
uv sync --locked --extra models --extra grammar --extra benchmarks --extra triton
source .venv/bin/activate
bitmask-speculation generate 'Return a colour:' \
  --target /path/to/target --draft /path/to/draft --local-files-only \
  --mode c4 --schema-json '{"enum":["red","blue"]}' --max-new-tokens 32
```

`--device`, `--dtype` and `--backend` default to `auto`: the current CUDA
device, bfloat16 (float16 below compute capability 8.0) and Triton, falling back
to the CUDA extension and then the reference backend. Without CUDA, `auto`
selects CPU, float32 and reference. Explicit flags override each choice, and the
JSON output records the resolved values. The CUDA backend compiles its extension
on first use and needs a CUDA toolkit, C++ compiler and Ninja.

For Hub models, supply full `--target-revision` and `--draft-revision` SHAs.
Target and draft tokenizer definitions must match.

| Mode | Draft selection | Target selection | Budget |
|---|---|---|---|
| `target_unconstrained` | No draft | Unconstrained | — |
| `target_constrained` | No draft | Grammar masked | — |
| `c1` | Unconstrained | Unconstrained | Fixed |
| `c2` | Unconstrained | Grammar masked | Fixed |
| `c3` | Grammar masked | Grammar masked | Fixed |
| `c4` | Grammar masked | Grammar masked | Mask density |

The target accepts only the consecutive matching prefix. A mismatch supplies a
target correction; an all-accepted round may supply a bonus token. EOS and the
remaining output budget bound both cases. Draft grammar state is forked from
committed state and discarded after verification.

The [usage guide](docs/usage.md) covers the Python APIs, CLI and termination
semantics. [Worked traces](docs/design.md#prefix-acceptance) show prefix rejection,
bonus-token alignment and proposal accounting.

## Backend contracts

| Capability | Reference | CUDA | Triton |
|---|---|---|---|
| Constrained greedy selection | CPU or CUDA PyTorch | Linux NVIDIA | Linux NVIDIA, compute capability ≥8.0 |
| Selection score dtype | Floating PyTorch dtype | float32 primitive | float32 primitive |
| Packed count | CPU signed-word reference | One row | Fused with argmax |
| Independent reduction rows | — | Per-row calls | Contiguous `[B,V]` |
| Scratch reuse | — | — | Device- and stream-bound workspace |
| Standalone masked sampling | CPU comparison function | Explicit CUDA generator | — |

Engine dispatch widens float16/bfloat16 scores to float32 for native selection;
native float64 selection is rejected. Grammar masks originate on the CPU and are
copied to the model device. Checked selection reads status before advancing state.
C4 density queries still use the CPU count reference. Returned kernel outputs own
their storage; workspace reuse applies only to intermediate reductions.

Drafting caches the committed prefix once per round, then evaluates one token
at a time. The cache is released before target verification. Use `--no-draft-cache`
for uncached comparisons. Target evaluation remains full-prefix with
`use_cache=False`. Generation is single-request, single-device and greedy.
Batched reduction primitives do not batch the request loop. The standalone
sampler is not a stochastic speculative decoder. See
[kernel contracts](docs/kernels.md) for shapes, limits and stream rules.

## Measure

```bash
python -m bitmask_speculation.benchmarks reductions \
  --backend triton --rows 6 --vocab 151936 --density .01 \
  --output /tmp/bitmask-reductions

python -m bitmask_speculation.benchmarks decode \
  --target /path/to/target --draft /path/to/draft --local-files-only \
  --workloads benchmarks/regimes.json --device cuda:0 --dtype bfloat16 \
  --backend triton --chat-template --budgets 1 2 4 8 --repetitions 5 \
  --output /tmp/bitmask-decoding

python -m bitmask_speculation.benchmarks report /tmp/bitmask-decoding
```

Reduction measurements separate graph-device replay, asynchronous wrapper latency
and checked wrapper latency. Decode trials include target forward passes,
drafting, verification, masks and scheduling; setup and output validation are
excluded. Reports record the draft cache mode. The published A10G trials used
uncached drafting; reproduce that mode with `--no-draft-cache`. They do not measure
cached drafting or a cached serving engine. `--chat-template` renders a user turn
with `enable_thinking=False`.

Each run uses a new output directory and retains source/model identities, hardware,
configuration, independent references and raw trials. A speedup is emitted only
when both sides complete, exactly match their target references and validate
against the full schema in every repetition. Errors and truncated outputs remain
visible. No aggregate speedup across unlike schemas is reported.

[Benchmark methodology](docs/benchmarks.md) defines the timing boundaries,
repetitions, schema regimes and aggregation rules.

[Profiling](docs/profiling.md) provides correctness-gated Nsight Systems and
Nsight Compute entry points for the installed reduction backends.

[A10G measurements](docs/benchmarks.md#a10g-measurements) include raw reduction
samples and per-schema decoding comparisons, including regressions.

## Development

The [development guide](docs/development.md) covers CPU-only installs, the test
suite, installed-wheel checks, native builds and sanitizers.
