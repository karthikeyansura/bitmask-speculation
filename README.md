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

## Install

Python 3.12 or later. From the checkout:

```bash
uv sync --locked --extra models --extra grammar --extra benchmarks
source .venv/bin/activate
bitmask-speculation generate --help
```

Add `--extra triton` on a Linux NVIDIA host for Triton. CUDA extension builds also
need a compatible CUDA toolkit, C++ compiler and Ninja. See the
[development guide](docs/development.md) for installed-wheel and GPU checks.
The base package imports without model dependencies or a CUDA compiler.

## Generate

```bash
bitmask-speculation generate 'Return a colour:' \
  --target /path/to/target --draft /path/to/draft --local-files-only \
  --mode c3 --schema-json '{"enum":["red","blue"]}' --max-new-tokens 32
```

For Hub models, supply full `--target-revision` and `--draft-revision` SHAs.
Target and draft tokenizer definitions must match. Use `--device cuda:0
--dtype bfloat16 --backend triton` for constrained Triton selection on a supported
GPU. CPU/float32 with `reference` selection is the default.

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

The model adapter evaluates full prefixes with `use_cache=False`. Generation is
single-request, single-device and greedy. Batched reduction primitives do not
batch the request loop. The standalone sampler is not a stochastic speculative
decoder. See [kernel contracts](docs/kernels.md) for shapes, limits and stream rules.

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
and checked wrapper latency. Decode trials include full-prefix model work,
verification, masks and scheduling; setup and output validation are excluded.
These comparisons describe the uncached implementation, not a cached serving
engine. `--chat-template` renders a user turn with `enable_thinking=False`.

Each run uses a new output directory and retains source/model identities, hardware,
configuration, independent references and raw trials. A speedup is emitted only
when both sides complete, exactly match their target references and validate
against the full schema in every repetition. Errors and truncated outputs remain
visible. No aggregate speedup across unlike schemas is reported.

[Benchmark methodology](docs/benchmarks.md) defines the timing boundaries,
repetitions, schema regimes and aggregation rules.

## Test

```bash
uv run --locked --extra models --extra grammar --extra benchmarks pytest -q
```

CPU hosts skip CUDA execution checks. Device-guard tests require two visible
CUDA devices in one process. [Development](docs/development.md) includes native
builds and sanitizer commands.
