<div align="center">
<img src="assets/banner.svg" alt="bitmask-speculation" height="100"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-custom%20kernels-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-fused%20kernel-orange.svg)](https://triton-lang.org/)
</div>

# bitmask-speculation

Packed grammar masks for draft scheduling and token verification in structured
LLM decoding. The project combines grammar-constrained draft proposals, ordered
prefix verification and density-based draft budgets, with standalone CUDA
primitives for popcount, masked argmax and categorical sampling.

A narrow grammar can reduce invalid proposals without making the draft agree
with the target. Mask density is a scheduling signal, not an acceptance guarantee;
its overhead and effect on throughput must be measured.

## Use

Python 3.12 or later. From this checkout:

```bash
uv sync --extra models --extra grammar
source .venv/bin/activate
bitmask-speculation generate --help
```

Generate from locally saved model/tokenizer directories:

```bash
bitmask-speculation generate 'Return a colour:' \
  --target /path/to/target --draft /path/to/draft \
  --mode c3 --schema-json '{"enum":["red","blue"]}' \
  --max-new-tokens 32 --local-files-only
```

For Hub models, provide full revision SHAs with `--target-revision` and
`--draft-revision`. Target and draft tokenizer definitions must match. Choose
`--device cuda --dtype bfloat16` on a compatible GPU host; CPU/float32 is the
default. Model loading never starts cloud resources.

The [Python API](docs/engine.md) reuses loaded models across requests. The
[CLI](docs/cli.md) returns JSON with content tokens, stop reason, accounting and
scheduling decisions. See the [runnable example](examples/quickstart.py).

## Decoding paths

| Mode | Target grammar | Draft grammar | Draft budget |
| --- | --- | --- | --- |
| `target_unconstrained` | No | No draft | None |
| `target_constrained` | Yes | No draft | None |
| `c1` | No | No | Fixed |
| `c2` | Yes | No | Fixed |
| `c3` | Yes | Yes | Fixed |
| `c4` | Yes | Yes | Mask-density policy |

Verification accepts only the consecutive matching prefix. On the first mismatch,
the target supplies a correction and discards the remaining proposals. An
all-accepted round can supply a target bonus token, subject to EOS and output
budget. Draft grammar state is isolated from the committed target state.

The [prefix walkthrough](docs/prefix-acceptance.md) explains alignment and state
transitions. [Proposal accounting](docs/proposal-accounting.md) distinguishes
proposed, verified, accepted and emitted tokens. C4 records each density query and
requested/effective budget so its decisions can be inspected.

## Native primitives

The optional CUDA extension builds explicitly from packaged sources. It provides
vocabulary-tail popcount, deterministic masked argmax and standalone masked
sampling for contiguous single rows. Python and native boundaries check inputs;
outputs have independent ownership and launches use the current CUDA stream.

Generation uses the full-prefix reference path. See [native contracts](docs/cuda-primitives.md)
and [CUDA validation](docs/cuda-validation.md).

## Evaluation

Speculative output must match target-only greedy output under the same grammar.
Schema validity, termination and exact token agreement are separate checks.
The [baseline workloads](docs/baseline-workloads.md) and
[budget sweep](docs/budget-sweeps.md) retain individual trials, failures and C4
traces across enum, nested and mixed schemas.

Correctness results do not establish a speedup. Kernel, wrapper and end-to-end
measurements need distinct timing boundaries, warmup and synchronization.
See [validation records](docs/validation.md) for evidence requirements.

## Development

```bash
uv sync --extra models --extra grammar --extra benchmarks
uv run pytest -q
```

CPU hosts skip GPU tests explicitly. A single GPU cannot execute the two-device
isolation check. The generation API is greedy, single-request and uncached; the
sampling primitive does not establish stochastic speculative decoding.

- [Documentation index](docs/index.md)
- [Design and decoding contracts](docs/design.md)
- [Development and installed-wheel checks](docs/development.md)
