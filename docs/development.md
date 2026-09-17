# Development

Use Python 3.12 for the development environment. The distribution is named
`bitmask-speculation`; Python imports use `bitmask_speculation` because a hyphen
cannot appear in a Python identifier.

## Local installation

From the repository root, with uv installed:

```bash
uv sync --locked
uv run --locked python -c "import bitmask_speculation as bs; print(bs.__version__)"
```

The lockfile records the Python development dependencies. `uv sync` creates an
editable installation in `.venv` and installs the development group by default.
It leaves the system Python environment alone. The token-level decoding API
also works with a caller-provided model without installing a framework.

The base package has no runtime dependencies. Optional dependencies are
installed through the extras listed in `pyproject.toml`.

## Reference tests

```bash
uv run --locked pytest tests/unit
uv sync --locked --extra models
uv run --locked --extra models pytest
```

The `models` extra installs PyTorch and Transformers. Integration tests create
small random models on the CPU, without downloading pretrained weights. They
compare token sequences with Transformers greedy generation and check adapter
failure cases. Without the extra, integration tests are skipped; a base-only
run does not establish model compatibility.

## Grammar integration tests

```bash
uv sync --locked --extra models --extra grammar
uv run --locked --extra models --extra grammar pytest
```

The `grammar` extra pins XGrammar 0.2.8. Tests compile small EBNF grammars and a
JSON enum schema against explicit vocabularies; no tokenizer or model download
is required. They cover state ownership, stop tokens, signed mask words and
partial final words.

## Request-loop tests

The same test command covers complete speculative requests. Framework-independent
fixtures compare against greedy decoding across proposal budgets, output limits,
and EOS placements. Tiny CPU model pairs repeat the comparison with independent
grammar states and both XGrammar stop-token modes. Accounting tests compare reported totals with
independently observed proposal and consumed-token sequences, including EOS and
grammar termination.

## Baseline workload checks

```bash
uv sync --locked --extra models --extra grammar --extra benchmarks
uv run --locked --extra models --extra grammar --extra benchmarks pytest
```

The `benchmarks` extra adds JSON Schema validation without adding a dependency
to the base decoding package. Tests check workload loading, record round-trips,
weighted aggregation and tiny CPU model integrations. Synthetic durations in
unit tests exercise arithmetic only; integration records are untimed.

See [baseline methodology](baseline-workloads.md) for record fields, eligibility
checks and timing boundaries. Workload data ships in the source distribution;
the installed wheel contains the record API and accepts an explicit manifest
path. It never searches the current directory for a benchmark checkout.

## Package boundaries

```text
pyproject.toml                 package metadata and build selection
uv.lock                       resolved development and optional dependencies
.python-version               development interpreter selection
src/bitmask_speculation/       installed Python package
docs/                         design and development documentation
tests/unit/                   framework-independent decoding tests
tests/integration/            optional model adapter tests
benchmarks/baseline.json       versioned baseline cases
benchmarks/schemas/            self-contained JSON schemas
```

The `src` layout keeps imports separate from the working directory. Install the
package before running code that imports it; do not modify `sys.path` in scripts
to make imports succeed.

The wheel contains only the selected package and its distribution metadata.
The source distribution explicitly includes that source, documentation, license,
project configuration, and the unit and integration tests. Unrelated files in the checkout are not automatically
included. Extend the selection when a new component needs additional build inputs.

Native kernel compilation is separate from installing the base package. GPU
backends must be opt-in and must not initialize CUDA during package import.
Backend build and compatibility checks accompany their implementation.

## Build and verify a distribution

```bash
uv run --locked python -m build
```

This builds a source distribution and then builds a wheel from it, checking that
the source distribution carries the inputs needed for a clean build. Build
backends run in isolated environments; their requirements are declared in
`pyproject.toml` separately from the development lockfile.

Validate the wheel from outside the checkout with a fresh environment:

```bash
uv venv --python 3.12 /tmp/bitmask-wheel-check
uv pip install --python /tmp/bitmask-wheel-check/bin/python \
  dist/bitmask_speculation-0.1.0.dev0-py3-none-any.whl
cd /tmp
/tmp/bitmask-wheel-check/bin/python -I -c \
  "import bitmask_speculation as bs; print(bs.__version__, bs.__file__)"
```

Use a fresh directory for each check and the wheel filename produced by the
current build. The import path must resolve inside the verification environment,
not the source checkout. Also inspect archive contents, license metadata, project
links, and runtime requirements before publishing a distribution.

## GPU work

Develop and check packaging locally. Run CUDA compilation, kernel correctness,
and inference benchmarks on Linux with an NVIDIA GPU. Record the GPU model,
driver, toolkit, Python, and library versions with each run. An instance type or
a successful SSH login alone does not establish a validated GPU environment.

## Budget comparisons

The [budget-sweep API](budget-sweeps.md) works with already loaded model adapters.
It requires the `models`, `grammar` and `benchmarks` extras for the XGrammar path.
CPU fixture tests exercise repeated comparisons without downloading weights.
Use the same installed package, pinned models, precision and schema for GPU
comparisons; retain actual run times and explicit device synchronization.

## CUDA validation

See [native primitives](cuda-primitives.md) and the [GPU validation procedure](cuda-validation.md).
The default suite collects `tests/gpu` and reports explicit skips on CPU hosts.
Do not count those skips as CUDA validation. Package installation requires no
CUDA compiler; building the optional extension does. Validate an installed wheel
outside the source checkout before enabling a generation backend.
