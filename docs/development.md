# Development

Python 3.12 or later. The distribution name is `bitmask-speculation`; the Python
import is `bitmask_speculation`. Source lives under `src/`, native build inputs
under `csrc/`, workload data under `benchmarks/`, and tests under `tests/`.

## Installation and CPU tests

```bash
uv sync --locked --extra models --extra grammar --extra benchmarks
uv run --locked --extra models --extra grammar --extra benchmarks pytest -q
```

`uv sync` installs the package in editable mode and includes the development
requirements. The base package has no runtime dependencies. Extras select
PyTorch/Transformers (`models`), XGrammar (`grammar`), JSON Schema validation
(`benchmarks`) and the Linux NVIDIA Triton backend (`triton`). Tests use locally
constructed models; they do not download pretrained weights. GPU tests skip on
hosts without CUDA. A build failure on a CUDA host is a test failure.

## Distribution checks

```bash
uv run --locked python -m build
uv venv --python 3.12 /tmp/bitmask-wheel-check
uv pip install --python /tmp/bitmask-wheel-check/bin/python \
  dist/bitmask_speculation-0.1.0.dev0-py3-none-any.whl
cd /tmp
/tmp/bitmask-wheel-check/bin/python -I -c \
  "import bitmask_speculation as bs; print(bs.__version__, bs.__file__)"
```

Use a fresh verification directory. Build first produces an sdist, then builds
the wheel from that sdist. Wheels include Python code and native source files
under `kernels/_src`; the optional extension compiles when explicitly loaded.
The source archive also contains guides, tests, examples and workload manifests.
Run the installed wheel's tests from a separate source archive so imports resolve
to site-packages. The base import and CLI help require no optional dependencies.

<a id="cuda-tests"></a>

## CUDA tests

Use Linux, CUDA-enabled PyTorch, a compatible CUDA toolkit, a C++ compiler and
Ninja. Add the `triton` extra for Triton checks. Activate the environment so Ninja
and Python resolve from the same environment. Set the architecture to the device
being tested; `8.6` below is for A10G.

```bash
source .venv/bin/activate
nvcc --version
ninja --version
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
python -c 'from bitmask_speculation.kernels._extension import load_extension; print(load_extension().runtime_version())'
python -m pytest -ra tests/unit tests/integration tests/gpu
python examples/validate_gpu.py
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q tests/gpu/test_popcount.py tests/gpu/test_argmax.py tests/gpu/test_random_masks.py
```

The suite covers signed mask words and tail bits, ties and infinities, invalid
metadata, NaNs, empty support, seeded sampling, output ownership and non-default
streams. Two-device guard and workspace-isolation tests need two CUDA devices
visible in one process; two separate servers cannot execute those checks.

## Guides

- [Decoding and grammar state](design.md)
- [Generation APIs](usage.md)
- [CUDA and Triton kernels](kernels.md)
- [Benchmark methodology](benchmarks.md)
- [Kernel profiling](profiling.md)
