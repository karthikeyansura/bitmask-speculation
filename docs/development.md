# Development

Python 3.12+. The distribution is `bitmask-speculation` and the import is
`bitmask_speculation`. Sources live in `src/`, CUDA sources in `csrc/`, workload
manifests and recorded results in `benchmarks/`, tests in `tests/`.

## Install and test

```bash
uv sync --locked --extra models --extra grammar --extra benchmarks
uv run --locked pytest -q
```

The base package has no runtime dependencies. Extras add PyTorch and Transformers
(`models`), XGrammar (`grammar`), JSON Schema validation (`benchmarks`) and Triton
(`triton`, Linux NVIDIA only). Tests build tiny models locally and download no
weights; GPU tests skip without CUDA, and a failed CUDA build on a GPU host fails
the run.

<a id="cuda-tests"></a>

## GPU tests

On a Linux host with CUDA-enabled PyTorch, `nvcc`, a C++ compiler and Ninja:

```bash
uv sync --locked --extra models --extra grammar --extra benchmarks --extra triton
source .venv/bin/activate
export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2   # architecture of the device under test
python -m pytest -ra tests
python examples/validate_gpu.py
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -m pytest -q tests/gpu/test_argmax.py tests/gpu/test_cuda_batch.py tests/gpu/test_random_masks.py
```

The GPU suite covers signed words and tail padding, ties and infinities across
lanes, warps and blocks, NaN and empty-support errors, batched rows with
independent status, seeded sampling and its CDF boundaries, output ownership and
non-default streams. Device-guard and workspace-isolation tests need two GPUs
visible to one process.

## Distribution

```bash
uv run --locked python -m build
uv venv --python 3.12 /tmp/wheel-check
uv pip install --python /tmp/wheel-check/bin/python dist/bitmask_speculation-*.whl
(cd /tmp && /tmp/wheel-check/bin/python -I -c "import bitmask_speculation as bs; print(bs.__version__)")
```

The sdist builds first and the wheel is built from it. Wheels carry the Python
package and the CUDA sources; the extension compiles on first use. Run installed
tests from outside the checkout so imports resolve to site-packages.
