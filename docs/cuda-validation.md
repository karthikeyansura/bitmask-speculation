# CUDA validation procedure

Run the installed-wheel GPU suite in a CUDA-enabled environment.

## CUDA environment

Use a compatible CUDA-enabled PyTorch environment, CUDA toolkit with `nvcc`, a
host compiler and Ninja. Record exact versions, GPU model, compute capability,
source revision, architecture flags and actual run time. Primitive tests do not load model weights.

Activate the test environment so both Python and Ninja resolve from its `bin`
directory. Calling its Python by absolute path alone does not put Ninja on `PATH`.

```bash
source .venv/bin/activate
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
nvcc --version
ninja --version
```

For an A10G, the architecture flag is `8.6`; choose the actual device capability
for other GPUs. Compile and run with a bounded number of compiler workers:

```bash
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
python -c 'from bitmask_speculation.kernels._extension import load_extension; print(load_extension().runtime_version())'
python -m pytest -q tests/unit tests/integration tests/gpu
```

Install the built wheel first and run tests from a separate source/test directory
so imports resolve to site-packages. The native sources must come from the wheel's
`kernels/_src`, not a checkout fallback. Keep the full test log, including skips.
A single-GPU host skips the two-device guard test; report that limitation.

## Required checks

- Signed mask words, sign-bit-only support, dirty padding tails, random masks and
  vocabulary sizes around word/block boundaries.
- Empty support and NaN rejection; lowest-ID ties across lanes/warps; valid
  negative infinity; read-only inputs and independently owned retained outputs.
- Native metadata rejection as well as Python validation, including dtype,
  layout, device, shape and word-count errors.
- Current-stream execution with non-default streams and repeated calls. Caller
  cross-stream producer/consumer ordering remains explicit.
- Sampling temperature-zero parity, seeded replay, generator advancement,
  explicit-uniform CDF comparisons, zero mass, extreme temperatures and an
  empirical categorical distribution check.
- Clean CPU-only collection with explicit GPU skips. No performance assertions
  are embedded in correctness tests.

Where Compute Sanitizer is available, run a focused memory check after the normal
suite; it is separate evidence from ordinary test success:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest -q tests/gpu/test_popcount.py tests/gpu/test_argmax.py tests/gpu/test_random_masks.py
```

The checked argmax and sampling wrappers read a status scalar and synchronize.
They are not CUDA-graph capture interfaces. Kernel-only timings must use a
separately defined measurement boundary and validate status outside that boundary.
Do not compare these checked wrappers with an unchecked baseline without stating
the difference.
