# Native CUDA primitives

The extension is optional and builds only when `load_extension()` is explicitly
called, or a native primitive first dispatches. Importing the package does not
compile code or select a fallback. Base installation needs no CUDA compiler.

Install the `models` extra on a Linux NVIDIA host with CUDA-enabled PyTorch,
a compatible CUDA toolkit (`nvcc`), a C++ compiler and Ninja. Set
`TORCH_CUDA_ARCH_LIST` explicitly for reproducible builds and `MAX_JOBS` to bound
compiler memory use. Compiler failures propagate rather than falling back to CPU.
The module cache key includes sources, headers, PyTorch/CUDA versions and the
requested architecture list. PyTorch manages the local build cache.

```python
from bitmask_speculation.kernels._extension import load_extension
native = load_extension()
print(native.runtime_version())
```

Development sources live in `csrc/`. Wheels include the same source bytes under
`bitmask_speculation/kernels/_src/`; source archives retain `csrc/`. Installation
outside the checkout therefore retains the sources needed for explicit builds.
The extension is not a precompiled binary distribution.

## Tensor contract

Primitives use a single contiguous row on one CUDA device: masks are signed int32
with exactly `ceil(V/32)` words; logits are float32 with V entries. V must be
positive and fit an int32 token index. Batched rows, strided views, implicit dtype
casts and implicit CPU/device transfers are unsupported. Python validates before
loading; native entry points independently enforce the same metadata contracts.

Each operation must guard the input device, launch on that device's current
PyTorch CUDA stream, check launch errors, and allocate owned outputs. Input
storage is read-only. Callers remain responsible for cross-stream dependencies
when passing tensors produced on another stream.

## Build and tests

CPU checks cover packaging, imports and rejected input metadata. GPU tests live
under `tests/gpu`; they skip explicitly when no CUDA device exists. On a CUDA
host, extension build failures fail the tests. Local CPU success does not establish
that the CUDA sources compile or run correctly. GPU build/runtime validation must
precede backend integration or performance claims.

The decoder still uses its existing reference selection path; native dispatch
into generation belongs to a later step. The relevant upstream contracts are
[PyTorch extension loading](https://docs.pytorch.org/docs/stable/cpp_extension.html)
and [CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Packed-mask count

`kernels.cuda.mask_count(mask, V)` returns a new scalar int64 CUDA tensor. A
256-thread block scans words, masks the final vocabulary tail, reduces per-warp
counts and writes the entire result once. No caller-supplied accumulator, atomic
initialization or host readback is required. Signed words are treated as uint32
bit patterns. Zero support returns zero; an all-valid row returns V.

`mask_density(mask, V)` performs a subsequent float64 conversion and division on
the device. Its cost includes those PyTorch operations; it is not a fused count
and division kernel. Calling `.item()` on either result synchronizes explicitly.
The reduction uses one thread block.

## Masked argmax

`kernels.cuda.masked_argmax(logits, mask)` fuses support filtering and maximum
selection without materializing a masked logit row. Exact score ties choose the
lowest valid token ID across lanes and warps. All-valid `-inf` scores still select
a valid token. Positive infinity is allowed. NaN anywhere in the row, including
an invalid position, raises; empty support also raises. Padding bits are ignored.

The native entry returns owned scalar output and status tensors. The checked
Python wrapper reads status and therefore synchronizes the current operation;
this overhead belongs in wrapper/end-to-end timings. The output stays on CUDA.
The implementation scans one row with one 256-thread block. Batched dispatch,
workspace reuse and generation-backend integration remain later steps.
