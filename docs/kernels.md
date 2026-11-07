# CUDA and Triton kernels

<a id="cuda-primitives"></a>

## Native CUDA primitives

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

### Tensor contract

Primitives use a single contiguous row on one CUDA device: masks are signed int32
with exactly `ceil(V/32)` words; logits are float32 with V entries. V must be
positive and fit an int32 token index. Batched rows, strided views, implicit dtype
casts and implicit CPU/device transfers are unsupported. Python validates before
loading; native entry points independently enforce the same metadata contracts.

Each operation must guard the input device, launch on that device's current
PyTorch CUDA stream, check launch errors, and allocate owned outputs. Input
storage is read-only. Callers remain responsible for cross-stream dependencies
when passing tensors produced on another stream.

### Build and tests

`TransformersGreedyModel` defaults to reference selection. `TransformersEngine`
and the CLI default to `auto`, which selects Triton, else the CUDA extension when
its build prerequisites are present, else reference. The relevant upstream contracts are
[PyTorch extension loading](https://docs.pytorch.org/docs/stable/cpp_extension.html)
and [CUDA stream semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).

### Packed-mask count

`kernels.cuda.mask_count(mask, V)` returns a new scalar int64 CUDA tensor. A
256-thread block scans words, masks the final vocabulary tail, reduces per-warp
counts and writes the entire result once. No caller-supplied accumulator, atomic
initialization or host readback is required. Signed words are treated as uint32
bit patterns. Zero support returns zero; an all-valid row returns V.

`mask_density(mask, V)` performs a subsequent float64 conversion and division on
the device. Its cost includes those PyTorch operations; it is not a fused count
and division kernel. Calling `.item()` on either result synchronizes explicitly.
The reduction uses one thread block.

### Masked argmax

`kernels.cuda.masked_argmax(logits, mask)` fuses support filtering and maximum
selection without materializing a masked logit row. Exact score ties choose the
lowest valid token ID across lanes and warps. All-valid `-inf` scores still select
a valid token. Positive infinity is allowed. NaN anywhere in the row, including
an invalid position, raises; empty support also raises. Padding bits are ignored.

The native entry returns owned scalar output and status tensors. The checked
Python wrapper reads status and therefore synchronizes the current operation;
this overhead belongs in wrapper/end-to-end timings. The output stays on CUDA.
The CUDA implementation scans one row with one 256-thread block. The engine
can select it explicitly for constrained argmax. Triton reduction capabilities
and scratch ownership are specified in [Triton contracts](kernels.md#triton-reductions).

### Standalone masked sampling

`kernels.cuda.masked_sample(logits, mask, temperature=..., generator=...)` samples
from masked softmax using inverse CDF. Positive temperature draws one float64
uniform on the input device through PyTorch. Pass an explicitly seeded CUDA
`torch.Generator` for repeatable sequences in a fixed environment; no private
cuRAND state or hidden seed reset is used. State advances across calls, including
value-error calls after the draw. Build and metadata errors precede the draw.
Temperature zero consumes no randomness and uses the same argmax primitive.

The CUDA sampler first reduces the valid maximum, then reduces exponentiated
weights, then walks the CDF serially on thread zero. It uses float64 arithmetic
for normalization and compares a uniform in [0, 1) using a strict CDF boundary.
A last-positive-weight fallback handles summation rounding at the endpoint.
There are multiple scans; this is not an online or single-pass softmax algorithm.
No probability tensor is materialized, but serial CDF cost may dominate.

Temperature must be finite and nonnegative. Positive-temperature sampling rejects
valid +inf, all-valid -inf and empty support; invalid +inf is ignored. NaN anywhere
raises. Valid -inf entries have zero mass and cannot win even for uniform zero.
Very small positive temperatures subtract the maximum before division to avoid
overflow of the largest scaled score. Output and status are independently owned.
The checked wrapper synchronizes to report errors, just like argmax.

`kernels.reference.masked_sample_reference` accepts CPU tensors and an explicit
uniform for deterministic boundary comparisons. RNG identity across devices or
PyTorch versions is not promised. This primitive is separate from the greedy
speculative engine and does not establish distribution-preserving stochastic
speculative decoding. GPU tests check seeded replay and the categorical distribution.

<a id="triton-reductions"></a>

## Triton masked reductions

`kernels.triton.masked_reduce(logits, mask)` fuses masked argmax and valid-token
count in two passes. The first pass scans 1,024 tokens per block; the second
reduces every partial block. The supported vocabulary range is 1–16,777,216.
Larger inputs fail before launch rather than truncating the final reduction.

The explicit backend requires Linux, NVIDIA compute capability 8.0 or newer,
PyTorch with CUDA, and the `triton` extra. Importing the package does not import
Triton or compile kernels. A requested backend never falls back silently.

Inputs are contiguous one-dimensional float32 logits and int32 packed masks on
the same CUDA device. Words are interpreted as unsigned bits, including bit 31;
padding bits beyond the vocabulary do not contribute to the count. Inputs remain
unchanged. Ties select the lowest valid token ID; valid negative infinity is
eligible. NaN anywhere in the row is an error, even at a masked-out position.

The result owns device scalars `token_id` (int64), `valid_count` (int64), and
`status` (int32). Status 0 means success, 1 means empty support, and 2 means NaN.
Calling `.check()` synchronizes and raises for an invalid result. Never consume
an unchecked token ID. `masked_argmax` performs that check automatically.

For a short installed-package check, run `python examples/validate_gpu.py`.
In Colab the same file can run from a `!python` cell after installing the wheel
and its GPU dependencies. No notebook-specific implementation is needed.
The runner exits 2 when CUDA is unavailable; a missing Triton dependency or a
failed assertion on a present GPU is a failure. Signed-word fixtures cover
`-2147483648` (only bit 31) and `-1` (all bits), including masked tail padding.

### Engine dispatch

Pass `backend="reference"`, `"cuda"`, or `"triton"` to
`TransformersEngine.from_pretrained`, or `--backend` to the CLI. Unsupported
hardware fails before model loading. There is no automatic backend choice.
Both target and constrained draft selection use the chosen backend; unconstrained
argmax remains PyTorch. Grammar transitions occur only after checked selection.

XGrammar exposes owned CPU packed-mask snapshots. Native dispatch copies a
snapshot to the model device each selection. Float16 and bfloat16 scores widen
exactly to float32; float64 is rejected by native selection. These primitives do
not manage model KV caches, GPU-resident grammar, or eliminate mask transfers. C4 scheduling still
uses the CPU mask-density reference; the fused Triton count does not replace it.

### Scratch ownership

Pass a `ReductionWorkspace(device)` as `workspace=` to reuse partial-reduction
buffers. It binds to the device's current CUDA stream at construction. Using it
on a different device or stream raises; concurrent host submissions raise rather
than race. A workspace grows to its largest requested capacity. `clear()` releases
scratch on its bound stream. `allocation_count` counts scratch allocation groups.

Returned token/count/status tensors are always newly owned and remain unchanged
by subsequent calls. Only intermediate buffers are reused. The Transformers
adapter creates one workspace lazily for each Triton adapter, binding it to the
first constrained-selection stream. Create separate adapters for other streams.
CPU masks and result tensors are still allocated per selection. Allocation tests
measure reuse; speedup requires a separate GPU timing experiment.

### Independent batch rows

`batched_masked_reduce` accepts contiguous float32 `[B,V]` logits and int32
`[B,ceil(V/32)]` masks on the same CUDA device. It returns owned token/count/status
vectors. Every row has its own mask and status; errors in one row do not alter
other rows. `.check()` reports the invalid row indices. Strided or broadcast
views fail explicitly. Batch sizes from 0 to 65,535 are supported; an empty vocabulary is not.

Single-row and batched calls share the same two kernels and workspace ownership
rules. A workspace caches enough partial storage for its largest `B * blocks`.
The primitives do not imply batched grammar matchers or a batched request loop.
