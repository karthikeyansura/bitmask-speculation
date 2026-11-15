# Kernels

Two GPU backends implement constrained selection over packed masks: a CUDA
extension (`csrc/`) and Triton kernels. Both return the lowest-ID masked argmax,
the exact valid-token count and a status code per row, and both reduce a whole
`[B, V]` batch in two launches.

<a id="build"></a>

## Build

The CUDA extension compiles on first use through PyTorch's JIT extension loader;
importing the package compiles nothing. It needs CUDA-enabled PyTorch, `nvcc`, a
C++ compiler and Ninja. Set `TORCH_CUDA_ARCH_LIST` (for example `8.6` for A10G,
`8.0` for A100, `9.0` for H100) and `MAX_JOBS`. Wheels ship the sources under
`bitmask_speculation/kernels/_src/`; the build cache keys on sources, PyTorch and
CUDA versions and the architecture list. Build failures raise.

```python
from bitmask_speculation.kernels._extension import load_extension
print(load_extension().runtime_version())
```

Triton requires Linux, an NVIDIA GPU of compute capability 8.0 or newer and the
`triton` extra.

<a id="contract"></a>

## Tensor contract

| Input | Type | Shape |
|---|---|---|
| logits | float32, contiguous | `[V]` or `[B, V]` |
| mask | int32, contiguous | `[ceil(V/32)]` or `[B, ceil(V/32)]` |

Inputs share one CUDA device, `1 ≤ V ≤ 2^31 − 1` and `B ≤ 65,535`. Strided
views, implicit casts and host-device transfers are rejected rather than repaired.
Python validates first and the native entry points check again. Every operation
guards the input device, launches on that device's current PyTorch stream and
returns newly allocated outputs; inputs are read-only.

| Status | Meaning |
|---|---|
| 0 | success |
| 1 | no valid token |
| 2 | NaN in the logits, at any position |
| 3 | sampler only: uniform outside [0, 1) |
| 4 | sampler only: no finite probability mass |
| 5 | sampler only: valid `+inf` |

Checked wrappers read the status and synchronize; unchecked tensors stay on the
device. Never consume a token whose status was not checked.

<a id="argmax"></a>

## Masked argmax and count

`argmax.cu` and the Triton kernels share one structure.

**Pass 1, partial reduction.** The grid spans `(blocks per row, rows)`. In CUDA a
128-thread block covers 4,096 tokens and a warp owns 1,024 consecutive tokens:
each lane loads one 32-bit mask word, clears padding bits past `V` and counts it
with `__popc`. The warp then iterates over its 32 words, broadcasting word `j`
with `__shfl_sync` while each lane reads logit `j*32 + lane`, so mask words are
read once and logit loads are coalesced. Warp shuffles and shared memory combine
lanes into one `(max, index, count, nan)` partial per block. Triton covers 1,024
tokens per program with vectorized loads.

**Pass 2, finish.** One warp per row (CUDA) or one program per row (Triton) folds
the row's partials into the token, count and status.

The combine operator keeps the larger value and, on equal values, the lower token
ID, which makes the result independent of block scheduling. Valid `-inf` scores
stay eligible; `+inf` is allowed.

```python
from bitmask_speculation.kernels import cuda, triton

token = cuda.masked_argmax(logits, mask)                    # checked, scalar int64
batch = cuda.batched_masked_reduce(logits_bv, masks_bw)    # token_ids, valid_counts, status
batch = triton.batched_masked_reduce(logits_bv, masks_bw, workspace=workspace)
```

`popcount.cu` provides a standalone count, `cuda.mask_count(mask, V)`: a
256-thread block applies `__popc` per word and reduces with warp shuffles.

<a id="sampling"></a>

## Masked sampling

`cuda.masked_sample(logits, mask, temperature=..., generator=...)` draws one token
from the masked softmax by inverse CDF. The single uniform comes from the supplied
`torch.Generator`, so seeded sequences replay; temperature zero consumes no
randomness and returns the argmax.

The kernel reads the logits once. Each of 256 threads folds a contiguous token
range with the online softmax recurrence: a running maximum `m` and a sum `s`
rescaled by `exp((m_old − m_new) / T)` whenever the maximum grows, in float64.
Rescaling each range to the block maximum gives per-range probability mass in
token-ID order. A block scan locates the range containing `u · total`, and only
that thread walks its range against a strict CDF boundary. Compared with an
unfused mask → softmax → multinomial pipeline, which writes and re-reads
vocabulary-sized tensors at each stage, the kernel moves about one fifth of the
memory traffic and materializes no probability vector. A last-positive-weight
fallback absorbs rounding at the upper boundary.
`kernels.reference.masked_sample_reference` reproduces the same CDF order on the
CPU for an explicit uniform.

<a id="dispatch"></a>

## Engine dispatch

The engine applies the chosen backend to target and constrained-draft selection;
unconstrained argmax stays in PyTorch. XGrammar produces an owned CPU mask, which
dispatch copies to the model device per selection; float16/bfloat16 scores widen
exactly to float32. The grammar advances only after a checked selection.

<a id="workspace"></a>

## Triton workspace

`ReductionWorkspace(device)` reuses partial-reduction buffers across Triton calls.
It binds to the device's current stream at construction, raises on use from another
device or stream, and grows to the largest `B × blocks` requested. Outputs are
always freshly owned; only scratch is reused. The Transformers adapter creates one
workspace per Triton adapter. CUDA calls allocate partials from PyTorch's caching
allocator on the current stream.

`python examples/validate_gpu.py` runs an installed-package smoke check of the
Triton reduction, including signed-word and tail-padding fixtures.
