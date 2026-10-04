# Triton masked reductions

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

Tests cover tails, ties, large final reductions, invalid support, NaN, and
retained outputs. Run `python -m pytest tests/gpu/test_triton_reduction.py` on a
compatible GPU. CPU skips do not establish compilation or runtime correctness.

For a short installed-package check, run `python examples/validate_gpu.py`.
In Colab the same file can run from a `!python` cell after installing the wheel
and its GPU dependencies. No notebook-specific implementation is needed.
The runner exits 2 when CUDA is unavailable; a missing Triton dependency or a
failed assertion on a present GPU is a failure. Signed-word fixtures cover
`-2147483648` (only bit 31) and `-1` (all bits), including masked tail padding.

## Engine dispatch

Pass `backend="reference"`, `"cuda"`, or `"triton"` to
`TransformersEngine.from_pretrained`, or `--backend` to the CLI. Unsupported
hardware fails before model loading. There is no automatic backend choice.
Both target and constrained draft selection use the chosen backend; unconstrained
argmax remains PyTorch. Grammar transitions occur only after checked selection.

XGrammar exposes owned CPU packed-mask snapshots. Native dispatch copies a
snapshot to the model device each selection. Float16 and bfloat16 scores widen
exactly to float32; float64 is rejected by native selection. No KV caching,
GPU-resident grammar, or transfer elimination is implied. C4 scheduling still
uses the CPU mask-density reference; the fused Triton count does not replace it.

## Scratch ownership

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
