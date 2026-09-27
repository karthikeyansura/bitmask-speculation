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
