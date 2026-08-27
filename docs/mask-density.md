# Packed-mask density

`bitmask_speculation.grammar.density` provides `packed_mask_count(mask, vocab_size)`
and `packed_mask_density(mask, vocab_size)`. The latter returns count divided by
vocabulary size. These are CPU references requiring the `models` extra.

Each row holds exactly `ceil(vocab_size / 32)` signed int32 words. Token `t` maps
to bit `t % 32` in word `t // 32`. Negative words carry ordinary bit patterns:
`-1` has 32 set bits; `-2147483648` has only bit 31 set. Padding at token IDs beyond
the vocabulary never contributes, even if every padding bit is set.

```python
import torch
from bitmask_speculation.grammar.density import packed_mask_count, packed_mask_density

mask = torch.tensor([-2147483648, -1], dtype=torch.int32)
assert packed_mask_count(mask, 33) == 2  # IDs 31 and 32; ignore 31 padding bits.
assert packed_mask_density(mask, 33) == 2 / 33
```

Vocabulary size must be a positive Python integer; booleans are rejected.
The mask must be a one-dimensional, strided CPU int32 tensor with the exact word
count. Noncontiguous views are supported. Scalars, batched rows, sparse tensors,
other dtypes and other devices fail explicitly. The input is never modified.
Zero valid bits give density 0; all vocabulary bits give density 1. Empty support
is still a selection error unless the grammar is already terminal.

`XGrammarState.mask_density()` fills its private workspace without accepting any
token. It returns 1 for XGrammar's all-valid fast path without reading stale mask
contents. A terminal state raises `GrammarTerminatedError`; callers must check
termination before querying. Forks remain independent.

The implementation converts the CPU row to Python words and uses integer
`bit_count`, normalizing signed words and masking the tail. It avoids per-word
tensor scalar extraction and does not expand a vocabulary-sized boolean mask.
It is not a CUDA/Triton kernel. Mask filling and counting still have a cost;
selection may refill the mask. Density measures support size, not draft-target
agreement, and does not establish a speedup.
