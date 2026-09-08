"""Explicit single-row CUDA primitives; no implicit fallback or tensor copies."""

from ._extension import load_extension
from ._validation import mask_input


def mask_count(mask, vocab_size: int):
    """Return an owned scalar int64 CUDA tensor; no host synchronization."""
    vocab_size = mask_input(mask, vocab_size)
    return load_extension().mask_count(mask, vocab_size)


def mask_density(mask, vocab_size: int):
    """Return a scalar float64 CUDA tensor including the count and division."""
    import torch
    return mask_count(mask, vocab_size).to(torch.float64) / vocab_size
