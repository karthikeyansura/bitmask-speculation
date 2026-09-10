"""Explicit single-row CUDA primitives; no implicit fallback or tensor copies."""

from ._extension import load_extension
from ._validation import mask_input, selection_inputs


def mask_count(mask, vocab_size: int):
    """Return an owned scalar int64 CUDA tensor; no host synchronization."""
    vocab_size = mask_input(mask, vocab_size)
    return load_extension().mask_count(mask, vocab_size)


def mask_density(mask, vocab_size: int):
    """Return a scalar float64 CUDA tensor including the count and division."""
    import torch
    return mask_count(mask, vocab_size).to(torch.float64) / vocab_size


def _checked_selection(result):
    output, status = result
    code = status.item()  # Explicit host synchronization to surface value errors.
    if code == 1:
        raise ValueError("mask has no valid token")
    if code == 2:
        raise ValueError("logits contain NaN")
    if code != 0:
        raise RuntimeError(f"unknown native selection status: {code}")
    return output


def masked_argmax(logits, mask):
    """Return the lowest-ID valid maximum as an owned scalar int64 CUDA tensor."""
    selection_inputs(logits, mask)
    return _checked_selection(load_extension().masked_argmax(logits, mask))
