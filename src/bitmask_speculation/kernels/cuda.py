"""Explicit single-row CUDA primitives; no implicit fallback or tensor copies."""

from ._extension import load_extension
from ._validation import mask_input, selection_inputs, temperature_input


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
    if code == 3:
        raise ValueError("uniform must be finite and in [0, 1)")
    if code == 4:
        raise ValueError("valid logits have no finite probability mass")
    if code == 5:
        raise ValueError("positive infinity is unsupported for stochastic sampling")
    if code != 0:
        raise RuntimeError(f"unknown native selection status: {code}")
    return output


def masked_argmax(logits, mask):
    """Return the lowest-ID valid maximum as an owned scalar int64 CUDA tensor."""
    selection_inputs(logits, mask)
    return _checked_selection(load_extension().masked_argmax(logits, mask))


def masked_sample(logits, mask, *, temperature: float = 1.0, generator=None):
    """Sample masked softmax; temperature zero uses deterministic argmax.

    Positive temperatures draw one float64 uniform on the input device using the
    supplied PyTorch generator (or its default). Checked status synchronizes.
    This is a standalone primitive, not stochastic speculative decoding.
    """
    import torch
    temperature = temperature_input(temperature)
    selection_inputs(logits, mask)
    if temperature == 0:
        return masked_argmax(logits, mask)
    native = load_extension()  # Build errors must not consume generator state.
    uniform = torch.rand((), dtype=torch.float64, device=logits.device, generator=generator)
    return _checked_selection(native.masked_sample(logits, mask, temperature, uniform))
