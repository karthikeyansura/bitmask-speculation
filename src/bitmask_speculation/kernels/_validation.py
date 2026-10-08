"""Metadata validation before native dispatch; no device copies or compilation."""

from ..decoding._validation import vocabulary_size


def mask_input(mask, vocab_size):
    import torch
    vocab_size = vocabulary_size(vocab_size)
    if vocab_size > 2147483647:
        raise ValueError("vocab_size exceeds the native int32 index range")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.int32:
        raise TypeError("mask must be an int32 tensor")
    if mask.layout != torch.strided or mask.ndim != 1 or not mask.is_contiguous():
        raise ValueError("mask must be a contiguous one-dimensional strided tensor")
    if mask.numel() != (vocab_size + 31) // 32:
        raise ValueError("mask word count must equal ceil(vocab_size / 32)")
    if mask.device.type != "cuda":
        raise ValueError("mask must be on CUDA; no implicit transfer")
    return vocab_size


def selection_inputs(logits, mask):
    import torch
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise TypeError("logits must be a float32 tensor")
    if logits.layout != torch.strided or logits.ndim != 1 or not logits.is_contiguous():
        raise ValueError("logits must be a contiguous one-dimensional strided tensor")
    vocab_size = mask_input(mask, logits.numel())
    if logits.device != mask.device:
        raise ValueError("logits and mask must be on the same CUDA device")
    return vocab_size


def temperature_input(temperature):
    import math
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be numeric")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    return float(temperature)


def batched_selection_inputs(logits, mask):
    import torch
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise TypeError("logits must be a float32 tensor")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.int32:
        raise TypeError("mask must be an int32 tensor")
    for value in (logits, mask):
        if value.layout != torch.strided or value.ndim != 2 or not value.is_contiguous():
            raise ValueError("batch inputs must be contiguous two-dimensional strided tensors")
    batch, vocab = logits.shape
    if batch > 65535:
        raise ValueError('batch size exceeds the CUDA grid-y limit of 65535')
    vocabulary_size(vocab)
    if mask.shape != (batch, (vocab + 31)//32):
        raise ValueError("mask shape must be [batch, ceil(vocab_size / 32)]")
    if logits.device.type != "cuda" or logits.device != mask.device:
        raise ValueError("batch inputs must be on the same CUDA device")
    return batch, vocab
