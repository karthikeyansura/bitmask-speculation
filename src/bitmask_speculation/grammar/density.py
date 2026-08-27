"""CPU reference popcount for one signed int32 grammar-mask row."""

from typing import TYPE_CHECKING

from ..decoding._validation import vocabulary_size

if TYPE_CHECKING:
    from torch import Tensor


def packed_mask_count(mask: "Tensor", vocab_size: int) -> int:
    """Count vocabulary bits, ignoring padding in the final 32-bit word.

    Accept a strided, one-dimensional CPU int32 tensor of exactly ceil(V/32)
    words. Input storage is never changed. GPU reductions are separate backends;
    this reference does not silently transfer a device mask to the host.
    """
    import torch

    vocab_size = vocabulary_size(vocab_size)
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a tensor")
    if mask.dtype != torch.int32:
        raise TypeError("mask must have dtype int32")
    if mask.layout != torch.strided or mask.ndim != 1:
        raise ValueError("mask must be a strided one-dimensional row")
    if mask.device.type != "cpu":
        raise ValueError("reference mask counting requires a CPU tensor")
    if mask.numel() != (vocab_size + 31) // 32:
        raise ValueError("mask word count must equal ceil(vocab_size / 32)")
    words = mask.tolist()
    # Python integers are unbounded; normalize signed words before bit_count.
    count = sum((word & 0xFFFFFFFF).bit_count() for word in words[:-1])
    tail_bits = vocab_size % 32 or 32
    return count + (words[-1] & ((1 << tail_bits) - 1)).bit_count()


def packed_mask_density(mask: "Tensor", vocab_size: int) -> float:
    """Return valid_count / vocab_size, including zero for an empty support."""
    return packed_mask_count(mask, vocab_size) / vocab_size
