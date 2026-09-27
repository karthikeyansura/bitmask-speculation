"""Explicit Triton masked argmax/count with owned outputs and device status."""

from dataclasses import dataclass

from ._validation import selection_inputs

BLOCK_SIZE = 1024
MAX_BLOCKS = 16384


def reduction_shape(vocab_size: int) -> tuple[int, int]:
    """Bound the final reduction; never silently discard partial blocks."""
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError('vocab_size must be an integer')
    if not 0 < vocab_size <= BLOCK_SIZE * MAX_BLOCKS:
        raise ValueError(f'vocab_size must be in [1, {BLOCK_SIZE * MAX_BLOCKS}]')
    blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    return blocks, 1 << (blocks - 1).bit_length()


def require_device(device) -> None:
    import torch
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Triton requires an NVIDIA CUDA device')
    if torch.version.hip is not None or torch.cuda.get_device_capability(device) < (8, 0):
        raise ValueError('Triton backend requires NVIDIA compute capability 8.0 or newer')
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('install the triton extra on Linux for this backend') from exc


@dataclass(frozen=True, slots=True)
class MaskedReduction:
    """Owned device scalars; status 0=success, 1=empty support, 2=NaN.

    Read token_id only after checking status. check() synchronizes to the host.
    NaN anywhere in the input row is an error, including outside the mask.
    """

    token_id: object
    valid_count: object
    status: object

    def check(self) -> 'MaskedReduction':
        status = int(self.status.item())
        if status == 1:
            raise ValueError('grammar mask has no valid tokens')
        if status == 2:
            raise ValueError('logits contain NaN')
        if status != 0:
            raise RuntimeError(f'unknown reduction status: {status}')
        return self


def masked_reduce(logits, mask) -> MaskedReduction:
    """Launch asynchronously on the input device's current stream.

    Inputs are contiguous float32 logits and signed int32 packed words already
    on the same CUDA device. Ties choose the lowest valid ID, including -inf.
    Input storage is never modified; each call allocates independent outputs.
    """
    import torch
    vocab = selection_inputs(logits, mask)
    blocks, final_width = reduction_shape(vocab)
    require_device(logits.device)
    from ._triton_kernels import partial_reduce, finish_reduce
    with torch.cuda.device(logits.device):
        maximum = torch.empty(blocks, device=logits.device, dtype=torch.float32)
        index = torch.empty(blocks, device=logits.device, dtype=torch.int32)
        count = torch.empty_like(index)
        bad = torch.empty_like(index)
        token_out = torch.empty((), device=logits.device, dtype=torch.int64)
        count_out = torch.empty_like(token_out)
        status = torch.empty((), device=logits.device, dtype=torch.int32)
        partial_reduce[(blocks,)](logits, mask, maximum, index, count, bad,
                                  vocab, BLOCK_SIZE)
        finish_reduce[(1,)](maximum, index, count, bad, token_out, count_out,
                            status, vocab, blocks, final_width)
        return MaskedReduction(token_out, count_out, status)


def masked_argmax(logits, mask):
    """Checked selection; returns an owned int64 device scalar."""
    return masked_reduce(logits, mask).check().token_id
