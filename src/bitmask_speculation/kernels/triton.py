"""Explicit Triton masked argmax/count with owned outputs and device status."""

from dataclasses import dataclass

from .workspace import ReductionWorkspace

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


def masked_reduce(logits, mask, *, workspace: ReductionWorkspace | None = None) -> MaskedReduction:
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
        scratch = workspace if workspace is not None else ReductionWorkspace(logits.device)
        token_out = torch.empty((), device=logits.device, dtype=torch.int64)
        count_out = torch.empty_like(token_out)
        status = torch.empty((), device=logits.device, dtype=torch.int32)
        with scratch.acquire(logits.device, blocks) as (maximum, index, count, bad):
            partial_reduce[(blocks,)](logits, mask, maximum, index, count, bad,
                                      vocab, BLOCK_SIZE)
            finish_reduce[(1,)](maximum, index, count, bad, token_out, count_out,
                                status, vocab, blocks, final_width)
        return MaskedReduction(token_out, count_out, status)


def masked_argmax(logits, mask, *, workspace: ReductionWorkspace | None = None):
    """Checked selection; returns an owned int64 device scalar."""
    return masked_reduce(logits, mask, workspace=workspace).check().token_id


@dataclass(frozen=True, slots=True)
class BatchedMaskedReduction:
    """Owned vectors, with one independent status per input row."""

    token_ids: object
    valid_counts: object
    status: object

    def check(self) -> 'BatchedMaskedReduction':
        codes = self.status.tolist()  # One explicit device-to-host synchronization.
        errors = [(row, code) for row, code in enumerate(codes) if code]
        if errors:
            raise ValueError(f'invalid reduction rows (row, status): {errors}')
        return self


def batched_masked_reduce(logits, mask, *, workspace=None) -> BatchedMaskedReduction:
    """Reduce [B,V] logits and [B,ceil(V/32)] masks without row broadcasting.

    Inputs must be contiguous; strided views are rejected instead of copied.
    B=0 returns empty owned outputs. V must satisfy the single-row bound.
    This primitive does not batch stateful speculative request execution.
    """
    import torch
    from ._validation import batched_selection_inputs
    batch, vocab = batched_selection_inputs(logits, mask)
    blocks, final_width = reduction_shape(vocab)
    require_device(logits.device)
    from ._triton_kernels import partial_reduce, finish_reduce
    with torch.cuda.device(logits.device):
        tokens = torch.empty(batch,device=logits.device,dtype=torch.int64)
        counts = torch.empty_like(tokens)
        status = torch.empty(batch,device=logits.device,dtype=torch.int32)
        if batch == 0:
            return BatchedMaskedReduction(tokens,counts,status)
        scratch = workspace if workspace is not None else ReductionWorkspace(logits.device)
        with scratch.acquire(logits.device, batch * blocks) as (maximum,index,count,bad):
            partial_reduce[(blocks,batch)](logits,mask,maximum,index,count,bad,vocab,BLOCK_SIZE)
            finish_reduce[(batch,)](maximum,index,count,bad,tokens,counts,status,
                                    vocab,blocks,final_width)
        return BatchedMaskedReduction(tokens,counts,status)
