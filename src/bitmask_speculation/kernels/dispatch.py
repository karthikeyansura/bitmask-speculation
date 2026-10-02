"""Explicit constrained-selection dispatch; reference remains the default."""

BACKENDS = ('reference', 'cuda', 'triton')


def validate_backend(backend, device):
    import torch
    if backend not in BACKENDS:
        raise ValueError(f'backend must be one of {BACKENDS}')
    if backend == 'reference':
        return
    if device.type != 'cuda' or not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError(f'{backend} backend requires NVIDIA CUDA')
    if backend == 'triton':
        from .triton import require_device
        require_device(device)


def packed_mask(grammar):
    """Get an owned CPU mask; generic grammar states use validated token IDs."""
    import torch
    from ..decoding._validation import token_id, vocabulary_size
    vocab = vocabulary_size(grammar.vocab_size)
    snapshot = getattr(grammar, 'packed_mask', None)
    if snapshot is not None:
        mask = snapshot()
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.int32
            or mask.device.type != 'cpu' or mask.layout != torch.strided
            or mask.ndim != 1 or mask.numel() != (vocab + 31)//32):
            raise ValueError('grammar packed_mask must return a CPU int32 vocabulary row')
        return mask.clone().contiguous()
    words = [0] * ((vocab + 31)//32)
    for value in grammar.allowed_token_ids():
        value = token_id(value, vocab, 'allowed token')
        words[value//32] |= 1 << (value % 32)
    return torch.tensor([w if w < 2**31 else w-2**32 for w in words],dtype=torch.int32)


def select_token(scores, grammar, backend='reference') -> int:
    """Select without advancing grammar; native error status is checked first.

    The engine explicitly copies CPU grammar masks to CUDA and widens fp16/bf16
    logits losslessly to fp32. Float64 native selection is rejected, not rounded.
    Native selection synchronizes for its checked token; no fallback is applied.
    """
    import torch
    validate_backend(backend, scores.device)
    if backend == 'reference':
        from ..grammar.selection import masked_argmax
        return masked_argmax(scores,grammar.allowed_token_ids())
    if scores.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise TypeError('native selection accepts float16, bfloat16 or float32 logits')
    mask = packed_mask(grammar).to(scores.device)
    values = scores.to(dtype=torch.float32).contiguous()
    if backend == 'cuda':
        from .cuda import masked_argmax
    else:
        from .triton import masked_argmax
    return int(masked_argmax(values, mask).item())
