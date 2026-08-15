"""Correctness-first selection over explicit valid token IDs."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..decoding._validation import token_id

if TYPE_CHECKING:
    from torch import Tensor


def masked_argmax(scores: "Tensor", allowed_token_ids: Iterable[int]) -> int:
    """Select the lowest-ID valid maximum without modifying scores.

    NaN anywhere in the row is rejected, including at invalid token positions.
    Valid negative infinity scores remain selectable; an empty valid set fails.
    """
    import torch

    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty one-dimensional tensor")
    if not scores.is_floating_point():
        raise TypeError("scores must have a floating-point dtype")
    allowed = sorted({token_id(t, scores.numel(), "allowed token") for t in allowed_token_ids})
    if not allowed:
        raise ValueError("grammar has no valid next token")
    with torch.inference_mode():
        if torch.isnan(scores).any().item():
            raise ValueError("next-token logits contain NaN")
        indices = torch.tensor(allowed, dtype=torch.long, device=scores.device)
        winner = scores.index_select(0, indices).argmax().item()
        return allowed[winner]
