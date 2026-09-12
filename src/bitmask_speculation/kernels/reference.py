"""CPU oracle for the standalone masked categorical sampler."""

import math

from ..grammar.density import packed_mask_count
from ..grammar.selection import masked_argmax
from ._validation import temperature_input


def masked_sample_reference(logits, mask, *, temperature: float, uniform: float) -> int:
    """Inverse CDF over masked softmax with an explicit U in [0, 1).

    Positive temperatures accept finite logits or -inf; valid +inf, all-valid
    -inf and NaN anywhere fail. Temperature zero follows greedy reference rules.
    """
    import torch
    temperature = temperature_input(temperature)
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise TypeError("logits must be float32")
    if logits.device.type != "cpu" or logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("reference requires a nonempty CPU logit row")
    packed_mask_count(mask, logits.numel())
    words = mask.tolist()
    allowed = [t for t in range(logits.numel()) if (words[t // 32] >> (t % 32)) & 1]
    if temperature == 0:
        return masked_argmax(logits, allowed)
    if isinstance(uniform, bool) or not isinstance(uniform, (int, float)):
        raise TypeError("uniform must be numeric")
    if not math.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("uniform must be in [0, 1)")
    values = logits.tolist()
    if any(math.isnan(v) for v in values):
        raise ValueError("logits contain NaN")
    if not allowed:
        raise ValueError("mask has no valid token")
    maximum = max(values[t] for t in allowed)
    if maximum == math.inf:
        raise ValueError("positive infinity is unsupported for stochastic sampling")
    if maximum == -math.inf:
        raise ValueError("valid logits have no finite probability mass")
    weights = [math.exp((values[t] - maximum) / temperature) for t in allowed]
    cutoff = uniform * math.fsum(weights)
    cumulative = 0.0
    last_positive = -1
    for token, weight in zip(allowed, weights):
        if weight == 0:
            continue
        last_positive = token
        cumulative += weight
        if cutoff < cumulative:
            return token
    return last_positive
