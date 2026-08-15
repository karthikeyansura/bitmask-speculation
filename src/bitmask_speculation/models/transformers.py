"""Full-prefix adapter for an already loaded Transformers causal language model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from ..decoding._validation import token_id, vocabulary_size
from ..decoding.verification import VerifiedToken, verify_prefix

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class TransformersGreedyModel:
    """Uncached, single-device reference with raw-logit greedy selection.

    Load the model, move it to its intended device, and call ``eval()`` before
    constructing the adapter. No tokenizer, weights, or device is chosen here.
    Generation-config processors, sampling and model offloading are not applied.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        import torch

        if model.training:
            raise ValueError("call model.eval() before constructing the adapter")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("the reference requires a decoder-only causal model")
        vocab_size = vocabulary_size(model.config.vocab_size)
        devices = {parameter.device for parameter in model.parameters()}
        devices.update(buffer.device for buffer in model.buffers())
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("model tensors must reside on one materialized device")
        self._device = next(iter(devices))
        self._model = model
        self._torch = torch
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _prefix(self, prefix: Sequence[int]) -> tuple[int, ...]:
        prefix = tuple(token_id(t, self.vocab_size, "prefix token") for t in prefix)
        if not prefix:
            raise ValueError("prefix must contain at least one token")
        return prefix

    def _forward_logits(self, prefix: tuple[int, ...]):
        if self._model.training:
            raise ValueError("model must remain in evaluation mode")
        torch = self._torch
        with torch.inference_mode():
            input_ids = torch.tensor([prefix], dtype=torch.long, device=self._device)
            output = self._model(input_ids=input_ids, use_cache=False, return_dict=True)
            logits = output.logits
            expected = (1, len(prefix), self.vocab_size)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
                raise ValueError(f"expected model logits with shape {expected}")
            if not logits.is_floating_point():
                raise TypeError("model logits must have a floating-point dtype")
            return logits[0]

    def _select_token(self, scores) -> int:
        with self._torch.inference_mode():
            if self._torch.isnan(scores).any().item():
                raise ValueError("next-token logits contain NaN")
            return int(scores.argmax().item())

    def next_token(self, prefix: tuple[int, ...]) -> int:
        """Predict after the full prefix; logits from earlier positions are unused."""
        logits = self._forward_logits(self._prefix(prefix))
        return self._select_token(logits[-1])

    def verify_prefix(
        self, prefix: Sequence[int], proposals: Sequence[int]
    ) -> Iterator[VerifiedToken]:
        """Run one forward pass, then lazily select a verified continuation.

        The fixed prefix length determines all row offsets. Selection stops at
        the first mismatch; later rows condition on rejected tokens. The caller
        must stop consuming at EOS, its output limit, or grammar termination.
        Full-prefix logits remain owned by this iterator until it is released.
        """
        prefix = self._prefix(prefix)
        draft = tuple(token_id(t, self.vocab_size, "proposal token") for t in proposals)
        logits = self._forward_logits(prefix + draft)
        start = len(prefix) - 1
        return verify_prefix(
            draft,
            lambda i: self._select_token(logits[start + i]),
            vocab_size=self.vocab_size,
        )
