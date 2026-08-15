"""Full-prefix adapter for an already loaded Transformers causal language model."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from ..decoding._validation import token_id, vocabulary_size
from ..decoding.verification import VerifiedToken, verify_prefix
from ..grammar import GrammarState, GrammarTerminatedError
from ..grammar.selection import masked_argmax

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class TransformersGreedyModel:
    """Uncached, single-device reference with optional grammar-constrained selection.

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

    def _check_grammar(self, grammar: GrammarState | None) -> None:
        if grammar is not None and vocabulary_size(grammar.vocab_size) != self.vocab_size:
            raise ValueError("grammar and model vocabularies differ")

    def _select_token(self, scores, grammar: GrammarState | None = None) -> int:
        if grammar is not None:
            if grammar.is_terminated():
                raise GrammarTerminatedError("grammar has terminated")
            token = masked_argmax(scores, grammar.allowed_token_ids())
            if not grammar.accept_token(token):
                raise ValueError("grammar rejected its selected token")
            return token
        with self._torch.inference_mode():
            if self._torch.isnan(scores).any().item():
                raise ValueError("next-token logits contain NaN")
            return int(scores.argmax().item())

    def next_token(
        self, prefix: tuple[int, ...], *, grammar: GrammarState | None = None
    ) -> int:
        """Predict after the prefix and advance optional grammar state once."""
        prefix = self._prefix(prefix)
        self._check_grammar(grammar)
        if grammar is not None and grammar.is_terminated():
            raise GrammarTerminatedError("grammar has terminated")
        logits = self._forward_logits(prefix)
        return self._select_token(logits[-1], grammar)

    def verify_prefix(
        self, prefix: Sequence[int], proposals: Sequence[int], *,
        grammar: GrammarState | None = None,
    ) -> Iterator[VerifiedToken]:
        """Run one forward pass, then lazily select a verified continuation.

        The fixed prefix length determines all row offsets. Selection stops at
        the first mismatch; later rows condition on rejected tokens. The caller
        must stop consuming at EOS or its output limit. With grammar state, the
        iterator advances only selected tokens and stops at grammar termination.
        Full-prefix logits remain owned by this iterator until it is released.
        """
        prefix = self._prefix(prefix)
        draft = tuple(token_id(t, self.vocab_size, "proposal token") for t in proposals)
        self._check_grammar(grammar)
        if grammar is not None and grammar.is_terminated():
            return iter(())
        logits = self._forward_logits(prefix + draft)
        start = len(prefix) - 1
        choices = verify_prefix(
            draft,
            lambda i: self._select_token(logits[start + i], grammar),
            vocab_size=self.vocab_size,
        )
        if grammar is None:
            return choices

        def constrained_choices() -> Iterator[VerifiedToken]:
            try:
                while not grammar.is_terminated():
                    event = next(choices, None)
                    if event is None:
                        return
                    yield event
            finally:
                choices.close()

        return constrained_choices()
