"""Greedy evaluation with request-scoped target and round-scoped draft caches."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ..decoding._validation import token_id, vocabulary_size
from ..decoding.verification import VerifiedToken, verify_prefix
from ..grammar import GrammarState, GrammarTerminatedError
from ..kernels.dispatch import select_token, validate_backend

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class TransformersGreedyModel:
    """Single-device greedy evaluation with optional grammar-constrained selection.

    Load the model, move it to its intended device, and call ``eval()`` before
    constructing the adapter. No tokenizer, weights, or device is chosen here.
    Generation-config processors, sampling and model offloading are not applied.
    Standalone ``next_token`` and ``verify_prefix`` calls recompute the full
    prefix; ``target_session()`` and ``draft_session()`` own the caches.
    """

    def __init__(
        self, model: PreTrainedModel, *, backend: str = "reference",
        draft_cache: bool = True, target_cache: bool = True,
    ) -> None:
        import torch

        if type(draft_cache) is not bool:
            raise TypeError("draft_cache must be a boolean")
        if type(target_cache) is not bool:
            raise TypeError("target_cache must be a boolean")
        self.draft_cache = draft_cache
        self.target_cache = target_cache
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
        validate_backend(backend, self._device)
        self.backend = backend
        self._reduction_workspace = None
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
            return self._logits(output, len(prefix))

    def _logits(self, output, length: int):
        logits = output.logits
        expected = (1, length, self.vocab_size)
        if not isinstance(logits, self._torch.Tensor) or tuple(logits.shape) != expected:
            raise ValueError(f"expected model logits with shape {expected}")
        if not logits.is_floating_point():
            raise TypeError("model logits must have a floating-point dtype")
        return logits[0]

    @contextmanager
    def draft_session(self):
        """Own one draft round's cache; discard it on completion or failure.

        Standalone next_token calls remain uncached. Sessions never share model
        cache state, including when the adapter is reused.
        """
        if not self.draft_cache:
            yield self
            return
        session = _DraftSession(self)
        try:
            yield session
        finally:
            session.close()

    @contextmanager
    def target_session(self):
        """Own one request's target cache; discard it on completion or failure.

        The cache covers full-attention layers only and is validated before any
        forward pass. Other layer types fail; use ``target_cache=False``.
        """
        if not self.target_cache:
            yield self
            return
        session = _TargetSession(self)
        try:
            yield session
        finally:
            session.close()

    def _new_target_cache(self):
        from transformers import DynamicCache

        try:
            cache = DynamicCache(config=self._model.config)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("target cache requires a Transformers model config; use target_cache=False") from exc
        _check_croppable(cache)
        return cache

    def _check_grammar(self, grammar: GrammarState | None) -> None:
        if grammar is not None and vocabulary_size(grammar.vocab_size) != self.vocab_size:
            raise ValueError("grammar and model vocabularies differ")

    def _select_token(self, scores, grammar: GrammarState | None = None) -> int:
        if grammar is not None:
            if grammar.is_terminated():
                raise GrammarTerminatedError("grammar has terminated")
            if self.backend == 'triton' and self._reduction_workspace is None:
                from ..kernels.workspace import ReductionWorkspace
                self._reduction_workspace = ReductionWorkspace(self._device)
            token = select_token(scores, grammar, self.backend, workspace=self._reduction_workspace)
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
        prefix, draft = self._verification_inputs(prefix, proposals, grammar)
        if grammar is not None and grammar.is_terminated():
            return iter(())
        return self._choices(draft, self._forward_logits(prefix + draft), len(prefix) - 1, grammar)

    def _verification_inputs(self, prefix, proposals, grammar):
        prefix = self._prefix(prefix)
        draft = tuple(token_id(t, self.vocab_size, "proposal token") for t in proposals)
        self._check_grammar(grammar)
        return prefix, draft

    def _choices(self, draft, logits, start, grammar) -> Iterator[VerifiedToken]:
        """Select lazily; logits row ``start + i`` predicts proposal ``i``."""
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


def _check_croppable(cache) -> None:
    from transformers.cache_utils import DynamicLayer

    unsupported = sorted({type(layer).__name__ for layer in cache.layers if type(layer) is not DynamicLayer})
    if unsupported:
        raise ValueError(
            f"target cache rollback supports full-attention layers only, found {', '.join(unsupported)}; "
            "use target_cache=False"
        )


class _TargetSession:
    """Target cache owned by a single request.

    The cache holds the keys and values of ``_tokens``. Each call reuses the
    longest cached prefix of its input, crops the rejected suffix, and feeds the
    remaining tokens, at least one, with absolute positions.
    """

    def __init__(self, adapter: TransformersGreedyModel):
        self.adapter = adapter
        self._cache = adapter._new_target_cache()
        self._tokens: tuple[int, ...] = ()
        self._closed = False

    @property
    def vocab_size(self) -> int:
        return self.adapter.vocab_size

    def close(self):
        self._cache = None
        self._tokens = ()
        self._closed = True

    def _check_open(self):
        if self._closed:
            raise RuntimeError("target session is closed")

    def _logits(self, sequence: tuple[int, ...]):
        """Return logits for ``sequence[start:]`` and ``start``."""
        adapter = self.adapter
        if adapter._model.training:
            raise ValueError("model must remain in evaluation mode")
        cached = self._tokens
        start, limit = 0, min(len(cached), len(sequence) - 1)
        while start < limit and cached[start] == sequence[start]:
            start += 1
        torch = adapter._torch
        try:
            with torch.inference_mode():
                if start == 0:
                    if self._cache is None or cached:
                        self._cache = adapter._new_target_cache()
                elif start < len(cached):
                    self._cache.crop(start - len(cached))
                self._tokens = sequence[:start]
                output = adapter._model(
                    input_ids=torch.tensor([sequence[start:]], dtype=torch.long, device=adapter._device),
                    attention_mask=torch.ones((1, len(sequence)), dtype=torch.long, device=adapter._device),
                    position_ids=torch.arange(start, len(sequence), device=adapter._device).unsqueeze(0),
                    past_key_values=self._cache, use_cache=True, return_dict=True,
                )
                logits = adapter._logits(output, len(sequence) - start)
                if (getattr(output, "past_key_values", None) is not self._cache
                        or self._cache.get_seq_length() != len(sequence)):
                    raise RuntimeError("target model did not extend the supplied cache; use target_cache=False")
                _check_croppable(self._cache)
        except BaseException:
            self._cache, self._tokens = None, ()
            raise
        self._tokens = sequence
        return logits, start

    def next_token(self, prefix: tuple[int, ...], *, grammar: GrammarState | None = None) -> int:
        self._check_open()
        adapter = self.adapter
        prefix = adapter._prefix(prefix)
        adapter._check_grammar(grammar)
        if grammar is not None and grammar.is_terminated():
            raise GrammarTerminatedError("grammar has terminated")
        logits, _ = self._logits(prefix)
        return adapter._select_token(logits[-1], grammar)

    def verify_prefix(
        self, prefix: Sequence[int], proposals: Sequence[int], *,
        grammar: GrammarState | None = None,
    ) -> Iterator[VerifiedToken]:
        """One cached forward over uncached prefix tokens plus proposals."""
        self._check_open()
        adapter = self.adapter
        prefix, draft = adapter._verification_inputs(prefix, proposals, grammar)
        if grammar is not None and grammar.is_terminated():
            return iter(())
        logits, start = self._logits(prefix + draft)
        return adapter._choices(draft, logits, len(prefix) - 1 - start, grammar)


class _DraftSession:
    """Append-only cache owned by a single speculative proposal round."""

    def __init__(self, adapter: TransformersGreedyModel):
        self.adapter = adapter
        self._cache = None
        self._expected_prefix = None
        self._closed = False

    def close(self):
        self._cache = None
        self._expected_prefix = None
        self._closed = True

    def next_token(self, prefix: tuple[int, ...], *, grammar: GrammarState | None = None) -> int:
        if self._closed:
            raise RuntimeError("draft session is closed")
        adapter = self.adapter
        prefix = adapter._prefix(prefix)
        adapter._check_grammar(grammar)
        if grammar is not None and grammar.is_terminated():
            raise GrammarTerminatedError("grammar has terminated")
        if adapter._model.training:
            raise ValueError("model must remain in evaluation mode")
        if self._expected_prefix is not None and prefix != self._expected_prefix:
            raise ValueError("draft session requires its previous choice appended to the prefix")
        tokens = prefix if self._cache is None else prefix[-1:]
        torch = adapter._torch
        with torch.inference_mode():
            output = adapter._model(
                input_ids=torch.tensor([tokens], dtype=torch.long, device=adapter._device),
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=adapter._device),
                position_ids=torch.arange(len(prefix) - len(tokens), len(prefix),
                                          device=adapter._device).unsqueeze(0),
                past_key_values=self._cache, use_cache=True, return_dict=True,
            )
            logits = adapter._logits(output, len(tokens))
            cache = getattr(output, "past_key_values", None)
            if cache is None:
                raise RuntimeError("draft model did not return past_key_values; disable draft_cache for this model")
            token = adapter._select_token(logits[-1], grammar)
            self._cache = cache
            self._expected_prefix = prefix + (token,)
            return token
