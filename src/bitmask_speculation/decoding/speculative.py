"""Request-level greedy speculation with target-controlled termination."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

from ..grammar import GrammarState
from ._validation import request_inputs, token_id, vocabulary_size
from .greedy import GreedyModel, GreedyResult
from .verification import VerifiedToken


class VerificationTarget(Protocol):
    """Verify one round lazily and advance optional target grammar state.

    The iterator must follow verify_prefix semantics, including a target-only
    choice for empty proposals. No state may advance past a yielded event until
    the consumer requests another. A close method, when present, releases state
    owned by the iterator but does not roll back already yielded choices.
    """

    @property
    def vocab_size(self) -> int: ...

    def verify_prefix(
        self, prefix: Sequence[int], proposals: Sequence[int], *,
        grammar: GrammarState | None = None,
    ) -> Iterator[VerifiedToken]: ...


def speculative_decode(
    target: VerificationTarget,
    draft: GreedyModel,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    draft_budget: int = 5,
    eos_token_ids: Iterable[int] = (),
    grammar: GrammarState | None = None,
) -> GreedyResult:
    """Generate a target-greedy continuation with unconstrained draft proposals.

    Drafting stops before its EOS choice; even an empty draft is verified by the
    target. Each round starts from the committed prefix and caps proposals at the
    remaining output budget. Only target-confirmed EOS terminates the request.
    Model vocabularies must have identical token meanings, not just equal sizes.

    A zero output budget validates inputs without calling either model. The
    result uses the same content-token and termination contract as greedy_decode.
    """
    vocab_size = vocabulary_size(target.vocab_size)
    prompt, eos = request_inputs(
        vocab_size, prompt_token_ids, max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        grammar_vocab_size=None if grammar is None else vocabulary_size(grammar.vocab_size),
    )
    if vocabulary_size(draft.vocab_size) != vocab_size:
        raise ValueError("draft and target vocabularies differ")
    if isinstance(draft_budget, bool) or not isinstance(draft_budget, int):
        raise TypeError("draft_budget must be an integer")
    if draft_budget <= 0:
        raise ValueError("draft_budget must be positive")
    generated: list[int] = []

    while len(generated) < max_new_tokens:
        if grammar is not None and grammar.is_terminated():
            return GreedyResult(tuple(generated), "grammar")
        prefix = prompt + tuple(generated)
        proposals: list[int] = []
        remaining = max_new_tokens - len(generated)
        for _ in range(min(draft_budget, remaining)):
            choice = draft.next_token(prefix + tuple(proposals))
            token = token_id(choice, vocab_size, "draft token")
            if token in eos:
                break
            proposals.append(token)

        choices = (
            target.verify_prefix(prefix, tuple(proposals)) if grammar is None
            else target.verify_prefix(prefix, tuple(proposals), grammar=grammar)
        )
        before = len(generated)
        try:
            for event in choices:
                token = token_id(event.token_id, vocab_size, "target token")
                if token in eos:
                    return GreedyResult(tuple(generated), "eos", token)
                generated.append(token)
                if grammar is not None and grammar.is_terminated():
                    return GreedyResult(tuple(generated), "grammar")
                if len(generated) == max_new_tokens:
                    return GreedyResult(tuple(generated), "length")
        finally:
            close = getattr(choices, "close", None)
            if close is not None:
                close()
        if len(generated) == before:
            raise RuntimeError("target verification made no progress")

    return GreedyResult(tuple(generated), "length")
