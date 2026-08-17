"""Target-only greedy decoding with explicit termination semantics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from ..grammar import GrammarState

from ._validation import request_inputs, token_id, vocabulary_size


class GreedyModel(Protocol):
    """Return the greedy next token for a complete, nonempty input prefix.

    Token IDs use one vocabulary throughout the request. Implementations select
    the lowest token ID on an exact score tie and reject NaN prediction scores.
    """

    @property
    def vocab_size(self) -> int: ...

    def next_token(self, prefix: tuple[int, ...]) -> int: ...


class ConstrainedGreedyModel(Protocol):
    """Select a valid token and advance the supplied grammar before returning."""

    @property
    def vocab_size(self) -> int: ...

    def next_token(self, prefix: tuple[int, ...], *, grammar: GrammarState) -> int: ...


@dataclass(frozen=True, slots=True)
class GreedyResult:
    """Generated content tokens, excluding the prompt and a selected EOS token."""

    token_ids: tuple[int, ...]
    stop_reason: Literal["eos", "length", "grammar"]
    eos_token_id: int | None = None


def greedy_decode(
    model: GreedyModel | ConstrainedGreedyModel,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    grammar: GrammarState | None = None,
) -> GreedyResult:
    """Generate at most ``max_new_tokens`` content tokens from one prompt.

    A selected EOS stops immediately and is recorded separately from content.
    Reaching the length limit does not trigger an extra forward pass to probe
    for EOS. A zero limit validates inputs but never calls the model.

    The caller tokenizes the prompt, including BOS or chat-template tokens when
    needed. Prompt EOS tokens are ordinary context, not a stop instruction.
    A constrained model advances grammar state once per choice. Sampling is
    outside this reference's scope.
    """
    vocab_size = vocabulary_size(model.vocab_size)
    prompt, eos = request_inputs(
        vocab_size, prompt_token_ids, max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        grammar_vocab_size=None if grammar is None else vocabulary_size(grammar.vocab_size),
    )
    generated: list[int] = []

    for _ in range(max_new_tokens):
        if grammar is not None and grammar.is_terminated():
            return GreedyResult(tuple(generated), "grammar")
        prefix = prompt + tuple(generated)
        choice = (
            cast(GreedyModel, model).next_token(prefix) if grammar is None
            else cast(ConstrainedGreedyModel, model).next_token(prefix, grammar=grammar)
        )
        token = token_id(choice, vocab_size, "model token")
        if token in eos:
            return GreedyResult(tuple(generated), "eos", token)
        generated.append(token)
        if grammar is not None and grammar.is_terminated():
            return GreedyResult(tuple(generated), "grammar")

    return GreedyResult(tuple(generated), "length")
