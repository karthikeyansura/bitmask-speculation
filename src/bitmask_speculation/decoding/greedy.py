"""Target-only greedy decoding with explicit termination semantics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ._validation import token_id, vocabulary_size


class GreedyModel(Protocol):
    """Return the greedy next token for a complete, nonempty input prefix.

    Token IDs use one vocabulary throughout the request. Implementations select
    the lowest token ID on an exact score tie and reject NaN prediction scores.
    """

    @property
    def vocab_size(self) -> int: ...

    def next_token(self, prefix: tuple[int, ...]) -> int: ...


@dataclass(frozen=True, slots=True)
class GreedyResult:
    """Generated content tokens, excluding the prompt and a selected EOS token."""

    token_ids: tuple[int, ...]
    stop_reason: Literal["eos", "length"]
    eos_token_id: int | None = None


def greedy_decode(
    model: GreedyModel,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
) -> GreedyResult:
    """Generate at most ``max_new_tokens`` content tokens from one prompt.

    A selected EOS stops immediately and is recorded separately from content.
    Reaching the length limit does not trigger an extra forward pass to probe
    for EOS. A zero limit validates inputs but never calls the model.

    The caller tokenizes the prompt, including BOS or chat-template tokens when
    needed. Prompt EOS tokens are ordinary context, not a stop instruction.
    Grammar constraints and sampling are outside this reference's scope.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    vocab_size = vocabulary_size(model.vocab_size)

    prompt = tuple(token_id(t, vocab_size, "prompt token") for t in prompt_token_ids)
    if not prompt:
        raise ValueError("prompt_token_ids must include at least one token")
    eos = frozenset(token_id(t, vocab_size, "EOS token") for t in eos_token_ids)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        prefix = prompt + tuple(generated)
        token = token_id(model.next_token(prefix), vocab_size, "model token")
        if token in eos:
            return GreedyResult(tuple(generated), "eos", token)
        generated.append(token)

    return GreedyResult(tuple(generated), "length")
