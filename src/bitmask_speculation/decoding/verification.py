"""Longest-prefix verification independent of model and grammar backends."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from ._validation import token_id, vocabulary_size


@dataclass(frozen=True, slots=True)
class VerifiedToken:
    """One target-confirmed choice, before request-level termination handling."""

    token_id: int
    kind: Literal["accepted", "correction", "bonus", "fallback"]


def verify_prefix(
    proposals: Sequence[int],
    target_token_at: Callable[[int], int],
    *,
    vocab_size: int,
) -> Iterator[VerifiedToken]:
    """Yield matching proposals followed by one correction or bonus choice.

    ``target_token_at(i)`` selects after the committed prefix plus proposals
    before index ``i``. Index ``len(proposals)`` selects the bonus. Calls are
    ordered and lazy: no later choice is requested after a mismatch or when
    the consumer stops. Empty proposals request one target-only fallback.

    Proposal IDs are validated and copied before returning the iterator. The
    caller handles EOS, output limits and grammar state between yielded tokens;
    this primitive has no special-token knowledge and does not emit text.
    """
    vocab_size = vocabulary_size(vocab_size)
    draft = tuple(token_id(t, vocab_size, "proposal token") for t in proposals)

    def choices() -> Iterator[VerifiedToken]:
        for i, proposal in enumerate(draft):
            target = token_id(target_token_at(i), vocab_size, "target token")
            if proposal != target:
                yield VerifiedToken(target, "correction")
                return
            yield VerifiedToken(proposal, "accepted")
        target = token_id(target_token_at(len(draft)), vocab_size, "target token")
        yield VerifiedToken(target, "bonus" if draft else "fallback")

    return choices()
