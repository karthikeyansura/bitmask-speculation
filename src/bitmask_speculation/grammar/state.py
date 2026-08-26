"""Request and proposal state for constrained token selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class GrammarTerminatedError(ValueError):
    """A token was requested after the grammar had terminated."""


class GrammarState(Protocol):
    """Grammar state for a committed suffix or an isolated proposal branch.

    Reading allowed IDs must not advance state. A failed acceptance must return
    False without advancing state. Do not share a state across active requests.
    """

    @property
    def vocab_size(self) -> int: ...

    def is_terminated(self) -> bool: ...

    def allowed_token_ids(self) -> Sequence[int]: ...

    def accept_token(self, token: int) -> bool: ...


class ForkableGrammarState(GrammarState, Protocol):
    """A state that can create independent proposal branches.

    fork() must preserve the current state without advancing it. The returned
    state has identical token meanings, allowed IDs and termination status, but
    independently mutable transitions and mask storage. Discarding a branch is
    the rollback operation; target verification alone advances the parent.
    """

    def fork(self) -> ForkableGrammarState: ...
