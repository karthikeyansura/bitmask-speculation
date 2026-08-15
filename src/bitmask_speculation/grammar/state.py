"""The state required for target-side constrained selection."""

from collections.abc import Sequence
from typing import Protocol


class GrammarTerminatedError(ValueError):
    """A token was requested after the grammar had terminated."""


class GrammarState(Protocol):
    """One request's generated-suffix state, never the prompt or draft tail.

    Reading allowed IDs must not advance state. A failed acceptance must return
    False without advancing state. Do not share a state across active requests.
    """

    @property
    def vocab_size(self) -> int: ...

    def is_terminated(self) -> bool: ...

    def allowed_token_ids(self) -> Sequence[int]: ...

    def accept_token(self, token: int) -> bool: ...
