"""Optional XGrammar adapter with a private matcher and CPU mask workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..decoding._validation import token_id, vocabulary_size
from .state import GrammarTerminatedError

if TYPE_CHECKING:
    from xgrammar import CompiledGrammar


class XGrammarState:
    """Create fresh request state from an already compiled grammar.

    The compiled tokenizer vocabulary must use the model's token-ID meanings.
    By default XGrammar requires a stop token after matching the grammar body.
    Set terminate_without_stop_token only when that is the intended contract.
    """

    def __init__(
        self, compiled_grammar: CompiledGrammar, *, terminate_without_stop_token: bool = False
    ) -> None:
        import torch
        import xgrammar as xgr

        self._vocab_size = vocabulary_size(compiled_grammar.tokenizer_info.vocab_size)
        self._matcher = xgr.GrammarMatcher(
            compiled_grammar, terminate_without_stop_token=terminate_without_stop_token
        )
        self._mask = xgr.allocate_token_bitmask(1, self.vocab_size)
        self._ids = torch.arange(self.vocab_size, dtype=torch.int64)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def fork(self) -> XGrammarState:
        """Copy matcher state for one draft round; keep mask writes isolated."""
        branch = object.__new__(XGrammarState)
        branch._vocab_size = self._vocab_size
        branch._matcher = self._matcher.fork()
        branch._mask = self._mask.new_empty(self._mask.shape)
        # This index tensor is read-only; only matcher and mask storage mutate.
        branch._ids = self._ids
        return branch

    def is_terminated(self) -> bool:
        return self._matcher.is_terminated()

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.is_terminated():
            raise GrammarTerminatedError("grammar has terminated")
        if not self._matcher.fill_next_token_bitmask(self._mask):
            return tuple(range(self.vocab_size))
        # Arithmetic shift preserves the selected bit even for signed int32 words.
        # Indexing only vocabulary IDs excludes padding bits in the final word.
        valid = ((self._mask[0, self._ids // 32] >> (self._ids % 32)) & 1).bool()
        return tuple(self._ids[valid].tolist())

    def accept_token(self, token: int) -> bool:
        token = token_id(token, self.vocab_size, "grammar token")
        if self.is_terminated():
            raise GrammarTerminatedError("grammar has terminated")
        return self._matcher.accept_token(token)
