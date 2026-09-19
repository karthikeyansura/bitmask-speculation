"""Reusable request facade over the reference decoding functions."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .decoding import (
    DensityPolicy, GreedyResult, ProposalAccounting, RoundSchedule,
    greedy_decode, speculative_decode,
)
from .decoding._validation import vocabulary_size
from .grammar import GrammarState

MODES = ("target_unconstrained", "target_constrained", "c1", "c2", "c3", "c4")
CONSTRAINED_MODES = frozenset({"target_constrained", "c2", "c3", "c4"})


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generation: GreedyResult
    accounting: ProposalAccounting | None = None
    schedule: tuple[RoundSchedule, ...] = ()
    text: str | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.generation.token_ids

    @property
    def stop_reason(self) -> str:
        return self.generation.stop_reason

    @property
    def eos_token_id(self) -> int | None:
        return self.generation.eos_token_id


class DecodingEngine:
    """Reuse loaded models; allocate grammar state for each request.

    Models must share token-ID meanings. The caller's grammar_factory must return
    fresh state. No model loading, device selection, timing or native dispatch
    occurs here. Calls use the established decoding loops unchanged.
    """

    def __init__(self, target, draft=None):
        self.target, self.draft = target, draft
        size = vocabulary_size(target.vocab_size)
        if draft is not None and vocabulary_size(draft.vocab_size) != size:
            raise ValueError("draft and target vocabularies differ")

    def generate(
        self, prompt_token_ids: Sequence[int], *, max_new_tokens: int,
        mode: str = "target_unconstrained", eos_token_ids: Iterable[int] = (),
        grammar_factory: Callable[[], GrammarState] | None = None,
        draft_budget: int = 5, draft_policy: DensityPolicy | None = None,
    ) -> GenerationResult:
        if mode not in MODES:
            raise ValueError("unknown decoding mode")
        constrained = mode in CONSTRAINED_MODES
        if constrained != (grammar_factory is not None):
            raise ValueError("grammar_factory is required exactly for constrained modes")
        if draft_policy is not None and mode != "c4":
            raise ValueError("draft_policy is only used by c4")
        if mode.startswith("c") and self.draft is None:
            raise ValueError("speculative modes require a draft model")
        grammar = None if grammar_factory is None else grammar_factory()
        if constrained and grammar is None:
            raise ValueError("grammar_factory returned no state")
        common = dict(max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids, grammar=grammar)
        if mode.startswith("target_"):
            return GenerationResult(greedy_decode(self.target, prompt_token_ids, **common))
        result = speculative_decode(
            self.target, self.draft, prompt_token_ids, **common,
            draft_budget=draft_budget, constrain_draft=mode in {"c3", "c4"},
            draft_policy=(draft_policy or DensityPolicy()) if mode == "c4" else None,
        )
        return GenerationResult(result.generation, result.accounting, result.schedule)
