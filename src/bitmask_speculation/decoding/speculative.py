"""Request-level greedy speculation with target-controlled termination."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from ..grammar import ForkableGrammarState, GrammarState
from ._validation import request_inputs, token_id, vocabulary_size
from .accounting import ProposalAccounting, _RoundAccounting
from .greedy import ConstrainedGreedyModel, GreedyModel, GreedyResult
from .scheduling import DensityPolicy, RoundSchedule
from .verification import VerifiedToken


@dataclass(frozen=True, slots=True)
class SpeculativeResult:
    """Generation semantics and proposal accounting from the same request."""

    generation: GreedyResult
    accounting: ProposalAccounting
    schedule: tuple[RoundSchedule, ...] = ()

    def __post_init__(self) -> None:
        if self.schedule and len(self.schedule) != self.accounting.rounds:
            raise ValueError("schedule must contain one decision per round")
        if self.accounting.emitted != len(self.generation.token_ids):
            raise ValueError("emitted count must equal the content token count")

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.generation.token_ids

    @property
    def stop_reason(self) -> Literal["eos", "length", "grammar"]:
        return self.generation.stop_reason

    @property
    def eos_token_id(self) -> int | None:
        return self.generation.eos_token_id


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


def _draft_proposals(
    draft: GreedyModel | ConstrainedGreedyModel,
    prefix: tuple[int, ...],
    budget: int,
    eos: frozenset[int],
    vocab_size: int,
    grammar: ForkableGrammarState | None,
) -> tuple[int, ...]:
    # The branch goes out of scope before target verification, including when
    # drafting fails. No rollback count or speculative EOS reaches the parent.
    branch = None if grammar is None else grammar.fork()
    if grammar is not None:
        if branch is None or branch is grammar:
            raise ValueError("grammar.fork() must return independent state")
        if vocabulary_size(branch.vocab_size) != vocab_size:
            raise ValueError("forked grammar vocabulary differs")
    proposals: list[int] = []
    for _ in range(budget):
        if branch is not None and branch.is_terminated():
            break
        context = prefix + tuple(proposals)
        choice = (
            draft.next_token(context) if branch is None
            else draft.next_token(context, grammar=branch)
        )
        token = token_id(choice, vocab_size, "draft token")
        if token in eos:
            break
        proposals.append(token)
    return tuple(proposals)


def speculative_decode(
    target: VerificationTarget,
    draft: GreedyModel | ConstrainedGreedyModel,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    draft_budget: int = 5,
    eos_token_ids: Iterable[int] = (),
    grammar: GrammarState | None = None,
    constrain_draft: bool = False,
    draft_policy: DensityPolicy | None = None,
) -> SpeculativeResult:
    """Generate a target-greedy continuation with optional constrained proposals.

    Drafting stops before its EOS choice; even an empty draft is verified by the
    target. Each round starts from the committed prefix and caps proposals at the
    remaining output budget. Only target-confirmed EOS terminates the request.
    Model vocabularies must have identical token meanings, not just equal sizes.

    With constrain_draft=True, grammar must support independent forks and draft
    must advance its supplied grammar once per choice. Each round drafts on a
    fork; discarding it restores the committed state without mutating the parent.

    A draft_policy enables C4: sample density at each committed round boundary,
    choose K from the policy, and cap it at remaining output. It requires draft
    constraints and a grammar mask_density() query; draft_budget is then unused.

    A zero output budget validates inputs without calling either model. The
    generation uses the same contract as greedy_decode; accounting is separate.
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
    if not isinstance(constrain_draft, bool):
        raise TypeError("constrain_draft must be a boolean")
    if constrain_draft and not callable(getattr(grammar, "fork", None)):
        raise TypeError("constrained drafting requires a forkable grammar")
    draft_grammar = cast(ForkableGrammarState, grammar) if constrain_draft else None
    if draft_policy is not None:
        if not isinstance(draft_policy, DensityPolicy):
            raise TypeError("draft_policy must be a DensityPolicy")
        if not constrain_draft or not callable(getattr(grammar, "mask_density", None)):
            raise ValueError("density scheduling requires constrained drafts and mask_density()")
    schedule: list[RoundSchedule] = []
    generated: list[int] = []
    proposed = verified = accepted = rounds = 0

    def finish(
        reason: Literal["eos", "length", "grammar"], eos_token: int | None = None
    ) -> SpeculativeResult:
        return SpeculativeResult(
            GreedyResult(tuple(generated), reason, eos_token),
            ProposalAccounting(proposed, verified, accepted, len(generated), rounds),
            tuple(schedule),
        )

    while len(generated) < max_new_tokens:
        if grammar is not None and grammar.is_terminated():
            return finish("grammar")
        prefix = prompt + tuple(generated)
        remaining = max_new_tokens - len(generated)
        budget = draft_budget
        if draft_policy is not None:
            density = grammar.mask_density()
            budget = draft_policy.budget(density)
            schedule.append(RoundSchedule(density, budget, min(budget, remaining), len(generated)))
        proposals = _draft_proposals(
            draft, prefix, min(budget, remaining), eos, vocab_size, draft_grammar
        )

        proposed += len(proposals)
        rounds += 1
        round_accounting = _RoundAccounting(proposals)
        choices = (
            target.verify_prefix(prefix, proposals) if grammar is None
            else target.verify_prefix(prefix, proposals, grammar=grammar)
        )
        before = len(generated)
        try:
            for event in choices:
                token = token_id(event.token_id, vocab_size, "target token")
                compared, matched = round_accounting.record(event)
                verified += compared
                accepted += matched
                if token in eos:
                    return finish("eos", token)
                generated.append(token)
                if grammar is not None and grammar.is_terminated():
                    return finish("grammar")
                if len(generated) == max_new_tokens:
                    return finish("length")
        finally:
            close = getattr(choices, "close", None)
            if close is not None:
                close()
        if len(generated) == before:
            raise RuntimeError("target verification made no progress")
        if not round_accounting.finished:
            raise RuntimeError("verification ended before correction or bonus")

    return finish("length")
