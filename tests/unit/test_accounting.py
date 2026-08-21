from dataclasses import FrozenInstanceError, asdict
from itertools import product

import pytest

from bitmask_speculation.decoding import (
    GreedyResult, ProposalAccounting, SpeculativeResult, VerifiedToken,
    speculative_decode, verify_prefix,
)


class Draft:
    vocab_size = 32

    def __init__(self, tokens):
        self.tokens = tokens

    def next_token(self, prefix):
        return self.tokens[(len(prefix) - 1) % len(self.tokens)]


class State:
    vocab_size = 32

    def __init__(self, length):
        self.length = length
        self.accepted = []

    def is_terminated(self):
        return len(self.accepted) == self.length

    def accept_token(self, token):
        self.accepted.append(token)
        return True


class Target:
    vocab_size = 32

    def __init__(self, tokens):
        self.tokens = tokens
        self.observations = []

    def verify_prefix(self, prefix, proposals, *, grammar=None):
        observed = []
        self.observations.append((proposals, observed))

        def select(i):
            token = self.tokens[(len(prefix) - 1 + i) % len(self.tokens)]
            if grammar is not None:
                assert grammar.accept_token(token)
            return token

        stream = verify_prefix(proposals, select, vocab_size=self.vocab_size)
        try:
            for event in stream:
                observed.append(event.token_id)
                yield event
        finally:
            stream.close()


def observed_counts(target, emitted):
    # Derive comparisons from lengths and token equality, not event labels.
    proposed = verified = accepted = 0
    for proposals, selections in target.observations:
        proposed += len(proposals)
        verified += min(len(proposals), len(selections))
        for draft, target_token in zip(proposals, selections):
            if draft != target_token:
                break
            accepted += 1
    return ProposalAccounting(proposed, verified, accepted, emitted, len(target.observations))


@pytest.mark.parametrize("draft,target,budget,limit,terminal,expected", [
    # The five scenarios from docs/prefix-acceptance.md.
    ([10, 12, 14, 15], [10, 12, 13], 4, 8, 3, (4, 3, 2, 3, 1)),
    ([10, 12, 13], [10, 12, 13, 15], 3, 8, 4, (3, 3, 3, 4, 1)),
    ([11, 12, 14, 15], [10], 4, 8, 1, (4, 1, 0, 1, 1)),
    ([10, 12], [10, 12], 4, 2, 4, (2, 2, 2, 2, 1)),
    ([31], [10], 4, 8, 1, (0, 0, 0, 1, 1)),
])
def test_walkthrough_counts(draft, target, budget, limit, terminal, expected):
    target_model = Target(target)
    result = speculative_decode(
        target_model, Draft(draft), [0], max_new_tokens=limit, draft_budget=budget,
        eos_token_ids=[31], grammar=State(terminal),
    )
    assert result.accounting == ProposalAccounting(*expected)
    assert result.accounting == observed_counts(target_model, len(result.token_ids))


@pytest.mark.parametrize("draft,target,budget,expected", [
    ([2, 2, 2, 2], [31], 4, (4, 1, 0, 0, 1)),  # Correction EOS.
    ([1, 2], [1, 2, 31], 2, (2, 2, 2, 2, 1)),  # Bonus EOS.
    ([31], [31], 4, (0, 0, 0, 0, 1)),          # Fallback EOS.
    ([31], [1, 2, 31], 4, (0, 0, 0, 2, 3)),    # Repeated empty drafts.
])
def test_eos_counts_comparison_but_never_emission(draft, target, budget, expected):
    result = speculative_decode(
        Target(target), Draft(draft), [0], max_new_tokens=8,
        draft_budget=budget, eos_token_ids=[31],
    )
    assert result.stop_reason == "eos"
    assert result.accounting == ProposalAccounting(*expected)


def test_zero_budget_and_initial_terminal_have_no_rounds():
    for limit, state in ((0, State(2)), (4, State(0))):
        result = speculative_decode(Target([1]), Draft([1]), [0], max_new_tokens=limit, grammar=state)
        assert result.accounting == ProposalAccounting()


def test_rates_use_different_denominators_and_preserve_undefined():
    counts = ProposalAccounting(4, 3, 2, 3, 1)
    assert counts.acceptance_rate == pytest.approx(2 / 3)
    assert counts.proposal_yield == 0.5
    assert ProposalAccounting().acceptance_rate is None
    assert ProposalAccounting().proposal_yield is None
    assert ProposalAccounting(4, 1, 0, 1, 1).acceptance_rate == 0.0
    assert ProposalAccounting(4, 0, 0).proposal_yield == 0.0
    assert ProposalAccounting(4, 0, 0).acceptance_rate is None


def test_results_are_immutable_and_compare_accounting():
    generation = GreedyResult((1,), "length")
    first = SpeculativeResult(generation, ProposalAccounting(1, 1, 1, 1, 1))
    second = SpeculativeResult(generation, ProposalAccounting(4, 1, 0, 1, 1))
    assert first.generation == second.generation and first != second
    assert first.token_ids == (1,) and first.stop_reason == "length" and first.eos_token_id is None
    assert asdict(first)["accounting"]["emitted"] == 1
    with pytest.raises(FrozenInstanceError):
        first.accounting.accepted = 4
    with pytest.raises(ValueError, match="emitted"):
        SpeculativeResult(generation, ProposalAccounting())


@pytest.mark.parametrize("field", ["proposed", "verified", "accepted", "emitted", "rounds"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_counter_values(field, value):
    with pytest.raises((ValueError, TypeError)):
        ProposalAccounting(**{field: value})


@pytest.mark.parametrize("values", [(1, 2, 0), (2, 1, 2)])
def test_counter_order_invariant(values):
    with pytest.raises(ValueError, match="accepted <= verified <= proposed"):
        ProposalAccounting(*values)


def test_request_matrix_matches_independent_observations():
    for draft, target, limit, budget in product(
        ([1, 2, 3], [1, 4, 3], [4], [31]),
        ([1, 2, 3], [1, 2, 31]), range(8), (1, 2, 5),
    ):
        target_model = Target(target)
        result = speculative_decode(
            target_model, Draft(draft), [0], max_new_tokens=limit,
            draft_budget=budget, eos_token_ids=[31],
        )
        assert result.accounting == observed_counts(target_model, len(result.token_ids))


@pytest.mark.parametrize("events", [
    [VerifiedToken(2, "accepted")],                # Mismatched token called accepted.
    [VerifiedToken(1, "correction")],              # Matching token called correction.
    [VerifiedToken(1, "bonus")],                   # Premature bonus.
    [VerifiedToken(1, "fallback")],                # Nonempty proposal.
    [VerifiedToken(1, "unknown")],                 # Unsupported label.
    [VerifiedToken(1, "accepted"), VerifiedToken(1, "accepted")],
    [VerifiedToken(2, "correction"), VerifiedToken(1, "bonus")],
])
def test_malformed_event_stream_cannot_report_counts(events):
    target = Target([1])
    target.verify_prefix = lambda prefix, proposals: iter(events)
    with pytest.raises(ValueError):
        speculative_decode(target, Draft([1]), [0], max_new_tokens=8, draft_budget=1)


def test_incomplete_verifier_does_not_report_success():
    target = Target([1])
    target.verify_prefix = lambda prefix, proposals: iter([VerifiedToken(1, "accepted")])
    with pytest.raises(RuntimeError, match="before correction or bonus"):
        speculative_decode(target, Draft([1]), [0], max_new_tokens=8, draft_budget=2)
