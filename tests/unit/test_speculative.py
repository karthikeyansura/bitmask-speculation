from itertools import product

import pytest

from bitmask_speculation.decoding import greedy_decode, speculative_decode, verify_prefix


class FunctionModel:
    vocab_size = 8

    def __init__(self, rule):
        self.rule = rule
        self.prefixes = []

    def next_token(self, prefix):
        self.prefixes.append(prefix)
        return self.rule(prefix)


class FunctionTarget(FunctionModel):
    def __init__(self, rule):
        super().__init__(rule)
        self.rounds = []
        self.closed = 0

    def next_token(self, prefix, *, grammar=None):
        choice = super().next_token(prefix)
        if grammar is not None:
            allowed = grammar.allowed_token_ids()
            choice = choice if choice in allowed else min(allowed)
            assert grammar.accept_token(choice)
        return choice

    def verify_prefix(self, prefix, proposals, *, grammar=None):
        self.rounds.append((prefix, proposals))
        choices = verify_prefix(
            proposals,
            lambda i: self.next_token(prefix + proposals[:i], grammar=grammar),
            vocab_size=self.vocab_size,
        )
        try:
            yield from choices
        finally:
            choices.close()
            self.closed += 1


def linear(tokens):
    def rule(prefix):
        return tokens[len(prefix) - 1]
    return rule


@pytest.mark.parametrize("budget", [1, 2, 5, 99])
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 7])
def test_output_limit_caps_drafting_and_never_selects_bonus(budget, limit):
    target = FunctionTarget(lambda prefix: 1)
    draft = FunctionModel(lambda prefix: 1)
    result = speculative_decode(target, draft, [0], max_new_tokens=limit, draft_budget=budget)
    assert result.token_ids == (1,) * limit
    assert result.stop_reason == "length"
    assert len(target.prefixes) == limit
    assert all(len(proposals) <= min(budget, limit - len(prefix) + 1)
               for prefix, proposals in target.rounds)
    assert target.closed == len(target.rounds)
    if limit == 0:
        assert draft.prefixes == [] and target.rounds == []


@pytest.mark.parametrize("eos_position", range(5))
@pytest.mark.parametrize("draft_kind", ["matches", "mismatches", "eos"])
def test_target_eos_at_every_position_and_draft_behavior(eos_position, draft_kind):
    tokens = [1] * eos_position + [7]
    target = FunctionTarget(linear(tokens))
    draft_rule = {
        "matches": lambda prefix: tokens[min(len(prefix) - 1, eos_position)],
        "mismatches": lambda prefix: 2,
        "eos": lambda prefix: 7,
    }[draft_kind]
    result = speculative_decode(
        target, FunctionModel(draft_rule), [0], max_new_tokens=12,
        draft_budget=3, eos_token_ids=[6, 7],
    )
    assert result.token_ids == (1,) * eos_position
    assert result.stop_reason == "eos" and result.eos_token_id == 7
    assert len(target.prefixes) == eos_position + 1
    assert target.closed == len(target.rounds)


def test_empty_draft_progresses_until_target_eos():
    target = FunctionTarget(linear([2, 3, 7]))
    result = speculative_decode(
        target, FunctionModel(lambda prefix: 7), [0], max_new_tokens=8, eos_token_ids=[7]
    )
    assert result.token_ids == (2, 3) and result.stop_reason == "eos"
    assert target.rounds == [((0,), ()), ((0, 2), ()), ((0, 2, 3), ())]


def test_rejection_restarts_draft_from_corrected_prefix():
    target = FunctionTarget(lambda prefix: (sum(prefix) + 1) % 8)
    draft = FunctionModel(lambda prefix: 5)
    prompt = [0]
    result = speculative_decode(target, draft, prompt, max_new_tokens=4, draft_budget=3)
    reference = greedy_decode(FunctionModel(target.rule), prompt, max_new_tokens=4)
    assert result.generation == reference
    assert draft.prefixes[:4] == [(0,), (0, 5), (0, 5, 5), (0, 1)]
    assert prompt == [0]


def test_eos_on_last_allowed_selection_wins_over_length():
    result = speculative_decode(
        FunctionTarget(linear([1, 7])), FunctionModel(lambda prefix: 2),
        [0], max_new_tokens=2, eos_token_ids=[7],
    )
    assert result.token_ids == (1,) and result.stop_reason == "eos"


def test_eos_after_limit_is_not_probed():
    target = FunctionTarget(linear([1, 7]))
    result = speculative_decode(target, FunctionModel(lambda prefix: 1), [0], max_new_tokens=1)
    assert result.token_ids == (1,) and result.stop_reason == "length"
    assert target.prefixes == [(0,)]


class PathGrammar:
    vocab_size = 8

    def __init__(self, path):
        self.path = path
        self.accepted = []

    def is_terminated(self):
        return len(self.accepted) == len(self.path)

    def allowed_token_ids(self):
        assert not self.is_terminated(), "no row selection after terminal state"
        return [self.path[len(self.accepted)]]

    def accept_token(self, token):
        assert token == self.path[len(self.accepted)]
        self.accepted.append(token)
        return True


@pytest.mark.parametrize("path", [[], [1], [1, 2], [1, 2, 3]])
@pytest.mark.parametrize("draft_token", [1, 2, 7])
def test_grammar_terminal_content_and_no_post_terminal_selection(path, draft_token):
    state = PathGrammar(path)
    target, draft = FunctionTarget(lambda prefix: 0), FunctionModel(lambda prefix: draft_token)
    result = speculative_decode(
        target, draft, [0], max_new_tokens=8, draft_budget=2, eos_token_ids=[7], grammar=state
    )
    assert result.token_ids == tuple(path) and result.stop_reason == "grammar"
    assert state.accepted == path and len(target.prefixes) == len(path)
    assert target.closed == len(target.rounds)
    if not path:
        assert draft.prefixes == [] and target.rounds == []


def test_budget_stops_grammar_at_exact_emitted_prefix():
    state = PathGrammar([1, 1, 1])
    result = speculative_decode(
        FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1),
        [0], max_new_tokens=2, draft_budget=5, grammar=state,
    )
    assert result.stop_reason == "length" and result.token_ids == (1, 1)
    assert state.accepted == [1, 1] and not state.is_terminated()


def test_grammar_termination_on_length_boundary_and_eos_precedence():
    for eos, expected in [([], "grammar"), ([1], "eos")]:
        state = PathGrammar([1])
        result = speculative_decode(
            FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1),
            [0], max_new_tokens=1, grammar=state, eos_token_ids=eos,
        )
        assert result.stop_reason == expected and state.accepted == [1]
        assert result.token_ids == (() if eos else (1,))


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_draft_budget_before_calls(budget):
    target, draft = FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1)
    with pytest.raises((ValueError, TypeError)):
        speculative_decode(target, draft, [0], max_new_tokens=0, draft_budget=budget)
    assert target.rounds == [] and draft.prefixes == []


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": -1}, {"max_new_tokens": True}, {"max_new_tokens": 1.5},
    {"eos_token_ids": [8]}, {"eos_token_ids": [False]},
])
def test_invalid_request_before_calls(kwargs):
    target, draft = FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1)
    with pytest.raises((ValueError, TypeError)):
        speculative_decode(target, draft, [0], **({"max_new_tokens": 0} | kwargs))
    assert target.rounds == [] and draft.prefixes == []


@pytest.mark.parametrize("prompt", [[], [-1], [8], [False], [1.5]])
def test_invalid_prompt_before_calls(prompt):
    target, draft = FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1)
    with pytest.raises((ValueError, TypeError)):
        speculative_decode(target, draft, prompt, max_new_tokens=0)
    assert target.rounds == [] and draft.prefixes == []


@pytest.mark.parametrize("vocab", [0, -1, True, 8.0, 9])
def test_invalid_or_mismatched_draft_vocabulary(vocab):
    target, draft = FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1)
    draft.vocab_size = vocab
    with pytest.raises((ValueError, TypeError)):
        speculative_decode(target, draft, [0], max_new_tokens=0)
    assert target.rounds == [] and draft.prefixes == []


@pytest.mark.parametrize("token", [-1, 8, True, 1.5])
def test_invalid_draft_output_is_rejected_before_verification(token):
    target = FunctionTarget(lambda prefix: 1)
    with pytest.raises((ValueError, TypeError)):
        speculative_decode(target, FunctionModel(lambda prefix: token), [0], max_new_tokens=1)
    assert target.rounds == []


def test_empty_verifier_fails_instead_of_looping():
    target = FunctionTarget(lambda prefix: 1)
    target.verify_prefix = lambda prefix, proposals: iter(())
    with pytest.raises(RuntimeError, match="no progress"):
        speculative_decode(target, FunctionModel(lambda prefix: 1), [0], max_new_tokens=1)


def test_verifier_closes_on_selection_failure():
    def fail(prefix):
        raise RuntimeError("selection failed")
    target = FunctionTarget(fail)
    with pytest.raises(RuntimeError, match="selection failed"):
        speculative_decode(target, FunctionModel(lambda prefix: 1), [0], max_new_tokens=2)
    assert target.closed == 1


def test_deterministic_request_matrix_matches_greedy():
    for prefix, limit, budget, draft_token, eos in product(
        ([0], [1, 2]), range(9), (1, 2, 4), range(4), ((), (3,)),
    ):
        rule = lambda prefix: (sum(prefix) + len(prefix)) % 4
        target, draft = FunctionTarget(rule), FunctionModel(lambda prefix: draft_token)
        result = speculative_decode(
            target, draft, prefix, max_new_tokens=limit, draft_budget=budget, eos_token_ids=eos
        )
        reference = greedy_decode(FunctionModel(rule), prefix, max_new_tokens=limit, eos_token_ids=eos)
        assert result.generation == reference
        assert target.closed == len(target.rounds)


@pytest.mark.parametrize("vocab", [None, 0, True, 9])
def test_invalid_grammar_vocabulary_matches_reference_validation(vocab):
    target, draft, state = FunctionTarget(lambda prefix: 1), FunctionModel(lambda prefix: 1), PathGrammar([1])
    state.vocab_size = vocab
    with pytest.raises((ValueError, TypeError)):
        speculative_decode(target, draft, [0], max_new_tokens=0, grammar=state)
    with pytest.raises((ValueError, TypeError)):
        greedy_decode(target, [0], max_new_tokens=0, grammar=state)
    assert target.prefixes == [] and target.rounds == [] and draft.prefixes == []


def test_invalid_target_event_closes_iterator():
    from bitmask_speculation.decoding import VerifiedToken

    target = FunctionTarget(lambda prefix: 1)
    closed = []

    def invalid(prefix, proposals):
        try:
            yield VerifiedToken(8, "correction")
        finally:
            closed.append(True)

    target.verify_prefix = invalid
    with pytest.raises(ValueError, match="target token"):
        speculative_decode(target, FunctionModel(lambda prefix: 1), [0], max_new_tokens=1)
    assert closed == [True]
