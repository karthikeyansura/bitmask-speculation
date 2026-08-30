import math

import pytest

from bitmask_speculation.decoding import DensityPolicy, RoundSchedule, speculative_decode, verify_prefix


@pytest.mark.parametrize("density,expected", [
    (0, 8), (math.nextafter(.005, 0), 8), (.005, 4),
    (math.nextafter(.02, 0), 4), (.02, 1), (1, 1),
])
def test_threshold_boundaries(density, expected):
    assert DensityPolicy().budget(density) == expected


@pytest.mark.parametrize("density", [math.nan, math.inf, -math.inf, -0.1, 1.1, True, None, "0.1"])
def test_invalid_density(density):
    with pytest.raises((ValueError, TypeError)):
        DensityPolicy().budget(density)


@pytest.mark.parametrize("options", [
    {"low_density": .03}, {"low_density": .02}, {"high_density": 2},
    {"low_density": math.nan}, {"high_density": True},
    {"min_budget": 0}, {"max_budget": -1}, {"min_budget": 9},
    {"max_budget": True}, {"min_budget": 1.5},
])
def test_invalid_policy(options):
    with pytest.raises((ValueError, TypeError)):
        DensityPolicy(**options)


def test_custom_bounds_and_constant_budget():
    assert [DensityPolicy(.1, .8, 2, 7).budget(d) for d in [0, .1, .8, 1]] == [7, 4, 2, 2]
    assert all(DensityPolicy(0, 1, 3, 3).budget(d) == 3 for d in [0, .3, 1])


class State:
    vocab_size = 4

    def __init__(self, tokens=(), limit=3):
        self.tokens = list(tokens)
        self.limit = limit
        self.queries = []

    def fork(self):
        return State(self.tokens, self.limit)

    def mask_density(self):
        self.queries.append(tuple(self.tokens))
        return [0, .01, 1][len(self.tokens) % 3]

    def is_terminated(self):
        return len(self.tokens) == self.limit

    def allowed_token_ids(self):
        return (1, 2)

    def accept_token(self, token):
        assert not self.is_terminated() and token in self.allowed_token_ids()
        self.tokens.append(token)
        return True


class Model:
    vocab_size = 4

    def __init__(self, token=1):
        self.token = token
        self.prefixes = []

    def next_token(self, prefix, *, grammar):
        self.prefixes.append(prefix)
        assert grammar.accept_token(self.token)
        return self.token

    def verify_prefix(self, prefix, proposals, *, grammar):
        yield from verify_prefix(proposals, lambda i: self.next_token(prefix + proposals[:i], grammar=grammar), vocab_size=4)


def run(state, **kwargs):
    return speculative_decode(Model(), Model(2), [0], grammar=state, **({
        "constrain_draft": True, "max_new_tokens": 3, "draft_policy": DensityPolicy(),
    } | kwargs))


def test_schedule_reads_corrected_prefix_once_per_round_and_caps_budget():
    state = State()
    result = run(state)
    assert result.token_ids == (1, 1, 1)
    assert state.queries == [(), (1,), (1, 1)]
    assert result.schedule == (
        RoundSchedule(0, 8, 3, 0), RoundSchedule(.01, 4, 2, 1), RoundSchedule(1, 1, 1, 2),
    )
    assert result.accounting.rounds == 3
    assert result.accounting.proposed == 6 and result.accounting.verified == 3


def test_all_valid_uses_policy_minimum_not_fixed_budget():
    state = State()
    state.mask_density = lambda: 1
    result = run(state, draft_budget=99, draft_policy=DensityPolicy(min_budget=2))
    assert all(s.requested_budget == 2 for s in result.schedule)


def test_fixed_budget_has_no_density_queries_or_trace():
    state = State()
    result = run(state, draft_policy=None)
    assert not result.schedule and not state.queries


@pytest.mark.parametrize("limit,budget", [(0, 3), (3, 0)])
def test_no_density_query_on_terminal_entry_or_zero_budget(limit, budget):
    state = State(limit=limit)
    result = run(state, max_new_tokens=budget)
    assert not state.queries and not result.schedule and result.accounting.rounds == 0


def test_missing_capabilities_or_wrong_policy_fail_at_entry():
    state = State()
    with pytest.raises(ValueError, match="constrained drafts"):
        run(state, constrain_draft=False)
    state.mask_density = None
    with pytest.raises(ValueError, match="mask_density"):
        run(state, max_new_tokens=0)
    with pytest.raises(TypeError, match="DensityPolicy"):
        run(State(), draft_policy="adaptive")


def test_invalid_density_fails_before_draft_or_target_transitions():
    state = State()
    state.mask_density = lambda: math.nan
    with pytest.raises(ValueError, match="density"):
        run(state)
    assert state.tokens == []


@pytest.mark.parametrize("args", [(0, 1, 2, 0), (0, 0, 0, 0), (0, 1, 1, -1), (0, 1, 1, True)])
def test_invalid_trace_fields(args):
    with pytest.raises((TypeError, ValueError)):
        RoundSchedule(*args)
