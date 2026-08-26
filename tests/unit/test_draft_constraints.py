import pytest

from bitmask_speculation.decoding import speculative_decode, verify_prefix


class State:
    vocab_size = 4

    def __init__(self, tokens=(), limit=4):
        self.tokens = list(tokens)
        self.limit = limit
        self.forks = []

    def fork(self):
        branch = State(self.tokens, self.limit)
        self.forks.append(branch)
        return branch

    def is_terminated(self):
        return len(self.tokens) == self.limit

    def allowed_token_ids(self):
        assert not self.is_terminated()
        return (1, 2)

    def accept_token(self, token):
        if token not in self.allowed_token_ids():
            return False
        self.tokens.append(token)
        return True


class Model:
    vocab_size = 4

    def __init__(self, token=1):
        self.token = token
        self.calls = []

    def next_token(self, prefix, *, grammar):
        self.calls.append(tuple(prefix))
        assert grammar.accept_token(self.token)
        return self.token

    def verify_prefix(self, prefix, proposals, *, grammar):
        yield from verify_prefix(
            proposals, lambda i: self.next_token(prefix + proposals[:i], grammar=grammar),
            vocab_size=self.vocab_size,
        )


def run(state, *, draft=None, target=None, **kwargs):
    return speculative_decode(
        target or Model(), draft or Model(), [0], grammar=state,
        **({"constrain_draft": True, "max_new_tokens": 4} | kwargs),
    )


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_flag_requires_boolean(flag):
    with pytest.raises(TypeError, match="boolean"):
        run(State(), constrain_draft=flag, max_new_tokens=0)


def test_constrained_drafting_requires_forkable_state_even_at_zero_budget():
    state = State()
    state.fork = None
    for grammar in (None, state):
        with pytest.raises(TypeError, match="forkable grammar"):
            run(grammar, max_new_tokens=0)


def test_zero_budget_and_terminal_entry_do_not_fork_or_call_models():
    for limit, budget, reason in [(4, 0, "length"), (0, 4, "grammar")]:
        state, draft, target = State(limit=limit), Model(), Model()
        result = run(state, draft=draft, target=target, max_new_tokens=budget)
        assert result.stop_reason == reason
        assert not state.forks and not draft.calls and not target.calls


@pytest.mark.parametrize("returns_parent", [False, True])
def test_fork_cannot_be_missing_or_alias_parent(returns_parent):
    state = State()
    state.fork = lambda: state if returns_parent else None
    with pytest.raises(ValueError, match="independent state"):
        run(state)
    assert state.tokens == []


def test_fork_must_preserve_vocabulary():
    state, branch = State(), State()
    branch.vocab_size = 5
    state.fork = lambda: branch
    with pytest.raises(ValueError, match="forked grammar vocabulary"):
        run(state)
    assert state.tokens == [] and branch.tokens == []


def test_draft_exception_after_transition_leaves_parent_untouched():
    state, target, draft = State(tokens=(1,)), Model(), Model()

    def fail(prefix, *, grammar):
        assert grammar.accept_token(2)
        raise RuntimeError("draft failed after transition")

    draft.next_token = fail
    with pytest.raises(RuntimeError, match="draft failed"):
        run(state, draft=draft, target=target)
    assert state.tokens == [1] and state.allowed_token_ids() == (1, 2)
    assert state.forks[0].tokens == [1, 2]
    assert target.calls == []


@pytest.mark.parametrize("token", [-1, 4, True, 1.5])
def test_invalid_draft_output_does_not_change_parent(token):
    state, draft, target = State(), Model(), Model()
    draft.next_token = lambda prefix, *, grammar: token
    with pytest.raises((ValueError, TypeError)):
        run(state, draft=draft, target=target)
    assert state.tokens == [] and target.calls == []


def test_target_failure_preserves_only_consumed_committed_transitions():
    state, target = State(), Model()
    choose = target.next_token

    def fail(prefix, *, grammar):
        if grammar.tokens:
            raise RuntimeError("target failed")
        return choose(prefix, grammar=grammar)

    target.next_token = fail
    with pytest.raises(RuntimeError, match="target failed"):
        run(state, target=target)
    assert state.tokens == [1]
    assert state.forks[0].tokens == [1, 1, 1, 1]


def test_draft_eos_does_not_terminate_parent_or_count_as_proposal():
    state = State(limit=3)
    result = run(state, draft=Model(2), eos_token_ids=[2], max_new_tokens=9)
    assert result.token_ids == (1, 1, 1) and result.stop_reason == "grammar"
    assert state.tokens == [1, 1, 1]
    assert [branch.tokens for branch in state.forks] == [[2], [1, 2], [1, 1, 2]]
    assert result.accounting.proposed == result.accounting.verified == 0
    assert result.accounting.rounds == 3


def test_continue_existing_suffix_forks_at_current_state():
    state = State(tokens=(2, 1), limit=5)
    result = speculative_decode(
        Model(), Model(2), [0, 2, 1], grammar=state, constrain_draft=True,
        max_new_tokens=3, draft_budget=2,
    )
    assert result.token_ids == (1, 1, 1) and result.stop_reason == "grammar"
    assert state.tokens == [2, 1, 1, 1, 1]
    assert [branch.tokens for branch in state.forks] == [
        [2, 1, 2, 2], [2, 1, 1, 2, 2], [2, 1, 1, 1, 2],
    ]
