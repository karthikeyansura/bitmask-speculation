from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

from bitmask_speculation.decoding import greedy_decode, speculative_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState
from bitmask_speculation.models.transformers import TransformersGreedyModel


def compiled_grammar():
    info = xgr.TokenizerInfo(["a", "b", "<eos>", "z"], stop_token_ids=[2])
    return xgr.GrammarCompiler(info, max_threads=1).compile_grammar(
        'root ::= "aaaa" | "baab" | "abab" | "aabb" | "aaab"'
    )


class PositionModel(torch.nn.Module):
    def __init__(self, tokens):
        super().__init__()
        scores = torch.zeros(len(tokens), 4)
        for position, token in enumerate(tokens):
            scores[position, token] = 10
        # Invalid z wins without a grammar: test actual masking, not just kwargs.
        scores[:, 3] = 100
        self.register_buffer("scores", scores)
        self.config = SimpleNamespace(vocab_size=4, is_encoder_decoder=False)

    def forward(self, *, input_ids, use_cache, return_dict):
        return SimpleNamespace(logits=self.scores[:input_ids.shape[1]].unsqueeze(0))


def equivalent_state(actual, expected):
    assert actual.is_terminated() == expected.is_terminated()
    if not actual.is_terminated():
        assert actual.allowed_token_ids() == expected.allowed_token_ids()


@pytest.mark.parametrize("terminate_without_stop", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("mismatch", [0, 1, 2, 3, None])
def test_every_rejection_position_restores_state_and_matches_reference(
    mismatch, limit, terminate_without_stop,
):
    compiled = compiled_grammar()

    def fresh():
        return XGrammarState(compiled, terminate_without_stop_token=terminate_without_stop)

    paths = {0: [1, 0, 0, 1], 1: [0, 1, 0, 1], 2: [0, 0, 1, 1], 3: [0, 0, 0, 1], None: [0, 0, 0, 0]}
    draft = TransformersGreedyModel(PositionModel(paths[mismatch] + [2] * 10).eval())
    target = TransformersGreedyModel(PositionModel([0] * 4 + [2] * 10).eval())
    reference_state = fresh()
    reference = greedy_decode(target, [3], max_new_tokens=limit, eos_token_ids=[2], grammar=reference_state)
    committed = fresh()
    original_verify = target.verify_prefix
    rounds = []

    def inspect(prefix, proposals, *, grammar):
        rounds.append((prefix, proposals))
        expected = fresh()
        for token in prefix[1:]:
            assert expected.accept_token(token)
        # Drafting must not mutate parent even if the draft has terminated.
        equivalent_state(grammar, expected)
        stream = original_verify(prefix, proposals, grammar=grammar)
        try:
            for event in stream:
                assert expected.accept_token(event.token_id)
                equivalent_state(grammar, expected)
                yield event
        finally:
            stream.close()

    target.verify_prefix = inspect
    result = speculative_decode(
        target, draft, [3], grammar=committed, constrain_draft=True,
        max_new_tokens=limit, draft_budget=4, eos_token_ids=[2],
    )
    assert result.generation == reference
    equivalent_state(committed, reference_state)
    assert rounds[0][1] == tuple(paths[mismatch][:min(limit, 4)])
    assert result.accounting.accepted <= result.accounting.verified <= result.accounting.proposed
    if mismatch is None and limit <= 4:
        assert result.accounting.rounds == 1
        assert result.accounting.accepted == result.accounting.proposed == limit


def test_matcher_forks_isolate_transitions_masks_and_nested_branches():
    state = XGrammarState(compiled_grammar())
    assert state.accept_token(0)
    branch = state.fork()
    assert branch._mask.data_ptr() != state._mask.data_ptr()
    assert branch.accept_token(1)
    assert branch.allowed_token_ids() == (0,)
    assert state.allowed_token_ids() == (0, 1)
    nested = branch.fork()
    assert nested.accept_token(0) and nested.accept_token(1) and nested.accept_token(2)
    assert nested.is_terminated() and nested.fork().is_terminated()
    assert branch.allowed_token_ids() == (0,)
    assert state.accept_token(0)
    assert not branch.accept_token(1)
    assert branch.allowed_token_ids() == (0,)
    assert state.allowed_token_ids() == (0, 1)
