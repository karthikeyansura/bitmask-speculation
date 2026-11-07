from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.grammar import GrammarTerminatedError
from bitmask_speculation.grammar.selection import masked_argmax
from bitmask_speculation.models.transformers import TransformersGreedyModel


@pytest.mark.parametrize(
    "scores,allowed,expected",
    [
        ([99.0, 2.0, 2.0], [2, 1], 1),
        ([99.0, -float("inf"), -float("inf")], [2, 1], 1),
        ([float("inf"), 2.0, 3.0], [2, 1], 2),
        ([0.0, float("inf"), float("inf")], [2, 1, 2], 1),
    ],
)
def test_masked_argmax_ties_and_infinities(scores, allowed, expected):
    scores = torch.tensor(scores)
    original = scores.clone()
    assert masked_argmax(scores, allowed) == expected
    torch.testing.assert_close(scores, original)


@pytest.mark.parametrize("allowed", [[], [-1], [3], [True], [1.5]])
def test_invalid_allowed_ids(allowed):
    with pytest.raises((ValueError, TypeError)):
        masked_argmax(torch.zeros(3), allowed)


@pytest.mark.parametrize("scores", [torch.zeros(2, 3), torch.tensor([]), [1.0]])
def test_invalid_score_shapes(scores):
    with pytest.raises(ValueError):
        masked_argmax(scores, [0])


def test_integer_scores_fail():
    with pytest.raises(TypeError):
        masked_argmax(torch.tensor([1, 2]), [0])


@pytest.mark.parametrize("allowed", [[0], [1]])
def test_nan_fails_even_at_invalid_position(allowed):
    with pytest.raises(ValueError, match="NaN"):
        masked_argmax(torch.tensor([float("nan"), 1.0]), allowed)


class PathGrammar:
    vocab_size = 5

    def __init__(self, path=(1, 2, 3), reject=False):
        self.path = path
        self.accepted = []
        self.reject = reject
        self.mask_reads = 0

    def is_terminated(self):
        return len(self.accepted) == len(self.path)

    def allowed_token_ids(self):
        assert not self.is_terminated()
        self.mask_reads += 1
        return [self.path[len(self.accepted)]]

    def accept_token(self, token):
        if self.reject or token != self.path[len(self.accepted)]:
            return False
        self.accepted.append(token)
        return True


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scores", torch.tensor([99.0, 4.0, 3.0, 2.0, 1.0]))
        self.config = SimpleNamespace(vocab_size=5, is_encoder_decoder=False)
        self.calls = 0

    def forward(self, *, input_ids, use_cache, return_dict):
        self.calls += 1
        return SimpleNamespace(logits=self.scores.expand(1, input_ids.shape[1], -1))


def adapter():
    return TransformersGreedyModel(ConstantModel().eval(), target_cache=False)


@pytest.mark.parametrize("draft", [[4, 4], [1, 4, 4], [1, 2], [1, 2, 3, 4], []])
def test_transitions_match_reference_and_rejected_tail_is_untouched(draft):
    target = adapter()
    grammar = PathGrammar()
    events = list(target.verify_prefix([0], draft, grammar=grammar))
    tokens = tuple(e.token_id for e in events)
    reference = greedy_decode(adapter(), [0], max_new_tokens=len(tokens), grammar=PathGrammar())
    assert tokens == reference.token_ids
    assert grammar.accepted == list(tokens)
    assert grammar.mask_reads == len(tokens)
    assert target._model.calls == 1
    assert tokens == (1, 2, 3)[:len(tokens)]


def test_consumer_stop_preserves_exact_state():
    grammar = PathGrammar()
    stream = adapter().verify_prefix([0], [1, 2, 3], grammar=grammar)
    assert grammar.accepted == []
    assert next(stream).token_id == 1
    stream.close()
    assert grammar.accepted == [1]
    assert grammar.mask_reads == 1


def test_failed_transition_never_returns_selected_token():
    grammar = PathGrammar(reject=True)
    stream = adapter().verify_prefix([0], [1], grammar=grammar)
    with pytest.raises(ValueError, match="rejected"):
        next(stream)
    assert grammar.accepted == []


def test_empty_nonterminal_set_does_not_advance_state():
    grammar = PathGrammar()
    grammar.allowed_token_ids = lambda: []
    with pytest.raises(ValueError, match="no valid"):
        adapter().next_token((0,), grammar=grammar)
    assert grammar.accepted == []


def test_terminal_state_skips_forward_and_bonus():
    grammar = PathGrammar(path=())
    target = adapter()
    assert list(target.verify_prefix([0], [1], grammar=grammar)) == []
    with pytest.raises(GrammarTerminatedError):
        target.next_token((0,), grammar=grammar)
    result = greedy_decode(target, [0], max_new_tokens=10, grammar=grammar)
    assert result.token_ids == () and result.stop_reason == "grammar"
    assert target._model.calls == 0


def test_final_content_is_preserved_when_grammar_terminates():
    result = greedy_decode(adapter(), [0], max_new_tokens=3, grammar=PathGrammar())
    assert result.token_ids == (1, 2, 3)
    assert result.stop_reason == "grammar"


def test_eos_takes_precedence_and_is_not_content():
    grammar = PathGrammar()
    result = greedy_decode(adapter(), [0], max_new_tokens=3, eos_token_ids=[3], grammar=grammar)
    assert result.token_ids == (1, 2)
    assert result.eos_token_id == 3 and result.stop_reason == "eos"
    assert grammar.accepted == [1, 2, 3]


def test_zero_budget_does_not_read_or_advance_grammar():
    target, grammar = adapter(), PathGrammar()
    result = greedy_decode(target, [0], max_new_tokens=0, grammar=grammar)
    assert result.stop_reason == "length" and result.token_ids == ()
    assert grammar.accepted == [] and grammar.mask_reads == 0
    assert target._model.calls == 0


def test_mismatched_vocabulary_fails_before_forward():
    target, grammar = adapter(), PathGrammar()
    grammar.vocab_size = 6
    with pytest.raises(ValueError, match="vocabularies"):
        target.verify_prefix([0], [], grammar=grammar)
    with pytest.raises(ValueError, match="vocabularies"):
        greedy_decode(target, [0], max_new_tokens=0, grammar=grammar)
    assert target._model.calls == 0
