from dataclasses import FrozenInstanceError

import pytest

from bitmask_speculation.decoding import greedy_decode


class ScriptedModel:
    vocab_size = 8

    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.prefixes = []

    def next_token(self, prefix):
        assert isinstance(prefix, tuple)
        self.prefixes.append(prefix)
        return next(self.tokens)


def test_complete_prefix_and_prompt_ownership():
    prompt = [1, 2]
    model = ScriptedModel([3, 4, 5])
    result = greedy_decode(model, prompt, max_new_tokens=3)
    assert model.prefixes == [(1, 2), (1, 2, 3), (1, 2, 3, 4)]
    assert prompt == [1, 2]
    assert result.token_ids == (3, 4, 5)
    assert result.stop_reason == "length"
    assert result.eos_token_id is None
    with pytest.raises(FrozenInstanceError):
        result.token_ids = ()


@pytest.mark.parametrize("eos", [0, 6, 7])
def test_immediate_eos_performs_one_prediction(eos):
    model = ScriptedModel([eos])
    result = greedy_decode(model, [1], max_new_tokens=10, eos_token_ids=[0, 6, 7])
    assert result.token_ids == ()
    assert result.stop_reason == "eos"
    assert result.eos_token_id == eos
    assert model.prefixes == [(1,)]


def test_eos_after_content_is_not_emitted_or_fed_back():
    model = ScriptedModel([3, 7])
    result = greedy_decode(model, [1], max_new_tokens=4, eos_token_ids=[7])
    assert result.token_ids == (3,)
    assert result.stop_reason == "eos"
    assert result.eos_token_id == 7
    assert model.prefixes == [(1,), (1, 3)]


def test_length_limit_does_not_probe_for_eos():
    model = ScriptedModel([3, 7])
    result = greedy_decode(model, [1], max_new_tokens=1, eos_token_ids=[7])
    assert result.token_ids == (3,)
    assert result.stop_reason == "length"
    assert model.prefixes == [(1,)]
    assert next(model.tokens) == 7


def test_eos_on_last_allowed_prediction_takes_precedence():
    model = ScriptedModel([3, 7])
    result = greedy_decode(model, [1], max_new_tokens=2, eos_token_ids=[7])
    assert result.token_ids == (3,)
    assert result.stop_reason == "eos"


def test_zero_budget_never_calls_model():
    model = ScriptedModel([])
    result = greedy_decode(model, [1], max_new_tokens=0)
    assert result.token_ids == ()
    assert result.stop_reason == "length"
    assert model.prefixes == []


def test_prompt_eos_does_not_terminate_generation():
    model = ScriptedModel([3])
    result = greedy_decode(model, [1, 7], max_new_tokens=1, eos_token_ids=[7])
    assert result.token_ids == (3,)
    assert model.prefixes == [(1, 7)]


@pytest.mark.parametrize(
    "limit,error",
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError)
    ],
)
def test_invalid_limits_fail_before_model_call(limit, error):
    model = ScriptedModel([])
    with pytest.raises(error):
        greedy_decode(model, [1], max_new_tokens=limit)
    assert model.prefixes == []


@pytest.mark.parametrize(
    "vocab,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (8.0, TypeError)
    ],
)
def test_invalid_vocabularies(vocab, error):
    model = ScriptedModel([])
    model.vocab_size = vocab
    with pytest.raises(error):
        greedy_decode(model, [1], max_new_tokens=0)


@pytest.mark.parametrize(
    "prompt,error",
    [
        ([], ValueError),
        ([-1], ValueError),
        ([8], ValueError),
        ([True], TypeError),
        ([1.0], TypeError)
    ],
)
def test_invalid_prompt_is_rejected_even_with_zero_budget(prompt, error):
    model = ScriptedModel([])
    with pytest.raises(error):
        greedy_decode(model, prompt, max_new_tokens=0)
    assert model.prefixes == []


@pytest.mark.parametrize(
    "token,error",
    [
        (-1, ValueError),
        (8, ValueError),
        (True, TypeError),
        (1.5, TypeError)
    ],
)
def test_invalid_eos_ids(token, error):
    model = ScriptedModel([])
    with pytest.raises(error):
        greedy_decode(model, [1], max_new_tokens=0, eos_token_ids=[token])


@pytest.mark.parametrize(
    "token,error",
    [
        (-1, ValueError),
        (8, ValueError),
        (False, TypeError),
        (2.0, TypeError)
    ],
)
def test_invalid_model_output_is_not_emitted(token, error):
    model = ScriptedModel([token])
    with pytest.raises(error):
        greedy_decode(model, [1], max_new_tokens=2)
    assert model.prefixes == [(1,)]
