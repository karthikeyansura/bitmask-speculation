from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from bitmask_speculation.decoding import VerifiedToken, greedy_decode, verify_prefix


@pytest.mark.parametrize(
    'draft,target,expected',
    [
        ([1, 2, 3], [0], [(0, 'correction')]),
        ([1, 2, 3], [1, 0], [(1, 'accepted'), (0, 'correction')]),
        ([1, 2, 3], [1, 2, 0], [(1, 'accepted'), (2, 'accepted'), (0, 'correction')]),
        ([1, 2], [1, 2, 3], [(1, 'accepted'), (2, 'accepted'), (3, 'bonus')]),
        ([], [2], [(2, 'fallback')]),
    ],
)
def test_longest_prefix_and_exact_selection_calls(draft, target, expected):
    calls = []

    def select(i):
        calls.append(i)
        return target[i]  # Any read beyond the expected prefix fails.

    result = list(verify_prefix(draft, select, vocab_size=4))
    assert [(t.token_id, t.kind) for t in result] == expected
    assert calls == list(range(len(expected)))


def test_consumer_stops_without_selecting_bonus_or_tail():
    calls = []

    def select(i):
        calls.append(i)
        assert i < 2
        return i + 1

    stream = verify_prefix([1, 2], select, vocab_size=4)
    assert calls == []
    assert next(stream) == VerifiedToken(1, 'accepted')
    assert calls == [0]
    assert next(stream) == VerifiedToken(2, 'accepted')
    stream.close()
    assert calls == [0, 1]


def test_inputs_are_snapshotted_and_events_are_immutable():
    draft = [1]
    stream = verify_prefix(draft, lambda i: [1, 2][i], vocab_size=3)
    draft[0] = 0
    event = next(stream)
    assert event == VerifiedToken(1, 'accepted')
    with pytest.raises(FrozenInstanceError):
        event.token_id = 0
    assert next(stream) == VerifiedToken(2, 'bonus')


@pytest.mark.parametrize('bad', [-1, 4, True, 1.5])
def test_invalid_proposal_fails_before_selection(bad):
    def select(i):
        pytest.fail('selection must not run')

    with pytest.raises((TypeError, ValueError)):
        verify_prefix([1, bad], select, vocab_size=4)


@pytest.mark.parametrize('bad', [-1, 4, False, 1.5])
def test_invalid_target_choice_is_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        next(verify_prefix([1], lambda i: bad, vocab_size=4))


@pytest.mark.parametrize('bad', [0, -1, True, 4.0])
def test_invalid_vocabulary(bad):
    with pytest.raises((TypeError, ValueError)):
        verify_prefix([], lambda i: 0, vocab_size=bad)


def test_selection_failure_propagates():
    def select(i):
        raise RuntimeError('model failure')

    with pytest.raises(RuntimeError, match='model failure'):
        next(verify_prefix([], select, vocab_size=4))


class PrefixDependentModel:
    vocab_size = 3

    def next_token(self, prefix):
        return (sum(prefix) + len(prefix)) % self.vocab_size


def test_exhaustive_small_vocabulary_matches_greedy_reference():
    model = PrefixDependentModel()
    # Every proposal of length 0..4, for two distinct committed prefixes.
    for prefix in ((0,), (1, 2, 0)):
        for length in range(5):
            for draft in product(range(model.vocab_size), repeat=length):
                events = list(verify_prefix(
                    draft,
                    lambda i: model.next_token(prefix + draft[:i]),
                    vocab_size=model.vocab_size,
                ))
                reference = greedy_decode(model, prefix, max_new_tokens=len(events))
                assert tuple(t.token_id for t in events) == reference.token_ids
                assert all(t.kind == 'accepted' for t in events[:-1])
                assert events[-1].kind in ('correction', 'bonus', 'fallback')
