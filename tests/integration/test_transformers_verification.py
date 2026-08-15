from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel


class PositionModel(torch.nn.Module):
    def __init__(self, choices, nan_from=None):
        super().__init__()
        self.register_buffer('anchor', torch.zeros(1))
        self.config = SimpleNamespace(vocab_size=8, is_encoder_decoder=False)
        self.choices = choices
        self.nan_from = nan_from
        self.calls = []

    def forward(self, *, input_ids, use_cache, return_dict):
        self.calls.append((input_ids.tolist(), use_cache, return_dict))
        logits = torch.zeros(1, input_ids.shape[1], self.config.vocab_size)
        for i, choice in enumerate(self.choices):
            logits[0, i, choice] = 10.0
        if self.nan_from is not None:
            logits[0, self.nan_from:] = float('nan')
        return SimpleNamespace(logits=logits)


def test_fixed_prefix_offset_and_bonus_with_one_forward():
    model = PositionModel([7, 7, 1, 2, 3]).eval()
    stream = TransformersGreedyModel(model).verify_prefix([6, 5, 4], [1, 2])
    assert model.calls == [([[6, 5, 4, 1, 2]], False, True)]
    assert not torch.is_inference_mode_enabled()
    events = list(stream)
    assert [t.token_id for t in events] == [1, 2, 3]
    assert [t.kind for t in events] == ['accepted', 'accepted', 'bonus']
    assert not torch.is_inference_mode_enabled()
    assert len(model.calls) == 1


def test_rejected_tail_is_never_selected_even_with_nan_logits():
    model = PositionModel([7, 1, 4, 0, 0], nan_from=3).eval()
    events = list(TransformersGreedyModel(model).verify_prefix([6, 5], [1, 2, 3]))
    assert [(t.token_id, t.kind) for t in events] == [(1, 'accepted'), (4, 'correction')]


def test_nan_in_consumed_row_fails():
    model = PositionModel([1, 2], nan_from=0).eval()
    stream = TransformersGreedyModel(model).verify_prefix([6], [1])
    with pytest.raises(ValueError, match='NaN'):
        next(stream)


def test_unused_bonus_is_not_selected():
    model = PositionModel([1, 2], nan_from=1).eval()
    stream = TransformersGreedyModel(model).verify_prefix([6], [1])
    assert next(stream).token_id == 1
    stream.close()


def test_empty_proposals_take_target_only_step():
    model = PositionModel([7, 4]).eval()
    events = list(TransformersGreedyModel(model).verify_prefix([6, 5], []))
    assert [(t.token_id, t.kind) for t in events] == [(4, 'fallback')]
    assert len(model.calls) == 1


@pytest.mark.parametrize('prefix,draft', [([], []), ([8], []), ([True], []), ([1], [-1])])
def test_invalid_inputs_do_not_run_model(prefix, draft):
    model = PositionModel([]).eval()
    with pytest.raises((ValueError, TypeError)):
        TransformersGreedyModel(model).verify_prefix(prefix, draft)
    assert model.calls == []


@pytest.mark.parametrize('seed', [7, 19, 41])
def test_real_causal_model_every_rejection_position_matches_greedy(seed):
    torch.manual_seed(seed)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=17, n_positions=32, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=1, eos_token_id=None, pad_token_id=0,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )).eval()
    adapter = TransformersGreedyModel(model)
    # A longer prefix models later rounds, not only an initial prompt.
    for prefix in ([1], [1, 3, 5, 2, 4]):
        expected = greedy_decode(adapter, prefix, max_new_tokens=5).token_ids
        for mismatch in (0, 1, 2, 3, None):
            draft = list(expected[:4])
            if mismatch is not None:
                draft[mismatch] = (draft[mismatch] + 1) % adapter.vocab_size
            calls = []
            handle = model.register_forward_hook(lambda *args: calls.append(1))
            try:
                events = list(adapter.verify_prefix(prefix, draft))
            finally:
                handle.remove()
            count = 5 if mismatch is None else mismatch + 1
            assert tuple(t.token_id for t in events) == expected[:count]
            assert events[-1].kind == ('bonus' if mismatch is None else 'correction')
            assert len(calls) == 1
