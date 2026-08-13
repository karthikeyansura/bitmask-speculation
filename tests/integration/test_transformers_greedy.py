from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores))
        self.config = SimpleNamespace(vocab_size=len(scores), is_encoder_decoder=False)
        self.calls = []

    def forward(self, *, input_ids, use_cache, return_dict):
        self.calls.append(
            (input_ids.tolist(), use_cache, return_dict, torch.is_inference_mode_enabled())
        )
        logits = torch.zeros(1, input_ids.shape[1], len(self.scores), dtype=self.scores.dtype)
        logits[0, -1] = self.scores
        return SimpleNamespace(logits=logits)


def test_last_position_alignment_and_inference_mode():
    model = FixedLogitsModel([0.0, 1.0, 9.0]).eval()
    result = greedy_decode(TransformersGreedyModel(model), [1, 0], max_new_tokens=2)
    assert result.token_ids == (2, 2)
    assert model.calls == [([[1, 0]], False, True, True), ([[1, 0, 2]], False, True, True)]


@pytest.mark.parametrize(
    "scores,expected",
    [
        ([2.0, 2.0, 1.0], 0),
        ([0.0, 3.0, 3.0], 1),
        ([-float("inf")] * 3, 0),
        ([0.0, float("inf"), float("inf")], 1)
    ],
)
def test_exact_ties_choose_lowest_token_id(scores, expected):
    model = FixedLogitsModel(scores).eval()
    assert TransformersGreedyModel(model).next_token((1,)) == expected


def test_nan_scores_are_rejected():
    model = FixedLogitsModel([0.0, float("nan"), 2.0]).eval()
    with pytest.raises(ValueError, match="NaN"):
        TransformersGreedyModel(model).next_token((1,))


def test_integer_logits_are_rejected():
    model = FixedLogitsModel([0, 1, 2]).eval()
    with pytest.raises(TypeError, match="floating-point"):
        TransformersGreedyModel(model).next_token((1,))


def test_wrong_vocabulary_dimension_is_rejected():
    model = FixedLogitsModel([0.0, 1.0, 2.0]).eval()
    model.config.vocab_size = 4
    with pytest.raises(ValueError, match="shape"):
        TransformersGreedyModel(model).next_token((1,))


def test_training_mode_is_not_silently_changed():
    model = FixedLogitsModel([0.0, 1.0])
    with pytest.raises(ValueError, match="eval"):
        TransformersGreedyModel(model)
    assert model.training
    adapter = TransformersGreedyModel(model.eval())
    model.train()
    with pytest.raises(ValueError, match="evaluation"):
        adapter.next_token((1,))


def test_encoder_decoder_is_rejected():
    model = FixedLogitsModel([0.0, 1.0]).eval()
    model.config.is_encoder_decoder = True
    with pytest.raises(ValueError, match="decoder-only"):
        TransformersGreedyModel(model)


def test_meta_device_is_rejected():
    model = FixedLogitsModel([0.0, 1.0]).eval().to("meta")
    with pytest.raises(ValueError, match="materialized"):
        TransformersGreedyModel(model)


@pytest.mark.parametrize("seed", [7, 19, 41])
def test_tiny_causal_model_matches_transformers_generate(seed):
    torch.manual_seed(seed)
    config = transformers.GPT2Config(
        vocab_size=17, n_positions=32, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=1, eos_token_id=None, pad_token_id=0,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    prompt = [1, 3, 5]
    with torch.inference_mode():
        expected = model.generate(
            input_ids=torch.tensor([prompt]),
            attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
            max_new_tokens=5, do_sample=False, num_beams=1, use_cache=False,
            eos_token_id=None, pad_token_id=0,
        )[0, len(prompt):].tolist()
    adapter = TransformersGreedyModel(model)
    result = greedy_decode(adapter, prompt, max_new_tokens=5)
    assert result.token_ids == tuple(expected)
    assert result.stop_reason == "length"
    stopped = greedy_decode(adapter, prompt, max_new_tokens=5, eos_token_ids=[expected[0]])
    assert stopped.token_ids == ()
    assert stopped.stop_reason == "eos"
    assert stopped.eos_token_id == expected[0]
