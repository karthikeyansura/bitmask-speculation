import math

import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.kernels.cuda import _checked_selection, masked_argmax, masked_sample
from bitmask_speculation.kernels.reference import masked_sample_reference

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("temperature", [0.0, 1e-320, .1, 1.0, 100.0])
@pytest.mark.parametrize("uniform", [0.0, .13, .51, .999999999])
def test_native_sampling_matches_explicit_uniform_reference(cuda_module, temperature, uniform):
    logits = torch.tensor([1000., -math.inf, -2., 1., .5])
    mask = torch.tensor([-2], dtype=torch.int32)
    expected = masked_sample_reference(logits, mask, temperature=temperature, uniform=uniform)
    result = _checked_selection(cuda_module.masked_sample(
        logits.cuda(), mask.cuda(), temperature,
        torch.tensor(uniform, dtype=torch.float64, device="cuda")))
    assert result.item() == expected


def test_seed_replay_and_greedy_consumes_no_rng(cuda_module):
    logits = torch.zeros(3, device="cuda")
    mask = torch.tensor([7], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device=logits.device).manual_seed(37)
    state = generator.get_state().clone()
    greedy = masked_sample(logits, mask, temperature=0, generator=generator)
    assert torch.equal(state, generator.get_state())
    assert greedy.item() == masked_argmax(logits, mask).item()
    first = [masked_sample(logits, mask, generator=generator).item() for _ in range(32)]
    assert not torch.equal(state, generator.get_state())
    generator.set_state(state)
    assert first == [masked_sample(logits, mask, generator=generator).item() for _ in range(32)]


def test_empirical_probabilities_match_masked_softmax(cuda_module):
    logits = torch.tensor([999., 0., math.log(3.)], device="cuda")
    mask = torch.tensor([6], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device=logits.device).manual_seed(123)
    tokens = [masked_sample(logits, mask, generator=generator).item() for _ in range(4096)]
    assert set(tokens) <= {1, 2}
    # Broad fixed bound (>6 binomial standard deviations), not a speed assertion.
    assert abs(tokens.count(1) / len(tokens) - .25) < .05


@pytest.mark.parametrize("values,word,match", [
    ([0.], 0, "no valid"), ([-math.inf], 1, "finite probability"),
    ([math.inf], 1, "positive infinity"), ([0., math.nan], 1, "NaN"),
])
def test_sampling_value_errors(cuda_module, values, word, match):
    with pytest.raises(ValueError, match=match):
        masked_sample(torch.tensor(values, device="cuda"), torch.tensor([word], dtype=torch.int32, device="cuda"))


def test_invalid_uniform_native_status(cuda_module):
    for uniform in [-.1, 1., math.nan]:
        with pytest.raises(ValueError, match="uniform"):
            _checked_selection(cuda_module.masked_sample(
                torch.zeros(1, device="cuda"), torch.ones(1, dtype=torch.int32, device="cuda"), 1.,
                torch.tensor(uniform, dtype=torch.float64, device="cuda")))
