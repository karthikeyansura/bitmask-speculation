import math

import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.kernels.reference import masked_sample_reference as sample


@pytest.mark.parametrize("uniform,expected", [(0, 1), (.249, 1), (.251, 2), (.999999, 2)])
def test_masked_categorical_cdf_against_known_weights(uniform, expected):
    logits = torch.tensor([999.0, 0.0, math.log(3.0)])
    assert sample(logits, torch.tensor([6], dtype=torch.int32), temperature=1, uniform=uniform) == expected


def test_zero_mass_prefix_and_exact_cdf_boundary():
    mask = torch.tensor([-1], dtype=torch.int32)
    logits = torch.tensor([-math.inf, 0.0, 0.0])
    assert sample(logits, mask, temperature=1, uniform=0) == 1
    assert sample(logits, mask, temperature=1, uniform=.5) == 2


def test_zero_temperature_reuses_greedy_contract():
    mask = torch.tensor([6], dtype=torch.int32)
    logits = torch.tensor([999., -math.inf, -math.inf])
    assert sample(logits, mask, temperature=0, uniform=None) == 1


@pytest.mark.parametrize("temperature", [1e-320, 1e-10, 1, 1e300])
def test_extreme_temperatures_remain_in_valid_support(temperature):
    logits = torch.tensor([-3e38, 3e38, -math.inf])
    assert sample(logits, torch.tensor([-1], dtype=torch.int32), temperature=temperature, uniform=.999) == 1


@pytest.mark.parametrize("temperature", [-1, math.nan, math.inf, True, "1"])
def test_invalid_temperature(temperature):
    with pytest.raises((TypeError, ValueError)):
        sample(torch.zeros(1), torch.tensor([1], dtype=torch.int32), temperature=temperature, uniform=0)


@pytest.mark.parametrize("uniform", [-.01, 1, math.nan, math.inf, True, None])
def test_invalid_uniform(uniform):
    with pytest.raises((ValueError, TypeError)):
        sample(torch.zeros(1), torch.tensor([1], dtype=torch.int32), temperature=1, uniform=uniform)


@pytest.mark.parametrize("values,word,match", [
    ([0.0], 0, "no valid"), ([-math.inf], 1, "finite probability"),
    ([math.inf], 1, "positive infinity"), ([0, math.nan], 1, "NaN"),
])
def test_undefined_distributions_fail(values, word, match):
    with pytest.raises(ValueError, match=match):
        sample(torch.tensor(values), torch.tensor([word], dtype=torch.int32), temperature=1, uniform=.5)


def test_invalid_positive_infinity_and_padding_are_ignored():
    assert sample(torch.tensor([math.inf, 2.]), torch.tensor([-2], dtype=torch.int32), temperature=1, uniform=0) == 1
