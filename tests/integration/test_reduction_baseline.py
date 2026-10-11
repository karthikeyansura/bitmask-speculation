import pytest

torch = pytest.importorskip('torch')
from bitmask_speculation.benchmarks.reductions import torch_reduce


def test_baseline_signed_words_tail_and_negative_infinity():
    scores = torch.full((2, 33), -float('inf'))
    mask = torch.tensor([[-2147483648, 1], [0, -1]], dtype=torch.int32)
    tokens, counts, status = torch_reduce(scores, mask)
    assert tokens.tolist() == [31, 32]
    assert counts.tolist() == [2, 1]
    assert status.tolist() == [0, 0]


def test_baseline_rejects_masked_nan_and_empty_rows():
    scores = torch.zeros(2, 3)
    scores[0, 2] = float('nan')
    _, counts, status = torch_reduce(scores, torch.tensor([[1], [0]], dtype=torch.int32))
    assert counts.tolist() == [1, 0] and status.tolist() == [2, 1]
