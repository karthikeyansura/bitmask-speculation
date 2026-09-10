import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.kernels.cuda import masked_argmax

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("vocab", [1, 31, 32, 33, 255, 256, 257, 8193])
@pytest.mark.parametrize("value", [0.0, float("-inf"), float("inf")])
def test_ties_choose_first_valid_across_lanes_and_warps(cuda_module, vocab, value):
    logits = torch.full((vocab,), value, dtype=torch.float32, device="cuda")
    mask = torch.full(((vocab + 31) // 32,), -1, dtype=torch.int32, device="cuda")
    assert masked_argmax(logits, mask).item() == 0


def test_sparse_support_excludes_higher_invalid_logit(cuda_module):
    logits = torch.zeros(257, device="cuda")
    logits[0] = 999
    logits[31] = logits[256] = 10
    mask = torch.zeros(9, dtype=torch.int32, device="cuda")
    mask[0] = -2147483648
    mask[-1] = -1  # Dirty tail must not create out-of-vocabulary candidates.
    before = logits.clone()
    assert masked_argmax(logits, mask).item() == 31
    assert torch.equal(logits, before)


def test_empty_support_and_nan_are_explicit_errors(cuda_module):
    logits = torch.zeros(33, device="cuda")
    mask = torch.zeros(2, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="no valid"):
        masked_argmax(logits, mask)
    mask[0] = 1
    logits[32] = float("nan")  # NaN at an invalid position must still fail.
    with pytest.raises(ValueError, match="NaN"):
        masked_argmax(logits, mask)
