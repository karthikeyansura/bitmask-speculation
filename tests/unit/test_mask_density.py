import random

import pytest

torch = pytest.importorskip("torch")

from bitmask_speculation.grammar.density import packed_mask_count, packed_mask_density


@pytest.mark.parametrize("vocab", [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 248320])
def test_counts_match_bitwise_oracle_with_dirty_tail(vocab):
    rng = random.Random(vocab)
    for words in ([0] * ((vocab + 31) // 32), [-1] * ((vocab + 31) // 32),
                  [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range((vocab + 31) // 32)]):
        mask = torch.tensor(words, dtype=torch.int32)
        before = mask.clone()
        expected = sum((words[t // 32] >> (t % 32)) & 1 for t in range(vocab))
        assert packed_mask_count(mask, vocab) == expected
        assert packed_mask_density(mask, vocab) == expected / vocab
        assert torch.equal(mask, before)


@pytest.mark.parametrize("bit", range(32))
def test_every_bit_including_sign_bit(bit):
    word = (1 << bit) if bit < 31 else -(1 << 31)
    mask = torch.tensor([word], dtype=torch.int32)
    assert packed_mask_count(mask, 32) == 1
    if bit:
        assert packed_mask_count(mask, bit) == 0


def test_noncontiguous_view_and_input_ownership():
    backing = torch.tensor([-2147483648, 123, -1, 456], dtype=torch.int32)
    mask = backing[::2]
    assert not mask.is_contiguous()
    assert packed_mask_count(mask, 33) == 2
    assert backing.tolist() == [-2147483648, 123, -1, 456]


@pytest.mark.parametrize("vocab", [0, -1, True, 32.0, None])
def test_invalid_vocabulary(vocab):
    with pytest.raises((ValueError, TypeError)):
        packed_mask_density(torch.tensor([-1], dtype=torch.int32), vocab)


@pytest.mark.parametrize("shape", [(), (1, 1), (0,), (2,)])
def test_bad_shape_or_word_count(shape):
    with pytest.raises(ValueError):
        packed_mask_count(torch.zeros(shape, dtype=torch.int32), 32)


@pytest.mark.parametrize("dtype", [torch.int64, torch.int16, torch.uint8, torch.float32, torch.bool])
def test_wrong_dtype(dtype):
    with pytest.raises(TypeError, match="int32"):
        packed_mask_count(torch.zeros(1, dtype=dtype), 32)


def test_wrong_container_layout_and_device():
    with pytest.raises(TypeError, match="tensor"):
        packed_mask_count([-1], 32)
    with pytest.raises(ValueError, match="strided"):
        packed_mask_count(torch.tensor([1], dtype=torch.int32).to_sparse(), 32)
    with pytest.raises(ValueError, match="CPU"):
        packed_mask_count(torch.empty(1, dtype=torch.int32, device="meta"), 32)
