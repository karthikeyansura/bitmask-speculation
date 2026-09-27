"""Portable mask fixtures use signed int32 storage, not overflowing literals."""
import pytest

from bitmask_speculation.grammar.density import packed_mask_count


@pytest.mark.parametrize('word,vocab,count', [(-2147483648,32,1),(-2147483648,31,0),
                                            (-1,32,32),(-1,31,31),(2147483647,32,31)])
def test_signed_words(word,vocab,count):
    torch = pytest.importorskip('torch')
    assert packed_mask_count(torch.tensor([word],dtype=torch.int32),vocab) == count
