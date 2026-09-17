import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.grammar.density import packed_mask_count
from bitmask_speculation.grammar.selection import masked_argmax as reference_argmax
from bitmask_speculation.kernels.cuda import mask_count, masked_argmax

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("vocab", [31, 32, 33, 255, 256, 257, 4097, 248321])
def test_random_words_and_logits_match_independent_references(cuda_module, vocab):
    generator = torch.Generator().manual_seed(vocab)
    for _ in range(4):
        mask = torch.randint(-(1 << 31), (1 << 31), ((vocab + 31) // 32,), dtype=torch.int32, generator=generator)
        mask[0] |= 1  # Ensure nonempty support while retaining signed words.
        logits = torch.randn(vocab, generator=generator)
        words = mask.tolist()
        allowed = [i for i in range(vocab) if (words[i // 32] >> (i % 32)) & 1]
        cuda_mask, cuda_logits = mask.cuda(), logits.cuda()
        assert mask_count(cuda_mask, vocab).item() == packed_mask_count(mask, vocab)
        assert masked_argmax(cuda_logits, cuda_mask).item() == reference_argmax(logits, allowed)
        assert torch.equal(cuda_mask.cpu(), mask) and torch.equal(cuda_logits.cpu(), logits)
