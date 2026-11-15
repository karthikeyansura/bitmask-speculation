import math

import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.kernels import cuda

pytestmark = pytest.mark.gpu


def reference_rows(logits, masks):
    vocab = logits.shape[1]
    ids = torch.arange(vocab, device=logits.device)
    tokens, counts = [], []
    for row in range(logits.shape[0]):
        allowed = ((masks[row, ids // 32].to(torch.int64) >> (ids % 32)) & 1).bool()
        valid = ids[allowed]
        tokens.append(valid[logits[row, valid].argmax()].item() if valid.numel() else -1)
        counts.append(int(allowed.sum()))
    return tokens, counts


@pytest.mark.parametrize("batch,vocab", [(0, 33), (1, 1), (1, 4095), (1, 4097), (3, 33),
                                         (6, 151936), (12, 248320)])
def test_batch_matches_reference_rows(cuda_module, batch, vocab):
    generator = torch.Generator(device="cuda").manual_seed(41)
    logits = torch.randn((batch, vocab), device="cuda", generator=generator)
    masks = torch.randint(-2147483648, 2147483647, (batch, (vocab + 31) // 32),
                          device="cuda", dtype=torch.int32, generator=generator)
    if batch:
        masks[:, 0] |= 1
    saved = logits.clone(), masks.clone()
    result = cuda.batched_masked_reduce(logits, masks).check()
    assert result.token_ids.shape == result.valid_counts.shape == result.status.shape == (batch,)
    tokens, counts = reference_rows(logits, masks)
    assert result.token_ids.tolist() == tokens and result.valid_counts.tolist() == counts
    for row in range(batch):
        assert cuda.masked_argmax(logits[row].contiguous(), masks[row].contiguous()).item() == tokens[row]
    assert torch.equal(logits, saved[0]) and torch.equal(masks, saved[1])


def test_batch_rows_fail_independently_and_ties_pick_lowest_id(cuda_module):
    logits = torch.zeros((4, 8193), device="cuda")
    masks = torch.full((4, 257), -1, device="cuda", dtype=torch.int32)
    masks[0] = 0
    logits[1, 8192] = math.nan
    logits[2, 4096] = logits[2, 8191] = 5.0  # equal maxima in different blocks
    result = cuda.batched_masked_reduce(logits, masks)
    assert result.status.tolist() == [1, 2, 0, 0]
    assert result.token_ids.tolist()[2:] == [4096, 0]
    assert result.valid_counts.tolist() == [0, 8193, 8193, 8193]
