import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.grammar.density import packed_mask_count
from bitmask_speculation.kernels.cuda import mask_count, mask_density

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("vocab", [1, 31, 32, 33, 255, 256, 257, 248320, 248321])
@pytest.mark.parametrize("fill", [0, -1, -2147483648])
def test_signed_and_tail_counts_match_cpu(cuda_module, vocab, fill):
    cpu = torch.full(((vocab + 31) // 32,), fill, dtype=torch.int32)
    mask = cpu.cuda()
    expected = packed_mask_count(cpu, vocab)
    output = mask_count(mask, vocab)
    assert output.shape == () and output.dtype == torch.int64 and output.is_cuda
    assert output.item() == expected
    # CUDA scalar division may multiply by a rounded reciprocal.
    # Integer popcount remains exact; density allows only FP64 rounding.
    assert mask_density(mask, vocab).item() == pytest.approx(expected / vocab, rel=1e-15, abs=0)
    assert torch.equal(mask.cpu(), cpu)


def test_repeated_calls_own_their_outputs(cuda_module):
    mask = torch.tensor([-1, -1], dtype=torch.int32, device="cuda")
    first = mask_count(mask, 33)
    second = mask_count(mask, 64)
    empty = mask_count(torch.zeros_like(mask), 64)
    assert (first.item(), second.item(), empty.item()) == (33, 64, 0)
    assert len({first.data_ptr(), second.data_ptr(), empty.data_ptr()}) == 3
