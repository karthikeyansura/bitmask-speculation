import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture
def triton_backend():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('NVIDIA CUDA required')
    from bitmask_speculation.kernels import triton
    # Missing dependency or failed compilation on an eligible GPU is a failure.
    triton.require_device(torch.device('cuda', torch.cuda.current_device()))
    return triton


@pytest.mark.parametrize('vocab', [1,31,32,33,1023,1024,1025,151936,1048577])
def test_tail_ties_and_full_reduction(triton_backend, vocab):
    import torch
    scores = torch.zeros(vocab, device='cuda')
    scores[-1] = 2
    mask = torch.full(((vocab+31)//32,), -1, dtype=torch.int32, device='cuda')
    result = triton_backend.masked_reduce(scores, mask).check()
    assert result.token_id.item() == vocab-1
    assert result.valid_count.item() == vocab
    scores.fill_(-float('inf'))
    assert triton_backend.masked_argmax(scores,mask).item() == 0


@pytest.mark.parametrize('bad_nan', [False,True])
def test_empty_and_masked_out_nan(triton_backend, bad_nan):
    import torch
    scores = torch.zeros(33, device='cuda')
    mask = torch.zeros(2, dtype=torch.int32, device='cuda')
    if bad_nan:
        scores[-1] = float('nan')
    result = triton_backend.masked_reduce(scores,mask)
    assert result.status.item() == (2 if bad_nan else 1)
    with pytest.raises(ValueError):
        result.check()


def test_outputs_and_inputs_are_owned(triton_backend):
    import torch
    scores = torch.tensor([0.,2.,1.],device='cuda')
    mask = torch.tensor([-1],device='cuda',dtype=torch.int32)
    saved = scores.clone(), mask.clone()
    first = triton_backend.masked_reduce(scores,mask).check()
    second = triton_backend.masked_reduce(-scores,mask).check()
    assert first.token_id.item() == 1
    assert second.token_id.item() == 0
    for a,b in zip((first.token_id,first.valid_count,first.status),
                   (second.token_id,second.valid_count,second.status)):
        assert a.data_ptr() != b.data_ptr()
    assert torch.equal(scores,saved[0]) and torch.equal(mask,saved[1])
