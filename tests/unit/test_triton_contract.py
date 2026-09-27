import pytest

from bitmask_speculation.kernels.triton import reduction_shape


@pytest.mark.parametrize('vocab,shape', [(1,(1,1)),(1024,(1,1)),(1025,(2,2)),
                                       (1048577,(1025,2048)),(16777216,(16384,16384))])
def test_reduction_shape(vocab, shape):
    assert reduction_shape(vocab) == shape


@pytest.mark.parametrize('vocab', [0,-1,16777217])
def test_reduction_limit(vocab):
    with pytest.raises(ValueError):
        reduction_shape(vocab)


@pytest.mark.parametrize('vocab', [True,1.0,'32'])
def test_reduction_type(vocab):
    with pytest.raises(TypeError):
        reduction_shape(vocab)


def test_cpu_is_not_a_silent_fallback():
    torch = pytest.importorskip('torch')
    from bitmask_speculation.kernels.triton import masked_reduce
    with pytest.raises(ValueError, match='CUDA'):
        masked_reduce(torch.zeros(32), torch.tensor([-1], dtype=torch.int32))
