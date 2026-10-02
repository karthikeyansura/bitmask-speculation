from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from bitmask_speculation.kernels.dispatch import packed_mask, select_token, validate_backend


@pytest.mark.parametrize('backend',['cuda','triton'])
def test_native_cpu_rejected(backend):
    with pytest.raises(ValueError,match='CUDA'):
        validate_backend(backend,torch.device('cpu'))


def test_unknown_backend():
    with pytest.raises(ValueError,match='backend'):
        validate_backend('auto',torch.device('cpu'))


def test_generic_signed_mask():
    grammar=SimpleNamespace(vocab_size=33,allowed_token_ids=lambda:(0,31,32,31))
    assert packed_mask(grammar).tolist()==[-2147483647,1]


@pytest.mark.parametrize('token',[-1,33,True])
def test_invalid_grammar_token(token):
    grammar=SimpleNamespace(vocab_size=33,allowed_token_ids=lambda:(token,))
    with pytest.raises((TypeError,ValueError)):
        packed_mask(grammar)


def test_snapshot_ownership():
    words=torch.tensor([-1],dtype=torch.int32)
    grammar=SimpleNamespace(vocab_size=3,packed_mask=lambda:words)
    snapshot=packed_mask(grammar)
    words.zero_()
    assert snapshot.item()==-1


def test_reference_selection_does_not_advance():
    grammar=SimpleNamespace(vocab_size=3,allowed_token_ids=lambda:(2,1))
    assert select_token(torch.tensor([100.,1.,1.]),grammar)==1
