from types import SimpleNamespace
import pytest

pytestmark=pytest.mark.gpu


@pytest.mark.parametrize('backend',['cuda','triton'])
@pytest.mark.parametrize('dtype',['float32','float16','bfloat16'])
def test_native_matches_reference(backend,dtype):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('NVIDIA CUDA required')
    from bitmask_speculation.kernels.dispatch import select_token
    grammar=SimpleNamespace(vocab_size=65,allowed_token_ids=lambda:(31,32,64))
    scores=torch.zeros(65,device='cuda',dtype=getattr(torch,dtype))
    scores[31]=2;scores[32]=2;scores[0]=100
    assert select_token(scores,grammar,backend)==select_token(scores,grammar,'reference')==31


@pytest.mark.parametrize('backend',['cuda','triton'])
def test_native_float64_not_silently_rounded(backend):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('NVIDIA CUDA required')
    from bitmask_speculation.kernels.dispatch import select_token
    grammar=SimpleNamespace(vocab_size=2,allowed_token_ids=lambda:(0,1))
    with pytest.raises(TypeError,match='float32'):
        select_token(torch.zeros(2,device='cuda',dtype=torch.float64),grammar,backend)
