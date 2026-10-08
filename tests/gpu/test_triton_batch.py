import pytest
pytestmark=pytest.mark.gpu


@pytest.fixture
def runtime():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('NVIDIA CUDA required')
    from bitmask_speculation.kernels import triton
    triton.require_device(torch.device('cuda',torch.cuda.current_device()))
    return torch,triton


@pytest.mark.parametrize('batch,vocab',[(0,33),(1,1),(3,33),(5,1025),(2,151936)])
def test_batch_matches_rows(runtime,batch,vocab):
    torch,triton=runtime
    generator=torch.Generator(device='cuda').manual_seed(93)
    logits=torch.randn((batch,vocab),device='cuda',generator=generator)
    masks=torch.randint(-2147483648,2147483647,(batch,(vocab+31)//32),
                        device='cuda',dtype=torch.int32,generator=generator)
    if batch:masks[:,0] |= 1
    saved=logits.clone(),masks.clone()
    result=triton.batched_masked_reduce(logits,masks).check()
    assert result.token_ids.shape == result.valid_counts.shape == result.status.shape == (batch,)
    for row in range(batch):
        single=triton.masked_reduce(logits[row],masks[row]).check()
        ids=torch.arange(vocab,device='cuda')
        allowed=((masks[row,ids//32].to(torch.int64) >> (ids%32)) & 1).bool()
        valid_ids=ids[allowed]
        expected=valid_ids[logits[row,valid_ids].argmax()]
        assert result.token_ids[row].item()==single.token_id.item()==expected.item()
        assert result.valid_counts[row].item()==single.valid_count.item()==allowed.sum().item()
    assert torch.equal(logits,saved[0]) and torch.equal(masks,saved[1])


def test_row_errors_are_independent(runtime):
    torch,triton=runtime
    logits=torch.zeros((3,33),device='cuda')
    mask=torch.full((3,2),-1,device='cuda',dtype=torch.int32)
    mask[0]=0;logits[1,32]=float('nan');logits[2,31]=3
    result=triton.batched_masked_reduce(logits,mask)
    assert result.status.tolist()==[1,2,0]
    assert result.token_ids[2].item()==31 and result.valid_counts.tolist()==[0,33,33]
    with pytest.raises(ValueError,match='row'):
        result.check()


def test_reuse_does_not_alias_batch_outputs(runtime):
    torch,triton=runtime
    from bitmask_speculation.kernels.workspace import ReductionWorkspace
    workspace=ReductionWorkspace('cuda')
    logits=torch.arange(66,device='cuda',dtype=torch.float32).reshape(2,33)
    mask=torch.full((2,2),-1,device='cuda',dtype=torch.int32)
    first=triton.batched_masked_reduce(logits,mask,workspace=workspace).check()
    second=triton.batched_masked_reduce(-logits,mask,workspace=workspace).check()
    assert workspace.allocation_count==1
    assert first.token_ids.tolist()==[32,32] and second.token_ids.tolist()==[0,0]
    assert first.token_ids.data_ptr()!=second.token_ids.data_ptr()


@pytest.mark.parametrize('kind',['logit_stride','mask_stride','broadcast','shape','dtype'])
def test_bad_batch_layout_rejected(runtime,kind):
    torch,triton=runtime
    logits=torch.zeros((2,33),device='cuda')
    mask=torch.zeros((2,2),device='cuda',dtype=torch.int32)
    if kind=='logit_stride':logits=torch.zeros((2,66),device='cuda')[:,::2]
    if kind=='mask_stride':mask=torch.zeros((2,4),device='cuda',dtype=torch.int32)[:,::2]
    if kind=='broadcast':mask=mask[:1].expand(2,-1)
    if kind=='shape':mask=mask[:1]
    if kind=='dtype':logits=logits.double()
    with pytest.raises((TypeError,ValueError)):
        triton.batched_masked_reduce(logits,mask)
