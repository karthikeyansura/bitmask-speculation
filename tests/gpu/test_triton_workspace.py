import pytest
pytestmark=pytest.mark.gpu


@pytest.fixture
def runtime():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('NVIDIA CUDA required')
    from bitmask_speculation.kernels import triton
    from bitmask_speculation.kernels.workspace import ReductionWorkspace
    triton.require_device(torch.device('cuda',torch.cuda.current_device()))
    return torch,triton,ReductionWorkspace


def test_reuse_growth_clear_and_retained_results(runtime):
    torch,triton,Workspace=runtime
    workspace=Workspace('cuda')
    saved=[]
    for vocab in [32,32,2049,32]:
        scores=torch.arange(vocab,device='cuda',dtype=torch.float32)
        mask=torch.full(((vocab+31)//32,),-1,device='cuda',dtype=torch.int32)
        result=triton.masked_reduce(scores,mask,workspace=workspace).check()
        saved.append((result,vocab))
    assert workspace.allocation_count==2 and workspace.capacity==3
    assert len({r.token_id.data_ptr() for r,_ in saved})==4
    for result,vocab in saved:
        assert result.token_id.item()==vocab-1 and result.valid_count.item()==vocab
    workspace.clear()
    assert workspace.capacity==0
    for result,vocab in saved:assert result.token_id.item()==vocab-1


def test_stream_binding_and_independent_workspaces(runtime):
    torch,triton,Workspace=runtime
    main=Workspace('cuda')
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    results=[]
    for stream in streams:
        with torch.cuda.stream(stream):
            workspace=Workspace('cuda')
            scores=torch.tensor([0.,2.],device='cuda')
            mask=torch.tensor([-1],device='cuda',dtype=torch.int32)
            with pytest.raises(ValueError,match='stream'):
                triton.masked_reduce(scores,mask,workspace=main)
            results.append(triton.masked_reduce(scores,mask,workspace=workspace))
    for stream in streams:stream.synchronize()
    for result in results:assert result.check().token_id.item()==1


def test_reentrant_workspace_rejected(runtime):
    torch,triton,Workspace=runtime
    workspace=Workspace('cuda')
    with workspace.acquire(workspace.device,1):
        with pytest.raises(RuntimeError,match='in use'):
            with workspace.acquire(workspace.device,1):pass


def test_device_guard_and_wrong_workspace_device(runtime):
    torch,triton,Workspace=runtime
    if torch.cuda.device_count()<2:pytest.skip('two CUDA devices required')
    with torch.cuda.device(0):workspace=Workspace('cuda:0')
    scores=torch.tensor([0.,2.],device='cuda:1')
    mask=torch.tensor([-1],device='cuda:1',dtype=torch.int32)
    with torch.cuda.device(0):
        assert triton.masked_reduce(scores,mask).check().token_id.item()==1
        assert torch.cuda.current_device()==0
        with pytest.raises(ValueError,match='device'):
            triton.masked_reduce(scores,mask,workspace=workspace)
