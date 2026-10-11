import json
from types import SimpleNamespace
import pytest

pytestmark = pytest.mark.gpu

@pytest.mark.parametrize('backend', ['torch', 'cuda', 'triton'])
def test_reduction_timing_scopes_keep_raw_samples(backend, tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('NVIDIA CUDA device required')
    if backend == 'triton':
        if torch.cuda.get_device_capability() < (8, 0): pytest.skip('Triton requires sm80')
        pytest.importorskip('triton')
    from bitmask_speculation.benchmarks.reductions import run_reductions
    run_reductions(SimpleNamespace(command='reductions', device='cuda:0', backend=backend,
        vocab=33, rows=2, density=.4, seed=1729, iterations=2, warmups=2,
        repetitions=2, output=str(tmp_path/'run')))
    samples=[json.loads(line) for line in (tmp_path/'run/samples.jsonl').read_text().splitlines()]
    assert len(samples)==6 and all(s['seconds_per_batch']>0 for s in samples)
    assert {s['scope'] for s in samples}=={'graph_device','async_wrapper','checked_wrapper'}
    assert json.loads((tmp_path/'run/status.json').read_text())['status']=='complete'
