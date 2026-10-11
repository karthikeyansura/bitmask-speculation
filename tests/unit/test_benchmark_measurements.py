import json
import math

import pytest

from bitmask_speculation.benchmarks.artifacts import RunDirectory, distribution
from bitmask_speculation.benchmarks.reductions import measure_wall


def test_failure_preserves_completed_samples(tmp_path):
    with pytest.raises(RuntimeError):
        with RunDirectory(tmp_path/'run', {'kind': 'fixture'}) as run:
            run.append('samples.jsonl', '{"seconds":1}')
            raise RuntimeError('device failed')
    assert json.loads((tmp_path/'run/status.json').read_text())['status'] == 'failed'
    assert (tmp_path/'run/samples.jsonl').read_text() == '{"seconds":1}\n'
    with pytest.raises(FileExistsError):
        RunDirectory(tmp_path/'run', {})


def test_timing_brackets_all_operations_and_drain():
    events = []
    times = iter((10., 16.))
    def sync(): events.append('sync')
    def clock(): events.append('clock'); return next(times)
    def op(): events.append('op'); return (1, 2, 0)
    elapsed, result = measure_wall(op, sync, iterations=3, clock=clock)
    assert events == ['sync', 'clock', 'op', 'op', 'op', 'sync', 'clock']
    assert elapsed == 2 and result == (1, 2, 0)


@pytest.mark.parametrize('values', [[], [0], [-1], [math.nan], [math.inf], [True]])
def test_invalid_measurements_rejected(values):
    with pytest.raises(ValueError): distribution(values)


def test_nearest_rank_percentile():
    assert distribution(range(1, 21)) == {'samples':20, 'median_seconds':10.5,
                                        'minimum_seconds':1, 'p95_seconds':19}
