import pytest
from bitmask_speculation.benchmarks.profiling import capture_range


def test_capture_brackets_operation_and_final_drain():
    events=[]
    def op():events.append('operation');return 4
    result=capture_range(op, lambda:events.append('sync'),lambda:events.append('start'),
                         lambda:events.append('stop'),iterations=2)
    assert result==4 and events==['sync','start','operation','operation','sync','stop']


def test_failed_operation_still_closes_range():
    events=[]
    def fail():raise RuntimeError('kernel failed')
    with pytest.raises(RuntimeError,match='kernel failed'):
        capture_range(fail,lambda:None,lambda:events.append('start'),lambda:events.append('stop'),iterations=1)
    assert events==['start','stop']


@pytest.mark.parametrize('iterations',[0,-1,True,1.5])
def test_invalid_iterations_never_start_profiler(iterations):
    def fail():raise AssertionError('should not run')
    with pytest.raises(ValueError):capture_range(fail,fail,fail,fail,iterations=iterations)
