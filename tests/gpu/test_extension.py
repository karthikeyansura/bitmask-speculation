import pytest

pytestmark = pytest.mark.gpu


def test_extension_builds_and_reports_cuda_runtime(cuda_module):
    assert cuda_module.runtime_version() > 0
