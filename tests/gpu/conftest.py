import pytest


@pytest.fixture(scope="session")
def cuda_module():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA device required")
    from bitmask_speculation.kernels._extension import load_extension
    # A present GPU with a failed build is a failure, not a skipped test.
    return load_extension()
