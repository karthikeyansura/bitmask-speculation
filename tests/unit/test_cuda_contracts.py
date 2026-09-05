import pytest

torch = pytest.importorskip("torch")

from bitmask_speculation.kernels._extension import source_directory
from bitmask_speculation.kernels._validation import mask_input, selection_inputs


def test_native_sources_are_available_without_compiling():
    root = source_directory()
    assert (root / "bindings.cpp").is_file()
    assert (root / "runtime.cu").is_file()


@pytest.mark.parametrize("vocab", [0, -1, True, 1.0, 2147483648])
def test_invalid_vocab_rejected_without_cuda(vocab):
    with pytest.raises((ValueError, TypeError)):
        mask_input(torch.zeros(1, dtype=torch.int32), vocab)


@pytest.mark.parametrize("mask", [
    [1], torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.float32),
    torch.zeros((1, 1), dtype=torch.int32), torch.zeros(4, dtype=torch.int32)[::2],
    torch.zeros(1, dtype=torch.int32).to_sparse(), torch.zeros(2, dtype=torch.int32),
])
def test_invalid_mask_contract_before_device_dispatch(mask):
    with pytest.raises((ValueError, TypeError)):
        mask_input(mask, 32)


def test_cpu_inputs_are_not_transferred_implicitly():
    with pytest.raises(ValueError, match="CUDA"):
        mask_input(torch.ones(1, dtype=torch.int32), 32)
    with pytest.raises(TypeError, match="float32"):
        selection_inputs(torch.zeros(32, dtype=torch.float16), torch.ones(1, dtype=torch.int32))
