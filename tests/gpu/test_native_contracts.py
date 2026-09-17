import pytest

torch = pytest.importorskip("torch")
from bitmask_speculation.kernels.cuda import mask_count, masked_argmax, masked_sample

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("kind", ["cpu", "dtype", "shape", "stride", "length"])
def test_native_mask_rejects_invalid_inputs(cuda_module, kind):
    masks = {
        "cpu": torch.ones(1, dtype=torch.int32),
        "dtype": torch.ones(1, dtype=torch.int64, device="cuda"),
        "shape": torch.ones((1, 1), dtype=torch.int32, device="cuda"),
        "stride": torch.ones(4, dtype=torch.int32, device="cuda")[::2],
        "length": torch.ones(2, dtype=torch.int32, device="cuda"),
    }
    with pytest.raises(RuntimeError):
        cuda_module.mask_count(masks[kind], 32)


@pytest.mark.parametrize("kind", ["cpu", "dtype", "shape", "stride"])
def test_native_selection_rejects_invalid_logits(cuda_module, kind):
    logits = {
        "cpu": torch.zeros(32),
        "dtype": torch.zeros(32, dtype=torch.float16, device="cuda"),
        "shape": torch.zeros((1, 32), device="cuda"),
        "stride": torch.zeros(64, device="cuda")[::2],
    }[kind]
    mask = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        cuda_module.masked_argmax(logits, mask)
    with pytest.raises(RuntimeError):
        cuda_module.masked_sample(logits, mask, 1., torch.tensor(.5, dtype=torch.float64, device="cuda"))


def test_native_sampler_uniform_metadata_and_temperature(cuda_module):
    logits = torch.zeros(1, device="cuda")
    mask = torch.ones(1, dtype=torch.int32, device="cuda")
    for uniform in [torch.tensor(.5), torch.tensor(.5, device="cuda"),
                    torch.tensor([.5], dtype=torch.float64, device="cuda")]:
        with pytest.raises(RuntimeError):
            cuda_module.masked_sample(logits, mask, 1., uniform)
    for temperature in [-1., float("nan"), float("inf")]:
        with pytest.raises(RuntimeError):
            cuda_module.masked_sample(logits, mask, temperature, torch.tensor(.5, dtype=torch.float64, device="cuda"))


@pytest.mark.parametrize("operation", ["count", "argmax", "sample"])
def test_current_stream_ordering_and_retained_outputs(cuda_module, operation):
    stream = torch.cuda.Stream()
    retained = []
    with torch.cuda.stream(stream):
        for turn in range(16):
            mask = torch.empty(9, dtype=torch.int32, device="cuda")
            mask.fill_(-1 if turn % 2 else 1)
            logits = torch.full((257,), -1000., device="cuda")
            logits[0] = 1000.
            if operation == "count":
                result = mask_count(mask, 257)
                expected = 257 if turn % 2 else 9
            elif operation == "argmax":
                result = masked_argmax(logits, mask)
                expected = 0
            else:
                result = masked_sample(logits, mask)
                expected = 0
            retained.append((result, expected))
    stream.synchronize()
    assert len({result.data_ptr() for result, _ in retained}) == len(retained)
    assert all(result.item() == expected for result, expected in retained)


def test_input_device_guard_and_mismatched_devices(cuda_module):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required for device-guard isolation")
    before = torch.cuda.current_device()
    other = (before + 1) % torch.cuda.device_count()
    mask = torch.tensor([-1], dtype=torch.int32, device=f"cuda:{other}")
    assert mask_count(mask, 32).item() == 32
    assert torch.cuda.current_device() == before
    logits = torch.zeros(32, device=f"cuda:{before}")
    with pytest.raises((ValueError, RuntimeError)):
        masked_argmax(logits, mask)
    with pytest.raises(RuntimeError):
        cuda_module.masked_argmax(logits, mask)
