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


def test_selection_status_contract_on_cpu():
    from bitmask_speculation.kernels.cuda import _checked_selection
    output = torch.tensor(31, dtype=torch.int64)
    assert _checked_selection((output, torch.tensor(0))) is output
    for code, text in [(1, "no valid"), (2, "NaN")]:
        with pytest.raises(ValueError, match=text):
            _checked_selection((output, torch.tensor(code)))
    with pytest.raises(RuntimeError, match="unknown"):
        _checked_selection((output, torch.tensor(99)))


def test_invalid_public_calls_never_compile(monkeypatch):
    from bitmask_speculation.kernels import cuda
    def forbidden():
        pytest.fail("invalid CPU inputs reached the extension loader")
    monkeypatch.setattr(cuda, "load_extension", forbidden)
    mask = torch.ones(1, dtype=torch.int32)
    for call in [lambda: cuda.mask_count(mask, 32), lambda: cuda.mask_density(mask, 32),
                 lambda: cuda.masked_argmax(torch.zeros(32), mask),
                 lambda: cuda.masked_sample(torch.zeros(32), mask)]:
        with pytest.raises(ValueError, match="CUDA"):
            call()


def test_loader_fails_explicitly_without_cuda(monkeypatch):
    from bitmask_speculation.kernels._extension import load_extension
    load_extension.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA-enabled"):
        load_extension()
    load_extension.cache_clear()


def test_loader_cache_and_packaged_source_selection(monkeypatch):
    import torch.utils.cpp_extension as cpp
    from bitmask_speculation.kernels._extension import load_extension
    calls = []
    marker = object()
    def fake_load(**kwargs):
        calls.append(kwargs)
        return marker
    load_extension.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "test")
    monkeypatch.setattr(cpp, "CUDA_HOME", "/unused-test-toolkit")
    monkeypatch.setattr(cpp, "load", fake_load)
    try:
        assert load_extension() is marker and load_extension() is marker
        assert len(calls) == 1
        assert calls[0]["with_cuda"] is True
        assert "--use_fast_math" not in calls[0]["extra_cuda_cflags"]
        assert any(p.endswith("sample.cu") for p in calls[0]["sources"])
    finally:
        load_extension.cache_clear()
