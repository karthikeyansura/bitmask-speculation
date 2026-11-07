import pytest

torch = pytest.importorskip('torch')
from bitmask_speculation.kernels import dispatch
from bitmask_speculation.models.loading import resolve_device, resolve_dtype


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Report one NVIDIA device with a chosen capability; no kernel runs."""
    capability = {'value': (8, 6)}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: capability['value'])
    monkeypatch.setattr(torch.version, 'hip', None)
    monkeypatch.setattr(torch.version, 'cuda', '12.8')
    return capability


def test_auto_device_falls_back_to_cpu(no_cuda):
    assert resolve_device('auto') == torch.device('cpu')


def test_auto_device_prefers_current_cuda_device(fake_cuda):
    assert resolve_device('auto') == torch.device('cuda', 0)


def test_explicit_device_overrides_auto(fake_cuda):
    assert resolve_device('cpu') == torch.device('cpu')
    assert resolve_device('cuda:1') == torch.device('cuda', 1)


def test_explicit_cuda_requires_cuda(no_cuda):
    with pytest.raises(ValueError, match='unavailable'):
        resolve_device('cuda')


def test_unsupported_device_rejected():
    with pytest.raises(ValueError, match='supported devices'):
        resolve_device('meta')


def test_auto_dtype_cpu_is_float32():
    assert resolve_dtype('auto', torch.device('cpu')) == 'float32'


@pytest.mark.parametrize('capability,expected', [((8, 0), 'bfloat16'), ((9, 0), 'bfloat16'), ((7, 5), 'float16')])
def test_auto_dtype_cuda_prefers_bfloat16(fake_cuda, capability, expected):
    fake_cuda['value'] = capability
    assert resolve_dtype('auto', torch.device('cuda', 0)) == expected


@pytest.mark.parametrize('supported,expected', [(True, 'bfloat16'), (False, 'float16')])
def test_auto_dtype_rocm_uses_runtime_support(fake_cuda, monkeypatch, supported, expected):
    monkeypatch.setattr(torch.version, 'hip', '6.0')
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda *a, **k: supported)
    assert resolve_dtype('auto', torch.device('cuda', 0)) == expected


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_explicit_dtype_overrides_auto(fake_cuda, dtype):
    assert resolve_dtype(dtype, torch.device('cuda', 0)) == dtype
    assert resolve_dtype(dtype, torch.device('cpu')) == dtype


def test_unknown_dtype_rejected():
    with pytest.raises(ValueError, match='dtype'):
        resolve_dtype('float64', torch.device('cpu'))


def test_auto_backend_cpu_is_reference(fake_cuda, monkeypatch):
    monkeypatch.setattr(dispatch, 'triton_available', lambda device: pytest.fail('CPU must not probe Triton'))
    assert dispatch.resolve_backend('auto', torch.device('cpu')) == 'reference'


def test_auto_backend_prefers_triton(fake_cuda, monkeypatch):
    monkeypatch.setattr(dispatch, 'triton_available', lambda device: True)
    monkeypatch.setattr(dispatch, 'cuda_extension_available', lambda: pytest.fail('Triton wins first'))
    monkeypatch.setattr('bitmask_speculation.kernels.triton.require_device', lambda device: None)
    assert dispatch.resolve_backend('auto', torch.device('cuda', 0)) == 'triton'


def test_auto_backend_falls_back_to_cuda(fake_cuda, monkeypatch):
    monkeypatch.setattr(dispatch, 'triton_available', lambda device: False)
    monkeypatch.setattr(dispatch, 'cuda_extension_available', lambda: True)
    assert dispatch.resolve_backend('auto', torch.device('cuda', 0)) == 'cuda'


def test_auto_backend_falls_back_to_reference(fake_cuda, monkeypatch):
    monkeypatch.setattr(dispatch, 'triton_available', lambda device: False)
    monkeypatch.setattr(dispatch, 'cuda_extension_available', lambda: False)
    assert dispatch.resolve_backend('auto', torch.device('cuda', 0)) == 'reference'


def test_auto_backend_rocm_is_reference(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', '6.0')
    monkeypatch.setattr(dispatch, 'triton_available', lambda device: pytest.fail('ROCm must not probe Triton'))
    assert dispatch.resolve_backend('auto', torch.device('cuda', 0)) == 'reference'


@pytest.mark.parametrize('backend', ['cuda', 'triton'])
def test_explicit_backend_is_still_validated(backend):
    with pytest.raises(ValueError, match='CUDA'):
        dispatch.resolve_backend(backend, torch.device('cpu'))


def test_explicit_reference_overrides_auto(fake_cuda, monkeypatch):
    monkeypatch.setattr(dispatch, 'triton_available', lambda device: pytest.fail('explicit backend is not probed'))
    assert dispatch.resolve_backend('reference', torch.device('cuda', 0)) == 'reference'


def test_triton_unavailable_below_compute_capability_8(fake_cuda):
    fake_cuda['value'] = (7, 5)
    assert not dispatch.triton_available(torch.device('cuda', 0))


def test_triton_unavailable_on_cpu():
    assert not dispatch.triton_available(torch.device('cpu'))


@pytest.mark.parametrize('cuda_home,ninja,expected', [('/usr/local/cuda', True, True),
                                                       (None, True, False), ('/usr/local/cuda', False, False)])
def test_cuda_extension_prerequisites(fake_cuda, monkeypatch, cuda_home, ninja, expected):
    from torch.utils import cpp_extension
    monkeypatch.setattr(cpp_extension, 'CUDA_HOME', cuda_home)
    monkeypatch.setattr(cpp_extension, 'is_ninja_available', lambda: ninja)
    assert dispatch.cuda_extension_available() is expected


def test_cuda_extension_requires_cuda_build(fake_cuda, monkeypatch):
    monkeypatch.setattr(torch.version, 'cuda', None)
    assert not dispatch.cuda_extension_available()
