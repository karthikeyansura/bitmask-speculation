"""Explicit, cached CUDA compilation from packaged sources."""

from functools import lru_cache
import hashlib
import os
from pathlib import Path


def source_directory() -> Path:
    packaged = Path(__file__).with_name("_src")
    if packaged.is_dir():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / "csrc"
    if checkout.is_dir():
        return checkout
    raise RuntimeError("CUDA sources are missing; reinstall a complete distribution")


@lru_cache(maxsize=1)
def load_extension():
    """Compile on explicit request; never substitute CPU code for a CUDA call."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("a CUDA-enabled PyTorch build and visible NVIDIA GPU are required")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit with nvcc is required to build the extension")
    root = source_directory()
    files = sorted(p for p in root.iterdir() if p.suffix in {".cpp", ".cu", ".h", ".cuh"})
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(str((torch.__version__, torch.version.cuda,
                       os.environ.get("TORCH_CUDA_ARCH_LIST", "auto"))).encode())
    return load(
        name="bitmask_cuda_" + digest.hexdigest()[:16],
        sources=[str(p) for p in files if p.suffix in {".cpp", ".cu"}],
        extra_include_paths=[str(root)], extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"], with_cuda=True, verbose=False,
    )
