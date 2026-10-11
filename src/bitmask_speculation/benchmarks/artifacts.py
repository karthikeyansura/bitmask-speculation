"""Exclusive run directories, source fingerprints and lossless measurements."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    dependencies = {}
    for name in ('bitmask-speculation', 'torch', 'transformers', 'xgrammar', 'triton', 'jsonschema'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob('*')) if p.suffix in {'.py', '.cu', '.cpp', '.h', '.cuh'}}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': dependencies, 'source_files': sources,
            'source_sha256': sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}


def gpu_environment(device):
    import torch
    props = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = 'unavailable'
    return {'device': str(device), 'gpu': props.name, 'memory_bytes': props.total_memory,
            'capability': list(torch.cuda.get_device_capability(device)),
            'torch_cuda': torch.version.cuda, 'driver_inventory': driver}


def distribution(samples):
    values = sorted(samples)
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('timings must be finite positive seconds')
    return {'samples': len(values), 'median_seconds': statistics.median(values),
            'minimum_seconds': values[0], 'p95_seconds': values[math.ceil(.95 * len(values)) - 1]}


class RunDirectory:
    """A failed run keeps its completed records and a distinct failure marker."""

    def __init__(self, path, manifest):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.write('manifest.json', {'version': 1, 'started_at': utc_now(), **manifest})

    def write(self, name, value):
        with (self.path / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, name, payload):
        # Flush each completed trial outside its measured interval.
        with (self.path / name).open('a', encoding='utf-8') as stream:
            stream.write(payload + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.write('status.json', {'finished_at': utc_now(),
                                  'status': 'complete' if kind is None else 'failed',
                                  'error': None if kind is None else f'{kind.__name__}: {error}'})
        return False
