"""Caller-owned, stream-bound scratch for Triton reductions."""

from contextlib import contextmanager
from threading import Lock


class ReductionWorkspace:
    """Reuse partial reductions, never result tensors.

    One workspace belongs to one CUDA device and its construction stream. Calls
    on other streams fail explicitly. A nonblocking lock prevents overlapping
    host submissions; same-stream GPU submissions remain ordered asynchronously.
    Growth retains the largest allocation until clear() or object destruction.
    """

    def __init__(self, device):
        import torch
        device = torch.device(device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('workspace requires CUDA')
        self.device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
        self._stream = torch.cuda.current_stream(self.device)
        self._lock = Lock()
        self._buffers = None
        self.capacity = 0
        self.allocation_count = 0

    def _check_stream(self):
        import torch
        if torch.cuda.current_stream(self.device) != self._stream:
            raise ValueError('workspace belongs to a different CUDA stream')

    @contextmanager
    def acquire(self, device, blocks):
        import torch
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
            raise ValueError('workspace capacity must be a positive integer')
        if torch.device(device) != self.device:
            raise ValueError('workspace belongs to a different CUDA device')
        self._check_stream()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('workspace is already in use')
        try:
            if blocks > self.capacity:
                with torch.cuda.device(self.device):
                    self._buffers = (torch.empty(blocks,device=self.device,dtype=torch.float32),
                                     *(torch.empty(blocks,device=self.device,dtype=torch.int32) for _ in range(3)))
                self.capacity = blocks
                self.allocation_count += 1
            yield self._buffers
        finally:
            self._lock.release()

    def clear(self):
        """Release cached scratch on its bound stream; counters remain cumulative."""
        self._check_stream()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('workspace is already in use')
        try:
            self._buffers = None
            self.capacity = 0
        finally:
            self._lock.release()
