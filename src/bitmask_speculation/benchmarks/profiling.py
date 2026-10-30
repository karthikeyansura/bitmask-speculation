"""Correctness-gated CUDA profiler range over installed reduction primitives."""

from .artifacts import RunDirectory, environment, gpu_environment
from .reductions import check, reduction_inputs, reduction_operation


def capture_range(operation, synchronize, start, stop, *, iterations):
    """Only the repeated operation and its device drain belong to the range."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    synchronize()
    start()
    try:
        result = None
        for _ in range(iterations):
            result = operation()
        synchronize()
        return result
    finally:
        stop()


def run_profile(args):
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('profiling requires an NVIDIA CUDA device')
    if not 0 < args.density <= 1 or not 1 <= args.vocab <= 16777216 or not 1 <= args.rows <= 65535:
        raise ValueError('invalid density, vocabulary or row count')
    if args.warmups < 1 or args.iterations < 1:
        raise ValueError('warmups and iterations must be positive')
    with RunDirectory(args.output, {'kind':'profile', 'configuration':vars(args),
            'environment':environment() | gpu_environment(device)}) as run:
        scores, mask, expected = reduction_inputs(args, device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            operation = reduction_operation(args.backend, scores, mask)
            for _ in range(args.warmups):
                check(operation(), expected)
            def start():
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push('bitmask::' + args.backend)
            def stop():
                try:
                    torch.cuda.nvtx.range_pop()
                finally:
                    torch.cuda.profiler.stop()
            actual = capture_range(operation, stream.synchronize, start, stop,
                                   iterations=args.iterations)
            check(actual, expected)
        run.write('result.json', {'reference_match':True, 'iterations':args.iterations,
                                  'valid_counts':expected[1].tolist(),
                                  'range':'unchecked reductions plus final stream synchronization'})
