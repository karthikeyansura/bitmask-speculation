"""Equivalent masked argmax + count reductions with explicit timing scopes."""

import json
from time import perf_counter

from .artifacts import RunDirectory, distribution, environment, gpu_environment


def torch_reduce(logits, mask):
    import torch
    ids = torch.arange(logits.shape[1], device=logits.device)
    valid = ((mask[:, ids // 32].to(torch.int64) >> (ids % 32)) & 1).bool()
    counts = valid.sum(1, dtype=torch.int64)
    maximum = logits.masked_fill(~valid, -float('inf')).amax(1, keepdim=True)
    tokens = torch.where(valid & (logits == maximum), ids, logits.shape[1]).amin(1)
    status = torch.where(torch.isnan(logits).any(1), 2, torch.where(counts == 0, 1, 0)).int()
    return tokens, counts, status


def check(actual, expected):
    import torch
    for value, reference in zip(actual, expected, strict=True):
        if not torch.equal(value, reference):
            raise ValueError('reduction differs from the reference')


def measure_wall(operation, synchronize, *, iterations, checked=False, clock=perf_counter):
    synchronize()
    start = clock()
    result = None
    for _ in range(iterations):
        result = operation()
        if checked and result[2].any().item():
            raise ValueError('reduction status is nonzero')
    synchronize()
    elapsed = (clock() - start) / iterations
    distribution([elapsed])
    return elapsed, result


def run_reductions(args):
    import torch
    if args.device == 'cpu' or not torch.cuda.is_available():
        raise ValueError('reduction timing requires an NVIDIA CUDA device')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('reduction timing requires CUDA')
    if not 0 < args.density <= 1 or not 1 <= args.vocab <= 16777216 or not 1 <= args.rows <= 65535:
        raise ValueError('invalid density, vocabulary or row count')
    if min(args.iterations, args.warmups) < 1 or args.repetitions < 2:
        raise ValueError('use positive iterations/warmups and at least two repetitions')
    metadata = environment()
    metadata.update(gpu_environment(device))
    with RunDirectory(args.output, {'kind': 'reductions', 'configuration': vars(args),
                                    'environment': metadata}) as run:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        scores = torch.randn((args.rows, args.vocab), device=device, generator=generator)
        valid = torch.rand((args.rows, args.vocab), device=device, generator=generator) < args.density
        valid[:, 0] = True
        words = (args.vocab + 31) // 32
        padded = torch.nn.functional.pad(valid, (0, words * 32 - args.vocab))
        shifts = torch.arange(32, dtype=torch.int64, device=device)
        mask = (padded.reshape(args.rows, words, 32).long() << shifts).sum(2).int().contiguous()
        expected = torch_reduce(scores, mask)
        run.write('inputs.json', {'seed': args.seed, 'valid_counts': expected[1].tolist(),
                                 'density_note': 'Bernoulli bits; token zero forced valid'})
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        samples = []
        with torch.cuda.device(device), torch.cuda.stream(stream):
            if args.backend == 'torch':
                operation = lambda: torch_reduce(scores, mask)
            elif args.backend == 'cuda':
                from ..kernels._extension import load_extension
                native = load_extension()
                def operation():
                    pairs = [native.masked_argmax(scores[i], mask[i]) for i in range(args.rows)]
                    counts = [native.mask_count(mask[i], args.vocab) for i in range(args.rows)]
                    return torch.stack([p[0] for p in pairs]), torch.stack(counts), torch.stack([p[1] for p in pairs])
            else:
                from ..kernels.triton import batched_masked_reduce
                from ..kernels.workspace import ReductionWorkspace
                workspace = ReductionWorkspace(device)
                def operation():
                    result = batched_masked_reduce(scores, mask, workspace=workspace)
                    return result.token_ids, result.valid_counts, result.status
            for _ in range(args.warmups):
                check(operation(), expected)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.iterations):
                    captured = operation()
            graph.replay()
            stream.synchronize()
            check(captured, expected)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            scopes = ('graph_device', 'async_wrapper', 'checked_wrapper')
            for repetition in range(args.repetitions):
                offset = repetition % len(scopes)
                for scope in scopes[offset:] + scopes[:offset]:
                    if scope == 'graph_device':
                        start_event.record(stream)
                        graph.replay()
                        end_event.record(stream)
                        end_event.synchronize()
                        elapsed = start_event.elapsed_time(end_event) / 1000 / args.iterations
                        actual = captured
                    else:
                        elapsed, actual = measure_wall(operation, stream.synchronize,
                            iterations=args.iterations, checked=scope == 'checked_wrapper')
                    check(actual, expected)
                    distribution([elapsed])
                    record = {'scope': scope, 'repetition': repetition, 'seconds_per_batch': elapsed}
                    samples.append(record)
                    run.append('samples.jsonl', json.dumps(record))
            run.write('summary.json', {scope: distribution([r['seconds_per_batch'] for r in samples
                                                           if r['scope'] == scope]) for scope in scopes})
