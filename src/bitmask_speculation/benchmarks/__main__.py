"""Explicit benchmark commands, independent of model-generation CLI startup."""

import argparse
import sys


def parser():
    root = argparse.ArgumentParser(prog='python -m bitmask_speculation.benchmarks')
    commands = root.add_subparsers(dest='command', required=True)
    kernel = commands.add_parser('reductions', help='time masked argmax and exact popcount')
    kernel.add_argument('--backend', choices=('torch', 'cuda', 'triton'), required=True)
    kernel.add_argument('--device', default='cuda:0')
    kernel.add_argument('--vocab', type=int, default=151936)
    kernel.add_argument('--rows', type=int, default=1)
    kernel.add_argument('--density', type=float, default=.01)
    kernel.add_argument('--seed', type=int, default=1729)
    kernel.add_argument('--iterations', type=int, default=20)
    kernel.add_argument('--warmups', type=int, default=5)
    kernel.add_argument('--repetitions', type=int, default=20)
    kernel.add_argument('--output', required=True)
    decode = commands.add_parser('decode', help='time speculative decoding over a workload manifest')
    decode.add_argument('--workloads', required=True)
    decode.add_argument('--target', required=True)
    decode.add_argument('--draft', required=True)
    decode.add_argument('--target-revision')
    decode.add_argument('--draft-revision')
    decode.add_argument('--device', default='cpu')
    decode.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    decode.add_argument('--backend', choices=('reference', 'cuda', 'triton'), default='reference')
    decode.add_argument('--local-files-only', action='store_true')
    decode.add_argument('--no-draft-cache', action='store_true',
                        help='recompute full draft prefixes for uncached comparisons')
    decode.add_argument('--no-target-cache', action='store_true',
                        help='recompute full target prefixes for uncached comparisons')
    decode.add_argument('--chat-template', action='store_true')
    decode.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 4, 8])
    decode.add_argument('--warmups', type=int, default=1)
    decode.add_argument('--repetitions', type=int, default=5)
    decode.add_argument('--output', required=True)
    profile = commands.add_parser('profile', help='profile warmed reductions after reference checks')
    profile.add_argument('--backend', choices=('torch', 'cuda', 'triton'), required=True)
    profile.add_argument('--device', default='cuda:0')
    profile.add_argument('--vocab', type=int, default=151936)
    profile.add_argument('--rows', type=int, default=1)
    profile.add_argument('--density', type=float, default=.01)
    profile.add_argument('--seed', type=int, default=1729)
    profile.add_argument('--iterations', type=int, default=10)
    profile.add_argument('--warmups', type=int, default=5)
    profile.add_argument('--output', required=True)
    report = commands.add_parser('report', help='recompute comparisons from completed raw trials')
    report.add_argument('run_directory')
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == 'reductions':
            from .reductions import run_reductions
            run_reductions(args)
        elif args.command == 'profile':
            from .profiling import run_profile
            run_profile(args)
        elif args.command == 'report':
            import json
            from .report import report_directory
            print(json.dumps(report_directory(args.run_directory), indent=2, allow_nan=False))
        else:
            from .decoding import run_decoding
            run_decoding(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
