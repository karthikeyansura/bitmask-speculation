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
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        from .reductions import run_reductions
        run_reductions(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
