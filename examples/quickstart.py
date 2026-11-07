"""Run the text API against two locally saved, tokenizer-compatible models."""
import argparse
from pathlib import Path

from bitmask_speculation import ModelSpec, TransformersEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('draft', type=Path)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if not args.target.is_dir() or not args.draft.is_dir():
        parser.error('target and draft must be local model directories')
    engine = TransformersEngine.from_pretrained(
        ModelSpec(str(args.target.resolve())), ModelSpec(str(args.draft.resolve())),
        device=args.device, local_files_only=True,
    )
    result = engine.generate('Return a colour:', mode='c3', schema={'enum':['red','blue']}, max_new_tokens=32)
    print(result.text)
    print(result.stop_reason, result.accounting)


if __name__ == '__main__':
    main()
