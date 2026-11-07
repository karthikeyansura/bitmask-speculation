"""Installed generation command with explicit inputs and machine-readable output."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .decoding import DensityPolicy
from .engine import CONSTRAINED_MODES, MODES
from .models.loading import ModelSpec, TransformersEngine, schema_document


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _positive(value):
    number = _nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(prog="bitmask-speculation", description="Grammar-constrained greedy speculation")
    commands = root.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate a single continuation")
    generate.add_argument("prompt", help="raw prompt, including any pre-rendered chat template")
    generate.add_argument("--target", required=True, help="local directory or Hub model ID")
    generate.add_argument("--target-revision", help="full Hub revision SHA")
    generate.add_argument("--draft", help="local directory or Hub model ID")
    generate.add_argument("--draft-revision", help="full Hub revision SHA")
    generate.add_argument("--mode", choices=MODES, default="target_unconstrained")
    schema = generate.add_mutually_exclusive_group()
    schema.add_argument("--schema-file", type=Path, help="JSON Schema file")
    schema.add_argument("--schema-json", help="inline JSON Schema document")
    generate.add_argument("--max-new-tokens", type=_nonnegative, default=128)
    generate.add_argument("--draft-budget", type=_positive, help="fixed K for c1, c2 or c3 (default: 5)")
    generate.add_argument("--max-draft-budget", type=_positive, help="maximum C4 K (default: 8)")
    generate.add_argument("--device", default="auto", help="auto, cpu or cuda[:index] (default: auto)")
    generate.add_argument("--backend", choices=("auto", "reference", "cuda", "triton"), default="auto")
    generate.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    generate.add_argument("--local-files-only", action="store_true")
    generate.add_argument("--no-draft-cache", action="store_true",
                          help="recompute full draft prefixes instead of caching within each round")
    generate.add_argument("--no-target-cache", action="store_true",
                          help="recompute full target prefixes instead of caching across rounds")
    generate.add_argument("--output-format", choices=("json", "text"), default="json")
    return root


def main(argv=None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.mode.startswith("c") and args.draft is None:
        arguments.error("speculative modes require --draft")
    if args.draft_revision is not None and args.draft is None:
        arguments.error("--draft-revision requires --draft")
    if args.mode.startswith("target_") and args.draft is not None:
        arguments.error("target-only modes do not use --draft")
    has_schema = args.schema_file is not None or args.schema_json is not None
    if (args.mode in CONSTRAINED_MODES) != has_schema:
        arguments.error("a schema is required exactly for constrained modes")
    if args.draft_budget is not None and args.mode not in {"c1", "c2", "c3"}:
        arguments.error("--draft-budget is only used by c1, c2 and c3")
    if args.max_draft_budget is not None and args.mode != "c4":
        arguments.error("--max-draft-budget is only used by c4")
    try:
        schema = None
        if has_schema:
            raw = args.schema_file.read_text() if args.schema_file is not None else args.schema_json
            schema = schema_document(raw)
        target = ModelSpec(args.target, args.target_revision)
        draft = None if args.draft is None else ModelSpec(args.draft, args.draft_revision)
        engine = TransformersEngine.from_pretrained(
            target, draft, device=args.device, dtype=args.dtype, backend=args.backend,
            local_files_only=args.local_files_only, draft_cache=not args.no_draft_cache,
            target_cache=not args.no_target_cache,
        )
        result = engine.generate(
            args.prompt, max_new_tokens=args.max_new_tokens, mode=args.mode, schema=schema,
            draft_budget=args.draft_budget or 5,
            draft_policy=DensityPolicy(max_budget=args.max_draft_budget or 8) if args.mode == "c4" else None,
        )
        if args.output_format == "text":
            print(result.text)
        else:
            print(json.dumps({
                "mode": args.mode, "device": engine.device, "dtype": engine.dtype,
                "backend": engine.backend, "text": result.text, "token_ids": result.token_ids,
                "stop_reason": result.stop_reason, "eos_token_id": result.eos_token_id,
                "accounting": None if result.accounting is None else asdict(result.accounting),
                "schedule": [asdict(item) for item in result.schedule],
            }, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        print(f"bitmask-speculation: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Install the models and grammar extras to enable generation.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
