"""Preserve the case order, raw prompts, schema property order and output caps."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from bitmask_speculation.benchmarking import load_workloads


def test_primary_manifest_matches_the_six_source_cases():
    manifest = Path(__file__).resolve().parents[2] / "benchmarks" / "primary.json"
    cases = load_workloads(manifest)
    entries = json.loads(manifest.read_text())["cases"]
    rows = [
        (case.case_id, case.prompt, json.loads((manifest.parent / entry["schema"]).read_text()), case.max_new_tokens)
        for case, entry in zip(cases, entries, strict=True)
    ]
    serialized = json.dumps(rows, separators=(",", ":")).encode()
    assert len(cases) == 6
    assert hashlib.sha256(serialized).hexdigest() == "1972100a6d128d8d6d2e91701b03a6cc6f363be65367400a32a6cce2ecc24116"
