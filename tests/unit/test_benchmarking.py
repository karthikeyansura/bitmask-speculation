from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from bitmask_speculation.benchmarking import (
    TrialRecord, Workload, aggregate_trials, load_workloads, record_trial,
)
from bitmask_speculation.decoding import GreedyResult, ProposalAccounting, SpeculativeResult

META = {
    "source_revision": "fixture:step09", "target_revision": "fixture:target",
    "draft_revision": "fixture:draft", "tokenizer_revision": "fixture:tokens",
    "device": "cpu", "precision": "fixture", "dependencies": "fixture",
    "timing_scope": "full_prefix_loop",
}
CASE = Workload("enum", 'Return "blue".', '{"enum":["red","blue"]}', 8)


def trial(*, tokens=(1,), text='"blue"', reference=None, reason="eos", counts=None, **kwargs):
    generation = GreedyResult(tokens, reason, 9 if reason == "eos" else None)
    result = SpeculativeResult(generation, counts or ProposalAccounting(4, 2, 1, len(tokens), 1))
    return record_trial(
        CASE, variant="c2", result=result, reference=reference or generation,
        decode=lambda ids: text, prompt_token_ids=[0], eos_token_ids=[9],
        metadata=META, draft_budget=4, **kwargs,
    )


def test_json_roundtrip_preserves_raw_values():
    original = trial(decode_seconds=0.123456789012345)
    rebuilt = TrialRecord.from_json(original.to_json())
    assert rebuilt == original
    assert rebuilt.decode_seconds == 0.123456789012345
    assert rebuilt.accounting.proposed == 4
    assert rebuilt.complete
    assert rebuilt.recorded_at.endswith("+00:00")
    assert replace(original, metadata_json=json.dumps(dict(reversed(list(META.items()))))) == original


def test_fingerprint_covers_prompt_schema_budget_and_id():
    same = Workload("enum", 'Return "blue".', '{ "enum": ["red", "blue"] }', 8)
    assert same.fingerprint == CASE.fingerprint
    for changed in [replace(CASE, prompt="different"), replace(CASE, schema_json='{"enum":["red"]}'), replace(CASE, max_new_tokens=9), replace(CASE, case_id="other")]:
        assert changed.fingerprint != CASE.fingerprint


@pytest.mark.parametrize("text", ['"green"', '{"not":"an enum"}', 'not JSON', 'NaN', 'Infinity'])
def test_schema_validation_is_not_json_parse_only(text):
    record = trial(text=text)
    assert record.exact_match
    assert not record.schema_valid


def test_length_limited_schema_valid_result_is_still_incomplete():
    record = trial(reason="length")
    assert record.schema_valid and not record.complete
    summary = aggregate_trials([record])
    assert summary["schema_valid"] == 1
    assert summary["completed_schema_valid"] == 0
    assert summary["length_limited"] == 1
    assert summary["observed_tokens_per_second"] is None


def test_reference_mismatch_remains_visible():
    summary = aggregate_trials([trial(reference=GreedyResult((2,), "eos", 9))])
    assert summary["exact_matches"] == 0
    assert summary["schema_valid"] == 1


def test_aggregation_uses_pooled_counts_and_duration():
    a = trial(counts=ProposalAccounting(2, 1, 1, 1, 1), decode_seconds=1.0)
    b = trial(tokens=(1, 1, 1), counts=ProposalAccounting(12, 9, 0, 3, 3), decode_seconds=9.0, repetition=1)
    result = aggregate_trials([a, b])
    assert result["acceptance_rate"] == 0.1  # Not mean(1/1, 0/9) == 0.5.
    assert result["proposal_yield"] == 1 / 14
    assert result["observed_tokens_per_second"] == 0.4  # Not mean(1/1, 3/9).
    assert result["decode_seconds"] == 10.0
    assert result["emitted"] == 4 and result["rounds"] == 4


def test_empty_denominators_and_mixed_timings_remain_undefined():
    zero = trial(tokens=(), counts=ProposalAccounting(0, 0, 0, 0, 1))
    summary = aggregate_trials([zero])
    assert summary["acceptance_rate"] is None
    assert summary["proposal_yield"] is None
    assert summary["observed_tokens_per_second"] is None
    assert aggregate_trials([trial(decode_seconds=1.0), trial(repetition=1)])["decode_seconds"] is None


def test_failure_is_preserved_and_suppresses_aggregate_rates():
    failed = TrialRecord(CASE, "c2", 4, (0,), (9,), json.dumps(META), repetition=1, error="RuntimeError: model unavailable")
    assert TrialRecord.from_json(failed.to_json()) == failed
    result = aggregate_trials([trial(decode_seconds=1), failed])
    assert result["trials"] == 2 and result["errors"] == 1
    assert result["observed_tokens_per_second"] is None
    assert result["acceptance_rate"] is None
    with pytest.raises(ValueError, match="failed trials"):
        replace(trial(), error="cannot claim both")


@pytest.mark.parametrize("change", [
    {"workload": replace(CASE, prompt="different")}, {"draft_budget": 5},
    {"prompt_token_ids": (1,)}, {"eos_token_ids": (9, 10)},
    {"metadata_json": json.dumps({**META, "target_revision": "different"})},
    {"variant": "c1"},
])
def test_mixed_experiments_cannot_be_aggregated(change):
    with pytest.raises(ValueError, match="identical"):
        aggregate_trials([trial(), replace(trial(repetition=1), **change)])


def test_duplicate_repetitions_and_empty_input_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_trials([trial(), trial()])
    with pytest.raises(ValueError, match="at least one"):
        aggregate_trials([])


@pytest.mark.parametrize("seconds", [0, -1, math.nan, math.inf, -math.inf])
def test_invalid_timings_rejected(seconds):
    with pytest.raises(ValueError):
        trial(decode_seconds=seconds)


@pytest.mark.parametrize("change", [
    {"variant": "unknown"}, {"draft_budget": 0}, {"prompt_token_ids": ()},
    {"token_ids": (9,)}, {"eos_token_id": 8}, {"token_ids": tuple(range(20))},
    {"metadata_json": "{}"}, {"recorded_at": "2026-11-06T12:00:00"},
    {"accounting": None}, {"stop_reason": "unknown"},
    {"accounting": ProposalAccounting(1, 1, 1, 2, 1)},
])
def test_inconsistent_records_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        replace(trial(), **change)


def test_boolean_numeric_fields_are_not_counts_or_timings():
    for change in [{"repetition": True}, {"draft_budget": True}, {"decode_seconds": True}, {"prompt_token_ids": (False,)}]:
        with pytest.raises(TypeError):
            replace(trial(), **change)


def test_target_only_record_has_no_proposal_rate():
    r = replace(trial(), variant="target_constrained", draft_budget=None, accounting=None)
    assert aggregate_trials([r])["acceptance_rate"] is None


def test_version_and_extra_fields_are_rejected():
    data = json.loads(trial().to_json())
    for version in [2, True]:
        with pytest.raises(ValueError):
            TrialRecord.from_json(json.dumps({**data, "version": version}))
    with pytest.raises(TypeError):
        TrialRecord.from_json(json.dumps({**data, "invented": 1}))


def test_manifest_is_location_independent_and_workloads_validate(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    manifest = root / "benchmarks/baseline.json"
    monkeypatch.chdir(tmp_path)
    cases = load_workloads(manifest)
    assert [c.case_id for c in cases] == ["contact-card", "tool-search", "nested-users"]
    assert len({c.prompt for c in cases}) == len(cases)
    assert len({c.fingerprint for c in cases}) == len(cases)
    outputs = [
        {"name": "Alice", "age": 30, "city": "Seattle", "email": "alice@example.com"},
        {"function": "search", "arguments": {"query": "CUDA masked argmax"}},
        {"users": [{"name": "Alice", "address": {"city": "Seattle", "country": "USA"}, "tags": ["systems"]}], "total_count": 1, "page": 1},
    ]
    for case, value in zip(cases, outputs):
        result = GreedyResult((1,), "eos", 9)
        record = record_trial(case, variant="target_constrained", result=result, reference=result, decode=lambda ids: json.dumps(value), prompt_token_ids=[0], eos_token_ids=[9], metadata=META)
        assert record.schema_valid


def test_manifest_rejects_duplicate_ids_and_path_escape(tmp_path):
    (tmp_path / "schema.json").write_text('{"type":"string"}')
    entry = {"case_id": "x", "prompt": "x", "schema": "schema.json", "max_new_tokens": 8}
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"version": 1, "cases": [entry, entry]}))
    with pytest.raises(ValueError, match="unique"):
        load_workloads(manifest)
    manifest.write_text(json.dumps({"version": 1, "cases": [{**entry, "schema": "../outside.json"}]}))
    with pytest.raises(ValueError, match="inside"):
        load_workloads(manifest)


@pytest.mark.parametrize("schema", ['{"$ref":"https://example.com/schema.json"}', '{"$schema":"draft-07"}', '[]'])
def test_unsupported_schema_references_and_dialect_rejected(schema):
    with pytest.raises(ValueError):
        Workload("bad", "prompt", schema, 8)


def test_decode_failure_is_not_mislabeled_invalid_json():
    def broken_decode(tokens):
        raise ValueError("decoder failed")
    result = GreedyResult((1,), "eos", 9)
    with pytest.raises(ValueError, match="decoder failed"):
        record_trial(CASE, variant="target_constrained", result=result, reference=result, decode=broken_decode, prompt_token_ids=[0], eos_token_ids=[9], metadata=META)


def test_untimed_metadata_cannot_label_a_timing():
    with pytest.raises(ValueError, match="untimed"):
        replace(trial(decode_seconds=1), metadata_json=json.dumps({**META, "timing_scope": "untimed"}))


def test_c3_records_roundtrip_without_pooling_c2():
    c2 = trial()
    c3 = replace(c2, variant="c3", repetition=1)
    assert TrialRecord.from_json(c3.to_json()) == c3
    assert aggregate_trials([c3])["exact_matches"] == 1
    with pytest.raises(ValueError):
        aggregate_trials([c2, c3])
    for change in ({"draft_budget": None}, {"accounting": None}):
        with pytest.raises((ValueError, TypeError)):
            replace(c3, **change)


def test_original_version_one_records_remain_readable():
    record = trial()
    payload = json.loads(record.to_json())
    payload.pop("policy")
    payload.pop("schedule")
    assert TrialRecord.from_json(json.dumps(payload)) == record
