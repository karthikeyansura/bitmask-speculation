"""Recompute case-level comparisons from raw, balanced trial records."""

import json
from pathlib import Path

from ..benchmarking import TrialRecord
from ..budget_sweep import summarize_budget_sweep


def compare_case(records, *, budgets, repetitions):
    if repetitions < 2 or not budgets or len(set(budgets)) != len(budgets) or min(budgets) < 1:
        raise ValueError('invalid expected sweep configuration')
    expected = {('target_unconstrained', None), ('target_constrained', None), ('c4', None)}
    expected.update((mode, k) for mode in ('c1', 'c2', 'c3') for k in budgets)
    if {(r.variant, r.draft_budget) for r in records} != expected:
        raise ValueError('missing or unexpected sweep configurations')
    groups = {}
    for record in records:
        groups.setdefault((record.variant, record.draft_budget), []).append(record)
    for group in groups.values():
        if len(group) != repetitions or {r.repetition for r in group} != set(range(repetitions)):
            raise ValueError('incomplete or duplicate repetition coverage')
    summaries = summarize_budget_sweep(records)
    def eligible(row):
        return (row['errors'] == 0 and row['completed'] == row['exact_matches'] ==
                row['schema_valid'] == row['trials'] and row['decode_seconds'] is not None)
    references = {row['variant']: row for row in summaries if row['variant'].startswith('target_')}
    for row in summaries:
        baseline = 'target_unconstrained' if row['variant'] in {'c1', 'target_unconstrained'} else 'target_constrained'
        reference = references[baseline]
        allowed = eligible(row) and eligible(reference)
        row.update(baseline=baseline, speedup_eligible=allowed,
                   latency_speedup=reference['decode_seconds'] / row['decode_seconds'] if allowed else None)
    return {'case_id': records[0].workload.case_id, 'fingerprint': records[0].workload.fingerprint,
            'configurations': summaries}


def report_directory(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    status = json.loads((path/'status.json').read_text())
    if manifest.get('kind') != 'decoding' or status.get('status') != 'complete':
        raise ValueError('report requires a completed decoding run')
    config = manifest['configuration']
    cases = {}
    for line in (path/'trials.jsonl').read_text().splitlines():
        record = TrialRecord.from_json(line)
        cases.setdefault(record.workload.case_id, []).append(record)
    expected = {case['case_id']: case for case in manifest['workloads']}
    if len(expected) != len(manifest['workloads']) or cases.keys() != expected.keys():
        raise ValueError('workload coverage differs from manifest')
    from dataclasses import asdict
    for name, records in cases.items():
        if any(asdict(record.workload) != expected[name] for record in records):
            raise ValueError('workload content differs from manifest')
    return {'version': 1, 'aggregation': 'per-case ratios of total matched-run latency; no cross-case average',
            'cases': [compare_case(records, budgets=config['budgets'], repetitions=config['repetitions'])
                      for records in cases.values()]}
