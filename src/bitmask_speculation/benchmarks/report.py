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
    if len({r.metadata_json for r in records}) != 1:
        raise ValueError('execution metadata differs across compared configurations')
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
    metadata = json.loads(records[0].metadata_json)
    execution = {key: metadata[key] for key in ('timing_scope', 'draft_cache', 'target_cache') if key in metadata}
    return {'case_id': records[0].workload.case_id, 'fingerprint': records[0].workload.fingerprint,
            'execution': execution,
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


def _cell(value, spec):
    return '-' if value is None else format(value, spec)


def format_report(report):
    """Render per-case tables: exactness, acceptance, yield, throughput and ratios."""
    lines = []
    for case in report['cases']:
        execution = ', '.join(f'{k}={v}' for k, v in case['execution'].items())
        lines += [f"{case['case_id']}" + (f'  ({execution})' if execution else ''),
                  f"  {'configuration':<22}{'exact':>7}{'accept':>8}{'yield':>8}{'tok/s':>8}{'vs target':>11}"]
        rows = {(row['variant'], row['draft_budget']): row for row in case['configurations']}
        for row in case['configurations']:
            name = row['variant'] + ('' if row['draft_budget'] is None else f" K={row['draft_budget']}")
            exact = f"{row['exact_matches']}/{row['trials']}"
            speedup = row['latency_speedup']
            ratio = '' if row['variant'].startswith('target_') else '-' if speedup is None else f'{speedup:.3f}x'
            lines.append(f"  {name:<22}{exact:>7}{_cell(row['acceptance_rate'], '.2f'):>8}"
                         f"{_cell(row['proposal_yield'], '.2f'):>8}"
                         f"{_cell(row['observed_tokens_per_second'], '.1f'):>8}{ratio:>11}")
        c4 = rows.get(('c4', None))
        for (variant, budget), c3 in sorted((key, row) for key, row in rows.items() if key[0] == 'c3'):
            if c4 and c3['speedup_eligible'] and c4['speedup_eligible']:
                lines.append(f"  C4/C3 K={budget}: {c3['decode_seconds'] / c4['decode_seconds']:.3f}x")
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'
