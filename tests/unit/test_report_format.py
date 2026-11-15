from bitmask_speculation.benchmarks.report import format_report


def row(variant, budget, seconds, eligible=True, accept=None, yield_=None, tps=None, speedup=None):
    return dict(variant=variant, draft_budget=budget, trials=2, exact_matches=2 if eligible else 1,
                acceptance_rate=accept, proposal_yield=yield_, observed_tokens_per_second=tps,
                decode_seconds=seconds, speedup_eligible=eligible, latency_speedup=speedup)


def test_table_lists_configurations_and_c4_over_c3():
    report = {'cases': [{'case_id': 'enum', 'execution': {'timing_scope': 'full_prefix_loop'}, 'configurations': [
        row('target_constrained', None, 2.0, tps=20.0),
        row('c3', 4, 1.6, accept=1.0, yield_=0.9, tps=25.0, speedup=1.25),
        row('c3', 8, 1.0, eligible=False),
        row('c4', None, 1.28, accept=1.0, yield_=0.95, tps=31.2, speedup=1.5625),
    ]}]}
    text = format_report(report)
    assert text.splitlines()[0] == 'enum  (timing_scope=full_prefix_loop)'
    assert '  c3 K=4                    2/2    1.00    0.90    25.0     1.250x' in text
    assert '  c3 K=8                    1/2       -       -       -          -' in text
    assert 'C4/C3 K=4: 1.250x' in text and 'C4/C3 K=8' not in text
