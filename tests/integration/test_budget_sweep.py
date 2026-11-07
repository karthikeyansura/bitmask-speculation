from dataclasses import replace
import itertools
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")
pytest.importorskip("jsonschema")

from bitmask_speculation.benchmarking import TrialRecord, Workload, aggregate_trials
from bitmask_speculation.budget_sweep import run_budget_sweep, summarize_budget_sweep
from bitmask_speculation.decoding import DensityPolicy
from bitmask_speculation.grammar.xgrammar import XGrammarState
from bitmask_speculation.models.transformers import TransformersGreedyModel

META = {"source_revision": "fixture:budget-sweep", "target_revision": "fixture:target",
        "draft_revision": "fixture:draft", "tokenizer_revision": "fixture:five-tokens",
        "device": "cpu", "precision": "float32", "dependencies": "fixture",
        "timing_scope": "full_prefix_loop"}


class PositionModel(torch.nn.Module):
    def __init__(self, middle):
        super().__init__()
        scores = torch.zeros(12, 5)
        for i, token in enumerate([0, middle, 0] + [3] * 9):
            scores[i, token] = 10
        self.register_buffer("scores", scores)
        self.config = SimpleNamespace(vocab_size=5, is_encoder_decoder=False)

    def forward(self, *, input_ids, use_cache, return_dict):
        return SimpleNamespace(logits=self.scores[:input_ids.shape[1]].unsqueeze(0))


@pytest.fixture
def setup():
    vocab = ['"', "a", "b", "<eos>", "z"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    case = Workload("tiny-enum", 'Return "a" or "b".', '{"enum":["a","b"]}', 8)
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_json_schema(case.schema_json)
    states = []
    events = []
    ticks = itertools.count()

    def fresh():
        state = XGrammarState(compiled)
        states.append(state)
        events.append("fresh")
        return state

    def sync():
        events.append("sync")

    def clock():
        events.append("clock")
        return next(ticks) * .25

    kwargs = dict(prompt_token_ids=[4], eos_token_ids=[3], grammar_factory=fresh,
                  decode=lambda ids: ''.join(vocab[t] for t in ids), metadata=META,
                  synchronize=sync, clock=clock, fixed_budgets=[1, 4], repetitions=3,
                  policy=DensityPolicy(.25, .75, 1, 4))
    target = TransformersGreedyModel(PositionModel(1).eval(), target_cache=False)
    draft = TransformersGreedyModel(PositionModel(2).eval(), draft_cache=False)
    return target, draft, case, kwargs, states, events


def test_repeated_sweep_roundtrips_outputs_traces_and_rotates_order(setup):
    target, draft, case, kwargs, states, events = setup
    records = run_budget_sweep(target, draft, case, **kwargs)
    assert len(records) == 27
    assert len({id(s) for s in states}) == len(states)
    assert all(r.error is None and r.exact_match and r.schema_valid for r in records)
    assert all(r.decode_seconds == .25 for r in records)
    assert [r.variant for r in records[::9]] == ["target_unconstrained", "target_constrained", "c1"]
    assert [r.draft_budget for r in records[::9]] == [None, None, 1]
    for i, event in enumerate(events):
        if event == "clock":
            assert events[i - 1] == "sync"
    restored = [TrialRecord.from_json(r.to_json()) for r in records]
    assert restored == list(records)
    c4 = next(r for r in records if r.variant == "c4")
    assert c4.schedule and c4.policy == kwargs["policy"] and c4.draft_budget is None
    assert len(c4.schedule) == c4.accounting.rounds
    summaries = summarize_budget_sweep(restored)
    assert len(summaries) == 9
    assert all(s["trials"] == s["exact_matches"] == 3 for s in summaries)
    assert all(s["decode_seconds"] == .75 for s in summaries)


def test_decode_failure_retained_and_rates_suppressed(setup):
    target, draft, case, kwargs, _, _ = setup
    def fail(ids):
        raise RuntimeError("decode failed")
    records = run_budget_sweep(target, draft, case, **(kwargs | {"decode": fail}))
    assert len(records) == 27 and all("decode failed" in r.error for r in records)
    assert all(not r.schedule and r.token_ids is None for r in records)
    assert all(s["errors"] == 3 and s["observed_tokens_per_second"] is None
               for s in summarize_budget_sweep(records))


def test_policy_identity_and_trace_validation(setup):
    target, draft, case, kwargs, _, _ = setup
    records = run_budget_sweep(target, draft, case, **kwargs)
    record = next(r for r in records if r.variant == "c4")
    for changes in [{"policy": None}, {"draft_budget": 4}, {"schedule": ()},
                    {"schedule": (replace(record.schedule[0], requested_budget=99), *record.schedule[1:])}]:
        with pytest.raises((ValueError, TypeError)):
            replace(record, **changes)
    alternate = replace(record, policy=DensityPolicy(.25, .8, 1, 4), repetition=1)
    with pytest.raises(ValueError, match="identical workloads"):
        aggregate_trials([record, alternate])
    with pytest.raises(ValueError, match="repetition IDs"):
        summarize_budget_sweep(records[:-1])
    with pytest.raises(ValueError, match="same workload"):
        summarize_budget_sweep([record, replace(record, metadata_json=json.dumps({**META, "precision": "bf16"}))])


@pytest.mark.parametrize("changes", [
    {"repetitions": 1}, {"repetitions": True}, {"warmups": 0},
    {"fixed_budgets": []}, {"fixed_budgets": [1, 1]}, {"fixed_budgets": [True]},
    {"fixed_budgets": [0]}, {"policy": None}, {"synchronize": None},
    {"metadata": {**META, "timing_scope": "untimed"}},
])
def test_bad_sweep_configuration_fails_before_state_creation(setup, changes):
    target, draft, case, kwargs, states, _ = setup
    with pytest.raises((ValueError, TypeError)):
        run_budget_sweep(target, draft, case, **(kwargs | changes))
    assert states == []


def test_missing_grammar_factory_result_cannot_silently_run_unconstrained(setup):
    target, draft, case, kwargs, _, _ = setup
    with pytest.raises(ValueError, match="fresh grammar"):
        run_budget_sweep(target, draft, case, **(kwargs | {"grammar_factory": lambda: None}))


def test_schema_validation_is_outside_timing_boundary(setup):
    target, draft, case, kwargs, _, events = setup
    decode = kwargs["decode"]
    def checked_decode(ids):
        assert events[-2:] == ["sync", "clock"]
        events.append("decode")
        return decode(ids)
    records = run_budget_sweep(target, draft, case, **(kwargs | {"decode": checked_decode}))
    assert all(r.error is None for r in records)
    assert events.count("decode") == len(records)


def test_warmup_failure_aborts_before_any_timing(setup):
    target, draft, case, kwargs, _, events = setup
    def fail(*args, **kwargs):
        raise RuntimeError("warmup verifier failed")
    target.verify_prefix = fail
    with pytest.raises(RuntimeError, match="warmup verifier"):
        run_budget_sweep(target, draft, case, **kwargs)
    assert "clock" not in events


def test_nonpositive_clock_intervals_become_errors(setup):
    target, draft, case, kwargs, _, _ = setup
    records = run_budget_sweep(target, draft, case, **(kwargs | {"clock": lambda: 1.0}))
    assert all(r.error and "positive" in r.error for r in records)
    assert all(r.decode_seconds is None for r in records)


def test_persistence_callback_runs_after_validation_and_failure_aborts(setup):
    target, draft, case, kwargs, _, events = setup
    saved = []
    def save(record):
        assert events[-1] == 'clock'
        saved.append(record)
        if len(saved) == 2:
            raise OSError('disk full')
    with pytest.raises(OSError, match='disk full'):
        run_budget_sweep(target, draft, case, **kwargs, on_record=save)
    assert len(saved) == 2 and all(r.exact_match for r in saved)


def test_report_gates_speedups_and_selects_correct_baselines(setup):
    from bitmask_speculation.benchmarks.report import compare_case
    target, draft, case, kwargs, _, _ = setup
    records = list(run_budget_sweep(target, draft, case, **kwargs))
    report = compare_case(records, budgets=[1,4], repetitions=3)
    assert all(row['latency_speedup'] == 1 for row in report['configurations'])
    assert {row['baseline'] for row in report['configurations'] if row['variant']=='c1'} == {'target_unconstrained'}
    records[0] = replace(records[0], schema_valid=False)
    report = compare_case(records, budgets=[1,4], repetitions=3)
    assert all(row['latency_speedup'] is None for row in report['configurations']
               if row['variant'] in {'c1','target_unconstrained'})
    assert all(row['latency_speedup'] == 1 for row in report['configurations']
               if row['variant'] in {'c2','c3','c4','target_constrained'})
    for changed in (records[:-1], records + [records[0]], [r for r in records if r.variant != 'c1']):
        with pytest.raises(ValueError): compare_case(changed, budgets=[1,4], repetitions=3)


@pytest.mark.parametrize('changes',[{'exact_match':False}, {'stop_reason':'length','eos_token_id':None}, {'schema_valid':False}])
def test_report_suppresses_incorrect_or_incomplete_configuration(setup,changes):
    from bitmask_speculation.benchmarks.report import compare_case
    target, draft, case, kwargs, _, _ = setup
    records = list(run_budget_sweep(target, draft, case, **kwargs))
    i = next(i for i,r in enumerate(records) if r.variant == 'c2')
    records[i] = replace(records[i], **changes)
    rows = compare_case(records,budgets=[1,4],repetitions=3)['configurations']
    assert next(r for r in rows if r['variant']=='c2' and r['draft_budget']==records[i].draft_budget)['latency_speedup'] is None


def test_reference_callback_precedes_measurements(setup):
    target, draft, case, kwargs, _, events = setup
    references = []
    def save(refs):
        assert 'clock' not in events
        references.append(refs)
    records = run_budget_sweep(target, draft, case, **kwargs, on_reference=save)
    assert len(references) == 1 and set(references[0]) == {False, True}
    assert all(r.token_ids == references[0][r.variant not in {'c1','target_unconstrained'}].token_ids
               for r in records)


def test_report_rejects_mixed_cache_modes(setup):
    from bitmask_speculation.benchmarks.report import compare_case
    target, draft, case, kwargs, _, _ = setup
    records = list(run_budget_sweep(target, draft, case, **kwargs))
    changed = []
    for record in records:
        metadata = json.loads(record.metadata_json)
        if record.variant == 'c3':
            metadata.update(timing_scope='round_cached_draft_loop', draft_cache='per_round')
            record = replace(record, metadata_json=json.dumps(metadata))
        changed.append(record)
    with pytest.raises(ValueError, match='metadata differs'):
        compare_case(changed, budgets=[1,4], repetitions=3)
