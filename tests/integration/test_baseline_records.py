import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
xgr = pytest.importorskip("xgrammar")
pytest.importorskip("jsonschema")

from bitmask_speculation.benchmarking import Workload, TrialRecord, aggregate_trials, record_trial
from bitmask_speculation.decoding import greedy_decode, speculative_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState
from bitmask_speculation.models.transformers import TransformersGreedyModel


@pytest.mark.parametrize("constrained", [False, True])
def test_cpu_model_records_roundtrip_and_keep_reference_contract(constrained):
    vocabulary = ['"', "a", "b", "<eos>", "z"]
    models = []
    for seed in [7, 19]:
        torch.manual_seed(seed)
        config = transformers.GPT2Config(vocab_size=5, n_positions=32, n_embd=16, n_layer=2, n_head=2, bos_token_id=4, eos_token_id=3, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
        models.append(TransformersGreedyModel(transformers.GPT2LMHeadModel(config).eval()))
    target, draft = models
    case = Workload("tiny-enum", 'Return "a" or "b".', '{"enum":["a","b"]}', 8)
    info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[3])
    grammar = xgr.GrammarCompiler(info, max_threads=1).compile_json_schema(case.schema_json)
    fresh = lambda: XGrammarState(grammar) if constrained else None
    decode = lambda ids: "".join(vocabulary[i] for i in ids)
    metadata = {
        "source_revision": "fixture:step09", "target_revision": "fixture:gpt2-seed7",
        "draft_revision": "fixture:gpt2-seed19", "tokenizer_revision": "fixture:five-tokens",
        "device": "cpu", "precision": "float32", "dependencies": "tiny-model-fixture",
        "timing_scope": "untimed",
    }
    reference = greedy_decode(target, [4], max_new_tokens=8, eos_token_ids=[3], grammar=fresh())
    for budget in [1, 4]:
        records = []
        for repetition in range(2):
            actual = speculative_decode(target, draft, [4], max_new_tokens=8, draft_budget=budget, eos_token_ids=[3], grammar=fresh())
            record = record_trial(case, variant="c2" if constrained else "c1", result=actual, reference=reference, decode=decode, prompt_token_ids=[4], eos_token_ids=[3], metadata=metadata, draft_budget=budget, repetition=repetition)
            records.append(TrialRecord.from_json(record.to_json()))
            assert record.exact_match
            if constrained:
                assert record.schema_valid and record.complete
        summary = aggregate_trials(records)
        assert summary["exact_matches"] == 2
        assert summary["emitted"] == len(reference.token_ids) * 2
        assert summary["observed_tokens_per_second"] is None
