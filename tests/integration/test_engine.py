import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from bitmask_speculation.decoding import DensityPolicy, greedy_decode, speculative_decode
from bitmask_speculation.engine import DecodingEngine
from bitmask_speculation.models.loading import ModelSpec, TransformersEngine, schema_document
from bitmask_speculation.grammar.xgrammar import XGrammarState


@pytest.fixture(scope='module')
def local_pair(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    root = tmp_path_factory.mktemp('engine-models')
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(alphabet + ['<eos>', '<unk>'])}
    backend = Tokenizer(models.BPE(vocab, [], unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', unk_token='<unk>')
    for name, seed in [('target', 3), ('draft', 19)]:
        torch.manual_seed(seed)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=len(vocab), n_positions=64, n_embd=16, n_layer=1, n_head=2,
            eos_token_id=vocab['<eos>'], bos_token_id=vocab['<eos>'],
        )).eval()
        path = root/name
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    return root/'target', root/'draft'


@pytest.fixture(scope='module')
def engine(local_pair):
    return TransformersEngine.from_pretrained(*(ModelSpec(str(p)) for p in local_pair), local_files_only=True)


@pytest.mark.parametrize('mode', ['target_unconstrained', 'target_constrained', 'c1', 'c2', 'c3', 'c4'])
def test_engine_functional_parity_and_request_isolation(engine, mode):
    schema = {'enum': ['a', 'b']} if mode in {'target_constrained','c2','c3','c4'} else None
    result = engine.generate('Choose:', mode=mode, schema=schema, max_new_tokens=8, draft_budget=2)
    again = engine.generate('Choose:', mode=mode, schema=schema, max_new_tokens=8, draft_budget=2)
    assert result == again
    prompt = engine.tokenizer.encode('Choose:', add_special_tokens=True)
    grammar = None if schema is None else XGrammarState(engine._compiler.compile_json_schema(json.dumps(schema)))
    common = dict(max_new_tokens=8, eos_token_ids=engine.eos_token_ids, grammar=grammar)
    if mode.startswith('target_'):
        reference = greedy_decode(engine.decoder.target, prompt, **common)
        assert result.generation == reference and result.accounting is None
    else:
        reference = speculative_decode(engine.decoder.target, engine.decoder.draft, prompt, **common,
            draft_budget=2, constrain_draft=mode in {'c3','c4'}, draft_policy=DensityPolicy() if mode=='c4' else None)
        assert (result.generation, result.accounting, result.schedule) == (reference.generation, reference.accounting, reference.schedule)
    if schema is not None:
        assert json.loads(result.text) in schema['enum']


def test_zero_budget_never_calls_models(engine, monkeypatch):
    def fail(*args, **kwargs): raise AssertionError('unexpected model forward')
    monkeypatch.setattr(engine.decoder.target, '_forward_logits', fail)
    monkeypatch.setattr(engine.decoder.draft, '_forward_logits', fail)
    for mode in ['target_unconstrained', 'c1', 'c2','c3','c4']:
        result=engine.generate('x', mode=mode, schema={'enum':['a']} if mode in {'c2','c3','c4'} else None, max_new_tokens=0)
        assert result.token_ids == () and result.text == '' and result.stop_reason == 'length'


@pytest.mark.parametrize('kwargs', [dict(mode='bad'), dict(mode='c2'), dict(schema={}),dict(mode='c1',draft_policy=DensityPolicy()),dict(max_new_tokens=-1)])
def test_request_errors(engine, kwargs):
    with pytest.raises((ValueError,TypeError)):engine.generate('x', **kwargs)


def test_target_only_loading_does_not_load_a_draft(local_pair, monkeypatch):
    original = transformers.AutoModelForCausalLM.from_pretrained
    calls=[]
    def load(*args, **kwargs):
        calls.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',load)
    engine=TransformersEngine.from_pretrained(ModelSpec(str(local_pair[0])),local_files_only=True)
    engine.generate('x',max_new_tokens=0)
    engine.generate('x',max_new_tokens=0)
    assert len(calls)==1
    assert calls[0][1]['trust_remote_code'] is False and calls[0][1]['local_files_only'] is True
    with pytest.raises(ValueError,match='draft model'):engine.generate('x',mode='c1')


def test_rejects_tokenizer_mismatch_before_model_loading(local_pair, tmp_path, monkeypatch):
    import shutil
    shutil.copytree(local_pair[1],tmp_path/'draft')
    tokenizer=transformers.AutoTokenizer.from_pretrained(tmp_path/'draft')
    tokenizer.add_tokens(['new-token']);tokenizer.save_pretrained(tmp_path/'draft')
    def fail(*args,**kwargs):raise AssertionError('models must not load')
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',fail)
    with pytest.raises(ValueError,match='tokenizer definitions'):
        TransformersEngine.from_pretrained(ModelSpec(str(local_pair[0])),ModelSpec(str(tmp_path/'draft')),local_files_only=True)


@pytest.mark.parametrize('revision',[None,'main','abc','z'*40])
def test_hub_models_require_immutable_revision(revision):
    with pytest.raises(ValueError,match='revision SHA'):ModelSpec('organization/model',revision)


def test_model_spec_accepts_full_revision():
    assert ModelSpec('organization/model','a'*40).revision=='a'*40


@pytest.mark.parametrize('schema',['[]','null','{"x":NaN}',{'$ref':'https://example.com/a'}, {'nested':{'$dynamicRef':False}}])
def test_schema_inputs_are_explicit_and_local(schema):
    with pytest.raises((ValueError,TypeError)):schema_document(schema)


def test_schema_canonicalization():
    assert schema_document({'enum':['a']})=='{"enum":["a"]}'
    assert schema_document(False)=='false'


def test_core_rejects_invalid_factory_and_mismatched_vocab(engine):
    with pytest.raises(ValueError,match='no state'):
        engine.decoder.generate([0],max_new_tokens=0,mode='target_constrained',grammar_factory=lambda:None)
    class Draft: vocab_size=2
    with pytest.raises(ValueError,match='vocabularies'):DecodingEngine(engine.decoder.target,Draft())


def test_installed_cli_from_another_directory(local_pair,tmp_path):
    import subprocess,sys
    command=[sys.executable,'-I','-m','bitmask_speculation','generate','Choose:',
        '--target',str(local_pair[0]),'--draft',str(local_pair[1]),'--mode','c3',
        '--schema-json','{"enum":["a","b"]}','--max-new-tokens','8','--local-files-only']
    process=subprocess.run(command,cwd=tmp_path,text=True,capture_output=True,timeout=60)
    assert process.returncode==0,process.stderr
    data=json.loads(process.stdout)
    assert json.loads(data['text']) in ['a','b'] and data['accounting']['emitted']==len(data['token_ids'])
    failed=subprocess.run(command+['--schema-file','missing.json'],cwd=tmp_path,text=True,capture_output=True,timeout=60)
    assert failed.returncode==2 and not failed.stdout


@pytest.mark.parametrize('uncached', [False, True])
def test_installed_decoding_benchmark_persists_trials(local_pair, tmp_path, uncached):
    pytest.importorskip('xgrammar')
    pytest.importorskip('jsonschema')
    import subprocess, sys
    from bitmask_speculation.benchmarking import TrialRecord
    (tmp_path/'schema.json').write_text('{"enum":["a","b"]}')
    manifest = tmp_path/'cases.json'
    manifest.write_text(json.dumps({'version':1, 'cases':[{'case_id':'enum', 'prompt':'Choose:',
        'schema':'schema.json', 'max_new_tokens':8}]}))
    output = tmp_path/'run'
    command = [sys.executable, '-I', '-m', 'bitmask_speculation.benchmarks', 'decode',
        '--target',str(local_pair[0]), '--draft',str(local_pair[1]), '--local-files-only',
        '--workloads',str(manifest), '--budgets','1', '--repetitions','2', '--output',str(output)]
    if uncached:
        command.append('--no-draft-cache')
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    records = [TrialRecord.from_json(line) for line in (output/'trials.jsonl').read_text().splitlines()]
    assert len(records) == 12 and all(r.exact_match for r in records)
    assert all(r.decode_seconds > 0 for r in records)
    assert json.loads((output/'status.json').read_text())['status'] == 'complete'
    assert json.loads(records[0].metadata_json)['target_revision'].startswith('sha256:')
    metadata = json.loads(records[0].metadata_json)
    assert metadata['draft_cache'] == ('disabled' if uncached else 'per_round')
    assert metadata['timing_scope'] == ('full_prefix_loop' if uncached else 'round_cached_draft_loop')
    assert metadata['target_cache'] == 'disabled'
    assert 'model_load_seconds' in json.loads((output/'setup.json').read_text())
    repeat = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert repeat.returncode == 2 and 'File exists' in repeat.stderr


def test_regime_manifest_loads_from_arbitrary_directory():
    from bitmask_speculation.benchmarking import load_workloads
    root = Path(__file__).resolve().parents[2]
    cases = load_workloads(root/'benchmarks/regimes.json')
    assert len(cases) == 6 and len({c.fingerprint for c in cases}) == 6
    assert next(c for c in cases if c.case_id == 'free-text').schema_json == '{"type":"string"}'


def test_benchmark_report_recomputes_and_rejects_partial_runs(local_pair, tmp_path):
    from bitmask_speculation.benchmarks.decoding import run_decoding
    from bitmask_speculation.benchmarks.report import report_directory
    from types import SimpleNamespace
    (tmp_path/'schema.json').write_text('{"enum":["a"]}')
    (tmp_path/'cases.json').write_text(json.dumps({'version':1,'cases':[
        {'case_id':'enum','prompt':'Choose:', 'schema':'schema.json','max_new_tokens':8}]}))
    output = tmp_path/'run'
    run_decoding(SimpleNamespace(command='decode',workloads=str(tmp_path/'cases.json'),
        target=str(local_pair[0]),draft=str(local_pair[1]),target_revision=None,draft_revision=None,
        device='cpu',dtype='float32',backend='reference',local_files_only=True,
        chat_template=False,no_draft_cache=False,budgets=[1],warmups=1,repetitions=2,output=str(output)))
    report=report_directory(output)
    assert len(report['cases'])==1 and len(report['cases'][0]['configurations'])==6
    assert len((output/'references.jsonl').read_text().splitlines()) == 1
    trials=output/'trials.jsonl'
    original=trials.read_text()
    trials.write_text('\n'.join(original.splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='coverage'):report_directory(output)
    trials.write_text(original)
    (output/'status.json').write_text('{"status":"failed"}')
    with pytest.raises(ValueError,match='completed'):report_directory(output)
