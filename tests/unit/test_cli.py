import json

import pytest

from bitmask_speculation import cli
from bitmask_speculation.decoding import GreedyResult
from bitmask_speculation.engine import GenerationResult


@pytest.fixture
def fake_loader(monkeypatch):
    calls=[]
    class Engine:
        def generate(self, prompt, **kwargs):
            calls.append(('generate',prompt,kwargs))
            return GenerationResult(GreedyResult((1,),'eos',2),text='a')
    def load(*args,**kwargs):
        calls.append(('load',args,kwargs));return Engine()
    monkeypatch.setattr(cli.TransformersEngine,'from_pretrained',load)
    return calls


def base(tmp_path):return ['generate','prompt','--target',str(tmp_path)]


def test_json_output_and_loader_options(tmp_path,fake_loader,capsys):
    assert cli.main(base(tmp_path)+['--local-files-only'])==0
    output=capsys.readouterr();data=json.loads(output.out)
    assert data['token_ids']==[1] and data['stop_reason']=='eos' and data['text']=='a'
    assert data['accounting'] is None and data['schedule']==[] and not output.err
    assert fake_loader[0][2]['local_files_only'] is True


def test_schema_file_and_c4_policy(tmp_path,fake_loader,capsys):
    path=tmp_path/'schema.json';path.write_text('{"enum":["a"]}')
    args=base(tmp_path)+['--draft',str(tmp_path),'--mode','c4','--schema-file',str(path),'--max-draft-budget','3','--output-format','text']
    assert cli.main(args)==0 and capsys.readouterr().out=='a\n'
    assert fake_loader[1][2]['schema']=='{"enum":["a"]}'
    assert fake_loader[1][2]['draft_policy'].max_budget==3


@pytest.mark.parametrize('extra',[
    ['--mode','c1'], ['--mode','target_constrained'], ['--schema-json','{}'],
    ['--max-new-tokens','-1'], ['--draft-budget','0'], ['--draft-budget','4'],
    ['--max-draft-budget','4'], ['--draft-revision','a'*40],
    ['--schema-json','{}','--schema-file','a.json'],
])
def test_usage_errors_do_not_load_models(tmp_path,fake_loader,capsys,extra):
    with pytest.raises(SystemExit) as exc:cli.main(base(tmp_path)+extra)
    assert exc.value.code==2 and not fake_loader
    assert capsys.readouterr().out==''


@pytest.mark.parametrize('schema',['[]','{invalid','{"$ref":"https://example.com/x"}'])
def test_invalid_schema_fails_before_loading(tmp_path,fake_loader,capsys,schema):
    assert cli.main(base(tmp_path)+['--mode','target_constrained','--schema-json',schema])==1
    output=capsys.readouterr();assert not output.out and output.err and not fake_loader


def test_missing_schema_file(tmp_path,fake_loader,capsys):
    assert cli.main(base(tmp_path)+['--mode','target_constrained','--schema-file',str(tmp_path/'missing.json')])==1
    assert not fake_loader and capsys.readouterr().err


def test_runtime_error_is_not_success(tmp_path,monkeypatch,capsys):
    def fail(*args,**kwargs):raise RuntimeError('model load failed')
    monkeypatch.setattr(cli.TransformersEngine,'from_pretrained',fail)
    assert cli.main(base(tmp_path))==1
    output=capsys.readouterr();assert not output.out and 'model load failed' in output.err


def test_explicit_backend_reaches_loader(tmp_path,fake_loader,capsys):
    assert cli.main(base(tmp_path)+['--backend','triton','--device','cuda']) == 0
    assert fake_loader[0][2]['backend'] == 'triton'


@pytest.mark.parametrize('uncached', [False, True])
def test_draft_cache_option_reaches_loader(tmp_path, fake_loader, capsys, uncached):
    options = ['--no-draft-cache'] if uncached else []
    assert cli.main(base(tmp_path) + options) == 0
    assert fake_loader[0][2]['draft_cache'] is not uncached
