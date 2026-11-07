from types import SimpleNamespace
import pytest
pytestmark=pytest.mark.gpu


@pytest.mark.parametrize('backend',['cuda','triton'])
@pytest.mark.parametrize('mode',['target_constrained','c2','c3','c4'])
def test_constrained_engine_matches_reference(backend,mode):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('NVIDIA CUDA required')
    import xgrammar as xgr
    from bitmask_speculation.models.transformers import TransformersGreedyModel
    from bitmask_speculation.engine import DecodingEngine
    from bitmask_speculation.grammar.xgrammar import XGrammarState
    from bitmask_speculation.decoding.scheduling import DensityPolicy
    class Model(torch.nn.Module):
        def __init__(self,values):
            super().__init__()
            self.register_buffer('values',torch.tensor(values,device='cuda',dtype=torch.float32))
            self.config=SimpleNamespace(vocab_size=4,is_encoder_decoder=False)
        def forward(self,input_ids,**kwargs):
            return SimpleNamespace(logits=self.values.expand(1,input_ids.shape[1],4))
    target=Model([4,2,1,0]).eval();draft=Model([4,1,3,0]).eval()
    info=xgr.TokenizerInfo(['a','b','c','<eos>'],stop_token_ids=[3])
    compiled=xgr.GrammarCompiler(info,max_threads=1).compile_grammar('root ::= "ab" | "ac"')
    kwargs=dict(mode=mode,max_new_tokens=8,eos_token_ids=(3,),
                grammar_factory=lambda:XGrammarState(compiled))
    if mode=='c4':kwargs['draft_policy']=DensityPolicy()
    def run(name):
        engine=DecodingEngine(TransformersGreedyModel(target,backend=name,target_cache=False),
                              TransformersGreedyModel(draft,backend=name,draft_cache=False))
        return engine.generate((3,),**kwargs)
    actual,expected=run(backend),run('reference')
    assert actual==expected
    assert actual.token_ids==(0,1) and actual.stop_reason=='eos'
