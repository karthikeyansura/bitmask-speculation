from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.grammar import GrammarTerminatedError
from bitmask_speculation.grammar.xgrammar import XGrammarState
from bitmask_speculation.models.transformers import TransformersGreedyModel


def compile_grammar(vocab, grammar, stop):
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[stop])
    return xgr.GrammarCompiler(info, max_threads=1).compile_grammar(grammar)


def test_real_matcher_stop_token_and_request_isolation():
    compiled = compile_grammar(["a", "b", "c", "<eos>"], 'root ::= "ab" | "ac"', 3)
    first, second = XGrammarState(compiled), XGrammarState(compiled)
    assert first.allowed_token_ids() == (0,)
    assert not first.accept_token(1)
    assert first.allowed_token_ids() == (0,)
    assert first.accept_token(0)
    assert first.allowed_token_ids() == (1, 2)
    assert second.allowed_token_ids() == (0,)
    assert first.accept_token(1)
    assert not first.is_terminated()
    assert first.allowed_token_ids() == (3,)
    assert first.accept_token(3) and first.is_terminated()
    with pytest.raises(GrammarTerminatedError):
        first.allowed_token_ids()
    with pytest.raises(GrammarTerminatedError):
        first.accept_token(3)


def test_real_matcher_termination_without_stop_token():
    compiled = compile_grammar(["a", "b", "<eos>"], 'root ::= "ab"', 2)
    state = XGrammarState(compiled, terminate_without_stop_token=True)
    assert state.accept_token(0) and state.accept_token(1)
    assert state.is_terminated()


@pytest.mark.parametrize("token", [31, 32, 34])
def test_signed_mask_word_and_partial_final_word(token):
    vocab = [f"unused{i}" for i in range(36)]
    vocab[token] = "a"
    vocab[35] = "<eos>"
    state = XGrammarState(compile_grammar(vocab, 'root ::= "a"', 35))
    assert state.allowed_token_ids() == (token,)
    assert state.accept_token(token)
    assert state.allowed_token_ids() == (35,)


def test_all_true_fast_path_ignores_stale_workspace(monkeypatch):
    state = XGrammarState(compile_grammar(["a", "<eos>"], 'root ::= "a"', 1))
    state._mask.fill_(0)
    monkeypatch.setattr(state._matcher, "fill_next_token_bitmask", lambda mask: False)
    assert state.allowed_token_ids() == (0, 1)


class JsonModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scores", torch.tensor([0.0, 2.0, 3.0, 100.0, 999.0]))
        self.config = SimpleNamespace(vocab_size=5, is_encoder_decoder=False)

    def forward(self, *, input_ids, use_cache, return_dict):
        return SimpleNamespace(logits=self.scores.expand(1, input_ids.shape[1], -1))


def test_json_schema_correction_matches_constrained_greedy():
    import json

    vocab = ['"', "red", "blue", "<eos>", "z"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[3])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_json_schema({"enum": ["red", "blue"]})
    target = TransformersGreedyModel(JsonModel().eval())
    expected = greedy_decode(target, [4], max_new_tokens=8, eos_token_ids=[3], grammar=XGrammarState(compiled))
    assert expected.token_ids == (0, 2, 0) and expected.stop_reason == "eos"
    assert json.loads(''.join(vocab[t] for t in expected.token_ids)) == "blue"
    grammar = XGrammarState(compiled)
    events = list(target.verify_prefix([4], [0, 1, 0], grammar=grammar))
    assert [(t.token_id, t.kind) for t in events] == [(0, "accepted"), (2, "correction")]
    assert grammar.allowed_token_ids() == (0,)
    tail = list(target.verify_prefix([4, 0, 2], [0, 3, 4], grammar=grammar))
    assert [t.token_id for t in tail] == [0, 3]
    assert grammar.is_terminated()


@pytest.mark.parametrize("seed", [7, 19, 41])
def test_tiny_causal_model_constrained_verification_matches_reference(seed):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(seed)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=4, n_positions=16, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=3, eos_token_id=2, pad_token_id=3,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )).eval()
    target = TransformersGreedyModel(model)
    compiled = compile_grammar(
        ["a", "b", "<eos>", "z"],
        'root ::= ("a" | "b") ("a" | "b") ("a" | "b")', 2,
    )
    for prefix in ([3], [3, 3]):
        reference = greedy_decode(
            target, prefix, max_new_tokens=8, eos_token_ids=[2], grammar=XGrammarState(compiled)
        )
        assert reference.stop_reason == "eos" and len(reference.token_ids) == 3
        for mismatch in (0, 1, 2, None):
            draft = list(reference.token_ids)
            if mismatch is not None:
                draft[mismatch] = 1 - draft[mismatch]
            grammar = XGrammarState(compiled)
            events = list(target.verify_prefix(prefix, draft, grammar=grammar))
            if mismatch is None:
                assert tuple(e.token_id for e in events) == reference.token_ids + (2,)
                assert events[-1].kind == "bonus" and grammar.is_terminated()
            else:
                assert tuple(e.token_id for e in events) == reference.token_ids[:mismatch + 1]
                assert events[-1].kind == "correction" and not grammar.is_terminated()


def test_density_matches_allowed_ids_without_advancing_state():
    state = XGrammarState(compile_grammar(["a", "b", "c", "<eos>"], 'root ::= "ab" | "ac"', 3))
    for token in (0, 1, 3):
        before = state.allowed_token_ids()
        assert state.mask_density() == len(before) / state.vocab_size
        assert state.mask_density() == len(before) / state.vocab_size
        assert state.allowed_token_ids() == before
        branch = state.fork()
        assert branch.mask_density() == state.mask_density()
        assert state.accept_token(token)
        assert branch.allowed_token_ids() == before
    with pytest.raises(GrammarTerminatedError):
        state.mask_density()


def test_density_all_valid_ignores_stale_mask(monkeypatch):
    state = XGrammarState(compile_grammar(["a", "<eos>"], 'root ::= "a"', 1))
    state._mask.fill_(0)
    monkeypatch.setattr(state._matcher, "fill_next_token_bitmask", lambda mask: False)
    assert state.mask_density() == 1.0


def test_packed_snapshot_survives_matcher_advance():
    state = XGrammarState(compile_grammar(['a','b','<eos>'], 'root ::= "ab"', 2))
    first = state.packed_mask()
    assert first.item() & 7 == 1
    state.accept_token(0)
    second = state.packed_mask()
    assert first.item() & 7 == 1 and second.item() & 7 == 2
    first.zero_()
    assert state.allowed_token_ids() == (1,)


def test_packed_snapshot_all_valid_fast_path(monkeypatch):
    state = XGrammarState(compile_grammar(['a','<eos>'], 'root ::= "a"', 1))
    state._mask.zero_()
    monkeypatch.setattr(state._matcher,'fill_next_token_bitmask',lambda mask:False)
    assert state.packed_mask().item() == -1
