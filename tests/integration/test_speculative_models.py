import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bitmask_speculation.decoding import greedy_decode, speculative_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel


def tiny_model(seed, vocab_size=7):
    torch.manual_seed(seed)
    return TransformersGreedyModel(transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=vocab_size, n_positions=32, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=0, eos_token_id=None, pad_token_id=0,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )).eval())


@pytest.mark.parametrize("seed", [7, 19, 41])
def test_distinct_causal_models_match_target_greedy_across_budgets(seed):
    target, draft = tiny_model(seed), tiny_model(seed + 100)
    for prompt in ([0], [0, 2, 3]):
        ordinary = greedy_decode(target, prompt, max_new_tokens=7)
        # Make the target's first choice an EOS in one request to exercise early stop.
        for eos in ((), (ordinary.token_ids[0],)):
            for limit in (0, 1, 2, 7):
                expected = greedy_decode(target, prompt, max_new_tokens=limit, eos_token_ids=eos)
                for budget in (1, 3, 10):
                    result = speculative_decode(
                        target, draft, prompt, max_new_tokens=limit,
                        draft_budget=budget, eos_token_ids=eos,
                    )
                    assert result.generation == expected


@pytest.mark.parametrize("terminate_without_stop", [False, True])
@pytest.mark.parametrize("seed", [7, 19, 41])
def test_xgrammar_request_matches_target_reference(seed, terminate_without_stop):
    xgr = pytest.importorskip("xgrammar")
    from bitmask_speculation.grammar.xgrammar import XGrammarState

    vocab = ["a", "b", "<eos>", "z"]
    info = xgr.TokenizerInfo(vocab, stop_token_ids=[2])
    compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(
        'root ::= ("a" | "b") ("a" | "b") ("a" | "b")'
    )
    target, draft = tiny_model(seed, 4), tiny_model(seed + 100, 4)

    def state():
        return XGrammarState(compiled, terminate_without_stop_token=terminate_without_stop)

    for prompt in ([3], [3, 3]):
        for limit in (0, 1, 2, 3, 4, 8):
            reference_state = state()
            expected = greedy_decode(
                target, prompt, max_new_tokens=limit, eos_token_ids=[2], grammar=reference_state
            )
            for budget in (1, 2, 8):
                actual_state = state()
                result = speculative_decode(
                    target, draft, prompt, max_new_tokens=limit, draft_budget=budget,
                    eos_token_ids=[2], grammar=actual_state,
                )
                assert result.generation == expected
                assert actual_state.is_terminated() == reference_state.is_terminated()
                if not actual_state.is_terminated():
                    assert actual_state.allowed_token_ids() == reference_state.allowed_token_ids()
                assert all(vocab[t] in ("a", "b") for t in result.token_ids)
