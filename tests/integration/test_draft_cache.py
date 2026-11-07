"""Cached proposal rounds preserve greedy decisions and isolate rejected tails."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from bitmask_speculation.decoding import greedy_decode, speculative_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel


def model(family, seed=73):
    torch.manual_seed(seed)
    if family == 'gpt2':
        config = transformers.GPT2Config(vocab_size=17, n_positions=64, n_embd=16,
            n_layer=2, n_head=2, eos_token_id=None, bos_token_id=0,
            attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.)
        return transformers.GPT2LMHeadModel(config).eval()
    config = transformers.Qwen3Config(vocab_size=17, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, eos_token_id=None, bos_token_id=0, attention_dropout=0.)
    return transformers.Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize('family', ['gpt2', 'qwen3'])
def test_prefill_once_then_one_token_with_correct_positions(family):
    raw = model(family)
    adapter = TransformersGreedyModel(raw)
    observed = []
    def record(module, args, kw):
        if kw['use_cache']:
            observed.append((kw['input_ids'].shape[1], kw['attention_mask'].shape[1],
                             kw['position_ids'].tolist(), kw['past_key_values'] is None))
    handle = raw.register_forward_pre_hook(record, with_kwargs=True)
    try:
        for prompt in ((0, 2, 3), (4, 5)):
            prefix = prompt
            with adapter.draft_session() as session:
                for _ in range(4):
                    token = session.next_token(prefix)
                    assert token == adapter.next_token(prefix)
                    prefix += (token,)
            assert session._cache is None
            with pytest.raises(RuntimeError, match='closed'):
                session.next_token(prefix)
    finally:
        handle.remove()
    assert observed == [(3,3,[[0,1,2]],True),(1,4,[[3]],False),(1,5,[[4]],False),(1,6,[[5]],False),
                        (2,2,[[0,1]],True),(1,3,[[2]],False),(1,4,[[3]],False),(1,5,[[4]],False)]


@pytest.mark.parametrize('family', ['gpt2', 'qwen3'])
def test_cached_and_uncached_requests_match_target_after_rejections(family):
    target = TransformersGreedyModel(model(family, 19))
    raw = model(family, 41)
    cached = TransformersGreedyModel(raw)
    uncached = TransformersGreedyModel(raw, draft_cache=False)
    rejected = 0
    for prompt in ((0,), (0,2,3), (7,8)):
        for budget in (1,3,8):
            for limit in (0,1,7):
                expected = greedy_decode(target, prompt, max_new_tokens=limit)
                a = speculative_decode(target, cached, prompt, max_new_tokens=limit, draft_budget=budget)
                b = speculative_decode(target, uncached, prompt, max_new_tokens=limit, draft_budget=budget)
                assert a == b
                assert a.generation == expected
                rejected += a.accounting.verified - a.accounting.accepted
    assert rejected > 0


def test_cache_is_released_on_failure_and_next_session_restarts():
    adapter = TransformersGreedyModel(model('gpt2'))
    with pytest.raises(ValueError, match='previous choice'):
        with adapter.draft_session() as session:
            token = session.next_token((0,2))
            assert session._cache is not None
            session.next_token((0,2,(token+1)%17))
    assert session._cache is None
    with adapter.draft_session() as fresh:
        assert fresh.next_token((0,2)) == token


def test_missing_cache_fails_explicitly_and_uncached_option_still_works():
    raw = model('gpt2')
    def strip_cache(module, args, output):
        return SimpleNamespace(logits=output.logits, past_key_values=None)
    hook = raw.register_forward_hook(strip_cache)
    try:
        adapter = TransformersGreedyModel(raw)
        with pytest.raises(RuntimeError, match='past_key_values'):
            with adapter.draft_session() as session:
                session.next_token((0,))
        assert session._cache is None
        plain = TransformersGreedyModel(raw, draft_cache=False)
        with plain.draft_session() as session:
            assert session.next_token((0,)) == plain.next_token((0,))
    finally:
        hook.remove()


@pytest.mark.parametrize('reason', ['eos', 'length', 'failure'])
def test_loop_closes_draft_before_verification_and_on_failure(reason):
    from contextlib import contextmanager
    from bitmask_speculation.decoding.verification import verify_prefix
    active = []
    class Draft:
        vocab_size = 3
        @contextmanager
        def draft_session(self):
            active.append(True)
            try:
                yield self
            finally:
                active.pop()
        def next_token(self, prefix):
            assert active
            if reason == 'failure':
                raise RuntimeError('draft failure')
            return 2 if reason == 'eos' else 1
    class Target:
        vocab_size = 3
        def verify_prefix(self, prefix, proposals):
            assert not active
            return verify_prefix(proposals, lambda _:1, vocab_size=3)
    if reason == 'failure':
        with pytest.raises(RuntimeError, match='draft failure'):
            speculative_decode(Target(), Draft(), [0], max_new_tokens=2)
    else:
        result = speculative_decode(Target(), Draft(), [0], max_new_tokens=2, eos_token_ids=[2])
        assert result.token_ids == (1,1)
    assert not active
