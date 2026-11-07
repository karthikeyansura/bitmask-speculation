"""Request-scoped target caches match full-prefix verification and release state."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from bitmask_speculation.decoding import DensityPolicy, greedy_decode, speculative_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel

V = 17


def model(family, seed=19, **options):
    torch.manual_seed(seed)
    if family == 'gpt2':
        config = transformers.GPT2Config(vocab_size=V, n_positions=96, n_embd=16,
            n_layer=2, n_head=2, eos_token_id=None, bos_token_id=0,
            attn_pdrop=0., embd_pdrop=0., resid_pdrop=0.)
        return transformers.GPT2LMHeadModel(config).eval()
    config = transformers.Qwen3Config(vocab_size=V, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=96, eos_token_id=None, bos_token_id=0, attention_dropout=0.,
        **options)
    return transformers.Qwen3ForCausalLM(config).eval()


def pair(family, seed=19):
    raw = model(family, seed)
    return TransformersGreedyModel(raw, target_cache=True), TransformersGreedyModel(raw)


def record_calls(raw):
    calls = []
    def record(module, args, kw):
        if not kw['use_cache']:
            return
        cache = kw['past_key_values']
        calls.append((kw['input_ids'][0].tolist(), kw['position_ids'][0].tolist(),
                      kw['attention_mask'].shape[1], cache.get_seq_length()))
    return calls, raw.register_forward_pre_hook(record, with_kwargs=True)


def capture_sessions(adapter):
    sessions, original = [], adapter.target_session
    @contextmanager
    def capture():
        with original() as session:
            sessions.append(session)
            yield session
    adapter.target_session = capture
    return sessions


class ScriptedDraft:
    """Propose the target path, replacing round index ``mismatch`` with a wrong token."""

    vocab_size = V

    def __init__(self, prompt, path, mismatch=None, fail_round=None):
        self.expected = tuple(prompt) + tuple(path)
        self.mismatch, self.fail_round, self.rounds, self.start = mismatch, fail_round, 0, None

    @contextmanager
    def draft_session(self):
        self.rounds += 1
        self.start = None
        yield self

    def next_token(self, prefix):
        if self.rounds == self.fail_round:
            raise RuntimeError('draft failure')
        if self.start is None:
            self.start = len(prefix)
        if len(prefix) >= len(self.expected) or prefix != self.expected[:len(prefix)]:
            return 1
        token = self.expected[len(prefix)]
        return (token + 1) % V if len(prefix) - self.start == self.mismatch else token


@pytest.mark.parametrize('family', ['gpt2', 'qwen3'])
def test_rounds_feed_uncached_tokens_and_crop_rejected_suffixes(family):
    cached, uncached = pair(family)
    raw = cached._model
    prompt = (0, 2, 3)
    t = greedy_decode(uncached, prompt, max_new_tokens=8).token_ids
    wrong = lambda token: (token + 1) % V
    rounds = [
        (prompt, (t[0], wrong(t[1]), 5)),          # mid rejection
        (prompt + t[:2], (t[2], t[3])),              # full acceptance with bonus
        (prompt + t[:5], (wrong(t[5]),)),            # first-token rejection
        (prompt + t[:6], ()),                        # empty proposals
        ((4, 5), (6,)),                              # unrelated prefix
    ]
    calls, hook = record_calls(raw)
    try:
        with cached.target_session() as session:
            for prefix, proposals in rounds:
                actual = list(session.verify_prefix(prefix, proposals))
                assert actual == list(uncached.verify_prefix(prefix, proposals))
            assert session.next_token(prompt + t[:7]) == t[7]
        assert session._cache is None
        with pytest.raises(RuntimeError, match='closed'):
            session.next_token(prompt)
    finally:
        hook.remove()
    assert calls == [
        ([0, 2, 3, t[0], wrong(t[1]), 5], [0, 1, 2, 3, 4, 5], 6, 0),
        ([t[1], t[2], t[3]], [4, 5, 6], 7, 4),
        ([t[4], wrong(t[5])], [7, 8], 9, 7),
        ([t[5]], [8], 9, 8),
        ([4, 5, 6], [0, 1, 2], 3, 0),
        ([0, 2, 3, t[0], t[1], t[2], t[3], t[4], t[5], t[6]], list(range(10)), 10, 0),
    ]


@pytest.mark.parametrize('family', ['gpt2', 'qwen3'])
def test_scripted_requests_match_uncached_target(family):
    cached, uncached = pair(family)
    reasons = set()
    for prompt in ((0,), (0, 2, 3)):
        path = greedy_decode(uncached, prompt, max_new_tokens=24).token_ids
        for eos in ((), (path[5],)):
            for mismatch in (None, 0, 1, 2):
                for budget in (1, 3, 5):
                    for limit in (0, 1, 4, 11):
                        options = dict(max_new_tokens=limit, eos_token_ids=eos)
                        expected = greedy_decode(uncached, prompt, **options)
                        assert greedy_decode(cached, prompt, **options) == expected
                        a = speculative_decode(cached, ScriptedDraft(prompt, path, mismatch), prompt,
                                               draft_budget=budget, **options)
                        b = speculative_decode(uncached, ScriptedDraft(prompt, path, mismatch), prompt,
                                               draft_budget=budget, **options)
                        assert a == b and a.generation == expected
                        reasons.add(a.stop_reason)
                        counts = a.accounting
                        if eos or limit < 4 or budget == 1:
                            continue
                        if mismatch is None:
                            assert counts.accepted == counts.verified == counts.proposed > 0
                        elif mismatch == 0:
                            assert counts.accepted == 0 < counts.verified
                        else:
                            assert 0 < counts.accepted < counts.verified
    assert reasons == {'eos', 'length'}


def alternating_grammar():
    xgr = pytest.importorskip('xgrammar')
    info = xgr.TokenizerInfo(list('abcdefghijklmnop') + ['<eos>'], stop_token_ids=[16])
    return xgr.GrammarCompiler(info, max_threads=1).compile_grammar(
        'root ::= ' + ' '.join(['wide narrow'] * 6) + '\nwide ::= [a-p]\nnarrow ::= [a-c]'
    )


@pytest.mark.parametrize('family', ['gpt2', 'qwen3'])
@pytest.mark.parametrize('mode', ['c1', 'c2', 'c3', 'c4'])
def test_model_drafts_match_uncached_target_in_every_mode(family, mode):
    from bitmask_speculation.grammar.xgrammar import XGrammarState
    compiled = alternating_grammar()
    cached, uncached = pair(family)
    draft = TransformersGreedyModel(model(family, 41))
    policy = DensityPolicy(low_density=.2, high_density=.5, min_budget=1, max_budget=4)
    fresh = (lambda: None) if mode == 'c1' else (lambda: XGrammarState(compiled))
    rejected, reasons = 0, set()
    for prompt in ((0,), (0, 2, 3), (7, 8)):
        for budget in ((1,) if mode == 'c4' else (1, 3, 8)):
            for limit in (0, 1, 7, 16):
                def run(target):
                    return speculative_decode(target, draft, prompt, max_new_tokens=limit,
                        draft_budget=budget, eos_token_ids=(16,), grammar=fresh(),
                        constrain_draft=mode in {'c3', 'c4'},
                        draft_policy=policy if mode == 'c4' else None)
                a, b = run(cached), run(uncached)
                assert a == b
                assert a.generation == greedy_decode(uncached, prompt, max_new_tokens=limit,
                                                     eos_token_ids=(16,), grammar=fresh())
                assert len(a.schedule) == (a.accounting.rounds if mode == 'c4' else 0)
                rejected += a.accounting.verified - a.accounting.accepted
                reasons.add(a.stop_reason)
    assert rejected > 0
    if mode != 'c1':
        assert 'eos' in reasons and 'length' in reasons


def test_failures_release_the_request_cache():
    cached, uncached = pair('gpt2')
    sessions = capture_sessions(cached)
    prompt = (0, 2, 3)
    path = greedy_decode(uncached, prompt, max_new_tokens=24).token_ids
    with pytest.raises(RuntimeError, match='draft failure'):
        speculative_decode(cached, ScriptedDraft(prompt, path, 1, fail_round=3), prompt,
                           max_new_tokens=12, draft_budget=3)
    assert sessions[-1]._cache is None and sessions[-1]._closed

    failing = []
    def fail(module, args, kw):
        if failing:
            raise RuntimeError('target failure')
    hook = cached._model.register_forward_pre_hook(fail, with_kwargs=True)
    try:
        with cached.target_session() as session:
            assert session.next_token(prompt) == path[0]
            failing.append(True)
            with pytest.raises(RuntimeError, match='target failure'):
                session.next_token(prompt + path[:1])
            assert session._cache is None and session._tokens == ()
            failing.clear()
            calls, recorder = record_calls(cached._model)
            try:
                assert session.next_token(prompt + path[:1]) == path[1]
            finally:
                recorder.remove()
            assert calls == [([*prompt, path[0]], [0, 1, 2, 3], 4, 0)]
        failing.append(True)
        with pytest.raises(RuntimeError, match='target failure'):
            greedy_decode(cached, prompt, max_new_tokens=4)
        assert sessions[-1]._cache is None and sessions[-1]._closed
    finally:
        hook.remove()


def test_missing_cache_update_fails_and_uncached_option_still_works():
    raw = model('gpt2')
    def strip_cache(module, args, output):
        return SimpleNamespace(logits=output.logits, past_key_values=None)
    hook = raw.register_forward_hook(strip_cache)
    try:
        adapter = TransformersGreedyModel(raw, target_cache=True)
        with pytest.raises(RuntimeError, match='did not extend'):
            with adapter.target_session() as session:
                session.next_token((0,))
        assert session._cache is None
        plain = TransformersGreedyModel(raw)
        assert greedy_decode(plain, (0,), max_new_tokens=3).token_ids
    finally:
        hook.remove()


def test_reused_adapter_prefills_each_request_once():
    cached, uncached = pair('qwen3')
    draft = TransformersGreedyModel(model('qwen3', 41))
    sessions = capture_sessions(cached)
    prompts = ((0,), (0, 2, 3), (0,), (7, 8))
    calls, hook = record_calls(cached._model)
    try:
        for prompt in prompts:
            a = speculative_decode(cached, draft, prompt, max_new_tokens=9, draft_budget=3)
            assert a == speculative_decode(uncached, draft, prompt, max_new_tokens=9, draft_budget=3)
    finally:
        hook.remove()
    assert [prefill for *_, prefill in calls].count(0) == len(prompts)
    assert len(sessions) == len(prompts) and all(s._cache is None for s in sessions)


def test_rejects_non_croppable_layers_before_any_forward():
    qwen_hybrid = getattr(transformers, 'Qwen3_5ForCausalLM', None)
    if qwen_hybrid is None:
        pytest.skip('Transformers lacks Qwen3.5')
    config = transformers.Qwen3_5TextConfig(vocab_size=V, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, eos_token_id=None, bos_token_id=0)
    targets = [(qwen_hybrid(config).eval(), 'LinearAttentionLayer'),
               (model('qwen3', use_sliding_window=True, sliding_window=4, max_window_layers=0),
                'DynamicSlidingWindowLayer')]
    for raw, layer in targets:
        calls = []
        hook = raw.register_forward_pre_hook(lambda *args: calls.append(True))
        try:
            with pytest.raises(ValueError, match=f'full-attention layers only, found {layer}.*target_cache=False'):
                speculative_decode(TransformersGreedyModel(raw, target_cache=True), TransformersGreedyModel(raw), (0, 2),
                                   max_new_tokens=4, draft_budget=2)
            assert not calls
        finally:
            hook.remove()
        plain = TransformersGreedyModel(raw)
        result = speculative_decode(plain, TransformersGreedyModel(raw), (0, 2), max_new_tokens=4, draft_budget=2)
        assert result.generation == greedy_decode(plain, (0, 2), max_new_tokens=4)


def test_target_cache_is_opt_in_and_boolean():
    adapter = TransformersGreedyModel(model('gpt2'))
    assert adapter.target_cache is False
    with adapter.target_session() as session:
        assert session is adapter
    with pytest.raises(TypeError, match='target_cache'):
        TransformersGreedyModel(model('gpt2'), target_cache=1)
