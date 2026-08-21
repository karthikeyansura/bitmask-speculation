import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from bitmask_speculation.decoding import ProposalAccounting, speculative_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel


class ObservedTarget:
    def __init__(self, target):
        self.target = target
        self.vocab_size = target.vocab_size
        self.rounds = []
        self.closed = 0

    def verify_prefix(self, prefix, proposals, *, grammar=None):
        selected = []
        self.rounds.append((tuple(proposals), selected))
        stream = self.target.verify_prefix(prefix, proposals, grammar=grammar)
        try:
            for event in stream:
                selected.append(event.token_id)
                yield event
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
            self.closed += 1


def tiny_model(seed):
    torch.manual_seed(seed)
    return TransformersGreedyModel(transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=4, n_positions=24, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=3, eos_token_id=2, pad_token_id=3,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )).eval())


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("seed", [7, 19, 41])
def test_model_accounting_matches_consumed_tokens(seed, constrained):
    model, draft = tiny_model(seed), tiny_model(seed + 100)
    compiled = None
    if constrained:
        xgr = pytest.importorskip("xgrammar")
        from bitmask_speculation.grammar.xgrammar import XGrammarState
        info = xgr.TokenizerInfo(["a", "b", "<eos>", "z"], stop_token_ids=[2])
        compiled = xgr.GrammarCompiler(info, max_threads=1).compile_grammar(
            'root ::= ("a" | "b") ("a" | "b") ("a" | "b")'
        )
    for budget in (1, 2, 8):
        for limit in (0, 1, 3, 8):
            target = ObservedTarget(model)
            grammar = XGrammarState(compiled) if constrained else None
            result = speculative_decode(
                target, draft, [3], max_new_tokens=limit, draft_budget=budget,
                eos_token_ids=[2], grammar=grammar,
            )
            proposed = verified = accepted = 0
            selected_count = 0
            for proposals, selected in target.rounds:
                proposed += len(proposals)
                verified += min(len(proposals), len(selected))
                selected_count += len(selected)
                for proposal, chosen in zip(proposals, selected):
                    if proposal != chosen:
                        break
                    accepted += 1
            emitted = selected_count - int(result.eos_token_id is not None)
            assert result.accounting == ProposalAccounting(
                proposed, verified, accepted, emitted, len(target.rounds)
            )
            assert emitted == len(result.token_ids)
            assert target.closed == result.accounting.rounds
