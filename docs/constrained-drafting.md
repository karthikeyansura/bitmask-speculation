# Constrained draft proposals

C3 uses the same grammar for draft selection and target verification, with a
fixed proposal budget. Enable it explicitly:

```python
from bitmask_speculation.decoding import speculative_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState

result = speculative_decode(
    target, draft, prompt_ids,
    max_new_tokens=128,
    draft_budget=5,
    eos_token_ids=eos_ids,
    grammar=XGrammarState(compiled_grammar),
    constrain_draft=True,
)
```

Both models must share token-ID meanings and special-token conventions. The
constrained draft adapter must select an allowed token and advance its supplied
grammar exactly once before returning it, just like the target adapter. The
Transformers adapter already implements this contract. Setting the flag without
a forkable grammar fails explicitly, including for a zero output budget.

## Fork and discard

`ForkableGrammarState` extends `GrammarState` with `fork()`. A fork begins at the
same generated suffix, with identical allowed IDs and termination status. Its
transitions and mutable mask workspace must be independent of its parent.
XGrammar shares the compiled grammar while copying matcher state; the adapter
allocates a separate mask workspace. It shares only the read-only token-ID index.

Each round follows this sequence:

1. Fork the grammar at the committed prefix, including any suffix consumed before
   this request. Never replay prompt tokens into the grammar.
2. Select draft tokens against the fork, advancing it once per selection. Stop
   at the proposal budget, remaining output budget, draft EOS or branch termination.
3. Discard the fork before target verification. This is the rollback operation:
   the parent has never advanced along the proposed path.
4. Verify against the parent. Accepted tokens and a correction or bonus advance
   committed state only as their events are consumed.
5. Start the next round from the corrected prefix and a fresh fork of its state.

A rejected tail cannot contaminate the next round. Full acceptance also discards
the fork; the target advances the parent once for every consumed choice. No
commit or merge of the draft branch is needed. There is no KV cache to roll back
in the current full-prefix model adapter.

## EOS, limits and errors

The draft adapter may consume EOS into its branch. That EOS is excluded from
proposals and counters, and branch termination cannot terminate the request.
An empty proposal still asks the target for a fallback token. Only the target's
choices control request termination. A final content token that terminates the
draft grammar remains a proposal for target verification.

Zero output budget calls neither models nor `fork()`. An already terminated
parent returns before drafting. Otherwise the proposal budget is capped by the
remaining content allowance. Target EOS, grammar termination and output limits
retain the precedence defined by the [request API](speculative-decoding.md).

A draft exception propagates with the parent unchanged for that round, including
when the draft advanced its branch before failing. A target exception can leave
the parent advanced through events already consumed; this is not transactional
rollback of the entire request. Custom grammar implementations are responsible
for true fork independence. The decoder rejects a fork returning the parent or
a different vocabulary, but cannot detect hidden shared mutable state.

## Constraints

Tests force rejection at each of four proposal positions and compare allowed
IDs and termination state with independently replayed reference state before
verification and after every consumed event. Additional cases cover full
acceptance, budgets within a round, draft EOS disagreement, terminal branches,
continuation of an existing suffix, and failures after draft transitions.
Tiny causal-model tests compare C2 and C3 against constrained target-only decoding.
Trial records retain the `c3` label and compare it with the constrained reference.

The draft fork isolates speculative grammar transitions from the committed target state.
