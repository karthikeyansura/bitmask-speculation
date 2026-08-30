# Speculative request decoding

`speculative_decode` combines draft proposals and lazy target verification into
one bounded request. Both models use greedy selection. The optional grammar
constrains the target. `constrain_draft=True` also applies that grammar to draft
selection through a fresh fork at each round boundary.

Given already initialized target and draft model adapters:

```python
from bitmask_speculation.decoding import speculative_decode

result = speculative_decode(
    target, draft, prompt_ids,
    max_new_tokens=128,
    draft_budget=5,
    eos_token_ids=eos_ids,
    grammar=state,  # Omit for unconstrained decoding.
)
```

The target implements `VerificationTarget`, including lazy prefix verification.
The draft implements `GreedyModel`, or `ConstrainedGreedyModel` when constrained
proposals are enabled. `TransformersGreedyModel` supports either role.
Both must agree on token-ID meanings, tokenizer behavior, and special-token
conventions. Vocabulary-size equality is checked but does not prove compatibility.

The immutable `SpeculativeResult` contains `generation` (a `GreedyResult`) and
`accounting` (a `ProposalAccounting`). The convenience properties `token_ids`,
`stop_reason`, and `eos_token_id` retain their meanings; prompt and EOS remain
excluded from content. Compare `result.generation` with the target-only reference
when checking decoding equivalence. Comparing two whole speculative results also
compares their counters. See [proposal accounting](proposal-accounting.md).

## One round

1. Construct the authoritative prefix from the prompt and emitted content.
2. Draft up to `min(draft_budget, remaining_output_tokens)` tokens. With draft
   constraints, advance only a fork of the committed grammar and stop when that
   branch terminates. Do not add a selected EOS to the proposal sequence. Discard
   the branch before verification, including its speculative EOS transition.
3. Verify the proposals against the target. Empty proposals request one
   target-only fallback choice; a draft EOS never ends the request by itself.
4. Consume target-confirmed choices in order, checking request termination after
   each one. Never request a bonus when the output budget is exhausted.
5. After rejection, discard the proposal tail and begin the next round from the
   corrected committed prefix. After full acceptance, a bonus is allowed only
   if the request still has room and has not terminated.

No model cache is carried across rounds in the current Transformers adapter.
Each draft prediction recomputes its full prefix; target verification performs
one forward over the committed prefix and proposals. Rejection therefore needs
no cache rollback. A future cached implementation must preserve the same prefix
and state semantics.

## Termination and validation

A target-selected EOS stops immediately and is stored separately from content.
Otherwise the selected content token is appended. If its grammar transition
terminates the request, the token is preserved and the stop reason is `grammar`.
Otherwise reaching the output limit returns `length`. Thus target EOS takes
precedence over grammar termination, which takes precedence over a simultaneous
length boundary. No extra prediction probes whether EOS would follow the limit.

Zero output budget validates inputs and returns `length` without calling either
model or reading/advancing matcher state. For positive budgets, an already
terminated grammar returns `grammar` before drafting. Prompt EOS tokens remain
ordinary context. Multiple explicit EOS IDs are supported.

`draft_budget` must be a positive integer; the output limit must be a nonnegative
integer. Empty prompts, invalid IDs, incompatible vocabulary sizes, booleans in
integer fields, and invalid model outputs fail explicitly. A verifier that emits
no event without an established terminal condition fails instead of looping.

The verifier must be lazy: selecting a token and advancing grammar state happen
only when the consumer requests that event. Its iterator is closed on early
return or failure when it exposes `close()`. Errors propagate rather than being
reported as successful partial output; earlier grammar transitions are not
rolled back automatically. State belongs exclusively to one active request.

## Scope

The reference path handles one request at a time, with a fixed or density-selected draft budget,
optional grammar constraints on the target or both models, and greedy selection.
It has request-level proposal counters, but no timing instrumentation, KV-cache
reuse, sampling, or batching yet. C4 optionally selects the budget using a
[density-threshold policy](draft-scheduling.md). The full prompt plus proposal sequence must
fit the model's context capacity; no implicit truncation is applied.

Tests compare content, EOS identity, stop reason, and final grammar state against
target-only greedy decoding. They cover partial and full acceptance, empty-draft
fallback, target EOS at different positions, output limits inside a round,
terminal grammar transitions, corrected-prefix reuse, and iterator cleanup.
Tiny randomly initialized causal models and real XGrammar matchers validate
integration on CPU without downloading pretrained weights. Constrained drafting
also checks rejection at every proposal position, independent branch transitions,
and parent state after draft errors. See [draft constraints](constrained-drafting.md).
