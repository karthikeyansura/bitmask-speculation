# Grammar-constrained target selection

The target model selects a maximum only among tokens valid in the current grammar
state. Grammar state describes the generated suffix, not the prompt. A fresh
state is required for each independent request; to continue an existing suffix,
replay its generated token IDs before calling the decoder.

## State and selection

`GrammarState` exposes a vocabulary size, termination status, allowed token IDs,
and token acceptance. Reading allowed IDs must not advance state. A rejected
acceptance must return `False` without advancing. The model and grammar must agree
on token-ID meanings; equal vocabulary sizes alone do not establish compatibility.

The Transformers adapter performs these operations for every consumed choice:

1. Read the valid IDs at the current committed grammar state.
2. Select their highest-scoring token, breaking exact ties by lowest token ID.
3. Accept that token into the grammar, failing explicitly if it is rejected.
4. Return or yield the token only after the transition succeeds.

Draft proposals never advance the target grammar. At a mismatch, the target
correction advances state once; rejected proposals and their tail do not. This
requires no rollback because target state never visits the rejected path. Draft
side constraints and speculative matcher rollback are separate work.

`masked_argmax` gathers valid IDs instead of setting invalid scores to negative
infinity. That preserves validity even when all valid scores are negative
infinity. It rejects empty valid sets, malformed IDs, non-floating scores, and
NaN anywhere in the score row. It leaves logits unchanged. This reference sorts
and gathers explicit IDs and synchronizes the winning index to the host; it is
not an optimized packed-mask GPU kernel.

## XGrammar adapter

Install the `models` and `grammar` extras as described in
[development](development.md). Given a compiled XGrammar grammar, an initialized
`TransformersGreedyModel` named `target`, and tokenized `prompt_ids`:

```python
from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState

state = XGrammarState(compiled_grammar)
result = greedy_decode(
    target, prompt_ids, max_new_tokens=128,
    eos_token_ids=eos_ids, grammar=state,
)
```

Compile with the target tokenizer's token mapping and model vocabulary size.
The adapter owns a private matcher and CPU mask workspace. It expands only IDs
below the vocabulary size, handling signed int32 words and excluding padding.
The all-valid fast path uses the matcher's documented `need_apply=False` result.
The base package imports without XGrammar or PyTorch; optional libraries are
loaded when the adapter is constructed or tensor selection is called.

By default XGrammar requires a stop token after completing the grammar body.
A complete body and a terminated matcher are therefore different states.
`terminate_without_stop_token=True` instead ends at the body; choose the mode
explicitly to match the request contract. See the upstream
[matcher API](https://xgrammar.mlc.ai/docs/latest/api/python/grammar_matcher.html).
Set the decoder's EOS IDs consistently with the grammar's stop tokens.

## Verification and termination boundaries

```python
stream = target.verify_prefix(committed_prefix, proposals, grammar=state)
```

The state is advanced before each event is yielded. Do not accept the yielded
token again. Stop consumption at the request's EOS or output limit; the adapter
stops automatically at grammar termination, including after an accepted proposal
that would otherwise trigger a bonus. A terminal state at entry causes no forward
pass in verification; direct `next_token` raises `GrammarTerminatedError`.

The constrained greedy reference preserves the final content token and returns
`grammar` when its transition terminates the matcher. A selected EOS takes
precedence and is excluded from content. Zero output budget returns `length`
without reading or advancing the matcher. Vocabulary compatibility is still
validated. The request-wide speculative loop and its output accounting remain
separate work.

A partially consumed iterator leaves state at exactly its yielded prefix. Keep
exclusive ownership of that state while the iterator is active; do not interleave
iterators, mutate it externally, or share it across requests. Close an abandoned
iterator to release its logits. Exceptions do not roll back earlier successful
transitions: discard the failed request or restore an explicit checkpoint.

## Validation scope

Tests cover valid-score ties, infinities, invalid IDs, NaNs, empty support,
transition failures, partial acceptance, correction state, and termination
without a bonus. Real XGrammar tests use EBNF and JSON Schema, including stop-token
modes and bits crossing word boundaries. Tiny causal models compare every
rejection position against constrained greedy decoding with an independent
matcher. These are CPU correctness checks, not GPU performance measurements.
