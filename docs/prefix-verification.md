# Prefix verification

`verify_prefix` compares proposals with target choices in order. It yields the
longest matching prefix and then one target correction. If every proposal
matches, it can yield one bonus choice. With no proposals it yields a target-only
fallback. It never skips a mismatch to accept later proposals.

```python
from bitmask_speculation.decoding import verify_prefix

rows = [10, 12, 13]  # Later rows must not be requested after the mismatch.
events = list(verify_prefix([10, 12, 14, 15], rows.__getitem__, vocab_size=32))
assert [(event.token_id, event.kind) for event in events] == [
    (10, "accepted"), (12, "accepted"), (13, "correction")
]
```

Each immutable `VerifiedToken` has a `token_id` and a `kind`: `accepted`,
`correction`, `bonus`, or `fallback`. These are target-confirmed choices, not yet
text emissions. EOS has no special meaning to this primitive.

## Selection order and ownership

The callback `target_token_at(i)` selects after the committed prefix followed by
`proposals[:i]`. It is called once per requested event, with indices starting at
zero. Index `len(proposals)` selects the bonus after full acceptance. The callback
owns score selection, including tie handling and NaN rejection, just as in the
[greedy reference](reference-decoding.md).

Proposal IDs are validated and copied immediately. Target choices are validated
as they are requested. A callback failure propagates; previously yielded events
are not rolled back. Callers must discard a failed request or restore their own
state rather than treat it as successful partial output.

The iterator never requests a row after a mismatch. It also never requests the
bonus unless the consumer advances beyond every accepted proposal. Consumers
must stop at their output limit, EOS, or grammar termination before requesting
another event. They should release or close an unused iterator promptly.

The primitive does not own a model cache, tokenizer, grammar matcher, output
budget, or EOS policy. It is a round-level building block, not a complete
speculative generation API. Collecting all events is appropriate only when no
request-level stop condition intervenes. The [speculative request
loop](speculative-decoding.md) owns these conditions across rounds.

## One-forward Transformers verification

For an initialized `TransformersGreedyModel` adapter, call:

```python
stream = adapter.verify_prefix(committed_prefix, proposals)
```

The call immediately runs one uncached forward pass over the prefix plus
proposals. For a fixed prefix length `L`, event index `i` reads logit row
`L - 1 + i`. The bonus reads `L + len(proposals) - 1`, the final row. These offsets
never depend on a growing output list. Empty proposals read row `L - 1`.

Argmax and NaN checks occur lazily, only for consumed rows. After a correction,
later rows condition on a rejected token and cannot predict the corrected prefix.
The next round must recompute from the corrected committed prefix. The iterator
retains its logits until exhausted, closed, or released; it never shares a mutable
workspace with another request. Inference mode is confined to model operations
and does not remain active in the caller while the iterator is suspended.

The adapter performs raw-logit selection with the same single-device and context
capacity restrictions as the greedy reference. The entire prefix plus proposal
sequence must fit the model context. Early iterator termination saves selection
work, not the forward computation already performed. This implementation makes
no speedup claim. Passing `grammar=state` enables [constrained selection and
state transitions](grammar-selection.md). This adapter stops at grammar
termination automatically; the caller still owns EOS and output limits.

## Validation

Controlled tests cover empty drafts, every mismatch position, full acceptance,
input ownership, callback failures, and stopping before a bonus. Exhaustive small
vocabularies compare verified continuations against sequential greedy decoding.
Tiny causal-model tests repeat this comparison for multiple prefix lengths and
random seeds, while asserting that verification performs exactly one forward.
Position-specific fixtures detect shifted rows and attempts to select rejected
or unused NaN rows. Real models may differ numerically across forward shapes;
those differences must be investigated before making equivalence claims.
