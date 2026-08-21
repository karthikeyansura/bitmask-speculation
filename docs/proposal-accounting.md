# Proposal accounting

`speculative_decode` returns an immutable `SpeculativeResult`. Generation and
accounting are separate values from the same successful request:

```python
result.generation.token_ids  # Same as result.token_ids.
result.accounting.proposed
result.accounting.verified
result.accounting.accepted
result.accounting.emitted
result.accounting.rounds
result.accounting.acceptance_rate
result.accounting.proposal_yield
```

## Count definitions

| Field | Counted | Excluded |
| --- | --- | --- |
| `proposed` | All tokens passed to target verification, including discarded tails | Prompt and draft EOS choices, which the current drafter omits |
| `verified` | Consumed draft/target comparisons, including the first mismatch | Uninspected tails, bonus and fallback choices |
| `accepted` | Proposals that match the target in the consumed matching prefix | Corrections, bonus and fallback choices |
| `emitted` | Generated content from accepted proposals, corrections, bonuses and fallbacks | Prompt, discarded proposals, and target EOS |
| `rounds` | Proposal/verification rounds, including empty-draft and EOS-ending rounds | Requests with zero budget or an initially terminated grammar |

These are logical decoding counts, not kernel launches or total model work.
A target forward may compute rows the verifier never consumes. A draft EOS
prediction costs a forward even though it contributes no proposal. The current
Transformers adapter uses one forward per verification round; that is an adapter
property, not the definition of the round counter.

Counts accumulate across the entire request. They are nonnegative integers and
must satisfy `accepted <= verified <= proposed`. `emitted` equals the number of
returned content token IDs. A final content token that terminates a grammar is
still emitted. An EOS correction counts as verified but not emitted; bonus and
fallback EOS choices count as neither verified nor emitted. All count their round.

The implementation validates verification event labels against proposal tokens
before accounting for them. Malformed streams and incomplete rounds fail rather
than publishing misleading counts. A legitimate request stop can interrupt a
round after accepted tokens without consuming its bonus.

## Ratios

`acceptance_rate = accepted / verified` measures agreement among comparisons
actually inspected. `proposal_yield = accepted / proposed` also accounts for
wasted proposal tails. Both return `None` when their denominator is zero.
A nonzero denominator with no accepted tokens produces `0.0`.

For example, a four-token proposal with two matches and a third-token mismatch
has `proposed=4`, `verified=3`, and `accepted=2`. Its acceptance is `2/3`, while
its proposal yield is `2/4`. If the correction completes the grammar, the request
emits three content tokens and takes one round.

Do not average request acceptance rates to get a corpus rate. Sum the relevant
integer counts first, then divide total accepted by total verified or total
proposed. Exclude undefined per-request ratios only for explicitly described
request-level summaries. Neither ratio establishes a throughput improvement.

## Serialization and ownership

Use `dataclasses.asdict(result)` for a nested record containing generation and
raw counts. Derived ratios are properties; include them explicitly if an output
format needs them, preserving `None` as JSON null rather than substituting zero.
Results and counts are frozen. Comparing full speculative results compares both
generation and accounting; use `result.generation` for greedy-reference checks.

No partial result is returned on a decoding failure. Accounting does not request
extra logits, consume an unused bonus, or advance grammar state. The
[prefix walkthrough](prefix-acceptance.md) gives five concrete count traces;
tests also cover zero budgets, early termination, EOS, multiple rounds, and
independent observations from real model adapters on CPU.
