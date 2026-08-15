# Prefix acceptance, one round at a time

This trace applies the [decoding contract](design.md) to a small, explicit token
vocabulary. Token IDs and target choices are specified for the example; they are
not a model run or the tokenization of a particular model.

The [verification primitive](prefix-verification.md) implements the prefix
comparison below. Grammar state and request-level termination are separate from
that primitive.

## Starting state

The generated prefix is `{"op":`. The complete model input, including the prompt
and this prefix, has length `L`. The grammar permits exactly these four objects:

```jsonl
{"op":"get","id":7}
{"op":"get","id":9}
{"op":"put","id":7}
{"op":"put","id":9}
```

For this example, the grammar terminates when the closing brace is committed.
An adapter that requires EOS after a complete object must handle that additional
step explicitly.

Use a vocabulary of 32 token IDs. At the states below, only the listed tokens
can be valid; every other ID is masked out.

| Token ID | Decoded text | Valid after |
| --- | --- | --- |
| 10 | `"get"` | `{"op":` |
| 11 | `"put"` | `{"op":` |
| 12 | `,"id":` | Either operation value |
| 13 | `7` | The `id` field separator |
| 14 | `9` | The `id` field separator |
| 15 | `}` | Either integer value |

Assume the target-only greedy continuation is `[10, 12, 13, 15]`, producing
`{"op":"get","id":7}`. Each scenario starts from the same prefix and uses
the same target choices. These are distinct proposal scenarios, not successive
rounds of one generation.

## Partial acceptance

Let `K = 4`, with at least four output-token slots remaining. The grammar-guided
draft proposes `[10, 12, 14, 15]`, which would complete the object with `id: 9`.
Every proposal is legal along the draft's path. Legality does not imply target
agreement.

The target evaluates the prefix plus all four proposals. Verification uses the
following rows of its logits; positions are zero-based over the full input.

| Proposal index | Target logit position | Draft token | Target choice | Action |
| --- | --- | --- | --- | --- |
| 0 | `L - 1` | 10: `"get"` | 10: `"get"` | Accept |
| 1 | `L` | 12: `,"id":` | 12: `,"id":` | Accept |
| 2 | `L + 1` | 14: `9` | 13: `7` | Append correction; stop verification |
| 3 | `L + 2` | 15: `}` | Not inspected | Discard |

The round emits `[10, 12, 13]`. The new generated prefix is
`{"op":"get","id":7`, with the closing brace still absent.

The row at `L + 2` conditions on the rejected `9`. It is not a prediction for
the corrected prefix ending in `7`. Even though both integers permit `}`, the
remaining proposal cannot be accepted by skipping over the mismatch. Recompute
the next target choice from the corrected prefix in the next round.

### Restoring state

If a matcher was advanced through the entire draft, its temporary state is after
the closing brace. Verification must instead begin from the round's saved
grammar state, using a separate matcher or restoring a checkpoint.

Advance committed grammar state only through the two accepted tokens and the
target correction. Alternatively, when retaining speculative matcher states,
restore the state after `[10, 12]`, discard the transitions for `[14, 15]`, and
then apply `13`. Both approaches must reach the state before the closing brace.

A cached model path must likewise discard cache entries for the rejected token
and its suffix. The correction is not already represented by the target forward
pass over the draft. Consume it before using the cache to predict what follows,
or recompute from the committed prefix. The draft cache must also be realigned.

## Full acceptance and the bonus token

Now let `K = 3`, still with four output-token slots remaining. The draft proposes
`[10, 12, 13]`. All three agree with the target.

The target row at `L + 2` now conditions on the accepted `7`. Apply the grammar
mask for that prefix and select `15`, the closing brace, as a bonus token. The
round emits `[10, 12, 13, 15]` and the grammar terminates.

The bonus token is a target emission, not an accepted draft proposal. A second
forward pass is unnecessary for this choice because the target evaluated the
accepted proposal sequence. No further token is appended after termination.

## First-token rejection

Let `K = 4` and the draft propose `[11, 12, 14, 15]`. The grammar permits `"put"`,
but the target chooses `"get"`. Append `10` and discard the entire draft suffix.
Only the first proposal was verified, and none was accepted.

At this state, the mask has two valid bits out of 32: density is `2 / 32`.
The draft and target still disagree. Density counts allowed tokens; it does not
measure the models' ranking of those tokens or guarantee acceptance.

## Output limits and draft termination

With only two output-token slots remaining, cap the effective draft budget at
two. If `[10, 12]` is accepted, the round has exhausted the budget. Do not inspect
or append a bonus token. The output ends at `{"op":"get","id":` and must be
reported as incomplete, not as a schema-valid object.

If an unmasked draft signals EOS before proposing any tokens, the target still
takes a constrained step and emits `10` in this example. Draft EOS is not a
target termination decision. A grammar-guided draft cannot select EOS at this
state if the mask forbids it. When a target-confirmed termination condition is
reached, stop immediately without filling the rest of the selected budget.

## Counters

All emitted tokens below are content tokens; none is an EOS token. Counts are
per scenario, starting from the prefix above.

| Scenario | Proposed | Verified | Accepted | Emitted | Acceptance | Proposal yield |
| --- | --- | --- | --- | --- | --- | --- |
| Partial acceptance | 4 | 3 | 2 | 3 | `2/3` | `2/4` |
| Full acceptance plus bonus | 3 | 3 | 3 | 4 | `3/3` | `3/3` |
| First-token rejection | 4 | 1 | 0 | 1 | `0/1` | `0/4` |
| Two-token output limit | 2 | 2 | 2 | 2 | `2/2` | `2/2` |
| No proposals; target step | 0 | 0 | 0 | 1 | Undefined | Undefined |

Here, verified means the prefix comparisons actually performed, including the
mismatch. A target forward pass can compute logits for more positions than the
verification loop inspects. Those unused rows still cost compute.

Acceptance is `accepted / verified`; proposal yield is `accepted / proposed`.
Neither is throughput. The full-acceptance scenario emits four tokens in one
round, but a speedup claim also needs the draft, target, grammar, transfer, and
synchronization costs measured against the same target-only workload.
