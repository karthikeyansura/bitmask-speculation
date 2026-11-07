# Decoding and grammar state

<a id="design"></a>

## Decoding contract

Greedy verification, grammar state and packed-mask semantics.

### Scope

The decoder handles one request on one device with greedy token selection.
A smaller draft model proposes a linear token sequence; the target verifies its
longest matching prefix. XGrammar supplies grammar state and packed token masks.
The project owns proposal scheduling, verification, and mask-processing kernels.

The correctness contract is equivalence to locally grammar-masked, target-only
greedy decoding. It does not establish distributional equivalence for sampling
or for conditioning a model on all complete strings accepted by a grammar.
Stochastic acceptance, continuous batching, and multi-GPU execution require
separate designs.

### Inputs and state

The draft and target must share token-ID meanings, tokenizer behavior, and
special-token conventions. Matching vocabulary sizes alone is insufficient.
Each run fixes the model and tokenizer revisions, prompt serialization, grammar,
numeric precision, and output-token limit.

At a round boundary, the committed token prefix is authoritative. Grammar state
describes the generated suffix of that prefix. Any model cache must have an
explicit correspondence to the tokens it has consumed; an appended token may
still be pending consumption. Recomputing the prefix is a valid reference path.

Drafting uses temporary state. Rejection must discard speculative grammar and
cache state beyond the accepted prefix before the next round begins.

With `target_cache=True`, the Transformers target cache spans one request. After
verification it covers the
round's prefix and proposals. The next round crops it to the longest prefix shared
with the new committed prefix: `L + j` after a correction at proposal `j`, or every
proposal after full acceptance. The correction or bonus is consumed first in that
round's forward. Rollback requires full-attention layers. Recurrent and
sliding-window layers cannot be cropped exactly, so they are rejected before any
forward pass. Without the option, each round recomputes the full prefix.

### Greedy verification

For a committed prefix of length `L`, let the draft propose `d[0:m]`, where `m`
is no greater than the selected budget `K` or the remaining output-token budget.
With a causal target forward pass over the prefix followed by the proposals,
logits at position `L - 1 + i` predict proposal `d[i]`. A cached implementation
must preserve this alignment even when its returned logits use different indices.

Verification walks proposals in order:

1. Compute the target's greedy choice using the grammar state before `d[i]`.
2. If the choice equals `d[i]`, append that token and advance committed state.
3. At the first mismatch, append the target's choice and discard the remaining
   proposals. Start the next round from the resulting committed prefix.
4. If every proposal matches, the target may append one additional token from
   the next-position logits, subject to grammar termination and the output limit.

Verification must stop immediately when a committed token terminates generation
or consumes the remaining output budget. A draft's EOS prediction alone cannot
terminate the target. If drafting returns no proposals, take one target-only
step unless generation has already finished.

After a mismatch, logits for later positions condition on a rejected token and
cannot be reused as predictions for the corrected prefix. Grammar masks for
those positions are likewise speculative and must be discarded.

The [prefix-acceptance walkthrough](design.md#prefix-acceptance) gives concrete traces
for rejection, full acceptance, and termination at an output limit.

### Mask semantics

For vocabulary size `V > 0`, a mask row contains `ceil(V / 32)` packed 32-bit
words. Token `t` is valid when bit `t % 32` in word `t // 32` is set. An `int32`
container carries a bit pattern: its sign does not change token validity.

Padding bits at indices `t >= V` never contribute to selection or density:

```text
valid_count = sum(mask[t] for t in range(V))
density     = valid_count / V
```

Masked argmax selects the highest-scoring valid token, breaking exact ties by
the lowest token ID. Invalid tokens must not win even when valid scores are
negative infinity. Reject NaN logits and malformed inputs explicitly. An empty
valid-token set is an error unless the grammar has already reached its defined
terminal state; it must not silently select token zero.

Kernel interfaces must specify supported shapes, dtypes, layouts, and devices.
Input masks and logits remain unchanged unless an operation is explicitly
documented as in-place. Reusable workspaces must not overwrite results still
owned by a caller. CUDA operations must respect the caller's current stream.

### Draft scheduling

The fixed-budget policy establishes the baseline. A density-based policy reads
the mask at the committed round boundary and chooses a positive budget within
configured bounds. The effective budget is capped by the remaining output limit.
Thresholds and bounds are experiment parameters, not universal constants.

Low density describes a restricted support, not the probability that draft and
target agree. Density at the round boundary also says nothing certain about the
states reached later in the proposal. Measure acceptance and total decoding cost
to determine whether this signal helps.

Mask construction, transfers, popcount reductions, and host synchronization all
contribute to scheduling overhead. A mask already needed by the grammar does not
make density computation free.

### Experiment matrix

| Variant | Draft mask | Target mask | Draft budget |
| --- | --- | --- | --- |
| Target-only reference | No draft | Enabled | None |
| C1 | Disabled | Disabled | Fixed |
| C2 | Disabled | Enabled | Fixed |
| C3 | Enabled | Enabled | Fixed |
| C4 | Enabled | Enabled | Density-based |

C1 is an unconstrained diagnostic and needs its own unconstrained target-only
reference. It is not a structured-output correctness baseline. C2 versus C3
isolates draft-side masking; C3 versus C4 isolates scheduling. Keep other settings
fixed for each comparison and report regressions as well as improvements.

### Accounting and measurement

Record proposed, verified, accepted, and emitted token counts separately.
Proposed tokens include discarded tails. Verified tokens include the mismatch
position, when present. Accepted tokens count only matching draft proposals.
Emitted tokens include target corrections and bonus tokens, never rejected tails.

Report acceptance as `accepted / verified`, using an undefined value when no
proposal was verified. Report proposal yield separately as `accepted / proposed`.
Maintain `accepted <= verified <= proposed`; acceptance alone is not a speedup.
For text throughput, exclude EOS and other termination-only tokens from the
emitted-token count and apply the same rule to every baseline.

Separate model loading, grammar compilation, prefill, and decode measurements.
State which phases an end-to-end interval includes. Warm up models and compiled
kernels, synchronize GPU work at timing boundaries, and repeat each workload.
Report sample counts, latency distributions, output lengths, and throughput.
Kernel timings and full-generation timings answer different questions.

Each result must retain its command, source revision, actual run timestamp,
hardware identity, dependency versions, model/tokenizer revisions, schema,
prompt, precision, scheduling parameters, and raw measurements. Schema validation
and termination status accompany timing results. Parsing as JSON alone does not
establish conformance to the requested schema.

### Correctness conditions

Before benchmarking, require:

- Exact token agreement with the appropriate greedy reference on controlled
  tests covering first-token rejection, partial acceptance, and full acceptance.
- Grammar-state restoration after rejection, target-controlled EOS handling,
  zero-proposal progress, and output limits reached inside a verification round.
- Mask/reference agreement for empty and full masks, signed words, partial final
  words, tied scores, and vocabulary sizes around word and reduction boundaries.
- Explicit errors for unsupported inputs and kernel checks for output ownership
  and non-default CUDA streams.
- Independent schema validation for completed structured outputs; incomplete
  outputs caused by length limits must be reported as incomplete.

Controlled fixtures isolate algorithmic errors from model numerical behavior.
Model integration checks must also investigate token differences caused by
precision or forward-pass shape rather than relaxing equality to a pass rate.
Performance claims follow these checks and must name the workload and baseline.

<a id="prefix-acceptance"></a>

## Prefix acceptance, one round at a time

This trace applies the [decoding contract](design.md#design) to a small, explicit token
vocabulary. Token IDs and target choices are specified for the example; they are
not a model run or the tokenization of a particular model.

The [verification primitive](usage.md#prefix-verification) implements the prefix
comparison below. Grammar state and request-level termination are separate from
that primitive.

### Starting state

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

### Partial acceptance

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

#### Restoring state

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
or recompute from the committed prefix. Here an enabled target cache keeps
`L + 2` positions, ending at `[10, 12]`, and the next round feeds `13` before its
proposals. The draft cache must also be realigned; this adapter discards it.

### Full acceptance and the bonus token

Now let `K = 3`, still with four output-token slots remaining. The draft proposes
`[10, 12, 13]`. All three agree with the target.

The target row at `L + 2` now conditions on the accepted `7`. Apply the grammar
mask for that prefix and select `15`, the closing brace, as a bonus token. The
round emits `[10, 12, 13, 15]` and the grammar terminates.

The bonus token is a target emission, not an accepted draft proposal. A second
forward pass is unnecessary for this choice because the target evaluated the
accepted proposal sequence. No further token is appended after termination.

### First-token rejection

Let `K = 4` and the draft propose `[11, 12, 14, 15]`. The grammar permits `"put"`,
but the target chooses `"get"`. Append `10` and discard the entire draft suffix.
Only the first proposal was verified, and none was accepted.

At this state, the mask has two valid bits out of 32: density is `2 / 32`.
The draft and target still disagree. Density counts allowed tokens; it does not
measure the models' ranking of those tokens or guarantee acceptance.

### Output limits and draft termination

With only two output-token slots remaining, cap the effective draft budget at
two. If `[10, 12]` is accepted, the round has exhausted the budget. Do not inspect
or append a bonus token. The output ends at `{"op":"get","id":` and must be
reported as incomplete, not as a schema-valid object.

If an unmasked draft signals EOS before proposing any tokens, the target still
takes a constrained step and emits `10` in this example. Draft EOS is not a
target termination decision. A grammar-guided draft cannot select EOS at this
state if the mask forbids it. When a target-confirmed termination condition is
reached, stop immediately without filling the rest of the selected budget.

### Counters

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

The request API returns these counts through `result.accounting`. The
[accounting contract](design.md#proposal-accounting) also defines EOS cases, empty
requests, round totals, and undefined ratios. Tests execute all five scenarios
above against the request loop.

<a id="proposal-accounting"></a>

## Proposal accounting

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

### Count definitions

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
Transformers adapter uses one forward per verification round, cached or not; that
is an adapter property, not the definition of the round counter.

Counts accumulate across the entire request. They are nonnegative integers and
must satisfy `accepted <= verified <= proposed`. `emitted` equals the number of
returned content token IDs. A final content token that terminates a grammar is
still emitted. An EOS correction counts as verified but not emitted; bonus and
fallback EOS choices count as neither verified nor emitted. All count their round.

The implementation validates verification event labels against proposal tokens
before accounting for them. Malformed streams and incomplete rounds fail rather
than publishing misleading counts. A legitimate request stop can interrupt a
round after accepted tokens without consuming its bonus.

### Ratios

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

### Serialization and ownership

Use `dataclasses.asdict(result)` for a nested record containing generation and
raw counts. Derived ratios are properties; include them explicitly if an output
format needs them, preserving `None` as JSON null rather than substituting zero.
Results and counts are frozen. Comparing full speculative results compares both
generation and accounting; use `result.generation` for greedy-reference checks.

No partial result is returned on a decoding failure. Accounting does not request
extra logits, consume an unused bonus, or advance grammar state. The
[prefix walkthrough](design.md#prefix-acceptance) gives five concrete count traces;
tests also cover zero budgets, early termination, EOS, multiple rounds, and
independent observations from real model adapters on CPU.

<a id="mask-density"></a>

## Packed-mask density

`bitmask_speculation.grammar.density` provides `packed_mask_count(mask, vocab_size)`
and `packed_mask_density(mask, vocab_size)`. The latter returns count divided by
vocabulary size. These are CPU references requiring the `models` extra.

Each row holds exactly `ceil(vocab_size / 32)` signed int32 words. Token `t` maps
to bit `t % 32` in word `t // 32`. Negative words carry ordinary bit patterns:
`-1` has 32 set bits; `-2147483648` has only bit 31 set. Padding at token IDs beyond
the vocabulary never contributes, even if every padding bit is set.

```python
import torch
from bitmask_speculation.grammar.density import packed_mask_count, packed_mask_density

mask = torch.tensor([-2147483648, -1], dtype=torch.int32)
assert packed_mask_count(mask, 33) == 2  # IDs 31 and 32; ignore 31 padding bits.
assert packed_mask_density(mask, 33) == 2 / 33
```

Vocabulary size must be a positive Python integer; booleans are rejected.
The mask must be a one-dimensional, strided CPU int32 tensor with the exact word
count. Noncontiguous views are supported. Scalars, batched rows, sparse tensors,
other dtypes and other devices fail explicitly. The input is never modified.
Zero valid bits give density 0; all vocabulary bits give density 1. Empty support
is still a selection error unless the grammar is already terminal.

`XGrammarState.mask_density()` fills its private workspace without accepting any
token. It returns 1 for XGrammar's all-valid fast path without reading stale mask
contents. A terminal state raises `GrammarTerminatedError`; callers must check
termination before querying. Forks remain independent.

The implementation converts the CPU row to Python words and uses integer
`bit_count`, normalizing signed words and masking the tail. It avoids per-word
tensor scalar extraction and does not expand a vocabulary-sized boolean mask.
It is not a CUDA/Triton kernel. Mask filling and counting still have a cost;
selection may refill the mask. Density measures support size, not draft-target
agreement, and does not establish a speedup.

<a id="draft-scheduling"></a>

## Density-selected draft budgets

C4 adds scheduling to C3. Pass `draft_policy=DensityPolicy(...)` together with
`constrain_draft=True` and a grammar implementing `fork()` and `mask_density()`.
`XGrammarState` provides both. Missing capabilities fail explicitly; there is no
silent fallback to a fixed budget.

```python
from bitmask_speculation.decoding import DensityPolicy, speculative_decode

policy = DensityPolicy(low_density=0.005, high_density=0.02, min_budget=1, max_budget=8)
result = speculative_decode(
    target, draft, prompt_ids, max_new_tokens=128, eos_token_ids=eos_ids,
    grammar=state, constrain_draft=True, draft_policy=policy,
)
for decision in result.schedule:
    print(decision.output_offset, decision.density,
          decision.requested_budget, decision.effective_budget)
```

The thresholds define half-open bands:

| Density | Requested budget |
| --- | --- |
| `d < low_density` | `max_budget` |
| `low_density <= d < high_density` | `(min_budget + max_budget) // 2` |
| `d >= high_density` | `min_budget` |

Density must be finite and in [0, 1]. Thresholds must be strictly increasing
within that interval. Budgets must be positive integers with min <= max;
booleans are rejected. Equal budget bounds make the policy constant. The default
thresholds are a starting configuration, not validated universal values.

Once per active round, before forking or drafting, the decoder reads density
from committed state. The effective budget is `min(requested, remaining)`.
An all-valid grammar reports density 1 and therefore selects the minimum;
it does not bypass the policy. Zero output budget and terminal state cause no
density query. A target correction determines the next round's density; rejected
proposal states never enter scheduling. Existing EOS and output-limit rules hold.

`draft_budget` remains the fixed C1–C3 parameter. A supplied policy replaces it
for C4. `result.schedule` is an immutable tuple with one `RoundSchedule` per C4
round and is empty for fixed-budget decoding. Effective budget describes the cap,
not the actual number proposed: draft EOS or termination can shorten a proposal.

Mask filling, CPU counting and policy decisions are part of decoding overhead.
Reading density may refill the same mask needed for token selection. No cache,
GPU reduction or performance improvement is implied. Tests verify threshold
boundaries, finite input validation, capped budgets, corrected-prefix state,
trace alignment and equality with target-only constrained greedy decoding.
