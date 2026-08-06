# Decoding contract

Greedy verification, grammar state and packed-mask semantics.

## Scope

The decoder handles one request on one device with greedy token selection.
A smaller draft model proposes a linear token sequence; the target verifies its
longest matching prefix. XGrammar supplies grammar state and packed token masks.
The project owns proposal scheduling, verification, and mask-processing kernels.

The correctness contract is equivalence to locally grammar-masked, target-only
greedy decoding. It does not establish distributional equivalence for sampling
or for conditioning a model on all complete strings accepted by a grammar.
Stochastic acceptance, continuous batching, and multi-GPU execution require
separate designs.

## Inputs and state

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

## Greedy verification

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

## Mask semantics

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

## Draft scheduling

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

## Experiment matrix

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

## Accounting and measurement

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

## Correctness conditions

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
