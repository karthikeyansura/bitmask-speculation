# Benchmark methodology

<a id="baseline-workloads"></a>

## Baseline workloads and records

The workload set compares unconstrained, target-constrained and jointly
constrained greedy speculation with their respective target-only references.
C2 versus C3 isolates draft-side masking; C3 versus C4 isolates scheduling.
The [budget sweep](benchmarks.md#budget-sweeps) repeats these comparisons.

### Cases and references

[baseline.json](../benchmarks/baseline.json) pairs three distinct prompts with
schemas for a contact card, a tool call and a nested user list. Limits are 128,
128 and 256 content tokens respectively. Schema paths resolve beside the
manifest, regardless of the working directory. Case IDs must be unique; the
workload fingerprint covers the ID, full prompt, schema and output limit.

| Variant | Draft grammar | Target grammar | Reference |
| --- | --- | --- | --- |
| `target_unconstrained` | No draft | Disabled | Defines unconstrained behavior |
| `target_constrained` | No draft | Enabled | Defines constrained behavior |
| `c1` | Disabled | Disabled | `target_unconstrained` |
| `c2` | Disabled | Enabled | `target_constrained` |
| `c3` | Enabled | Enabled | `target_constrained` |
| `c4` | Enabled, adaptive budget | Enabled | `target_constrained` |

Use the same prompt serialization, token IDs, output limit, EOS IDs, precision
and target model revision for each paired comparison. Construct fresh grammar
state for each constrained request. Never reuse state consumed by a reference.
For C3, pass `constrain_draft=True` and a forkable grammar to `speculative_decode`.
A trial's variant label describes how its caller configured generation; the
recording API does not infer or verify that configuration from output tokens.
C1 is a diagnostic, not a structured-generation baseline. Its schema validity
is still recorded, but invalid JSON does not alone indicate a C1 decoding bug.
Distinct prompts avoid repeating an identical unconstrained request under three
schema labels. Cases are not pooled across schemas, prompt IDs or variants.

These schemas allow variable text and some optional fields. They are baseline
fixtures, not an enum-heavy performance claim or an exhaustive schema matrix.
The independent validator uses JSON Schema draft 2020-12. `format` remains an
annotation rather than an assertion; semantic prompt compliance is not tested.
Schemas must be self-contained; external references are rejected.

### Recording a trial

Install the `benchmarks` extra and use
`bitmask_speculation.benchmarking.load_workloads(path)` to read the cases.
`record_trial` accepts the existing decoder result and an independently run
`GreedyResult` reference. It decodes and validates output after generation.
It does not load models, run AWS commands, mutate grammar state, or measure time.

Every record retains:

- The complete workload and prompt/EOS token IDs.
- Variant, fixed draft budget or C4 policy configuration, and repetition ID.
- C4 density, requested/effective budget and output offset for every round.
- Actual UTC recording time, source revision, model/tokenizer revisions, device,
  precision, dependency versions and timing scope.
- Raw content token IDs, stop reason and a separate selected EOS token.
- Raw proposal counts for speculative variants; target-only records have none.
- Exact comparison of tokens **and termination**, independent schema validity,
  and an optional unrounded decode duration.

Metadata values are strings. Use identifiable model names plus immutable
revisions, for example `organization/model@commit`, and record the complete
prompt template in additional metadata when applicable. Use detailed hardware
and library identifiers for measurements. Tiny local fixtures should identify
seeds and vocabulary definitions instead of pretending to use pretrained weights.

```python
from bitmask_speculation.benchmarking import TrialRecord, aggregate_trials

# `records` contains TrialRecord objects collected for one configuration.
# Use a new path for each run; exclusive creation protects completed records.
with open("trial-records.jsonl", "x", encoding="utf-8") as output:
    for record in records:
        output.write(record.to_json() + "\n")

with open("trial-records.jsonl", encoding="utf-8") as source:
    restored = [TrialRecord.from_json(line) for line in source]
summary = aggregate_trials(restored)
```

If generation or decoding fails, retain a `TrialRecord` with `error` set to the
exception type and message and leave output/comparison fields unset. Do not
replace failures with empty successful generations or drop them from a run.
`record_trial` propagates decoder and validator failures so a caller can record
the failure explicitly. Invalid JSON is a successful generation with
`schema_valid=False`, not an infrastructure exception.

### Metrics

`complete` means termination by EOS or grammar. Reaching the length limit is
incomplete even if the text happens to validate against the schema. Conversely,
EOS alone does not establish schema validity. Always examine exact agreement,
completion and schema outcomes together before making a correctness claim.

For repetitions of the **same** configuration:

```text
acceptance_rate = sum(accepted) / sum(verified)
proposal_yield  = sum(accepted) / sum(proposed)
observed_tokens_per_second = sum(content_tokens) / sum(decode_seconds)
```

Content token counts exclude the prompt and EOS. Unconsumed proposal tails
remain in `proposed`. Never average trial acceptance percentages or per-trial
throughputs to obtain these aggregate rates. Missing denominators return `None`,
which serializes as JSON `null`. Target-only acceptance rates are undefined.

Aggregation rejects mixed workloads, model/environment metadata, prompt/EOS IDs,
variants, draft budgets or policy parameters, and rejects duplicate repetition IDs. Any error
suppresses aggregate rates; counts from successful trials remain visible beside
the error count. Any missing timing suppresses throughput. Untimed records must
not contain durations. Raw observed throughput is not a validation verdict:
inspect the mismatch, length-limit and schema counts before interpreting it.

### Timing boundary

Leave `decode_seconds=None` and use `timing_scope="untimed"` for correctness
checks.

For the uncached reference, time the whole decode
loop including repeated prefix computation, drafting, target verification and
mask work. Exclude model loading, tokenization, schema compilation, independent
reference generation, detokenization, schema validation and serialization.
Warm up beforehand, synchronize device work at both timing boundaries and state
that the scope is `full_prefix_loop`. Prefill and decode are not separated by
this implementation. Cached implementations need a distinct timing scope.

Store every repetition, rotate variant order and retain environment details.

### Adaptive records

C4 uses `draft_budget=None`, an explicit `DensityPolicy`, and the decoder's
`RoundSchedule` tuple. A successful record has one decision per round, beginning
at offset zero with strictly increasing committed-output offsets. Requested K
must match the policy; effective K must match the remaining output cap. Error
records retain policy configuration without claiming a completed schedule.
The version-1 reader accepts older fixed-budget records without these optional
fields. Policy configurations remain distinct aggregation groups.

<a id="budget-sweeps"></a>

## Fixed and adaptive budget comparisons

`bitmask_speculation.budget_sweep.run_budget_sweep` compares target-only decoding,
C1/C2/C3 at each fixed budget, and C4 with one density policy. It accepts already
loaded model adapters, a workload, serialized prompt token IDs, EOS IDs, a fresh
grammar factory, a decode function and reproducibility metadata.

| Variant | Draft selection | Target selection | Budget |
| --- | --- | --- | --- |
| `target_unconstrained` | None | Unconstrained | None |
| `target_constrained` | None | Grammar constrained | None |
| `c1` | Unconstrained | Unconstrained | Fixed |
| `c2` | Unconstrained | Grammar constrained | Fixed |
| `c3` | Grammar constrained | Grammar constrained | Fixed |
| `c4` | Grammar constrained | Grammar constrained | Density policy |

C1 compares with the unconstrained target reference; C2–C4 compare with the
constrained reference. Every pair shares the exact prompt IDs, target revision,
precision, EOS IDs, schema and output limit. Match model token-ID meanings before
running; vocabulary size alone does not establish tokenizer compatibility.

### Running a comparison

Given loaded `target` and `draft` adapters, a compiled grammar for `case`, a
matching tokenizer and complete [metadata](benchmarks.md#baseline-workloads):

```python
from bitmask_speculation.budget_sweep import run_budget_sweep, summarize_budget_sweep
from bitmask_speculation.decoding import DensityPolicy
from bitmask_speculation.grammar.xgrammar import XGrammarState

records = run_budget_sweep(
    target, draft, case,
    prompt_token_ids=prompt_ids, eos_token_ids=eos_ids,
    grammar_factory=lambda: XGrammarState(compiled_grammar),
    decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=False),
    metadata=metadata,  # timing_scope="full_prefix_loop"
    synchronize=synchronize_models,
    fixed_budgets=(1, 2, 4, 8),
    policy=DensityPolicy(.005, .02, 1, 8),
    repetitions=5, warmups=1,
)
with open("budget-trials.jsonl", "x", encoding="utf-8") as output:
    for record in records:
        output.write(record.to_json() + "\n")
summaries = summarize_budget_sweep(records)
```

`synchronize_models` must wait for every device used by both models. For both
models on one CUDA device, use a callback calling `torch.cuda.synchronize(device)`.
For CPU-only adapters use a no-op. This callback is explicit because the runner
cannot infer the devices used by an arbitrary adapter. A faulty callback can
produce invalid timings. Model loading, compilation and cloud resources are
outside this API.

The factory must return a fresh matcher on every call, sharing only immutable
compiled grammar. The same consumed matcher cannot be reused for repetitions,
variants, warmups or references. The decode callback must preserve all output
content needed for schema validation; do not remove invalid tokens to make a
result pass.

### Execution and timing

1. Validate configuration and metadata. Budgets are unique positive integers;
   use at least one warmup and two measured repetitions.
2. Generate independent constrained and unconstrained references.
3. Warm every configuration equally and require exact reference agreement.
   Reference or warmup failures abort before measured records are produced.
4. For each repetition, rotate configuration order left by its repetition index.
   This limits fixed order bias; a partial rotation is not fully balanced.
5. Create fresh matcher state, synchronize, start the clock, decode, synchronize,
   and stop the clock. Validate decoded JSON and schema only afterward.
6. Retain each measured trial with actual UTC recording time, raw output,
   accounting, exact-match/schema verdicts and C4 policy/trace when applicable.

The interval includes full-prefix forward passes, drafting, mask work, density
queries and policy decisions. It excludes model loading, prompt tokenization,
grammar compilation, initial matcher construction, reference generation,
detokenization, schema validation and serialization. This implementation does
not separate prefill from decode or reuse KV caches.

Measured failures become explicit error records and the runner continues once
pending device work is drained. They suppress aggregate rates for the affected
configuration. A broken synchronizer aborts the sweep. `on_record(record)` can persist each completed trial outside the timed interval;
`on_reference(references)` receives the independent target generations before
warmup. Persistence errors abort the sweep. Records are also returned at completion.
The runner does not resume partial runs.

### Interpreting records

Summaries keep variants, fixed budgets and policy parameters separate, require
identical repetition coverage, and reject mixed workloads or execution metadata.
They expose exact agreement, completion, schema validity, errors, pooled proposal
ratios and observed throughput. They do not automatically declare a winning
policy or calculate a headline speedup. Inspect all correctness counts first.

Different policies may expose different numerical behavior through target
forward-pass shape. Investigate any token mismatch; do not relax equality to a
percentage. Keep regressions and incomplete output visible. Density thresholds
are experiment settings, not a guarantee of acceptance.

<a id="validation"></a>

## Validation artifacts

Benchmark output contains the configuration, source hashes, model and tokenizer
identities, hardware and dependency versions, raw trials, and a derived summary.
Each trial carries token IDs, termination, schema validity, exact-reference
agreement, proposal counts and elapsed time. Failed trials remain in the output.

### Device coverage

Single-device tests cover kernel correctness, streams, output ownership, seeded
sampling and memory/synchronization checks. Device-guard tests require two visible
CUDA devices in one process. They check input-device selection, restoration of
the caller's device and rejection of mismatched-device tensors. These tests do
not exercise distributed inference or multi-GPU throughput.

## Runnable measurements

The installed module `python -m bitmask_speculation.benchmarks` provides
`reductions`, `decode` and `report`. Use `--help` on each subcommand.

`reductions` computes equivalent masked argmax, exact count and status outputs
with `torch`, `cuda` or `triton`. CUDA uses per-row native reductions plus stacking;
Triton uses a batched reduction. Inputs are deterministic seeded float32 scores
and Bernoulli mask bits, with token zero forced valid in every row. Padding bits
are zero. Latency is per whole batch.

| Scope | Measured work |
|---|---|
| `graph_device` | CUDA-event duration of graph replay, divided by captured iterations; no Python allocation |
| `async_wrapper` | Host wall time for unchecked calls plus final stream synchronization |
| `checked_wrapper` | Host wall time with a status read/check on every call plus final synchronization |

Compilation and warmup precede capture and timing. Every result is compared with
the reference outside the interval. Graph replay amortizes dispatch; it is not
an isolated-kernel profiler. The report stores raw samples, median, minimum and
nearest-rank p95. Increase repetitions for meaningful tail estimates.

`decode` loads models once. Remote sources require full revision SHAs; local
model contents are hashed. Raw prompt text is the default. `--chat-template`
renders a user turn with a generation prompt and `enable_thinking=False`, then
encodes without adding special tokens a second time. The rendered text, prompt
IDs, tokenizer definition and independent constrained/unconstrained references
are saved. Model loading and per-case tokenization/grammar setup have separate
timings. Measured records are flushed before the next trial.

`benchmarks/regimes.json` contains six self-contained cases: enum-heavy object,
scalar enum, nested object, mixed categorical/numeric/text fields, free-text JSON
string and an open object. Free text still follows JSON string syntax. Each case
fixes its prompt, full schema and output-token budget; it is not a corpus average.

`report` recomputes per-case comparisons from a completed run. It requires every
manifest case, configuration and repetition exactly once. C1 compares with the
unconstrained target; C2–C4 compare with the constrained target. Ratios divide
summed baseline latency by summed configuration latency over matched repetitions.
Both sides must finish, match exactly, pass full-schema validation and contain
finite timings. Failed, incorrect or length-limited trials suppress the ratio.
No cross-case average is emitted. A partial run retains its raw records and
failure status; it cannot be resumed into the same directory.

## A10G measurements

[Raw samples and trial data](../benchmarks/results/a10g.json) contain the complete
schema cases, pinned model revisions, source hashes and per-trial outcomes.
The model pair is Qwen3-4B / Qwen3-0.6B, using BF16, greedy selection, a chat
template with thinking disabled, and Triton constrained selection on one A10G.
The decoder evaluates full prefixes without a KV cache.

The sweep used one warmup and two measured repetitions per configuration.
All 108 trials completed and matched their respective target-only references.
All 72 grammar-constrained trials passed schema validation; 18 unconstrained
outputs failed schema validation. Those failures remain in the data and suppress
ratios for the affected configurations.

| Schema regime | C3, K=4 / constrained target | C4 / constrained target |
|---|---:|---:|
| enum-heavy | 1.234× | 1.211× |
| enum-small | 0.992× | 0.984× |
| free-text | 0.617× | 0.766× |
| mixed-record | 0.812× | 0.832× |
| nested-record | 1.101× | 1.036× |
| open-object | 0.807× | 0.752× |

Ratios divide total baseline latency by total configuration latency; values below
1 indicate slower decoding. These are measurements of this configuration with
two repetitions, not a general speedup across workloads or serving engines.

Reduction medians use 20 repetitions of 20 operations, six rows, vocabulary size
151,936, and approximately 1% valid bits. Values are microseconds per whole batch.

| Backend | Graph replay | Async wrapper | Checked wrapper |
|---|---:|---:|---:|
| torch | 257.55 | 422.86 | 460.94 |
| cuda | 1517.59 | 1530.46 | 1569.62 |
| triton | 6.21 | 116.44 | 161.82 |

The CUDA comparison uses six per-row argmax/count calls plus output stacking;
Triton processes all rows in a batched two-pass reduction. Wrapper scopes include
Python dispatch and output allocation; checked wrappers also read batch status.
Graph replay excludes Python allocation. These scopes do not measure LLM
throughput. The single-row CUDA implementation is slower than both alternatives
on this workload.
