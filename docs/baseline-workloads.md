# Baseline workloads and records

The workload set compares unconstrained, target-constrained and jointly
constrained greedy speculation with their respective target-only references.
C2 versus C3 isolates draft-side masking; C3 versus C4 isolates scheduling.
The [budget sweep](budget-sweeps.md) repeats these comparisons.

## Cases and references

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

## Recording a trial

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
# Use a new path for each run; exclusive creation protects previous evidence.
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

## Metrics

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

## Timing boundary

Leave `decode_seconds=None` and use `timing_scope="untimed"` for correctness
checks. The initial integration tests do this; they publish no latency numbers.

For the uncached reference, time the whole decode
loop including repeated prefix computation, drafting, target verification and
mask work. Exclude model loading, tokenization, schema compilation, independent
reference generation, detokenization, schema validation and serialization.
Warm up beforehand, synchronize device work at both timing boundaries and state
that the scope is `full_prefix_loop`. Prefill and decode are not separated by
this implementation. Cached implementations need a distinct timing scope.

Store every repetition, rotate variant order and retain environment details.

## Adaptive records

C4 uses `draft_budget=None`, an explicit `DensityPolicy`, and the decoder's
`RoundSchedule` tuple. A successful record has one decision per round, beginning
at offset zero with strictly increasing committed-output offsets. Requested K
must match the policy; effective K must match the remaining output cap. Error
records retain policy configuration without claiming a completed schedule.
The version-1 reader accepts older fixed-budget records without these optional
fields. Policy configurations remain distinct aggregation groups.
