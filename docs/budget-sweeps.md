# Fixed and adaptive budget comparisons

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

## Running a comparison

Given loaded `target` and `draft` adapters, a compiled grammar for `case`, a
matching tokenizer and complete [metadata](baseline-workloads.md):

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

## Execution and timing

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
configuration. A broken synchronizer aborts the sweep. Results are returned at
completion; this is not a crash-resumable runner. Choose bounded workloads and
persist each completed sweep under a new output filename.

## Interpreting records

Summaries keep variants, fixed budgets and policy parameters separate, require
identical repetition coverage, and reject mixed workloads or execution metadata.
They expose exact agreement, completion, schema validity, errors, pooled proposal
ratios and observed throughput. They do not automatically declare a winning
policy or calculate a headline speedup. Inspect all correctness counts first.

Different policies may expose different numerical behavior through target
forward-pass shape. Investigate any token mismatch; do not relax equality to a
percentage. Keep regressions and incomplete output visible. Density thresholds
are experiment settings, not a guarantee of acceptance.

CPU fixture runs validate orchestration and evidence serialization. They do not
establish pretrained-model latency or GPU throughput. GPU comparisons require
separate runs with pinned model/tokenizer revisions and complete hardware and
software metadata.
