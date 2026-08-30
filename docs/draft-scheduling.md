# Density-selected draft budgets

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
