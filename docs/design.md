# Design

<a id="contract"></a>

## Correctness contract

Speculative output equals grammar-masked, target-only greedy decoding token for
token, including the stop reason. The draft influences only how many target
forwards a request needs, never what the target emits. The contract covers one
request on one device with greedy selection; stochastic acceptance and batched
requests are separate designs.

<a id="round"></a>

## One round

For a committed prefix of length `L`:

1. **Schedule.** C4 reads the density of the committed mask and picks K; other
   modes use a fixed K. K is capped by the remaining output budget.
2. **Draft.** The draft proposes up to K tokens. Under C3/C4 it selects against a
   fork of the committed grammar state, advancing only the fork.
3. **Discard.** The fork is dropped before verification. The committed state never
   visited the proposed path, so rejection needs no rollback.
4. **Verify.** One target forward covers prefix plus proposals; row `L - 1 + i`
   predicts proposal `i`. The target accepts the longest matching prefix, emits its
   own token at the first mismatch, and emits a bonus token from the final row when
   every proposal matches. Each emitted token advances committed grammar state once.
5. **Stop.** Verification halts at EOS, grammar termination or the output limit,
   even mid-round. Rows after a mismatch condition on a rejected token and are
   never reused.

<a id="masks"></a>

## Packed masks

A mask row holds `ceil(V / 32)` int32 words; token `t` is valid when bit `t % 32`
of word `t // 32` is set. Words are bit patterns: `-1` has 32 valid bits and
`-2147483648` has only bit 31. Padding bits at `t ≥ V` never count and never win.

Masked argmax returns the highest-scoring valid token with ties broken by the
lowest ID. Valid `-inf` scores remain eligible, NaN anywhere in the row is an error
and an empty valid set is an error. Inputs are never modified.

<a id="trace"></a>

## Worked trace

The committed output is `{"op":` and the grammar allows four objects: `op` is
`"get"` or `"put"` and `id` is `7` or `9`. Six tokens matter:

| ID | Text | Valid after |
|---|---|---|
| 10 | `"get"` | `{"op":` |
| 11 | `"put"` | `{"op":` |
| 12 | `,"id":` | either operation |
| 13 | `7` | the `id` separator |
| 14 | `9` | the `id` separator |
| 15 | `}` | either integer |

The target-only continuation is `[10, 12, 13, 15]`, i.e. `{"op":"get","id":7}`.

**Partial acceptance (K = 4).** The constrained draft proposes `[10, 12, 14, 15]`.
Every proposal is legal along the draft's own path; legality is not agreement.

| i | Row | Draft | Target | Action |
|---|---|---|---|---|
| 0 | `L - 1` | 10 | 10 | accept |
| 1 | `L` | 12 | 12 | accept |
| 2 | `L + 1` | 14 | 13 | emit correction, stop |
| 3 | `L + 2` | 15 | — | discard |

The round emits `[10, 12, 13]`. Row `L + 2` conditions on the rejected `9`, so
`}` cannot be accepted by skipping the mismatch; the next round recomputes it.
With target caching enabled, the cache keeps `L + 2` positions and the next round
feeds `13` before its proposals.

**Full acceptance (K = 3).** Proposals `[10, 12, 13]` all match; row `L + 2` now
conditions on the accepted `7`, and the target emits `15` as a bonus. Four tokens
come from one target forward and the grammar terminates.

**First-token rejection (K = 4).** Proposals `[11, 12, 14, 15]`: the target prefers
`10`, so one comparison is made and nothing is accepted. Density here is `2 / 32`;
it bounds the support but does not rank the candidates.

**Output limit.** With two slots left, K is capped at 2. Accepting `[10, 12]`
exhausts the budget: no bonus is requested and the result, `{"op":"get","id":`,
is reported as `length`.

<a id="accounting"></a>

## Accounting

| Scenario | Proposed | Verified | Accepted | Emitted | Acceptance | Yield |
|---|---|---|---|---|---|---|
| Partial acceptance | 4 | 3 | 2 | 3 | 2/3 | 2/4 |
| Full acceptance + bonus | 3 | 3 | 3 | 4 | 3/3 | 3/3 |
| First-token rejection | 4 | 1 | 0 | 1 | 0/1 | 0/4 |
| Output limit | 2 | 2 | 2 | 2 | 2/2 | 2/2 |
| Empty proposal | 0 | 0 | 0 | 1 | — | — |

- `proposed` counts every token sent to verification, including discarded tails.
- `verified` counts comparisons actually made, including the first mismatch.
- `accepted` counts matching proposals; corrections and bonuses are emitted, not accepted.
- `emitted` equals the returned content length; EOS is excluded.

`accepted ≤ verified ≤ proposed` always holds. Acceptance (`accepted / verified`)
measures agreement; yield (`accepted / proposed`) also charges wasted tails. Both
are `None` when their denominator is zero. Corpus rates sum the counts first and
divide once; averaging per-request rates is incorrect.

<a id="density-schedule"></a>

## Density schedule

C4 reads `density = valid_count / V` from the committed grammar state once per
round, before forking. `DensityPolicy(low_density, high_density, min_budget,
max_budget)` maps it to half-open bands:

| Density | Requested K | Default |
|---|---|---|
| `d < low_density` | `max_budget` | `d < 0.005 → 8` |
| `low_density ≤ d < high_density` | `(min_budget + max_budget) // 2` | `d < 0.02 → 4` |
| `d ≥ high_density` | `min_budget` | otherwise `1` |

The effective K is `min(requested, remaining)` and every decision is recorded in
`result.schedule` with its output offset. An all-valid state reports density 1 and
takes the minimum budget. A correction determines the next round's density;
rejected draft states never reach the scheduler.

The count runs on the CPU mask that XGrammar fills for selection, through a
byte-wise popcount over the int32 storage (about 57 μs for V = 248,320).
Sparse masks mean the schema dictates the continuation and a constrained draft
rarely diverges; dense masks mark free text, where long proposals waste draft
forwards.
