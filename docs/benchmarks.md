# Benchmarks

`python -m bitmask_speculation.benchmarks` provides four commands: `reductions`
times the masked reduction backends, `decode` runs end-to-end decoding sweeps,
`report` turns a completed sweep into per-case tables, and `profile` brackets
reductions for Nsight. Every run writes to a new directory with its configuration,
source hashes, model and tokenizer revisions, hardware and dependency versions,
the independent references and every raw trial.

<a id="decode"></a>

## Decode sweeps

```bash
python -m bitmask_speculation.benchmarks decode \
  --target /models/target --draft /models/draft --local-files-only \
  --device cuda:0 --dtype bfloat16 --backend triton --chat-template \
  --workloads benchmarks/regimes.json --budgets 1 4 --repetitions 5 \
  --output runs/decode-a10g
python -m bitmask_speculation.benchmarks report runs/decode-a10g
```

A sweep generates the constrained and unconstrained target-only references,
warms every configuration, then runs C1–C3 at each fixed budget and C4 under its
policy. Configuration order rotates with each repetition and each trial starts
from a fresh grammar matcher.

**Timing boundary.** The clock starts after a device synchronization and stops
after the next one. It covers target and draft forwards, mask filling and
transfer, selection, density queries and scheduling. Model loading, tokenization,
grammar compilation, reference generation, detokenization and validation fall
outside it. Records name the scope: `full_prefix_loop`, `round_cached_draft_loop`
or `request_cached_target_loop`, together with the draft and target cache modes.

**Gates.** A ratio is reported only when both sides completed by EOS or grammar
termination, matched their target-only reference token for token, including the
stop reason, and validated against the full schema (JSON Schema 2020-12) in every
repetition. Length-limited, mismatched and failed trials stay in the record and
suppress the ratio.

**Metrics.** For each configuration: `acceptance = Σ accepted / Σ verified`,
`yield = Σ accepted / Σ proposed` and `tokens/s = Σ content tokens / Σ seconds`.
Speedups divide summed baseline time by summed configuration time over matched
repetitions. `report` prints exactness, acceptance, yield, tokens/s and the ratio
to the target for every configuration, followed by C4/C3 per fixed budget;
`--json` emits the full record.

<a id="workloads"></a>

## Workloads

| Manifest | Cases | Use |
|---|---|---|
| `benchmarks/primary.json` | The six schemas of the A100 study: pure enum, nested enum, enum + short text, enum + text, long and short free text | Qwen3.5-4B / 0.8B (pinned revisions), FP16, raw prompts, C3 K=5 vs C4 K ≤ 8 |
| `benchmarks/regimes.json` | Enum-heavy object, scalar enum, nested record, mixed record, free-text string, open object | Mask-density regimes for the A10G study |
| `benchmarks/baseline.json` | Contact card, tool call, nested user list | Long-output correctness fixtures |

Each case fixes its prompt, a self-contained schema and an output cap.

<a id="a10g-decode"></a>

## A10G decode study

A10G (24 GB) · Qwen3-4B / Qwen3-0.6B · BF16 · Triton selection · chat template
with thinking disabled · full-prefix target and draft · one warmup and two
measured repetitions. All 108 trials matched their references; all 72
grammar-constrained outputs validated against their schemas.

| Regime | C3 K=4 / target | C4 / target | C3 acceptance | C4 acceptance | C4 / C3 |
|---|:---:|:---:|:---:|:---:|:---:|
| enum-heavy | **1.234×** | **1.211×** | 100% | 100% | 0.98× |
| nested-record | **1.101×** | **1.036×** | 96% | 96% | 0.94× |
| enum-small | 0.992× | 0.984× | 100% | 100% | 0.99× |
| mixed-record | 0.812× | 0.832× | 87% | 94% | 1.02× |
| open-object | 0.807× | 0.752× | 76% | 69% | 0.93× |
| free-text | 0.617× | 0.766× | 64% | 50% | **1.24×** |

Constrained drafting beats the target where the schema dictates most tokens. The
density schedule earns its keep on free text: dropping to K=1 recovers 24% over
fixed K=4 by not spending draft forwards that the target would reject. The ceiling
on this GPU comes from the draft's relative cost, which the profile below
quantifies. Raw trials: [`a10g.json`](../benchmarks/results/a10g.json),
[`a10g-trials.jsonl`](../benchmarks/results/a10g-trials.jsonl).

<a id="reductions"></a>

## Reduction timing

```bash
python -m bitmask_speculation.benchmarks reductions \
  --backend cuda --rows 6 --vocab 248320 --density .01 --output runs/reduce-cuda
```

Inputs are seeded float32 scores and Bernoulli masks with token zero forced
valid. Every backend's tokens, counts and statuses are checked against a PyTorch
reference before timing. Three scopes are reported per whole batch:

| Scope | Measures |
|---|---|
| `graph_device` | CUDA-event time of a captured CUDA graph replay, per operation |
| `async_wrapper` | Host wall time of unchecked calls plus a final synchronization |
| `checked_wrapper` | As above, reading the status after every call |

A10G · vocabulary 248,320 · 1% valid bits · median μs per batch over 20
repetitions of 20 operations:

| Rows | Graph replay: torch | CUDA | Triton | Eager call: torch | CUDA | Triton |
|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 99.7 | 11.1 | 5.0 | 423.4 | **19.0** | 120.1 |
| 2 | 154.8 | 11.2 | 5.4 | 420.2 | **19.5** | 120.0 |
| 6 | 462.5 | 18.3 | 8.4 | 494.4 | **20.8** | 119.2 |
| 8 | 609.7 | 25.3 | 20.5 | 640.8 | **27.9** | 120.2 |
| 12 | 894.4 | 33.8 | 28.8 | 926.7 | **36.5** | 119.9 |

From 1 to 12 rows, device time grows 3× for CUDA and 6× for Triton against 9×
for PyTorch, which reduces row by row. On the device alone Triton is fastest;
per eager call, the scope the decoder actually uses, the CUDA extension is about 6× faster than Triton because
a C++ launch costs a fraction of Triton's Python dispatch. That makes the
extension the default backend. End to end, the faster selection is worth about
1% of decode throughput on the six A10G regimes, consistent with selection being
a small share of each round. Vocabulary 151,936 and every sample are in
[`a10g-reductions.json`](../benchmarks/results/a10g-reductions.json).

<a id="decode-profile"></a>

## Decode profile

Nsight Systems trace of the enum-heavy regime on the A10G configuration above,
with NVTX ranges per forward and CUPTI kernel activity, each decode captured once:

| Span | CPU span (median) | GPU kernel time (median) | Kernel launches |
|---|---:|---:|---:|
| Target forward (4B) | 64 ms | 25.3 ms | 2,081 |
| Draft forward (0.6B) | 50 ms | 5.9 ms | 1,591 |
| Mask copy + selection + status read | 0.8–2.0 ms | 0.01 ms | 3 |

The GPU is busy for 38% of target-only decoding, 18% under C3 K=4 and 15% under
C4. Forwards issued back to back without synchronization take as long as
synchronized ones, so the host launch rate is the limit. A draft forward costs 77%
of a target forward in wall time but 23% in GPU time; grammar work and selection
stay under 3% of decode time. At GPU-time cost a K=4 round would take about
4 × 5.9 + 25.3 ms for up to five tokens, which is the headroom CUDA-graph capture of
both forwards targets.

<a id="profiling"></a>

## Profiling commands

`profile` prepares seeded inputs, warms the backend, checks it against the
reference, then repeats unchecked reductions inside a CUDA profiler range on one
stream:

```bash
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --output=runs/reduce-nsys \
  python -m bitmask_speculation.benchmarks profile --backend cuda \
  --rows 6 --vocab 248320 --density .01 --iterations 10 --output runs/reduce-nsys-inputs

ncu --profile-from-start off --set basic --launch-count 2 --export runs/reduce-ncu \
  python -m bitmask_speculation.benchmarks profile --backend cuda \
  --rows 6 --vocab 248320 --density .01 --iterations 1 --output runs/reduce-ncu-inputs
```

Each call issues a partial and a final kernel, hence `--launch-count 2`. Nsight
Compute needs GPU performance-counter access on the host. Use the uninstrumented
`reductions` runner for latency.
