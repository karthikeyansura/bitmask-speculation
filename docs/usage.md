# Usage

<a id="engine"></a>

## Engine and modes

`TransformersEngine` loads a target and an optional draft model and serves one
request at a time. Every mode is greedy; speculative modes produce exactly the
tokens of their target-only reference.

| Mode | Draft grammar | Target grammar | Budget | Reference |
|---|---|---|---|---|
| `target_unconstrained` | — | No | — | itself |
| `target_constrained` | — | Yes | — | itself |
| `c1` | No | No | Fixed K | `target_unconstrained` |
| `c2` | No | Yes | Fixed K | `target_constrained` |
| `c3` | Yes | Yes | Fixed K | `target_constrained` |
| `c4` | Yes | Yes | Density schedule | `target_constrained` |

C2 versus C3 isolates draft-side masking; C3 versus C4 isolates scheduling.

## Python API

```python
from bitmask_speculation import ModelSpec, TransformersEngine

engine = TransformersEngine.from_pretrained(
    ModelSpec("/models/target"), ModelSpec("/models/draft"),
    local_files_only=True,
)
result = engine.generate(
    "Return a colour:", mode="c3", schema={"enum": ["red", "blue"]},
    max_new_tokens=32, draft_budget=5,
)
```

`ModelSpec` takes a local directory or a Hub ID with a full 40-character revision
SHA. Both models load onto one device without offloading or remote code; their
tokenizers must define the same token IDs. The prompt is tokenized as raw text
with special tokens, so apply any chat template beforehand. Schemas are JSON
documents (dict, string or boolean) with local `$ref`s only; each request gets an
independent XGrammar matcher.

| Result field | Meaning |
|---|---|
| `text`, `token_ids` | Generated content, excluding prompt and EOS |
| `stop_reason` | `eos`, `grammar` (terminal transition) or `length` |
| `eos_token_id` | The selected EOS, kept separate from content |
| `accounting` | Proposed, verified, accepted, emitted tokens and rounds |
| `schedule` | Per-round C4 density, requested and effective K |
| `acceptance_rate` | Accepted / verified draft tokens |
| `proposal_yield` | Accepted / proposed, counting discarded tails |
| `decode_seconds`, `tokens_per_second` | Device-synchronized decode time and throughput |

A `length` result can be an incomplete document; validate structured output
before use. C4 accepts `draft_policy=DensityPolicy(...)` to change thresholds or
bounds ([scheduling](design.md#density-schedule)).

<a id="runtime"></a>

## Device, precision and backend

`device`, `dtype` and `backend` default to `auto`: the current CUDA device;
bfloat16 at compute capability 8.0 and above, float16 below; the CUDA extension
when `nvcc` and Ninja are present, then Triton, then reference selection. The
extension wins on per-call latency in eager decoding ([timing](benchmarks.md#reductions)). A host without CUDA resolves to CPU, float32
and reference. Explicit values are validated and never fall back. The engine
records the resolved choices in `engine.device`, `engine.dtype`, `engine.backend`.

Native selection copies the grammar's CPU mask to the model device and widens
float16/bfloat16 logits losslessly to float32 ([kernels](kernels.md#dispatch)).

<a id="caching"></a>

## KV caching

- **Draft:** each round prefills the committed prefix once and feeds one token per
  proposal; the cache is released before verification. Disable with
  `draft_cache=False` or `--no-draft-cache`.
- **Target:** off by default. `target_cache=True` or `--target-cache` keeps one
  `DynamicCache` per request. Verification feeds only the uncached committed tokens
  plus the proposals; the next round crops rejected positions with
  `DynamicCache.crop` and feeds the correction or bonus first.

Cropping is exact only for full-attention layers. Models with linear-attention
(Qwen3.5 Gated DeltaNet) or sliding-window layers are rejected when the target
session opens; draft caching works for them because draft caches only append. In
bfloat16 and float16, full-prefix, one-token and chunked forwards round
differently, so an exact logit tie can resolve to a different token; float32 is
bit-identical across these paths.

<a id="cli"></a>

## Command line

```bash
bitmask-speculation generate 'Return a colour:' \
  --target /models/target --draft /models/draft --local-files-only \
  --mode c4 --schema-json '{"enum":["red","blue"]}' --max-new-tokens 32 --stats
```

Hub IDs take `--target-revision` and `--draft-revision`. `--schema-file` and
`--schema-json` are mutually exclusive; constrained modes require one.
`--draft-budget` sets K for C1–C3 (default 5) and `--max-draft-budget` caps C4
(default 8). `--device`, `--dtype`, `--backend`, `--no-draft-cache` and
`--target-cache` mirror the Python options.

Standard output is one JSON object with the resolved runtime, text, token IDs,
stop reason, accounting, schedule, rates and timing; `--output-format text` prints
only the content. `--stats` writes a summary line to stderr:

```
c4 | 24 tokens | stop=eos | 4 rounds | acceptance 0.96 | yield 0.92 | 31.4 tok/s
```

Exit code 2 signals a usage error and 1 a load, schema or generation error.

<a id="lower-level"></a>

## Token-level APIs

The engine composes three primitives that operate on token IDs and caller-owned
models.

```python
from bitmask_speculation.decoding import DensityPolicy, greedy_decode, speculative_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState

reference = greedy_decode(target, prompt_ids, max_new_tokens=128,
                          eos_token_ids=eos_ids, grammar=XGrammarState(compiled))
result = speculative_decode(target, draft, prompt_ids, max_new_tokens=128,
                            eos_token_ids=eos_ids, grammar=XGrammarState(compiled),
                            constrain_draft=True, draft_policy=DensityPolicy())
assert result.generation == reference
```

- `greedy_decode(model, prompt_ids, ...)` is the reference loop. The model returns
  the lowest token ID among tied maxima and rejects NaN scores.
- `verify_prefix(proposals, target_token_at, vocab_size)` lazily yields
  `accepted` events for the matching prefix, then one `correction`, a `bonus` after
  full acceptance, or a `fallback` for an empty proposal. It never requests a row
  past the first mismatch.
- `speculative_decode(target, draft, ...)` runs rounds of drafting and
  verification. `constrain_draft=True` drafts under a fork of the grammar;
  `draft_policy` enables C4.

`TransformersGreedyModel` wraps a loaded causal LM for either role. Verification
runs one forward over prefix plus proposals and reads row `L - 1 + i` for proposal
`i`. Sharded, offloaded and encoder-decoder models are out of scope.

<a id="grammar"></a>

## Grammar state

`XGrammarState(compiled_grammar)` wraps an XGrammar matcher compiled against the
target tokenizer and model vocabulary. It exposes allowed tokens, an owned packed
mask, `mask_density()`, `accept_token()` and `fork()`. Reading the mask never
advances state, and a fork shares only the compiled grammar.

XGrammar requires a stop token after the grammar body by default;
`terminate_without_stop_token=True` ends at the body instead. Pass EOS IDs that
match the grammar's stop tokens.

<a id="termination"></a>

## Termination

A target-selected EOS ends the request and is excluded from content. Otherwise
the token is appended; if its transition terminates the grammar the stop reason
is `grammar`, and reaching the output limit gives `length`. Precedence is EOS,
then grammar, then length. A draft EOS never ends a request: an empty proposal
still takes one target step. A zero output budget validates inputs and calls no
model.
