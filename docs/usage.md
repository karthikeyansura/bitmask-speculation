# Generation APIs

<a id="engine"></a>

## Request engine

`DecodingEngine` owns references to already loaded target/draft adapters. Its
`generate` method delegates to `greedy_decode` or `speculative_decode`; it contains
no second decoding loop. `GenerationResult` exposes content token IDs, stop reason,
EOS ID, optional proposal accounting and C4 scheduling decisions.

| Mode | Target grammar | Draft grammar | Budget |
| --- | --- | --- | --- |
| `target_unconstrained` | No | No draft | None |
| `target_constrained` | Yes | No draft | None |
| `c1` | No | No | Fixed |
| `c2` | Yes | No | Fixed |
| `c3` | Yes | Yes | Fixed |
| `c4` | Yes | Yes | Density policy |

Pass a `grammar_factory` for constrained token-level calls. It must return fresh
state; request state is never retained by the engine. A factory that reuses a
mutable matcher violates this API contract. Model adapters must share token-ID
meanings, and concurrent use of one model instance is not promised.

### Text and model loading

`TransformersEngine.from_pretrained` is the optional text facade. `ModelSpec`
accepts a local model directory or a Hub ID with a full revision SHA. Target-only
use does not load a draft. Both models reside on the explicitly selected device;
there is no offloading, automatic sharding or remote model-code execution.
Tokenizer definitions must agree before loading the model pair.

```python
from bitmask_speculation.models.loading import ModelSpec, TransformersEngine

engine = TransformersEngine.from_pretrained(
    ModelSpec("/path/to/target"), ModelSpec("/path/to/draft"),
    device="cuda", dtype="bfloat16", local_files_only=True,
)
result = engine.generate(
    "Return a colour:", mode="c3", schema={"enum": ["red", "blue"]},
    max_new_tokens=32, draft_budget=5,
)
print(result.text, result.stop_reason, result.accounting)
```

The text facade tokenizes a raw prompt with special tokens enabled; it does not
apply a chat template. Render any model-specific template before calling it.
Schema arguments are JSON documents, not implicit paths. Each request receives
an independent XGrammar matcher; compiled grammar caching belongs to XGrammar.
Only local schema references are accepted. Grammar-constrained generation is not
a substitute for application-level schema validation, and a length-limited result
may be incomplete. Inspect the stop reason and validate the complete output.

Zero output budget calls no model forward pass. Content excludes prompt and EOS.
The facade decodes content without stripping special tokens or normalizing spaces.
No throughput metric is inferred from these calls. Model loading, grammar
compilation and decoding timing belong to the benchmark harness.

The engine keeps one target KV cache per request and a fresh draft KV cache per
proposal round. Set `target_cache=False` or `draft_cache=False` when loading the
engine, or pass `--no-target-cache` or `--no-draft-cache` to either CLI, for
full-prefix evaluation. `device`, `dtype` and
`backend` default to `"auto"`: the current CUDA device when available, bfloat16
on compute capability 8.0 or newer and float16 below it, and Triton, else the CUDA
extension, else reference selection. Without CUDA, `auto` selects CPU, float32 and
reference. Explicit `"reference"`, `"cuda"` or `"triton"` values are validated and
never fall back. The engine's `device`, `dtype` and `backend` attributes record the
resolved values. See [backend contracts](kernels.md#triton-reductions)
for precision, mask transfers and synchronization. This is still single-request
inference; batched primitives do not batch grammar transitions.

<a id="cli"></a>

## Generation CLI

Install from the repository with `uv sync --extra models --extra grammar`, then
activate `.venv`. Both `bitmask-speculation generate` and
`python -m bitmask_speculation generate` use the same entry point. `--help` works
without optional model dependencies.

For locally saved, compatible model/tokenizer directories:

```bash
bitmask-speculation generate 'Return a colour:' \
  --target /path/to/target --draft /path/to/draft \
  --mode c3 --schema-json '{"enum":["red","blue"]}' \
  --max-new-tokens 32 --local-files-only
```

For Hub IDs, supply full `--target-revision` and `--draft-revision` SHAs.
Both model and tokenizer loads use those revisions; remote model code is disabled.
`--device`, `--dtype` and `--backend` default to `auto` with the same resolution
as the Python API. Pass explicit values to override any of them.

`--schema-file schema.json` and `--schema-json '{...}'` are mutually exclusive.
Constrained modes require a schema; unconstrained modes reject one. Schemas are
explicit JSON documents and external references are rejected. Target-only modes
reject draft options. Fixed `--draft-budget` applies to C1–C3; C4 instead accepts
`--max-draft-budget` and uses the default density thresholds.

The default stdout is one JSON object containing the resolved device, dtype and
backend, text, content token IDs, stop reason, selected EOS, proposal accounting
and schedule. `--output-format text`
prints only generated content. Errors go to stderr. Exit code 2 indicates command
usage errors; exit code 1 indicates a load, schema or generation error. A successful
length-limited request still exits 0; inspect `stop_reason` and validate the output
before treating a structured document as complete.

No latency or throughput number is printed. The benchmark harness must define
warmup, synchronization and timing boundaries separately. See [engine contracts](usage.md#engine) and the
[Python example](../examples/quickstart.py).

Use `--backend reference|cuda|triton` to select constrained token processing
explicitly. Native backends require a CUDA device; unsupported hardware fails
before model loading. See [backend contracts](kernels.md#triton-reductions).

<a id="reference-decoding"></a>

## Greedy reference decoding

`greedy_decode` operates on token IDs for one unpadded prompt. Tokenization,
model loading, and device placement belong to the caller. The model must return
the lowest token ID among tied maximum scores and reject NaN prediction scores.

```python
from bitmask_speculation.decoding import greedy_decode

class NextTokenModel:
    vocab_size = 8

    def next_token(self, prefix: tuple[int, ...]) -> int:
        return 3 if prefix[-1] == 1 else 7

result = greedy_decode(
    NextTokenModel(), [1], max_new_tokens=8, eos_token_ids=[7]
)
assert result.token_ids == (3,)
assert result.stop_reason == "eos"
assert result.eos_token_id == 7
```

### Output contract

The immutable result contains newly generated content, excluding the prompt and
any selected EOS. `eos_token_id` records which stop token was selected, or `None`
when the output limit or grammar termination was reached. A grammar terminal
content token is preserved, with stop reason `grammar`; a selected EOS takes
precedence and remains separate from content. Multiple EOS IDs are supported. They are
explicit inputs, never inferred from a tokenizer or generation configuration.

Each prediction receives the complete prompt plus accepted content. The loop
preserves the caller's prompt and performs at most `max_new_tokens` predictions.
A zero limit validates inputs and makes no model call. At the length limit it
does not make an extra prediction to discover whether EOS would follow. An EOS
selected on the last allowed prediction still produces an `eos` stop reason.
EOS tokens already in the prompt are ordinary context.

A prompt must be nonempty. Token IDs and the vocabulary size are Python integers;
booleans and out-of-range IDs are rejected. The output limit is a nonnegative
integer. A model failure is propagated rather than converted to partial output.

### Transformers adapter

Install the `models` extra as described in [development](development.md).
Given an already loaded decoder-only `model` and tokenized `prompt_ids`:

```python
from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.models.transformers import TransformersGreedyModel

model.eval()
reference = TransformersGreedyModel(model)
result = greedy_decode(
    reference, prompt_ids, max_new_tokens=128, eos_token_ids=eos_ids
)
```

Place all model parameters and buffers on one device before constructing the
adapter, and leave the device and vocabulary unchanged while using it. Sharded,
offloaded, and encoder-decoder models are outside this adapter's contract.
The caller must keep the total prefix within the model's context capacity.

For prefix length `L`, the adapter consumes logits shaped `[1, L, vocab_size]`
and selects from position `L - 1`. It runs in inference mode, requires evaluation
mode, and recomputes the full prefix for standalone `next_token` and
`verify_prefix` calls. Cached evaluation happens only inside sessions.
`greedy_decode` and `speculative_decode` open `target_session()` once per request.
The speculative loop also opens `draft_session()` for each round: it prefills once,
passes only each new token with its cached keys/values, and releases the cache
before verification. Attention masks cover cached and current tokens; position IDs
follow the unpadded prefix. Each session owns its cache exclusively and drops it
on exit or failure. Models must return `past_key_values`; otherwise use
`draft_cache=False`. The target session's requirements are listed under
[cached verification](usage.md#cached-verification).
Exact ties choose the lowest token ID; NaN next-token logits fail explicitly.
Infinities retain normal argmax ordering, including lowest-ID selection when
all scores are negative infinity. This unmasked adapter has no validity mask;
a constrained decoder must separately detect an empty valid-token set.

No repetition penalties, forced tokens, suppression lists, sampling, or other
generation-config processors are applied. Comparisons with `model.generate`
require those settings to be disabled and matching EOS/output-limit semantics.


With `target_cache=False`, full-prefix recomputation is a correctness reference,
not a throughput baseline. Compare speculative and target-only decoding under the
same cache modes, grammar and generation settings. Pass `grammar=state` for
[constrained greedy decoding](usage.md#grammar-selection). The [speculative request loop](usage.md#speculative-decoding) uses the same result and
termination contract.
The [prefix verifier](usage.md#prefix-verification) uses this greedy behavior to check
one proposal round, without adding a generation loop.

<a id="prefix-verification"></a>

## Prefix verification

`verify_prefix` compares proposals with target choices in order. It yields the
longest matching prefix and then one target correction. If every proposal
matches, it can yield one bonus choice. With no proposals it yields a target-only
fallback. It never skips a mismatch to accept later proposals.

```python
from bitmask_speculation.decoding import verify_prefix

rows = [10, 12, 13]  # Later rows must not be requested after the mismatch.
events = list(verify_prefix([10, 12, 14, 15], rows.__getitem__, vocab_size=32))
assert [(event.token_id, event.kind) for event in events] == [
    (10, "accepted"), (12, "accepted"), (13, "correction")
]
```

Each immutable `VerifiedToken` has a `token_id` and a `kind`: `accepted`,
`correction`, `bonus`, or `fallback`. These are target-confirmed choices, not yet
text emissions. EOS has no special meaning to this primitive.

### Selection order and ownership

The callback `target_token_at(i)` selects after the committed prefix followed by
`proposals[:i]`. It is called once per requested event, with indices starting at
zero. Index `len(proposals)` selects the bonus after full acceptance. The callback
owns score selection, including tie handling and NaN rejection, just as in the
[greedy reference](usage.md#reference-decoding).

Proposal IDs are validated and copied immediately. Target choices are validated
as they are requested. A callback failure propagates; previously yielded events
are not rolled back. Callers must discard a failed request or restore their own
state rather than treat it as successful partial output.

The iterator never requests a row after a mismatch. It also never requests the
bonus unless the consumer advances beyond every accepted proposal. Consumers
must stop at their output limit, EOS, or grammar termination before requesting
another event. They should release or close an unused iterator promptly.

The primitive does not own a model cache, tokenizer, grammar matcher, output
budget, or EOS policy. It is a round-level building block, not a complete
speculative generation API. Collecting all events is appropriate only when no
request-level stop condition intervenes. The [speculative request
loop](usage.md#speculative-decoding) owns these conditions across rounds.

### One-forward Transformers verification

For an initialized `TransformersGreedyModel` adapter, call:

```python
stream = adapter.verify_prefix(committed_prefix, proposals)
```

The standalone call immediately runs one uncached forward pass over the prefix plus
proposals. For a fixed prefix length `L`, event index `i` reads logit row
`L - 1 + i`. The bonus reads `L + len(proposals) - 1`, the final row. These offsets
never depend on a growing output list. Empty proposals read row `L - 1`.

Argmax and NaN checks occur lazily, only for consumed rows. After a correction,
later rows condition on a rejected token and cannot predict the corrected prefix.
The next round must recompute from the corrected committed prefix. The iterator
retains its logits until exhausted, closed, or released; it never shares a mutable
workspace with another request. Inference mode is confined to model operations
and does not remain active in the caller while the iterator is suspended.

The adapter performs raw-logit selection with the same single-device and context
capacity restrictions as the greedy reference. The entire prefix plus proposal
sequence must fit the model context. Early iterator termination saves selection
work, not the forward computation already performed. This implementation makes
no speedup claim. Passing `grammar=state` enables [constrained selection and
state transitions](usage.md#grammar-selection). This adapter stops at grammar
termination automatically; the caller still owns EOS and output limits.

<a id="cached-verification"></a>

### Cached verification

```python
with adapter.target_session() as target:
    stream = target.verify_prefix(committed_prefix, proposals)
```

The session cache holds keys and values for the tokens of its last call. Each
call keeps the longest cached prefix of `committed_prefix + proposals`, crops the
rest with `DynamicCache.crop`, and feeds at least the final prefix token with the
proposals. After a correction at proposal `j`, the cache holds `L + len(proposals)`
positions. The next round keeps `L + j`, crops the rejected suffix, and feeds the
correction first. After full acceptance nothing is cropped; the bonus is fed
first. Event `i` reads local row `L - 1 - c + i`, where `c` is the reused length.
A prefix that shares nothing with the cache starts a fresh cache. Session
`next_token` follows the same rule.

The cache is a `DynamicCache` built from the model config before the first forward.
Rollback supports full-attention layers only. Linear-attention layers such as
Qwen3.5 Gated DeltaNet, and sliding-window layers, cannot be cropped exactly. They
fail with `ValueError` when the session opens; use `target_cache=False` for them.
Draft caching still works for these models, since draft caches are append-only.
A forward that fails or returns a different cache drops the session cache;
the next call recomputes from the full prefix. Closing the session releases it,
and a closed session rejects further calls.

### Validation

Controlled tests cover empty drafts, every mismatch position, full acceptance,
input ownership, callback failures, and stopping before a bonus. Exhaustive small
vocabularies compare verified continuations against sequential greedy decoding.
Tiny causal-model tests repeat this comparison for multiple prefix lengths and
random seeds, while asserting that verification performs exactly one forward.
Position-specific fixtures detect shifted rows and attempts to select rejected
or unused NaN rows. Real models may differ numerically across forward shapes;
those differences must be investigated before making equivalence claims.

<a id="grammar-selection"></a>

## Grammar-constrained target selection

The target model selects a maximum only among tokens valid in the current grammar
state. Grammar state describes the generated suffix, not the prompt. A fresh
state is required for each independent request; to continue an existing suffix,
replay its generated token IDs before calling the decoder.

### State and selection

`GrammarState` exposes a vocabulary size, termination status, allowed token IDs,
and token acceptance. Reading allowed IDs must not advance state. A rejected
acceptance must return `False` without advancing. The model and grammar must agree
on token-ID meanings; equal vocabulary sizes alone do not establish compatibility.

The Transformers adapter performs these operations for every consumed choice:

1. Read the valid IDs at the current committed grammar state.
2. Select their highest-scoring token, breaking exact ties by lowest token ID.
3. Accept that token into the grammar, failing explicitly if it is rejected.
4. Return or yield the token only after the transition succeeds.

Draft proposals never advance the target grammar. At a mismatch, the target
correction advances state once; rejected proposals and their tail do not. This
requires no rollback because target state never visits the rejected path.
[Constrained drafting](usage.md#constrained-drafting) uses a separate matcher fork;
discarding that fork restores the round boundary without rewinding the target.

`masked_argmax` gathers valid IDs instead of setting invalid scores to negative
infinity. That preserves validity even when all valid scores are negative
infinity. It rejects empty valid sets, malformed IDs, non-floating scores, and
NaN anywhere in the score row. It leaves logits unchanged. This reference sorts
and gathers explicit IDs and synchronizes the winning index to the host; it is
not an optimized packed-mask GPU kernel.

### XGrammar adapter

Install the `models` and `grammar` extras as described in
[development](development.md). Given a compiled XGrammar grammar, an initialized
`TransformersGreedyModel` named `target`, and tokenized `prompt_ids`:

```python
from bitmask_speculation.decoding import greedy_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState

state = XGrammarState(compiled_grammar)
result = greedy_decode(
    target, prompt_ids, max_new_tokens=128,
    eos_token_ids=eos_ids, grammar=state,
)
```

Compile with the target tokenizer's token mapping and model vocabulary size.
The adapter owns a private matcher and CPU mask workspace. It expands only IDs
below the vocabulary size, handling signed int32 words and excluding padding.
The all-valid fast path uses the matcher's documented `need_apply=False` result.
The base package imports without XGrammar or PyTorch; optional libraries are
loaded when the adapter is constructed or tensor selection is called.

By default XGrammar requires a stop token after completing the grammar body.
A complete body and a terminated matcher are therefore different states.
`terminate_without_stop_token=True` instead ends at the body; choose the mode
explicitly to match the request contract. See the upstream
[matcher API](https://xgrammar.mlc.ai/docs/latest/api/python/grammar_matcher.html).
Set the decoder's EOS IDs consistently with the grammar's stop tokens.

### Verification and termination boundaries

```python
stream = target.verify_prefix(committed_prefix, proposals, grammar=state)
```

The state is advanced before each event is yielded. Do not accept the yielded
token again. Stop consumption at the request's EOS or output limit; the adapter
stops automatically at grammar termination, including after an accepted proposal
that would otherwise trigger a bonus. A terminal state at entry causes no forward
pass in verification; direct `next_token` raises `GrammarTerminatedError`.

The constrained greedy reference preserves the final content token and returns
`grammar` when its transition terminates the matcher. A selected EOS takes
precedence and is excluded from content. Zero output budget returns `length`
without reading or advancing the matcher. Vocabulary compatibility is still
validated. The [speculative request loop](usage.md#speculative-decoding) enforces these
boundaries across rounds. [Proposal accounting](design.md#proposal-accounting) retains
the discarded draft tail in proposed counts without counting it as verified.

A partially consumed iterator leaves state at exactly its yielded prefix. Keep
exclusive ownership of that state while the iterator is active; do not interleave
iterators, mutate it externally, or share it across requests. Close an abandoned
iterator to release its logits. Exceptions do not roll back earlier successful
transitions: discard the failed request or restore an explicit checkpoint.

<a id="constrained-drafting"></a>

## Constrained draft proposals

C3 uses the same grammar for draft selection and target verification, with a
fixed proposal budget. Enable it explicitly:

```python
from bitmask_speculation.decoding import speculative_decode
from bitmask_speculation.grammar.xgrammar import XGrammarState

result = speculative_decode(
    target, draft, prompt_ids,
    max_new_tokens=128,
    draft_budget=5,
    eos_token_ids=eos_ids,
    grammar=XGrammarState(compiled_grammar),
    constrain_draft=True,
)
```

Both models must share token-ID meanings and special-token conventions. The
constrained draft adapter must select an allowed token and advance its supplied
grammar exactly once before returning it, just like the target adapter. The
Transformers adapter already implements this contract. Setting the flag without
a forkable grammar fails explicitly, including for a zero output budget.

### Fork and discard

`ForkableGrammarState` extends `GrammarState` with `fork()`. A fork begins at the
same generated suffix, with identical allowed IDs and termination status. Its
transitions and mutable mask workspace must be independent of its parent.
XGrammar shares the compiled grammar while copying matcher state; the adapter
allocates a separate mask workspace. It shares only the read-only token-ID index.

Each round follows this sequence:

1. Fork the grammar at the committed prefix, including any suffix consumed before
   this request. Never replay prompt tokens into the grammar.
2. Select draft tokens against the fork, advancing it once per selection. Stop
   at the proposal budget, remaining output budget, draft EOS or branch termination.
3. Discard the fork before target verification. This is the rollback operation:
   the parent has never advanced along the proposed path.
4. Verify against the parent. Accepted tokens and a correction or bonus advance
   committed state only as their events are consumed.
5. Start the next round from the corrected prefix and a fresh fork of its state.

A rejected tail cannot contaminate the next round. Full acceptance also discards
the fork; the target advances the parent once for every consumed choice. No
commit or merge of the draft branch is needed. The round-local draft cache is
also discarded, on normal exit, EOS, grammar termination or an exception. The
request-scoped target cache is cropped at the next round boundary, not merged.

### EOS, limits and errors

The draft adapter may consume EOS into its branch. That EOS is excluded from
proposals and counters, and branch termination cannot terminate the request.
An empty proposal still asks the target for a fallback token. Only the target's
choices control request termination. A final content token that terminates the
draft grammar remains a proposal for target verification.

Zero output budget calls neither models nor `fork()`. An already terminated
parent returns before drafting. Otherwise the proposal budget is capped by the
remaining content allowance. Target EOS, grammar termination and output limits
retain the precedence defined by the [request API](usage.md#speculative-decoding).

A draft exception propagates with the parent unchanged for that round, including
when the draft advanced its branch before failing. A target exception can leave
the parent advanced through events already consumed; this is not transactional
rollback of the entire request. Custom grammar implementations are responsible
for true fork independence. The decoder rejects a fork returning the parent or
a different vocabulary, but cannot detect hidden shared mutable state.

<a id="speculative-decoding"></a>

## Speculative request decoding

`speculative_decode` combines draft proposals and lazy target verification into
one bounded request. Both models use greedy selection. The optional grammar
constrains the target. `constrain_draft=True` also applies that grammar to draft
selection through a fresh fork at each round boundary.

Given already initialized target and draft model adapters:

```python
from bitmask_speculation.decoding import speculative_decode

result = speculative_decode(
    target, draft, prompt_ids,
    max_new_tokens=128,
    draft_budget=5,
    eos_token_ids=eos_ids,
    grammar=state,  # Omit for unconstrained decoding.
)
```

The target implements `VerificationTarget`, including lazy prefix verification.
The draft implements `GreedyModel`, or `ConstrainedGreedyModel` when constrained
proposals are enabled. `TransformersGreedyModel` supports either role.
Both must agree on token-ID meanings, tokenizer behavior, and special-token
conventions. Vocabulary-size equality is checked but does not prove compatibility.

The immutable `SpeculativeResult` contains `generation` (a `GreedyResult`) and
`accounting` (a `ProposalAccounting`). The convenience properties `token_ids`,
`stop_reason`, and `eos_token_id` retain their meanings; prompt and EOS remain
excluded from content. Compare `result.generation` with the target-only reference
when checking decoding equivalence. Comparing two whole speculative results also
compares their counters. See [proposal accounting](design.md#proposal-accounting).

### One round

1. Construct the authoritative prefix from the prompt and emitted content.
2. Draft up to `min(draft_budget, remaining_output_tokens)` tokens. With draft
   constraints, advance only a fork of the committed grammar and stop when that
   branch terminates. Do not add a selected EOS to the proposal sequence. Discard
   the branch before verification, including its speculative EOS transition.
3. Verify the proposals against the target. Empty proposals request one
   target-only fallback choice; a draft EOS never ends the request by itself.
4. Consume target-confirmed choices in order, checking request termination after
   each one. Never request a bonus when the output budget is exhausted.
5. After rejection, discard the proposal tail and begin the next round from the
   corrected committed prefix. After full acceptance, a bonus is allowed only
   if the request still has room and has not terminated.

The Transformers target cache spans every round of one request. Each
verification forward feeds the uncached committed tokens, normally just the
previous correction or bonus, followed by the proposals. The next round crops
cache rows for rejected proposals before feeding the new correction or bonus.
Drafting prefills the committed prefix once per round, then feeds one token per
prediction; that cache is released before verification. Adapters without
`target_session()` or `draft_session()` retain full-prefix calls.

### Termination and validation

A target-selected EOS stops immediately and is stored separately from content.
Otherwise the selected content token is appended. If its grammar transition
terminates the request, the token is preserved and the stop reason is `grammar`.
Otherwise reaching the output limit returns `length`. Thus target EOS takes
precedence over grammar termination, which takes precedence over a simultaneous
length boundary. No extra prediction probes whether EOS would follow the limit.

Zero output budget validates inputs and returns `length` without calling either
model or reading/advancing matcher state. For positive budgets, an already
terminated grammar returns `grammar` before drafting. Prompt EOS tokens remain
ordinary context. Multiple explicit EOS IDs are supported.

`draft_budget` must be a positive integer; the output limit must be a nonnegative
integer. Empty prompts, invalid IDs, incompatible vocabulary sizes, booleans in
integer fields, and invalid model outputs fail explicitly. A verifier that emits
no event without an established terminal condition fails instead of looping.

The verifier must be lazy: selecting a token and advancing grammar state happen
only when the consumer requests that event. Its iterator is closed on early
return or failure when it exposes `close()`. Errors propagate rather than being
reported as successful partial output; earlier grammar transitions are not
rolled back automatically. State belongs exclusively to one active request.

### Scope

The reference path handles one request at a time, with a fixed or density-selected draft budget,
optional grammar constraints on the target or both models, and greedy selection.
It has request-level proposal counters, target KV reuse across rounds and draft
KV reuse within each round. Sampling and request batching are not implemented. C4 selects the budget using a
[density-threshold policy](design.md#draft-scheduling). The full prompt plus proposal sequence must
fit the model's context capacity; no implicit truncation is applied.
