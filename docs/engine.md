# Request engine

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

## Text and model loading

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

This engine uses the existing full-prefix reference selection path. It does not
invoke the standalone CUDA primitives.
