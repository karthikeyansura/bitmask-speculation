# Greedy reference decoding

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

## Output contract

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

## Transformers adapter

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
mode, disables the KV cache, and recomputes the full prefix on every prediction.
Exact ties choose the lowest token ID; NaN next-token logits fail explicitly.
Infinities retain normal argmax ordering, including lowest-ID selection when
all scores are negative infinity. This unmasked adapter has no validity mask;
a constrained decoder must separately detect an empty valid-token set.

No repetition penalties, forced tokens, suppression lists, sampling, or other
generation-config processors are applied. Comparisons with `model.generate`
require those settings to be disabled and matching EOS/output-limit semantics.
The tests use small CPU models with neutral generation settings.

This is a correctness reference. Full-prefix recomputation is intentionally
simple and must not be used as the sole throughput baseline for speculation.
Performance comparisons need a validated cached target-only implementation
under the same grammar and generation settings. Pass `grammar=state` for
[constrained greedy decoding](grammar-selection.md).
The [prefix verifier](prefix-verification.md) uses this greedy behavior to check
one proposal round, without adding a generation loop.
