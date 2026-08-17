# bitmask-speculation

Packed grammar masks for draft scheduling and token verification in structured
LLM decoding.

Structured generation restricts which tokens a model may emit at each position.
A draft model that ignores those restrictions can propose tokens the target must
reject. Applying the same grammar to drafting narrows the proposal space, but
does not guarantee agreement between the models.

This project investigates two decisions: how many tokens to draft from a given
grammar state, and how to compute masked token selection with less GPU overhead.
The starting point is greedy decoding with one draft model, one target model,
and a shared token vocabulary.

## Approach

- Apply grammar constraints to both draft proposals and target verification.
- Use the fraction of valid vocabulary entries to select a draft budget.
- Evaluate packed-mask reductions and masked argmax as CUDA/Triton kernels.

Mask density is a scheduling heuristic. A small valid-token set need not produce
high acceptance, and computing the signal has a cost. Both the policy and its
overhead need to be measured against fixed-budget speculation.

## Prefix acceptance

If the draft proposes four tokens and the third disagrees with the target, only
the first two are accepted. The target supplies the third token; the fourth
proposal is discarded even if it would otherwise match.

`verify_prefix` implements this ordered comparison. The Transformers adapter
uses one target forward pass over the committed prefix and proposals, then
selects only the rows requested by the consumer. See the
[verification API](docs/prefix-verification.md) for its round-level contract.

The [worked trace](docs/prefix-acceptance.md) follows token alignment, grammar
rollback, the all-accepted bonus token, and the resulting counters.

## Greedy reference

The target-only token loop accepts an explicit prompt, output limit, and EOS IDs.
Its result separates content tokens from the termination reason. An optional
Transformers adapter selects from the last prefix position, optionally restricted
by request-owned grammar state.

See the [reference API](docs/reference-decoding.md) for usage and edge cases.
The adapter recomputes the full prefix; it establishes token behavior before
cache management is introduced. Both greedy decoding and prefix verification
share the [constrained selection contract](docs/grammar-selection.md).

## Speculative requests

`speculative_decode` runs unconstrained drafting followed by target verification,
with optional target-side grammar state. It caps each proposal at the remaining
output budget and stops only on a target-confirmed EOS, grammar termination, or
the length limit. An empty draft still takes a target step.

The [request API](docs/speculative-decoding.md) defines termination precedence,
state ownership, and the current full-prefix implementation's limits.

## Evaluation

The reference is target-only greedy decoding under the same grammar. Speculative
variants must preserve its token sequence before throughput comparisons are
meaningful. Workloads should cover enums, nested objects, and free-text fields,
including cases where speculation is slower.

The [design document](docs/design.md) defines the decoding contract, experiment
matrix, and validation requirements. Measured results will accompany validated
implementations.

## Development

The Python package uses a `src` layout and a Python 3.12 development environment.
See [development setup](docs/development.md) for an editable installation and
checks against a built wheel. GPU backends are introduced separately from the
base package.

## License

[MIT](LICENSE).
