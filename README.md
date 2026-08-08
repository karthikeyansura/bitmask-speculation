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

The [worked trace](docs/prefix-acceptance.md) follows token alignment, grammar
rollback, the all-accepted bonus token, and the resulting counters.

## Evaluation

The reference is target-only greedy decoding under the same grammar. Speculative
variants must preserve its token sequence before throughput comparisons are
meaningful. Workloads should cover enums, nested objects, and free-text fields,
including cases where speculation is slower.

The [design document](docs/design.md) defines the decoding contract, experiment
matrix, and validation requirements. Installation instructions and measured
results will accompany validated implementations.

## License

MIT.
