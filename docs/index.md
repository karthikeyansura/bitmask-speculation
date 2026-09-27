# Documentation

Start with the [engine API](engine.md), [generation CLI](cli.md) or
[local-model example](../examples/quickstart.py).

| Area | Documents |
| --- | --- |
| Design | [Contracts](design.md), [worked prefix trace](prefix-acceptance.md) |
| Reference decoding | [Greedy loop](reference-decoding.md), [prefix verification](prefix-verification.md), [request termination](speculative-decoding.md) |
| Grammar | [Target selection](grammar-selection.md), [draft constraints](constrained-drafting.md), [mask density](mask-density.md) |
| Scheduling | [Density policy](draft-scheduling.md), [proposal accounting](proposal-accounting.md) |
| Native code | [CUDA primitives](cuda-primitives.md), [CUDA validation procedure](cuda-validation.md) |
| Evaluation | [Workloads](baseline-workloads.md), [budget sweeps](budget-sweeps.md), [validation records](validation.md) |
| Development | [Environment, packaging and tests](development.md) |

The engine shares the existing decoding loops. Its model-loading facade checks
compatible tokenizer definitions and keeps grammar state request-local. Native
kernel dispatch is a separate capability; importing the package does not compile
CUDA code or load model weights.

- [Triton masked reductions](triton-reductions.md): signed words, exact counts and checked selection.
