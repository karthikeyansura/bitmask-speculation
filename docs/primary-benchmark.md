# Fixed and adaptive speculation on six JSON schemas

The [primary manifest](../benchmarks/primary.json) fixes the raw prompts, schema
documents, case order and output limits. The target is Qwen3.5-4B at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The draft is Qwen3.5-0.8B at
`2fc06364715b967f1860aea9cf38778875588b17`. Use FP16, one request per batch,
raw prompt encoding with special tokens, round-local draft cache and uncached
target verification. The six cases form a separate series from the Qwen3 BF16
chat workloads.

The constrained target supplies the token reference. C2 masks verification,
C3 also masks drafting, and both use five proposals per round. C4 samples the
committed mask before drafting. Densities below 0.005 request eight proposals,
densities below 0.02 request four, and all other densities request one. The
effective budget is capped by the remaining output length. VeloSpec requests
eight when the matcher can skip mask application. The two implementations can
therefore follow different schedules even on the same prefix. Preserve schema
field order in the source documents. The migration canonicalizes schema keys
before compilation, so compare output IDs across engines before interpreting
a cross-engine latency difference.

Report C4/C3 and each variant against the constrained target using total
paired warm decode time over identical outputs. Count acceptance as accepted
over verified proposals and yield as accepted over all proposals. Report
content tokens, stop reason, schema validity and failed trials. One mismatch
invalidates the cell's ratio. Inspect both paths' logits at the first divergent
selection. FP16 and BF16 argmax can change with the forward shape.
[Thinking Machines, *Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

The primary table requires five or more paired repetitions per case. Keep
cold process and compilation timings separate. The reduction sweep reports
microseconds per batch at 1, 6, 8 and 12 rows, with vocabulary width, mask
density and timing boundary alongside each measurement.
