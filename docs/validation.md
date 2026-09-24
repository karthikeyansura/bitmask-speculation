# Validation artifacts

Benchmark output contains the configuration, source hashes, model and tokenizer
identities, hardware and dependency versions, raw trials, and a derived summary.
Each trial carries token IDs, termination, schema validity, exact-reference
agreement, proposal counts and elapsed time. Failed trials remain in the output.

## Device coverage

Single-device tests cover kernel correctness, streams, output ownership, seeded
sampling and memory/synchronization checks. Device-guard tests require two visible
CUDA devices in one process. They check input-device selection, restoration of
the caller's device and rejection of mismatched-device tensors. These tests do
not exercise distributed inference or multi-GPU throughput.
