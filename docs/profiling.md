# Kernel profiling

The installed `profile` command prepares seeded inputs, loads the selected
backend, warms it, and compares all outputs with the PyTorch reference before
opening a CUDA profiler range. It repeats unchecked reductions on one explicit
stream, drains that stream, closes the range, and checks outputs again. Input
creation, compilation and reference comparisons are outside the range.

```bash
python -m bitmask_speculation.benchmarks profile \
  --backend triton --rows 6 --vocab 151936 --density .01 \
  --iterations 10 --output /tmp/bitmask-profile-check
```

The command runs without an attached profiler and writes configuration,
environment and correctness results. It does not infer performance counters or
latency from that execution. Use a new output directory for each invocation.

## Nsight Systems

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/bitmask-systems \
  python -m bitmask_speculation.benchmarks profile \
  --backend triton --rows 6 --vocab 151936 --density .01 \
  --iterations 10 --output /tmp/bitmask-systems-inputs
```

The CUDA profiler API bounds capture; the NVTX range names the selected backend.
Use the timeline to inspect launch gaps, synchronization and kernel sequencing.
See the [Nsight Systems capture documentation](https://docs.nvidia.com/nsight-systems/UserGuide/).

## Nsight Compute

```bash
ncu --profile-from-start off --set basic --launch-count 2 \
  --export /tmp/bitmask-compute \
  python -m bitmask_speculation.benchmarks profile \
  --backend triton --rows 6 --vocab 151936 --density .01 \
  --iterations 1 --output /tmp/bitmask-compute-inputs
```

Triton emits a partial reduction and a final reduction per call. CUDA's batched
comparison invokes per-row primitives and stacking; adjust `--launch-count` when
profiling that path. Profiling replay and instrumentation change execution cost;
use the uninstrumented benchmark runner for latency comparisons.

Counter access is a host permission independent of CUDA execution. A counter
permission failure is not a successful profile, and querying metric names does
not establish counter access. The command does not modify driver permissions.
See the [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
and [counter access requirements](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
