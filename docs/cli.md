# Generation CLI

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
Use `--device cuda --dtype bfloat16` on a compatible NVIDIA host. CPU/float32 is the
default. The command never starts a cloud instance or silently changes device.

`--schema-file schema.json` and `--schema-json '{...}'` are mutually exclusive.
Constrained modes require a schema; unconstrained modes reject one. Schemas are
explicit JSON documents and external references are rejected. Target-only modes
reject draft options. Fixed `--draft-budget` applies to C1–C3; C4 instead accepts
`--max-draft-budget` and uses the default density thresholds.

The default stdout is one JSON object containing text, content token IDs, stop
reason, selected EOS, proposal accounting and schedule. `--output-format text`
prints only generated content. Errors go to stderr. Exit code 2 indicates command
usage errors; exit code 1 indicates a load, schema or generation error. A successful
length-limited request still exits 0; inspect `stop_reason` and validate the output
before treating a structured document as complete.

No latency or throughput number is printed. The benchmark harness must define
warmup, synchronization and timing boundaries separately. The native kernels are
not yet connected to generation. See [engine contracts](engine.md) and the
[Python example](../examples/quickstart.py).
