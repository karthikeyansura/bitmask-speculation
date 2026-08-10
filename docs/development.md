# Development

Use Python 3.12 for the development environment. The distribution is named
`bitmask-speculation`; Python imports use `bitmask_speculation` because a hyphen
cannot appear in a Python identifier.

## Local installation

From the repository root, with uv installed:

```bash
uv sync --locked
uv run --locked python -c "import bitmask_speculation as bs; print(bs.__version__)"
```

The lockfile records the Python development dependencies. `uv sync` creates an
editable installation in `.venv` and installs the development group by default.
It leaves the system Python environment alone. The current package exposes
version metadata; the decoding API is introduced with the implementation.

The base package has no runtime dependencies. Optional dependencies are
installed through the extras listed in `pyproject.toml`.

## Package boundaries

```text
pyproject.toml                 package metadata and build selection
uv.lock                       resolved development dependencies
.python-version               development interpreter selection
src/bitmask_speculation/       installed Python package
docs/                         design and development documentation
```

The `src` layout keeps imports separate from the working directory. Install the
package before running code that imports it; do not modify `sys.path` in scripts
to make imports succeed.

The wheel contains only the selected package and its distribution metadata.
The source distribution explicitly includes that source, documentation, license,
and project configuration. Unrelated files in the checkout are not automatically
included. Extend the selection when a new component needs additional build inputs.

Native kernel compilation is separate from installing the base package. GPU
backends must be opt-in and must not initialize CUDA during package import.
Backend build and compatibility checks accompany their implementation.

## Build and verify a distribution

```bash
uv run --locked python -m build
```

This builds a source distribution and then builds a wheel from it, checking that
the source distribution carries the inputs needed for a clean build. Build
backends run in isolated environments; their requirements are declared in
`pyproject.toml` separately from the development lockfile.

Validate the wheel from outside the checkout with a fresh environment:

```bash
uv venv --python 3.12 /tmp/bitmask-wheel-check
uv pip install --python /tmp/bitmask-wheel-check/bin/python \
  dist/bitmask_speculation-0.1.0.dev0-py3-none-any.whl
cd /tmp
/tmp/bitmask-wheel-check/bin/python -I -c \
  "import bitmask_speculation as bs; print(bs.__version__, bs.__file__)"
```

Use a fresh directory for each check and the wheel filename produced by the
current build. The import path must resolve inside the verification environment,
not the source checkout. Also inspect archive contents, license metadata, project
links, and runtime requirements before publishing a distribution.

## GPU work

Develop and check packaging locally. Run CUDA compilation, kernel correctness,
and inference benchmarks on Linux with an NVIDIA GPU. Record the GPU model,
driver, toolkit, Python, and library versions with each run. An instance type or
a successful SSH login alone does not establish a validated GPU environment.
