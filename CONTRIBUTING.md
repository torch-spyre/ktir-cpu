# Contributing to ktir_cpu

Thank you for your interest in contributing! This project is an experimental KTIR subset interpreter and validator for the IBM Spyre accelerator.

## Getting Started

```bash
uv sync --extra dev
```

## MLIR Frontend Bindings (optional)

The `tests/mlir_frontend/` tests require the optional `mlir-frontend` dependency.
See [README.md — MLIR frontend bindings](README.md#mlir-frontend-bindings-optional)
for install instructions; the setup will simplify to a single `uv sync --extra mlir-frontend`
once wheels are available.

## Running Tests

```bash
uv run pytest -v
```

## Adding a kernel

If your change adds a KTIR kernel, see [docs/kernelentry.md](docs/kernelentry.md) —
the path from "here is a kernel" to "the simulator supports it", and what the machine
checks along it versus what only a person can. Start before you write anything, by
pointing the tool at the IR:

```bash
uv run python -m ktir_cpu.kernelentry probe examples/latency/my_kernel.mlir
```

## Pull Requests

1. Fork the repository and create a feature branch.
2. Ensure all tests pass locally.
3. Add tests for new functionality.
4. Keep commits focused — one logical change per commit.
5. All source files must include the Apache 2.0 copyright header.

## Reporting Issues

Please use the GitHub issue tracker. Include MLIR input, expected output, and actual output when reporting bugs.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
