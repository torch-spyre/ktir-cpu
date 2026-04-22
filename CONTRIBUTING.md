# Contributing to ktir_cpu

Thank you for your interest in contributing! This project is an experimental KTIR subset interpreter and validator for the IBM Spyre accelerator.

## Getting Started

```bash
uv sync --extra dev
```

## Running Tests

```bash
uv run pytest -v
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
