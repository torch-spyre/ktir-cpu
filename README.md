# ktir_cpu

> **Experimental** — This is a research prototype. It implements a subset of the KTIR specification and is not a production-ready tool. See [Supported Subset](#supported-subset) for details.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

CPU validation interpreter for KTIR (Kernel Tile IR) — the MLIR dialect targeting IBM's Spyre accelerator. Parses KTDP MLIR kernels, executes them with NumPy on a simulated multi-core grid, and optionally estimates execution latency.

## Why

ktir_cpu validates KTIR kernels entirely on CPU — no Spyre hardware required. It catches correctness bugs early by executing kernels with NumPy on a simulated multi-core grid, and provides roofline latency estimates that identify memory, compute, and communication bottlenecks without a hardware run.

More broadly, ktir_cpu serves as an **environment and reward model** for AI-driven compiler development. A frontend compiler pipeline (e.g., Inductor → Triton → KTIR) can emit candidate kernels and use ktir_cpu to score them: correctness via numerical output comparison, and performance via the latency estimator's bottleneck analysis. This makes ktir_cpu a natural feedback loop for agentic compiler workflows — an LLM-based agent can generate or modify compiler passes, run the resulting kernels through ktir_cpu, and use the correctness and performance signals to iteratively improve code generation without access to physical hardware.

This works in practice because ktir_cpu is **fast**, **actionable**, and **deterministic**. It runs in seconds on any laptop, so an agent can evaluate hundreds of candidate kernels without waiting for hardware queues. The latency report breaks down into memory, compute, and communication components with a bottleneck classification, giving an agent direction on *what* to optimize — not just *whether* to optimize. And results are reproducible across runs, which matters when using output as a reward signal — noisy rewards make search harder.

## Setup

```bash
uv sync
```

### MLIR frontend bindings (optional)

The `tests/mlir_frontend/` tests use `mlir_ktdp` from
[ktir-mlir-frontend](https://github.com/torch-spyre/ktir-mlir-frontend).
Until a PyPI release is available, build from source:

**Prerequisites:** CMake ≥ 3.20, Ninja, C++17 compiler.

```bash
KTIR_MLIR_FRONTEND_COMMIT=55a8a7eaf3a3344d95c2e8793da0bc220c62d9c0
LLVM_HASH=e9846648fd6183ee6d8cbdb4502213fcf902a211

# Fetch setup_mlir.py from the pinned commit
curl -fsSL \
  "https://raw.githubusercontent.com/fabianlim/ktir-mlir-frontend/$KTIR_MLIR_FRONTEND_COMMIT/scripts/setup_mlir.py" \
  -o /tmp/setup_mlir.py

# Resolve MLIR — --wheel activates the mlir_wheel fallback (no GitHub token needed);
# --hash records the targeted LLVM build (drop --wheel once artifacts are public)
MLIR_DIR=$(uv run python /tmp/setup_mlir.py --wheel --hash "$LLVM_HASH")

# Build and install
CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv sync --extra mlir-frontend
```

Once [torch-spyre/ktir-mlir-frontend#12](https://github.com/torch-spyre/ktir-mlir-frontend/issues/12)
is resolved and wheels are published, this will simplify to:

```bash
uv sync --extra mlir-frontend
```

## Quick start

```python
from ktir_cpu import KTIRInterpreter, HardwareConfig
import numpy as np

interp = KTIRInterpreter()
interp.load("examples/triton-ktir/vector_add_ktir.mlir")

# Query expected tensor shapes from the parsed MLIR
sizes = interp.tensor_input_output_sizes("add_kernel")
# {'x_ptr': {'shape': (1024,), 'dtype': 'f16'}, 'y_ptr': ..., 'output_ptr': ...}

n = sizes["x_ptr"]["shape"][0]
x = np.random.randn(n).astype(np.float16)
y = np.random.randn(n).astype(np.float16)
out = np.zeros(n, dtype=np.float16)

outputs = interp.execute_function("add_kernel", x_ptr=x, y_ptr=y, output_ptr=out)
print(outputs["output_ptr"])  # x + y
```

## Latency estimation

Pass a `HardwareConfig` to enable cycle-approximate latency estimation:

```python
interp = KTIRInterpreter(latency_config=HardwareConfig())
interp.load("examples/triton-ktir/vector_add_ktir.mlir")
interp.execute_function("add_kernel", x_ptr=x, y_ptr=y, output_ptr=out)

report = interp.get_latency_report()
print(report)
print(report.bottleneck)      # "memory", "compute", or "comm"
print(report.kernel_time_us)  # estimated wall time in microseconds
```

See [docs/latency.md](docs/latency.md) for the full cycle model, hardware parameters, systolic array model, and a worked vector_add example.

## Architecture

### Loading and parsing

`interp.load(source)` accepts either a file path or inline MLIR text. The parser produces an `IRModule` containing `IRFunction`s, each with a list of `Operation` nodes and a grid shape inferred from the MLIR.

`tensor_input_output_sizes(func_name)` queries the parsed IR for tensor argument shapes and dtypes, so callers can allocate inputs without hardcoding sizes.

**Assumptions**:
- Single core if `grid` attribute is absent.
- `construct_access_tile` evaluates `base_map` to compute the sub-tile's base offset; `access_tile_set` and `access_tile_order` drive coordinate-set iteration in `ktdp.load` / `ktdp.store` when present.
- `ktdp.load` / `ktdp.store` use a gather/scatter path when `access_tile_set` is specified. This involves building a list of flat element indices and doing a numpy fancy-index read or read-modify-write — which is slower than a contiguous copy. Full rectangular sets are normalised to `None` at parse time so they take the direct path; only genuinely non-rectangular tiles pay the gather/scatter cost.
- `execute_function` always allocates array arguments in HBM. LX is used only for intermediate `Tile` values produced during execution.
- If `sizes:` tokens are SSA names (e.g. `%Nb`) rather than integer literals, the concrete dimensions are taken from the `memref<NxMxdtype>` result type instead.

### Execution model

`execute_function` allocates HBM, builds a `CoreContext` per core (which holds the core's LX scratchpad, HBM reference, and SSA value map), and runs each operation through registry-based dispatch:

```
load(source)  →  IRModule { IRFunction { [Operation, ...], grid } }
                                │
execute_function("fn", **inputs)
    │
    ├─ for each core in grid:
    │      CoreContext(core_id, hbm, lx, values={})
    │          │
    │          ├─ _execute_operation(op, context, env)
    │          │      handler = dispatch(op.op_type)   # registry lookup
    │          │      result = handler(op, context, env)
    │          │      context.set_value(op.result, result)
    │          │
    │          └─ ... next op ...
    │
    └─ collect output tensors from HBM
```

### Memory hierarchy

Two memory spaces are simulated:

- **HBM** (128 GB, shared) — holds host-provided input/output tensors. All function arguments are HBM addresses.
- **LX** (2 MB per core) — on-chip scratchpad holding all live SSA tensor values. Every `Tile` produced by `ktdp.load` or compute ops resides in LX.

`MemoryOps.load` and `.store` inspect `TileRef.memory_space` to determine whether data crosses the HBM-LX boundary (DMA) or stays on-chip. LX lifetime is region-scoped: when an MLIR region exits (`pop_scope`), its SSA values are discarded and LX is freed.

**Layer separation — flat access vs. stride-aware access:**

`memory.py` (`HBMSimulator`, `LXScratchpad`) is a pure flat byte-addressed store. Its `read(ptr, n_elements, dtype)` and `write(ptr, data)` methods know nothing about shapes, strides, or coordinate sets — they only translate byte addresses to array indices via `_find_allocation`. All stride and coordinate logic lives exclusively in `MemoryOps` (`ops/memory_ops.py`):

- `tile_access` evaluates `base_map` with the access indices to compute `base_coords`, then multiplies by `parent_ref.strides` to get the byte offset for `base_ptr`.
- `_gather_indices` converts a list of local coordinate tuples (plus the tile's strides) into flat element offsets from `base_ptr`.
- `load` / `store` call `mem.read(base_ptr, span)` once to cover the entire element footprint, then use numpy fancy indexing to gather or scatter the relevant elements.

This keeps `memory.py` simple and testable in isolation, and means stride-related bugs are always in `MemoryOps`, never in the memory simulator.

### Dialect handler registry

Dialect modules (`dialects/arith_ops.py`, `dialects/ktdp_ops.py`, etc.) register handlers at import time using the `@register()` decorator from `dialects/registry.py`. Each registration stores:

- the execution handler in `_REGISTRY[op_name]`
- the latency category in `_LATENCY_CATEGORIES[op_name]`

This keeps operation knowledge (behavior + cost classification) co-located in one place per op.

### Key types

| Type | Module | Purpose |
|------|--------|---------|
| `IRModule` / `IRFunction` | `ir_types` | Parsed MLIR structure |
| `Operation` | `ir_types` | Single IR operation node |
| `Tile` | `ir_types` | Data value backed by a NumPy array |
| `TileRef` | `ir_types` | Memory layout descriptor (memref equivalent) |
| `AccessTile` | `ir_types` | Sub-region reference into a `TileRef` |
| `CoreContext` | `grid` | Per-core state: core ID, LX scratchpad, HBM reference, SSA scope stack |
| `ExecutionEnv` | `dialects/registry` | Shared resources passed to handlers (grid executor, ring network, `execute_region`) |
| `LatencyCategory` | `latency` | `StrEnum` classifying op cost (`ZERO`, `MEMORY`, `COMPUTE_FLOAT`, etc.) |

## Supported Subset

This interpreter covers a **subset** of KTIR ([RFC 0682](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)). The following are supported:

- Embarrassingly parallel kernels (no inter-core communication required)
- `ktdp.load` / `ktdp.store` with rectangular-slice semantics
- `ktdp.construct_access_tile` (rectangular tiles only)
- Arithmetic, math, and linalg dialect ops (see `ktir_cpu/dialects/`)
- `scf.for` / `scf.if` control flow
- Multi-core grid execution
- Cycle-approximate latency estimation

**Not yet supported or unreliable:**

- `ktdp.construct_distributed_memory_view` — not implemented
- `ktdp.construct_indirect_access_tile` — not implemented
- `ktdp.transfer` / `ktdp.reduce` (communication ops) — present but **unreliable**: the multi-round communication model re-executes the entire function per round, causing incorrect latency accumulation and potential correctness issues with cyclic communication patterns. See `docs/gap_analysis.md` for details.
- `tensor.extract_slice` / `memref.subview`

## RFC conformance

The interpreter targets RFC 0682 but does not yet implement all KTDP ops. Known gaps are tracked as `xfail(strict=True)` tests in `tests/test_spec_gaps.py` — an unexpected pass (XPASS) signals that a gap has been closed and the marker should be promoted to a regular test. Full gap analysis: [`docs/gap_analysis.md`](docs/gap_analysis.md).

```bash
uv run pytest -m spec_gap        # run only gap tests
uv run pytest -m "not spec_gap"  # skip gap tests
```

RFC example files live in `examples/rfc/`.

## Tests

```bash
uv run pytest tests/ -v
```

### Update the lockfile

After adding or changing dependencies in `pyproject.toml`:

```bash
uv lock
git add uv.lock
```
