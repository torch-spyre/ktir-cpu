# KTIR emulator (Rust)

A Rust implementation of the KTIR execution stack (RFC 0682): a **Spyre emulator**
that parses KTIR/MLIR, interprets the `ktdp` dialect against an emulated machine
(HBM + per-core LX scratchpads), and offloads heavy math to Apple-Silicon
accelerators (Metal/NAX GPU, AMX via Accelerate).

## Getting started

```sh
# Build + run the full test suite (macOS: Accelerate/AMX + Metal are auto-on).
cargo test

# The strongest check — real-model end-to-end (Metal backend is auto-on on macOS):
cargo test -p ktir-emulator --test e2e_real_forward -- --test-threads=1

# Lint / format
cargo clippy --all-targets
cargo fmt
```

First run of the e2e tests fetches the model weights from public HuggingFace
(`HuggingFaceTB/SmolLM2-135M`, `unsloth/Llama-3.2-1B-Instruct`; no `HF_TOKEN`) and
caches them in `~/.cache/huggingface`.

> If `cargo` can't find `rustc` (broken rustup shims), put the toolchain on PATH:
> `export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"`

## Workspace layout

```
ktir-core         The KTIR language + shared data types (no machine model):
                  ir (AST) · parser · parser_ast · dtypes · affine · memref ·
                  codec (f16/bf16) · tile · fxhash
ktir-optimizer    IR→IR passes: function fusion, flash-attention cap-tiling,
                  head-parallel attention re-roll. Depends only on ktir-core.
ktir-emulator     The execution layer (the emulator). Depends on core + optimizer.
```

### `ktir-emulator` modules

| Module | Role |
|---|---|
| `interpreter` + `dialects/` | the ktdp/arith/math/linalg/scf/tensor op handlers (the eval loop) |
| `machine_state/` | the emulated Spyre machine: `memory` (HBM + per-core 2 MB LX hierarchy) and per-core `context` (SSA values, scope stack, LX accounting, grid id / comm) |
| `ops_memory` | the `ktdp.load`/`ktdp.store` data path (reads/writes against the machine) |
| `comm` / `comm_sched` | cross-core comm seam + the scheduled execution driver (GPU offload dispatch) |
| `segmented` / `resident` | the fused/serving execution paths (`execute_segmented`, the resident weight-cached session) |
| `program` | turnkey entrypoints (`program::execute` / `Session`) |
| `blas` | the CPU/cblas fallback GEMM (naive reference + Accelerate/OpenBLAS/MKL) |
| `metal` *(cfg(metal))* | the Apple-Silicon accelerator backend: NAX/simdgroup GPU GEMM, map-window fusion, the matmul-loop recognizer + offload, AMX transpose-B |
| `latency` | the Spyre cost model |

`cfg(metal)` is emitted by `build.rs` on macOS (or with `--features metal`); off it,
everything runs on the portable interpreter + `blas` path.

## Testing story

- **Parity tests** (`tests/port_*.rs`) check the Rust port against the Python
  `ktir_cpu` reference, module by module (parser, dialects, interpreter, latency,
  distributed views, …). These run in default `cargo test`.
- **Differential conformance** (`rust/crates/ktir-emulator/tests/equiv/diff_py_vs_rust.py`, under rust/)
  is the head-to-head proof of port faithfulness: it generates seeded random
  inputs, runs the **same** inputs through **both** the Python `KTIRInterpreter`
  and the Rust `execute_function`, and diffs the outputs (NOT both-vs-a-hardcoded
  answer-key, like the `port_*.rs` tests — those never run Python). On the CPU/AMX
  interpreter every conforming program is **bit-identical** (max-abs 0); and the
  **Metal fast path** (NAX/simdgroup GEMM + fused map) is covered too — the same
  programs are forced through the resident/GPU executor (`KTIR_DIFF_RESIDENT=1` /
  `KTIR_DIFF_GPU=1`) and diffed within a principled bf16/f16 band, each asserting an
  offload proof so a silent CPU fallback FAILS (see `tests/metal_conformance.rs`).
  See the [Python ↔ Rust conformance](PERFORMANCE.md#python--rust-conformance) table.
  The `rust-conformance.yml` workflow runs it on every relevant push.
- **Real-model e2e** (`tests/e2e_real_forward.rs`) — the headline test. It runs a
  real forward of SmolLM2-135M and Llama-3.2-1B (prefill *and* decode) through the
  production path (`execute_segmented`) and asserts the next-token argmax matches a
  **real `transformers` golden**, within a loose f16 band. Hermetic: KTIR programs
  are vendored in `tests/fixtures/<model>/` (weights-free, `matmul_transpose_b` →
  HF weights bind verbatim `[out,in]`, zero transpose); weights come from public HF;
  goldens + inputs are vendored as **f16 + gzip-9**. See
  [`tests/fixtures/README.md`](crates/ktir-emulator/tests/fixtures/README.md) for the
  program origin (scratchy `SCRATCHY_KTIR_DUMP`), re-vendoring, and how to regenerate
  the goldens (`gen_golden.py`, run in an ephemeral `uv` env).
- **Fusion / attention** goldens (`fuse_run_*`, `flash_attn_*`, `head_rewrite_*`)
  cover the optimizer passes against the production execution.
- Prefill (m>1) e2e gates on `cfg(metal)` (the fused [1,1] segments need the GPU/AMX
  offload for full-M reconstruction). `cfg(metal)` is auto-on on macOS, so plain
  `cargo test` runs prefill there; on non-mac add `--features metal`.

## BLAS / acceleration (the `blas` module)

`linalg` matmul runs through cblas where it's free, else a naive Rust loop (the
`cblas_sgemm` call is identical across providers — only the linked library differs):

- **macOS:** Apple **Accelerate** (AMX-backed) is used **by default, no feature
  flag** — it ships with the OS.
- **Linux / other:** naive loop by default; name a provider for hardware BLAS:

```sh
cargo test --features openblas-system   # system OpenBLAS (libopenblas-dev) [light]
cargo test --features mkl               # Intel MKL (x86_64)
cargo test --features blis              # portable BLIS (good on AMD)
cargo test --features openblas          # builds OpenBLAS from source (needs gcc/gfortran)
```

The `blas_matches_naive` test gates that the active backend agrees with the naive
oracle (runs by default on macOS).
