# KTIR Rust Performance

A trackable record of the **Rust** execution layer against the **Python** reference
interpreter (`ktir_cpu`). Everything here is **HERMETIC** — the program is vendored
in `rust/crates/ktir-emulator/tests/fixtures/` and the weights are fetched from
public HuggingFace at run time (no token). There is **NO dependency on the scratchy
`~/.cache/cudaforge` bundle** (the old bundle-based variants were removed; see git
history if you need them).

Two comparos:

- **E2E whole-model** — the production Rust path (`resident::ResidentExecutor`:
  weights marshaled into one persistent HBM **once**, segments chained on-device)
  vs the Python **per-node** reference interpreter, on real models (smollm2-135m,
  llama-3.2-1b; decode + prefill).
- **Per-kernel** — one kernel's `execute_function`, Python vs Rust (AMX/Metal), on
  the in-repo example MLIR.

## Machine

| | |
|---|---|
| Host | Apple **M5** (Mac17,2) — has AMX/Accelerate **and** a Metal GPU (NAX tensor engine) |
| Toolchain | `rustc 1.96.0` (stable-aarch64-apple-darwin) |
| Metal | `cfg(metal)` is auto-enabled on macOS by `build.rs` (no feature flag needed); the `--features metal` flag is for cross builds. **Every macOS build is a Metal build.** |

## How to regenerate

Everything is hermetic — **NO `~/.cache/cudaforge` bundle**. Run benches **one at a
time** (concurrency skews wall-clock; one warm-up pass is excluded inside every
harness; use `--test-threads=1` for the cargo tests).

### E2E whole-model — Python vs Rust

```bash
# --- PYTHON (per-node reference interpreter) -----------------------------------
# Program from tests/fixtures/<model>/, weights from public HF (no token). The
# fast three fit the faithful 2 MB LX; llama-3.2-1b-prefill transiently needs more
# (coarse scope-level LX reclaim) → raise the cap. KTIR_LX_MB gates ALLOCATION
# only, not compute, so the ms/pass is unchanged.
uv run --with huggingface_hub python \
  rust/crates/ktir-emulator/tests/fixtures/bench_e2e_hermetic.py \
  smollm2-135m smollm2-135m-prefill llama-3.2-1b
KTIR_LX_MB=512 ITERS=1 SKIP_WARMUP=1 uv run --with huggingface_hub python \
  rust/crates/ktir-emulator/tests/fixtures/bench_e2e_hermetic.py llama-3.2-1b-prefill
#   env: ITERS (timed passes, default 5), SKIP_WARMUP=1, KTIR_LX_MB (LX size in MB).

# --- RUST (production RESIDENT path) -------------------------------------------
# Same fixtures + HF weights; weights marshaled ONCE, then best-of-N passes.
cd rust && cargo test --release --test e2e_real_forward resident \
  -- --ignored --nocapture --test-threads=1
#   env: ITERS (timed passes, default 5).
```

### Per-kernel — Python vs Rust (AMX / Metal)

```bash
# PYTHON (one kernel's execute_function, on examples/triton-ktir/*.mlir):
uv run python bench_py_vs_rust.py
cd rust
# RUST AMX/CPU interpreter path (matmul tiles route to Accelerate below the NAX gate):
cargo test --release --test bench_py_vs_rust  -- --ignored --nocapture --test-threads=1
# RUST AMX-vs-Metal matmul PRIMITIVE (blas::sgemm_rowmajor vs NaxGemm::run, full GEMM shape):
cargo test --release --test bench_amx_vs_metal -- --ignored --nocapture --test-threads=1
```

---

## Latest snapshot (2026-06-20, branch `rust`)

### E2E whole-model (ms/pass; lower is better) — hermetic Python vs Rust RESIDENT

| Model / mode | Python (per-node) | Rust (RESIDENT) | Speedup |
|---|---:|---:|---:|
| smollm2-135m **decode** (m=1)   |    2,536.3 |  17.6 |  **144×** |
| smollm2-135m **prefill** (m=8)  |   60,397.4 |  45.6 | **1324×** |
| llama-3.2-1b **decode** (m=1)   |   31,777.6 |  71.6 |  **444×** |
| llama-3.2-1b **prefill** (m=32) |  977,688.3 | 169.1 | **5782×** |

**Python** = the `ktir_cpu` per-node reference interpreter (f16 numpy, which has no
BLAS); **Rust** = `resident::ResidentExecutor` (weights marshaled once; full-M GEMMs
on **NAX or AMX**, fused map-window kernels, head-parallel native attention), median of
5, one warm-up excluded. Both run the **identical** vendored KTIR program on the
**identical** public-HF weights and inputs. Prefill is the production **last-token**
default (only the final position's logits — what generation samples); the all-rows
prefill (every position, the comprehensive golden) is llama 186.6 / smollm2 45.8 ms.

The NAX `matmul2d` kernel uses a **vectorized threadgroup loader** — wide 4-element
coalesced device loads + threadgroup stores in place of the per-element `div`/`mod` +
bounds checks in the staging path (the GEMMs were loader-bound, streaming weights at
only ~12 GB/s). It lifted this E2E by **−20% llama decode / −14% llama prefill** vs the
scalar loader (121.6→97.4 / 196.6→169.1), bit-identically; smollm2's tiny m=8/m=1 GEMMs
are in the noise. The NAX kernels are also AOT-precompiled now (embedded metallibs, with
the mandatory `-mmacosx-version-min=26.2` workaround for the SDK-26.5 `matmul2d` half-K
miscompile, JIT fallback retained) — startup-only, so it does not move these
steady-state numbers.

Decode dropped again with **fused m=1 attention**. Per-token attention had been running
as a ~1500-op interpreter storm per layer (the `H` heads unrolled in the node body — the
BLAS GEMVs are near-free; the cost was the `const`/`splat`/`access_tile` plumbing around
them). A structural m=1 attention recognizer + a fused CPU dispatch — per head, QKᵀ GEMV
→ softmax → scores·V GEMV (f32 accumulate, GQA + per-position context mask + 1/√d scale +
context/diagonal split) — collapses it to ~`3·H` BLAS+softmax primitives, golden-faithful
(max-abs identical to the decomposed oracle). Resident decode: llama **97.4→71.6** (1.35×),
smollm2 **33.9→17.6** (2.05×). Because it eliminates ~20k op-dispatches/token of pure CPU
plumbing, it wins **more on slower-CPU devices**: real vLLM llama-3.2-1b decode on an **M1
Max went 5.4→11 tok/s** — clearing the CPU bottleneck lets the M1 Max's ~2.5× memory
bandwidth win the (now-dominant) weight-streaming GEMMs and pull **ahead of the M5**.
`KTIR_NO_FUSE_ATTN` restores the decomposed path.

The Python figures are measured **after** the reference-interpreter perf fixes in
[PR #124](https://github.com/torch-spyre/ktir-cpu/pull/124) (vectorized `ktdp.load`
offset calc, `ravel`-not-`flatten` allocation reads, + O(log n) allocation lookup),
which cut gratuitous overhead **12–20×** so the comparo reflects the interpreter, not
artifacts. Without those fixes the same three measured configs read 36,757.6 /
1,182,455.8 / 395,961.2 ms (i.e. 370× / 2702× / 1514×).

Why the gap is still large: the production program is **finely column-tiled** so each
GEMM tile's `[m, tile_n]` accumulator fits the 2 MB LX — a storm of small
`scf.for`-looped tiles. The Rust resident path runs each tile on NAX/AMX with the
weights already resident in HBM (no per-pass marshal); the Python reference pays genuine
per-node interpreter overhead (numpy tile reads/gathers, HBM stick-count modeling, op
dispatch) plus, at prefill scale, f16 numpy matmul with no BLAS. It is a fair
like-for-like (same KTIR, same weights, same inputs) — the production tiling is simply
hostile to a per-node Python interpreter and ideal for a resident GPU executor.

### Per-kernel (Python vs Rust AMX/Metal) — prior snapshot (2026-06-14, `1149970`)

The first three rows are the Rust **interpreter** path (`execute_function`); on this
M5 it routes matmul tiles to Accelerate (the per-tile 32×128@128×512 blocks are below
the NAX gate `NAX_MIN_BLOCKS=32`) and elementwise/layernorm to the CPU, so they are
the **AMX/CPU** column. The last two rows time the matmul **primitive directly** at
the kernel's logical full-GEMM shape (the apples-to-apples AMX-vs-Metal comparison).

| Kernel | Python | Rust AMX/CPU | speedup vs Py | Rust Metal | AMX→Metal |
|---|---:|---:|---:|---:|---:|
| vector_add (n=4096, f16) | 861 µs | **630.9 µs** | 1.37× | CPU only at this size¹ | — |
| matmul (64×2048×8192, f16) — interpreter (tiled SPMD K-loop) | 11.95 s | **428.5 ms** | 27.9× | n/a (tiles below NAX gate)² | — |
| layernorm (1151×8192, f16) | 41.9 s | **695.1 ms** | 60.3× | CPU only at this size¹ | — |
| **matmul PRIMITIVE 64×2048×8192** (sgemm vs NaxGemm) | — | **3.75 ms** (573 GFLOP/s) | — | **2.93 ms** (733 GFLOP/s) | **1.28×** |
| **matmul PRIMITIVE 512×4096×4096** (prefill-scale) | — | **12.53 ms** (1371 GFLOP/s) | — | **6.55 ms** (2622 GFLOP/s) | **1.91×** |

¹ The map-window GPU offload only fires inside a single-core fused function; the
  standalone elementwise/layernorm kernels run on a multi-core SPMD grid and stay on
  the CPU interpreter. At these sizes the GPU dispatch latency would not pay off.
² The matmul kernel's inner tiles are tiny, so even on Metal they route to Accelerate.
  The 428.5 ms is the *interpreter* path (32-core SPMD × K-tiles, a small Accelerate
  call + marshaling per tile) — ~100× the 3.75 ms raw primitive, i.e. per-tile
  dispatch/marshal dominates this microbench. The primitive rows are the real
  AMX-vs-Metal matmul comparison.

---

## Python ↔ Rust conformance

The headline proof that the Rust port is **faithful** to the Python reference is a
**direct differential** test, not a speed number: the SAME seeded random inputs run
through **both** the Python `ktir_cpu.KTIRInterpreter` AND the Rust
`execute_function`, and the outputs are diffed head-to-head. This is fundamentally
stronger than the `tests/port_*.rs` parity tests — those check Rust against a
**hardcoded answer-key** and never run Python, so they cannot catch a divergence
where Python and Rust *agree with the key but disagree with each other under fuzzed
input*. Here every output is checked Python-byte ⟷ Rust-byte.

Harness: driver `rust/crates/ktir-emulator/tests/equiv/diff_py_vs_rust.py` (seeded numpy input gen + the
reference interpreter), Rust CLI `rust/crates/ktir-emulator/examples/ktir_diff_run.rs`
(`parse_module` + `execute_function`, raw little-endian byte marshalling). The driver
batches every (program × seed) case into **one** Rust invocation, so the fuzz count is
cheap. Tolerances: **f16 1e-2** abs, **f32 1e-4**, integer/index **EXACT (0)**.

Across **all 19 shared example programs** (validated to `FUZZ_ITERS=50` locally; the CI
gate runs the full corpus): **17 bit-exact PASS + 2 matched-failure, 0 gaps** — measured
against `origin/main`'s reference interpreter.

| program | what | result |
|---|---|---|
| `vector_add` | elementwise add | PASS |
| `vector_add_dynamic` | dynamic-shape add (f32) | PASS |
| `matmul` | [64,2048]×[2048,8192] GEMM | PASS |
| `layernorm` | fused layernorm (Y + Mean) | PASS |
| `softmax` | rowwise softmax | PASS |
| `softmax_wide` | wide softmax (per-row LX-liveness) | PASS |
| `sdpa` | scaled dot-product attention | PASS |
| `reduce_generic` | explicit-region reduce | PASS |
| `indexed_add` | non-identity indirect gather (`%idx[%grid0+%d0]`) | PASS |
| `paged_attention` | paged-KV attention | PASS |
| `indirect_access_copy` | indirect gather copy | PASS |
| `indirect_scatter` | indirect scatter | PASS |
| `add_with_control_flow` | add inside `scf.for` | PASS |
| `distributed_view_copy` | HBM + LX distributed view | PASS |
| `ring_reduce` | inter-tile ring all-reduce | PASS |
| `ring_reduce_inner_loop` | comm op inside `scf.for` | PASS |
| `ring_reduce_multi_group` | grouped ring all-reduce | PASS |
| `paged_tensor_copy` | 16 MB load > 2 MB LX | **MATCH-FAIL** (`lx_overflow`) |
| `paged_tensor_write` | out-of-bounds `BoxSet` | **MATCH-FAIL** (`shape_mismatch`) |

Every PASS row is **max-abs 0** — bit-identical, not merely within the f16 band.
**MATCH-FAIL is a conformant result, not a defect:** the program is an intentional error
fixture, and Rust raises the **same normalized error category** as Python. A program that
is supposed to fail but where Rust *succeeds* — or raises a *different* error — flips to a
real FAIL. (That fired this cycle: after the LX-liveness port `softmax_wide` stopped
overflowing in Python, so its stale `lx_overflow` matched-failure became a lie and was
promoted to a bit-exact PASS.)

**The harness earns its keep — it has caught real bugs the parity/golden tests missed**
and every one is now closed: a non-identity indirect subscript `ind(%index_view[%grid0 +
%d0])` the Rust IR model dropped (`indexed_add`, `paged_attention`); an f16-**subnormal**
`arith.truncf` rounding bug (half-value at the normal/subnormal boundary); a Metal GEMM
weight reader that treated the pointer SSA as a stick instead of an element index (an
e2e-golden-breaker, max-abs 32.9→0); and a fusion `rename_attrs` bug that shared one
`reduce` accumulator slot across layers (RMSNorm sum-of-squares grew unboundedly). The
port tracks the latest reference — element-index `base_ptr` (#110), multi-dim
`linalg.reduce` (#106), comm-in-`scf` (#133), and LX-liveness (#134/#118) — at **100%**.

### Metal fast-path conformance (tolerance-banded)

The table above is **bit-exact** because the example programs tile below the NAX gate, so
their compute runs on **AMX/CPU** (f32→f16, which matches numpy). The production **Metal
fast path** — NAX `matmul2d` / simdgroup GEMM + the fused map-window kernel + fused
attention — is conformance-checked **separately and tolerance-banded**: NAX rounds f16
inputs to **bf16**, so it *cannot* be bit-exact with Python's f16. The harness runs every
program through the Metal path (`KTIR_DIFF_RESIDENT=1` via the resident/segmented executor;
`KTIR_DIFF_GPU=1` via the per-op GPU path), **forcing every offload**, and diffs within a
first-principles band `tol(|v|) = 4·f16_ulp(|v|) + 2⁻⁸·|v|` (f16 output quant + bf16 input
rounding). Each program asserts a **mandatory `OffloadProof > 0`** — a silent CPU fallback
is a FALSE pass that FAILS.

| program | Metal kernel (proof) | max-abs Py-vs-Rust-Metal | result |
|---|---|---:|---|
| `vector_add` | map | 0 | PASS |
| `softmax` | map | 2.9e-6 | PASS |
| `softmax_wide` | map | 0 | PASS |
| `layernorm` | map | 0.00195 | PASS |
| `matmul` | NAX GEMM (512 tiles) | 0.00195 (1 f16 ULP) | PASS |
| `sdpa` | GEMM + map | 9.5e-7 | PASS |
| `paged_attention` | NAX GEMM (512) | 1.5e-5 | PASS |
| `vector_add_dynamic` | map | 0 | PASS |
| `indexed_add` | map | 0 | PASS |

**9/9 Metal-eligible programs conform within band — zero divergence**, no Metal kernel
needed a fix. (The CPU-only programs — `reduce_generic`, the HBM gather/scatter/comm
fixtures — have no Metal compute op, so there is nothing Metal to test.) A **gated descend**
routes `scf.for`-nested maps (softmax/layernorm) to the Metal map kernel **only under
`KTIR_FORCE_GPU_MAP`**, so the golden/production path stays byte-identical (golden 6/6
unforced). A fault injector proves the check catches a real Metal divergence. Gated test:
`tests/metal_conformance.rs`.

### Run it

```bash
# From the repo ROOT (the examples/ MLIR lives there).
cargo build --release --example ktir_diff_run -p ktir-emulator    # in rust/
FUZZ_ITERS=8 uv run --with numpy rust/crates/ktir-emulator/tests/equiv/diff_py_vs_rust.py
#   FUZZ_ITERS          seeded iterations per program (default 8; proven to 50)
#   KTIR_DIFF_PROGRAMS  comma-separated subset (default all)
#   KTIR_DIFF_RUN_BIN   prebuilt CLI path (skip the per-run cargo build)
```

CI (`rust-conformance.yml`) runs this on every push that touches `ktir_cpu/`,
`examples/`, `rust/`, or `rust/crates/ktir-emulator/tests/equiv/` — cheap (tiny programs, numpy-only, no model
weights), so it gates port faithfulness on every relevant change.

---

## Interpretation

- **Residency is the architectural win.** Marshaling weights into one persistent HBM
  and chaining kernels on-device makes the production Rust path 370–2700× the Python
  per-node reference on the same hermetic program. The decode→prefill jump (370× →
  2702× on smollm2) is the resident path amortizing the per-tile marshal that the
  per-pass model pays every token.
- **Metal wins biggest where the use case lives — big-model prefill.** At prefill
  scale (512×4096×4096) the raw NAX primitive is **1.9×** faster than Accelerate; the
  resident path compounds that with no per-pass marshal. NAX's edge grows with M
  (1.28× at M=64 → 1.91× at M=512), so the larger wins live at batched / long-context
  prefill, the actual target regime.
- **Size-gate the backend, not just GPU-vs-interpreter.** Inside the resident GEMM
  offload the gate picks NAX vs AMX over the same reconstructed full-M GEMM
  (`KTIR_GEMM_GPU_MIN_KN`, default `k·n` ≥ 3M → NAX, else AMX), for decode AND prefill.
  NAX only wins once a weight amortizes its ~300 µs dispatch; below that AMX
  (Accelerate on already-resident f32, no dispatch) is faster. M>1 is ALWAYS full-M
  (NAX or AMX) — never the interpreter `scf.for`.
- **Emulation overhead, not arithmetic.** The matmul microbench shows the interpreter's
  tiled SPMD path (428 ms) is ~100× the raw primitive (3.75 ms): most per-kernel time
  on small tensors is dispatch/marshal. The fastest backend is only as fast as the path
  feeding it — which is why residency matters more than the backend choice.

GPU-offload toggles (read by `comm_sched` / the resident executor):

| Env | Effect when set |
|---|---|
| `KTIR_NO_GPU_GEMM=1` | K-loop GEMMs stay on Accelerate (no NAX) |
| `KTIR_NO_GPU_MAP=1`  | map-window elementwise stays on the CPU interpreter |
| `KTIR_GEMM_GPU_MIN_KN` | min weight `k·n` to run an offloaded full-M GEMM on NAX vs AMX (default 3,000,000; 0 = always NAX). Also the m==1 offload-vs-interpreter gate. |
| `KTIR_MAP_GPU_MIN_ELEMS` | min output elems to offload a fused map window (default 16,384; 0 = always GPU) |
| `KTIR_LX_MB` | Python reference only: LX scratchpad size in MB (default 2). Raise (e.g. 512) to run programs whose coarse scope-level reclaim transiently exceeds 2 MB (llama prefill). Gates allocation, not compute. |

---

## History

Newest on top. The pre-2026-06-16 snapshots were **bundle-based** (the
`~/.cache/cudaforge/ktir/<bundle>/` scratchy dump, per-node / fused-AMX / fused-Metal /
RESIDENT 4-path tables); they were removed when the comparo moved to the hermetic
fixtures + HF-weights methodology above. They remain in git history.

### 2026-06-19 · branch `rust` · Apple M5 — NAX matmul loader + AOT

Two NAX changes landed since the 2026-06-16 snapshot, plus the cumulative resident /
fusion / f16-weight / last-token work in this PR — together cutting llama decode
261.5→97.4 and the production prefill 920.9(all-rows)→169.1(last-token) ms/pass:

- **Vectorized `matmul2d` threadgroup loader** — wide 4-element coalesced device loads
  + threadgroup stores replace the per-element `div`/`mod` + bounds-checked staging that
  capped weight streaming at ~12 GB/s. Bit-identical output; **−20% llama decode / −14%
  llama prefill** vs the scalar loader (interleaved, same thermal state: m=128 qkv
  1.9× / gate 1.6× on the raw GEMM). BK=32 / double-buffer-32 variants regressed (this
  GPU's 32 KB threadgroup cap can't fit a double-buffered 32-wide panel).
- **AOT-precompiled NAX kernels** — `build.rs` embeds all 12 kernel-variant metallibs,
  compiled `-mmacosx-version-min=26.2` to dodge the SDK-26.5 offline-toolchain
  `matmul2d` half-K miscompile (verified: without the flag a ones-GEMM reduces to 64
  not 128), with a JIT `newLibraryWithSource` fallback. Startup-only (~36 ms one-time,
  0.1% of serving), so the steady-state table is unchanged by it.

### 2026-06-16 · branch `rust` · Apple M5 — hermetic Python-vs-Rust E2E

Moved the whole-model comparo off the scratchy bundle: the program is vendored in
`tests/fixtures/` and weights come from public HuggingFace, so the Python AND Rust
sides run the same thing with no bundle. To make this work the Python interpreter
gained a `linalg.matmul_transpose_b` handler (the production emit binds weights
verbatim `[out,in]` and contracts the last axis) and a `KTIR_LX_MB` LX-size override
(llama prefill transiently exceeds the faithful 2 MB under the reference's coarse LX
reclaim). Rust side: `e2e_real_forward` gained `time_*_resident` (best-of-N resident
ms/pass on the fixtures); Python side: `tests/fixtures/bench_e2e_hermetic.py`.

The Python reference was also de-gratuitized (PR #124: vectorized `ktdp.load` offset
calc, `ravel`-not-`flatten` allocation reads, + O(log n) allocation lookup) so the
comparo measures the interpreter, not artifacts — 12–20× faster Python; the table
below is post-fix.

E2E ms/pass (Python per-node reference vs Rust RESIDENT), best-of-7 Rust:

| Model / mode | Python | Rust RESIDENT | speedup |
|---|---:|---:|---:|
| smollm2-135m decode  |    2,536.3 |  99.2 |   26× |
| smollm2-135m prefill |   60,397.4 | 437.5 |  138× |
| llama-3.2-1b decode  |   31,777.6 | 261.5 |  122× |
| llama-3.2-1b prefill |  977,688.3 | 920.9 | 1062× |
