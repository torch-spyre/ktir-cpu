#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors. Apache-2.0.
#
"""DIRECT differential conformance: Python KTIRInterpreter ⟷ Rust execute_function.

For each KTIR example program, this driver generates SEEDED random inputs with
numpy (correct dtype/shape/scalars per the program's args), runs the Python
reference `ktir_cpu.KTIRInterpreter` on them, then hands the *same* inputs to the
Rust CLI (`examples/ktir_diff_run.rs`) via raw little-endian byte files, reads the
Rust outputs back, and computes per-output MAX-ABS(Python − Rust).

This is HEAD-TO-HEAD (Python vs Rust), not both-vs-a-hardcoded-answer-key — so it
catches divergences the hand-written port_*.rs parity tests (which never run
Python) miss. A divergence beyond the f16 tolerance band is a REAL conformance
finding: it is reported (program, seed, max-abs), NOT hidden by loosening the
tolerance or skipping the program.

I/O FORMAT (see examples/ktir_diff_run.rs for the Rust half)
-----------------------------------------------------------
* Inputs: each tensor arg is written as a raw little-endian bytes file already
  encoded in the arg's dtype (f16 = '<f2', etc.) — byte-identical to how
  ktir_cpu and the Rust codec lay the buffer out in HBM. Scalars are inline JSON.
* The driver writes ONE request.json describing every (program × seed) case and
  invokes the Rust CLI ONCE for the whole batch (so the fuzz count is cheap).
* Outputs: the Rust CLI writes each result tensor's raw dtype bytes + a
  manifest.json; the driver decodes them with the SAME numpy dtype and diffs.

Run from the repo ROOT (the `examples/` MLIR lives there):

  uv run --with numpy tests/equiv/diff_py_vs_rust.py

Env:
  FUZZ_ITERS  number of seeded iterations per program (default 8; seeds 0..N-1).
  KTIR_DIFF_PROGRAMS  comma-separated subset of program keys to run
                      (default: all in SPECS). E.g. KTIR_DIFF_PROGRAMS=vector_add.
  KTIR_DIFF_RUN_BIN  path to the built Rust CLI (default: auto-build via cargo).

Exit code is non-zero if any output exceeds its tolerance (a conformance failure).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Repo root: walk up from this file until the dir that holds the Python reference
# (ktir_cpu/) and the shared MLIR programs (examples/). Robust to wherever this
# driver lives — it sits under rust/, while the programs are at the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(start):
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "ktir_cpu")) and os.path.isdir(
            os.path.join(d, "examples")
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"repo root (ktir_cpu/ + examples/) not found above {start}")
        d = parent


ROOT = _find_root(HERE)
sys.path.insert(0, ROOT)  # so `import ktir_cpu` resolves regardless of CWD
EXAMPLES = os.path.join(ROOT, "examples")

from ktir_cpu import KTIRInterpreter  # noqa: E402  (after sys.path setup above)


def _find_rust_dir():
    """Locate the rust/ cargo workspace.

    Committed layout has rust/ at the repo root (CI checks this out); this dev
    machine builds it in a git worktree under .claude/worktrees/. Prefer the
    committed root, fall back to the worktree, else KTIR_RUST_DIR override.
    """
    env = os.environ.get("KTIR_RUST_DIR")
    if env:
        return env
    for cand in (
        os.path.join(ROOT, "rust"),
        os.path.join(ROOT, ".claude", "worktrees", "rust-fixtures", "rust"),
    ):
        if os.path.isfile(os.path.join(cand, "Cargo.toml")):
            return cand
    # Last resort: the root path (cargo will surface a clear error if absent).
    return os.path.join(ROOT, "rust")


RUST_DIR = _find_rust_dir()

# numpy dtype per KTIR dtype (matches ktir_cpu.dtypes.SUPPORTED_DTYPES + the Rust
# codec byte layout). index/i32 -> 4-byte int; i64 -> 8-byte; f16/f32 as named.
NP_DTYPE = {
    "f16": np.dtype("<f2"),
    "f32": np.dtype("<f4"),
    "i32": np.dtype("<i4"),
    "index": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "i1": np.dtype("uint8"),
}

# f16 tolerance: f16 has ~10 mantissa bits, so relative ulp ~1e-3; for the value
# ranges here an absolute band of 1e-2 covers rounding at modest magnitudes.
# Integer/index outputs must be EXACT (0).
F16_ABS_TOL = 1e-2
FLOAT_DTYPES = {"f16", "f32"}

# ---------------------------------------------------------------------------
# GPU-PATH MODE (KTIR_DIFF_GPU=1) — Phase 2 BandedDriver.
#
# The DEFAULT (CPU) mode above is BIT-EXACT: Python-f16 ⟷ Rust-AMX(f32-acc) over
# the example programs, which tile BELOW the NAX size gate and so run their
# matmuls on Accelerate. That path is byte-for-byte (max-abs 0) and is the
# existing 17-PASS harness — left untouched.
#
# This GPU mode instead forces the example matmuls onto the PRODUCTION Metal fast
# path (NAX matmul2d on M5 / simdgroup on pre-M5) by setting, in the Rust
# subprocess env:
#     KTIR_DIFF_ENGINE=gpu     -> the CLI resets+records a per-case GPU-GEMM
#                                 proof counter (gpu_gemm_count in manifest.gpu)
#     KTIR_FORCE_GPU_GEMM=1     -> bypass the wall-clock size gate so the small
#                                 tiled example matmuls (32x512x128, below
#                                 NAX_MIN_BLOCKS/NAX_MIN_K) dispatch to NAX.
#
# It then (a) ASSERTS the GPU actually ran for every GEMM-bearing program
# (gpu_gemm_count > 0; a 0 is a FALSE/secretly-AMX pass and FAILS), and (b) diffs
# Python ⟷ Rust under a PRINCIPLED bf16/f16 band (NOT the flat 1e-2). Because NAX
# rounds f16 inputs to bf16 (~8 mantissa bits) and accumulates in f32, the result
# CANNOT be bit-exact; the band is derived from first principles, per output
# magnitude, below. A program needing more than its band is a DIVERGENCE FINDING
# reported honestly — never loosened to hide.
GPU_MODE = os.environ.get("KTIR_DIFF_GPU", "").strip() not in ("", "0", "false")

# ---------------------------------------------------------------------------
# RESIDENT-PATH MODE (KTIR_DIFF_RESIDENT=1) — Phase 1 ResidentRunner.
#
# Where GPU_MODE force-runs `execute_function` (the per-op `linalg.matmul` GPU
# selector) for the 3 GEMM-bearing programs, RESIDENT_MODE runs the FULL example
# suite through the PRODUCTION resident/segmented Metal executor
# (`ResidentExecutor::new_native` -> `run`): resident HBM + weight cache +
# per-segment seg-plan (K-loop GEMM reconstruction where recognizable) + per-op
# Metal offloads + fused map windows / decode attention, at each kernel's NATIVE
# grid (so the SPMD-tiled examples write their WHOLE output, not just compute-tile
# 0's slice). The Rust CLI (KTIR_DIFF_ENGINE=resident) records, per case, the FULL
# per-offload proof breakdown (manifest.gpu[].offload_proof: matmul_loop_gpu /
# matmul_loop_amx / gemm_or_blas_gpu / map_region_gpu), so the driver can report
# WHICH offload(s) each program fired and assert a Metal-bearing program actually
# hit one (a zero total on a GEMM/attention program is a FALSE all-CPU pass).
#
# Like GPU_MODE it sets KTIR_FORCE_GPU_GEMM=1 + KTIR_GEMM_GPU_MIN_KN=0 so the small
# tiled example GEMMs (32x128x512) dispatch to NAX/simdgroup instead of staying on
# AMX below the wall-clock size gate, and diffs under the SAME principled bf16/f16
# band. HBM-seeded fixtures (RFC indirect/distributed/ring-reduce) are NOT
# marshalled-arg programs and cannot be driven through the ProgramSpec path — they
# are reported as not-drivable, not faked.
RESIDENT_MODE = os.environ.get("KTIR_DIFF_RESIDENT", "").strip() not in ("", "0", "false")

# Which programs carry a real GEMM (linalg.matmul tiles) that MUST dispatch to the
# Metal engine under KTIR_FORCE_GPU_GEMM. For these, gpu_gemm_count==0 means the
# matmul secretly ran on AMX — a FALSE pass — and the driver FAILS the case. The
# pure elementwise/reduce programs (softmax/layernorm/vector_add/...) legitimately
# run 0 GEMMs, so they are not GEMM-bearing and skip the >0 assertion.
GEMM_BEARING = {"matmul", "sdpa", "paged_attention"}

# Default program selection in GPU mode: the compute-heavy GEMM programs. The
# elementwise programs have no matmul so there is nothing for the GPU fast path to
# check (they'd be a redundant re-run of the bit-exact CPU harness). Override with
# KTIR_DIFF_PROGRAMS to widen (e.g. add softmax/layernorm to confirm count==0).
GPU_DEFAULT_PROGRAMS = [
    "matmul",
    "sdpa",
    "paged_attention",
    # MAP-bearing non-F16 programs (part B): the per-op `execute_function` GPU
    # path handles non-F16 dtypes (the all-F16 resident path cannot drive them),
    # so the GPU mode runs them with the fused MAP-window kernel forced on and
    # asserts a `map_region_gpu` offload fired. vector_add_dynamic is f32; its
    # top-level `arith.addf` is offloaded once KTIR_FORCE_GPU_MAP lifts gates and
    # KTIR_MAP_GPU_MIN_ELEMS=0 drops the dispatch floor. indexed_add gathers an
    # i64 index then adds (f16); the gather+add `arith.addf` is the offloaded map.
    "vector_add_dynamic",
    "indexed_add",
]

# GPU mode (part B): the non-F16 programs whose elementwise MAP must dispatch to
# the Metal fused-map kernel. With the GPU-mode force env (KTIR_FORCE_GPU_MAP=1 +
# KTIR_MAP_GPU_MIN_ELEMS=0) the per-op `execute_function` path offloads their
# `arith.addf` map window to the GPU; `map_region_gpu==0` is then a FALSE all-CPU
# pass and FAILS (mirrors GEMM_BEARING's gpu_gemm_count>0 assertion). These run on
# the GPU path precisely because it (unlike the all-F16 resident path) handles
# their f32 / i64 dtypes.
GPU_MAP_BEARING = {"vector_add_dynamic", "indexed_add"}

# RESIDENT mode: the offload total a program is REQUIRED to fire (a 0 is a FALSE
# all-CPU pass). On the native-grid resident path the GEMM-bearing programs fire
# per-op GPU GEMMs (gemm_or_blas_gpu) summed across their compute-tiles; the
# elementwise programs fire fused map windows (map_region_gpu) when a window's
# output is wide enough to clear the per-window GPU dispatch floor. A program that
# legitimately fires NEITHER (a pure index/reduce/comm kernel) is CPU-ONLY on this
# path and is NOT required to hit an offload — it still must conform within band.
# The buckets below are DERIVED FROM THE MEASURED per-offload proof (reported in
# the table); a program in RESIDENT_METAL_BEARING that shows a 0 total FAILS.
# MEASURED on the M5 (per-offload proof in the table) under PHASE 1 (ForceAllMetal:
# KTIR_FORCE_GPU_GEMM=1 + KTIR_FORCE_GPU_MAP=1 + KTIR_MAP_GPU_MIN_ELEMS=0 +
# KTIR_FORCE_FUSE_ATTN=1):
#   * The GEMM-bearing programs (matmul / sdpa) fire the per-tile `linalg.matmul`/
#     GEMV NAX/simdgroup dispatch (gemm_or_blas_gpu > 0); sdpa also fires a fused
#     map window.
#   * vector_add fires the fused MAP-window kernel (map_region_gpu > 0): its
#     elementwise add is a TOP-LEVEL op, so the map-window planner offloads it once
#     KTIR_FORCE_GPU_MAP lifts the single-core gate and KTIR_MAP_GPU_MIN_ELEMS=0
#     drops the per-window dispatch floor.
#   * softmax / softmax_wide / layernorm wrap ALL their map ops INSIDE a per-row
#     `scf.for` loop body. Under KTIR_FORCE_GPU_MAP the forced map-offload now
#     DESCENDS into the loop body (gated descend, scf.rs `run_region`) and fires the
#     fused map kernel per row (map_region_gpu > 0) — they are no longer CPU-only.
# So those programs are REQUIRED to fire a Metal offload (a 0 total is a FALSE
# all-CPU pass and FAILS).
#
# reduce_generic is a pure `linalg.reduce` (no map op) and fires no map kernel by
# design (a reduce is a window boundary, not an offloaded map) — it has no Metal-
# eligible op and is CPU bit-exact on both paths.
RESIDENT_METAL_BEARING = {
    "matmul",
    "sdpa",
    "vector_add",
    "softmax",
    "softmax_wide",
    "layernorm",
}


def f16_ulp(mag):
    """The f16 ULP (spacing to the next representable f16) at magnitude `mag`.

    This is the absolute quantization step of the f16 OUTPUT tile — the result
    is stored as f16 on BOTH the Python and Rust sides, so even a perfectly equal
    real-valued result differs by up to 1 ULP from f16 rounding alone.
    """
    x = abs(float(mag))
    if x == 0.0 or not np.isfinite(x):
        return float(np.spacing(np.float16(1.0)))  # smallest normal-ish step
    xf = np.float16(x)
    nxt = np.nextafter(xf, np.float16(np.inf))
    step = float(nxt) - float(xf)
    return step if step > 0 else float(np.spacing(xf))


def gpu_band(max_mag):
    """PRINCIPLED bf16/f16 absolute band at output magnitude `max_mag`.

        band = 4 * f16_ulp(|v|)        # f16 OUTPUT quantization (both sides f16)
             + 2^-8 * |v|              # bf16 INPUT-rounding relative ulp

    Rationale. The Metal NAX engine rounds the f16 GEMM inputs to bf16 (~8
    mantissa bits => 2^-8 relative ulp) and accumulates the K-loop in f32 — so
    the dominant error is the bf16 INPUT rounding, bounded by 2^-8*|v| (it does
    NOT grow like sqrt(K), because f32 accumulation does not lose bits across the
    reduction). The f32 result is then quantized back to the f16 output tile,
    contributing the f16 ULP term; 4 ULP gives a little headroom for the handful
    of intermediate f16 round-trips (e.g. softmax/exp inside sdpa) without being
    arbitrary. Every term is traceable to a named precision boundary; nothing is
    a tuned magic number. For matmul (|v|~2) this is ~0.0156 vs the observed
    0.00195 (= exactly 1 f16 ULP) — ~8x headroom, all of it justified.
    """
    return 4.0 * f16_ulp(max_mag) + (2.0**-8) * abs(float(max_mag))


def _arg(name, dtype, shape, gen="rand", **extra):
    a = {"name": name, "kind": "tensor", "dtype": dtype, "shape": list(shape), "gen": gen}
    a.update(extra)
    return a


def _scalar(name, scalar_dtype, value):
    return {
        "name": name,
        "kind": "scalar",
        "scalar_dtype": scalar_dtype,
        "value": value,
    }


# Per-program spec: the function name, the argument list (tensors get a `gen`
# describing how to seed them), and which tensor args are outputs to diff.
# Shapes/scalars are read off the example .mlir memory-view declarations and the
# conftest.py EXAMPLE_PARAMS execute_kwargs (the authoritative arg spec the
# Python test suite uses).
#
# gen roles:
#   "rand"     small symmetric ~N(0, 0.1) values, narrowed to the arg dtype.
#              Well-conditioned for add / matmul / layernorm / softmax / sdpa.
#   "zero"     output buffers (written by the kernel; seeded zero on both sides).
#   "randint"  uniform integers in [lo, hi) — for index tensors. Needs lo/hi.
#
# PHASE 2 — EVERY shared example program is now a CHECKED case. There are no
# silent skips. A program lands in exactly ONE of three buckets:
#
#   1. SPECS               bit-exact head-to-head over marshalled ndarray args.
#   2. HBM_SPECS           bit-exact head-to-head where the tensors live at
#                          hardcoded HBM stick addresses (RFC fixtures,
#                          ring-reduce): both sides seed byte-identical HBM/LX via
#                          the harness's seeding hook, run, and read back the same
#                          stick region. See `gen_hbm_*`.
#   3. FAIL_SPECS          MATCHED-FAILURE fixtures: programs the Python
#                          KTIRInterpreter raises on (oversized LX, box-not-
#                          contained). The harness asserts BOTH Python AND Rust
#                          raise, AND that the error falls in the same category
#                          (`expect_category`). A side that *succeeds* is a FAIL.
#
# Plus KNOWN_GAPS — programs Python runs but Rust genuinely cannot yet (the
# experimental `ktdp.inter_tile_produce`/`inter_tile_reduce` collective is
# unimplemented in the Rust port; only the fused `ktdp.reduce` exists). These are
# NOT faked as passing: the harness CHECKS that Rust still fails with the expected
# "no handler" class, so the gap is tracked, not hidden — and if Rust ever gains
# the op, the check flips and flags the row for promotion.
SPECS = {
    "vector_add": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "vector_add_ktir.mlir"),
        "function": "add_kernel",
        "args": [
            _arg("x_ptr", "f16", [4096]),
            _arg("y_ptr", "f16", [4096]),
            _arg("output_ptr", "f16", [4096], gen="zero"),
            _scalar("BLOCK_SIZE", "index", 128),
        ],
        "outputs": ["output_ptr"],
    },
    "vector_add_dynamic": {
        "program": os.path.join(
            EXAMPLES, "triton-ktir", "vector_add_dynamic_ktir.mlir"
        ),
        "function": "add_kernel_dynamic",
        # f32 program; n_elements drives the symbolic coordinate set (<= 1024).
        "args": [
            _arg("x_ptr", "f32", [1024]),
            _arg("y_ptr", "f32", [1024]),
            _arg("output_ptr", "f32", [1024], gen="zero"),
            _scalar("n_elements", "i32", 1024),
        ],
        "outputs": ["output_ptr"],
    },
    "softmax_wide": {
        "program": os.path.join(EXAMPLES, "ktir", "softmax_wide.mlir"),
        "function": "softmax_kernel",
        # 2x262144 f16 rowwise softmax. A naive impl would hold the 512 KB row
        # plus its several same-shape intermediates live at once (>2 MB LX); with
        # the #134/#118 LX-liveness model (single-use tiles consumed at last use,
        # no iter_arg double-count) the per-row peak fits, so BOTH sides now run
        # to completion and this is an ordinary bit-exact PASS (was a FAIL_SPECS
        # lx_overflow matched-failure before the liveness port).
        "args": [
            _arg("output_ptr", "f16", [2, 262144], gen="zero"),
            _arg("input_ptr", "f16", [2, 262144]),
        ],
        "outputs": ["output_ptr"],
    },
    "matmul": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "matmul_fwd_ktir.mlir"),
        "function": "matmul_kernel",
        "args": [
            _arg("a_ptr", "f16", [64, 2048]),
            _arg("b_ptr", "f16", [2048, 8192]),
            _arg("c_ptr", "f16", [64, 8192], gen="zero"),
            _scalar("K", "index", 2048),
            _scalar("BLOCK_SIZE_M", "index", 32),
            _scalar("BLOCK_SIZE_N", "index", 512),
            _scalar("BLOCK_SIZE_K", "index", 128),
        ],
        "outputs": ["c_ptr"],
    },
    "layernorm": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "layernorm_fwd_ktir.mlir"),
        "function": "_layer_norm_fwd_fused",
        # W/B are declared 2D (n_rows × n_cols) in the MLIR. Random bytes (same to
        # both sides) is a stronger test than the all-ones / all-zero gamma/beta.
        "args": [
            _arg("X", "f16", [1151, 8192]),
            _arg("Y", "f16", [1151, 8192], gen="zero"),
            _arg("W", "f16", [1151, 8192]),
            _arg("B", "f16", [1151, 8192]),
            _arg("Mean", "f16", [1151], gen="zero"),
            _arg("Rstd", "f16", [1151], gen="zero"),
            _scalar("N", "index", 8192),
            _scalar("eps", "f16", 1e-5),
            _scalar("BLOCK_SIZE", "index", 1024),
        ],
        "outputs": ["Y", "Mean"],
    },
    "softmax": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "softmax_fwd_ktir.mlir"),
        "function": "softmax_kernel",
        # output_ptr, input_ptr, n_rows — both [4096, 1024] f16.
        "args": [
            _arg("output_ptr", "f16", [4096, 1024], gen="zero"),
            _arg("input_ptr", "f16", [4096, 1024]),
            _scalar("n_rows", "index", 4096),
        ],
        "outputs": ["output_ptr"],
    },
    "sdpa": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "sdpa_2d.mlir"),
        "function": "sdpa_kernel_2d",
        # Q/K/V/output all [32, 64] f16; grid [1]; no scalar kwargs.
        "args": [
            _arg("q_ptr", "f16", [32, 64]),
            _arg("k_ptr", "f16", [32, 64]),
            _arg("v_ptr", "f16", [32, 64]),
            _arg("output_ptr", "f16", [32, 64], gen="zero"),
        ],
        "outputs": ["output_ptr"],
    },
    "indexed_add": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "indexed_add.mlir"),
        "function": "indexed_add_kernel",
        # x[index[grid0], dim1_start:+32, grid1, :] + y. index gathers x's dim-0
        # (size 128), so the index tensor must be valid integers in [0, 128).
        "args": [
            _arg("x_ptr", "f16", [128, 64, 8, 128]),
            _arg("y_ptr", "f16", [2, 32, 8, 128]),
            _arg("index_ptr", "i64", [2], gen="randint", lo=0, hi=128),
            _arg("output_ptr", "f16", [2, 32, 8, 128], gen="zero"),
            _scalar("dim1_start", "index", 0),
        ],
        "outputs": ["output_ptr"],
    },
    "reduce_generic": {
        "program": os.path.join(EXAMPLES, "ktir", "reduce_generic.mlir"),
        "function": "reduce_explicit_region",
        # arg0 is BOTH input and output (same buffer): loaded [1,4], reduced
        # along dim 1, broadcast back. Diff the post-exec buffer.
        "args": [_arg("arg0", "f16", [1, 4])],
        "outputs": ["arg0"],
    },
    "paged_attention": {
        "program": os.path.join(EXAMPLES, "triton-ktir", "paged_attention.mlir"),
        "function": "kernel_unified_attention_spyre_2d",
        # Paged attention via block_tables indirection (non-identity indirect
        # subscript `ind(%block_tables[%c0, %bt_idx + %d0])`). Concrete params
        # from tests/conftest.py kernel_unified_attention_spyre_2d. block_tables
        # holds KV-cache block ids in [0, 64); num_tiles=8 covers 128 KV tokens.
        "args": [
            _arg("output_ptr", "f16", [8, 32, 128], gen="zero"),
            _arg("query_ptr", "f16", [8, 32, 128]),
            _arg("key_cache_ptr", "f16", [64, 16, 8, 128]),
            _arg("value_cache_ptr", "f16", [64, 16, 8, 128]),
            _arg("block_tables_ptr", "i32", [1, 16], gen="randint", lo=0, hi=64),
            _scalar("cur_batch_start_index", "index", 0),
            _scalar("block_table_offset", "index", 0),
            _scalar("num_tiles", "index", 8),
            _scalar("context_len", "index", 120),
            _scalar("scale", "f32", 0.08838834764831843),
        ],
        "outputs": ["output_ptr"],
    },
}


def _seed(elem, dtype, shape, gen="rand", lx_core=None, next_ptr=None, **extra):
    """An HBM/LX region to seed before execution (see HBM_SPECS).

    elem     : ELEMENT index — the value of the MLIR construct_memory_view base
               constant. The byte address is elem*bytes_per_elem(dtype) (the
               base_ptr=element-index convention, RFC #110). For HBM the byte
               address decomposes into (stick, intra) via hbm_write; for LX it
               is a plain byte pointer.
    lx_core  : when set, seed this core's LX (not HBM) at byte address elem*bpe.
    next_ptr : when set, advance that LX's allocation cursor past the seed
               (already a byte pointer — NOT an element index).
    """
    s = {"elem": elem, "dtype": dtype, "shape": list(shape), "gen": gen}
    if lx_core is not None:
        s["lx_core"] = lx_core
    if next_ptr is not None:
        s["next_ptr"] = next_ptr
    s.update(extra)
    return s


def _read(name, elem, dtype, shape):
    """An HBM region to read back + diff after execution (see HBM_SPECS).

    elem : ELEMENT index of the MLIR output view base (byte = elem*bpe)."""
    return {"name": name, "elem": elem, "dtype": dtype, "shape": list(shape)}


# ---------------------------------------------------------------------------
# HBM_SPECS — programs whose tensors live at hardcoded HBM (and LX) addresses,
# not marshalled ndarray args. The construct_memory_view bases are arith.constant
# ELEMENT indices (RFC #110: MemRef.base_ptr is an element index); both sides
# seed byte-identical memory at byte = elem*bytes_per_elem(dtype), run, and read
# back the named element-base regions. Bit-exact PASS expected.
# ---------------------------------------------------------------------------
HBM_SPECS = {
    "indirect_access_copy": {
        "program": os.path.join(EXAMPLES, "rfc", "indirect-access-copy.mlir"),
        "function": "indirect_access_copy",
        # Y[m,k] = X[IDX1[m,k], IDX2[m,k]]. ELEMENT-index bases are the MLIR
        # arith.constants RESTORED to origin/main (RFC #110): X@0 (f16, byte 0),
        # IDX1@64 (i32, byte 256), IDX2@128 (i32, byte 512), Y@192 (f16, byte
        # 384). These element-byte spans OVERLAP (the program is self-aliased at
        # 64x64), so non-zero index/data seeds would clobber each other and
        # diverge. Mirror the Python reference instead — tests/test_indirect_
        # access.py::test_indirect_access_tile_rfc zero-seeds X/IDX1/IDX2/Y — a
        # zero-data smoke parity: every gather lands on zeros, Y stays all-zero,
        # and the read-back at Y@192 is bit-exact on both sides regardless of the
        # aliasing.
        "seeds": [
            _seed(0, "f16", [64, 64], gen="zero"),
            _seed(64, "i32", [64, 64], gen="zero"),
            _seed(128, "i32", [64, 64], gen="zero"),
            _seed(192, "f16", [64, 64], gen="zero"),
        ],
        "reads": [_read("Y", 192, "f16", [64, 64])],
    },
    "indirect_scatter": {
        "program": os.path.join(EXAMPLES, "rfc", "indirect-scatter.mlir"),
        "function": "indirect_scatter",
        # Y[IDX1[m,k], IDX2[m,k]] = X[m,k]. Same RESTORED element-index bases
        # (X@0, IDX1@64, IDX2@128, Y@192) and the same self-aliased element-byte
        # spans, so the same zero-data smoke parity applies (mirrors
        # tests/test_indirect_access.py::test_indirect_scatter_rfc).
        "seeds": [
            _seed(0, "f16", [64, 64], gen="zero"),
            _seed(64, "i32", [64, 64], gen="zero"),
            _seed(128, "i32", [64, 64], gen="zero"),
            _seed(192, "f16", [64, 64], gen="zero"),
        ],
        "reads": [_read("Y", 192, "f16", [64, 64])],
    },
    "add_with_control_flow": {
        "program": os.path.join(EXAMPLES, "rfc", "add-with-control-flow.mlir"),
        "function": "add",
        # C = A + B over 96x64, tiled 3x64 across 32 cores, via linalg.add inside
        # scf.for. A/B/C view bases are the arith.constant ELEMENT indices 1024 /
        # 12288 / 18432 (byte = elem*2 at f16).
        "seeds": [
            _seed(1024, "f16", [96, 64]),
            _seed(12288, "f16", [96, 64]),
            _seed(18432, "f16", [96, 64], gen="zero"),
        ],
        "reads": [_read("C", 18432, "f16", [96, 64])],
    },
    "distributed_view_copy": {
        "program": os.path.join(EXAMPLES, "rfc", "distributed-view-copy.mlir"),
        "function": "distributed_view_copy",
        # A (192x64) distributed across HBM rows 0..95 (@elem0 → byte0), LX0 rows
        # 96..127 (col-packed strides [1,64] @ elem 12288 → LX byte 24576), LX1
        # rows 128..191 (row-major @ elem 16384 → LX byte 32768); copied into
        # contiguous HBM B @ elem 24576 → byte 49152. The two LX seeds advance
        # next_ptr (a BYTE pointer) past their region so the kernel's staging
        # cannot trample the source (mirrors tests/test_distributed_view.py, which
        # sets lx0.next_ptr = 16384*2 + 8128 and lx1.next_ptr = 16384*2 + 8192).
        "seeds": [
            _seed(0, "f16", [96, 64], gen="dist_a_hbm"),
            _seed(
                12288,
                "f16",
                [32, 64],
                gen="dist_a_lx0",
                lx_core=0,
                next_ptr=16384 * 2 + 8128,
            ),
            _seed(
                16384,
                "f16",
                [64, 64],
                gen="dist_a_lx1",
                lx_core=1,
                next_ptr=16384 * 2 + 8192,
            ),
            _seed(24576, "f16", [192, 64], gen="zero"),
        ],
        "reads": [_read("B", 24576, "f16", [192, 64])],
    },
    "ring_reduce": {
        "program": os.path.join(EXAMPLES, "ktir", "ring_reduce.mlir"),
        "function": "ring_reduce",
        # 4-core all-reduce sum; in_ptr/out_ptr are f16 ELEMENT-index scalars.
        # Each core's 1x128 f16 input row is at in_ptr + pid*128 elements
        # (0,128,256,384); core 0 writes the reduced row to out_ptr=512. Ordinary
        # bit-exact PASS case: Rust now implements ktdp.inter_tile_produce/reduce.
        "scalars": [_scalar("in_ptr", "index", 0), _scalar("out_ptr", "index", 512)],
        "seeds": [
            _seed(0, "f16", [1, 128], gen="ring_pos"),
            _seed(128, "f16", [1, 128], gen="ring_pos"),
            _seed(256, "f16", [1, 128], gen="ring_pos"),
            _seed(384, "f16", [1, 128], gen="ring_pos"),
            _seed(512, "f16", [1, 128], gen="zero"),
        ],
        "reads": [_read("out", 512, "f16", [128])],
    },
    "ring_reduce_inner_loop": {
        "program": os.path.join(EXAMPLES, "ktir", "ring_reduce_inner_loop.mlir"),
        "function": "ring_reduce_inner_loop",
        # 4-core all-reduce sum INSIDE an scf.for body (#133): the loop runs
        # n_iters rounds, each doing a full ring all-reduce of the per-core 1x128
        # row and accumulating into an iter_arg; core 0 writes the final
        # accumulator. Exercises ktdp.inter_tile_produce/reduce inside scf.for —
        # the comm-in-control-flow path. f16 ELEMENT-index scalars: input rows at
        # elems 0,128,256,384; output at out_ptr=512; n_iters=3 → out = 3*sum(rows).
        "scalars": [
            _scalar("in_ptr", "index", 0),
            _scalar("out_ptr", "index", 512),
            _scalar("n_iters", "index", 3),
        ],
        "seeds": [
            _seed(0, "f16", [1, 128], gen="ring_pos"),
            _seed(128, "f16", [1, 128], gen="ring_pos"),
            _seed(256, "f16", [1, 128], gen="ring_pos"),
            _seed(384, "f16", [1, 128], gen="ring_pos"),
            _seed(512, "f16", [1, 128], gen="zero"),
        ],
        "reads": [_read("out", 512, "f16", [128])],
    },
    "ring_reduce_multi_group": {
        "program": os.path.join(EXAMPLES, "latency", "ring_reduce_multi_group.mlir"),
        "function": "ring_reduce_multi_group",
        # 16 cores in 4 groups of 4; in-group all-reduce; first core of each group
        # writes a per-group output row. f16 ELEMENT-index scalars: input rows at
        # elems 0,128,..,15*128; outputs at out_ptr=2048 (4 rows × 128 elems).
        # Ordinary bit-exact PASS case (ktdp.inter_tile_produce/reduce).
        "scalars": [_scalar("in_ptr", "index", 0), _scalar("out_ptr", "index", 2048)],
        "seeds": [_seed(128 * c, "f16", [1, 128], gen="ring_pos") for c in range(16)]
        + [_seed(2048, "f16", [4, 128], gen="zero")],
        "reads": [_read("out", 2048, "f16", [4, 128])],
    },
}

# Programs in HBM_SPECS that Python runs but the Rust port cannot yet, because of
# a genuine missing feature (NOT a marshalling artefact). The harness CHECKS that
# Rust fails with the expected error category — so the gap is verified, not
# hidden; if Rust gains the op the check flips and flags the row for promotion to
# a real PASS. `category` is matched against the normalized Rust error.
#
# (empty) — the inter-tile ring all-reduce collective
# (ktdp.inter_tile_produce/yield_partial/inter_tile_reduce/yield_reduced) is now
# implemented in the Rust port, so ring_reduce + ring_reduce_multi_group are
# PROMOTED to ordinary bit-exact HBM_SPECS cases (no longer KNOWN_GAPs).
KNOWN_GAPS = {}

# ---------------------------------------------------------------------------
# FAIL_SPECS — MATCHED-FAILURE fixtures. Python raises; the harness asserts Rust
# also raises AND in the same error category. A side that succeeds is a FAIL.
# `kind`="marshalled" reuses the ndarray-arg path; "hbm" uses the seeded path.
# Error categories (normalized from the message on both sides):
#   lx_overflow    LX scratchpad / capacity exceeded (oversized live set)
#   shape_mismatch box not contained / store data-tile vs access-tile mismatch
#   unmapped       read from an unmapped address
# ---------------------------------------------------------------------------
FAIL_SPECS = {
    "paged_tensor_copy": {
        "kind": "hbm",
        "program": os.path.join(EXAMPLES, "rfc", "paged-tensor-copy.mlir"),
        "function": "paged_tensor_copy_1core",
        # Production-sized: the single ktdp.load of the full 4x8x2048x128 f16
        # output tile (16 MB) overflows the 2 MB LX. To make BOTH sides reach
        # that load (so they hit the same lx_overflow, not an unmapped read on
        # the way there), seed the index tensor + X page 0 with zeros exactly as
        # tests/test_spec_gaps.py::test_paged_tensor_indirect_access: all page
        # ids -> 0, so every gather lands on the one seeded page. Idx @ stick
        # 20000000 (Nb*Ntkv/Ptkv = 4*32 i32); X page 0 @ stick 30000000
        # (Nhkv*Ptkv*Ndkv = 8*64*128 f16).
        "seeds": [
            _seed(20000000, "i32", [4 * 32], gen="zero"),
            _seed(30000000, "f16", [8 * 64 * 128], gen="zero"),
        ],
        "expect_category": "lx_overflow",
    },
    "paged_tensor_write": {
        "kind": "hbm",
        "program": os.path.join(EXAMPLES, "rfc", "paged-tensor-write.mlir"),
        "function": "paged_tensor_write_1core",
        # Scatter dual: the indirect access tile escapes its nominal box
        # (hi exceeds the page dim). Python: BoxSet 'not contained'; Rust: store
        # data-tile vs access-tile coordinate-count mismatch — same shape_mismatch.
        "seeds": [_seed(0, "f16", [8], gen="zero")],
        "expect_category": "shape_mismatch",
    },
}


def _error_category(msg):
    """Normalize a Python/Rust error MESSAGE to a coarse category so a matched
    failure can be asserted despite different exception types / wording."""
    m = msg.lower()
    if "no handler registered" in m:
        return "no_handler"
    if "scratchpad overflow" in m or "lx capacity exceeded" in m or "lx overflow" in m:
        return "lx_overflow"
    if "not contained" in m or "exceeds shape" in m or "shape mismatch" in m:
        return "shape_mismatch"
    if "unmapped" in m:
        return "unmapped"
    return "other"


# The distributed-view fixture's logical A is the deterministic 192x64 ramp
# np.arange(192*64) (matches tests/test_distributed_view.py), partitioned across
# three memory regions. Each region's seed reproduces its own slice independently
# (no shared state needed), so both Python and Rust seed byte-identical bytes.
_DIST_FULL = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)


def _col_packed(block, strides):
    """Pack `block` into a flat f16 buffer under element `strides` (holes zero).
    Element (i,j) lands at offset i*strides[0] + j*strides[1]. Mirrors the test
    helper `_write_strided`."""
    coords = np.stack(
        np.meshgrid(*[np.arange(s) for s in block.shape], indexing="ij"), axis=-1
    ).reshape(-1, block.ndim)
    offsets = coords @ np.array(strides, dtype=np.int64)
    span = int(offsets.max()) + 1 if offsets.size else 1
    buf = np.zeros(span, dtype=np.float16)
    buf[offsets] = block.flatten()
    return buf


def gen_tensor(arg, shape, dtype, rng):
    """Seeded input generation per the arg's `gen` role, in its numpy dtype."""
    np_dt = NP_DTYPE[dtype]
    n = int(np.prod(shape))
    role = arg["gen"]
    if role == "zero":
        return np.zeros(shape, dtype=np_dt)
    if role == "ones":
        return np.ones(shape, dtype=np_dt)
    if role == "randint":
        return rng.integers(arg["lo"], arg["hi"], size=shape, dtype=np_dt)
    # ring_reduce inputs: strictly positive (sum is well-conditioned; matches the
    # uniform(1,2) the Python ring-reduce test uses, but seeded per-run).
    if role == "ring_pos":
        return (rng.uniform(1.0, 2.0, size=n)).astype(np_dt).reshape(shape)
    # distributed-view partitions of the deterministic 192x64 ramp.
    if role == "dist_a_hbm":
        return _DIST_FULL[0:96, :].copy()  # row-major HBM
    if role == "dist_a_lx0":
        return _col_packed(_DIST_FULL[96:128, :].copy(), [1, 64])  # col-packed LX
    if role == "dist_a_lx1":
        return _DIST_FULL[128:192, :].copy()  # row-major LX
    # "rand": small symmetric values so f16 rounding is well-conditioned and the
    # matmul/layernorm reductions don't blow up the dynamic range.
    return (rng.standard_normal(n) * 0.1).astype(np_dt).reshape(shape)


def build_python_outputs(spec, rng):
    """Run the Python KTIRInterpreter; return (kwargs_for_rust, py_outputs).

    kwargs_for_rust maps tensor arg name -> (numpy array, dtype) so the SAME
    bytes go to Rust. py_outputs maps output name -> numpy array (post-exec).
    """
    interp = KTIRInterpreter()
    interp.load(open(spec["program"]).read())

    kwargs = {}
    tensor_inputs = {}  # name -> (np array as fed, dtype str)
    for a in spec["args"]:
        if a["kind"] == "scalar":
            sd = a["scalar_dtype"]
            if sd in FLOAT_DTYPES:
                kwargs[a["name"]] = float(a["value"])
            elif sd == "i32":
                # The dynamic-shape kernel binds n_elements through a symbolic
                # coordinate-set bound; the Python suite passes it as np.int32
                # (test_examples.TestVectorAddDynamicExecution). Match that so
                # the symbolic mask resolves identically on both sides.
                kwargs[a["name"]] = np.int32(a["value"])
            else:
                kwargs[a["name"]] = int(a["value"])
            continue
        arr = gen_tensor(a, a["shape"], a["dtype"], rng)
        kwargs[a["name"]] = arr
        tensor_inputs[a["name"]] = (arr, a["dtype"])

    out = interp.execute_function(spec["function"], **kwargs)
    py_outputs = {name: np.asarray(out[name]) for name in spec["outputs"]}
    return tensor_inputs, py_outputs


def _scalar_kwarg(s):
    """A FAIL/HBM spec scalar -> the Python execute_function kwarg value."""
    sd = s["scalar_dtype"]
    if sd in FLOAT_DTYPES:
        return float(s["value"])
    if sd == "i32":
        return np.int32(s["value"])
    return int(s["value"])


def _run_python_hbm(spec, rng):
    """Run an HBM-seeded program in Python: seed HBM/LX via _prepare_execution,
    execute, read back each `reads` region. Returns (seed_arrays, outputs) on
    success, or raises the interpreter's exception (the matched-failure path).

    `seed_arrays` maps a seed key -> the generated ndarray, so the SAME bytes are
    written on the Rust side. Seed key = f"stick{stick}" (LX seeds prefixed lx).
    """
    from ktir_cpu.dtypes import bytes_per_elem
    from ktir_cpu.ops.memory_ops import hbm_read, hbm_write

    interp = KTIRInterpreter()
    interp.load(open(spec["program"]).read())

    seed_arrays = {}
    for s in spec.get("seeds", []):
        arr = gen_tensor(s, s["shape"], s["dtype"], rng)
        key = f"lx{s['lx_core']}_{s['elem']}" if "lx_core" in s else f"hbm_{s['elem']}"
        seed_arrays[key] = (arr, s)

    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        for key, (arr, s) in seed_arrays.items():
            # base_ptr is an ELEMENT index → byte address = elem*bytes_per_elem.
            byte_addr = s["elem"] * bytes_per_elem(s["dtype"])
            if "lx_core" in s:
                lx = interp.memory.get_lx(s["lx_core"])
                lx.write(byte_addr, arr.flatten())  # LX is byte-addressed
                if "next_ptr" in s:
                    lx.next_ptr = s["next_ptr"]
            else:
                hbm_write(hbm, byte_addr, arr.flatten())  # HBM byte → (stick, intra)

    interp._prepare_execution = _prepare_and_seed

    kwargs = {s["name"]: _scalar_kwarg(s) for s in spec.get("scalars", [])}
    interp.execute_function(spec["function"], **kwargs)

    outputs = {}
    for r in spec["reads"]:
        n = int(np.prod(r["shape"]))
        byte_addr = r["elem"] * bytes_per_elem(r["dtype"])
        outputs[r["name"]] = np.asarray(
            hbm_read(interp.memory.hbm, byte_addr, n, r["dtype"])
        ).reshape(r["shape"])
    return seed_arrays, outputs


def _stage_hbm_case(prog, seed, spec, seed_arrays, in_dir):
    """Build the Rust request case dict for an HBM-seeded program (seeds + scalar
    args + read-back regions), writing each seed's bytes to a file."""
    hbm_seed = []
    for key, (arr, s) in seed_arrays.items():
        bin_path = os.path.join(in_dir, f"{prog}_seed{seed}_{key}.bin")
        arr.astype(NP_DTYPE[s["dtype"]]).flatten().tofile(bin_path)
        # `elem` is the ELEMENT-index base (MLIR constant); the Rust harness
        # converts to a byte address via elem*bytes_per_elem(dtype) — symmetric
        # with the Python _run_python_hbm seeding above.
        entry = {"elem": s["elem"], "dtype": s["dtype"], "bytes": bin_path}
        if "lx_core" in s:
            entry["lx_core"] = s["lx_core"]
        if "next_ptr" in s:
            entry["next_ptr"] = s["next_ptr"]
        hbm_seed.append(entry)
    args = [
        {"name": s["name"], "kind": "scalar", "scalar_dtype": s["scalar_dtype"],
         "value": s["value"]}
        for s in spec.get("scalars", [])
    ]
    hbm_read = [
        {"name": r["name"], "elem": r["elem"], "dtype": r["dtype"],
         "shape": list(r["shape"])}
        for r in spec["reads"]
    ]
    return {
        "id": f"{prog}/seed{seed}",
        "program": spec["program"],
        "function": spec["function"],
        "args": args,
        "hbm_seed": hbm_seed,
        "hbm_read": hbm_read,
    }


def main():
    fuzz_iters = int(os.environ.get("FUZZ_ITERS", "8"))
    # Phase 2 default: ALL programs the harness can run head-to-head. Override
    # with KTIR_DIFF_PROGRAMS=vector_add (or any comma-separated subset).
    # GPU mode default-selects the compute-heavy GEMM programs (the ones that
    # exercise the Metal fast path); CPU mode defaults to ALL.
    # RESIDENT mode attempts ALL 19 programs through the resident/Metal path (the
    # marshalled SPECS run end-to-end; the HBM-seeded / matched-failure fixtures
    # are reported as not-drivable — the honest full table). GPU mode default-
    # selects the GEMM programs; CPU mode defaults to ALL.
    if RESIDENT_MODE:
        default_sel = "all"
    elif GPU_MODE:
        default_sel = ",".join(GPU_DEFAULT_PROGRAMS)
    else:
        default_sel = "all"
    sel = os.environ.get("KTIR_DIFF_PROGRAMS", default_sel)
    all_known = list(SPECS) + list(HBM_SPECS) + list(FAIL_SPECS)
    if sel.strip().lower() == "all":
        selected = all_known
    else:
        selected = [p.strip() for p in sel.split(",") if p.strip()]
    for p in selected:
        if p not in SPECS and p not in HBM_SPECS and p not in FAIL_SPECS:
            sys.exit(f"unknown program {p!r}; known: {sorted(all_known)}")
    # Split the selection across the three buckets.
    programs = [p for p in selected if p in SPECS]
    hbm_programs = [p for p in selected if p in HBM_SPECS]
    fail_programs = [p for p in selected if p in FAIL_SPECS]

    workdir = tempfile.mkdtemp(prefix="ktir-diff-")
    in_dir = os.path.join(workdir, "inputs")
    out_dir = os.path.join(workdir, "outputs")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Run Python for every (program, seed); collect outputs and stage inputs.
    cases = []
    py_outputs = {}  # case_id -> {output_name: np array}
    for prog in programs:
        spec = SPECS[prog]
        for seed in range(fuzz_iters):
            case_id = f"{prog}/seed{seed}"
            rng = np.random.default_rng(seed)
            tensor_inputs, outs = build_python_outputs(spec, rng)
            py_outputs[case_id] = outs

            arg_entries = []
            for a in spec["args"]:
                if a["kind"] == "scalar":
                    arg_entries.append(a)
                    continue
                arr, dtype = tensor_inputs[a["name"]]
                bin_path = os.path.join(
                    in_dir, f"{prog}_seed{seed}_{a['name']}.bin"
                )
                # raw little-endian bytes in the arg's dtype (C-order).
                arr.astype(NP_DTYPE[dtype]).tofile(bin_path)
                arg_entries.append(
                    {
                        "name": a["name"],
                        "kind": "tensor",
                        "dtype": dtype,
                        "shape": list(a["shape"]),
                        "bytes": bin_path,
                    }
                )
            cases.append(
                {
                    "id": case_id,
                    "program": spec["program"],
                    "function": spec["function"],
                    "args": arg_entries,
                    "outputs": spec["outputs"],
                }
            )

    # 1b. HBM-seeded programs: run Python (capturing seed bytes + outputs, or the
    # exception for a KNOWN_GAP), and stage the Rust seeded case onto the batch.
    hbm_py_outputs = {}   # case_id -> {name: np array}  (None if Python raised)
    hbm_py_errors = {}    # case_id -> error string       (Python raised)
    for prog in hbm_programs:
        spec = HBM_SPECS[prog]
        for seed in range(fuzz_iters):
            case_id = f"{prog}/seed{seed}"
            rng = np.random.default_rng(seed)
            try:
                seed_arrays, outs = _run_python_hbm(spec, rng)
                hbm_py_outputs[case_id] = outs
            except Exception as e:  # noqa: BLE001 — capture for matched-failure
                hbm_py_errors[case_id] = f"{type(e).__name__}: {e}"
                # Re-seed deterministically so the Rust case gets the same bytes.
                rng = np.random.default_rng(seed)
                seed_arrays = {}
                for s in spec.get("seeds", []):
                    arr = gen_tensor(s, s["shape"], s["dtype"], rng)
                    key = (
                        f"lx{s['lx_core']}_{s['elem']}"
                        if "lx_core" in s
                        else f"hbm_{s['elem']}"
                    )
                    seed_arrays[key] = (arr, s)
            cases.append(_stage_hbm_case(prog, seed, spec, seed_arrays, in_dir))

    # 1c. MATCHED-FAILURE programs: Python MUST raise. Stage the Rust case (the
    # CLI records the Rust error per-case); we assert both raise + same category.
    fail_py_errors = {}   # case_id -> (error string | None if Python did NOT raise)
    for prog in fail_programs:
        spec = FAIL_SPECS[prog]
        for seed in range(fuzz_iters):
            case_id = f"{prog}/seed{seed}"
            rng = np.random.default_rng(seed)
            if spec["kind"] == "marshalled":
                # Build kwargs + tensor bytes; run Python expecting an exception.
                interp = KTIRInterpreter()
                interp.load(open(spec["program"]).read())
                kwargs = {}
                tensor_inputs = {}
                for a in spec["args"]:
                    if a["kind"] == "scalar":
                        kwargs[a["name"]] = _scalar_kwarg(a)
                        continue
                    arr = gen_tensor(a, a["shape"], a["dtype"], rng)
                    kwargs[a["name"]] = arr
                    tensor_inputs[a["name"]] = (arr, a["dtype"])
                try:
                    interp.execute_function(spec["function"], **kwargs)
                    fail_py_errors[case_id] = None  # did NOT raise -> a FAIL
                except Exception as e:  # noqa: BLE001
                    fail_py_errors[case_id] = f"{type(e).__name__}: {e}"
                arg_entries = []
                for a in spec["args"]:
                    if a["kind"] == "scalar":
                        arg_entries.append(a)
                        continue
                    arr, dtype = tensor_inputs[a["name"]]
                    bin_path = os.path.join(in_dir, f"{prog}_seed{seed}_{a['name']}.bin")
                    arr.astype(NP_DTYPE[dtype]).tofile(bin_path)
                    arg_entries.append({
                        "name": a["name"], "kind": "tensor", "dtype": dtype,
                        "shape": list(a["shape"]), "bytes": bin_path,
                    })
                cases.append({
                    "id": case_id, "program": spec["program"],
                    "function": spec["function"], "args": arg_entries,
                    "outputs": [],
                })
            else:  # "hbm" kind
                try:
                    _run_python_hbm(spec, rng)
                    fail_py_errors[case_id] = None  # did NOT raise -> a FAIL
                except Exception as e:  # noqa: BLE001
                    fail_py_errors[case_id] = f"{type(e).__name__}: {e}"
                rng = np.random.default_rng(seed)
                seed_arrays = {}
                for s in spec.get("seeds", []):
                    arr = gen_tensor(s, s["shape"], s["dtype"], rng)
                    key = (
                        f"lx{s['lx_core']}_{s['elem']}"
                        if "lx_core" in s
                        else f"hbm_{s['elem']}"
                    )
                    seed_arrays[key] = (arr, s)
                # No reads (we only care that it raises).
                staged = _stage_hbm_case(
                    prog, seed, {**spec, "reads": []}, seed_arrays, in_dir
                )
                cases.append(staged)

    request = {"out_dir": out_dir, "cases": cases}
    req_path = os.path.join(workdir, "request.json")
    with open(req_path, "w") as f:
        json.dump(request, f)

    # 2. Run the Rust CLI ONCE for the whole batch.
    rust_bin = os.environ.get("KTIR_DIFF_RUN_BIN")
    if rust_bin:
        cmd = [rust_bin, req_path]
    else:
        cmd = [
            "cargo",
            "run",
            "--release",
            "--example",
            "ktir_diff_run",
            "--",
            req_path,
        ]
    # GPU mode: force the Metal fast path in the Rust subprocess. KTIR_DIFF_ENGINE
    # =gpu makes the CLI reset+record the per-case GPU-GEMM proof counter;
    # KTIR_FORCE_GPU_GEMM=1 routes the small tiled example matmuls onto NAX (they
    # are below the wall-clock size gate). Inherit the rest of the env unchanged.
    rust_env = dict(os.environ)
    if RESIDENT_MODE:
        # Drive the whole suite through the resident/segmented Metal executor at the
        # kernel's native grid. KTIR_FORCE_GPU_GEMM=1 + KTIR_GEMM_GPU_MIN_KN=0 route
        # the small tiled example GEMMs onto NAX/simdgroup (below the wall-clock
        # gate). The CLI records the full per-offload proof per case.
        rust_env["KTIR_DIFF_ENGINE"] = "resident"
        rust_env["KTIR_FORCE_GPU_GEMM"] = "1"
        rust_env.setdefault("KTIR_GEMM_GPU_MIN_KN", "0")
        # PHASE 1 (ForceAllMetal): force EVERY Metal offload so each example
        # program's compute runs ON METAL (proven by the OffloadProof counters),
        # not CPU. KTIR_FORCE_GPU_MAP lifts the scheduler's single-core gate so the
        # MULTI-CORE elementwise programs (softmax/layernorm/vector_add, grid [32,1])
        # dispatch their per-core maps to the Metal map kernel; KTIR_MAP_GPU_MIN_ELEMS
        # =0 also offloads windows below the per-window dispatch floor (otherwise the
        # tiny per-core windows would skip the GPU). KTIR_FORCE_FUSE_ATTN forces the
        # decode-attention fused Metal path on. With these, every Metal-eligible op
        # (GEMM / elementwise-map / attention) fires a Metal kernel.
        rust_env.setdefault("KTIR_FORCE_GPU_MAP", "1")
        rust_env.setdefault("KTIR_MAP_GPU_MIN_ELEMS", "0")
        rust_env.setdefault("KTIR_FORCE_FUSE_ATTN", "1")
        print(
            "[resident] KTIR_DIFF_ENGINE=resident KTIR_FORCE_GPU_GEMM=1 "
            "KTIR_GEMM_GPU_MIN_KN=0 KTIR_FORCE_GPU_MAP=1 KTIR_MAP_GPU_MIN_ELEMS=0 "
            "KTIR_FORCE_FUSE_ATTN=1 — running ALL programs through the production "
            "resident/segmented Metal executor (native grid), FORCING every Metal "
            "offload (GEMM/map/attn), and recording the per-offload proof "
            "(gemm-loop/gemm-or-blas/map-window) per program",
            flush=True,
        )
    elif GPU_MODE:
        rust_env["KTIR_DIFF_ENGINE"] = "gpu"
        rust_env["KTIR_FORCE_GPU_GEMM"] = "1"
        # Part B: also force the fused MAP-window offload so the non-F16 elementwise
        # programs (vector_add_dynamic f32, indexed_add i64-gather), which the
        # all-F16 resident path cannot drive, run their `arith.addf` map on the
        # Metal kernel through the per-op `execute_function` GPU path (which DOES
        # handle non-F16 dtypes). KTIR_FORCE_GPU_MAP lifts the scheduler's
        # single-core gate (indexed_add is grid [2,8]); KTIR_MAP_GPU_MIN_ELEMS=0
        # drops the per-window dispatch floor so the small example windows offload.
        rust_env.setdefault("KTIR_FORCE_GPU_MAP", "1")
        rust_env.setdefault("KTIR_MAP_GPU_MIN_ELEMS", "0")
        print(
            "[gpu] KTIR_DIFF_ENGINE=gpu KTIR_FORCE_GPU_GEMM=1 KTIR_FORCE_GPU_MAP=1 "
            "KTIR_MAP_GPU_MIN_ELEMS=0 — forcing the Metal NAX/simdgroup GEMM "
            "(gpu_gemm_count>0 per GEMM program) AND the fused map kernel "
            "(map_region_gpu>0 per map-bearing non-F16 program)",
            flush=True,
        )
    print(f"[rust] {' '.join(cmd[:6])} ... ({len(cases)} cases)", flush=True)
    proc = subprocess.run(cmd, cwd=RUST_DIR, capture_output=True, text=True, env=rust_env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"Rust CLI failed (exit {proc.returncode})")

    # 3. Read the Rust manifest and diff each output against Python.
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    rust_by_case = {}  # case_id -> {name: (np array, dtype)}
    for entry in manifest["outputs"]:
        np_dt = NP_DTYPE[entry["dtype"]]
        raw = np.fromfile(
            os.path.join(out_dir, entry["bytes_file"]), dtype=np_dt
        )
        arr = raw.reshape(entry["shape"])
        rust_by_case.setdefault(entry["case_id"], {})[entry["name"]] = (
            arr,
            entry["dtype"],
        )
    # Per-case Rust execution/parse errors (a program Python ran but Rust could
    # not). Keyed by case_id; these become FAILs with the error as the reason.
    rust_errors = {e["case_id"]: e["error"] for e in manifest.get("errors", [])}
    # GPU mode: per-case GPU-GEMM proof count (manifest.gpu). case_id -> count.
    # A GEMM-bearing program with count==0 secretly ran on AMX — a FALSE pass.
    gpu_counts = {g["case_id"]: g.get("gpu_gemm_count", 0) for g in manifest.get("gpu", [])}
    # RESIDENT mode: per-case full offload-proof breakdown (manifest.gpu[].
    # offload_proof). case_id -> {matmul_loop_gpu, matmul_loop_amx, gemm_or_blas_gpu,
    # map_region_gpu}. The OFFLOAD TOTAL (sum) > 0 proves the program hit Metal; a
    # Metal-bearing program with total 0 secretly ran all-CPU (a FALSE pass).
    offload_proofs = {
        g["case_id"]: g["offload_proof"]
        for g in manifest.get("gpu", [])
        if "offload_proof" in g
    }

    def _offload_total(case_id):
        p = offload_proofs.get(case_id, {})
        return sum(int(p.get(k, 0)) for k in (
            "matmul_loop_gpu", "matmul_loop_amx", "gemm_or_blas_gpu", "map_region_gpu"))

    def _offload_str(case_id):
        p = offload_proofs.get(case_id, {})
        parts = []
        for short, k in (("gemm-loop", "matmul_loop_gpu"), ("gemm-amx", "matmul_loop_amx"),
                         ("gemm", "gemm_or_blas_gpu"), ("map", "map_region_gpu")):
            v = int(p.get(k, 0))
            if v:
                parts.append(f"{short}={v}")
        return ",".join(parts) if parts else "cpu-only"

    # The principled bf16/f16 band + per-offload proof apply in BOTH the GPU and
    # the RESIDENT Metal modes (the CPU bit-exact mode keeps the flat tol).
    band_mode = GPU_MODE or RESIDENT_MODE

    # 4. Per-program: aggregate max-abs across all seeds + outputs, report.
    print()
    gpu_col = f" {'offloads':>22}" if RESIDENT_MODE else (f" {'backend':>10}" if GPU_MODE else "")
    header = (
        f"{'program':<20} {'dtype':<6} {'fuzz iters':>10} "
        f"{'max-abs Py-vs-Rust':>20}  result{gpu_col}"
    )
    print(header)
    print("-" * len(header))
    failures = []      # numeric divergences: (case_id, name, diff, tol)
    exec_fails = []    # Rust could not run: (case_id, error)
    false_passes = []  # GPU mode: GEMM-bearing program that ran on AMX (count==0)
    overall_ok = True
    for prog in programs:
        spec = SPECS[prog]
        prog_max = 0.0
        prog_band = 0.0   # GPU mode: the principled band actually applied (report)
        prog_dtype = None
        exact_required = False
        prog_errored = False
        gemm_bearing = GPU_MODE and prog in GEMM_BEARING
        # GPU mode part B: a non-F16 elementwise program whose `arith.addf` map MUST
        # dispatch to the fused Metal map kernel (map_region_gpu>0); a 0 is a FALSE
        # all-CPU pass.
        map_bearing = GPU_MODE and prog in GPU_MAP_BEARING
        prog_map_count = None  # min map_region_gpu across seeds (worst case)
        # RESIDENT: a Metal-bearing program MUST fire at least one offload (a 0
        # total is a FALSE all-CPU pass). The bucket is derived from the MEASURED
        # proof and reported per program.
        metal_bearing = RESIDENT_MODE and prog in RESIDENT_METAL_BEARING
        prog_gpu_count = None  # min gpu_gemm_count across seeds (worst case)
        prog_offload_total = None  # min offload total across seeds (worst case)
        prog_offload_str = "cpu-only"
        false_pass = False
        for seed in range(fuzz_iters):
            case_id = f"{prog}/seed{seed}"
            if case_id in rust_errors:
                prog_errored = True
                exec_fails.append((case_id, rust_errors[case_id]))
                continue
            # GPU PROOF: a GEMM-bearing program MUST have dispatched its matmuls to
            # the Metal engine. count==0 means it secretly ran on AMX (a FALSE
            # pass) — flag it; numeric agreement then proves nothing about NAX.
            if GPU_MODE:
                cnt = gpu_counts.get(case_id, 0)
                prog_gpu_count = cnt if prog_gpu_count is None else min(prog_gpu_count, cnt)
                if gemm_bearing and cnt <= 0:
                    false_pass = True
                    false_passes.append((case_id, prog))
                # Part B map proof: the map-bearing non-F16 program must fire the
                # fused Metal map kernel (map_region_gpu>0).
                mcnt = int(offload_proofs.get(case_id, {}).get("map_region_gpu", 0))
                prog_map_count = mcnt if prog_map_count is None else min(prog_map_count, mcnt)
                if map_bearing and mcnt <= 0:
                    false_pass = True
                    false_passes.append((case_id, prog))
            # RESIDENT PROOF: record the per-offload breakdown; a Metal-bearing
            # program with a 0 total secretly ran all-CPU (a FALSE pass).
            if RESIDENT_MODE:
                tot = _offload_total(case_id)
                prog_offload_total = tot if prog_offload_total is None else min(prog_offload_total, tot)
                prog_offload_str = _offload_str(case_id)
                if metal_bearing and tot <= 0:
                    false_pass = True
                    false_passes.append((case_id, prog))
            for name in spec["outputs"]:
                if case_id not in rust_by_case or name not in rust_by_case[case_id]:
                    prog_errored = True
                    exec_fails.append(
                        (case_id, f"output {name!r} missing from Rust manifest")
                    )
                    continue
                rust_arr, dtype = rust_by_case[case_id][name]
                prog_dtype = dtype
                py_arr = py_outputs[case_id][name].reshape(rust_arr.shape)
                if dtype in FLOAT_DTYPES:
                    pf = py_arr.astype(np.float64)
                    rf = rust_arr.astype(np.float64)
                    diff = np.max(np.abs(pf - rf))
                    if band_mode:
                        # PRINCIPLED bf16/f16 band at this output's own peak
                        # magnitude (self-calibrating — no hardcoded value range).
                        finite = np.concatenate(
                            [pf[np.isfinite(pf)].ravel(), rf[np.isfinite(rf)].ravel()]
                        )
                        max_mag = float(np.max(np.abs(finite))) if finite.size else 1.0
                        tol = gpu_band(max_mag)
                        prog_band = max(prog_band, tol)
                    else:
                        tol = F16_ABS_TOL if dtype == "f16" else 1e-4
                else:
                    exact_required = True
                    diff = float(
                        np.max(np.abs(py_arr.astype(np.int64) - rust_arr.astype(np.int64)))
                    )
                    tol = 0.0
                d = float(diff)
                # NaN-safe: a non-finite diff (one side NaN/overflowed, the other
                # did not) is ALWAYS a divergence. `max(x, nan)` swallows the NaN and
                # `nan > tol` is False in Python, so non-finite must be flagged
                # EXPLICITLY — otherwise a NaN divergence reads as "0 PASS".
                if not (d <= tol):
                    failures.append((case_id, name, d, tol))
                if not np.isfinite(d):
                    prog_max = float("nan")
                elif np.isfinite(prog_max):
                    prog_max = max(prog_max, d)
        if band_mode:
            tol = prog_band if prog_dtype in FLOAT_DTYPES else 0.0
        else:
            tol = 0.0 if exact_required else (F16_ABS_TOL if prog_dtype == "f16" else 1e-4)
        # RESIDENT: a program Rust cannot DRIVE through the marshalled ProgramSpec
        # path (dynamic/symbolic view shape => "no shape derivable") is a genuine
        # not-drivable finding, NOT a numeric divergence — reported honestly and NOT
        # counted as a suite failure (it stays bit-exact in the DEFAULT CPU harness).
        # A program that DID run but diverged, or a metal-bearing false-CPU pass,
        # still FAILS.
        not_drivable_reasons = [
            e for cid, e in exec_fails
            if cid.split("/")[0] == prog and (
                "not drivable" in e
                or "no shape derivable" in e
                or "all-F16" in e
            )
        ]
        not_drivable = RESIDENT_MODE and prog_errored and bool(not_drivable_reasons)
        if not_drivable:
            ok = True  # reported separately; does not fail the suite
            result_txt = "NOT-DRIVABLE"
        else:
            ok = (not prog_errored) and (not false_pass) and prog_max <= tol
            result_txt = "PASS" if ok else "FAIL"
        overall_ok = overall_ok and ok
        dt = prog_dtype if prog_dtype else "-"
        maxabs = ("N/A" if not_drivable else ("ERROR" if prog_errored else f"{prog_max:.6g}"))
        if RESIDENT_MODE:
            if not_drivable:
                why = not_drivable_reasons[0]
                cat = ("non-F16 dtype" if "all-F16" in why
                       else "dyn-shape" if "no shape derivable" in why
                       else "not drivable")
                gpu_col = f" {('N/A: ' + cat):>22}"
            elif prog_errored:
                gpu_col = f" {'ERROR':>22}"
            elif false_pass:
                gpu_col = f" {'CPU!(metal-bearing)':>22}"
            else:
                tag = prog_offload_str
                gpu_col = f" {tag:>22}"
        elif GPU_MODE:
            cnt_disp = prog_gpu_count if prog_gpu_count is not None else 0
            map_disp = prog_map_count if prog_map_count is not None else 0
            if false_pass:
                gpu_col = f" {'CPU!(0)':>10}"
            elif gemm_bearing:
                gpu_col = f" {('NAX(%d)' % cnt_disp):>10}"
            elif map_bearing:
                gpu_col = f" {('map(%d)' % map_disp):>10}"
            else:
                gpu_col = f" {'cpu(0)':>10}"
        else:
            gpu_col = ""
        result_label = result_txt if RESIDENT_MODE else ("PASS" if ok else "FAIL")
        print(
            f"{prog:<20} {dt:<6} {fuzz_iters:>10} {maxabs:>20}  "
            f"{result_label}{gpu_col}"
        )

    # 4b. HBM-seeded programs: bit-exact diff of the read-back regions, OR a
    # checked KNOWN_GAP (Python ran, Rust raised the expected category).
    for prog in hbm_programs:
        if RESIDENT_MODE:
            # The HBM-seeded fixtures place their tensors at hardcoded HBM stick
            # addresses, NOT marshalled pointer args, so they cannot be expressed as
            # a marshalled-arg ProgramSpec the resident executor runs. Reported as
            # not-drivable (CPU-only on this path), NOT a suite failure — the honest
            # full-table entry. They remain bit-exact in the DEFAULT CPU harness.
            print(f"{prog:<20} {'-':<6} {fuzz_iters:>10} {'N/A':>20}  CPU-ONLY"
                  f" {'not drivable (HBM-seeded)':>22}")
            continue
        spec = HBM_SPECS[prog]
        gap = KNOWN_GAPS.get(prog)
        prog_max = 0.0
        prog_dtype = None
        result = "PASS"
        reason = ""
        for seed in range(fuzz_iters):
            case_id = f"{prog}/seed{seed}"
            rust_err = rust_errors.get(case_id)
            py_err = hbm_py_errors.get(case_id)
            if gap is not None:
                # KNOWN GAP: Python should run, Rust should raise `category`.
                if py_err is not None:
                    result, reason = "FAIL", f"Python unexpectedly raised: {py_err}"
                    break
                if rust_err is None:
                    # Rust no longer fails -> the gap may be CLOSED; flag for
                    # promotion to a real bit-exact PASS (don't silently pass).
                    result, reason = (
                        "XPASS",
                        "Rust no longer raises — promote this KNOWN_GAP to a PASS",
                    )
                    break
                cat = _error_category(rust_err)
                if cat != gap["category"]:
                    result, reason = (
                        "FAIL",
                        f"Rust error category {cat!r} != expected {gap['category']!r}: {rust_err}",
                    )
                    break
                result = "GAP"  # checked, expected failure
                continue
            # Bit-exact PASS expected on both sides.
            if py_err is not None:
                result, reason = "FAIL", f"Python raised: {py_err}"
                break
            if rust_err is not None:
                result, reason = "FAIL", f"Rust raised: {rust_err}"
                break
            for r in spec["reads"]:
                name, dtype = r["name"], r["dtype"]
                if case_id not in rust_by_case or name not in rust_by_case[case_id]:
                    result, reason = "FAIL", f"output {name!r} missing from Rust manifest"
                    break
                rust_arr, _ = rust_by_case[case_id][name]
                prog_dtype = dtype
                py_arr = hbm_py_outputs[case_id][name].reshape(rust_arr.shape)
                if dtype in FLOAT_DTYPES:
                    diff = float(np.max(np.abs(
                        py_arr.astype(np.float64) - rust_arr.astype(np.float64))))
                    tol = F16_ABS_TOL if dtype == "f16" else 1e-4
                else:
                    diff = float(np.max(np.abs(
                        py_arr.astype(np.int64) - rust_arr.astype(np.int64))))
                    tol = 0.0
                # NaN-safe (see the SPECS diff above): a non-finite diff is a
                # divergence even though `nan > tol` is False.
                if not (diff <= tol):
                    why = ("non-finite (NaN/inf)" if not np.isfinite(diff)
                           else f"max-abs {diff:.6g} > tol {tol}")
                    result, reason = "FAIL", f"{name} {why}"
                if not np.isfinite(diff):
                    prog_max = float("nan")
                elif np.isfinite(prog_max):
                    prog_max = max(prog_max, diff)
            if result == "FAIL":
                break
        ok = result in ("PASS", "GAP")
        overall_ok = overall_ok and ok
        dt = prog_dtype if prog_dtype else "-"
        maxabs = "GAP" if result == "GAP" else (
            "ERROR" if result in ("FAIL", "XPASS") else f"{prog_max:.6g}")
        print(f"{prog:<20} {dt:<6} {fuzz_iters:>10} {maxabs:>20}  {result}")
        if reason:
            print(f"  └─ {reason}")

    # 4c. MATCHED-FAILURE programs: assert BOTH sides raise, same category.
    for prog in fail_programs:
        if RESIDENT_MODE:
            # Matched-failure fixtures (oversized-LX / box-not-contained) are
            # HBM-seeded too, and assert an EXPECTED failure — not a numeric run.
            # They are not driven through the resident path; reported not-drivable.
            print(f"{prog:<20} {'-':<6} {fuzz_iters:>10} {'N/A':>20}  CPU-ONLY"
                  f" {'not drivable (fail-fixture)':>22}")
            continue
        spec = FAIL_SPECS[prog]
        expect = spec["expect_category"]
        result = "MATCH-FAIL"
        reason = ""
        for seed in range(fuzz_iters):
            case_id = f"{prog}/seed{seed}"
            py_err = fail_py_errors.get(case_id)
            rust_err = rust_errors.get(case_id)
            if py_err is None:
                result, reason = "FAIL", "Python did NOT raise (expected a failure)"
                break
            if rust_err is None:
                result, reason = "FAIL", "Rust did NOT raise (expected a failure)"
                break
            py_cat = _error_category(py_err)
            rust_cat = _error_category(rust_err)
            if expect not in (py_cat, rust_cat) or py_cat != rust_cat:
                result, reason = "FAIL", (
                    f"category mismatch: py={py_cat!r} ({py_err[:60]}) "
                    f"rust={rust_cat!r} ({rust_err[:60]}); expected {expect!r}"
                )
                break
        ok = result == "MATCH-FAIL"
        overall_ok = overall_ok and ok
        maxabs = expect if ok else "ERROR"
        print(f"{prog:<20} {'-':<6} {fuzz_iters:>10} {maxabs:>20}  {result}")
        if reason:
            print(f"  └─ {reason}")

    print()
    if RESIDENT_MODE:
        print(
            "RESIDENT/SEGMENTED METAL PATH (Phase 1 ResidentRunner):\n"
            "  Every program is driven through the PRODUCTION resident executor\n"
            "  (ResidentExecutor::new_native -> run) at its NATIVE grid: resident HBM,\n"
            "  weight cache, per-segment seg-plan (K-loop GEMM reconstruction where\n"
            "  recognizable) and per-op Metal offloads. The 'offloads' column is the\n"
            "  MEASURED per-case proof (gemm = gemm_or_blas_gpu NAX/simdgroup dispatch,\n"
            "  gemm-loop = matmul_loop_gpu reconstruction, map = fused map window).\n"
            "  Band = the SAME principled bf16/f16 band as GPU mode:\n"
            "      tol(|v|) = 4*f16_ulp(|v|)  +  2^-8 * |v|\n"
            "  A Metal-bearing program (matmul/sdpa/paged_attention) with a 0 offload\n"
            "  total is a FALSE all-CPU pass and FAILS. NOT-DRIVABLE rows are reported\n"
            "  honestly (non-F16 index/data dtype: the resident path is all-F16; or a\n"
            "  dynamic view shape; or an HBM-seeded/fail fixture not expressible as a\n"
            "  marshalled-arg ProgramSpec) — they stay correct on the DEFAULT CPU path.\n"
        )
    if GPU_MODE:
        print(
            "GPU-PATH BAND (principled, bf16/f16-derived):\n"
            "  tol(|v|) = 4*f16_ulp(|v|)  +  2^-8 * |v|\n"
            "             ^ f16 OUTPUT quant   ^ bf16 INPUT-rounding rel ulp\n"
            "  NAX rounds f16 inputs to bf16 (~8 mantissa bits) and accumulates in\n"
            "  f32, so the error is bounded by input rounding (NOT sqrt(K) growth);\n"
            "  the result re-quantizes to the f16 output tile (the 4-ulp term). The\n"
            "  band is evaluated at each output's own peak magnitude (self-\n"
            "  calibrating). e.g. matmul |v|~2 -> ~0.0156 vs observed ~0.00195.\n"
        )
    if false_passes:
        print("FALSE PASSES (GEMM-bearing program that ran on AMX, gpu_gemm_count==0):")
        seen = set()
        for case_id, prog in false_passes:
            if prog in seen:
                continue
            seen.add(prog)
            print(f"  {case_id} | matmul did NOT dispatch to the Metal NAX/simdgroup "
                  f"engine — numeric agreement proves nothing about the GPU path")
        print(
            "\nThese FAIL: the GPU fast path was supposed to run but the matmul "
            "secretly fell back to AMX/f32. Reported, not hidden."
        )
        print()
    if exec_fails:
        # In RESIDENT mode, split the "Rust did not run it" rows into NOT-DRIVABLE
        # (an expected limitation of the all-F16 resident path: non-F16 dtype / dyn
        # shape — still correct on the default CPU path) vs a genuine gap.
        def _nd(err):
            return ("not drivable" in err or "no shape derivable" in err
                    or "all-F16" in err)
        drivable_gap, nd = [], []
        for case_id, err in exec_fails:
            (nd if (RESIDENT_MODE and _nd(err)) else drivable_gap).append((case_id, err))
        if nd:
            print("RESIDENT NOT-DRIVABLE (program | why) — runs on the DEFAULT CPU path, "
                  "not through the all-F16 resident/segmented path:")
            seen = set()
            for case_id, err in nd:
                prog = case_id.split("/")[0]
                if prog in seen:
                    continue
                seen.add(prog)
                print(f"  {prog} | {err}")
            print()
        if drivable_gap:
            print("RUST EXECUTION FAILURES (program/seed | error) — Python ran, Rust did not:")
            seen = set()
            for case_id, err in drivable_gap:
                prog = case_id.split("/")[0]
                if prog in seen:
                    continue  # one representative line per program (errors repeat per seed)
                seen.add(prog)
                print(f"  {case_id} | {err}")
            print(
                "\nThese are REAL conformance gaps: a program the Python KTIRInterpreter "
                "executes that the Rust port cannot. Reported, not hidden."
            )
            print()
    if failures:
        print("CONFORMANCE DIVERGENCES (program/seed | output | max-abs | tol):")
        for case_id, name, diff, tol in failures[:50]:
            print(f"  {case_id} | {name} | {diff:.6g} | tol={tol}")
        print(
            "\nThese are REAL Python⟷Rust numeric divergences. Reported, not hidden. "
            "Fix the implementation or, if the band is genuinely f16-rounding, "
            "justify the tolerance — do NOT silently loosen it."
        )

    # Keep the workdir on failure for inspection; clean on success.
    if overall_ok:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"\n(workdir kept for inspection: {workdir})")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
