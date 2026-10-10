// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! GATED Metal differential conformance — the example programs (matmul, sdpa,
//! softmax, layernorm, vector_add, reduce_generic, paged_attention, ...) run
//! through the PRODUCTION Metal fast path (resident/segmented executor + the
//! per-op NAX/simdgroup GEMM selector) and are diffed HEAD-TO-HEAD against the
//! Python `ktir_cpu.KTIRInterpreter` reference under a principled bf16/f16 band.
//!
//! This is the cargo-test wrapper around `tests/equiv/diff_py_vs_rust.py`: where
//! the Python driver is the engine (it generates seeded inputs, runs Python, hands
//! the SAME bytes to the Rust CLI, diffs the outputs and asserts the per-kernel
//! offload proof), this test makes that conformance a REAL GATED check that runs
//! under `cargo test` on every relevant change — not just in CI. It:
//!
//!   1. (positive gate) runs the FULL example suite through the resident/segmented
//!      Metal executor (`KTIR_DIFF_RESIDENT=1`, forcing EVERY Metal offload:
//!      `KTIR_FORCE_GPU_GEMM` + `KTIR_FORCE_GPU_MAP` + `KTIR_MAP_GPU_MIN_ELEMS=0` +
//!      `KTIR_FORCE_FUSE_ATTN`) and asserts the driver exits 0 — i.e. every
//!      Metal-bearing program (vector_add map / matmul / sdpa GEMM) fired its
//!      offload (proof > 0) AND matched Python within band;
//!   2. (positive gate) runs the GEMM-bearing programs through the per-op
//!      `execute_function` GPU selector (`KTIR_DIFF_GPU=1`) so paged_attention —
//!      not drivable on the all-F16 resident path because of its i32 block_tables —
//!      is ALSO proven on NAX (`gpu_gemm_count > 0`) within band;
//!   3. (NEGATIVE CONTROL) re-runs the resident gate with a +5% perturbation
//!      injected into the Metal GEMM output (`KTIR_DIFF_INJECT_DIVERGENCE=0.05`,
//!      well outside the band) and asserts the driver exits NON-ZERO — proving the
//!      band/offload-proof actually catches a real Metal-path divergence rather
//!      than silently swallowing garbage. A green positive gate is worthless if the
//!      negative control does not fail.
//!
//! It is `#![cfg(metal)]`: on Linux there is no Metal device (and the Linux CI runs
//! the SAME driver bit-exact on the CPU path), so the test compiles out there
//! rather than spuriously skipping. It needs `uv` (the Python reference is
//! numpy-only — no torch / weights / MLIR build) and is `#[ignore]` by default so
//! `cargo test` stays self-contained; run it explicitly:
//!
//!   cargo test --release -p ktir-emulator --test metal_conformance -- --ignored --nocapture
//!
//! On a machine without `uv` it reports that and skips (it does NOT fail) — the
//! conformance is exercised in macOS CI regardless (see rust-conformance.yml).

#![cfg(metal)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// Repo root: the crate is at `<root>/rust/crates/ktir-emulator`, so the root is
/// three parents up from `CARGO_MANIFEST_DIR`'s `rust/` (four up from the crate).
/// The driver itself re-derives the root (it walks up for `ktir_cpu/` + `examples/`),
/// but we need it here to locate `uv`'s working corpus and the driver script.
fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/rust/crates/ktir-emulator
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root is 3 ancestors above the crate manifest dir")
        .to_path_buf()
}

/// `<root>/rust` — the cargo workspace the diff CLI lives in (and the dir the
/// driver `cwd`s into to invoke it).
fn rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("rust/ is 2 ancestors above the crate manifest dir")
        .to_path_buf()
}

/// The Python differential driver.
fn driver_py() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/equiv/diff_py_vs_rust.py")
}

/// The prebuilt `ktir_diff_run` example binary, building it once if absent so the
/// driver skips its own (nested) `cargo run` and just invokes the binary. We build
/// the example out-of-band (a plain `cargo build`, NOT from inside this test's
/// `cargo test` target lock) before returning its path.
fn diff_run_bin() -> PathBuf {
    let bin = rust_dir()
        .join("target")
        .join("release")
        .join("examples")
        .join("ktir_diff_run");
    if !bin.is_file() {
        let status = Command::new(env!("CARGO"))
            .current_dir(rust_dir())
            .args([
                "build",
                "--release",
                "--example",
                "ktir_diff_run",
                "-p",
                "ktir-emulator",
            ])
            .status()
            .expect("spawn cargo build --example ktir_diff_run");
        assert!(status.success(), "cargo build of ktir_diff_run failed");
    }
    assert!(
        bin.is_file(),
        "ktir_diff_run example binary not found at {bin:?} after build"
    );
    bin
}

/// Whether `uv` (the Python reference launcher) is on PATH. The reference is
/// numpy-only via `uv run --with numpy`; with no `uv` the conformance can't run
/// locally, so we SKIP (not fail) — CI provides `uv` on the macOS runner.
fn have_uv() -> bool {
    Command::new("uv")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run the Python differential driver with the given extra env, returning whether
/// it exited 0. Inherits the parent env, pins `KTIR_DIFF_RUN_BIN` to the prebuilt
/// example (so no nested cargo build), and streams its output (the per-program
/// PASS/FAIL + offload-proof table) so `--nocapture` shows exactly which kernels
/// fired and the max-abs vs Python.
fn run_driver(extra_env: &[(&str, &str)]) -> bool {
    let mut cmd = Command::new("uv");
    cmd.current_dir(repo_root())
        .args(["run", "--with", "numpy"])
        .arg(driver_py())
        .env("KTIR_DIFF_RUN_BIN", diff_run_bin());
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let status = cmd.status().expect("spawn uv run diff_py_vs_rust.py");
    status.success()
}

/// FULL-SUITE positive gate: every example program through the resident/segmented
/// Metal executor, forcing every offload, banded vs Python, with the per-kernel
/// offload proof MANDATORY (a Metal-bearing program firing 0 offloads is a FALSE
/// all-CPU pass and the driver FAILS it). PLUS the per-op GPU selector over the
/// GEMM-bearing programs so paged_attention (i32 block_tables, not drivable on the
/// all-F16 resident path) is proven on NAX too. PLUS the negative control.
#[test]
#[ignore = "needs uv + a Metal GPU; run with --release --ignored --nocapture"]
fn metal_differential_conformance() {
    if !have_uv() {
        eprintln!(
            "metal_conformance: `uv` not found on PATH — SKIPPING the Python<->Rust \
             Metal differential (it is exercised in macOS CI). Install uv to run locally."
        );
        return;
    }

    // Modest fuzz count keeps the gated test fast; CI runs more seeds. Each seed is
    // one extra execute through both interpreters.
    let fuzz = "3";

    // 1. RESIDENT positive gate — full suite, force ALL Metal offloads. The driver
    //    asserts proof>0 for the Metal-bearing set (vector_add/matmul/sdpa) and band.
    let resident_ok = run_driver(&[
        ("KTIR_DIFF_RESIDENT", "1"),
        ("KTIR_DIFF_PROGRAMS", "all"),
        ("FUZZ_ITERS", fuzz),
    ]);
    assert!(
        resident_ok,
        "RESIDENT/Metal differential FAILED: a program either diverged beyond the \
         principled bf16/f16 band or a Metal-bearing program fired 0 offloads (a \
         FALSE all-CPU pass). See the per-program table above."
    );

    // 2. GPU per-op gate — the GEMM-bearing programs incl. paged_attention, proven
    //    on NAX (gpu_gemm_count>0) within band.
    let gpu_ok = run_driver(&[
        ("KTIR_DIFF_GPU", "1"),
        ("KTIR_DIFF_PROGRAMS", "matmul,sdpa,paged_attention"),
        ("FUZZ_ITERS", fuzz),
    ]);
    assert!(
        gpu_ok,
        "GPU per-op differential FAILED: a GEMM-bearing program diverged beyond the \
         band, or secretly ran on AMX (gpu_gemm_count==0 — a FALSE pass)."
    );

    // 3. NEGATIVE CONTROL — inject a +5% perturbation into the Metal GEMM output
    //    (well outside the band) and require the resident differential to FAIL. This
    //    proves the gate is LIVE: matmul/sdpa must diverge WITH their offload proof
    //    still > 0 (a real Metal-path divergence, not a CPU fallback). If this
    //    PASSED, the band would be swallowing garbage and gate (1) is worthless.
    let injected_ok = run_driver(&[
        ("KTIR_DIFF_RESIDENT", "1"),
        ("KTIR_DIFF_PROGRAMS", "matmul,sdpa,vector_add"),
        ("KTIR_DIFF_INJECT_DIVERGENCE", "0.05"),
        ("FUZZ_ITERS", "2"),
    ]);
    assert!(
        !injected_ok,
        "NEGATIVE CONTROL DID NOT FIRE: the resident differential PASSED with a +5% \
         injected Metal-GEMM divergence — the band/offload-proof is NOT catching a \
         real Metal-output divergence, so the positive gate is vacuous."
    );
}
