// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! RUST-ONLY (not a port of a Python test): end-to-end execution of the
//! `layernorm_fwd_ktir.mlir` example fixture, which exercises the `%x2 =
//! arith.mulf %x, %x` variance squaring that the operand-dedup bug silently
//! broke (the fixture had no Rust test). Validates the kernel-stored `Mean` and
//! `Rstd` against a reference, and that `Y` is finite — proving the real
//! RMSNorm/LayerNorm path runs after the dedup fix.

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ir::Scalar;
use ktir_emulator::parser::parse_module;

const SRC: &str = include_str!("../../../../examples/triton-ktir/layernorm_fwd_ktir.mlir");
const ROWS: usize = 1151;
const COLS: usize = 8192; // N
const EPS: f32 = 1e-5;

fn f16(x: f32) -> f32 {
    ktir_emulator::codec::f16_bits_to_f32(ktir_emulator::codec::f32_to_f16_bits(x))
}

#[test]
fn layernorm_fixture_runs_end_to_end() {
    let module = parse_module(SRC).expect("parse layernorm");

    // Small f16 inputs; weight = 1, bias = 0 so Y = (X - mean) * rstd.
    let x: Vec<f32> = (0..ROWS * COLS)
        .map(|i| f16(((i % 17) as f32 - 8.0) * 0.01))
        .collect();
    let w = vec![1.0f32; ROWS * COLS];
    let b = vec![0.0f32; ROWS * COLS];
    let zeros_big = vec![0.0f32; ROWS * COLS];
    let zeros_vec = vec![0.0f32; ROWS];

    let big = |data: Vec<f32>| Arg::Tensor {
        data,
        shape: vec![ROWS, COLS],
        dtype: DType::F16,
    };
    let vec1 = |data: Vec<f32>| Arg::Tensor {
        data,
        shape: vec![ROWS],
        dtype: DType::F16,
    };
    let args: Vec<(&str, Arg)> = vec![
        ("X", big(x.clone())),
        ("Y", big(zeros_big)),
        ("W", big(w)),
        ("B", big(b)),
        ("Mean", vec1(zeros_vec.clone())),
        ("Rstd", vec1(zeros_vec)),
        ("N", Arg::Scalar(Scalar::I64(COLS as i64))),
        ("eps", Arg::Scalar(Scalar::F32(EPS))),
        ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(1024))),
    ];

    let out =
        execute_function(&module, "_layer_norm_fwd_fused", &args).expect("run layernorm fixture");
    let mean = &out.get("Mean").expect("Mean output").data;
    let rstd = &out.get("Rstd").expect("Rstd output").data;
    let y = &out.get("Y").expect("Y output").data;

    assert!(y.iter().all(|v| v.is_finite()), "Y must be finite");

    // Reference per row: mean = E[X], var = E[X²], rstd = 1/sqrt(var+eps).
    // (This kernel's variance is the second moment E[X²], matching line 91-104.)
    let mut max_mean_err = 0.0f32;
    let mut max_rstd_err = 0.0f32;
    for r in 0..ROWS {
        let row = &x[r * COLS..(r + 1) * COLS];
        let m: f32 = row.iter().sum::<f32>() / COLS as f32;
        let v: f32 = row.iter().map(|&e| e * e).sum::<f32>() / COLS as f32;
        let rs = 1.0 / (v + EPS).sqrt();
        max_mean_err = max_mean_err.max((mean[r] - m).abs() / m.abs().max(1e-3));
        max_rstd_err = max_rstd_err.max((rstd[r] - rs).abs() / rs.abs().max(1e-3));
    }
    // Generous f16 tolerance: 8192-wide reductions accumulate rounding.
    assert!(
        max_mean_err < 0.1,
        "Mean max rel err {max_mean_err} too large"
    );
    assert!(
        max_rstd_err < 0.1,
        "Rstd max rel err {max_rstd_err} too large"
    );
    eprintln!(
        "e2e layernorm ({ROWS}×{COLS}, grid 32) ran — Mean err {max_mean_err:.4}, Rstd err {max_rstd_err:.4} ✓"
    );
}
