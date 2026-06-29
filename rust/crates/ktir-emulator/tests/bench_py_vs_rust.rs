// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! RUST-ONLY benchmark harness for the Python-vs-Rust comparison. Times ONLY
//! `execute_function` (parse/load excluded, one warm-up run excluded), matching
//! the Python `bench_py_vs_rust.py` methodology and inputs. Run with:
//!   cargo test --release --test bench_py_vs_rust <name> -- --ignored --nocapture
//! NOT COMMITTED — review-only.

use std::time::Instant;

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ir::Scalar;
use ktir_emulator::parser::parse_module;

const VECTOR_ADD: &str = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
const MATMUL: &str = include_str!("../../../../examples/triton-ktir/matmul_fwd_ktir.mlir");
const LAYERNORM: &str = include_str!("../../../../examples/triton-ktir/layernorm_fwd_ktir.mlir");

fn f16(x: f32) -> f32 {
    ktir_emulator::codec::f16_bits_to_f32(ktir_emulator::codec::f32_to_f16_bits(x))
}

fn time_call<F: Fn()>(iters: usize, f: F) -> f64 {
    f(); // warm-up
    let t0 = Instant::now();
    for _ in 0..iters {
        f(); // f() executes a kernel (side-effecting), so it is not elided
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn bench_vector_add() {
    let module = parse_module(VECTOR_ADD).expect("parse vector_add");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| f16((i % 13) as f32 * 0.1)).collect();
    let y: Vec<f32> = (0..n).map(|i| f16((i % 11) as f32 * 0.1)).collect();
    let iters = 500;
    let each = time_call(iters, || {
        let args = [
            (
                "x_ptr",
                Arg::Tensor {
                    data: x.clone(),
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            (
                "y_ptr",
                Arg::Tensor {
                    data: y.clone(),
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            (
                "output_ptr",
                Arg::Tensor {
                    data: vec![0.0; n],
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
        ];
        execute_function(&module, "add_kernel", &args).unwrap();
    });
    eprintln!(
        "vector_add: {:.1} µs/run (n=4096 f16, grid[32], {iters} iters)",
        each * 1e6
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn bench_matmul() {
    let module = parse_module(MATMUL).expect("parse matmul");
    let (m, k, n) = (64usize, 2048usize, 8192usize);
    let a: Vec<f32> = (0..m * k).map(|i| f16((i % 13) as f32 * 0.01)).collect();
    let b: Vec<f32> = (0..k * n).map(|i| f16((i % 11) as f32 * 0.01)).collect();
    let iters = 20;
    let each = time_call(iters, || {
        let args = [
            (
                "a_ptr",
                Arg::Tensor {
                    data: a.clone(),
                    shape: vec![m, k],
                    dtype: DType::F16,
                },
            ),
            (
                "b_ptr",
                Arg::Tensor {
                    data: b.clone(),
                    shape: vec![k, n],
                    dtype: DType::F16,
                },
            ),
            (
                "c_ptr",
                Arg::Tensor {
                    data: vec![0.0; m * n],
                    shape: vec![m, n],
                    dtype: DType::F16,
                },
            ),
            ("K", Arg::Scalar(Scalar::I64(k as i64))),
            ("BLOCK_SIZE_M", Arg::Scalar(Scalar::I64(32))),
            ("BLOCK_SIZE_N", Arg::Scalar(Scalar::I64(512))),
            ("BLOCK_SIZE_K", Arg::Scalar(Scalar::I64(128))),
        ];
        execute_function(&module, "matmul_kernel", &args).unwrap();
    });
    eprintln!(
        "matmul: {:.1} µs/run (M=64,K=2048,N=8192 f16, grid[2,16], {iters} iters)",
        each * 1e6
    );
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn bench_layernorm() {
    let module = parse_module(LAYERNORM).expect("parse layernorm");
    let (rows, cols) = (1151usize, 8192usize);
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| f16(((i % 17) as f32 - 8.0) * 0.01))
        .collect();
    let w = vec![1.0f32; rows * cols];
    let b = vec![0.0f32; rows * cols];
    let iters = 20;
    let big = |data: Vec<f32>| Arg::Tensor {
        data,
        shape: vec![rows, cols],
        dtype: DType::F16,
    };
    let vec1 = |data: Vec<f32>| Arg::Tensor {
        data,
        shape: vec![rows],
        dtype: DType::F16,
    };
    let each = time_call(iters, || {
        let args: Vec<(&str, Arg)> = vec![
            ("X", big(x.clone())),
            ("Y", big(vec![0.0; rows * cols])),
            ("W", big(w.clone())),
            ("B", big(b.clone())),
            ("Mean", vec1(vec![0.0; rows])),
            ("Rstd", vec1(vec![0.0; rows])),
            ("N", Arg::Scalar(Scalar::I64(cols as i64))),
            ("eps", Arg::Scalar(Scalar::F32(1e-5))),
            ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(1024))),
        ];
        execute_function(&module, "_layer_norm_fwd_fused", &args).unwrap();
    });
    eprintln!(
        "layernorm: {:.1} µs/run (1151×8192 f16, grid[32], {iters} iters)",
        each * 1e6
    );
}
