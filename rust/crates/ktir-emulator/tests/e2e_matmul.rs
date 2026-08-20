// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! RUST-ONLY (not a port of a Python test): the Python matmul tests assert on
//! the latency report; this is the first check of the matmul *output* value.
//!
//! End-to-end matmul: drive a real multi-core KTIR matmul kernel through the
//! full interpreter pipeline (parse → HBM marshal → grid → scf.for K-loop →
//! ktdp.load / linalg.matmul / arith.addf accumulate / ktdp.store → readback)
//! and check the matmul *output* against a reference GEMM — the first e2e
//! correctness check of the result (latency tests only assert on the report).
//!
//! Uses a 1-D core grid (`grid = [G]`, single-result `ktdp.get_compute_tile_id`)
//! that splits M across cores. The 2-D form (`%pid_m, %pid_n = ...`) is a known
//! parser gap (multi-result SSA binding) — see `matmul_2d_grid_gap`.

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ir::Scalar;
use ktir_emulator::parser::parse_module;

// A[M,K] · B[K,N] = C[M,N]; `cores` split the M rows (block_m each), shared B.
const M: usize = 32;
const K: usize = 64;
const N: usize = 32;
const CORES: usize = 4;
const BM: usize = M / CORES; // 8 rows/core

fn matmul_1d_kernel() -> String {
    format!(
        r#"module {{
  func.func @matmul_1d(%a_ptr: index, %b_ptr: index, %c_ptr: index, %K: index)
      attributes {{grid = [{CORES}]}} {{
    %pid = ktdp.get_compute_tile_id : index
    %bm = arith.constant {BM} : index
    %a_view = ktdp.construct_memory_view %a_ptr, sizes: [{M}, {K}], strides: [{K}, 1] {{
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {m1} >= 0, d1 >= 0, -d1 + {k1} >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{M}x{K}xf16>
    %b_view = ktdp.construct_memory_view %b_ptr, sizes: [{K}, {N}], strides: [{N}, 1] {{
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {k1} >= 0, d1 >= 0, -d1 + {n1} >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{K}x{N}xf16>
    %c_view = ktdp.construct_memory_view %c_ptr, sizes: [{M}, {N}], strides: [{N}, 1] {{
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {m1} >= 0, d1 >= 0, -d1 + {n1} >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{M}x{N}xf16>
    %offs_am = arith.muli %pid, %bm : index
    %accum_zero = arith.constant dense<0.0> : tensor<{BM}x{N}xf16>
    %c0 = arith.constant 0 : index
    %bk = arith.constant 32 : index
    %c = scf.for %off_k = %c0 to %K step %bk iter_args(%accum_itr = %accum_zero) -> (tensor<{BM}x{N}xf16>) {{
      %a_acc = ktdp.construct_access_tile %a_view[%offs_am, %off_k] {{
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {bm1} >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      }} : memref<{M}x{K}xf16> -> !ktdp.access_tile<{BM}x32xindex>
      %b_acc = ktdp.construct_access_tile %b_view[%off_k, %c0] {{
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + {n1} >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      }} : memref<{K}x{N}xf16> -> !ktdp.access_tile<32x{N}xindex>
      %a = ktdp.load %a_acc : !ktdp.access_tile<{BM}x32xindex> -> tensor<{BM}x32xf16>
      %b = ktdp.load %b_acc : !ktdp.access_tile<32x{N}xindex> -> tensor<32x{N}xf16>
      %c_init = tensor.empty() : tensor<{BM}x{N}xf16>
      %a_dot_b = linalg.matmul ins(%a, %b : tensor<{BM}x32xf16>, tensor<32x{N}xf16>)
                               outs(%c_init : tensor<{BM}x{N}xf16>) -> tensor<{BM}x{N}xf16>
      %accum_next = arith.addf %accum_itr, %a_dot_b : tensor<{BM}x{N}xf16>
      scf.yield %accum_next : tensor<{BM}x{N}xf16>
    }}
    %c_acc = ktdp.construct_access_tile %c_view[%offs_am, %c0] {{
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {bm1} >= 0, d1 >= 0, -d1 + {n1} >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    }} : memref<{M}x{N}xf16> -> !ktdp.access_tile<{BM}x{N}xindex>
    ktdp.store %c, %c_acc : tensor<{BM}x{N}xf16>, !ktdp.access_tile<{BM}x{N}xindex>
    return
  }}
}}"#,
        m1 = M - 1,
        k1 = K - 1,
        n1 = N - 1,
        bm1 = BM - 1,
    )
}

fn args<'a>(a: &'a [f32], b: &'a [f32], c: &'a [f32]) -> Vec<(&'a str, Arg)> {
    vec![
        (
            "a_ptr",
            Arg::Tensor {
                data: a.to_vec(),
                shape: vec![M, K],
                dtype: DType::F16,
            },
        ),
        (
            "b_ptr",
            Arg::Tensor {
                data: b.to_vec(),
                shape: vec![K, N],
                dtype: DType::F16,
            },
        ),
        (
            "c_ptr",
            Arg::Tensor {
                data: c.to_vec(),
                shape: vec![M, N],
                dtype: DType::F16,
            },
        ),
        ("K", Arg::Scalar(Scalar::I64(K as i64))),
    ]
}

fn f16(x: f32) -> f32 {
    ktir_emulator::codec::f16_bits_to_f32(ktir_emulator::codec::f32_to_f16_bits(x))
}

#[test]
fn matmul_kernel_end_to_end_correct() {
    let module = parse_module(&matmul_1d_kernel()).expect("parse matmul kernel");
    let a: Vec<f32> = (0..M * K)
        .map(|i| f16(((i % 13) as f32 - 6.0) * 0.1))
        .collect();
    let b: Vec<f32> = (0..K * N)
        .map(|i| f16(((i % 11) as f32 - 5.0) * 0.1))
        .collect();
    let c0 = vec![0.0f32; M * N];

    let outputs =
        execute_function(&module, "matmul_1d", &args(&a, &b, &c0)).expect("run matmul_1d");
    let got = &outputs.get("c_ptr").expect("c_ptr output").data;

    let want = ktir_emulator::blas::naive_sgemm(M, K, N, &a, &b);
    assert_eq!(got.len(), M * N);
    let mut max_rel = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
    }
    assert!(
        max_rel < 0.05,
        "e2e matmul max relative error {max_rel} exceeds f16 tolerance"
    );
    eprintln!(
        "e2e matmul ({M}×{K}×{N}, grid [{CORES}], K-tiled) correct — max rel err {max_rel:.4} ✓"
    );
}

/// E2E perf baseline: whole-program run time (parse excluded). The per-core tile
/// matmuls run on the CPU (Accelerate via the size gate — LX-sized tiles are
/// below the GPU crossover). The number a GPU-accelerated full-program path
/// must beat.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn matmul_kernel_end_to_end_perf() {
    let module = parse_module(&matmul_1d_kernel()).expect("parse");
    let a: Vec<f32> = (0..M * K).map(|i| f16((i % 13) as f32 * 0.1)).collect();
    let b: Vec<f32> = (0..K * N).map(|i| f16((i % 11) as f32 * 0.1)).collect();
    let c0 = vec![0.0f32; M * N];
    let iters = 500;
    execute_function(&module, "matmul_1d", &args(&a, &b, &c0)).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(execute_function(&module, "matmul_1d", &args(&a, &b, &c0)).unwrap());
    }
    let each = t0.elapsed().as_secs_f64() / iters as f64;
    eprintln!(
        "e2e matmul whole-program: {:.1} µs/run ({CORES} cores, K-tiled)",
        each * 1e6
    );
}
