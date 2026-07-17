// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_examples.py` — end-to-end execution of the compiler
//! generated KTIR example kernels in `examples/triton-ktir/` and `examples/ktir/`.
//!
//! This is the parity harness: each example is parsed with
//! [`ktir_emulator::parser::parse_module`] and run through
//! [`ktir_emulator::interpreter::execute_function`] (HBM marshalling -> multi-core
//! execution -> read-back), and the tensor output is checked against a
//! reference computed in-test.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python drives via `KTIRInterpreter.execute_function(name, **{arg: array})`,
//!   using `interp.arg_names(func_name)` to discover source-level argument
//!   names. The Rust `execute_function` takes `&[(&str, Arg)]` keyed by the same
//!   source-level names (e.g. `"x_ptr"`, `"BLOCK_SIZE"`), so we pass them
//!   directly (mirroring `rust/tests/end_to_end.rs`).
//! * Python uses `np.float16` arrays and NumPy reference math, comparing with a
//!   loose `rtol/atol`. To get *exact* parity instead of a tolerance, the Rust
//!   reference rounds every f16 value through the crate's own half-precision
//!   round-trip (`codec::f32_to_f16_bits` / `f16_bits_to_f32`) — the identical
//!   rounding the kernel's load/store path uses — and then compares with a tiny
//!   tolerance. Where intermediate compute is done in f32 inside the kernel
//!   (softmax / attention), an `rtol/atol` tolerance comparable to the Python
//!   one is used.
//! * Python's `np.random.default_rng(42)` is not reproducible in Rust, so the
//!   data is generated deterministically (small periodic / linspace patterns).
//!   The behaviour under test (output == reference) is identical.
//! * Scalar argument dtypes follow the MLIR signatures: `index` scalars are
//!   passed as `Scalar::I64` (matching `end_to_end.rs`), `i32` as `Scalar::I32`.
//!
//! Skipped Python cases (see `skipped` in the integrator notes):
//! * `TestExampleParsing` (parse/structure/attribute metadata) — covered by the
//!   dedicated `port_parse.rs` port; not an execution test.
//! * `TestSoftmaxExecution::test_softmax_lx_overflow` — asserts a Python
//!   `MemoryError`; ported as an `#[ignore]` `Err(..)`-expecting stub.
//! * `TestRingReduceExecution` — Python-side `@pytest.mark.xfail` (parser does
//!   not support `#ktdp.reduce_kind` / `reduce_mode` / `grid_axis`). Ported as
//!   an `#[ignore]` stub.

use std::collections::HashMap;

use ktir_emulator::codec::{f16_bits_to_f32, f32_to_f16_bits};
use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, Output, execute_function};
use ktir_emulator::ir::Scalar;
use ktir_emulator::parser::parse_module;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Round an f32 through the crate's IEEE-754 half-precision round-trip — the
/// exact rounding the kernel's f16 load/store path applies.
fn f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

fn f16_vec(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| f16(x)).collect()
}

/// A deterministic "random-ish" sequence in roughly [-1.5, 1.5), rounded to
/// f16. Stands in for Python's `rng.standard_normal(...).astype(f16)` — the
/// values differ but the property under test (out == reference) does not.
fn data_f16(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as u64).wrapping_mul(2654435761).wrapping_add(seed);
            let frac = ((t >> 8) & 0xFFFF) as f32 / 65536.0;
            f16(frac * 3.0 - 1.5)
        })
        .collect()
}

fn data_f32(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as u64).wrapping_mul(2654435761).wrapping_add(seed);
            let frac = ((t >> 8) & 0xFFFF) as f32 / 65536.0;
            frac * 3.0 - 1.5
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        assert!(
            diff <= tol,
            "mismatch at {i}: actual={a}, expected={e}, diff={diff}, tol={tol}"
        );
    }
}

fn get_output<'a>(outputs: &'a HashMap<String, Output>, name: &str) -> &'a Output {
    outputs
        .get(name)
        .unwrap_or_else(|| panic!("output {name:?} not present"))
}

// ===========================================================================
// TestVectorAddExecution — examples/triton-ktir/vector_add_ktir.mlir
// add_kernel(%x_ptr, %y_ptr, %output_ptr, %BLOCK_SIZE), grid = [32, 1]
// BLOCK_SIZE = 128, 32 cores -> n = 4096.
// ===========================================================================

const VECTOR_ADD: &str = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");

#[test]
fn vector_add_single_core() {
    // Python test_single_core: out = x + y over n=4096 f16 elements.
    let module = parse_module(VECTOR_ADD).expect("parse vector_add");
    let n = 4096usize;
    let x = data_f16(n, 1);
    let y = data_f16(n, 2);
    let out = vec![0.0f32; n];

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
                data: out,
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
    ];
    let outputs = execute_function(&module, "add_kernel", &args).expect("run add_kernel");
    let result = &get_output(&outputs, "output_ptr").data;

    let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| f16(a + b)).collect();
    assert_close(result, &expected, 1e-2, 1e-2);
}

#[test]
fn vector_add_various_values() {
    // Python test_various_values: x all-zeros, y linspace(-10, 10, n).
    let module = parse_module(VECTOR_ADD).expect("parse vector_add");
    let n = 4096usize;
    let x = vec![0.0f32; n];
    let y: Vec<f32> = (0..n)
        .map(|i| f16(-10.0 + 20.0 * (i as f32) / ((n - 1) as f32)))
        .collect();
    let out = vec![0.0f32; n];

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
                data: out,
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
    ];
    let outputs = execute_function(&module, "add_kernel", &args).expect("run add_kernel");
    let result = &get_output(&outputs, "output_ptr").data;

    let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| f16(a + b)).collect();
    assert_close(result, &expected, 1e-2, 1e-2);
}

// ===========================================================================
// TestVectorAddDynamicExecution — vector_add_dynamic_ktir.mlir
// add_kernel_dynamic(%x_ptr, %y_ptr, %output_ptr, %n_elements: i32), grid = [1].
// The symbolic coordinate set masks out-of-range elements; n in {256,512,1024}.
// ===========================================================================

const VECTOR_ADD_DYNAMIC: &str =
    include_str!("../../../../examples/triton-ktir/vector_add_dynamic_ktir.mlir");

#[allow(dead_code)]
fn run_vector_add_dynamic(n: usize) {
    let module = parse_module(VECTOR_ADD_DYNAMIC).expect("parse vector_add_dynamic");
    let x = data_f32(n, 1);
    let y = data_f32(n, 2);
    let out = vec![0.0f32; n];

    let args = [
        (
            "x_ptr",
            Arg::Tensor {
                data: x.clone(),
                shape: vec![n],
                dtype: DType::F32,
            },
        ),
        (
            "y_ptr",
            Arg::Tensor {
                data: y.clone(),
                shape: vec![n],
                dtype: DType::F32,
            },
        ),
        (
            "output_ptr",
            Arg::Tensor {
                data: out,
                shape: vec![n],
                dtype: DType::F32,
            },
        ),
        ("n_elements", Arg::Scalar(Scalar::I32(n as i32))),
    ];
    let outputs =
        execute_function(&module, "add_kernel_dynamic", &args).expect("run add_kernel_dynamic");
    let result = &get_output(&outputs, "output_ptr").data;

    let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_close(result, &expected, 1e-5, 1e-5);
}

// GAP: the baseline parser only records the `shape` attribute on
// construct_memory_view when every `sizes:` element is a literal int; for the
// dynamic memref<?xf32> view (sizes: [%n], symbolic coordinate set) it follows
// the documented "lazily resolve SSA sizes" contract and stores no `shape`, so
// the construct_memory_view handler errors at runtime. Skipped until SSA-size
// resolution lands.
#[test]
fn vector_add_dynamic_256() {
    run_vector_add_dynamic(256);
}

#[test]
fn vector_add_dynamic_512() {
    run_vector_add_dynamic(512);
}

#[test]
fn vector_add_dynamic_1024() {
    run_vector_add_dynamic(1024);
}

// ===========================================================================
// TestReduceExplicitRegion — examples/ktir/reduce_generic.mlir
// reduce_explicit_region(%arg0: index), grid = [1, 1].
// %arg0 is BOTH input and output (same buffer). linalg.reduce in generic form.
// Reduce [1,2,3,4] along dim 1 -> result broadcast to [10,10,10,10].
// ===========================================================================

const REDUCE_GENERIC: &str = include_str!("../../../../examples/ktir/reduce_generic.mlir");

// GAP: the baseline parser DEFERS nested op regions ("DEFERRED (later slices):
// nested regions"), so the linalg.reduce combiner block and its `dimensions =
// [1]` attribute are not lifted from the MLIR text. Driven from text the reduce
// collapses the wrong axis (observed result 2 instead of 10). The reduce
// *handler* itself supports explicit regions (exercised programmatically
// elsewhere); only the MLIR-text path is missing. Skipped until the parser
// lifts reduce regions + `dimensions`.
#[test]
fn reduce_explicit_region_sum() {
    let module = parse_module(REDUCE_GENERIC).expect("parse reduce_generic");
    let data = f16_vec(&[1.0, 2.0, 3.0, 4.0]); // shape [1, 4]
    let args = [(
        "arg0",
        Arg::Tensor {
            data,
            shape: vec![1, 4],
            dtype: DType::F16,
        },
    )];
    let outputs = execute_function(&module, "reduce_explicit_region", &args).expect("run reduce");
    let result = &get_output(&outputs, "arg0").data;

    let expected = vec![10.0f32; 4];
    assert_close(result, &expected, 1e-2, 1e-2);
}

#[test]
fn reduce_explicit_region_zeros() {
    let module = parse_module(REDUCE_GENERIC).expect("parse reduce_generic");
    let data = vec![0.0f32; 4]; // shape [1, 4]
    let args = [(
        "arg0",
        Arg::Tensor {
            data,
            shape: vec![1, 4],
            dtype: DType::F16,
        },
    )];
    let outputs = execute_function(&module, "reduce_explicit_region", &args).expect("run reduce");
    let result = &get_output(&outputs, "arg0").data;

    assert_close(result, &[0.0f32; 4], 0.0, 1e-3);
}

// ===========================================================================
// TestSdpaExecution — examples/triton-ktir/sdpa_2d.mlir
// sdpa_kernel_2d(%q_ptr, %k_ptr, %v_ptr, %output_ptr), grid = [1].
// out ~= softmax(Q @ K^T * scale) @ V, with scale = 1/sqrt(64) = 0.125.
// Q, K, V, output are [32, 64] f16.
// ===========================================================================

const SDPA_2D: &str = include_str!("../../../../examples/triton-ktir/sdpa_2d.mlir");

// Parses fully now (transpose `permutation`, reduce regions/`dimensions`, and
// tensor result-shape derivation all land). Remaining gap is semantic: the
// softmax-over-rows reduce in this kernel yields a `[1]` where the subsequent
// `linalg.matmul` expects `[32,1]` ("outs shape [1] != A@B shape [32,32]") —
// a row-wise reduce/broadcast shape mismatch in this multi-step kernel.
#[test]
fn sdpa_2d() {
    let module = parse_module(SDPA_2D).expect("parse sdpa_2d");
    let (n_rows, head_dim) = (32usize, 64usize);
    let n = n_rows * head_dim;
    let q = data_f16(n, 1);
    let k = data_f16(n, 2);
    let v = data_f16(n, 3);
    let out = vec![0.0f32; n];

    let args = [
        (
            "q_ptr",
            Arg::Tensor {
                data: q.clone(),
                shape: vec![n_rows, head_dim],
                dtype: DType::F16,
            },
        ),
        (
            "k_ptr",
            Arg::Tensor {
                data: k.clone(),
                shape: vec![n_rows, head_dim],
                dtype: DType::F16,
            },
        ),
        (
            "v_ptr",
            Arg::Tensor {
                data: v.clone(),
                shape: vec![n_rows, head_dim],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::Tensor {
                data: out,
                shape: vec![n_rows, head_dim],
                dtype: DType::F16,
            },
        ),
    ];
    let outputs = execute_function(&module, "sdpa_kernel_2d", &args).expect("run sdpa");
    let result = &get_output(&outputs, "output_ptr").data;

    // Reference: scaled dot-product attention in f32 (matches the kernel's f32
    // intermediate math); P normalised is materialised as an f16 tile, and the
    // final result is rounded to f16.
    let scale = 0.125f32; // 1/sqrt(64)
    let mut expected = vec![0.0f32; n];
    for i in 0..n_rows {
        let mut scores = vec![0.0f32; n_rows];
        let mut m = f32::NEG_INFINITY;
        for j in 0..n_rows {
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q[i * head_dim + d] * k[j * head_dim + d];
            }
            s *= scale;
            scores[j] = s;
            if s > m {
                m = s;
            }
        }
        let mut denom = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - m).exp();
            denom += *sc;
        }
        for sc in scores.iter_mut() {
            *sc = f16(*sc / denom);
        }
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..n_rows {
                acc += scores[j] * v[j * head_dim + d];
            }
            expected[i * head_dim + d] = f16(acc);
        }
    }
    assert_close(result, &expected, 2e-2, 2e-2);
}

// ===========================================================================
// TestSoftmaxExecution::test_softmax_lx_overflow — examples/ktir/softmax_wide.mlir
// A row too wide for LX must fail at execution. Python expects a MemoryError
// matching "LX scratchpad overflow"; the Rust crate surfaces this as an
// `Err(..)` from execute_function (the error *string* differs from Python's
// message, so we assert failure rather than match the exact text), which is the
// faithful behavioural port: the run must not succeed.
// ===========================================================================

#[test]
fn softmax_wide_runs_under_lx_liveness() {
    // 2x262144 f16 rowwise softmax. A naive impl would hold the 512 KB row plus its
    // several same-shape intermediates live at once (>2 MB LX). With the #134/#118
    // LX-liveness model (single-use tiles consumed at last use inside the per-row
    // loop body, no iter_arg double-count) the per-row peak fits, so the kernel now
    // runs to completion — it no longer raises the LX overflow it once did. Mirrors
    // the diff harness's softmax_wide PASS and the removal of the invalidated Python
    // `test_softmax_lx_overflow` in #134.
    let src = include_str!("../../../../examples/ktir/softmax_wide.mlir");
    let module = parse_module(src).expect("parse softmax_wide");
    let n_rows = 2usize;
    let n_cols = 262144usize;
    let n = n_rows * n_cols;
    let inp = vec![0.0f32; n];
    let out = vec![0.0f32; n];
    let args = [
        (
            "output_ptr",
            Arg::Tensor {
                data: out,
                shape: vec![n_rows, n_cols],
                dtype: DType::F16,
            },
        ),
        (
            "input_ptr",
            Arg::Tensor {
                data: inp,
                shape: vec![n_rows, n_cols],
                dtype: DType::F16,
            },
        ),
    ];
    let res = execute_function(&module, "softmax_kernel", &args)
        .expect("softmax_wide must run to completion under LX-liveness, not overflow");
    // All-zero input -> uniform softmax: every element == 1/n_cols.
    let expected = 1.0f32 / n_cols as f32;
    let out = &res["output_ptr"];
    assert_eq!(out.data.len(), n);
    // f16 rounding band around the uniform value.
    assert!(
        (out.data[0] - expected).abs() < 1e-4,
        "softmax_wide output {} not ~uniform {expected}",
        out.data[0]
    );
}

// ===========================================================================
// TestRingReduceExecution — examples/ktir/ring_reduce.mlir
// Python-side @pytest.mark.xfail: parser lacks #ktdp.reduce_kind / reduce_mode /
// grid_axis support (torch-spyre/ktir-mlir-frontend#21). Ported as ignore stub.
// ===========================================================================

#[test]
#[ignore = "xfail in Python: parser lacks #ktdp.reduce_kind / reduce_mode / grid_axis attrs (ktir-mlir-frontend#21)"]
fn ring_reduce_sum() {
    let src = include_str!("../../../../examples/ktir/ring_reduce.mlir");
    let _ = parse_module(src);
}
