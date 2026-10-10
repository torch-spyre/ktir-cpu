#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::approx_constant
)]
// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_interpreter.py` — interpreter edge cases and
//! previously-uncovered paths.
//!
//! The Python file exercises four areas of `KTIRInterpreter`:
//!   1. Scalar (non-NumPy) arguments to `execute_function`.
//!   2. `execute_region` in isolation (empty / single / multi-op).
//!   3. Unknown op dispatch raising `ValueError`.
//!   4. Multi-result operation unpacking (registry patched at runtime).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python's `execute_function(name, **kwargs)` splits kwargs into NumPy arrays
//!   (marshalled into HBM, echoed in the returned `outputs` dict) and plain
//!   scalars (bound directly, NOT echoed). The Rust port draws the same line:
//!   `Arg::Tensor` is marshalled into HBM and read back into the `Output` map;
//!   `Arg::Scalar` is bound directly and never appears in the returned map. We
//!   assert on exactly that membership split.
//! * Python's `execute_region(core, ops)` runs a straight-line op list against a
//!   `CoreContext` and *returns the last op's result*. The Rust `execute_region`
//!   has the locked signature `(&[Operation], &mut CoreContext, &ExecutionEnv)
//!   -> Result<(), String>` (it threads results into the context but does not
//!   surface the final value). So the "returns last result" assertions are
//!   re-expressed as context-state assertions: after running, `ctx.get_value` of
//!   each result name holds the expected value (an equivalent, non-weaker check
//!   of the same execution). The empty-list case asserts `Ok(())` with no values
//!   bound.
//! * Python's `_execute_op(unknown_op, core)` raises `ValueError` matching the
//!   op name. Rust's `execute_op` returns `Err(String)`; we assert the error
//!   surfaces and names the op (`"no handler registered for op '<name>'"`).
//! * The two multi-result cases patch `registry._REGISTRY` with a fake handler
//!   and rely on `op.result` being a Python `list`. The Rust `Operation.result`
//!   is a single `Option<String>` (see port_parse.rs notes) and the dispatch
//!   registry is not runtime-patchable from an integration test (no public API
//!   to insert a handler). Both cases are therefore Python-only test infra with
//!   no faithful Rust analogue and are `#[ignore]`d (see `skipped`). We DO cover
//!   the genuinely portable half — a real multi-value handler binding through
//!   `execute_op` — via `scf.yield`, whose handler returns a `Value::Tuple`.

use ktir_emulator::dialects::Dispatch;
use ktir_emulator::dtypes::DType;
use ktir_emulator::env::{ExecutionEnv, GridExecutor};
use ktir_emulator::interpreter::{
    Arg, execute_function, execute_op, execute_region, single_core_context,
};
use ktir_emulator::ir::{Attr, Operation, Scalar, Value};
use ktir_emulator::parser::parse_module;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an `ExecutionEnv` + single-core `CoreContext`, mirroring the Python
/// `_minimal_core(interp)` setup (a `(1,1,1)` grid, core 0).
fn env_and_ctx() -> (Dispatch, GridExecutor) {
    (Dispatch::new(), GridExecutor::new((1, 1, 1)))
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Scalar(s) => s.as_i64().expect("int scalar"),
        Value::Index(i) => *i,
        other => panic!("expected int-like, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1. Scalar (non-NumPy) argument handling
// ---------------------------------------------------------------------------

const SCALAR_KTIR: &str = r#"
module {
    func.func @scalar_fn(%n: index) -> () attributes { grid = [1, 1, 1] } {
        return
    }
}
"#;

/// Non-tensor args are bound directly and are not echoed in `outputs`
/// (Python: scalar takes the else-branch, not allocated in HBM).
#[test]
fn execute_function_scalar_arg() {
    let module = parse_module(SCALAR_KTIR).expect("parse");
    let outputs = execute_function(
        &module,
        "scalar_fn",
        &[("%n", Arg::Scalar(Scalar::I64(42)))],
    )
    .expect("exec");
    assert!(
        !outputs.contains_key("%n"),
        "scalar arg must not be echoed in outputs"
    );
    assert!(outputs.is_empty(), "no tensor args => empty outputs");
}

/// Mixed scalar + tensor args: the tensor is read back into `outputs`, the
/// scalar is not.
#[test]
fn execute_function_scalar_and_array_args() {
    let ktir = r#"
module {
    func.func @mixed(%buf: memref<4xf16, "HBM">, %n: index) -> ()
            attributes { grid = [1, 1, 1] } {
        return
    }
}
"#;
    let module = parse_module(ktir).expect("parse");
    let outputs = execute_function(
        &module,
        "mixed",
        &[
            (
                "%buf",
                Arg::Tensor {
                    data: vec![0.0; 4],
                    shape: vec![4],
                    dtype: DType::F16,
                },
            ),
            ("%n", Arg::Scalar(Scalar::I64(7))),
        ],
    )
    .expect("exec");
    assert!(outputs.contains_key("%buf"), "tensor arg must be echoed");
    assert!(!outputs.contains_key("%n"), "scalar arg must not be echoed");
}

/// Both integer and float scalars are accepted without error.
#[test]
fn execute_function_scalar_int_and_float() {
    let module = parse_module(SCALAR_KTIR).expect("parse");
    execute_function(&module, "scalar_fn", &[("%n", Arg::Scalar(Scalar::I64(0)))])
        .expect("int scalar");
    execute_function(
        &module,
        "scalar_fn",
        &[("%n", Arg::Scalar(Scalar::F32(3.14)))],
    )
    .expect("float scalar");
}

// ---------------------------------------------------------------------------
// 2. execute_region in isolation
// ---------------------------------------------------------------------------

/// `execute_region` with an empty op list is a no-op: it succeeds and binds
/// nothing. (Python returns `None`; Rust returns `Ok(())`.)
#[test]
fn execute_region_empty() {
    let (dispatch, grid) = env_and_ctx();
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    let r = execute_region(&[], &mut ctx, &env);
    assert!(r.is_ok());
    assert!(
        ctx.get_value("%anything").is_err(),
        "no values should be bound"
    );
}

/// `execute_region` runs each op and threads its result into the context.
/// Python asserts the *return* equals the constant's value (99) and that the
/// context holds it; Rust checks the context binding (the equivalent state).
#[test]
fn execute_region_single_op() {
    let (dispatch, grid) = env_and_ctx();
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();

    let op = Operation::new(Some("%c"), "arith.constant", &[]).with_attr("value", Attr::Int(99));
    execute_region(std::slice::from_ref(&op), &mut ctx, &env).expect("region");
    assert_eq!(as_i64(ctx.get_value("%c").expect("%c bound")), 99);
}

/// With multiple ops, every result is bound; the last op's result reflects the
/// final value (Python: `execute_region` returns the last result == 2).
#[test]
fn execute_region_multiple_ops_returns_last() {
    let (dispatch, grid) = env_and_ctx();
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();

    let ops = vec![
        Operation::new(Some("%a"), "arith.constant", &[]).with_attr("value", Attr::Int(1)),
        Operation::new(Some("%b"), "arith.constant", &[]).with_attr("value", Attr::Int(2)),
    ];
    execute_region(&ops, &mut ctx, &env).expect("region");
    assert_eq!(as_i64(ctx.get_value("%a").expect("%a")), 1);
    // The final op's result (the Python "return value") is 2.
    assert_eq!(as_i64(ctx.get_value("%b").expect("%b")), 2);
}

// ---------------------------------------------------------------------------
// 3. Unknown op dispatch — error, names the op
// ---------------------------------------------------------------------------

/// An unregistered op_type surfaces an error naming the op (Python raises
/// `ValueError` matching `"totally.unknown_op"`).
#[test]
fn unknown_op_raises() {
    let (dispatch, grid) = env_and_ctx();
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();

    let unknown = Operation::new(None, "totally.unknown_op", &[]);
    let err = execute_op(&unknown, &mut ctx, &env).expect_err("unknown op must error");
    assert!(
        err.contains("totally.unknown_op"),
        "error must name the op: {err}"
    );
}

// ---------------------------------------------------------------------------
// 4. Multi-result operation handling
// ---------------------------------------------------------------------------

/// The genuinely portable half of Python's multi-result coverage: a handler
/// that returns multiple values. `scf.yield` returns a `Value::Tuple` whose
/// elements carry each yielded operand. Drive it through `execute_op` and
/// confirm both seeded values surface in the produced tuple, in order.
#[test]
fn multi_value_yield_produces_tuple() {
    let (dispatch, grid) = env_and_ctx();
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    ctx.set_value("%x", Value::Index(10));
    ctx.set_value("%y", Value::Index(20));

    let op = Operation::new(None, "scf.yield", &["%x", "%y"]);
    let produced = execute_op(&op, &mut ctx, &env).expect("yield runs");
    match produced {
        Some(Value::Tuple(vals)) => {
            assert_eq!(vals.len(), 2, "two yielded values");
            assert_eq!(as_i64(&vals[0]), 10);
            assert_eq!(as_i64(&vals[1]), 20);
        }
        other => panic!("expected Value::Tuple, got {other:?}"),
    }
}

/// Python `test_multi_result_tuple_unpacked`: patches `registry._REGISTRY` with
/// a fake handler returning `(10, 20)` and binds `op.result = ["%x", "%y"]`.
/// No Rust analogue: `Operation.result` is a single `Option<String>` (not a
/// list) and the dispatch registry is not runtime-patchable from an integration
/// test. Python-only test infra. See `multi_value_yield_produces_tuple` for the
/// portable behavior (a multi-valued handler result through `execute_op`).
#[test]
#[ignore = "GAP: Python-only infra — list-valued op.result + runtime registry patch have no Rust analogue (result is Option<String>; registry not patchable)"]
fn multi_result_tuple_unpacked() {}

/// Python `test_multi_result_single_value_raises_error`: documents the
/// unguarded behavior where a list `op.result` plus a non-tuple handler return
/// makes set_value blow up (unhashable list key -> TypeError). Same infra gap:
/// Rust's `op.result` is `Option<String>`, so this exact edge case is
/// structurally impossible to construct. Python-only.
#[test]
#[ignore = "GAP: Python-only infra — depends on list-valued op.result (Rust result is Option<String>; edge case cannot be constructed)"]
fn multi_result_single_value_raises_error() {}
