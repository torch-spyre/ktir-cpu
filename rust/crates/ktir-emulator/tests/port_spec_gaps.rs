// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_spec_gaps.py` — one marker per known RFC conformance gap.
//!
//! In Python each case is a `@pytest.mark.xfail(strict=True)` whose *intent* is:
//! "this feature is in the RFC examples but not yet implemented; when it starts
//! working, the strict xfail flips to XPASS and reminds us to promote it." A
//! strict xfail is a test that is EXPECTED TO FAIL — i.e. it asserts the gap
//! still exists.
//!
//! Translation strategy (Python -> Rust)
//! -------------------------------------
//! There is no Rust `xfail(strict=true)` primitive, and `#[ignore]` on its own
//! would not *verify* the gap is still present (an ignored test never runs, so a
//! silently-fixed feature would go unnoticed). To keep the same guarantee with
//! a stronger assertion, each gap is ported as a *running* `#[test]` that asserts
//! the current behaviour: the feature genuinely fails (`parse_module` /
//! `execute_function` returns `Err`). These assertions are the faithful dual of
//! the Python strict-xfail: if the crate gains the feature, the kernel will run
//! to `Ok(..)` and the `assert!(.. .is_err())` here will fail loudly — exactly
//! the "unexpected pass" signal `strict=True` provides in pytest.
//!
//! Each test documents, in a comment, the *exact* error the crate currently
//! surfaces (probed against the real crate) so a future reader can see precisely
//! how the gap manifests. We assert only that execution does not succeed (not the
//! exact error string), so an evolving-but-still-incomplete implementation does
//! not produce false alarms — but a *complete* one does.
//!
//! Notes on individual Python cases that are NOT a plain "running gap assertion":
//! * `test_paged_tensor_indirect_access` / `test_paged_tensor_indirect_scatter`
//!   monkeypatch `interp._prepare_execution` to seed HBM (Idx/X tensors) before
//!   running. The Rust `execute_function` API has no equivalent pre-execution
//!   seeding hook — that is Python-only test infrastructure. The gap itself
//!   (the kernel cannot run end-to-end) is observable without the seeding: the
//!   crate errors on the `scf.forall` induction variable (`%b`) before it ever
//!   reaches the LX-overflow capacity limit the Python reason cites. We assert
//!   the run fails; see `skipped` for the classification.

use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::parser::parse_module;

// ---------------------------------------------------------------------------
// helper
// ---------------------------------------------------------------------------

/// Parse `src`, then attempt to run `func` with `args`. Returns `Ok(())` only if
/// BOTH parse and execution succeed; otherwise returns the surfaced error string.
/// Mirrors `interp.load(...) ; interp.execute_function(...)`.
fn run(src: &str, func: &str, args: &[(&str, Arg)]) -> Result<(), String> {
    let module = parse_module(src)?;
    execute_function(&module, func, args)?;
    Ok(())
}

// ===========================================================================
// ktdp.construct_indirect_access_tile (RFC §C.5)
//
// Python: test_paged_tensor_indirect_access
//   xfail(strict=True, reason="ktdp.load of 4x8x2048x128 f16 tile (16 MB) exceeds
//   2 MB LX scratchpad"). Uses examples/rfc/paged-tensor-copy.mlir, func
//   @paged_tensor_copy_1core. The Python test monkeypatches _prepare_execution to
//   seed Idx (all zeros) and X (page 0) into HBM — Python-only infrastructure with
//   no Rust analogue. Without that seeding the kernel still cannot run: the crate
//   errors on the scf.forall induction variable `%b` ("undefined SSA value: %b")
//   before reaching the cited LX capacity limit. Either way the run does not
//   succeed, which is the behaviour the strict xfail guards.
// ===========================================================================

const PAGED_TENSOR_COPY: &str = include_str!("../../../../examples/rfc/paged-tensor-copy.mlir");

#[test]
fn paged_tensor_indirect_access_gap() {
    // Current behaviour: parses, but execution fails (observed:
    // "undefined SSA value: %b" from the scf.forall induction binding; the
    // LX-overflow capacity limit the Python reason cites is downstream of this).
    let res = run(PAGED_TENSOR_COPY, "paged_tensor_copy_1core", &[]);
    assert!(
        res.is_err(),
        "expected paged-tensor-copy gap to persist (it ran to completion)"
    );
}

// ===========================================================================
// Python: test_paged_tensor_indirect_scatter
//   xfail(strict=True, same LX-overflow reason). Uses
//   examples/rfc/paged-tensor-write.mlir, func @paged_tensor_write_1core. Scatter
//   dual of the copy kernel; Python seeds X (zeros) and Idx (zeros) via the same
//   _prepare_execution monkeypatch (Python-only infra). Without seeding the crate
//   errors on the scf.forall induction variable before reaching LX overflow.
// ===========================================================================

const PAGED_TENSOR_WRITE: &str = include_str!("../../../../examples/rfc/paged-tensor-write.mlir");

#[test]
fn paged_tensor_indirect_scatter_gap() {
    // Current behaviour: parses, execution fails (observed:
    // "undefined SSA value: %b").
    let res = run(PAGED_TENSOR_WRITE, "paged_tensor_write_1core", &[]);
    assert!(
        res.is_err(),
        "expected paged-tensor-write gap to persist (it ran to completion)"
    );
}

// ===========================================================================
// RFC-explicit non-ktdp ops: linalg.add / tensor.empty inside scf.for/forall
//
// Python: test_linalg_add_tensor_empty
//   xfail(reason="linalg.add not implemented")  (note: NON-strict).
//   Uses examples/rfc/add-with-control-flow.mlir, func @add. tensor.empty is
//   implemented; linalg.add is the named gap.
//
// The Python xfail is NON-strict and marks a *Python* gap: `linalg.add` is not
// implemented there. The Rust crate DOES implement `linalg.add` (dialects/
// linalg.rs), and with `scf.for` parsing now in place the kernel runs to
// completion — so this gap is closed in Rust (it is ahead of Python here). A
// non-strict xfail tolerates an xpass, so a passing Rust run is compliant.
// ===========================================================================

const ADD_WITH_CONTROL_FLOW: &str =
    include_str!("../../../../examples/rfc/add-with-control-flow.mlir");

#[test]
fn linalg_add_tensor_empty_runs() {
    // Rust implements both `scf.for` and `linalg.add`, so the kernel executes
    // (unlike Python, whose non-strict xfail reflects its missing `linalg.add`).
    let res = run(ADD_WITH_CONTROL_FLOW, "add", &[]);
    assert!(
        res.is_ok(),
        "add-with-control-flow should now run to completion: {res:?}"
    );
}

// ===========================================================================
// tensor.extract_slice
//
// Python: test_tensor_extract_slice
//   xfail(strict=True, reason="tensor.extract_slice not implemented").
//   Inline MLIR (no RFC fixture). The slice result is stored back so an
//   unknown-op skip causes a hard failure rather than a silent pass.
//   Rust: no handler registered for op 'tensor.extract_slice'.
// ===========================================================================

const EXTRACT_SLICE_MLIR: &str = r#"
module {
  func.func @extract_slice_kernel() attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %src = arith.constant 0 : index
    %dst = arith.constant 256 : index
    %src_view = ktdp.construct_memory_view %src, sizes: [8, 8], strides: [8, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x8xf16>
    %dst_view = ktdp.construct_memory_view %dst, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>
    %src_access = ktdp.construct_access_tile %src_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8x8xf16> -> !ktdp.access_tile<8x8xindex>
    %dst_access = ktdp.construct_access_tile %dst_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>
    %tile = ktdp.load %src_access : !ktdp.access_tile<8x8xindex> -> tensor<8x8xf16>
    %slice = tensor.extract_slice %tile[0, 0][4, 4][1, 1] : tensor<8x8xf16> to tensor<4x4xf16>
    ktdp.store %slice, %dst_access : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>
    return
  }
}
"#;

#[test]
fn tensor_extract_slice_runs() {
    // GAP CLOSED (increment 2): the emulator now executes tensor.extract_slice
    // (parser captures the [offsets][sizes][strides] triple; the handler
    // materializes the strided sub-view). The kernel runs to completion.
    let res = run(EXTRACT_SLICE_MLIR, "extract_slice_kernel", &[]);
    assert!(
        res.is_ok(),
        "tensor.extract_slice should now run to completion: {res:?}"
    );
}

// ===========================================================================
// scf.parallel / scf.forall
//
// Python: test_scf_parallel
//   xfail(strict=True, reason="scf.parallel / scf.forall not implemented").
//   Inline MLIR. The loop result is stored so an unknown-op skip fails hard.
//   Rust: no handler registered for op 'scf.parallel'.
// ===========================================================================

const SCF_PARALLEL_MLIR: &str = r#"
module {
  func.func @parallel_kernel() attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %dst = arith.constant 0 : index
    %dst_view = ktdp.construct_memory_view %dst, sizes: [4, 1], strides: [1, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x1xf16>
    %dst_access = ktdp.construct_access_tile %dst_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<4x1xf16> -> !ktdp.access_tile<4x1xindex>
    %result = scf.parallel (%i) = (%c0) to (%c4) step (%c1) -> tensor<4x1xf16> {
      scf.reduce
    }
    ktdp.store %result, %dst_access : tensor<4x1xf16>, !ktdp.access_tile<4x1xindex>
    return
  }
}
"#;

#[test]
fn scf_parallel_gap() {
    // Current behaviour: parses, execution fails (observed:
    // "no handler registered for op 'scf.parallel'").
    let res = run(SCF_PARALLEL_MLIR, "parallel_kernel", &[]);
    assert!(
        res.is_err(),
        "expected scf.parallel gap to persist (it ran to completion)"
    );
}

// ===========================================================================
// scf.reduce / scf.reduce.return
//
// Python: test_scf_reduce
//   xfail(strict=True, reason="scf.reduce / scf.reduce.return not implemented").
//   Inline MLIR. The reduction result feeds an arith op so an unknown-op skip
//   fails hard. The reduction sits inside an scf.parallel, so the crate errors on
//   the scf.parallel op first ("no handler registered for op 'scf.parallel'") —
//   either way the kernel cannot execute, which is the gap this guards.
// ===========================================================================

const SCF_REDUCE_MLIR: &str = r#"
module {
  func.func @reduce_kernel() attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0.0 : f16
    %result = scf.parallel (%i) = (%c0) to (%c4) step (%c1) init (%init) -> f16 {
      %val = arith.constant 1.0 : f16
      scf.reduce(%val : f16) {
        ^bb0(%lhs: f16, %rhs: f16):
          %sum = arith.addf %lhs, %rhs : f16
          scf.reduce.return %sum : f16
      }
    }
    %check = arith.addf %result, %init : f16
    return
  }
}
"#;

#[test]
fn scf_reduce_gap() {
    // Current behaviour: parses, execution fails (observed:
    // "no handler registered for op 'scf.parallel'", the enclosing op).
    let res = run(SCF_REDUCE_MLIR, "reduce_kernel", &[]);
    assert!(
        res.is_err(),
        "expected scf.reduce gap to persist (it ran to completion)"
    );
}
