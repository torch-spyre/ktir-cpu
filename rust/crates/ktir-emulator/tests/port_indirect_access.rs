#![allow(clippy::needless_range_loop, clippy::type_complexity)]
// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_indirect_access.py` — `ktdp.construct_indirect_access_tile`
//! plus the indirect (gather/scatter) load/store data path (RFC 0682 §473,
//! implemented in `ops_memory::indirect_load` / `indirect_store` and built by
//! `dialects::ktdp_extra::construct_indirect_access_tile`).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * The Python suite drives whole MLIR kernels through `KTIRInterpreter`,
//!   seeding HBM with a monkey-patched `_prepare_execution` hook that writes the
//!   parent tensor `X`, the index tensors `IDX1`/`IDX2`, and the output `Y` to
//!   fixed stick addresses (the `arith.constant N : index` operands in the MLIR).
//!   The Rust `execute_function` exposes no HBM-seeding hook (it only marshals
//!   tensor args into freshly-allocated sticks), so — exactly like
//!   `port_distributed_view.rs` — the indirect path is driven at the ops layer:
//!   build an `IndirectAccessTile` directly over seeded `MemRef`s, run
//!   `indirect_load` / `indirect_store`, and check the SAME values the Python
//!   asserts. This exercises the SAME code (`ops_memory::indirect_*`,
//!   `build_indirect_coords`, the negative-index guard, the vso permutation
//!   guard, the vso sort-key ordering) the kernel path reaches.
//! * Python seeds each tensor as a separate `hbm.write(stick, ...)`; the Rust HBM
//!   keys allocations by base byte address and `read_bytes` does not span across
//!   allocations, so each tensor is `allocate`d independently and seeded at
//!   `stick * STICK_BYTES`.
//! * The index view in the Rust model is addressed by the enumeration point
//!   projected through the view's strides (`offset = Σ pt[d]*stride[d]`), which
//!   for a 4x4 `IDX` with strides `[4,1]` reads `IDX[m,k]` at point `(m,k)` —
//!   matching the Python `IDX[%m, %k]` identity-subscript case.
//! * `variables_space_set` is a row-major box affine set `[0, n-1]` per axis, so
//!   its `enumerate` yields the same `vss.enumerate` row-major point order Python
//!   iterates; a non-identity `variables_space_order` re-sorts those points by
//!   the map's image (lexicographic), per `enumerate_in_vso_order`.
//! * `test_*_rfc` (RFC-sized example `.mlir` files loaded by path) are smoke
//!   tests on all-zero input in Python; their faithful crate analogue is a 64x64
//!   all-zero indirect copy / scatter at the ops layer (parse-and-run end to end
//!   is covered by the example-file driver elsewhere). Kept as real tests.
//! * The two `test_ssa_intermediate_var_*` cases exercise a Python-only
//!   construction-time guard ("outer SSA value listed as an intermediate variable
//!   with non-zero range"). The Rust `construct_indirect_access_tile` models
//!   intermediate variables structurally (`DimSubscript::Direct { var_index }`)
//!   and never binds an outer SSA scalar as a variable, so that guard does not
//!   exist in the crate. Both cases are `#[ignore]`d with that reason.

use std::collections::HashMap;
use std::rc::Rc;

use ktir_emulator::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
use ktir_emulator::codec;
use ktir_emulator::context::CoreContext;
use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::single_core_context;
use ktir_emulator::memory::STICK_BYTES;
use ktir_emulator::memref::{DimSubscript, IndirectAccessTile, MemRef, MemorySpace};
use ktir_emulator::ops_memory::{indirect_load, indirect_store};
use ktir_emulator::tile::Tile;

// ===========================================================================
// helpers
// ===========================================================================

/// Row-major box affine set `[0, n-1]` per axis (`d_i >= 0`, `n_i-1 - d_i >= 0`).
/// Its `enumerate(&shape, &[])` yields the row-major variable-space point order
/// the Python `vss.enumerate` iterates.
fn box_set(sizes: &[i64]) -> AffineSet {
    let mut constraints = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        constraints.push(Constraint {
            expr: AffineExpr::Dim(i),
            kind: ConstraintKind::GreaterEq,
        });
        constraints.push(Constraint {
            expr: AffineExpr::Sub(
                Rc::new(AffineExpr::Const(n - 1)),
                Rc::new(AffineExpr::Dim(i)),
            ),
            kind: ConstraintKind::GreaterEq,
        });
    }
    AffineSet {
        num_dims: sizes.len(),
        num_syms: 0,
        constraints,
    }
}

/// Affine map `(d0,..) -> (perm[0], perm[1], ..)` over `perm.len()` dims.
fn perm_map(perm: &[usize]) -> AffineMap {
    AffineMap {
        num_dims: perm.len(),
        num_syms: 0,
        exprs: perm.iter().map(|&d| AffineExpr::Dim(d)).collect(),
    }
}

/// Row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

/// Allocate an HBM region for `data` encoded as `dtype`, seed it, and return a
/// `MemRef` over it with the given (row-major) shape/strides. Mirrors the Python
/// `hbm.write(stick, ...)` seeding step.
fn seed_hbm(ctx: &mut CoreContext, data: &[f32], dtype: DType, shape: &[usize]) -> MemRef {
    let raw = codec::encode(data, dtype);
    let stick = ctx.hbm.borrow_mut().allocate(raw.len().max(1) as i64);
    ctx.hbm.borrow_mut().write_bytes(stick * STICK_BYTES, &raw);
    MemRef {
        // base_ptr is an ELEMENT index (RFC #110): elem = stick*STICK_BYTES/bpe
        // so byte_address() == stick*STICK_BYTES (where the data was seeded).
        base_ptr: stick * STICK_BYTES / dtype.bytes_per_elem() as i64,
        shape: shape.to_vec(),
        strides: row_major_strides(shape),
        space: MemorySpace::Hbm,
        dtype,
        coordinate_set: None,
    }
}

/// Read `n` elements of `dtype` back from an HBM `MemRef`.
fn read_hbm(ctx: &CoreContext, mr: &MemRef, n: usize, dtype: DType) -> Vec<f32> {
    let nbytes = n * dtype.bytes_per_elem();
    // byte_address() = base_ptr*bytes_per_elem (element-index convention).
    let raw = ctx.hbm.borrow().read_bytes(mr.byte_address(), nbytes);
    codec::decode(&raw, n, dtype)
}

/// Build an IAT whose `dim_subscripts` come from `(kind, payload)` pairs:
/// `"indirect"` -> `Indirect { view: payload }`, `"direct"` -> `Direct { var_index: payload }`.
fn make_iat(
    parent: MemRef,
    shape: Vec<usize>,
    dims: &[(&str, usize)],
    index_views: Vec<MemRef>,
    vss: AffineSet,
    vso: Option<AffineMap>,
) -> IndirectAccessTile {
    let dim_subscripts = dims
        .iter()
        .map(|(kind, payload)| match *kind {
            "indirect" => DimSubscript::Indirect {
                view: *payload,
                idx_exprs: vec![],
            },
            "direct" => DimSubscript::Direct {
                var_index: *payload,
            },
            other => panic!("unknown dim kind {other}"),
        })
        .collect();
    IndirectAccessTile {
        parent_ref: parent,
        shape,
        dim_subscripts,
        index_views,
        variables_space_set: vss,
        variables_space_order: vso,
        extra: HashMap::new(),
    }
}

// ===========================================================================
// RFC-sized smoke tests (port of test_indirect_access_tile_rfc /
// test_indirect_scatter_rfc): all-zero 64x64 2-D gather / scatter, end to end.
// ===========================================================================

#[test]
fn indirect_access_tile_rfc() {
    // 64x64 gather Y[m,k] = X[IDX1[m,k], IDX2[m,k]] with everything zero-seeded.
    let n = 64usize;
    let mut ctx = single_core_context();
    let x = seed_hbm(&mut ctx, &vec![0.0; n * n], DType::F16, &[n, n]);
    let idx1 = seed_hbm(&mut ctx, &vec![0.0; n * n], DType::I32, &[n, n]);
    let idx2 = seed_hbm(&mut ctx, &vec![0.0; n * n], DType::I32, &[n, n]);

    let iat = make_iat(
        x,
        vec![n, n],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1, idx2],
        box_set(&[n as i64, n as i64]),
        None,
    );
    let tile = indirect_load(&mut ctx, &iat, None).unwrap();
    assert_eq!(tile.shape, vec![n, n]);
    assert!(tile.as_f32().iter().all(|&v| v == 0.0));
}

#[test]
fn indirect_scatter_rfc() {
    // 64x64 scatter Y[IDX1[m,k], IDX2[m,k]] = X[m,k] with everything zero-seeded.
    let n = 64usize;
    let mut ctx = single_core_context();
    let y = seed_hbm(&mut ctx, &vec![0.0; n * n], DType::F16, &[n, n]);
    let idx1 = seed_hbm(&mut ctx, &vec![0.0; n * n], DType::I32, &[n, n]);
    let idx2 = seed_hbm(&mut ctx, &vec![0.0; n * n], DType::I32, &[n, n]);

    let iat = make_iat(
        y.clone(),
        vec![n, n],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1, idx2],
        box_set(&[n as i64, n as i64]),
        None,
    );
    let src = Tile::compute(vec![0.0; n * n], DType::F16, vec![n, n]);
    indirect_store(&mut ctx, &src, &iat).unwrap();
    let out = read_hbm(&ctx, &y, n * n, DType::F16);
    assert!(out.iter().all(|&v| v == 0.0));
}

// ===========================================================================
// Small 4x4 indirect gather with data verification
// (port of test_small_indirect_gather)
// ===========================================================================
//
// X[i,j] = i*4+j (0..15); IDX1[m,k] = 3-k (each row [3,2,1,0]);
// IDX2[m,k] = k (each row [0,1,2,3]).  Y[m,k] = X[IDX1[m,k], IDX2[m,k]] = X[3-k, k].

#[test]
fn small_indirect_gather() {
    let mut ctx = single_core_context();
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // IDX1: each row [3,2,1,0].
    let idx1: Vec<f32> = (0..4).flat_map(|_| [3.0, 2.0, 1.0, 0.0]).collect();
    // IDX2: each row [0,1,2,3].
    let idx2: Vec<f32> = (0..4).flat_map(|_| [0.0, 1.0, 2.0, 3.0]).collect();

    let x_mr = seed_hbm(&mut ctx, &x, DType::F16, &[4, 4]);
    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);

    let iat = make_iat(
        x_mr,
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        None,
    );
    let tile = indirect_load(&mut ctx, &iat, None).unwrap();

    // Y[m,k] = X[3-k, k]: [12,9,6,3] in every row.
    let expected: Vec<f32> = (0..4).flat_map(|_| [12.0, 9.0, 6.0, 3.0]).collect();
    assert_eq!(tile.shape, vec![4, 4]);
    assert_eq!(tile.as_f32().to_vec(), expected);
}

// ===========================================================================
// Outer-SSA intermediate-variable guard (port of
// test_ssa_intermediate_var_nonzero_range_raises / _zero_range_ok).
//
// SKIPPED: the "outer SSA value listed as an intermediate variable with a
// non-zero range" construction-time guard is Python-only. The Rust
// `construct_indirect_access_tile` models intermediate variables structurally
// (DimSubscript::Direct { var_index }) and never binds an outer SSA scalar as a
// variable, so there is no such guard (and no analogue) in the crate.
// ===========================================================================

#[test]
#[ignore = "outer-SSA-as-intermediate-variable guard is Python-only; not modeled in the Rust crate"]
fn ssa_intermediate_var_nonzero_range_raises() {}

#[test]
#[ignore = "outer-SSA-as-intermediate-variable guard is Python-only; not modeled in the Rust crate"]
fn ssa_intermediate_var_zero_range_ok() {}

// ===========================================================================
// Small 4x4 indirect scatter with data verification (bijection)
// (port of test_small_indirect_scatter)
// ===========================================================================
//
// X[i,j] = i*4+j; IDX1[m,k] = 3-m; IDX2[m,k] = 3-k.
// Y[IDX1[m,k], IDX2[m,k]] = X[m,k] -> 180° rotation: Y[r,c] = X[3-r, 3-c].

#[test]
fn small_indirect_scatter() {
    let mut ctx = single_core_context();
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // IDX1[m,k] = 3-m: rows [3,3,3,3],[2,2,2,2],[1,1,1,1],[0,0,0,0].
    let idx1: Vec<f32> = [3.0, 2.0, 1.0, 0.0].iter().flat_map(|&v| [v; 4]).collect();
    // IDX2[m,k] = 3-k: each row [3,2,1,0].
    let idx2: Vec<f32> = (0..4).flat_map(|_| [3.0, 2.0, 1.0, 0.0]).collect();

    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);
    let y_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::F16, &[4, 4]);

    let iat = make_iat(
        y_mr.clone(),
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        None,
    );
    let src = Tile::compute(x, DType::F16, vec![4, 4]);
    indirect_store(&mut ctx, &src, &iat).unwrap();

    let y = read_hbm(&ctx, &y_mr, 16, DType::F16);
    // Y[r,c] = (3-r)*4 + (3-c).
    let expected: Vec<f32> = (0..4)
        .flat_map(|r| (0..4).map(move |c| ((3 - r) * 4 + (3 - c)) as f32))
        .collect();
    assert_eq!(y, expected);
}

#[test]
fn small_indirect_scatter_collision() {
    // IDX1, IDX2 all zeros -> every (m,k) writes Y[0,0]; last writer in
    // vss.enumerate (row-major) order is (m=3,k=3) -> X[3,3] = 15.
    let mut ctx = single_core_context();
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let idx1 = vec![0.0; 16];
    let idx2 = vec![0.0; 16];

    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);
    // Y seeded with sentinel -1 so untouched cells are verifiable.
    let y_mr = seed_hbm(&mut ctx, &[-1.0; 16], DType::F16, &[4, 4]);
    let iat = make_iat(
        y_mr.clone(),
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        None,
    );
    let src = Tile::compute(x, DType::F16, vec![4, 4]);
    indirect_store(&mut ctx, &src, &iat).unwrap();

    let y = read_hbm(&ctx, &y_mr, 16, DType::F16);
    assert_eq!(y[0], 15.0);
    for r in 0..4 {
        for c in 0..4 {
            if (r, c) == (0, 0) {
                continue;
            }
            assert_eq!(y[r * 4 + c], -1.0, "Y[{r},{c}] should be untouched");
        }
    }
}

// ===========================================================================
// Negative-index guard, both directions (port of
// test_negative_indirect_index_raises[indirect_load/indirect_store]).
// ===========================================================================

#[test]
fn negative_indirect_index_load_raises() {
    let mut ctx = single_core_context();
    let x = vec![0.0; 16];
    let mut idx1 = vec![0.0; 16];
    idx1[0] = -1.0; // negative entry must be rejected, not wrapped.
    let idx2 = vec![0.0; 16];

    let x_mr = seed_hbm(&mut ctx, &x, DType::F16, &[4, 4]);
    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);
    let iat = make_iat(
        x_mr,
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        None,
    );
    let err = indirect_load(&mut ctx, &iat, None).unwrap_err();
    assert!(err.contains("negative"), "unexpected error: {err}");
}

#[test]
fn negative_indirect_index_store_raises() {
    let mut ctx = single_core_context();
    let mut idx1 = vec![0.0; 16];
    idx1[0] = -1.0;
    let idx2 = vec![0.0; 16];

    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);
    let y_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::F16, &[4, 4]);
    let iat = make_iat(
        y_mr,
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        None,
    );
    let src = Tile::compute((0..16).map(|i| i as f32).collect(), DType::F16, vec![4, 4]);
    let err = indirect_store(&mut ctx, &src, &iat).unwrap_err();
    assert!(err.contains("negative"), "unexpected error: {err}");
}

// ===========================================================================
// Non-identity variables_space_order — swap (involution)
// (port of test_swap_vso[indirect_load_4x4_swap / indirect_store_4x4_swap])
// ===========================================================================
//
// X[m,k] = m*4+k; IDX1[m,k] = m; IDX2[m,k] = k (identity-coord gather/scatter).
// vso (d0,d1)->(d1,d0) reorders iteration to (d1,d0). With identity-coord IDX
// both the load gather tile and the store result Y equal X transposed:
// Y[r,c] = X[c,r] = c*4+r (matching the Python Y = X^T expectation).

fn swap_seed_4x4() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // IDX1[m,k] = m -> [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3].
    let idx1: Vec<f32> = (0..4).flat_map(|m| [m as f32; 4]).collect();
    // IDX2[m,k] = k -> tile([0,1,2,3], 4).
    let idx2: Vec<f32> = (0..4).flat_map(|_| [0.0, 1.0, 2.0, 3.0]).collect();
    (x, idx1, idx2)
}

#[test]
fn swap_vso_indirect_load() {
    let mut ctx = single_core_context();
    let (x, idx1, idx2) = swap_seed_4x4();
    let x_mr = seed_hbm(&mut ctx, &x, DType::F16, &[4, 4]);
    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);

    let iat = make_iat(
        x_mr,
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        Some(perm_map(&[1, 0])),
    );
    // The gather visits points in vso (swap) sort order, so the gathered data
    // tile lands in that reordered layout: with identity-coord IDX the value at
    // sorted position (c,r) is X[c,r], i.e. the gather tile is X transposed —
    // matching the Python Y = X^T expectation (gather-via-IAT then direct store).
    let tile = indirect_load(&mut ctx, &iat, None).unwrap();
    let expected: Vec<f32> = (0..4)
        .flat_map(|r| (0..4).map(move |c| (c * 4 + r) as f32))
        .collect();
    assert_eq!(tile.as_f32().to_vec(), expected);
}

#[test]
fn swap_vso_indirect_store() {
    // Read X identity-direct (full X), scatter through Y with the swap vso.
    // Y[r,c] = X[c,r] = c*4+r.
    let mut ctx = single_core_context();
    let (x, idx1, idx2) = swap_seed_4x4();
    let idx1_mr = seed_hbm(&mut ctx, &idx1, DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &idx2, DType::I32, &[4, 4]);
    let y_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::F16, &[4, 4]);

    let iat = make_iat(
        y_mr.clone(),
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        Some(perm_map(&[1, 0])),
    );
    let src = Tile::compute(x, DType::F16, vec![4, 4]);
    indirect_store(&mut ctx, &src, &iat).unwrap();

    let y = read_hbm(&ctx, &y_mr, 16, DType::F16);
    // Y[r,c] = X[c,r] = c*4 + r.
    let expected: Vec<f32> = (0..4)
        .flat_map(|r| (0..4).map(move |c| (c * 4 + r) as f32))
        .collect();
    assert_eq!(y, expected);
}

// ===========================================================================
// 3-D non-involution vso (3-cycle) — gather & scatter
// (port of test_indirect_load_with_3cycle_vso / _store_with_3cycle_vso)
// ===========================================================================
//
// X[m,k,l] = m*4+k*2+l (0..7); IDX[m,k,l] = m; dims 1+2 direct (k,l).
// vso (d0,d1,d2)->(d2,d0,d1) -> sort key (l,m,k).

#[test]
fn indirect_load_with_3cycle_vso() {
    let mut ctx = single_core_context();
    let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    // IDX[m,k,l] = m -> [0,0,0,0,1,1,1,1].
    let idx: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let x_mr = seed_hbm(&mut ctx, &x, DType::F16, &[2, 2, 2]);
    let idx_mr = seed_hbm(&mut ctx, &idx, DType::I32, &[2, 2, 2]);

    // dim0 indirect via IDX; dim1 direct var 1 (k); dim2 direct var 2 (l).
    let iat = make_iat(
        x_mr,
        vec![2, 2, 2],
        &[("indirect", 0), ("direct", 1), ("direct", 2)],
        vec![idx_mr],
        box_set(&[2, 2, 2]),
        Some(perm_map(&[2, 0, 1])),
    );
    let tile = indirect_load(&mut ctx, &iat, None).unwrap();

    // Sorted-by-(l,m,k) gather, reshaped (2,2,2) row-major.
    let expected = vec![0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0];
    assert_eq!(tile.shape, vec![2, 2, 2]);
    assert_eq!(tile.as_f32().to_vec(), expected);
}

#[test]
fn indirect_store_with_3cycle_vso() {
    let mut ctx = single_core_context();
    let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let idx: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let idx_mr = seed_hbm(&mut ctx, &idx, DType::I32, &[2, 2, 2]);
    let y_mr = seed_hbm(&mut ctx, &[0.0; 8], DType::F16, &[2, 2, 2]);

    let iat = make_iat(
        y_mr.clone(),
        vec![2, 2, 2],
        &[("indirect", 0), ("direct", 1), ("direct", 2)],
        vec![idx_mr],
        box_set(&[2, 2, 2]),
        Some(perm_map(&[2, 0, 1])),
    );
    let src = Tile::compute(x, DType::F16, vec![2, 2, 2]);
    indirect_store(&mut ctx, &src, &iat).unwrap();

    let y = read_hbm(&ctx, &y_mr, 8, DType::F16);
    let expected = vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
    assert_eq!(y, expected);
}

// ===========================================================================
// Non-permutation vso rejection, both directions (port of
// test_non_permutation_vso_raises[indirect_load_non_perm / indirect_store_non_perm]).
//
// vso (d0,d1)->(d0,d0) collapses two inputs to one output -> not a permutation;
// must be rejected at op-execution time.
// ===========================================================================

#[test]
fn non_permutation_vso_load_raises() {
    let mut ctx = single_core_context();
    let x_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::F16, &[4, 4]);
    let idx1_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::I32, &[4, 4]);
    let iat = make_iat(
        x_mr,
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        Some(perm_map(&[0, 0])), // non-permutation
    );
    let err = indirect_load(&mut ctx, &iat, None).unwrap_err();
    assert!(err.contains("permute"), "unexpected error: {err}");
}

#[test]
fn non_permutation_vso_store_raises() {
    let mut ctx = single_core_context();
    let y_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::F16, &[4, 4]);
    let idx1_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::I32, &[4, 4]);
    let idx2_mr = seed_hbm(&mut ctx, &[0.0; 16], DType::I32, &[4, 4]);
    let iat = make_iat(
        y_mr,
        vec![4, 4],
        &[("indirect", 0), ("indirect", 1)],
        vec![idx1_mr, idx2_mr],
        box_set(&[4, 4]),
        Some(perm_map(&[0, 0])),
    );
    let src = Tile::compute(vec![0.0; 16], DType::F16, vec![4, 4]);
    let err = indirect_store(&mut ctx, &src, &iat).unwrap_err();
    assert!(err.contains("permute"), "unexpected error: {err}");
}
