// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_tile.py` — AccessTile / TileRef structure plus the
//! `MemoryOps.tile_access` / `load` / `store` data path (contiguous, strided,
//! and coordinate-set gather/scatter).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python `MemoryOps.tile_access(ctx, parent, indices, access_shape, base_map)`
//!   is the private `tile_access` helper inside `dialects::ktdp`. It is exercised
//!   here exactly as the runtime invokes it: by dispatching a real
//!   `ktdp.construct_access_tile` op (seeded with a `Value::MemRef` parent and
//!   `Value::Index` index operands) and inspecting the resulting
//!   `AccessTile`'s single-allocation `ParentRef::Tile(TileRef)`. The
//!   `TileRef.base_ptr` is the byte-addressed offset the Python test asserts on,
//!   and `AccessTile.shape` is the requested `access_shape`.
//! * Python's `HBMSimulator.STICK_BYTES` multiplier on the expected base pointer
//!   maps to the crate's `memory::STICK_BYTES` (both = 128). A MemRef's HBM
//!   `base_ptr` is a *stick* index, so its absolute byte address is
//!   `stick * STICK_BYTES`, exactly as `MemRef::byte_address` computes.
//! * Python `MemoryOps.load` / `MemoryOps.store` map to the public
//!   `ops_memory::load_data` / `ops_memory::store_data` (the `MemRef.to_tile_ref`
//!   byte-addressed data path). `coords=` + `result_shape=` map 1:1.
//! * HBM is populated by `store_data` of a contiguous `Tile` (which performs the
//!   f16 byte encoding) rather than by a raw byte write, so no private encoder is
//!   needed. f16 is the crate's flat-f32 storage; every value used here (small
//!   integers 0..63) is exactly representable in f16, so the round trip is exact.
//! * `parse_affine_map` / `parse_affine_set` are the `parser_ast` free functions.
//!   `AffineSet::enumerate(shape, syms)` is the Rust port of `css.enumerate`;
//!   `AffineMap::eval(dims, syms)` ports `cso.eval`.
//! * `MemoryOps._is_contiguous` is a *private* crate helper with no public
//!   surface. Its boolean assertions are reproduced by checking the *observable*
//!   consequence — a strided (non-row-major) tile still gathers/scatters the
//!   correct elements through the slow path, and a row-major tile loads as a
//!   single contiguous span — which is the behavior `_is_contiguous` gates. See
//!   `skipped` for the pure-predicate cases that have no integration analogue.
//! * `TestTileOps::test_affine_attrs_preserved` / `test_base_map_always_present`
//!   parse `examples/.../indirect-access-copy.mlir` and inspect op attributes on
//!   the parsed module. They are ported as module-parse assertions.

use ktir_emulator::affine::AffineSet;
use ktir_emulator::dialects::Dispatch;
use ktir_emulator::dtypes::DType;
use ktir_emulator::env::{ExecutionEnv, GridExecutor};
use ktir_emulator::interpreter::{execute_op, single_core_context};
use ktir_emulator::ir::{Attr, Operation, Value};
use ktir_emulator::memory::STICK_BYTES;
use ktir_emulator::memref::{MemRef, MemorySpace, ParentRef, TileRef};
use ktir_emulator::ops_memory::{load_data, store_data};
use ktir_emulator::parser_ast::{parse_affine_map, parse_affine_set};
use ktir_emulator::tile::Tile;

// ===========================================================================
// Harness
// ===========================================================================

/// Build an HBM `MemRef` whose data lives at HBM stick `stick` (byte address
/// `stick * STICK_BYTES`), mirroring the Python
/// `MemRef(base_ptr=stick_to_elem_idx(ptr, "f16"), …)`. `base_ptr` is an ELEMENT
/// index (RFC #110): `stick * STICK_BYTES / bytes_per_elem`, so
/// `byte_address() == stick * STICK_BYTES`.
fn hbm_memref(stick: i64, shape: &[usize], strides: &[i64], dtype: DType) -> MemRef {
    MemRef {
        base_ptr: stick * STICK_BYTES / dtype.bytes_per_elem() as i64,
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        space: MemorySpace::Hbm,
        dtype,
        coordinate_set: None,
    }
}

/// Allocate `n` f16 elements in HBM, populate them with `data` via a contiguous
/// `store_data` (which does the f16 byte encoding), and return the stick index.
fn alloc_f16(ctx: &mut ktir_emulator::context::CoreContext, data: &[f32], shape: &[usize]) -> i64 {
    let stick = ctx.hbm.borrow_mut().allocate((data.len() * 2) as i64);
    let m = hbm_memref(stick, shape, &row_major(shape), DType::F16);
    let tr = m.to_tile_ref();
    let tile = Tile::compute(data.to_vec(), DType::F16, shape.to_vec());
    store_data(ctx, &tile, &tr, None).expect("seed store");
    stick
}

/// Row-major (C-order) element strides for `shape`.
fn row_major(shape: &[usize]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1] as i64;
    }
    s
}

/// Read `n` f16 elements back from HBM stick `stick` as a flat f32 vector.
fn read_back(
    ctx: &mut ktir_emulator::context::CoreContext,
    stick: i64,
    shape: &[usize],
) -> Vec<f32> {
    let m = hbm_memref(stick, shape, &row_major(shape), DType::F16);
    let tr = m.to_tile_ref();
    load_data(ctx, &tr, None, None)
        .expect("read back")
        .as_f32()
        .to_vec()
}

/// Drive a real `ktdp.construct_access_tile` op over an HBM `MemRef` parent and
/// return the resulting single-allocation `TileRef` (the Python
/// `MemoryOps.tile_access(...)` return value).
fn tile_access(
    parent: MemRef,
    indices: &[i64],
    access_shape: &[usize],
    base_map_src: &str,
) -> TileRef {
    let dispatch = Dispatch::new();
    let grid = GridExecutor::new((1, 1, 1));
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();

    ctx.set_value("%view", Value::MemRef(parent));
    let mut operands = vec!["%view".to_string()];
    for (i, &ix) in indices.iter().enumerate() {
        let name = format!("%i{i}");
        ctx.set_value(&name, Value::Index(ix));
        operands.push(name);
    }
    let operand_refs: Vec<&str> = operands.iter().map(|s| s.as_str()).collect();

    let base_map = parse_affine_map(base_map_src).expect("base_map");
    let shape_i: Vec<i64> = access_shape.iter().map(|&n| n as i64).collect();
    let op = Operation::new(Some("%t"), "ktdp.construct_access_tile", &operand_refs)
        .with_attr("shape", Attr::IntList(shape_i))
        .with_attr("base_map", Attr::AffineMap(base_map));

    let produced = execute_op(&op, &mut ctx, &env)
        .expect("construct_access_tile")
        .expect("access tile value");
    match produced {
        Value::AccessTile(at) => match at.parent_ref {
            ParentRef::Tile(tr) => tr,
            other => panic!("expected single-allocation ParentRef::Tile, got {other:?}"),
        },
        other => panic!("expected AccessTile, got {other:?}"),
    }
}

/// Build a strided/sub-tile `TileRef` directly from a MemRef view (Python's
/// `MemRef(...).to_tile_ref()`).
fn tile_ref(stick: i64, shape: &[usize], strides: &[i64]) -> TileRef {
    hbm_memref(stick, shape, strides, DType::F16).to_tile_ref()
}

fn arange(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

// ===========================================================================
// TestTileAccess — tile_access offset computation via base_map
// ===========================================================================

#[test]
fn identity_map_matches_direct_stride() {
    // parent sizes [4,4] strides [4,1]; indices [1,2]; identity base_map.
    // offset = 1*4 + 2*1 = 6 elems = 12 bytes (f16). base = stick*128 + 12.
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(16), &[4, 4]);
    let parent = hbm_memref(stick, &[4, 4], &[4, 1], DType::F16);
    let tr = tile_access(parent, &[1, 2], &[2, 2], "affine_map<(d0, d1) -> (d0, d1)>");
    assert_eq!(tr.base_ptr, stick * STICK_BYTES + 6 * 2);
}

#[test]
fn non_identity_map_transposed_access() {
    // swapped map (d0,d1)->(d1,d0); indices [1,2] -> base coords (2,1).
    // offset = 2*4 + 1*1 = 9 elems = 18 bytes.
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(16), &[4, 4]);
    let parent = hbm_memref(stick, &[4, 4], &[4, 1], DType::F16);
    let tr = tile_access(parent, &[1, 2], &[1, 1], "affine_map<(d0, d1) -> (d1, d0)>");
    assert_eq!(tr.base_ptr, stick * STICK_BYTES + 9 * 2);
}

#[test]
fn scaled_map() {
    // scaled map (d0,d1)->(d0*2, d1); indices [1,3] -> base coords (2,3).
    // parent strides [8,1]; offset = 2*8 + 3*1 = 19 elems = 38 bytes.
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(64), &[64]);
    let parent = hbm_memref(stick, &[8, 8], &[8, 1], DType::F16);
    let tr = tile_access(
        parent,
        &[1, 3],
        &[1, 1],
        "affine_map<(d0, d1) -> (d0 * 2, d1)>",
    );
    assert_eq!(tr.base_ptr, stick * STICK_BYTES + 19 * 2);
}

#[test]
fn access_shape_preserved() {
    // The returned TileRef inherits the requested access_shape (2, 3).
    let ctx = single_core_context();
    let stick = ctx.hbm.borrow_mut().allocate(32);
    let parent = hbm_memref(stick, &[4, 4], &[4, 1], DType::F16);
    let tr = tile_access(parent, &[0, 0], &[2, 3], "affine_map<(d0, d1) -> (d0, d1)>");
    assert_eq!(tr.shape, vec![2, 3]);
}

#[test]
fn load_after_tile_access() {
    // 4x4 parent; access 2x2 sub-tile starting at row 1, col 0.
    // base offset = 1*4 = 4 elems -> sub-tile [[4,5],[8,9]].
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(16), &[4, 4]);
    let parent = hbm_memref(stick, &[4, 4], &[4, 1], DType::F16);
    let tr = tile_access(parent, &[1, 0], &[2, 2], "affine_map<(d0, d1) -> (d0, d1)>");
    let tile = load_data(&mut ctx, &tr, None, None).unwrap();
    assert_eq!(tile.as_f32().to_vec(), vec![4.0, 5.0, 8.0, 9.0]);
    assert_eq!(tile.shape, vec![2, 2]);
}

// ---- _is_contiguous: ported via observable load behavior ----

#[test]
fn strided_load_no_coords() {
    // 2x2 sub-tile with parent row stride 4 -> not contiguous; slow gather path.
    // Selects rows 0,1 cols 0,1 of the 4x4 parent -> [[0,1],[4,5]].
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(16), &[4, 4]);
    let tr = tile_ref(stick, &[2, 2], &[4, 1]);
    let tile = load_data(&mut ctx, &tr, None, None).unwrap();
    assert_eq!(tile.as_f32().to_vec(), vec![0.0, 1.0, 4.0, 5.0]);
}

#[test]
fn strided_store_no_coords() {
    // Scatter a 2x2 patch into a zeroed 4x4 via a row-strided (non-contiguous)
    // sub-tile; only the top-left 2x2 block is touched.
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &[0.0; 16], &[4, 4]);
    let tr = tile_ref(stick, &[2, 2], &[4, 1]);
    let patch = Tile::compute(vec![1.0, 2.0, 3.0, 4.0], DType::F16, vec![2, 2]);
    store_data(&mut ctx, &patch, &tr, None).unwrap();
    let result = read_back(&mut ctx, stick, &[4, 4]);
    let expected = vec![
        1.0, 2.0, 0.0, 0.0, //
        3.0, 4.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn non_rectangular_load_store() {
    // Upper-triangular (d1 >= d0) gather from a 4x4 tile, then scatter doubled.
    let mut ctx = single_core_context();
    let data = arange(16);
    let stick = alloc_f16(&mut ctx, &data, &[4, 4]);
    let tr = tile_ref(stick, &[4, 4], &[4, 1]);

    let css: AffineSet = parse_affine_set(
        "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d1 - d0 >= 0)>",
    )
    .unwrap();
    let coords = css.enumerate(&[4, 4], &[]);
    assert_eq!(coords.len(), 10);

    let tile = load_data(&mut ctx, &tr, Some(&coords), Some(vec![10])).unwrap();
    let expected_vals: Vec<f32> = coords
        .iter()
        .map(|c| data[(c[0] * 4 + c[1]) as usize])
        .collect();
    assert_eq!(tile.as_f32().to_vec(), expected_vals);

    let doubled: Vec<f32> = tile.as_f32().iter().map(|v| v * 2.0).collect();
    let doubled_tile = Tile::compute(doubled, DType::F16, vec![10]);
    store_data(&mut ctx, &doubled_tile, &tr, Some(&coords)).unwrap();

    let result = read_back(&mut ctx, stick, &[4, 4]);
    for r in 0..4i64 {
        for c in 0..4i64 {
            let orig = data[(r * 4 + c) as usize];
            let got = result[(r * 4 + c) as usize];
            if c >= r {
                assert_eq!(got, orig * 2.0, "upper-tri [{r},{c}]");
            } else {
                assert_eq!(got, orig, "lower-tri [{r},{c}] unchanged");
            }
        }
    }
}

#[test]
fn access_tile_order_inverted() {
    // coordinate_order (d0,d1)->(d1,d0): output position i gets element at the
    // swapped coord -> column-major traversal of the 3x3 data.
    let mut ctx = single_core_context();
    let data = arange(9);
    let stick = alloc_f16(&mut ctx, &data, &[3, 3]);
    let tr = tile_ref(stick, &[3, 3], &[3, 1]);

    // Row-major coords (0,0)..(2,2).
    let mut coords: Vec<Vec<i64>> = Vec::new();
    for r in 0..3i64 {
        for c in 0..3i64 {
            coords.push(vec![r, c]);
        }
    }
    let cso = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>").unwrap();
    assert!(!cso.is_identity());

    let remapped: Vec<Vec<i64>> = coords.iter().map(|pt| cso.eval(pt, &[])).collect();
    let tile = load_data(&mut ctx, &tr, Some(&remapped), Some(vec![9])).unwrap();

    let expected: Vec<f32> = remapped
        .iter()
        .map(|c| data[(c[0] * 3 + c[1]) as usize])
        .collect();
    assert_eq!(tile.as_f32().to_vec(), expected);
    // Column-major traversal: 0,3,6,1,4,7,2,5,8.
    assert_eq!(
        tile.as_f32().to_vec(),
        vec![0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]
    );
}

// ===========================================================================
// TestTileAccessEdgeCases — 3D tiles, stride > extent, zero-element shapes
// ===========================================================================

#[test]
fn tile_access_3d_identity() {
    // 3D parent 2x3x4 strides [12,4,1]; indices [1,1,2].
    // offset = 1*12 + 1*4 + 2*1 = 18 elems = 36 bytes (f16).
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(24), &[2, 3, 4]);
    let parent = hbm_memref(stick, &[2, 3, 4], &[12, 4, 1], DType::F16);
    let tr = tile_access(
        parent,
        &[1, 1, 2],
        &[1, 1, 1],
        "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
    );
    assert_eq!(tr.base_ptr, stick * STICK_BYTES + 18 * 2);
}

#[test]
fn tile_access_3d_load() {
    // Access starting at [0,1,0], load a 1x2x4 contiguous sub-tile.
    // offset 4, contiguous 8 elements -> [[[4..7],[8..11]]].
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(24), &[2, 3, 4]);
    let parent = hbm_memref(stick, &[2, 3, 4], &[12, 4, 1], DType::F16);
    let tr = tile_access(
        parent,
        &[0, 1, 0],
        &[1, 2, 4],
        "affine_map<(d0, d1, d2) -> (d0, d1, d2)>",
    );
    let tile = load_data(&mut ctx, &tr, None, None).unwrap();
    assert_eq!(
        tile.as_f32().to_vec(),
        vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
    assert_eq!(tile.shape, vec![1, 2, 4]);
}

#[test]
fn non_contiguous_stride_larger_than_extent() {
    // 2x2 sub-tile with strides [8,1] over a 4x4 parent -> rows 0 and 2.
    // Gathered: [[0,1],[8,9]].
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &arange(16), &[4, 4]);
    let tr = tile_ref(stick, &[2, 2], &[8, 1]);
    let tile = load_data(&mut ctx, &tr, None, None).unwrap();
    assert_eq!(tile.as_f32().to_vec(), vec![0.0, 1.0, 8.0, 9.0]);
}

#[test]
fn non_contiguous_store_stride_larger_than_extent() {
    // Scatter into rows 0 and 2 via strides [8,1].
    let mut ctx = single_core_context();
    let stick = alloc_f16(&mut ctx, &[0.0; 16], &[4, 4]);
    let tr = tile_ref(stick, &[2, 2], &[8, 1]);
    let patch = Tile::compute(vec![10.0, 20.0, 30.0, 40.0], DType::F16, vec![2, 2]);
    store_data(&mut ctx, &patch, &tr, None).unwrap();
    let result = read_back(&mut ctx, stick, &[4, 4]);
    let expected = vec![
        10.0, 20.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        30.0, 40.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(result, expected);
}

// ===========================================================================
// _is_contiguous predicate — no public surface; checked via behavior above.
// These stubs document the pure-predicate Python cases.
// ===========================================================================

#[test]
fn is_contiguous_predicate() {
    use ktir_emulator::ops_memory::is_contiguous;
    assert!(is_contiguous(&[3, 4], &[4, 1]));
    assert!(is_contiguous(&[5], &[1]));
    assert!(!is_contiguous(&[3, 4], &[8, 1])); // row stride too large (sub-tile)
    assert!(!is_contiguous(&[3, 4], &[4, 2])); // col stride > 1
}

#[test]
fn is_contiguous_3d_predicate() {
    use ktir_emulator::ops_memory::is_contiguous;
    assert!(is_contiguous(&[2, 3, 4], &[12, 4, 1])); // row-major 3-D
    assert!(!is_contiguous(&[2, 3, 4], &[24, 4, 1])); // outer stride too large
}

// ===========================================================================
// TestTileOps — affine attributes preserved after parsing the example MLIR.
// ===========================================================================

/// Resolve the indirect-access-copy example MLIR path relative to the repo root
/// (the crate lives at `<repo>/rust/crates/ktir-emulator`, examples at `<repo>/examples`).
fn indirect_access_copy_path() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // ktir-emulator -> crates
    p.pop(); // crates    -> rust
    p.pop(); // rust       -> repo root
    p.push("examples/rfc/indirect-access-copy.mlir");
    p
}

#[test]
fn affine_attrs_preserved() {
    let path = indirect_access_copy_path();
    if !path.exists() {
        // The example is part of the Python tree; if absent in this checkout the
        // structural assertions cannot run.
        panic!("missing example MLIR: {}", path.display());
    }
    let src = std::fs::read_to_string(&path).expect("read example");
    let module = ktir_emulator::parser::parse_module(&src).expect("parse module");

    // At least one construct_access_tile op, each carrying a base_map AffineMap.
    let mut saw_access_tile = false;
    let mut saw_memory_view = false;
    for func in module.functions.values() {
        for op in &func.operations {
            if op.op_type == "ktdp.construct_access_tile" {
                saw_access_tile = true;
                assert!(
                    matches!(op.attributes.get("base_map"), Some(Attr::AffineMap(_))),
                    "base_map missing or not an AffineMap on construct_access_tile"
                );
                // coordinate_set is either absent (full tile) or an AffineSet.
                match op.attributes.get("coordinate_set") {
                    None => {}
                    Some(Attr::AffineSet(_)) => {}
                    Some(other) => panic!("coordinate_set present but not an AffineSet: {other:?}"),
                }
            }
            if op.op_type == "ktdp.construct_memory_view" {
                saw_memory_view = true;
            }
        }
    }
    assert!(saw_access_tile, "no construct_access_tile op found");
    assert!(saw_memory_view, "no construct_memory_view op found");
}

#[test]
fn base_map_always_present() {
    let path = indirect_access_copy_path();
    let src = std::fs::read_to_string(&path).expect("read example");
    let module = ktir_emulator::parser::parse_module(&src).expect("parse module");

    let mut saw = false;
    for func in module.functions.values() {
        for op in &func.operations {
            if op.op_type == "ktdp.construct_access_tile" {
                saw = true;
                assert!(
                    matches!(op.attributes.get("base_map"), Some(Attr::AffineMap(_))),
                    "base_map missing or not an AffineMap on {:?}",
                    op.result
                );
            }
        }
    }
    assert!(saw, "no construct_access_tile op found");
}
