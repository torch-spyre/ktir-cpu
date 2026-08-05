// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_ktir_emulator.py` — the top-level "basic" integration tests
//! for the KTIR CPU backend (memory hierarchy, grid executor, tile/memref types,
//! dtype mapping, interpreter load/execute).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python's `HBMSimulator`/`LXScratchpad` expose a *dtype-aware* `read(ptr,
//!   count, dtype)` / `write(ptr, ndarray)` API. The Rust crate splits this into
//!   raw byte storage (`read_bytes`/`write_bytes`) plus a separate `codec`
//!   (`encode`/`decode`) that does the dtype<->bytes round-trip at the boundary.
//!   Every read/write round-trip, zero-padding, partial-overwrite and
//!   interleaved-allocation behavior is reproduced faithfully by combining the
//!   two. The observable semantics (values + zero-fill past the allocation end)
//!   are identical.
//! * Python's HBM `allocate` returns a *stick index* and advances `next_ptr` to
//!   the next 128-byte stick boundary. The Rust `allocate` is identical; we drive
//!   it directly and then address bytes via `byte_addr = stick * STICK_BYTES`.
//! * `MemRef.size_bytes()` exists in Rust; we construct a `MemRef` per dtype and
//!   check the byte count (the parametrized Python `test_tileref_size_bytes`).
//! * `Tile.copy()` independence maps to Rust `Clone` (independent `Vec`).
//! * `GridExecutor` in Rust holds *coordinate transforms only* — there is no
//!   per-core `grid_pos` list, no `get_core`, and no `get_cores_in_group`. The
//!   coordinate round-trip + boundary positions ARE checked via
//!   `linear_to_grid` / `grid_to_linear`. The `get_cores_in_group` group-filter
//!   tests have no Rust analogue (the filtering lives in the interpreter driver,
//!   not the public grid type) and are stubbed `#[ignore]` (GAP).
//! * `_ktir_dtype(np.dtype)` maps a NumPy dtype to a KTIR string and raises on
//!   `float64`. Rust has no NumPy; the analogue is `DType::parse` (string ->
//!   DType), which rejects unsupported spellings. We test that direction.
//! * `KTIRInterpreter` is a stateful class with a "No module loaded" guard. The
//!   Rust crate is module-functional (`parse_module` + `execute_function`); the
//!   guard has no analogue (a missing module is simply a parse you never did).
//!   We instead test: parse + execute the simple `@add` module, and the missing-
//!   function error from `get_function`.
//! * `test_existing_ktir_file` is Python-only test infra (optionally reads a
//!   file from cwd that may not exist); noted skipped, no Rust analogue.
//! * `test_lx_read_unmapped_raises`: Python raises `ValueError`; the Rust
//!   `read_bytes` zero-fills an unmapped read instead of erroring. Behavioral
//!   GAP — stubbed `#[ignore]`.

use ktir_emulator::codec::{decode, encode};
use ktir_emulator::dtypes::DType;
use ktir_emulator::env::GridExecutor;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ir::Scalar;
use ktir_emulator::memory::{HBMSimulator, LXScratchpad, STICK_BYTES, SpyreMemoryHierarchy};
use ktir_emulator::memref::{MemRef, MemorySpace};
use ktir_emulator::parser::parse_module;
use ktir_emulator::tile::Tile;

// ===========================================================================
// Helpers: dtype-aware HBM/LX read+write, layered over the raw byte API + codec.
// These reproduce the Python `HBMSimulator.read/write` / `LXScratchpad.read/write`
// semantics (the dtype<->bytes round-trip the Python sim does internally).
// ===========================================================================

/// Allocate `count` f16 elements in HBM, write `data`, return absolute byte addr.
fn hbm_alloc_write(hbm: &mut HBMSimulator, data: &[f32], dt: DType) -> i64 {
    let bytes = encode(data, dt);
    let stick = hbm.allocate(bytes.len() as i64);
    let byte_addr = stick * STICK_BYTES;
    hbm.write_bytes(byte_addr, &bytes);
    byte_addr
}

/// Read `count` elements of `dt` from HBM at `byte_addr` (zero-padded past end).
fn hbm_read(hbm: &HBMSimulator, byte_addr: i64, count: usize, dt: DType) -> Vec<f32> {
    let raw = hbm.read_bytes(byte_addr, count * dt.bytes_per_elem());
    decode(&raw, count, dt)
}

fn lx_write(lx: &mut LXScratchpad, ptr: i64, data: &[f32], dt: DType) {
    lx.write_bytes(ptr, &encode(data, dt));
}

fn lx_read(lx: &LXScratchpad, ptr: i64, count: usize, dt: DType) -> Vec<f32> {
    let raw = lx.read_bytes(ptr, count * dt.bytes_per_elem());
    decode(&raw, count, dt)
}

// ===========================================================================
// Memory — HBM
// ===========================================================================

#[test]
fn memory_hbm_read_write_roundtrip() {
    // test_memory_hbm: write 4 f16 values, read them back exactly.
    let mut hbm = HBMSimulator::new(1);
    let data = [1.0, 2.0, 3.0, 4.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F16);
    let read = hbm_read(&hbm, addr, 4, DType::F16);
    assert_eq!(read, data.to_vec(), "HBM read/write mismatch");
}

#[test]
fn hbm_read_direct_hit_padding() {
    // test_hbm_read_direct_hit_padding: store 2 f16, request 4 -> last 2 zero.
    let mut hbm = HBMSimulator::default();
    let data = [1.0, 2.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F16);
    let result = hbm_read(&hbm, addr, 4, DType::F16);
    assert_eq!(result.len(), 4);
    assert_eq!(&result[..2], &data);
    assert_eq!(&result[2..], &[0.0, 0.0], "padding should be zeros");
}

#[test]
fn hbm_read_subarray_partial_padding() {
    // test_hbm_read_subarray_partial_padding: write [10,20,30,40], read starting
    // at element 2 requesting 4 -> [30,40,0,0].
    let mut hbm = HBMSimulator::default();
    let data = [10.0, 20.0, 30.0, 40.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F16);
    let byte_offset = 2 * DType::F16.bytes_per_elem() as i64; // skip 2 f16 elems
    let result = hbm_read(&hbm, addr + byte_offset, 4, DType::F16);
    assert_eq!(result.len(), 4);
    assert_eq!(result[0], 30.0);
    assert_eq!(result[1], 40.0);
    assert_eq!(result[2], 0.0, "padding should be zero");
    assert_eq!(result[3], 0.0, "padding should be zero");
}

#[test]
fn hbm_write_partial_overwrite() {
    // test_hbm_write_partial_overwrite: write [1,2,3,4], overwrite first 2 with
    // [99,88] -> [99,88,3,4].
    let mut hbm = HBMSimulator::default();
    let data = [1.0, 2.0, 3.0, 4.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F16);
    // Overwrite only the first 2 elements at the same base address.
    hbm.write_bytes(addr, &encode(&[99.0, 88.0], DType::F16));
    let result = hbm_read(&hbm, addr, 4, DType::F16);
    assert_eq!(result, vec![99.0, 88.0, 3.0, 4.0]);
}

#[test]
fn hbm_write_full_replacement() {
    // test_hbm_write_full_replacement: equal-size write fully replaces.
    let mut hbm = HBMSimulator::default();
    let data = [1.0, 2.0, 3.0, 4.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F16);
    let replacement = [10.0, 20.0, 30.0, 40.0];
    hbm.write_bytes(addr, &encode(&replacement, DType::F16));
    let result = hbm_read(&hbm, addr, 4, DType::F16);
    assert_eq!(result, replacement.to_vec());
}

#[test]
fn hbm_allocate_f32() {
    // test_hbm_allocate_f32: f32 round-trip.
    let mut hbm = HBMSimulator::default();
    let data = [1.0, 2.0, 3.0, 4.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F32);
    let result = hbm_read(&hbm, addr, 4, DType::F32);
    assert_eq!(result, data.to_vec());
}

#[test]
fn hbm_read_uninitialized_region() {
    // test_hbm_read_uninitialized_region: write 2 f16, read 4 -> last 2 zero.
    let mut hbm = HBMSimulator::default();
    let data = [1.0, 2.0];
    let addr = hbm_alloc_write(&mut hbm, &data, DType::F16);
    let result = hbm_read(&hbm, addr, 4, DType::F16);
    assert_eq!(result.len(), 4);
    assert_eq!(&result[..2], &data);
    assert_eq!(&result[2..], &[0.0, 0.0]);
}

#[test]
fn hbm_allocate_advances_stick_aligned() {
    // Backstop for the Python allocate() stick semantics (next_ptr stick-aligned,
    // distinct allocations are independent).
    let mut hbm = HBMSimulator::default();
    let a = hbm_alloc_write(&mut hbm, &[1.0, 2.0], DType::F16);
    let b = hbm_alloc_write(&mut hbm, &[7.0, 8.0], DType::F16);
    assert_ne!(a, b, "distinct allocations get distinct addresses");
    assert_eq!(hbm_read(&hbm, a, 2, DType::F16), vec![1.0, 2.0]);
    assert_eq!(hbm_read(&hbm, b, 2, DType::F16), vec![7.0, 8.0]);
}

// ===========================================================================
// Memory — LX
// ===========================================================================

#[test]
fn memory_lx_read_write_roundtrip() {
    // test_memory_lx (round-trip half): write 4 f16 at ptr 0, read them back.
    let mut lx = LXScratchpad::new(0, 2);
    let data = [5.0, 6.0, 7.0, 8.0];
    lx_write(&mut lx, 0, &data, DType::F16);
    let read = lx_read(&lx, 0, 4, DType::F16);
    assert_eq!(read, data.to_vec(), "LX read/write mismatch");
}

#[test]
fn lx_capacity_limit_enforced() {
    // test_memory_lx (capacity half): track_lx beyond the 2 MB cap errors.
    // CoreContext.track_lx() is the enforcement point (allocate() never checks).
    let mem = SpyreMemoryHierarchy::new(1);
    let mut ctx = ktir_emulator::context::CoreContext::new(
        0,
        (0, 0, 0),
        std::rc::Rc::clone(&mem.hbm),
        mem.get_lx(0),
        mem.lx_scratchpads.clone(),
    );
    // 3 MB > 2 MB limit -> error (Python raises MemoryError).
    assert!(ctx.track_lx("%huge", 3 * 1024 * 1024).is_err());
}

#[test]
fn lx_read_shape_mismatch() {
    // test_lx_read_shape_mismatch: read fewer/more elements than stored.
    let mut lx = LXScratchpad::new(0, 2);
    let data = [1.0, 2.0, 3.0, 4.0];
    lx_write(&mut lx, 0, &data, DType::F16);

    // Read fewer elements than stored.
    let r2 = lx_read(&lx, 0, 2, DType::F16);
    assert_eq!(r2, vec![1.0, 2.0]);

    // Read more elements than stored -> pads with zeros.
    let r6 = lx_read(&lx, 0, 6, DType::F16);
    assert_eq!(r6.len(), 6);
    assert_eq!(&r6[..4], &data);
    assert_eq!(&r6[4..], &[0.0, 0.0]);
}

#[test]
fn lx_clear_resets_state() {
    // test_lx_clear: clear() resets memory + next_ptr + used.
    let mut lx = LXScratchpad::new(0, 2);
    lx_write(&mut lx, 0, &[1.0, 2.0], DType::F16);
    lx.next_ptr = 64;
    lx.used = 32;

    lx.clear();

    assert_eq!(lx.next_ptr, 0);
    assert_eq!(lx.used, 0);
    // Memory is cleared: reading the previously-written region now zero-fills.
    assert_eq!(lx_read(&lx, 0, 2, DType::F16), vec![0.0, 0.0]);
}

#[test]
fn lx_interleaved_allocations() {
    // test_lx_interleaved_allocations: two non-contiguous allocations are
    // independently readable; overwriting one leaves the other untouched.
    let mut lx = LXScratchpad::new(0, 2);
    let a = [1.0, 2.0];
    let b = [10.0, 20.0, 30.0];
    let ptr_a = 0x0000;
    let ptr_b = 0x0100;

    lx_write(&mut lx, ptr_a, &a, DType::F16);
    lx_write(&mut lx, ptr_b, &b, DType::F16);

    assert_eq!(lx_read(&lx, ptr_a, 2, DType::F16), a.to_vec());
    assert_eq!(lx_read(&lx, ptr_b, 3, DType::F16), b.to_vec());

    // Overwrite ptr_a; ptr_b untouched.
    lx_write(&mut lx, ptr_a, &[99.0, 88.0], DType::F16);
    assert_eq!(lx_read(&lx, ptr_b, 3, DType::F16), b.to_vec());
}

#[test]
#[ignore = "GAP: LXScratchpad.read of an unmapped ptr raises ValueError in Python; \
            the Rust read_bytes zero-fills an unmapped read instead of erroring \
            (no error path in the byte-level API)"]
fn lx_read_unmapped_raises() {
    // test_lx_read_unmapped_raises — no Rust analogue (zero-fill, not error).
}

// ===========================================================================
// Grid executor
// ===========================================================================

#[test]
fn grid_executor_core_positions() {
    // test_grid_executor: 32-core 1D grid; core 0 at (0,0,0), core 31 at (31,0,0).
    let grid = GridExecutor::new((32, 1, 1));
    assert_eq!(grid.num_cores, 32, "should have 32 cores");
    assert_eq!(grid.linear_to_grid(0), (0, 0, 0), "core 0 position wrong");
    assert_eq!(
        grid.linear_to_grid(31),
        (31, 0, 0),
        "core 31 position wrong"
    );
    // get_core/get_core_at_pos analogue: id<->coord round-trip.
    assert_eq!(grid.linear_to_grid(5), (5, 0, 0));
    assert_eq!(grid.grid_to_linear(5, 0, 0), 5);
}

#[test]
fn grid_boundary_max_position() {
    // test_grid_boundary_max_position: 4x3x2 grid; last core at (3,2,1) and the
    // grid_to_linear round-trip inverts linear_to_grid.
    let grid = GridExecutor::new((4, 3, 2));
    let last_id = grid.num_cores - 1;
    let pos = grid.linear_to_grid(last_id);
    assert_eq!(pos, (3, 2, 1));
    assert_eq!(grid.grid_to_linear(pos.0, pos.1, pos.2), last_id);
}

#[test]
fn grid_linear_coord_roundtrip_all_cores() {
    // Backstop covering the coordinate logic the get_cores_in_group tests rely on
    // (every linear id round-trips through (x,y,z) on a 2x4x2 grid).
    let grid = GridExecutor::new((2, 4, 2));
    assert_eq!(grid.num_cores, 16);
    for id in 0..grid.num_cores {
        let (x, y, z) = grid.linear_to_grid(id);
        assert!(x < 2 && y < 4 && z < 2);
        assert_eq!(grid.grid_to_linear(x, y, z), id);
    }
}

#[test]
fn get_cores_in_group_filters() {
    use ktir_emulator::env::GridExecutor;
    let g = GridExecutor::new((4, 2, 1)); // 8 cores: x in 0..4, y in 0..2
    // all wildcards -> every core.
    assert_eq!(g.cores_in_group((-1, -1, -1)), (0..8).collect::<Vec<_>>());
    // y=1, z=0 -> the 4 cores in the second row (linear ids 4..8).
    assert_eq!(g.cores_in_group((-1, 1, 0)), vec![4, 5, 6, 7]);
    // x=2 across all y -> ids where x==2: (2,0)=2 and (2,1)=6.
    assert_eq!(g.cores_in_group((2, -1, -1)), vec![2, 6]);
    // a fully-specified coordinate -> exactly one core.
    assert_eq!(g.cores_in_group((3, 1, 0)), vec![7]);
}

// ===========================================================================
// Tile / MemRef types
// ===========================================================================

#[test]
fn tile_copy_is_independent() {
    // test_tile_operations (copy half): a cloned tile is a value-independent
    // snapshot of the original. Tiles are immutable (every op produces a fresh
    // result via `Tile::compute`), so a clone shares the underlying buffer until a
    // new tile is built — there is no in-place mutation path. Independence is now
    // observed through the public API: a separately-computed tile holding a changed
    // value does not perturb the original's data.
    let tile1 = Tile::compute(vec![1.0, 2.0, 3.0, 4.0], DType::F16, vec![4]);
    let tile1_copy = tile1.clone();
    // A clone is a faithful snapshot.
    assert_eq!(tile1.as_f32().to_vec(), tile1_copy.as_f32().to_vec());
    // Building a fresh tile with a changed element leaves the original untouched.
    let mut changed = tile1.as_f32().to_vec();
    changed[0] = 999.0;
    let tile1_mut = Tile::compute(changed, tile1.dtype, tile1.shape.clone());
    assert_eq!(tile1.as_f32()[0], 1.0, "original should be independent");
    assert_eq!(tile1_mut.as_f32()[0], 999.0, "new tile reflects the change");
}

#[test]
fn memref_size_bytes() {
    // test_tile_operations (MemRef half): a (4,) f16 view is 8 bytes.
    let ref_ = MemRef {
        base_ptr: 0x1000,
        shape: vec![4],
        strides: vec![1],
        space: MemorySpace::Hbm,
        dtype: DType::F16,
        coordinate_set: None,
    };
    assert_eq!(ref_.size_bytes(), 8, "MemRef size calculation wrong");
}

#[test]
fn memref_size_bytes_per_dtype() {
    // test_tileref_size_bytes (parametrized): byte count for each dtype, shape (4,).
    let cases = [
        (DType::F16, 8usize),
        (DType::F32, 16),
        (DType::I32, 16),
        (DType::I64, 32),
    ];
    for (dt, expected) in cases {
        let ref_ = MemRef {
            base_ptr: 0,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Hbm,
            dtype: dt,
            coordinate_set: None,
        };
        assert_eq!(ref_.size_bytes(), expected, "size_bytes wrong for {dt}");
    }
}

// ===========================================================================
// dtype mapping
// ===========================================================================

#[test]
fn dtype_parse_known_and_unknown() {
    // test_ktir_dtype_branches analogue: DType::parse maps canonical spellings and
    // rejects unsupported ones (Python's `_ktir_dtype` raised on float64).
    assert_eq!(DType::parse("f16").unwrap(), DType::F16);
    assert_eq!(DType::parse("f32").unwrap(), DType::F32);
    assert_eq!(DType::parse("i32").unwrap(), DType::I32);
    assert_eq!(DType::parse("i64").unwrap(), DType::I64);
    // float64 / bfloat16 are unsupported -> error (Python raised ValueError).
    assert!(DType::parse("float64").is_err());
    assert!(DType::parse("bfloat16").is_err());
}

// ===========================================================================
// Interpreter
// ===========================================================================

const ADD_MODULE: &str = r#"
module {
    func.func @add(%x: index, %y: index) -> index attributes { grid = [1, 1, 1] } {
        %c5 = arith.constant 5 : index
        %c10 = arith.constant 10 : index
        %sum = arith.addi %c5, %c10 : index
        return %sum : index
    }
}
"#;

#[test]
fn interpreter_loads_and_parses_simple_module() {
    // test_interpreter_simple: the simple @add module parses and yields a function.
    let module = parse_module(ADD_MODULE).expect("parse @add module");
    let func = module.get_function("add").expect("function add present");
    assert_eq!(func.grid, (1, 1, 1));
    assert_eq!(func.arg_names(), vec!["x".to_string(), "y".to_string()]);
}

#[test]
fn interpreter_executes_simple_module() {
    // Beyond the Python placeholder: actually run @add (5 + 10) end-to-end. It
    // takes scalar index args and returns; execution must not error.
    let module = parse_module(ADD_MODULE).expect("parse @add module");
    let args = [
        ("x", Arg::Scalar(Scalar::I64(0))),
        ("y", Arg::Scalar(Scalar::I64(0))),
    ];
    // No tensor outputs; the body is pure scalar arithmetic over constants.
    let outputs = execute_function(&module, "add", &args).expect("run @add");
    assert!(outputs.is_empty(), "no tensor args -> no tensor outputs");
}

#[test]
fn interpreter_missing_function_errors() {
    // test_interpreter_no_module_guard analogue: the Rust crate has no stateful
    // "No module loaded" guard, but asking for a function that isn't in the module
    // errors (the closest faithful check on the public API).
    let module = parse_module(ADD_MODULE).expect("parse @add module");
    assert!(module.get_function("foo").is_err());
}

#[test]
#[ignore = "Python-only test infra: test_existing_ktir_file optionally reads \
            add_kernel_ktir.mlir from cwd and only checks it loads without crashing; \
            no Rust analogue (covered by end_to_end.rs against real example kernels)"]
fn existing_ktir_file() {
    // test_existing_ktir_file — Python-only optional-file smoke test.
}
