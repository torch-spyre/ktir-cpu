// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_ktir_simple.py` — simple end-to-end / integration tests.
//!
//! The Python file mixes three loose top-level functions (`test_basic_execution`,
//! `test_memory_hierarchy`, `test_grid_execution`), plus a `TestMemorySimulator`
//! class covering the flat byte-addressed memory simulators in isolation.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * `test_basic_execution` loads a minimal `grid = [1,1,1]` module and confirms
//!   it parses. The Rust analogue parses with `parse_module` and additionally
//!   runs it through `execute_function` (the body is just `return`, so the result
//!   map is empty) to confirm end-to-end execution.
//!
//! * Memory model divergence (load-bearing): the Python `HBMSimulator` /
//!   `LXScratchpad` expose a *typed* `read(ptr, n, dtype)` / `write(ptr, arr)`
//!   API, raise `ValueError("unmapped ...")` on stray reads, and `HBMSimulator
//!   .allocate` returns a *byte* pointer. The Rust simulators expose a *byte*
//!   API (`read_bytes` / `write_bytes`), ZERO-PAD unmapped/out-of-range reads
//!   (no error), and `HBMSimulator::allocate` returns a *stick* address
//!   (`byte_ptr / STICK_BYTES`). Typed round-trips are reproduced here by
//!   encoding/decoding through `ktir_emulator::codec`. Cases that assert the
//!   `ValueError("unmapped")` behaviour have NO faithful Rust analogue (Rust
//!   zero-pads by design) and are `#[ignore]`d with a GAP note; the zero-pad
//!   behaviour itself IS positively checked instead.
//!
//! * `track_lx` capacity enforcement: Python raises `MemoryError`; Rust
//!   `CoreContext::track_lx` returns `Err(..)`. Checked via `is_err()`.
//!
//! * `test_grid_execution`'s `GridExecutor.cores` / `core.grid_pos` /
//!   `get_cores_in_group(group)` API is NOT present on the Rust `GridExecutor`
//!   (which exposes only `num_cores` + `linear_to_grid` / `grid_to_linear`). The
//!   per-core list and the wildcard core-group selection are a real feature gap;
//!   that sub-behaviour is `#[ignore]`d with a GAP note. The coordinate transforms
//!   that DO exist (grid shape, num_cores, per-core grid position) are checked
//!   faithfully via `linear_to_grid`.

use ktir_emulator::codec;
use ktir_emulator::context::CoreContext;
use ktir_emulator::dtypes::DType;
use ktir_emulator::env::GridExecutor;
use ktir_emulator::interpreter::execute_function;
use ktir_emulator::memory::{HBMSimulator, LXScratchpad, STICK_BYTES, SpyreMemoryHierarchy};
use ktir_emulator::parser::parse_module;

// ===========================================================================
// Helpers: typed read/write over the byte-addressed Rust simulators, via codec.
// ===========================================================================

/// Encode a flat f16 tile into raw bytes (Python `np.float16` array).
fn f16_bytes(data: &[f32]) -> Vec<u8> {
    codec::encode(data, DType::F16)
}

/// Decode `n` f16 elements from raw bytes.
fn read_f16(bytes: &[u8], n: usize) -> Vec<f32> {
    codec::decode(bytes, n, DType::F16)
}

fn close_slice(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-3, "{a:?} != {b:?}");
    }
}

// ===========================================================================
// Test 1: Basic execution (test_basic_execution)
// ===========================================================================

#[test]
fn basic_execution() {
    // Minimal module with a grid attribute that just returns. Confirm it parses
    // AND executes end-to-end (no tensor args -> empty output map).
    let ktir_text = r#"
module {
  func.func @add_test() attributes { grid = [1, 1, 1] } {
    return
  }
}
"#;
    let module = parse_module(ktir_text).expect("should parse KTIR text");
    let func = module.get_function("add_test").expect("function add_test");
    assert_eq!(func.grid, (1, 1, 1));

    let outputs = execute_function(&module, "add_test", &[]).expect("execution should succeed");
    assert!(outputs.is_empty(), "no tensor args -> no outputs");
}

// ===========================================================================
// Test 2: Memory hierarchy (test_memory_hierarchy)
// ===========================================================================

#[test]
fn hbm_typed_roundtrip() {
    // Python: hbm = HBMSimulator(size_gb=1); allocate, write f16 data, read back.
    let mut hbm = HBMSimulator::new(1);
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let bytes = f16_bytes(&data);
    // allocate returns a STICK address in Rust (Python returns a byte ptr).
    let stick = hbm.allocate(bytes.len() as i64);
    let byte_addr = stick * STICK_BYTES;
    hbm.write_bytes(byte_addr, &bytes);
    let read_back = read_f16(&hbm.read_bytes(byte_addr, bytes.len()), data.len());
    close_slice(&read_back, &data);
}

#[test]
fn lx_typed_roundtrip() {
    // Python: lx = LXScratchpad(size_mb=2, core_id=0); write/read f16 at ptr 0.
    let mut lx = LXScratchpad::new(0, 2);
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let bytes = f16_bytes(&data);
    lx.write_bytes(0, &bytes);
    let read_back = read_f16(&lx.read_bytes(0, bytes.len()), data.len());
    close_slice(&read_back, &data);
}

#[test]
fn lx_capacity_enforcement() {
    // Python: ctx.track_lx("%huge", 3MB) raises MemoryError against a 2MB LX.
    // Rust: CoreContext::track_lx returns Err on overflow.
    let mem = SpyreMemoryHierarchy::new(1);
    let mut ctx = CoreContext::new(
        0,
        (0, 0, 0),
        std::rc::Rc::clone(&mem.hbm),
        mem.get_lx(0),
        mem.lx_scratchpads.clone(),
    );
    // Default LX capacity is 2MB; 3MB must overflow.
    let three_mb = 3 * 1024 * 1024;
    assert!(
        ctx.track_lx("%huge", three_mb).is_err(),
        "3MB allocation should exceed the 2MB LX capacity"
    );
    // A within-capacity allocation succeeds.
    assert!(ctx.track_lx("%ok", 1024).is_ok());
}

// ===========================================================================
// Test 3: Grid execution (test_grid_execution)
// ===========================================================================

#[test]
fn grid_32_cores_positions() {
    // Python: GridExecutor(grid_shape=(32,1,1)); 32 cores; check core 0 and 31
    //         grid positions. Rust exposes num_cores + linear_to_grid.
    let grid = GridExecutor::new((32, 1, 1));
    assert_eq!(grid.num_cores, 32);
    assert_eq!(grid.linear_to_grid(0), (0, 0, 0));
    assert_eq!(grid.linear_to_grid(31), (31, 0, 0));
}

#[test]
fn grid_8x4_shape() {
    // Python: GridExecutor(grid_shape=(8,4,1)) -> 32 cores.
    let grid = GridExecutor::new((8, 4, 1));
    assert_eq!(grid.num_cores, 32);
    // Round-trip every core's (x,y,z) <-> linear id (coordinate transforms exist).
    for id in 0..grid.num_cores {
        let (x, y, z) = grid.linear_to_grid(id);
        assert_eq!(grid.grid_to_linear(x, y, z), id);
    }
}

#[test]
fn grid_core_group_selection() {
    use ktir_emulator::env::GridExecutor;
    // 4x2 grid: row 0 (y=0) -> ids 0..4.
    let g = GridExecutor::new((4, 2, 1));
    assert_eq!(g.cores_in_group((-1, 0, 0)), vec![0, 1, 2, 3]);
    // 8x4 grid: column 2 (x=2) across y -> one id per row at x=2.
    let g2 = GridExecutor::new((8, 4, 1));
    assert_eq!(g2.cores_in_group((2, -1, 0)), vec![2, 10, 18, 26]);
}

// ===========================================================================
// TestMemorySimulator: flat byte-addressed simulators in isolation.
//
// Python uses np.arange(16, f16).reshape(4,4) and typed read(ptr, n, dtype)
// with an optional `intra_byte` sub-offset. The Rust analogue is byte-addressed:
// element [r,c] of a 4x4 f16 array is at byte offset (r*4 + c) * 2.
// ===========================================================================

/// Build an HBM with a 4x4 f16 arange(16) written at a fresh allocation;
/// return (hbm, byte_addr_of_base).
fn make_hbm() -> (HBMSimulator, i64) {
    let mut hbm = HBMSimulator::default();
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let bytes = f16_bytes(&data);
    let stick = hbm.allocate(bytes.len() as i64);
    let byte_addr = stick * STICK_BYTES;
    hbm.write_bytes(byte_addr, &bytes);
    (hbm, byte_addr)
}

/// A fresh 2MB LX scratchpad (LXScratchpad has no Default impl).
fn fresh_lx() -> LXScratchpad {
    LXScratchpad::new(0, 2)
}

/// Build an LX with a 4x4 f16 arange(16) written at ptr 0; return (lx, 0).
fn make_lx() -> (LXScratchpad, i64) {
    let mut lx = fresh_lx();
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let bytes = f16_bytes(&data);
    lx.write_bytes(0, &bytes);
    (lx, 0)
}

// --- direct full-array read ---

#[test]
fn hbm_direct_read_exact_shape() {
    let (hbm, ptr) = make_hbm();
    let result = read_f16(&hbm.read_bytes(ptr, 32), 16);
    let expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
    assert_eq!(result, expected);
}

#[test]
fn lx_direct_read_exact_shape() {
    let (lx, ptr) = make_lx();
    let result = read_f16(&lx.read_bytes(ptr, 32), 16);
    let expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
    assert_eq!(result, expected);
}

// --- sub-allocation read (ptr inside an existing block) ---

#[test]
fn hbm_sub_allocation_read() {
    // Element [1,2] is at flat offset 6, byte offset 12.
    let (hbm, ptr) = make_hbm();
    let result = read_f16(&hbm.read_bytes(ptr + 12, 2), 1);
    assert_eq!(result[0], 6.0);
}

#[test]
fn lx_sub_allocation_read() {
    let (lx, ptr) = make_lx();
    let result = read_f16(&lx.read_bytes(ptr + 12, 2), 1);
    assert_eq!(result[0], 6.0);
}

#[test]
fn hbm_sub_allocation_read_row() {
    // Row 2 starts at flat offset 8, byte offset 16.
    let (hbm, ptr) = make_hbm();
    let result = read_f16(&hbm.read_bytes(ptr + 16, 8), 4);
    assert_eq!(result, vec![8.0, 9.0, 10.0, 11.0]);
}

#[test]
fn lx_sub_allocation_read_row() {
    let (lx, ptr) = make_lx();
    let result = read_f16(&lx.read_bytes(ptr + 16, 8), 4);
    assert_eq!(result, vec![8.0, 9.0, 10.0, 11.0]);
}

// --- sub-allocation write (ptr inside an existing block) ---

#[test]
fn hbm_sub_allocation_write() {
    // Write a single f16 element 99.0 at byte offset 12 (element [1,2]).
    let (mut hbm, ptr) = make_hbm();
    hbm.write_bytes(ptr + 12, &f16_bytes(&[99.0]));
    let result = read_f16(&hbm.read_bytes(ptr, 32), 16);
    assert_eq!(result[6], 99.0);
    let mut expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
    expected[6] = 99.0;
    assert_eq!(result, expected);
}

#[test]
fn lx_sub_allocation_write() {
    let (mut lx, ptr) = make_lx();
    lx.write_bytes(ptr + 12, &f16_bytes(&[99.0]));
    let result = read_f16(&lx.read_bytes(ptr, 32), 16);
    assert_eq!(result[6], 99.0);
    let mut expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
    expected[6] = 99.0;
    assert_eq!(result, expected);
}

// --- unmapped address: Python raises ValueError("unmapped"); Rust zero-pads. ---

#[test]
#[ignore = "GAP: Rust simulators zero-pad unmapped/out-of-range reads by design \
            (read_bytes returns zeros); they do NOT raise the Python \
            ValueError(\"unmapped ...\"). No faithful error-path analogue. The \
            zero-pad behaviour itself is checked by hbm_unmapped_zero_pads / \
            lx_unmapped_zero_pads."]
fn hbm_unmapped_raises() {
    // Python: hbm.read(0xDEAD, 4, "f16") raises ValueError(match="unmapped").
}

#[test]
#[ignore = "GAP: see hbm_unmapped_raises — Rust LXScratchpad zero-pads unmapped \
            reads instead of raising ValueError."]
fn lx_unmapped_raises() {
    // Python: lx.read(0xDEAD, 4, "f16") raises ValueError(match="unmapped").
}

#[test]
fn hbm_unmapped_zero_pads() {
    // Positive check of the actual Rust contract for an unmapped read.
    let hbm = HBMSimulator::default();
    assert_eq!(hbm.read_bytes(0xDEAD, 4), vec![0, 0, 0, 0]);
}

#[test]
fn lx_unmapped_zero_pads() {
    let lx = fresh_lx();
    assert_eq!(lx.read_bytes(0xDEAD, 4), vec![0, 0, 0, 0]);
}

// --- HBM and LX produce identical results for the same operations ---

#[test]
fn hbm_lx_sub_read_identical() {
    let (hbm, hbm_ptr) = make_hbm();
    let (lx, lx_ptr) = make_lx();
    for byte_offset in [0i64, 2, 12, 24, 30] {
        let hbm_val = hbm.read_bytes(hbm_ptr + byte_offset, 2);
        let lx_val = lx.read_bytes(lx_ptr + byte_offset, 2);
        assert_eq!(hbm_val, lx_val, "mismatch at byte_offset={byte_offset}");
    }
}

#[test]
fn hbm_lx_sub_write_identical() {
    let (mut hbm, hbm_ptr) = make_hbm();
    let (mut lx, lx_ptr) = make_lx();
    let patch = f16_bytes(&[77.0]);
    hbm.write_bytes(hbm_ptr + 12, &patch);
    lx.write_bytes(lx_ptr + 12, &patch);
    assert_eq!(hbm.read_bytes(hbm_ptr, 32), lx.read_bytes(lx_ptr, 32));
}
