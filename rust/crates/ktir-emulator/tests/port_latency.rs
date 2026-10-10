// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_latency.py` — the execution-latency model.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! The Python suite splits into two groups:
//!
//! 1. **End-to-end interpreter tests** — `TestVectorAddLatency`,
//!    `TestRoofline`, `TestSoftmaxLatency`, `TestReduceLatency`,
//!    `TestMatmulLatency`, `TestLatencyDisabled`, and the end-to-end half of
//!    `TestIndirectAccessLatency`. These build a `KTIRInterpreter`, load an
//!    example `.mlir`, run `execute_function`, and read back
//!    `interp.get_latency_report()`. The Rust crate's `LatencyTracker` is **not
//!    yet wired into the interpreter** — there is no public path to thread a
//!    tracker through `execute_function` and recover a report. Every test in
//!    this group is therefore **skipped** (see the integrator note at the
//!    bottom of this file). They are not ported as `#[ignore]` stubs because
//!    they need an entire interpreter feature that does not exist, not merely a
//!    fixture.
//!
//! 2. **Unit-level cost-formula / helper tests** — `TestLatencyEdgeCases`
//!    (`_data_size` / `_num_elements` on zero-element and LX tiles, empty
//!    counters bottleneck) and the helper half of `TestIndirectAccessLatency`
//!    (`_data_size` over int sidebands, IAT operands, etc.). These exercise the
//!    cost model directly and **are ported here**.
//!
//! Rust API mapping:
//!   * Python's `LatencyTracker._data_size(result, operands)` and
//!     `_num_elements(result, operands)` are **private** free functions in
//!     `latency.rs` and are not callable from an integration test. Their
//!     behaviour is observable through the public surface: `record_op` charges
//!     `total_bytes` from `_data_size` and `compute_cycles` / `total_flops`
//!     from `_num_elements`, so each Python helper assertion is reproduced by
//!     driving `record_op` and reading the resulting `CoreLatencyCounters`.
//!   * Python's `LatencyReport(config=..., counters={})` (empty report) maps to
//!     a freshly-constructed `LatencyTracker` whose `report()` has no counters.
//!   * The store-sideband behaviour (Python passes a bare `int` as the
//!     `result`) maps to `Value::Index(n)`.

use ktir_emulator::dtypes::DType;
use ktir_emulator::ir::Value;
use ktir_emulator::latency::{HardwareConfig, LatencyCategory, LatencyTracker};
use ktir_emulator::memref::{DimSubscript, IndirectAccessTile, MemRef, MemorySpace};
use ktir_emulator::parser_ast::parse_affine_set;
use ktir_emulator::tile::Tile;

use std::collections::HashMap;

const STICK_BYTES: u64 = 128;

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn hbm_memref(shape: Vec<usize>, dtype: DType) -> MemRef {
    MemRef {
        base_ptr: 0,
        shape,
        strides: vec![1],
        space: MemorySpace::Hbm,
        dtype,
        coordinate_set: None,
    }
}

fn lx_memref(shape: Vec<usize>, dtype: DType) -> MemRef {
    MemRef {
        base_ptr: 0,
        shape,
        strides: vec![1],
        space: MemorySpace::Lx { core_id: None },
        dtype,
        coordinate_set: None,
    }
}

/// HBM load result Tile with the stick bookkeeping the load path stamps.
fn load_result(unique_sticks: usize, idx_sticks: Option<usize>) -> Option<Value> {
    let mut t = Tile::compute(vec![0.0; 16], DType::F16, vec![4, 4]);
    t.unique_sticks = Some(unique_sticks);
    t.index_unique_sticks = idx_sticks;
    Some(Value::Tile(t))
}

/// Build the all-LX-index-views IAT from
/// `test_lx_index_views_excluded_from_hbm_bytes`: HBM parent, two LX index
/// views.
fn lx_index_iat() -> IndirectAccessTile {
    let vss = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>").unwrap();
    let lx_idx = lx_memref(vec![4, 4], DType::I32);
    let parent = hbm_memref(vec![4, 4], DType::F16);
    IndirectAccessTile {
        parent_ref: parent,
        shape: vec![4, 4],
        dim_subscripts: vec![],
        index_views: vec![lx_idx.clone(), lx_idx],
        variables_space_set: vss,
        variables_space_order: None,
        extra: HashMap::new(),
    }
}

/// Build the single-HBM-index-view IAT used by the int-sideband / operand
/// branch tests. `_idx_unique_sticks_no_reads(iat)` over an HBM index view
/// would itself give a positive count if the operand branch ever fired.
fn hbm_index_iat(shape: Vec<usize>) -> IndirectAccessTile {
    let vss = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>").unwrap();
    let idx_view = hbm_memref(shape.clone(), DType::I32);
    IndirectAccessTile {
        parent_ref: idx_view.clone(),
        shape,
        dim_subscripts: vec![DimSubscript::Indirect {
            view: 0,
            idx_exprs: vec![],
        }],
        index_views: vec![idx_view],
        variables_space_set: vss,
        variables_space_order: None,
        extra: HashMap::new(),
    }
}

// ===========================================================================
// TestLatencyEdgeCases — zero-element / LX-only tiles, empty counters.
// ===========================================================================

#[test]
fn zero_element_tile_latency() {
    // Port of test_zero_element_tile_latency. A zero-element Tile reports
    // zero bytes and zero FLOPs. Python calls LatencyTracker._data_size /
    // _num_elements directly; here we observe the same through record_op on
    // an HBM memory op (so _data_size runs) and a compute op (so
    // _num_elements runs).
    //
    // unique_sticks=0 honours the HBM-load contract: a zero-element load
    // spans zero sticks => zero bytes => zero memory cycles.
    let mut t = LatencyTracker::new(HardwareConfig::default());
    let mut zero_tile = Tile::compute(vec![], DType::F16, vec![0]);
    zero_tile.unique_sticks = Some(0);
    let operands = [Some(Value::MemRef(hbm_memref(vec![0], DType::F16)))];
    t.record_op(
        0,
        "ktdp.load",
        LatencyCategory::Memory,
        &Some(Value::Tile(zero_tile)),
        &operands,
    );
    let c = &t.counters()[&0];
    // _data_size => 0 bytes.
    assert_eq!(c.total_bytes, 0);
    assert_eq!(c.memory_cycles, 0.0);

    // _num_elements => 0 elements on a (0,) compute tile => 0 flops / cycles.
    let mut t2 = LatencyTracker::new(HardwareConfig::default());
    let res = Some(Value::Tile(Tile::compute(vec![], DType::F32, vec![0])));
    t2.record_op(0, "arith.addf", LatencyCategory::ComputeFloat, &res, &[]);
    let c2 = &t2.counters()[&0];
    assert_eq!(c2.total_flops, 0.0);
    assert_eq!(c2.compute_cycles, 0.0);
}

#[test]
fn lx_index_views_excluded_from_hbm_bytes() {
    // Port of test_lx_index_views_excluded_from_hbm_bytes.
    //   data side = unique_sticks * 128 = 1 * 128
    //   idx side  = index_unique_sticks * 128 = 0 * 128 (all-LX views)
    // _memory_space([iat]) falls back to the HBM parent, so the HBM memory
    // path runs and charges 1 stick of data + 0 idx sticks = 128 bytes.
    let mut t = LatencyTracker::new(HardwareConfig::default());
    let iat = lx_index_iat();
    // 4x4 f16 = 32 bytes — fits within one 128-byte stick (unique_sticks=1);
    // index_unique_sticks=0 honours the IAT-load contract for an all-LX IAT.
    let result = load_result(1, Some(0));
    let operands = [Some(Value::IndirectAccessTile(iat))];
    t.record_op(0, "ktdp.load", LatencyCategory::Memory, &result, &operands);
    let c = &t.counters()[&0];
    // Total stays stick-granular (1 * 128), not data.nbytes; LX idx adds nothing.
    assert_eq!(c.total_bytes, STICK_BYTES);
    // Parent is HBM => memory cycles charged (not the LX free path).
    assert!(c.memory_cycles > 0.0);
}

#[test]
fn empty_counters_bottleneck() {
    // Port of test_empty_counters_bottleneck. A report with no counters
    // reports bottleneck="none", kernel_cycles=0, kernel_time_us=0.
    let t = LatencyTracker::new(HardwareConfig::default());
    let rep = t.report();
    assert_eq!(rep.bottleneck(), "none");
    assert_eq!(rep.kernel_cycles(), 0.0);
    assert_eq!(rep.kernel_time_us(), 0.0);
}

#[test]
fn empty_report_roofline() {
    // Port of TestRoofline::test_empty_report_roofline. roofline() on an
    // empty report returns the empty answer (Python: {}; Rust: None).
    let t = LatencyTracker::new(HardwareConfig::default());
    assert!(t.report().roofline().is_none());
}

// ===========================================================================
// TestIndirectAccessLatency — the helper-level (_data_size) cases.
//
// These exercise the stick-counting cost formula directly. Python calls
// LatencyTracker._data_size(result, operands); here the same formula is
// observed through record_op's total_bytes on the Memory category.
// ===========================================================================

#[test]
fn data_size_uses_unique_sticks_for_gather_result() {
    // Port of test_data_size_uses_unique_sticks_for_gather_result.
    // 64 f16 elements = 128 bytes packed, but scattered across 64 sticks
    // (each element on its own stick): actual traffic = 64 * 128 = 8192.
    let mut t = LatencyTracker::new(HardwareConfig::default());
    let mut result = Tile::compute(vec![0.0; 64], DType::F16, vec![64]);
    result.unique_sticks = Some(64);
    // HBM operand so the memory path runs through _data_size.
    let operands = [Some(Value::MemRef(hbm_memref(vec![64], DType::F16)))];
    t.record_op(
        0,
        "ktdp.load",
        LatencyCategory::Memory,
        &Some(Value::Tile(result)),
        &operands,
    );
    assert_eq!(t.counters()[&0].total_bytes, 64 * STICK_BYTES);
}

#[test]
fn data_size_charges_index_unique_sticks() {
    // Port of test_data_size_charges_index_unique_sticks.
    // data side  = unique_sticks * 128 = 1 * 128
    // idx side   = index_unique_sticks * 128 = 3 * 128
    // _data_size returns the sum => (1 + 3) * 128 = 512.
    let mut t = LatencyTracker::new(HardwareConfig::default());
    let mut result = Tile::compute(vec![0.0; 64], DType::F16, vec![64]);
    result.unique_sticks = Some(1);
    result.index_unique_sticks = Some(3);
    let operands = [Some(Value::MemRef(hbm_memref(vec![64], DType::F16)))];
    t.record_op(
        0,
        "ktdp.load",
        LatencyCategory::Memory,
        &Some(Value::Tile(result)),
        &operands,
    );
    assert_eq!(t.counters()[&0].total_bytes, (1 + 3) * STICK_BYTES);
}

#[test]
fn data_size_iat_load_skips_operand_branch_when_result_field_set() {
    // Port of the parametrized
    // test_data_size_iat_load_skips_operand_branch_when_result_field_set.
    // When result.index_unique_sticks is set (any int, including 0), the IAT
    // operand branch is skipped — the load routes through the result field,
    // sidestepping the side-channel _idx_unique_sticks_no_reads(iat) charge.
    // Expected: (data sticks + idx sticks from result field) * 128, with NO
    // extra operand-branch double-charge from the HBM index view.
    for idx_sticks in [5usize, 0usize] {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let iat = hbm_index_iat(vec![4]);
        let mut result = Tile::compute(vec![0.0; 4], DType::F16, vec![4]);
        result.unique_sticks = Some(2);
        result.index_unique_sticks = Some(idx_sticks);
        let operands = [Some(Value::IndirectAccessTile(iat))];
        t.record_op(
            0,
            "ktdp.load",
            LatencyCategory::Memory,
            &Some(Value::Tile(result)),
            &operands,
        );
        assert_eq!(
            t.counters()[&0].total_bytes,
            (2 + idx_sticks as u64) * STICK_BYTES,
            "idx_sticks={idx_sticks}: operand branch must not double-charge",
        );
    }
}

#[test]
fn data_size_int_sideband_charges_stick_bytes() {
    // Port of test_data_size_int_sideband_charges_stick_bytes. The store
    // handler propagates the int unique_sticks as the op result (Value::Index
    // in Rust). _data_size returns result * STICK_BYTES; operands are ignored
    // on the int branch.
    let mut t = LatencyTracker::new(HardwareConfig::default());
    // operands [iat, src] are ignored on the int branch.
    let iat = hbm_index_iat(vec![4]);
    let src = Tile::compute(vec![0.0; 4], DType::F16, vec![4]);
    let operands = [Some(Value::IndirectAccessTile(iat)), Some(Value::Tile(src))];
    t.record_op(
        0,
        "ktdp.store",
        LatencyCategory::Memory,
        &Some(Value::Index(4)),
        &operands,
    );
    assert_eq!(t.counters()[&0].total_bytes, 4 * STICK_BYTES);
}

#[test]
fn data_size_int_sideband_direct_store_64x64_scatter() {
    // Port of test_data_size_int_sideband_direct_store_64x64_scatter.
    // Direct store cost is stick-granular, not source-tile bytes. For a 64×64
    // f16 tile scattered to 100 distinct sticks, HBM traffic is 100 * 128 =
    // 12800 bytes — differs from the logical 64*64*2 = 8192 bytes.
    let mut t = LatencyTracker::new(HardwareConfig::default());
    t.record_op(
        0,
        "ktdp.store",
        LatencyCategory::Memory,
        &Some(Value::Index(100)),
        &[],
    );
    assert_eq!(t.counters()[&0].total_bytes, 100 * STICK_BYTES);
    // Stick-granular cost differs from logical bytes by a non-trivial margin.
    assert_ne!(100 * STICK_BYTES, 64 * 64 * 2);
}

// ---------------------------------------------------------------------------
// Cross-checks that the int-sideband path is independent of the operand list.
// (Python asserts the operands are ignored on the int branch via separate
// 64×64-scatter and IAT-operand cases above; this pins the invariant
// directly: the same sideband int yields the same bytes regardless of
// operands.)
// ---------------------------------------------------------------------------

#[test]
fn int_sideband_ignores_operands() {
    let mut bare = LatencyTracker::new(HardwareConfig::default());
    bare.record_op(
        0,
        "ktdp.store",
        LatencyCategory::Memory,
        &Some(Value::Index(7)),
        &[],
    );

    let mut with_ops = LatencyTracker::new(HardwareConfig::default());
    let iat = hbm_index_iat(vec![4]);
    let src = Tile::compute(vec![0.0; 4], DType::F16, vec![4]);
    let operands = [Some(Value::IndirectAccessTile(iat)), Some(Value::Tile(src))];
    with_ops.record_op(
        0,
        "ktdp.store",
        LatencyCategory::Memory,
        &Some(Value::Index(7)),
        &operands,
    );

    assert_eq!(
        bare.counters()[&0].total_bytes,
        with_ops.counters()[&0].total_bytes
    );
    assert_eq!(bare.counters()[&0].total_bytes, 7 * STICK_BYTES);
}
