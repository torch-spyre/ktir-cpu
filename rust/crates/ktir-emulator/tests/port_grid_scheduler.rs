// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_grid_scheduler.py` — the cross-core scheduler + ring
//! reduce, exercised through the crate's PUBLIC comm surface
//! (`ktir_emulator::interpreter::execute_function`, which drives
//! `ktir_emulator::comm_sched::execute_with_communication`).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * The Python test speaks to the *stable per-core comm surface* `CommOps`
//!   through a bespoke stub-handler harness (`_h_reduce`, `run_spec`,
//!   `RingReduceBackend`), seeds a distinct tile into each core's scope, runs the
//!   scheduler, and reads each core's scope back. That harness (the in-crate
//!   `run_capturing` analogue) is `#[cfg(test)]`/private in Rust, so it is not
//!   reachable from an integration test. Instead we drive the SAME scheduler and
//!   ring all-reduce through the public path: a real KTIR kernel executed by
//!   `execute_function`.
//!
//! * The Rust ring reduce is the comm op `ktdp.reduce`, driven by the scheduler
//!   (`comm_sched::make_comm_op` reads operands `[tile, core_group]`). The
//!   `core_group` is a `Value::Tuple` of core ids, which a kernel builds with
//!   `ktdp.coreid` (wildcard `-1` = "all cores in that axis"). Each test kernel
//!   therefore mirrors a Python spec: every core seeds its own tile from its
//!   compute-tile-id, runs `ktdp.reduce` over the right group, and stores the
//!   result so the harness can read per-core values back from HBM.
//!
//! * Per-core observation: Python checks `grid.cores[id].get_value(name)`. We
//!   cannot read a core's scope through the public API, so every core stores its
//!   result tile into a distinct row of a shared HBM output tensor; the test
//!   reads that tensor back via `execute_function`'s return value and checks one
//!   value per row. (1x128 f16 rows — element 0 of each row carries the reduced
//!   scalar, matching the Python `tile.data[0]` check.)
//!
//! Faithful coverage
//! -----------------
//! * `test_ring_reduce[2x1x1]`  -> [`ring_reduce_2x1x1`]      (5+7 = 12 on both)
//! * `test_ring_reduce[4x1x1]`  -> [`ring_reduce_4x1x1`]      (1+2+3+4 = 10 on all)
//! * out-of-group / singleton identity (the spec's "non-participant returns its
//!   input unchanged" invariant, exercised by the in-crate
//!   `core_outside_group_is_identity` and relied on by the multi-group specs)
//!   -> [`ring_reduce_singleton_group_is_identity`].
//! * independent cores with no comm op all run to completion
//!   -> [`independent_cores_run_to_completion`] (the public analogue of the
//!   "framework cannot deadlock under normal usage" claim).
//!
//! Skipped Python cases (see the run report's `skipped` field)
//! -----------------------------------------------------------
//! * `test_ring_reduce[4x4x1_rows]` / `[4x4x1_cols]`: these need each core's
//!   grid `y` (resp. `x`) coordinate to both pick its group and seed its tile.
//!   That requires the multi-result `%x, %y = ktdp.get_compute_tile_id` form,
//!   but the Rust parser keeps only the first result name (documented in
//!   `port_parse.rs`), so `%y` is unbound. The underlying behavior (concurrent
//!   disjoint-group ring reductions) is still covered by the 1-D ring tests; the
//!   2-D tiling is a parser limitation, not a scheduler one.
//! * `test_scheduler_detects_deadlock[mutual_recv|wrong_dest|extra_recv]`: these
//!   monkeypatch `RingReduceBackend.run` with a deliberately broken send/recv
//!   protocol and assert the scheduler raises "Deadlock detected". Rust exposes
//!   no public hook to inject a broken comm op — `RingReduce` (the only
//!   registered comm op) is hardwired to the correct protocol, so a deadlock is
//!   unreachable from the public API. The scheduler's deadlock *detector* is
//!   present (`comm_sched::execute_with_communication` returns
//!   `Err("Deadlock detected: ...")` when no core can progress); only the
//!   fault-injection seam is private. An `#[ignore]` stub records the gap.

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::parser::parse_module;

/// Build a ring-reduce kernel over an `n`-core 1-D grid.
///
/// Every core:
///   1. reads its compute-tile-id `%pid` (= its linear id on a `[n,1,1]` grid),
///   2. builds the reduction group with `ktdp.coreid` from `group_mask`
///      (`-1` = wildcard "all cores in that axis"),
///   3. seeds a `1x128` f16 tile splatting `pid*scale + base`,
///   4. runs `ktdp.reduce` (the scheduler-driven ring all-reduce),
///   5. stores the reduced tile into row `%pid` of the `n x 128` output.
///
/// Element 0 of each output row is the per-core reduced scalar — the analogue of
/// the Python spec's `tile.data[0]` check.
fn ring_kernel(n: usize, group_mask: (i64, i64, i64), base: f32, scale: f32) -> String {
    let upper = n - 1;
    let (mx, my, mz) = group_mask;
    format!(
        r#"
#full_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {upper} >= 0, d1 >= 0, -d1 + 127 >= 0)>
#row_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {{
  func.func @reduce_ring(%out_ptr: index) attributes {{grid = [{n}, 1, 1]}} {{
    %c0  = arith.constant 0 : index
    %mx  = arith.constant {mx} : index
    %my  = arith.constant {my} : index
    %mz  = arith.constant {mz} : index
    %pid = ktdp.get_compute_tile_id : index
    %group = ktdp.coreid %mx, %my, %mz
    %pidf  = arith.index_cast %pid : index to i32
    %pf    = arith.sitofp %pidf : i32 to f16
    %scale = arith.constant {scale:?} : f16
    %base  = arith.constant {base:?} : f16
    %sc    = arith.mulf %pf, %scale : f16
    %val   = arith.addf %sc, %base : f16
    %t = tensor.splat %val : tensor<1x128xf16>
    %r = ktdp.reduce %t, %group : tensor<1x128xf16> -> tensor<1x128xf16>
    %view = ktdp.construct_memory_view %out_ptr, sizes: [{n}, 128], strides: [128, 1] {{
      coordinate_set = #full_set, memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{n}x128xf16>
    %acc = ktdp.construct_access_tile %view[%pid, %c0] {{
      access_tile_set = #row_set, access_tile_order = #identity
    }} : memref<{n}x128xf16> -> !ktdp.access_tile<1x128xindex>
    ktdp.store %r, %acc : tensor<1x128xf16>, !ktdp.access_tile<1x128xindex>
    return
  }}
}}
"#
    )
}

/// Run a ring-reduce kernel and return element 0 of each of the `n` output rows
/// — one reduced scalar per core, in core-id order.
fn run_ring(n: usize, group_mask: (i64, i64, i64), base: f32, scale: f32) -> Vec<f32> {
    let src = ring_kernel(n, group_mask, base, scale);
    let module = parse_module(&src).unwrap_or_else(|e| panic!("parse failed: {e}\n{src}"));
    let out = execute_function(
        &module,
        "reduce_ring",
        &[(
            "%out_ptr",
            Arg::Tensor {
                data: vec![0.0; n * 128],
                shape: vec![n, 128],
                dtype: DType::F16,
            },
        )],
    )
    .expect("execute_function");
    let row = &out["%out_ptr"].data;
    (0..n).map(|i| row[i * 128]).collect()
}

// ===========================================================================
// Ring reduction (test_ring_reduce parametrize)
// ===========================================================================

/// SPEC_RING_REDUCE_2X1X1: seeds 5, 7 on a 2-core ring; both cores end at 12.
/// Group = wildcard over axis 0 = `[0, 1]`. Seed = `pid*2 + 5` -> {5, 7}.
#[test]
fn ring_reduce_2x1x1() {
    let results = run_ring(2, (-1, 0, 0), /*base=*/ 5.0, /*scale=*/ 2.0);
    // After the single ring round both participating cores hold a + b = 12.
    assert_eq!(results, vec![12.0, 12.0], "2-core ring sum");
}

/// SPEC_RING_REDUCE_4X1X1: seeds 1,2,3,4 on a 4-core ring; after N-1=3 rounds
/// every participating core holds the full sum 10. Group = `[0,1,2,3]`.
/// Seed = `pid*1 + 1` -> {1, 2, 3, 4}.
#[test]
fn ring_reduce_4x1x1() {
    let results = run_ring(4, (-1, 0, 0), /*base=*/ 1.0, /*scale=*/ 1.0);
    assert_eq!(results, vec![10.0, 10.0, 10.0, 10.0], "4-core ring sum");
}

// ===========================================================================
// Out-of-group / singleton identity
// ===========================================================================
// The multi-group Python specs (4x4 rows/cols) rely on `CommOps.reduce`
// returning the input tile unchanged for a core that is not in the active
// group; the in-crate `core_outside_group_is_identity` pins the same Rust
// behavior. Exercised here through the public path with a singleton group.

/// Group = `ktdp.coreid(0,0,0)` = `[0]`. Core 0 is a singleton group (one ring
/// member, no rounds -> identity); core 1 is not in the group (identity). Each
/// core therefore keeps its own seed: `pid + 1` -> {1, 2}. No core blocks.
#[test]
fn ring_reduce_singleton_group_is_identity() {
    let results = run_ring(2, (0, 0, 0), /*base=*/ 1.0, /*scale=*/ 1.0);
    assert_eq!(
        results,
        vec![1.0, 2.0],
        "singleton/out-of-group reduce is identity"
    );
}

// ===========================================================================
// Independent cores (no comm op) run to completion
// ===========================================================================
// The public analogue of the Python module's claim that "the framework cannot
// deadlock under normal usage": a kernel with no comm op drives every core
// straight to completion through the same scheduler, with no recv ever parking
// a core.

/// Four independent cores, no `ktdp.reduce`: each writes `pid + 1` to its own
/// output row. All four must complete (the scheduler removes each core on
/// `Poll::Done`), leaving rows {1, 2, 3, 4}.
#[test]
fn independent_cores_run_to_completion() {
    const N: usize = 4;
    let src = format!(
        r#"
#full_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#row_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {{
  func.func @independent(%out_ptr: index) attributes {{grid = [{N}, 1, 1]}} {{
    %c0  = arith.constant 0 : index
    %pid = ktdp.get_compute_tile_id : index
    %pidf = arith.index_cast %pid : index to i32
    %pf   = arith.sitofp %pidf : i32 to f16
    %one  = arith.constant 1.0 : f16
    %val  = arith.addf %pf, %one : f16
    %t = tensor.splat %val : tensor<1x128xf16>
    %view = ktdp.construct_memory_view %out_ptr, sizes: [{N}, 128], strides: [128, 1] {{
      coordinate_set = #full_set, memory_space = #ktdp.spyre_memory_space<HBM>
    }} : memref<{N}x128xf16>
    %acc = ktdp.construct_access_tile %view[%pid, %c0] {{
      access_tile_set = #row_set, access_tile_order = #identity
    }} : memref<{N}x128xf16> -> !ktdp.access_tile<1x128xindex>
    ktdp.store %t, %acc : tensor<1x128xf16>, !ktdp.access_tile<1x128xindex>
    return
  }}
}}
"#
    );
    let module = parse_module(&src).unwrap_or_else(|e| panic!("parse failed: {e}"));
    let out = execute_function(
        &module,
        "independent",
        &[(
            "%out_ptr",
            Arg::Tensor {
                data: vec![0.0; N * 128],
                shape: vec![N, 128],
                dtype: DType::F16,
            },
        )],
    )
    .expect("execute_function");
    let row = &out["%out_ptr"].data;
    let vals: Vec<f32> = (0..N).map(|i| row[i * 128]).collect();
    assert_eq!(
        vals,
        vec![1.0, 2.0, 3.0, 4.0],
        "every core ran to completion"
    );
}

// ===========================================================================
// Deadlock detection — skipped (no public fault-injection seam)
// ===========================================================================

/// Port of `test_scheduler_detects_deadlock`. The Python test monkeypatches
/// `RingReduceBackend.run` with a broken send/recv protocol and asserts the
/// scheduler raises "Deadlock detected". Rust exposes no hook to register a
/// broken comm op: `RingReduce` (the sole comm op) is hardwired to the correct
/// protocol, so a deadlock is unreachable from the public API. The detector
/// itself exists — `comm_sched::execute_with_communication` returns
/// `Err("Deadlock detected: ...")` when no core can make progress — but cannot
/// be triggered without private fault injection. Left as a documented gap.
#[test]
#[ignore = "no public seam to inject a broken comm op; RingReduce protocol is hardwired correct"]
fn scheduler_detects_deadlock() {
    // Intentionally empty: see the doc comment for why this is unreachable via
    // the public API.
}
