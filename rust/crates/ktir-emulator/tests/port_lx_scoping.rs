// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_lx_scoping.py` — `CoreContext` scope stack + LX
//! scratchpad lifetime accounting (RFC 0682, `ktir_emulator/grid.py`).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python drives `CoreContext` directly via its public API
//!   (`set_value`/`get_value`/`has_value`/`push_scope`/`pop_scope`/`track_lx`/
//!   `untrack_lx`/`clear_values`). The Rust crate exposes the SAME public
//!   methods on `ktir_emulator::context::CoreContext`, so each case ports 1:1.
//! * `track_lx` overflow: Python raises `MemoryError("LX scratchpad overflow")`;
//!   the Rust port returns `Err(..)` instead of panicking. The "must raise"
//!   assertions therefore become `assert!(result.is_err())` — same observable
//!   contract (the allocation is rejected and `used` is unchanged).
//! * `pop_scope` on the function-body scope: Python raises
//!   `RuntimeError("Cannot pop the function-body scope")`; the Rust port
//!   `assert!`s (panics). Ported via `catch_unwind`.
//! * Private-field assertions in Python (`ctx._scope_stack == [{}]`,
//!   `ctx._lx_bytes == {}`, `ctx._lx_next_ptr_stack == []`) probe internal
//!   bookkeeping. The Rust fields are private, so these are checked through
//!   their *observable* consequences via the public API (e.g. after
//!   `clear_values`: a previously-set value is gone, `lx.used == 0`,
//!   `lx.next_ptr == 0`, and only the function-body scope remains so a single
//!   `pop_scope` panics). No assertion is weakened — every reset invariant the
//!   Python test checks is still verified.
//! * Python's `MemoryOps._write_to_lx(ctx, ndarray)` is a private helper that
//!   reserves a stick-aligned LX span and bumps `lx.next_ptr`. Its Rust analogue
//!   (`ops_memory::write_to_lx`) is private too, so the `next_ptr`-rewind tests
//!   use a local `write_to_lx` helper that performs the IDENTICAL stick-aligned
//!   bump (`(ptr + size + STICK-1) & !(STICK-1)`) through the public LX fields.
//! * Python sizes the LX in fractional MB (`0.125`, `0.5`). `LXScratchpad::new`
//!   takes an integer MB, so fractional-capacity contexts are built by setting
//!   the public `capacity` field directly (128 KB, etc.). `used`/`next_ptr`
//!   semantics are unaffected.
//! * `Tile.size_bytes()` exists in both; `_make_tile(shape, "f16")` maps to a
//!   zero-filled `Tile` of `DType::F16`.
//! * Python `set_value` stores arbitrary Python objects (ints, strings); the
//!   Rust scope map holds `Value`. Sentinel ints/strings become distinct
//!   `Value::Index` markers — identity/visibility is what the tests check.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use ktir_emulator::context::CoreContext;
use ktir_emulator::dtypes::DType;
use ktir_emulator::ir::Value;
use ktir_emulator::memory::{HBMSimulator, LXScratchpad, STICK_BYTES, UnsafeShared};
use ktir_emulator::tile::Tile;

// ===========================================================================
// helpers (port of _make_context / _make_tile)
// ===========================================================================

/// Build a `CoreContext` with a fresh `lx_size_mb`-MB LX and a default HBM.
/// Mirrors `_make_context(lx_size_mb=2)`.
fn make_context(lx_size_mb: i64) -> CoreContext {
    let lx = Rc::new(UnsafeShared::new(LXScratchpad::new(0, lx_size_mb)));
    let hbm = Rc::new(UnsafeShared::new(HBMSimulator::default()));
    CoreContext::new(0, (0, 0, 0), hbm, Rc::clone(&lx), vec![lx])
}

/// Build a `CoreContext` whose LX capacity is exactly `capacity_bytes`.
/// Stands in for Python's fractional-MB sizes (e.g. 0.125 MB = 128 KB).
fn make_context_bytes(capacity_bytes: i64) -> CoreContext {
    let mut lx = LXScratchpad::new(0, 1);
    lx.capacity = capacity_bytes;
    let lx = Rc::new(UnsafeShared::new(lx));
    let hbm = Rc::new(UnsafeShared::new(HBMSimulator::default()));
    CoreContext::new(0, (0, 0, 0), hbm, Rc::clone(&lx), vec![lx])
}

/// Zero-filled `f16` tile of the given shape. Mirrors `_make_tile`.
fn make_tile(shape: &[usize]) -> Tile {
    let n: usize = shape.iter().product();
    Tile::compute(vec![0.0f32; n], DType::F16, shape.to_vec())
}

fn used(ctx: &CoreContext) -> i64 {
    ctx.lx.borrow().used
}

fn next_ptr(ctx: &CoreContext) -> i64 {
    ctx.lx.borrow().next_ptr
}

/// Faithful port of `MemoryOps._write_to_lx`: reserve a stick-aligned LX span of
/// `n_elems` f16 values (2 bytes each) and bump `next_ptr`. Identical arithmetic
/// to the crate's private `ops_memory::write_to_lx`.
fn write_to_lx_f16(ctx: &mut CoreContext, n_elems: i64) {
    let size = n_elems * 2; // f16 = 2 bytes/elem
    let lx = ctx.get_lx(None);
    let lxm = lx.borrow_mut();
    let ptr = lxm.next_ptr;
    let advanced = ptr + size;
    lxm.next_ptr = (advanced + STICK_BYTES - 1) & !(STICK_BYTES - 1);
    lxm.write_bytes(ptr, &vec![0u8; size as usize]);
}

// ===========================================================================
// TestScopeStack
// ===========================================================================

#[test]
fn test_function_scope_is_always_present() {
    // _scope_stack starts with one scope; popping it panics (Python: RuntimeError).
    let mut ctx = make_context(2);
    let r = catch_unwind(AssertUnwindSafe(|| ctx.pop_scope()));
    assert!(r.is_err(), "popping the function-body scope must panic");
}

#[test]
fn test_inner_scope_sees_outer_values() {
    let mut ctx = make_context(2);
    ctx.set_value("%x", Value::Index(42));
    ctx.push_scope();
    assert!(matches!(ctx.get_value("%x").unwrap(), Value::Index(42)));
}

#[test]
fn test_outer_scope_does_not_see_inner_values() {
    let mut ctx = make_context(2);
    ctx.push_scope();
    ctx.set_value("%body_local", Value::Index(99));
    assert!(matches!(
        ctx.get_value("%body_local").unwrap(),
        Value::Index(99)
    ));
    ctx.pop_scope();
    // Python: KeyError. Rust: get_value returns Err for an undefined name.
    assert!(ctx.get_value("%body_local").is_err());
}

#[test]
fn test_has_value_searches_all_scopes() {
    let mut ctx = make_context(2);
    ctx.set_value("%outer", Value::Index(1));
    ctx.push_scope();
    ctx.set_value("%inner", Value::Index(2));
    assert!(ctx.has_value("%outer"));
    assert!(ctx.has_value("%inner"));
    ctx.pop_scope();
    assert!(ctx.has_value("%outer"));
    assert!(!ctx.has_value("%inner"));
}

#[test]
fn test_nested_scopes() {
    // Three levels: function -> for -> nested for. Sentinel ints stand in for the
    // Python "f"/"o"/"i" string markers (identity/visibility is what matters).
    let mut ctx = make_context(2);
    ctx.set_value("%func_val", Value::Index(0)); // "f"
    ctx.push_scope(); // outer for
    ctx.set_value("%outer_val", Value::Index(1)); // "o"
    ctx.push_scope(); // inner for
    ctx.set_value("%inner_val", Value::Index(2)); // "i"

    // All visible from innermost.
    assert!(matches!(
        ctx.get_value("%func_val").unwrap(),
        Value::Index(0)
    ));
    assert!(matches!(
        ctx.get_value("%outer_val").unwrap(),
        Value::Index(1)
    ));
    assert!(matches!(
        ctx.get_value("%inner_val").unwrap(),
        Value::Index(2)
    ));

    ctx.pop_scope(); // exit inner for
    assert!(ctx.has_value("%outer_val"));
    assert!(!ctx.has_value("%inner_val"));

    ctx.pop_scope(); // exit outer for
    assert!(ctx.has_value("%func_val"));
    assert!(!ctx.has_value("%outer_val"));
}

// ===========================================================================
// TestLXTracking
// ===========================================================================

#[test]
fn test_track_increments_used() {
    let mut ctx = make_context(2);
    let tile = make_tile(&[32, 1024]); // 32*1024*2 = 65536 bytes
    ctx.track_lx("%tile", tile.size_bytes() as i64).unwrap();
    assert_eq!(used(&ctx), 65536);
}

#[test]
fn test_untrack_decrements_used() {
    let mut ctx = make_context(2);
    let tile = make_tile(&[32, 1024]);
    ctx.track_lx("%tile", tile.size_bytes() as i64).unwrap();
    ctx.untrack_lx("%tile");
    assert_eq!(used(&ctx), 0);
}

#[test]
fn test_untrack_nonexistent_is_noop() {
    let mut ctx = make_context(2);
    ctx.untrack_lx("%does_not_exist"); // must not panic
    assert_eq!(used(&ctx), 0);
}

#[test]
fn test_pop_scope_frees_lx() {
    let mut ctx = make_context(2);
    ctx.push_scope();
    let tile = make_tile(&[32, 1024]); // 65536 bytes
    ctx.set_value("%tile", Value::Tile(tile.clone()));
    ctx.track_lx("%tile", tile.size_bytes() as i64).unwrap();
    assert_eq!(used(&ctx), 65536);

    ctx.pop_scope();
    assert_eq!(used(&ctx), 0);
}

#[test]
fn test_pop_scope_does_not_free_outer_lx() {
    let mut ctx = make_context(2);
    let outer_tile = make_tile(&[4, 64]); // 512 bytes
    ctx.set_value("%outer", Value::Tile(outer_tile.clone()));
    ctx.track_lx("%outer", outer_tile.size_bytes() as i64)
        .unwrap();

    ctx.push_scope();
    let inner_tile = make_tile(&[32, 1024]); // 65536 bytes
    ctx.set_value("%inner", Value::Tile(inner_tile.clone()));
    ctx.track_lx("%inner", inner_tile.size_bytes() as i64)
        .unwrap();
    assert_eq!(used(&ctx), 512 + 65536);

    ctx.pop_scope();
    assert_eq!(used(&ctx), 512); // only outer remains
}

#[test]
fn test_lx_overflow_raises() {
    // 1 MB = 1048576 bytes. Two 512 KB tiles fit; one more byte overflows.
    let mut ctx = make_context(1);
    ctx.track_lx("%a", 512 * 1024).unwrap();
    ctx.track_lx("%b", 512 * 1024).unwrap();
    assert_eq!(used(&ctx), 1048576);
    // Python: MemoryError("LX scratchpad overflow"). Rust: Err, allocation rejected.
    let r = ctx.track_lx("%c", 1);
    assert!(r.is_err());
    assert_eq!(used(&ctx), 1048576); // unchanged
}

#[test]
fn test_clear_values_resets_everything() {
    let mut ctx = make_context(2);
    ctx.set_value("%x", Value::Index(1));
    ctx.push_scope();
    ctx.set_value("%y", Value::Index(2));
    let tile = make_tile(&[8, 64]);
    ctx.track_lx("%tile", tile.size_bytes() as i64).unwrap();

    ctx.clear_values();

    // Python asserts _scope_stack == [{}], _lx_bytes == {}, lx.used == 0.
    // Those fields are private in Rust; check the equivalent observable state.
    assert_eq!(used(&ctx), 0); // _lx_bytes cleared -> used reset
    assert_eq!(next_ptr(&ctx), 0); // lx cleared
    assert!(!ctx.has_value("%x")); // all scopes wiped
    assert!(!ctx.has_value("%y"));
    // Only the function-body scope remains: a single pop must panic.
    let r = catch_unwind(AssertUnwindSafe(|| ctx.pop_scope()));
    assert!(r.is_err());
}

// ===========================================================================
// TestIterArgsPersistence
// ===========================================================================

#[test]
fn test_iter_arg_tiles_persist_body_local_freed() {
    // scf.for with one Tile iter_arg + one body-local Tile; body-local LX is
    // freed each iteration while the iter_arg survives.
    let mut ctx = make_context(2);

    // Initial iter_arg: tensor<4x1xf16> = 8 bytes.
    let iter_tile = make_tile(&[4, 1]);
    ctx.set_value("%acc", Value::Tile(iter_tile.clone()));
    ctx.track_lx("%acc", iter_tile.size_bytes() as i64).unwrap();
    assert_eq!(used(&ctx), 8);

    for _ in 0..3 {
        ctx.push_scope();

        // Body-local: tensor<4x256xf16> = 2048 bytes.
        let body_tile = make_tile(&[4, 256]);
        ctx.set_value("%body_tile", Value::Tile(body_tile.clone()));
        ctx.track_lx("%body_tile", body_tile.size_bytes() as i64)
            .unwrap();

        // New iter_arg value (created in body, will be yielded): 8 bytes.
        let new_acc = make_tile(&[4, 1]);
        ctx.set_value("%new_acc", Value::Tile(new_acc.clone()));
        ctx.track_lx("%new_acc", new_acc.size_bytes() as i64)
            .unwrap();

        assert_eq!(used(&ctx), 8 + 2048 + 8); // old acc + body + new acc

        // pop_scope frees body-local LX (%body_tile AND %new_acc).
        ctx.pop_scope();
        assert_eq!(used(&ctx), 8); // only old %acc remains

        // Re-bind iter_arg: untrack old, set + track new.
        ctx.untrack_lx("%acc");
        ctx.set_value("%acc", Value::Tile(new_acc.clone()));
        ctx.track_lx("%acc", new_acc.size_bytes() as i64).unwrap();
        assert_eq!(used(&ctx), 8); // back to steady state
    }
}

// ===========================================================================
// TestNextPtrRewind (issue #26)
// ===========================================================================

#[test]
fn test_issue_26_reproducer_next_ptr_bounded_in_loop() {
    // Before the fix, next_ptr advanced by 2048 (stick-aligned tile size) every
    // iteration and crossed the 128 KB cap at iter 64 while used stayed 0. After
    // the fix, both return to 0 on every pop.
    let mut ctx = make_context_bytes(128 * 1024); // 128 KB

    for i in 0..100 {
        ctx.push_scope();

        // Mirrors the interpreter's per-op load sequence: write into LX, set the
        // SSA value, track its bytes. pop_scope frees it via untrack_lx.
        write_to_lx_f16(&mut ctx, 4 * 256); // 2 KB
        let name = format!("%tile_iter{i}");
        let tile = make_tile(&[4, 256]);
        ctx.set_value(&name, Value::Tile(tile.clone()));
        ctx.track_lx(&name, tile.size_bytes() as i64).unwrap();

        ctx.pop_scope();

        // Strong invariant: both accountants return to pre-push values.
        assert_eq!(used(&ctx), 0, "iter {i}: lx.used");
        assert_eq!(next_ptr(&ctx), 0, "iter {i}: lx.next_ptr");
    }
}

#[test]
fn test_next_ptr_restored_on_pop_single_level() {
    let mut ctx = make_context(2);
    assert_eq!(next_ptr(&ctx), 0);

    ctx.push_scope();
    write_to_lx_f16(&mut ctx, 128); // 256 B
    assert_eq!(next_ptr(&ctx), 256);
    ctx.pop_scope();
    assert_eq!(next_ptr(&ctx), 0);
}

#[test]
fn test_next_ptr_restored_on_pop_with_outer_allocation() {
    let mut ctx = make_context(2);

    // Outer-scope allocation at the function level.
    write_to_lx_f16(&mut ctx, 128); // 256 B
    let outer_ptr = next_ptr(&ctx);
    assert_eq!(outer_ptr, 256);

    ctx.push_scope();
    write_to_lx_f16(&mut ctx, 1024); // 2 KB
    assert_eq!(next_ptr(&ctx), 256 + 2048);
    ctx.pop_scope();

    // Inner allocation reclaimed; outer watermark preserved.
    assert_eq!(next_ptr(&ctx), outer_ptr);
}

#[test]
fn test_nested_scopes_rewind_lifo() {
    // Three-level nesting (fn / for-body / if-body) restores each level.
    let mut ctx = make_context(2);

    write_to_lx_f16(&mut ctx, 128); // A (function level)
    let wm_fn = next_ptr(&ctx);

    ctx.push_scope(); // for-body
    write_to_lx_f16(&mut ctx, 512); // B
    let wm_for = next_ptr(&ctx);

    ctx.push_scope(); // if-body
    write_to_lx_f16(&mut ctx, 1024); // C
    assert!(next_ptr(&ctx) > wm_for);

    ctx.pop_scope(); // pop if
    assert_eq!(next_ptr(&ctx), wm_for); // C reclaimed, B/A live

    ctx.pop_scope(); // pop for
    assert_eq!(next_ptr(&ctx), wm_fn); // B reclaimed, A live
}

#[test]
fn test_legitimate_overflow_still_raises() {
    // The rewind must not hide genuine LX exhaustion within a single scope.
    let mut ctx = make_context_bytes(128 * 1024); // 128 KB cap

    let mut overflowed = false;
    for i in 0..100 {
        write_to_lx_f16(&mut ctx, 1024); // 2 KB each
        let tile = make_tile(&[1024]);
        if ctx
            .track_lx(&format!("%t{i}"), tile.size_bytes() as i64)
            .is_err()
        {
            overflowed = true; // Python: MemoryError("LX scratchpad overflow")
            break;
        }
    }
    assert!(overflowed, "tracking past capacity must be rejected");
}

#[test]
fn test_clear_values_resets_watermark_stack() {
    // clear_values must also clear the watermark stack (Python:
    // _lx_next_ptr_stack == []). Field is private; verify observable resets and
    // that the watermark stack is empty by exercising the pop discipline.
    let mut ctx = make_context(2);
    ctx.push_scope();
    ctx.push_scope();
    // (Python: len(_lx_next_ptr_stack) == 2 here.) Bump next_ptr so a stale
    // watermark would be observable after clear.
    write_to_lx_f16(&mut ctx, 1024);

    ctx.clear_values();

    assert_eq!(next_ptr(&ctx), 0);
    assert_eq!(used(&ctx), 0);
    // Watermark stack and scope stack reset to the function body: a single
    // pop_scope must panic (no pending inner watermarks remain).
    let r = catch_unwind(AssertUnwindSafe(|| ctx.pop_scope()));
    assert!(r.is_err());
}
