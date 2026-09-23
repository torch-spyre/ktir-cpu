// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! WALL-CLOCK timing gate for the flash-attention cap-tiling pass (TODO #2).
//!
//! The golden gate (`flash_attn_golden.rs`) proves SEMANTICS at SHORT cap (64-256)
//! where the pass correctly no-ops. This file answers the orthogonal question the
//! project actually cares about (arbitrary LLM inference == long context): at LONG
//! context, is flash-tiled attention a REAL, MEASURED wall-clock SPEEDUP over the
//! un-tiled re-rolled form?
//!
//! THE COMPARISON (apples-to-apples, honest):
//!
//! * un-tiled = the head_rewrite RE-ROLLED whole-`[m, cap]` form
//!   (`rewrite_head_attention` — the EXACT real model node structure), scaled to a
//!   large cap.
//! * flash = the SAME computation with the context cap axis split into KV blocks
//!   `[m, blk]`, online softmax (`tile_rerolled_attention`).
//!
//! Both run through the SAME UNCHANGED `interpreter::execute_function` (the generic
//! per-core CPU path — NOT the GPU/batched executor; this is NOT a GPU-kernel-time
//! vs CPU-wall-time comparison). Best-of-N release wall-clock, weight-free inputs.
//!
//! This is the REAL STRUCTURE, long-context-scaled: `rewrite_head_attention` emits
//! byte-for-byte the same two-block re-rolled IR that `apply_head_rewrite` produces
//! on the real llama / smollm2 `node111` (verified in `head_rewrite_golden.rs` and
//! `flash_attn_golden.rs`). We construct that island at a representative shape
//! (m=32, d=64, a few heads) and sweep ONLY the context `cap`. It is NOT a synthetic
//! shape that misrepresents the model — it is exactly the real attention body, at a
//! longer context (which is precisely the long-context use case).
//!
//! SEMANTICS: at every cap we assert flash == un-tiled through the unchanged
//! interpreter on arbitrary weight-free inputs, max-abs < 0.05.
//!
//! Ignored by default (release-only, serial; debug overheads mask the allocation
//! cost). Run:
//!   cargo test --release -p ktir-emulator --test flash_attn_timing -- --ignored \
//!     --nocapture --test-threads=1

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, Output, execute_function};
use ktir_emulator::ir::{IRFunction, IRModule};
use ktir_optimizer::flash_attn::{
    apply_flash_attention, recognize_rerolled_attention, tile_rerolled_attention,
};
use ktir_optimizer::fusion::attention_needs_flash;
use ktir_optimizer::head_rewrite::{HeadAttnIsland, rewrite_head_attention};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------

/// Representative prefill head shape (the real models are m≈32 prefill rows, d=64).
const M: i64 = 32;
const D: i64 = 64;
/// A few heads so the grid is the genuine head-parallel `[H,1,1]` SPMD form. gqac=1
/// keeps kv_cols = H*d (no GQA sharing) so the synthetic args are simple to size.
const H: i64 = 4;
const GQAC: i64 = 1;

/// The context caps to sweep (long-context regime). 128 is short (flash no-ops by
/// Contract B); the rest are progressively longer.
const CAPS: &[i64] = &[128, 512, 1024, 2048, 4096, 8192, 16384];

/// Best-of-N: take the FASTEST of N release runs (ignores scheduler/GC outliers).
const N_RUNS: usize = 5;

/// A REALISTIC LX budget (bytes). Chosen so the per-block `[m, blk=128]` context
/// tile (m*128*2 = 8 KB at f16) FITS, while the full `[m, cap]` tile overflows for
/// cap >= 256 — i.e. flash fires (with a sensible blk=128, NOT a pathological blk=1)
/// exactly in the long-context regime, and no-ops at cap=128 (short context).
///
/// attention_needs_flash(sb, lx) == (sb*8 >= lx*7). With lx=16384:
///   blk=128 tile  8192 B -> 65536 >= 114688? no  -> FITS (good block).
///   cap=128 tile  8192 B -> no-op (short context, identity).
///   cap=256 tile 16384 B -> 131072 >= 114688? yes -> flash fires.
const LX_BUDGET: usize = 16384;

// ---------------------------------------------------------------------------
// Build the REAL re-rolled attention structure at an arbitrary cap
// ---------------------------------------------------------------------------

/// A `HeadAttnIsland` at the representative shape with the swept `cap`. Feeding it
/// to `rewrite_head_attention` yields the EXACT re-rolled two-block IR the head pass
/// emits on the real model node — the un-tiled long-context body we measure.
fn island_at_cap(cap: i64) -> HeadAttnIsland {
    HeadAttnIsland {
        q_arg: "%q".into(),
        o_arg: "%o".into(),
        mask_arg: "%mask".into(),
        kc_arg: "%kc".into(),
        kd_arg: "%kd".into(),
        vc_arg: "%vc".into(),
        vd_arg: "%vd".into(),
        q_cols: H * D,
        kv_cols: (H / GQAC) * D,
        m: M,
        cap,
        d: D,
        gqac: GQAC,
        hdc: D,
        h: H,
        scale: 1.0 / (D as f32).sqrt(),
        ninf: -1.0e38,
        dtype: "f16".into(),
    }
}

/// The un-tiled re-rolled module (one function `attn`, grid `[H,1,1]`).
fn untiled_module(cap: i64) -> (IRModule, String) {
    let mut f = rewrite_head_attention(&island_at_cap(cap));
    f.name = "attn".into();
    let mut m = IRModule::default();
    m.add_function(f);
    (m, "attn".into())
}

/// The flash-tiled module: take the un-tiled re-rolled function, recognize it, and
/// cap-tile its context block with a budget-chosen block (blk=128 in this regime).
/// Uses the SAME public `apply_flash_attention` entrypoint the program pipeline
/// uses, with a realistic LX budget — NOT a pathological tiny forced budget.
fn flash_module(cap: i64) -> (IRModule, String, bool) {
    let (mut module, name) = untiled_module(cap);
    let fired = apply_flash_attention(&mut module, |sb| attention_needs_flash(sb, LX_BUDGET));
    (module, name, fired > 0)
}

// ---------------------------------------------------------------------------
// Weight-free arbitrary inputs, sized from the island shapes
// ---------------------------------------------------------------------------

/// Deterministic arbitrary f16 data in a modest range (so f16 rounding does not
/// dominate the < 0.05 gate). Same spirit as `head_rewrite_golden`'s arg builder.
fn arb(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed) % 23) as f32 - 11.0) * 0.03)
        .collect()
}

/// The seven pointer args for the re-rolled function, sized from the island:
///   %q,%o   [m, H*d]    %mask [1, cap]
///   %kc,%vc [cap, kv_cols]   %kd,%vd [m, kv_cols]
fn build_args(cap: i64) -> Vec<(&'static str, Arg)> {
    let m = M as usize;
    let d = D as usize;
    let h = H as usize;
    let cap = cap as usize;
    let kv_cols = (H / GQAC) as usize * d;
    let qcols = h * d;
    let f16 = DType::F16;
    let mk = |name: &'static str, rows: usize, cols: usize, seed: usize| {
        (
            name,
            Arg::Tensor {
                data: arb(rows * cols, seed),
                shape: vec![rows, cols],
                dtype: f16,
            },
        )
    };
    vec![
        mk("q", m, qcols, 1),
        // %o is an output; seed it too (overwritten by the store).
        mk("o", m, qcols, 2),
        // per-head context mask [1, cap]: 0 (visible) everywhere here (weight-free;
        // semantics equivalence holds for ANY mask since both paths read the same).
        (
            "mask",
            Arg::Tensor {
                data: vec![0.0; cap],
                shape: vec![1, cap],
                dtype: f16,
            },
        ),
        mk("kc", cap, kv_cols, 3),
        mk("kd", m, kv_cols, 4),
        mk("vc", cap, kv_cols, 5),
        mk("vd", m, kv_cols, 6),
    ]
}

// ---------------------------------------------------------------------------
// Timing + semantics
// ---------------------------------------------------------------------------

fn run(module: &IRModule, name: &str, args: &[(&str, Arg)]) -> HashMap<String, Output> {
    execute_function(module, name, args).expect("attention run")
}

/// Fallible run: at very long context the UN-TILED full-`[m,cap]` intermediates
/// overflow the interpreter's real 2 MB LX budget (`SpyreMemoryHierarchy`) and
/// `execute_function` returns `Err(..)`. That is itself the decisive long-context
/// result (un-tiled CANNOT run; flash can) — so we surface it instead of panicking.
fn try_run(
    module: &IRModule,
    name: &str,
    args: &[(&str, Arg)],
) -> Result<HashMap<String, Output>, String> {
    execute_function(module, name, args)
}

/// Best-of-N wall-clock, propagating an LX-overflow `Err` from the FIRST run so the
/// caller can report "overflowed LX" rather than fabricate a number.
fn best_of_n(
    module: &IRModule,
    name: &str,
    args: &[(&str, Arg)],
) -> Result<(f64, HashMap<String, Output>), String> {
    // One warm-up (page-in, allocator warm) outside the timing; also where an LX
    // overflow surfaces.
    let _ = try_run(module, name, args)?;
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..N_RUNS {
        let t = std::time::Instant::now();
        let out = try_run(module, name, args)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
        last = Some(out);
    }
    Ok((best, last.unwrap()))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// One sweep row: cap, un-tiled ms (`None` = overflowed LX), flash ms, fired,
/// max-abs vs un-tiled (`None` when un-tiled overflowed), chosen block size.
type Row = (i64, Option<f64>, f64, bool, Option<f32>, i64);

#[test]
#[ignore = "release wall-clock sweep: cargo test --release --test flash_attn_timing -- --ignored --nocapture --test-threads=1"]
fn flash_vs_untiled_wall_clock_sweep() {
    eprintln!(
        "\n== flash-tiled vs un-tiled re-rolled attention (REAL structure, m={M} d={D} H={H}) ==\n\
         cap      | un-tiled ms | flash ms | ratio(un/flash) | flash-fired | max-abs | blk"
    );
    eprintln!("---------+-------------+----------+-----------------+-------------+---------+----");

    let mut crossover: Option<i64> = None;
    let mut rows: Vec<Row> = Vec::new();

    for &cap in CAPS {
        let args = build_args(cap);

        // (a) un-tiled re-rolled module.
        let (un_mod, un_name) = untiled_module(cap);
        // (b) flash-tiled module (realistic budget; fires only for cap >= 256).
        let (fl_mod, fl_name, fired) = flash_module(cap);

        // Recover the chosen block size for the report (from the emitted scf.for, if any).
        let blk = if fired {
            recover_blk(&fl_mod, &fl_name).unwrap_or(0)
        } else {
            0
        };

        // Flash MUST always run (it tiles below LX). Time it.
        let (fl_ms, fl_out) =
            best_of_n(&fl_mod, &fl_name, &args).expect("flash-tiled must run within LX");

        // Un-tiled may OVERFLOW LX at very long context (the real long-context wall
        // flash exists to remove). Catch it.
        match best_of_n(&un_mod, &un_name, &args) {
            Ok((un_ms, un_out)) => {
                // Semantics: flash == un-tiled, max-abs < 0.05.
                let mut worst = 0.0f32;
                for (k, o) in &un_out {
                    let r = fl_out
                        .get(k)
                        .unwrap_or_else(|| panic!("flash missing output {k}"));
                    worst = worst.max(max_abs_diff(&o.data, &r.data));
                }
                assert!(
                    worst < 0.05,
                    "cap={cap}: flash diverged from un-tiled by {worst} (>= 0.05) — NOT semantics-preserving"
                );
                let ratio = un_ms / fl_ms;
                eprintln!(
                    "{cap:<8} | {un_ms:>11.3} | {fl_ms:>8.3} | {ratio:>15.3} | {fired:>11} | {worst:>7.4} | {blk}"
                );
                rows.push((cap, Some(un_ms), fl_ms, fired, Some(worst), blk));
                if fired && ratio >= 1.0 && crossover.is_none() {
                    crossover = Some(cap);
                }
            }
            Err(e) => {
                // Un-tiled cannot even execute — flash is the ONLY runnable form.
                eprintln!(
                    "{cap:<8} | {:>11} | {fl_ms:>8.3} | {:>15} | {fired:>11} | {:>7} | {blk}   (un-tiled LX-overflow: {})",
                    "OVERFLOW", "inf", "n/a", e
                );
                rows.push((cap, None, fl_ms, fired, None, blk));
                if crossover.is_none() {
                    crossover = Some(cap);
                }
            }
        }
    }

    eprintln!("\nSummary:");
    match crossover {
        Some(c) => eprintln!("  flash-tiled BEATS un-tiled at cap >= {c} (ratio >= 1.0)."),
        None => eprintln!(
            "  flash-tiled does NOT beat un-tiled at any swept cap (honest: no crossover)."
        ),
    }
    if let Some((cap, un, fl, fired, _, blk)) = rows.last().copied() {
        match un {
            Some(un) => eprintln!(
                "  largest cap={cap}: un-tiled {un:.3} ms, flash {fl:.3} ms (blk={blk}, fired={fired}), \
                 flash {} ({:.2}x).",
                if fl < un { "WINS" } else { "LOSES" },
                un / fl
            ),
            None => eprintln!(
                "  largest cap={cap}: un-tiled OVERFLOWED LX (cannot run); flash {fl:.3} ms \
                 (blk={blk}) — flash is the ONLY runnable form (decisive long-context win)."
            ),
        }
    }

    // This test ALWAYS asserts semantics (above, when un-tiled runs). It does NOT
    // assert a speedup: whether flash wins is the EMPIRICAL finding the harness
    // reports. A genuine loss is a real result, not a test failure. Flash MUST run
    // at every cap (asserted above), which is itself the long-context guarantee.
}

/// Read the chosen KV block size off the emitted `scf.for` body: the per-head
/// context MASK slice `[1, blk]` uniquely encodes `blk` (first dim is 1).
fn recover_blk(module: &IRModule, name: &str) -> Option<i64> {
    use ktir_emulator::ir::Attr;
    let f: &IRFunction = module.functions.get(name)?;
    let forop = f.operations.iter().find(|o| o.op_type == "scf.for")?;
    forop.regions[0]
        .iter()
        .filter_map(|o| match o.attributes.get("shape") {
            Some(Attr::IntList(v)) if v.len() == 2 && v[0] == 1 => Some(v[1]),
            _ => None,
        })
        .min()
}

/// Sanity: at cap=128 (short) the pass no-ops; at cap>=256 it fires and tiles with a
/// sensible block strictly smaller than cap. (Fast, default-run — not ignored.)
#[test]
fn budget_fires_only_long_context_with_sensible_block() {
    // short: no-op (identity).
    let (_m128, _n128, fired128) = flash_module(128);
    assert!(
        !fired128,
        "cap=128 short context must NOT fire (identity no-op)"
    );

    // long: fires, tiles, blk < cap, recognized re-rolled form.
    for &cap in &[512i64, 1024, 4096] {
        let (un_mod, un_name) = untiled_module(cap);
        assert!(
            recognize_rerolled_attention(un_mod.functions.get(&un_name).unwrap()).is_some(),
            "cap={cap}: un-tiled re-rolled form must be recognized"
        );
        let (fl_mod, fl_name, fired) = flash_module(cap);
        assert!(fired, "cap={cap}: long context must fire flash");
        let blk = recover_blk(&fl_mod, &fl_name).expect("blk from scf.for");
        assert!(blk < cap, "cap={cap}: blk {blk} must be < cap");
        assert_eq!(blk, 128, "cap={cap}: realistic budget should pick blk=128");
    }
}

/// Direct `tile_rerolled_attention` path (no budget machinery): confirms the tiler
/// builds a runnable module that matches the un-tiled form at a long cap. Default-run.
#[test]
fn direct_tiler_matches_untiled_long_cap() {
    let cap = 1024i64;
    let (un_mod, un_name) = untiled_module(cap);
    let isl = recognize_rerolled_attention(un_mod.functions.get(&un_name).unwrap())
        .expect("recognize re-rolled");
    let mut tiled = tile_rerolled_attention(&isl, 128);
    tiled.name = un_name.clone();
    let mut fl_mod = IRModule::default();
    fl_mod.add_function(tiled);

    let args = build_args(cap);
    let un_out = run(&un_mod, &un_name, &args);
    let fl_out = run(&fl_mod, &un_name, &args);
    let mut worst = 0.0f32;
    for (k, o) in &un_out {
        let r = fl_out.get(k).unwrap_or_else(|| panic!("flash missing {k}"));
        worst = worst.max(max_abs_diff(&o.data, &r.data));
    }
    eprintln!("direct tiler cap=1024: max-abs {worst:.6}");
    assert!(worst < 0.05, "direct tiler diverged by {worst}");
}
