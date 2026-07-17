// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Flash-attention IR-rewrite pass — TODO #2 (the ABOVE-cap path of Contract B).
//!
//! Long-context attention's `[m, cap]` scores matrix is an INTRA-node tile that
//! overflows the 2 MB LX as `cap` (the KV length) grows. Segmentation cannot
//! help — attention is ONE node, and the scores tile lives inside it. This pass
//! tiles the cap/KV dimension with **online softmax**, emitting STANDARD tiled
//! MLIR (an `scf.for` over KV blocks + online-softmax arith/reduce + matmul)
//! that the EXISTING generic interpreter runs unchanged. It is NOT a hand-written
//! kernel and introduces NO new `ktdp` ops.
//!
//! ## Why this is a legitimate, semantics-preserving optimization
//!
//! Online softmax (Milakov & Gimelshein 2018; the FlashAttention recurrence) is
//! mathematically equal to the two-pass `max → exp → sum → div` softmax up to
//! floating-point re-association — it visits the same `exp` terms, just folds the
//! running max/sum block by block instead of after a full pass. It is in fact
//! *more* numerically stable (the running max bounds every `exp` argument), so it
//! comfortably stays inside the project's 0.05 golden gate. This is the same
//! class of rewrite as the existing GEMM K-loop recognizer in `metal_backend`:
//! recognize a high-level idiom from raw IR, re-emit a tiled equivalent.
//!
//! ## Contract B (see `fusion::attention_needs_flash`)
//!
//! This pass owns the ABOVE-cap regime ONLY. A node is rewritten IFF its
//! `[m, cap]` scores footprint would overflow LX. Below the cap the node is left
//! NAIVE (untouched) — that regime belongs to the head-batching fleet. The
//! predicate is monotone and exhaustive, so every attention node receives exactly
//! one transform. The rewritten node is region-bearing (`scf.for`) by design and
//! runs on the generic interpreter (the batched-executor's region-free gate makes
//! it `Err → fall back`, which is the intended path above the cap).
//!
//! ## Two recognizers
//!
//! The real cached prefill/decode nodes are an *unrolled, per-query-row,
//! two-KV-block* hand lowering. They do NOT match the single-block canonical
//! idiom — but [`crate::head_rewrite`] (which runs FIRST) RE-ROLLS them into a
//! whole-tensor two-block form: a CONTEXT `QKᵀ` producing `[m, cap]` scores (the
//! tile that overflows LX as context grows) + a small `[m, m]` masked DIAGONAL
//! block + an online-softmax combine + two `A·V` matmuls, all stored as ONE
//! `arith.addf(ov_context, ov_diag)`.
//!
//! This pass therefore has TWO structural recognizers, tried in order:
//!   1. [`recognize_rerolled_attention`] — the head-rewrite OUTPUT (the REAL node
//!      form). It anchors on the `arith.addf`-of-two-matmuls store value, recovers
//!      the context/diagonal softmax chains, and [`tile_rerolled_attention`]
//!      cap-tiles ONLY the CONTEXT `[m, cap]` block with online softmax (the
//!      `[m, m]` diagonal stays whole). THIS is the long-context fix.
//!   2. [`recognize_attention`] — the single-block synthetic canonical idiom
//!      (unchanged; keeps the synthetic golden green).
//!
//! ## Fail-safe recognition
//!
//! Both recognizers return `None` whenever the function is not PROVABLY their
//! idiom (a top-level `scf.*`, a deviating op sequence, the wrong shapes).
//! Returning `None` leaves the node untouched — the same Err→fallback discipline
//! as the GPU offloads. We never rewrite a node we cannot prove equivalent.

use ktir_core::ir::{Attr, IRFunction, IRModule, Operation};
use std::collections::HashMap;

/// Apply the flash-attention pass to every function in `module`, in place.
///
/// For each function: recognize the canonical naive-attention idiom; if it is
/// PROVABLY that idiom AND its `[m, cap]` scores tile would overflow LX
/// (`needs_flash(scores_bytes, lx_budget)` — Contract B's
/// `fusion::attention_needs_flash`), replace it with the tiled online-softmax
/// rewrite (same function NAME, args, grid). Otherwise leave it untouched
/// (fail-safe: below the cap, or not recognized → naive).
///
/// `needs_flash` is injected (rather than calling `fusion::attention_needs_flash`
/// directly) so the caller threads its OWN `lx_budget` and any force/threshold
/// env override, keeping the cap-partition decision in one place. Returns the
/// number of functions rewritten (for diagnostics / test assertions).
pub fn apply_flash_attention(module: &mut IRModule, needs_flash: impl Fn(usize) -> bool) -> usize {
    let names: Vec<String> = module.functions.keys().cloned().collect();
    let mut rewritten = 0usize;
    for name in names {
        let Some(func) = module.functions.get(&name) else {
            continue;
        };

        // (A) RE-ROLLED form FIRST — this is the REAL model node after
        // `head_rewrite` runs (it stores `arith.addf(ovc, ovd)`, the exact hop the
        // single-block `recognize_attention` bails on). Its CONTEXT `[m, cap]`
        // scores tile is what overflows LX as context grows; cap-tile ONLY that
        // block, leaving the small `[m, m]` diagonal whole. Same `scores_bytes`
        // formula (`m*cap*bytes`) as the head island, so Contract B's monotone
        // predicate routes each node to exactly one pass.
        if let Some(island) = recognize_rerolled_attention(func) {
            // Fire flash when the `[m, cap]` SCORES tile is over the LX budget
            // (`needs_flash` — the large-query regime) OR the `[cap, d]` CONTEXT
            // K/V tile is too large to keep whole in LX (the long-context /
            // small-query regime: decode `m=1`, chunked prefill, where scores stay
            // tiny but the context read overflows LX — the case the scores-only
            // gate missed and left un-tiled ⇒ `LX capacity exceeded`).
            if !needs_flash(island.scores_bytes())
                && island.context_bytes() <= FLASH_CONTEXT_TILE_MAX
            {
                continue; // both tiles fit whole: leave head_rewrite's form.
            }
            // Choose the KV block so BOTH the per-block scores tile `[m, blk]` fits
            // the injected budget AND the per-block context K/V tile `[blk, d]`
            // fits `FLASH_CONTEXT_TILE_MAX`. This is the actual long-context fix:
            // `blk < cap`.
            let blk = choose_block_budgeted(
                island.m,
                island.cap,
                island.d,
                dtype_bytes(&island.dtype),
                &needs_flash,
            );
            let mut tiled = tile_rerolled_attention(&island, blk);
            tiled.name = name.clone();
            module.functions.insert(name, tiled);
            rewritten += 1;
            continue;
        }

        // (B) Single-block canonical idiom (the synthetic golden path) — unchanged.
        let Some(island) = recognize_attention(func) else {
            continue;
        };
        if !needs_flash(island.scores_bytes()) {
            continue; // below the cap: stay naive (head-batching fleet's regime).
        }
        let grid = func.grid;
        let mut tiled = tile_attention(&island);
        // Preserve the original function identity so the program's node→tensor
        // bindings and the segmenter still resolve it. The rewrite is single-grid
        // by design (region-bearing, runs on the generic interpreter).
        tiled.name = name.clone();
        let _ = grid; // grid intentionally collapsed to [1,1] in tile_attention.
        module.functions.insert(name, tiled);
        rewritten += 1;
    }
    rewritten
}

/// The recovered configuration of a recognized naive-attention function.
///
/// All shapes are re-derived from the IR (never assumed). `causal` and `scale`
/// are likewise recovered from the actual ops, so the rewrite reproduces the
/// node's exact arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionIsland {
    /// Function argument (pointer) carrying Q, its memory-view shape `[m, d]`.
    pub q_arg: String,
    pub q_shape: Vec<i64>,
    /// Function argument carrying K, view shape `[cap, d]` (NOT transposed).
    pub k_arg: String,
    pub k_shape: Vec<i64>,
    /// Function argument carrying V, view shape `[cap, d]`.
    pub v_arg: String,
    pub v_shape: Vec<i64>,
    /// Output pointer arg, view shape `[m, d]`.
    pub o_arg: String,
    pub o_shape: Vec<i64>,
    /// Query rows.
    pub m: i64,
    /// KV length (the cap axis to tile).
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// `1/sqrt(d)` scale recovered from the `arith.mulf` by a splat constant.
    pub scale: f32,
    /// True when a causal mask add (`-inf` upper triangle) is present.
    pub causal: bool,
    /// Storage dtype string (e.g. `"f16"`).
    pub dtype: String,
}

/// Storage bytes per element for a KTIR dtype string (f16/bf16 default 2).
fn dtype_bytes(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" => 4,
        "f64" | "i64" => 8,
        "i1" => 1,
        _ => 2, // f16/bf16 default
    }
}

impl AttentionIsland {
    /// Scores-tile byte footprint `[m, cap]` × storage-dtype bytes — the value
    /// Contract B's `attention_needs_flash` consumes to decide whether to fire.
    pub fn scores_bytes(&self) -> usize {
        (self.m as usize)
            .saturating_mul(self.cap as usize)
            .saturating_mul(dtype_bytes(&self.dtype))
    }
}

/// The recovered configuration of a recognized RE-ROLLED head-attention function
/// (the [`crate::head_rewrite`] OUTPUT — i.e. the REAL model node). Carries the
/// SAME fields [`crate::head_rewrite::HeadAttnIsland`] recovers, so the rewrite
/// reproduces the node's exact per-head arithmetic; `scores_bytes` uses the
/// IDENTICAL `m*cap*bytes` CONTEXT-tile formula so the two passes' Contract-B
/// partition stays disjoint (a node routes to head-reroll XOR flash, never both).
#[derive(Clone, Debug, PartialEq)]
pub struct ReRolledIsland {
    /// Q pointer arg (view0), `[m, q_cols]` where `q_cols = H*d`.
    pub q_arg: String,
    /// O pointer arg (view1), `[m, q_cols]`.
    pub o_arg: String,
    /// Per-head context mask pointer arg (view2), `[1, cap]`.
    pub mask_arg: String,
    /// Context K pointer arg (view5), `[cap, kv_cols]`.
    pub kc_arg: String,
    /// Diagonal (current-segment) K pointer arg (view6), `[m, kv_cols]`.
    pub kd_arg: String,
    /// Context V pointer arg (view7), `[cap, kv_cols]`.
    pub vc_arg: String,
    /// Diagonal V pointer arg (view8), `[m, kv_cols]`.
    pub vd_arg: String,
    /// Q/O view column width `H*d`.
    pub q_cols: i64,
    /// KV view column width (`num_kv_heads * d`).
    pub kv_cols: i64,
    /// Query rows.
    pub m: i64,
    /// Context KV length (the `cap` axis to tile).
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// GQA divisor recovered from `arith.divui %hpid, %gqac`.
    pub gqac: i64,
    /// Per-head column stride recovered from `arith.muli %hpid, %hdc`.
    pub hdc: i64,
    /// Grid head count `H` (grid.0).
    pub h: i64,
    /// `1/sqrt(d)` scale recovered from the `arith.mulf` by a splat constant.
    pub scale: f32,
    /// `-inf` mask constant recovered from the diagonal triangular mask.
    pub ninf: f32,
    /// Storage dtype string (e.g. `"f16"`).
    pub dtype: String,
}

impl ReRolledIsland {
    /// CONTEXT scores-tile footprint `[m, cap]` × dtype bytes — IDENTICAL to
    /// `HeadAttnIsland::scores_bytes` so Contract B's monotone predicate routes a
    /// node to exactly one of {head-reroll, flash}. (The `[m, m]` diagonal stays
    /// whole and is NOT counted — only the cap tile overflows.)
    pub fn scores_bytes(&self) -> usize {
        (self.m as usize)
            .saturating_mul(self.cap as usize)
            .saturating_mul(dtype_bytes(&self.dtype))
    }

    /// CONTEXT K/V-tile footprint `[cap, d]` × dtype bytes. UNLIKE `scores_bytes`
    /// (`m·cap`) this is m-INDEPENDENT: it grows with the context length `cap`
    /// alone. In the small-query/long-context regime (decode `m=1`, chunked
    /// prefill `m≪cap`) the whole `[cap, d]` context K (and V) read is what
    /// overflows LX while the `[m, cap]` scores tile stays tiny — the case the
    /// scores-only gate misses. Flash-tiling the cap axis shrinks BOTH tiles.
    pub fn context_bytes(&self) -> usize {
        (self.cap as usize)
            .saturating_mul(self.d as usize)
            .saturating_mul(dtype_bytes(&self.dtype))
    }
}

/// Max per-block CONTEXT K/V tile `[blk, d]` bytes flash keeps whole in LX. The
/// cap-tiled online-softmax loop holds one K block AND one V block resident at a
/// time, so `2 · this` must fit alongside the fused segment's other resident
/// live-set (~1.5 MiB of a 2 MiB per-core LX on Llama-3B). 128 KiB ⇒ 256 KiB for
/// K+V, leaving comfortable headroom. Numerics are exact for ANY block size
/// (online softmax), so this only trades a few extra blocks for fitting LX.
const FLASH_CONTEXT_TILE_MAX: usize = 128 * 1024;

// ===========================================================================
// Recognition
// ===========================================================================

/// Decoded `ktdp.construct_memory_view %ptr` -> (pointer arg, view shape).
struct ViewInfo {
    arg: String,
    shape: Vec<i64>,
    dtype: String,
}

/// Decoded `ktdp.construct_access_tile %view[..]` -> the view SSA it reads.
struct TileInfo {
    view: String,
}

fn shape_attr(op: &Operation) -> Vec<i64> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) => v.clone(),
        _ => Vec::new(),
    }
}

fn dtype_attr(op: &Operation) -> String {
    match op.attributes.get("dtype") {
        Some(Attr::Str(s)) => s.clone(),
        _ => "f16".to_string(),
    }
}

/// Recognize the canonical single-block naive-attention idiom in `func`.
///
/// Returns `Some(island)` only when the body is PROVABLY:
///   load Q[m,d], load K[cap,d], `Kt = transpose(K)`, `S = Q@Kt`, scale `S`,
///   (optional) causal-mask add, `mx = reduce_max(S, dim 1)`,
///   `P = exp(S - mx)`, `l = reduce_sum(P, dim 1)`, `W = P / l`,
///   load V[cap,d], `O = W@V`, store O[m,d].
///
/// Any structural deviation (an `scf.for`, multiple stores, an unrecognized op
/// sequence, a transposed/odd layout, multi-head grid unrolling) yields `None`,
/// leaving the node naive (fail-safe).
pub fn recognize_attention(func: &IRFunction) -> Option<AttentionIsland> {
    // An already-tiled body (an `scf.for` / `scf.if` control-flow region) is never
    // the flat canonical idiom — bail. (Leaf region-bodied ops like
    // `tensor.generate` for a mask or an explicit-region `linalg.reduce` are fine;
    // only top-level CONTROL FLOW disqualifies.)
    if func.operations.iter().any(|op| {
        matches!(
            op.op_type.as_str(),
            "scf.for" | "scf.if" | "scf.while" | "scf.parallel" | "scf.forall"
        )
    }) {
        return None;
    }

    // Index views/tiles by their result SSA so we can walk the load/store chains.
    let mut views: HashMap<String, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<String, TileInfo> = HashMap::new();
    // load result SSA -> (pointer arg, view shape) it reads.
    let mut load_src: HashMap<String, (String, Vec<i64>, String)> = HashMap::new();
    // SSA -> op (for the compute chain).
    let mut def: HashMap<String, &Operation> = HashMap::new();

    for op in &func.operations {
        match op.op_type.as_str() {
            "ktdp.construct_memory_view" => {
                if let (Some(res), Some(arg)) = (&op.result, op.operands.first()) {
                    views.insert(
                        res.clone(),
                        ViewInfo {
                            arg: arg.clone(),
                            shape: shape_attr(op),
                            dtype: dtype_attr(op),
                        },
                    );
                }
            }
            "ktdp.construct_access_tile" => {
                if let (Some(res), Some(view)) = (&op.result, op.operands.first()) {
                    tiles.insert(res.clone(), TileInfo { view: view.clone() });
                }
            }
            "ktdp.load" => {
                if let (Some(res), Some(tile)) = (&op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(
                        res.clone(),
                        (vi.arg.clone(), vi.shape.clone(), vi.dtype.clone()),
                    );
                }
            }
            _ => {}
        }
        if let Some(res) = &op.result {
            def.insert(res.clone(), op);
        }
    }

    // Exactly one store: the attention output. (The real unrolled nodes store
    // many times — they fail here, which is the fail-safe we want.)
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == "ktdp.store")
        .collect();
    let store = match stores.as_slice() {
        [s] => s,
        _ => return None,
    };
    // store %value, %tile
    let stored_val = store.operands.first()?;
    let store_tile = store.operands.get(1)?;
    let o_ti = tiles.get(store_tile)?;
    let o_view = views.get(&o_ti.view)?;
    let o_arg = o_view.arg.clone();
    let o_shape = o_view.shape.clone();

    // Walk back from the stored value: it must be `O = linalg.matmul(W, V)`.
    let av = def.get(stored_val)?;
    if av.op_type != "linalg.matmul" {
        return None;
    }
    let w_ssa = av.operands.first()?; // probabilities W = P / l
    let v_loaded = av.operands.get(1)?; // V (loaded straight)
    let (v_arg, v_shape, _vdt) = load_src.get(v_loaded)?.clone();

    // W = arith.divf(P, l_broadcast)
    let divw = def.get(w_ssa)?;
    if divw.op_type != "arith.divf" {
        return None;
    }
    let p_ssa = divw.operands.first()?;
    let lbcast = divw.operands.get(1)?;
    // P = math.exp(shifted)
    let pexp = def.get(p_ssa)?;
    if pexp.op_type != "math.exp" {
        return None;
    }
    let shifted = pexp.operands.first()?;
    // shifted = arith.subf(scaled_masked, mx_broadcast)
    let sub = def.get(shifted)?;
    if sub.op_type != "arith.subf" {
        return None;
    }
    let scores_masked = sub.operands.first()?;

    // l_broadcast must trace (broadcast -> reshape) to `reduce_sum(P, 1)`.
    if !broadcast_traces_to_reduce(lbcast, p_ssa, "arith.addf", &def) {
        return None;
    }
    // mx_broadcast must trace to `reduce_max(scores_masked, 1)`.
    let mxb = sub.operands.get(1)?;
    if !broadcast_traces_to_reduce(mxb, scores_masked, "arith.maximumf", &def) {
        return None;
    }

    // scores_masked is either `arith.addf(scaled, mask)` (causal) or `scaled`.
    let (scaled, causal) = {
        let smop = def.get(scores_masked)?;
        if smop.op_type == "arith.addf" {
            // one operand is the scaled scores, the other the causal mask tensor.
            (smop.operands.first()?.clone(), true)
        } else {
            (scores_masked.clone(), false)
        }
    };

    // scaled = arith.mulf(raw_scores, scale_splat)
    let mulop = def.get(&scaled)?;
    if mulop.op_type != "arith.mulf" {
        return None;
    }
    let raw_scores = mulop.operands.first()?;
    let scale_splat = mulop.operands.get(1)?;
    let scale = recover_scale(scale_splat, &def)?;

    // raw_scores = linalg.matmul(Q, Kt)
    let qk = def.get(raw_scores)?;
    if qk.op_type != "linalg.matmul" {
        return None;
    }
    let q_loaded = qk.operands.first()?;
    let kt_ssa = qk.operands.get(1)?;
    let (q_arg, q_shape, dtype) = load_src.get(q_loaded)?.clone();

    // Kt = linalg.transpose(K_loaded)
    let ktop = def.get(kt_ssa)?;
    if ktop.op_type != "linalg.transpose" {
        return None;
    }
    let k_loaded = ktop.operands.first()?;
    let (k_arg, k_shape, _kdt) = load_src.get(k_loaded)?.clone();

    // ---- shape sanity: Q[m,d], K[cap,d], V[cap,d], O[m,d] ----
    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 || o_shape.len() != 2 {
        return None;
    }
    let (m, d) = (q_shape[0], q_shape[1]);
    let (cap, kd) = (k_shape[0], k_shape[1]);
    if kd != d || v_shape != k_shape || o_shape != q_shape {
        return None;
    }
    if m <= 0 || cap <= 0 || d <= 0 {
        return None;
    }

    Some(AttentionIsland {
        q_arg,
        q_shape,
        k_arg,
        k_shape,
        v_arg,
        v_shape,
        o_arg,
        o_shape,
        m,
        cap,
        d,
        scale,
        causal,
        dtype,
    })
}

/// True if `bcast_ssa` is a `linalg.broadcast` whose source traces back through
/// an optional `tensor.reshape`/`tensor.extract` to `linalg.reduce { reduce_fn }`
/// over `target` (a per-row reduce of the scores/probabilities). This is the
/// `mx_broadcast` / `l_broadcast` chain the canonical softmax emits.
fn broadcast_traces_to_reduce(
    bcast_ssa: &str,
    target: &str,
    reduce_fn: &str,
    def: &HashMap<String, &Operation>,
) -> bool {
    let Some(bop) = def.get(bcast_ssa) else {
        return false;
    };
    if bop.op_type != "linalg.broadcast" {
        return false;
    }
    let Some(src) = bop.operands.first() else {
        return false;
    };
    traces_to_reduce_of(src, target, reduce_fn, def)
}

/// True if `ssa` is `linalg.reduce { reduce_fn }(target)` over the last axis,
/// possibly via a `tensor.reshape` / `tensor.extract` wrapper.
fn traces_to_reduce_of(
    ssa: &str,
    target: &str,
    reduce_fn: &str,
    def: &HashMap<String, &Operation>,
) -> bool {
    let mut cur = ssa.to_string();
    // Skip a chain of reshape/extract wrappers (reduce -> [m] -> reshape [m,1]).
    for _ in 0..4 {
        let Some(op) = def.get(&cur) else {
            return false;
        };
        if matches!(
            op.op_type.as_str(),
            "tensor.reshape" | "tensor.extract" | "tensor.expand_shape" | "tensor.collapse_shape"
        ) {
            match op.operands.first() {
                Some(src) => cur = src.clone(),
                None => return false,
            }
        } else {
            break;
        }
    }
    let Some(op) = def.get(&cur) else {
        return false;
    };
    if op.op_type != "linalg.reduce" {
        return false;
    }
    let fn_ok = matches!(op.attributes.get("reduce_fn"), Some(Attr::Str(s)) if s == reduce_fn);
    let target_ok = op.operands.first().map(|o| o == target).unwrap_or(false);
    fn_ok && target_ok
}

/// Recover the `1/sqrt(d)` scale from a `tensor.splat %c` whose `%c` is an
/// `arith.constant` float.
fn recover_scale(splat_ssa: &str, def: &HashMap<String, &Operation>) -> Option<f32> {
    let splat = def.get(splat_ssa)?;
    if splat.op_type != "tensor.splat" {
        return None;
    }
    let c = def.get(splat.operands.first()?)?;
    if c.op_type != "arith.constant" {
        return None;
    }
    match c.attributes.get("value") {
        Some(Attr::Float(f)) => Some(*f as f32),
        Some(Attr::Int(i)) => Some(*i as f32),
        _ => None,
    }
}

// ===========================================================================
// Tiling (online-softmax rewrite)
// ===========================================================================

/// Block size for the KV/cap loop. Chosen so the per-block scores tile `[m, BC]`
/// is comfortably below LX for the `m` the model uses; `cap` is split into
/// `ceil(cap / BC)` blocks. A power of two that divides the common caps (256,
/// 512, 1024, 2048, 4096) cleanly when possible; the loop handles a ragged tail
/// via a clamped block size.
const DEFAULT_KV_BLOCK: i64 = 128;

/// Choose a KV block size that (a) does not exceed the cap and (b) divides it
/// when a clean divisor near the default exists, else falls back to the default
/// (the loop's static unroll below handles any remainder by clamping).
fn choose_block(cap: i64) -> i64 {
    if cap <= DEFAULT_KV_BLOCK {
        return cap;
    }
    // Prefer the largest divisor of `cap` that is <= DEFAULT_KV_BLOCK and a power
    // of two, to keep every block equal-sized (no ragged tail to special-case).
    for b in [DEFAULT_KV_BLOCK, 64, 32, 16, 8, 4, 2, 1] {
        if cap % b == 0 {
            return b;
        }
    }
    1
}

/// All divisors of `cap` that are `<= DEFAULT_KV_BLOCK`, descending (so the first
/// fitting one is the largest equal-sized block). Always includes `1`.
fn cap_divisors(cap: i64) -> Vec<i64> {
    let mut ds: Vec<i64> = (1..=cap.min(DEFAULT_KV_BLOCK))
        .filter(|b| cap % b == 0)
        .collect();
    ds.sort_unstable_by(|a, b| b.cmp(a));
    ds
}

/// Budget-aware KV block size for the RE-ROLLED context tiling: the LARGEST
/// divisor `b` of `cap` (`b <= DEFAULT_KV_BLOCK`) whose per-block scores tile
/// `[m, b]` is BELOW the cap (`!needs_flash(m*b*bytes)`), so each block fits LX.
///
/// If even the smallest divisor still overflows (a pathologically tiny forced
/// budget), fall back to that smallest divisor — the most aggressive tiling we
/// can emit. In all cases `b <= cap`; when `cap` has a proper divisor `< cap`
/// (the real caps are 64-multiples) and the full `[m, cap]` tile overflows, the
/// returned `b` is strictly `< cap`, so REAL tiling happens.
/// Constrains BOTH per-block tiles: the `[m, blk]` scores tile must fit the
/// injected LX budget (`!needs_flash`) AND the `[blk, d]` context K/V tile must
/// fit [`FLASH_CONTEXT_TILE_MAX`]. In the small-query/long-context regime the KV
/// constraint binds (scores are already tiny), so a scores-only chooser would
/// pick `blk = cap` (no tiling) and overflow LX; this picks the largest cap
/// divisor that satisfies both.
fn choose_block_budgeted(
    m: i64,
    cap: i64,
    d: i64,
    bytes: usize,
    needs_flash: &impl Fn(usize) -> bool,
) -> i64 {
    let divisors = cap_divisors(cap);
    let scores = |b: i64| {
        (m as usize)
            .saturating_mul(b as usize)
            .saturating_mul(bytes)
    };
    let kv = |b: i64| {
        (d as usize)
            .saturating_mul(b as usize)
            .saturating_mul(bytes)
    };
    // Largest divisor whose per-block scores AND context K/V tiles both fit.
    for &b in &divisors {
        if !needs_flash(scores(b)) && kv(b) <= FLASH_CONTEXT_TILE_MAX {
            return b;
        }
    }
    // None fits: take the smallest divisor (the minimal achievable tile). When the
    // full tile overflows but no sub-block formally "fits", we STILL tile to the
    // smallest block (strictly smaller footprint) — honest best effort.
    *divisors.last().unwrap_or(&1)
}

/// A tiny monotonic counter so the rewritten function's fresh SSA names never
/// collide with each other across a multi-node rewrite.
struct NameGen {
    n: usize,
    prefix: String,
}
impl NameGen {
    fn new(prefix: &str) -> Self {
        NameGen {
            n: 0,
            prefix: prefix.to_string(),
        }
    }
    fn next(&mut self, tag: &str) -> String {
        let s = format!("%{}_{}_{}", self.prefix, tag, self.n);
        self.n += 1;
        s
    }
}

fn const_index(name: &str, v: i64) -> Operation {
    Operation::new(Some(name), "arith.constant", &[]).with_attr("value", Attr::Int(v))
}
fn const_f(name: &str, v: f64) -> Operation {
    Operation::new(Some(name), "arith.constant", &[]).with_attr("value", Attr::Float(v))
}

/// Build a `ktdp.construct_memory_view %ptr {shape, strides, memory_space, dtype}`
/// — a logical view only (RFC 0682: does NOT allocate).
fn mk_view(res: &str, ptr: &str, shape: &[i64], dtype: &str) -> Operation {
    // Row-major strides.
    let mut strides = vec![1i64; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    Operation::new(Some(res), "ktdp.construct_memory_view", &[ptr])
        .with_attr("shape", Attr::IntList(shape.to_vec()))
        .with_attr("strides", Attr::IntList(strides))
        .with_attr("memory_space", Attr::Str("HBM".into()))
        .with_attr("dtype", Attr::Str(dtype.into()))
}

/// Whole-tensor `ktdp.load` of a view: build the full-shape access tile then load.
fn mk_whole_load(g: &mut NameGen, ops: &mut Vec<Operation>, view: &str, shape: &[i64]) -> String {
    let tile = g.next("at");
    ops.push(
        Operation::new(Some(&tile), "ktdp.construct_access_tile", &[view])
            .with_attr("shape", Attr::IntList(shape.to_vec())),
    );
    let loaded = g.next("ld");
    ops.push(Operation::new(Some(&loaded), "ktdp.load", &[&tile]));
    loaded
}

/// A `tensor.splat %scalar -> tensor<shape×dtype>`.
fn mk_splat(res: &str, scalar: &str, shape: &[i64], dtype: &str) -> Operation {
    Operation::new(Some(res), "tensor.splat", &[scalar])
        .with_attr("shape", Attr::IntList(shape.to_vec()))
        .with_attr("dtype", Attr::Str(dtype.into()))
}

/// A `tensor.empty() -> tensor<shape×dtype>` (zero-filled init for matmul outs).
fn mk_empty(res: &str, shape: &[i64], dtype: &str) -> Operation {
    Operation::new(Some(res), "tensor.empty", &[])
        .with_attr("shape", Attr::IntList(shape.to_vec()))
        .with_attr("dtype", Attr::Str(dtype.into()))
}

/// A `linalg.reduce { reduce_fn } ins(%x) outs(%init) dimensions = [1]` over the
/// last axis of a `[r, c]` tile -> `[r]`.
fn mk_reduce(res: &str, x: &str, init: &str, reduce_fn: &str) -> Operation {
    Operation::new(Some(res), "linalg.reduce", &[x])
        .with_attr("reduce_fn", Attr::Str(reduce_fn.into()))
        .with_attr("dimensions", Attr::IntList(vec![1]))
        .with_attr("outs_var", Attr::Str(init.into()))
}

/// Rewrite a recognized [`AttentionIsland`] into a tiled online-softmax function.
///
/// The emitted body is, for `nb = ceil(cap / BC)` KV blocks of size `BC`:
/// ```text
///   Q = load Q[m,d]
///   m0 = splat(-inf, [m,1]); l0 = splat(0, [m,1]); acc0 = empty([m,d])
///   (m_f, l_f, acc_f) = scf.for j = 0 to nb step 1 iter_args(m_i, l_i, acc):
///       Kj = extract_slice K[j*BC .. , :]      ([BC, d])
///       Vj = extract_slice V[j*BC .. , :]      ([BC, d])
///       Sj = (Q @ Kjᵀ) * scale  (+ causal mask_j)         ([m, BC])
///       rmax = reduce_max(Sj, 1) -> [m]
///       m_new = max(m_i, rmax_bcast)                       ([m,1])
///       P = exp(Sj - m_new_bcast)                          ([m, BC])
///       alpha = exp(m_i - m_new)                           ([m,1])
///       rsum = reduce_sum(P, 1) -> [m]
///       l_new = alpha*l_i + rsum_bcast                     ([m,1])
///       acc_new = alpha_bcast*acc + P @ Vj                 ([m,d])
///       yield m_new, l_new, acc_new
///   O = acc_f / l_f_bcast
///   store O -> O[m,d]
/// ```
/// Causal masking is applied as `-inf` on KV positions `> (q_row + (cap - m))`
/// per block, matched to the naive form's mask. The `[m, BC]` block scores tile
/// fits LX by construction (BC = `choose_block(cap)` ≤ 128).
pub fn tile_attention(island: &AttentionIsland) -> IRFunction {
    let isl = island;
    let dt = isl.dtype.as_str();
    let bc = choose_block(isl.cap);
    let nb = isl.cap / bc; // choose_block guarantees bc | cap
    let mut g = NameGen::new("fa");
    let mut ops: Vec<Operation> = Vec::new();

    // ---- constants ----
    let c_neg_inf = g.next("ninf");
    ops.push(const_f(&c_neg_inf, -1.0e30));
    let c_zero = g.next("zero");
    ops.push(const_f(&c_zero, 0.0));
    let c_scale = g.next("scale");
    ops.push(const_f(&c_scale, isl.scale as f64));

    // A zero column-offset constant for the per-block KV access tiles. Kept an
    // SSA operand (not a literal in an attribute) so whole-program fusion's
    // operand renaming threads it through correctly.
    let c0 = g.next("c0");
    ops.push(const_index(&c0, 0));

    // ---- load Q whole ----
    let q_view = g.next("qv");
    ops.push(mk_view(&q_view, &isl.q_arg, &isl.q_shape, dt));
    let q = mk_whole_load(&mut g, &mut ops, &q_view, &isl.q_shape);

    // ---- build K / V views (loaded per-block inside the loop) ----
    // Each KV block is read straight from HBM with a `ktdp.construct_access_tile`
    // at the dynamic block offset `[j*BC, 0]` (its index operands are renamed by
    // fusion) — exactly the "only the `[BC, d]` block is resident" behavior that
    // keeps the scores tile inside LX. This is the KTIR-native analogue of an
    // `extract_slice` of the KV block and avoids materializing the whole `[cap,d]`
    // tensor in LX.
    let k_view = g.next("kv");
    ops.push(mk_view(&k_view, &isl.k_arg, &isl.k_shape, dt));
    let v_view = g.next("vv");
    ops.push(mk_view(&v_view, &isl.v_arg, &isl.v_shape, dt));

    // ---- iter-arg inits ----
    let m0 = g.next("m0");
    ops.push(mk_splat(&m0, &c_neg_inf, &[isl.m, 1], dt));
    let l0 = g.next("l0");
    ops.push(mk_splat(&l0, &c_zero, &[isl.m, 1], dt));
    let acc0 = g.next("acc0");
    ops.push(mk_empty(&acc0, &[isl.m, isl.d], dt));

    // ---- loop bounds ----
    let lb = g.next("lb");
    ops.push(const_index(&lb, 0));
    let ub = g.next("ub");
    ops.push(const_index(&ub, nb));
    let step = g.next("st");
    ops.push(const_index(&step, 1));
    let bc_c = g.next("bc");
    ops.push(const_index(&bc_c, bc));

    // iter-arg body-visible names.
    let mi = "%fa_mi".to_string();
    let li = "%fa_li".to_string();
    let acci = "%fa_acci".to_string();
    let iv = "%fa_j".to_string();

    // ---- loop body ----
    let mut body: Vec<Operation> = Vec::new();
    // block start offset = j * BC
    let off = g.next("off");
    body.push(Operation::new(Some(&off), "arith.muli", &[&iv, &bc_c]));

    // Kj = load K[off.., :]  -> [BC, d]  (KTIR access tile at the block offset)
    let kj = block_load(&mut g, &mut body, &k_view, &off, &c0, bc, isl.d);
    // Vj = load V[off.., :]  -> [BC, d]
    let vj = block_load(&mut g, &mut body, &v_view, &off, &c0, bc, isl.d);

    // Kjt = transpose(Kj) -> [d, BC]
    let kjt_init = g.next("kjti");
    body.push(mk_empty(&kjt_init, &[isl.d, bc], dt));
    let kjt = g.next("kjt");
    body.push(
        Operation::new(Some(&kjt), "linalg.transpose", &[&kj, &kjt_init])
            .with_attr("permutation", Attr::IntList(vec![1, 0])),
    );

    // raw = Q @ Kjt -> [m, BC]
    let raw_init = g.next("rawi");
    body.push(mk_empty(&raw_init, &[isl.m, bc], dt));
    let raw = g.next("raw");
    body.push(Operation::new(
        Some(&raw),
        "linalg.matmul",
        &[&q, &kjt, &raw_init],
    ));

    // scaled = raw * scale_splat
    let scale_t = g.next("sct");
    body.push(mk_splat(&scale_t, &c_scale, &[isl.m, bc], dt));
    let scaled = g.next("scaled");
    body.push(Operation::new(
        Some(&scaled),
        "arith.mulf",
        &[&raw, &scale_t],
    ));

    // sj = scaled (+ causal mask for this block, if causal)
    let sj = if isl.causal {
        let mask = causal_mask_block(&mut g, &mut body, &off, isl.m, isl.cap, bc, dt);
        let masked = g.next("sjm");
        body.push(Operation::new(
            Some(&masked),
            "arith.addf",
            &[&scaled, &mask],
        ));
        masked
    } else {
        scaled
    };

    // rmax = reduce_max(sj, 1) -> [m]
    let rmax_init = g.next("rmi");
    body.push(mk_splat(&rmax_init, &c_neg_inf, &[isl.m], dt));
    let rmax = g.next("rmax");
    body.push(mk_reduce(&rmax, &sj, &rmax_init, "arith.maximumf"));
    // reduce yields [m]; reshape to [m,1] for elementwise with the [m,1] iter-args.
    let rmax2 = g.next("rmax2");
    body.push(reshape_to(&rmax2, &rmax, &[isl.m, 1]));

    // m_new = max(m_i, rmax2)
    let mnew = g.next("mnew");
    body.push(Operation::new(
        Some(&mnew),
        "arith.maximumf",
        &[&mi, &rmax2],
    ));

    // m_new broadcast to [m, BC]
    let mnew_b = broadcast_col_to(&mut g, &mut body, &mnew, isl.m, bc, dt);
    // shifted = sj - m_new_b
    let shifted = g.next("shift");
    body.push(Operation::new(
        Some(&shifted),
        "arith.subf",
        &[&sj, &mnew_b],
    ));
    // P = exp(shifted) -> [m, BC]
    let p = g.next("p");
    body.push(Operation::new(Some(&p), "math.exp", &[&shifted]));

    // alpha = exp(m_i - m_new) -> [m,1]
    let mdiff = g.next("mdiff");
    body.push(Operation::new(Some(&mdiff), "arith.subf", &[&mi, &mnew]));
    let alpha = g.next("alpha");
    body.push(Operation::new(Some(&alpha), "math.exp", &[&mdiff]));

    // rsum = reduce_sum(P, 1) -> [m] -> [m,1]
    let rsum_init = g.next("rsi");
    body.push(mk_splat(&rsum_init, &c_zero, &[isl.m], dt));
    let rsum = g.next("rsum");
    body.push(mk_reduce(&rsum, &p, &rsum_init, "arith.addf"));
    let rsum2 = g.next("rsum2");
    body.push(reshape_to(&rsum2, &rsum, &[isl.m, 1]));

    // l_new = alpha * l_i + rsum2
    let al = g.next("al");
    body.push(Operation::new(Some(&al), "arith.mulf", &[&alpha, &li]));
    let lnew = g.next("lnew");
    body.push(Operation::new(Some(&lnew), "arith.addf", &[&al, &rsum2]));

    // acc_new = alpha_b * acc + P @ Vj
    let alpha_b = broadcast_col_to(&mut g, &mut body, &alpha, isl.m, isl.d, dt);
    let acc_scaled = g.next("accs");
    body.push(Operation::new(
        Some(&acc_scaled),
        "arith.mulf",
        &[&alpha_b, &acci],
    ));
    let pv_init = g.next("pvi");
    body.push(mk_empty(&pv_init, &[isl.m, isl.d], dt));
    let pv = g.next("pv");
    body.push(Operation::new(
        Some(&pv),
        "linalg.matmul",
        &[&p, &vj, &pv_init],
    ));
    let accnew = g.next("accnew");
    body.push(Operation::new(
        Some(&accnew),
        "arith.addf",
        &[&acc_scaled, &pv],
    ));

    // yield m_new, l_new, acc_new
    body.push(Operation::new(None, "scf.yield", &[&mnew, &lnew, &accnew]));

    // ---- the scf.for ----
    let m_f = g.next("mf");
    let l_f = g.next("lf");
    let acc_f = g.next("accf");
    let mut forop = Operation::new(None, "scf.for", &[&lb, &ub, &step, &m0, &l0, &acc0])
        .with_attr("iter_var", Attr::Str(iv.clone()))
        .with_attr(
            "iter_args",
            Attr::StrList(vec![mi.clone(), li.clone(), acci.clone()]),
        )
        .with_attr(
            "result_names",
            Attr::StrList(vec![m_f.clone(), l_f.clone(), acc_f.clone()]),
        );
    forop.regions = vec![body];
    ops.push(forop);

    // ---- final normalize: O = acc_f / l_f_b ----
    let l_f_b = broadcast_col_to(&mut g, &mut ops, &l_f, isl.m, isl.d, dt);
    let o = g.next("o");
    ops.push(Operation::new(Some(&o), "arith.divf", &[&acc_f, &l_f_b]));

    // ---- store O -> O[m,d] ----
    let o_view = g.next("ov");
    ops.push(mk_view(&o_view, &isl.o_arg, &isl.o_shape, dt));
    let o_at = g.next("oat");
    ops.push(
        Operation::new(Some(&o_at), "ktdp.construct_access_tile", &[&o_view])
            .with_attr("shape", Attr::IntList(isl.o_shape.clone())),
    );
    ops.push(Operation::new(None, "ktdp.store", &[&o, &o_at]));
    ops.push(Operation::new(None, "func.return", &[]));

    IRFunction {
        name: String::new(), // caller stamps the original name
        arguments: vec![
            (isl.q_arg.clone(), "index".into()),
            (isl.k_arg.clone(), "index".into()),
            (isl.v_arg.clone(), "index".into()),
            (isl.o_arg.clone(), "index".into()),
        ],
        operations: ops,
        grid: (1, 1, 1),
        return_type: None,
    }
}

/// Load a `[rows, cols]` block of an HBM `[*, cols]` view at dynamic row offset
/// `%off` (column offset `%c0`): `construct_access_tile %view[%off, %c0]` (block
/// shape) then `ktdp.load`. The access-tile index operands `%off`/`%c0` are real
/// SSA operands, so whole-program fusion's operand renaming threads them through
/// a fused segment correctly (a `tensor.extract_slice`'s `slice_offsets` live in
/// an attribute fusion does not rewrite — using the KTIR access tile sidesteps
/// that, and is the hardware-native "only the resident block is in LX" form).
fn block_load(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    view: &str,
    off: &str,
    c0: &str,
    rows: i64,
    cols: i64,
) -> String {
    let at = g.next("kvat");
    ops.push(
        Operation::new(Some(&at), "ktdp.construct_access_tile", &[view, off, c0])
            .with_attr("shape", Attr::IntList(vec![rows, cols])),
    );
    let loaded = g.next("kvld");
    ops.push(Operation::new(Some(&loaded), "ktdp.load", &[&at]));
    loaded
}

/// `%r = tensor.reshape %x -> tensor<shape>` (a pure reinterpretation; the
/// interpreter reads `target_shape`).
fn reshape_to(res: &str, x: &str, shape: &[i64]) -> Operation {
    Operation::new(Some(res), "tensor.reshape", &[x])
        .with_attr("target_shape", Attr::IntList(shape.to_vec()))
}

/// Broadcast a `[m, 1]` column tile to `[m, cols]`, pushing the ops onto `ops`
/// and returning the result SSA. Uses `linalg.broadcast ins(%col) outs(%init)`
/// with an empty `dimensions` list: the interpreter then NumPy right-aligned-
/// broadcasts the `[m,1]` input up to the `[m,cols]` outs shape (each row's
/// single value filled across the `cols` columns). The `outs` tile supplies the
/// target shape, so we materialize it with `tensor.empty` first.
fn broadcast_col_to(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    col: &str,
    m: i64,
    cols: i64,
    dt: &str,
) -> String {
    let init = g.next("bci");
    ops.push(mk_empty(&init, &[m, cols], dt));
    let res = g.next("bc");
    ops.push(
        Operation::new(Some(&res), "linalg.broadcast", &[col, &init])
            .with_attr("dimensions", Attr::IntList(vec![])),
    );
    res
}

/// Emit the per-block causal mask `[m, BC]` (pushing ops, returning its SSA),
/// using only ELEMENTWISE + CONSTANT ops (no region block-args) so it survives
/// whole-program fusion's operand renaming.
///
/// Visibility rule (matched to the naive form): query row `qr` has absolute KV
/// position `cap - m + qr` and attends to key block position `off + kc` iff
/// `off + kc <= cap - m + qr`, i.e. `kc - qr <= (cap - m) - off`. The left side
/// `D[qr,kc] = kc - qr` is a STATIC `[m, BC]` integer constant (baked at emit
/// time); the right side `rhs = (cap - m) - off` is a per-iteration scalar. The
/// mask is then `select(D <= rhs, 0, -inf)` — all elementwise.
fn causal_mask_block(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    off: &str,
    m: i64,
    cap: i64,
    bc: i64,
    dt: &str,
) -> String {
    // D[qr,kc] = kc - qr, baked as a dense i32 constant tensor.
    let mut d_vals = Vec::with_capacity((m * bc) as usize);
    for qr in 0..m {
        for kc in 0..bc {
            d_vals.push(kc - qr);
        }
    }
    let d = g.next("maskD");
    ops.push(
        Operation::new(Some(&d), "arith.constant", &[])
            .with_attr("is_tensor", Attr::Bool(true))
            .with_attr("dense_list", Attr::Bool(true))
            .with_attr("shape", Attr::IntList(vec![m, bc]))
            .with_attr("dtype", Attr::Str("i32".into()))
            .with_attr("value", Attr::IntList(d_vals)),
    );
    // rhs = (cap - m) - off  (scalar index).
    let base = g.next("maskBase");
    ops.push(const_index(&base, cap - m));
    let rhs = g.next("maskRhs");
    ops.push(Operation::new(Some(&rhs), "arith.subi", &[&base, off]));
    let rhs_t = g.next("maskRhsT");
    ops.push(
        Operation::new(Some(&rhs_t), "tensor.splat", &[&rhs])
            .with_attr("shape", Attr::IntList(vec![m, bc]))
            .with_attr("dtype", Attr::Str("i32".into())),
    );
    // cond = D <= rhs_t  (elementwise i1 tile).
    let cond = g.next("maskCond");
    ops.push(
        Operation::new(Some(&cond), "arith.cmpi", &[&d, &rhs_t])
            .with_attr("predicate", Attr::Str("sle".into())),
    );
    // visible -> 0, masked -> -inf  (elementwise select into f16).
    let zero_c = g.next("maskZ");
    ops.push(const_f(&zero_c, 0.0));
    let zero_t = g.next("maskZT");
    ops.push(mk_splat(&zero_t, &zero_c, &[m, bc], dt));
    let ninf_c = g.next("maskN");
    ops.push(const_f(&ninf_c, -1.0e30));
    let ninf_t = g.next("maskNT");
    ops.push(mk_splat(&ninf_t, &ninf_c, &[m, bc], dt));
    let mask = g.next("mask");
    ops.push(Operation::new(
        Some(&mask),
        "arith.select",
        &[&cond, &zero_t, &ninf_t],
    ));
    mask
}

// ===========================================================================
// RE-ROLLED recognizer + tiler (the REAL model node, post head_rewrite)
// ===========================================================================

/// Decoded `ktdp.load` source: the pointer arg, full view shape, and dtype.
#[derive(Clone)]
struct RrLoad {
    arg: String,
    view_shape: Vec<i64>,
    dtype: String,
}

/// One recognized softmax block (context OR diagonal) walked back from its `A·V`
/// matmul: the loaded V source, the loaded Q source, the loaded K source, the
/// scaled-scores SSA (`mulf(Q·Kᵀ, scale)`), the exp argument (`subf(S, gm_bc)`),
/// the masked-scores SSA `S` (`addf(scaled, mask)`), and the running-sum SSA fed
/// into the global `gs`.
struct RrBlock {
    v: RrLoad,
    q: RrLoad,
    k: RrLoad,
    scale: f32,
    /// The masked-scores tensor `S` (post mask add) — the reduce / subf operand.
    masked_scores: String,
    /// The per-block exp probabilities (`pc` / `pd`).
    probs: String,
    /// The per-block row-sum SSA (`scs` / `sds`) — the `gs = addf(.,.)` operand.
    row_sum: String,
}

/// Recognize the RE-ROLLED two-block head-attention idiom (the
/// [`crate::head_rewrite`] OUTPUT — the REAL model node).
///
/// Returns `Some(island)` only when the body is PROVABLY:
///   * grid `(H, 1, 1)` with `H > 1`; no top-level `scf.*` (so a re-tiled body is
///     never re-recognized);
///   * exactly ONE `ktdp.store` whose value is `arith.addf(ovc, ovd)` with both
///     args `linalg.matmul` (the two `A·V`);
///   * the CONTEXT block (V/K from a `[cap, kv_cols]` view, mask a
///     `linalg.broadcast` of a `[1, cap]` load) and the DIAGONAL block (V/K from
///     a `[m, kv_cols]` view, mask a dense `[m, m]` `arith.constant`), each a
///     `divf(exp(subf(addf(mulf(matmul(Q,Kᵀ),scale),mask), gm_bc)), gs_bc)`;
///   * ONE shared `gm = arith.maximumf(mc, md)` and ONE shared
///     `gs = arith.addf(reduce_sum(pc), reduce_sum(pd))`;
///   * Q is the SAME load arg for both blocks.
///
/// Any deviation → `None` → identity. `cap`, `m`, `d`, `gqac`, `hdc`, `scale`,
/// `ninf` are all RE-DERIVED from the IR (never model names or literal shapes).
pub fn recognize_rerolled_attention(func: &IRFunction) -> Option<ReRolledIsland> {
    // grid = [H,1,1], H > 1 (head-parallel).
    let (h, gy, gz) = func.grid;
    if gy != 1 || gz != 1 || h <= 1 {
        return None;
    }
    let h = h as i64;

    // No top-level control flow: a body that already contains an `scf.for` is the
    // already-cap-tiled form (or something else) — never re-recognize it.
    if func.operations.iter().any(|op| {
        matches!(
            op.op_type.as_str(),
            "scf.for" | "scf.if" | "scf.while" | "scf.parallel" | "scf.forall"
        )
    }) {
        return None;
    }

    // Index views / access tiles / loads / defs / int-constants.
    let mut views: HashMap<String, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<String, TileInfo> = HashMap::new();
    let mut load_src: HashMap<String, RrLoad> = HashMap::new();
    let mut def: HashMap<String, &Operation> = HashMap::new();
    let mut int_const: HashMap<String, i64> = HashMap::new();

    for op in &func.operations {
        match op.op_type.as_str() {
            "ktdp.construct_memory_view" => {
                if let (Some(res), Some(arg)) = (&op.result, op.operands.first()) {
                    views.insert(
                        res.clone(),
                        ViewInfo {
                            arg: arg.clone(),
                            shape: shape_attr(op),
                            dtype: dtype_attr(op),
                        },
                    );
                }
            }
            "ktdp.construct_access_tile" => {
                if let (Some(res), Some(view)) = (&op.result, op.operands.first()) {
                    tiles.insert(res.clone(), TileInfo { view: view.clone() });
                }
            }
            "ktdp.load" => {
                if let (Some(res), Some(tile)) = (&op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(
                        res.clone(),
                        RrLoad {
                            arg: vi.arg.clone(),
                            view_shape: vi.shape.clone(),
                            dtype: vi.dtype.clone(),
                        },
                    );
                }
            }
            "arith.constant" => {
                if let (Some(res), Some(Attr::Int(v))) = (&op.result, op.attributes.get("value")) {
                    int_const.insert(res.clone(), *v);
                }
            }
            _ => {}
        }
        if let Some(res) = &op.result {
            def.insert(res.clone(), op);
        }
    }

    // Per-head selection arithmetic (PRESERVED verbatim by the tiler): a
    // `get_compute_tile_id`, a `divui %hpid, %gqac`, a `muli %hpid, %hdc`.
    if !func
        .operations
        .iter()
        .any(|o| o.op_type == "ktdp.get_compute_tile_id")
    {
        return None;
    }
    let gqac = func
        .operations
        .iter()
        .find(|o| o.op_type == "arith.divui")
        .and_then(|o| o.operands.get(1))
        .and_then(|c| int_const.get(c).copied())?;
    if gqac < 1 {
        return None;
    }

    // Exactly ONE store; its value = arith.addf(ovc, ovd).
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == "ktdp.store")
        .collect();
    let store = match stores.as_slice() {
        [s] => s,
        _ => return None,
    };
    let stored_val = store.operands.first()?;
    let add = def.get(stored_val)?;
    if add.op_type != "arith.addf" {
        return None;
    }
    let ov0 = add.operands.first()?;
    let ov1 = add.operands.get(1)?;

    // Walk back BOTH `ov = matmul(w, v)` summands into softmax blocks.
    let blk0 = rr_walk_block(ov0, &def, &load_src)?;
    let blk1 = rr_walk_block(ov1, &def, &load_src)?;

    // Disambiguate context vs diagonal by V-view ROWS (cap vs m), NOT operand
    // order. The context V view has `rows == cap`, the diagonal `rows == m`. They
    // must differ (otherwise we cannot tell them apart → fail-safe).
    if blk0.v.view_shape.len() != 2 || blk1.v.view_shape.len() != 2 {
        return None;
    }
    let (ctx, diag) = if rr_is_context(&blk0, &def) && !rr_is_context(&blk1, &def) {
        (&blk0, &blk1)
    } else if rr_is_context(&blk1, &def) && !rr_is_context(&blk0, &def) {
        (&blk1, &blk0)
    } else {
        return None; // ambiguous: both or neither look like the context block.
    };

    // Q must be the SAME load arg for both blocks (one query tile).
    if ctx.q.arg != diag.q.arg {
        return None;
    }
    if (ctx.scale - diag.scale).abs() > 1e-4 {
        return None;
    }

    // ONE shared gm = maximumf(mc, md): both blocks' exp args subtract the SAME
    // broadcast of `gm`, and that gm is `maximumf(reduce_max(Sc), reduce_max(Sd))`.
    let gm = rr_shared_gm(ctx, diag, &def)?;
    if !rr_gm_is_max_of_reduces(&gm, &ctx.masked_scores, &diag.masked_scores, &def) {
        return None;
    }

    // ONE shared gs = addf(reduce_sum(pc), reduce_sum(pd)); each block's divf
    // denominator broadcasts THIS gs.
    rr_check_shared_gs(ctx, diag, &def)?;

    // ---- shape recovery (all RE-DERIVED) ----
    // Q/O view [m, q_cols]; q_cols = H*d.
    let q_shape = &ctx.q.view_shape;
    if q_shape.len() != 2 {
        return None;
    }
    let (m, q_cols) = (q_shape[0], q_shape[1]);
    if m <= 0 || q_cols % h != 0 {
        return None;
    }
    let d = q_cols / h;
    if d <= 0 {
        return None;
    }
    // Context K/V view [cap, kv_cols]; diagonal K/V view [m, kv_cols].
    let cap = ctx.v.view_shape[0];
    let kv_cols = ctx.v.view_shape[1];
    if cap <= 0 || kv_cols <= 0 || kv_cols % d != 0 {
        return None;
    }
    if ctx.k.view_shape != [cap, kv_cols] {
        return None;
    }
    if diag.k.view_shape != [m, kv_cols] || diag.v.view_shape != [m, kv_cols] {
        return None;
    }

    // O view [m, q_cols] + its pointer arg, from the store tile.
    let store_tile = store.operands.get(1)?;
    let o_ti = tiles.get(store_tile)?;
    let o_view = views.get(&o_ti.view)?;
    if o_view.shape != *q_shape {
        return None;
    }

    // Mask view [1, cap] (context per-head mask) + pointer arg.
    let mask_arg = rr_context_mask_arg(ctx, &def, &load_src, cap)?;

    // -inf recovered from the diagonal triangular mask constant (else default).
    let ninf = rr_recover_tri_ninf(diag, &def).unwrap_or(-1.0e38);

    Some(ReRolledIsland {
        q_arg: ctx.q.arg.clone(),
        o_arg: o_view.arg.clone(),
        mask_arg,
        kc_arg: ctx.k.arg.clone(),
        kd_arg: diag.k.arg.clone(),
        vc_arg: ctx.v.arg.clone(),
        vd_arg: diag.v.arg.clone(),
        q_cols,
        kv_cols,
        m,
        cap,
        d,
        gqac,
        hdc: d,
        h,
        scale: ctx.scale,
        ninf,
        dtype: ctx.q.dtype.clone(),
    })
}

/// Walk back one `ov = linalg.matmul(w, v_loaded)` summand into a softmax block.
/// `w = divf(exp(subf(addf(mulf(matmul(Q, transpose(K)), scale_splat), mask),
/// gm_bc)), gs_bc)`. Returns `None` on any deviation.
fn rr_walk_block(
    ov: &str,
    def: &HashMap<String, &Operation>,
    load_src: &HashMap<String, RrLoad>,
) -> Option<RrBlock> {
    let av = def.get(ov)?;
    if av.op_type != "linalg.matmul" {
        return None;
    }
    let w = av.operands.first()?;
    let v_loaded = av.operands.get(1)?;
    let v = load_src.get(v_loaded)?.clone();

    // w = divf(probs, gs_bc)
    let divw = def.get(w)?;
    if divw.op_type != "arith.divf" {
        return None;
    }
    let probs = divw.operands.first()?.clone();
    // probs = exp(subf(S, gm_bc))
    let pexp = def.get(&probs)?;
    if pexp.op_type != "math.exp" {
        return None;
    }
    let sub = def.get(pexp.operands.first()?)?;
    if sub.op_type != "arith.subf" {
        return None;
    }
    let masked_scores = sub.operands.first()?.clone();

    // S = addf(scaled, mask)
    let sop = def.get(&masked_scores)?;
    if sop.op_type != "arith.addf" {
        return None;
    }
    let scaled = sop.operands.first()?;
    // scaled = mulf(raw, scale_splat)
    let mulop = def.get(scaled)?;
    if mulop.op_type != "arith.mulf" {
        return None;
    }
    let raw = mulop.operands.first()?;
    let scale = recover_scale(mulop.operands.get(1)?, def)?;
    // raw = matmul(Q_loaded, Kt); Kt = transpose(K_loaded)
    let qk = def.get(raw)?;
    if qk.op_type != "linalg.matmul" {
        return None;
    }
    let q_loaded = qk.operands.first()?;
    let q = load_src.get(q_loaded)?.clone();
    let ktop = def.get(qk.operands.get(1)?)?;
    if ktop.op_type != "linalg.transpose" {
        return None;
    }
    let k_loaded = ktop.operands.first()?;
    let k = load_src.get(k_loaded)?.clone();

    // row_sum = the reduce_sum CONSUMING probs (feeds the global gs).
    let row_sum = rr_reduce_consuming(&probs, "arith.addf", def)?;

    Some(RrBlock {
        v,
        q,
        k,
        scale,
        masked_scores,
        probs,
        row_sum,
    })
}

/// True if `blk` is the CONTEXT block: its mask add operand is a
/// `linalg.broadcast` (the per-head `[1, cap]` mask), as opposed to the diagonal
/// block whose mask is a dense `arith.constant` `[m, m]` triangle.
fn rr_is_context(blk: &RrBlock, def: &HashMap<String, &Operation>) -> bool {
    let Some(sop) = def.get(&blk.masked_scores) else {
        return false;
    };
    let Some(mask) = sop.operands.get(1) else {
        return false;
    };
    let Some(mop) = def.get(mask) else {
        return false;
    };
    mop.op_type == "linalg.broadcast"
}

/// Verify both blocks' exp args subtract the SAME `gm` broadcast and return that
/// `gm` SSA. (Each `gm_bc` is `linalg.broadcast(reshape(gm))`.)
fn rr_shared_gm(
    ctx: &RrBlock,
    diag: &RrBlock,
    def: &HashMap<String, &Operation>,
) -> Option<String> {
    let gm_c = rr_broadcast_src(&ctx.probs, def)?;
    let gm_d = rr_broadcast_src(&diag.probs, def)?;
    if gm_c != gm_d {
        return None;
    }
    Some(gm_c)
}

/// From a `probs = exp(subf(S, gm_bc))` SSA, recover the pre-broadcast `gm` SSA by
/// peeling `exp -> subf -> (operand 1) gm_bc -> linalg.broadcast -> reshape`.
fn rr_broadcast_src(probs: &str, def: &HashMap<String, &Operation>) -> Option<String> {
    let pexp = def.get(probs)?;
    let sub = def.get(pexp.operands.first()?)?;
    let gm_bc = sub.operands.get(1)?;
    rr_peel_broadcast_reshape(gm_bc, def)
}

/// Peel `linalg.broadcast(reshape(x))` (or `broadcast(x)`) → `x`.
fn rr_peel_broadcast_reshape(ssa: &str, def: &HashMap<String, &Operation>) -> Option<String> {
    let bop = def.get(ssa)?;
    if bop.op_type != "linalg.broadcast" {
        return None;
    }
    let src = bop.operands.first()?;
    let sop = def.get(src)?;
    if matches!(
        sop.op_type.as_str(),
        "tensor.reshape" | "tensor.expand_shape"
    ) {
        Some(sop.operands.first()?.clone())
    } else {
        Some(src.clone())
    }
}

/// True if `gm = arith.maximumf(reduce_max(sc), reduce_max(sd))` (order-free).
fn rr_gm_is_max_of_reduces(
    gm: &str,
    sc: &str,
    sd: &str,
    def: &HashMap<String, &Operation>,
) -> bool {
    let Some(mop) = def.get(gm) else { return false };
    if mop.op_type != "arith.maximumf" {
        return false;
    }
    let Some(a) = mop.operands.first() else {
        return false;
    };
    let Some(b) = mop.operands.get(1) else {
        return false;
    };
    let a_red = rr_reduce_of(a, "arith.maximumf", def);
    let b_red = rr_reduce_of(b, "arith.maximumf", def);
    // a reduces sc & b reduces sd, OR vice-versa.
    (a_red.as_deref() == Some(sc) && b_red.as_deref() == Some(sd))
        || (a_red.as_deref() == Some(sd) && b_red.as_deref() == Some(sc))
}

/// Find the `linalg.reduce { reduce_fn }` whose input operand is `target`, and
/// return its result SSA (the row vector). `None` if no such reduce exists.
fn rr_reduce_consuming(
    target: &str,
    reduce_fn: &str,
    def: &HashMap<String, &Operation>,
) -> Option<String> {
    let red = def.values().find(|o| {
        o.op_type == "linalg.reduce"
            && o.operands.first().map(|x| x == target).unwrap_or(false)
            && matches!(o.attributes.get("reduce_fn"), Some(Attr::Str(s)) if s == reduce_fn)
    })?;
    red.result.clone()
}

/// If `ssa` is `linalg.reduce { reduce_fn } (target)` (possibly via a reshape
/// wrapper), return the reduced `target`; else `None`.
fn rr_reduce_of(ssa: &str, reduce_fn: &str, def: &HashMap<String, &Operation>) -> Option<String> {
    let mut cur = ssa.to_string();
    for _ in 0..3 {
        let op = def.get(&cur)?;
        if matches!(
            op.op_type.as_str(),
            "tensor.reshape" | "tensor.expand_shape" | "tensor.collapse_shape"
        ) {
            cur = op.operands.first()?.clone();
        } else {
            break;
        }
    }
    let op = def.get(&cur)?;
    if op.op_type != "linalg.reduce" {
        return None;
    }
    if !matches!(op.attributes.get("reduce_fn"), Some(Attr::Str(s)) if s == reduce_fn) {
        return None;
    }
    Some(op.operands.first()?.clone())
}

/// Verify both blocks' `divf` denominators broadcast ONE shared
/// `gs = arith.addf(scs, sds)` where `scs`/`sds` are the two blocks' row sums.
fn rr_check_shared_gs(
    ctx: &RrBlock,
    diag: &RrBlock,
    def: &HashMap<String, &Operation>,
) -> Option<()> {
    let gs_c = rr_divf_denom_src(&ctx.probs, def)?;
    let gs_d = rr_divf_denom_src(&diag.probs, def)?;
    if gs_c != gs_d {
        return None;
    }
    let gsop = def.get(&gs_c)?;
    if gsop.op_type != "arith.addf" {
        return None;
    }
    let a = gsop.operands.first()?;
    let b = gsop.operands.get(1)?;
    let ok = (a == &ctx.row_sum && b == &diag.row_sum) || (a == &diag.row_sum && b == &ctx.row_sum);
    if ok { Some(()) } else { None }
}

/// From a block's `probs`, find the `w = divf(probs, gs_bc)` consumer and peel
/// `gs_bc = broadcast(reshape(gs))` → `gs`.
fn rr_divf_denom_src(probs: &str, def: &HashMap<String, &Operation>) -> Option<String> {
    // Find the divf whose first operand is `probs`.
    let divf = def.values().find(|o| {
        o.op_type == "arith.divf" && o.operands.first().map(|x| x == probs).unwrap_or(false)
    })?;
    let gs_bc = divf.operands.get(1)?;
    rr_peel_broadcast_reshape(gs_bc, def)
}

/// Recover the context-mask pointer arg: the `addf(scaled, mask)` second operand
/// is `linalg.broadcast(mask_load)` where `mask_load` reads a `[1, cap]` view.
fn rr_context_mask_arg(
    ctx: &RrBlock,
    def: &HashMap<String, &Operation>,
    load_src: &HashMap<String, RrLoad>,
    cap: i64,
) -> Option<String> {
    let sop = def.get(&ctx.masked_scores)?;
    let mask = sop.operands.get(1)?;
    let bop = def.get(mask)?;
    if bop.op_type != "linalg.broadcast" {
        return None;
    }
    let mask_loaded = bop.operands.first()?;
    let mc = load_src.get(mask_loaded)?;
    if mc.view_shape != [1, cap] {
        return None;
    }
    Some(mc.arg.clone())
}

/// Recover the `-inf` constant from the diagonal block's dense `[m, m]`
/// triangular mask (`addf(scaled, tri)` where `tri` is an `arith.constant` with a
/// `value` FloatList). The most-negative entry is the `-inf` fill.
fn rr_recover_tri_ninf(diag: &RrBlock, def: &HashMap<String, &Operation>) -> Option<f32> {
    let sop = def.get(&diag.masked_scores)?;
    let tri = sop.operands.get(1)?;
    let top = def.get(tri)?;
    if top.op_type != "arith.constant" {
        return None;
    }
    match top.attributes.get("value") {
        Some(Attr::FloatList(v)) => v
            .iter()
            .cloned()
            .fold(None, |acc, x| match acc {
                Some(a) if a <= x => Some(a),
                _ => Some(x),
            })
            .map(|x| x as f32),
        _ => None,
    }
}

/// Rewrite a recognized [`ReRolledIsland`] — cap-tile ONLY the CONTEXT `[m, cap]`
/// block with online softmax (an `scf.for` over `cap/blk` KV blocks carrying the
/// running max / sum / acc), leaving the small `[m, m]` DIAGONAL whole, then
/// COMBINE both with the SAME global re-association `head_rewrite` uses.
///
/// Preserves grid `[H,1,1]` and the per-head GQA column arithmetic verbatim. The
/// per-block context scores tile is `[m, blk]` (`blk` is a divisor of `cap` chosen
/// by [`choose_block_budgeted`] so the tile fits LX). Emits ONLY RFC-0682 ops
/// (`ktdp` load/store + Arith/Math/LinAlg/Tensor + ONE `scf.for`); NO
/// `tensor.insert_slice`.
pub fn tile_rerolled_attention(isl: &ReRolledIsland, blk: i64) -> IRFunction {
    let dt = isl.dtype.as_str();
    let (m, d, cap) = (isl.m, isl.d, isl.cap);
    // `blk` is a divisor of `cap` (`cap_divisors` only returns divisors), so the
    // block count is exact (no ragged tail).
    let blk = if blk >= 1 && cap % blk == 0 {
        blk
    } else {
        choose_block(cap)
    };
    let mut g = NameGen::new("fa");
    let mut ops: Vec<Operation> = Vec::new();

    // ---- constants ----
    let c0 = g.next("c0");
    ops.push(const_index(&c0, 0));
    let scale_c = g.next("scl");
    ops.push(const_f(&scale_c, isl.scale as f64));
    let ninf_c = g.next("ninf");
    ops.push(const_f(&ninf_c, isl.ninf as f64));
    let zero_c = g.next("zero");
    ops.push(const_f(&zero_c, 0.0));

    // ---- per-head selection arithmetic (PRESERVED verbatim) ----
    let hpid = g.next("hpid");
    ops.push(Operation::new(Some(&hpid), "ktdp.get_compute_tile_id", &[]));
    let hdc = g.next("hdc");
    ops.push(const_index(&hdc, isl.hdc));
    let gqac = g.next("gqac");
    ops.push(const_index(&gqac, isl.gqac));
    let qcol = g.next("qcol");
    ops.push(Operation::new(Some(&qcol), "arith.muli", &[&hpid, &hdc]));
    let kvh = g.next("kvh");
    ops.push(Operation::new(Some(&kvh), "arith.divui", &[&hpid, &gqac]));
    let kvcol = g.next("kvcol");
    ops.push(Operation::new(Some(&kvcol), "arith.muli", &[&kvh, &hdc]));

    // ---- views ----
    let q_view = g.next("qv");
    ops.push(mk_view(&q_view, &isl.q_arg, &[m, isl.q_cols], dt));
    let o_view = g.next("ov");
    ops.push(mk_view(&o_view, &isl.o_arg, &[m, isl.q_cols], dt));
    let mask_view = g.next("mv");
    ops.push(mk_view(&mask_view, &isl.mask_arg, &[1, cap], dt));
    let kc_view = g.next("kcv");
    ops.push(mk_view(&kc_view, &isl.kc_arg, &[cap, isl.kv_cols], dt));
    let kd_view = g.next("kdv");
    ops.push(mk_view(&kd_view, &isl.kd_arg, &[m, isl.kv_cols], dt));
    let vc_view = g.next("vcv");
    ops.push(mk_view(&vc_view, &isl.vc_arg, &[cap, isl.kv_cols], dt));
    let vd_view = g.next("vdv");
    ops.push(mk_view(&vd_view, &isl.vd_arg, &[m, isl.kv_cols], dt));

    // ---- whole-Q load [m, d] at [0, qcol] (per-head column slice) ----
    let q = block_load(&mut g, &mut ops, &q_view, &c0, &qcol, m, d);

    // ---- loop bounds + block-size constant ----
    let lb = g.next("lb");
    ops.push(const_index(&lb, 0));
    let step = g.next("st");
    ops.push(const_index(&step, 1));
    let blk_c = g.next("blk");
    ops.push(const_index(&blk_c, blk));

    // ---- RUNTIME loop bound: iterate ONLY the KV blocks that hold valid context.
    // The context mask [1, cap] is 0 on valid columns and -inf past valid_len, so
    // exp(mask) is a 1/0 valid-column indicator. Sum it (WIDENED to f32 — an f16
    // sum saturates integer precision past 2048 and would undercount valid_len at a
    // block boundary, silently dropping context) to recover valid_len, then run
    // ceil(valid_len / blk) blocks. valid_len = 0 (empty prefix) => 0 blocks: the
    // diagonal alone carries the result. This makes attention O(actual context)
    // instead of O(cap) — a 32-token prefill chunk runs 1 block, not cap/blk — which
    // is the whole point of the rewrite for the long-context/small-query regime.
    let mask_full = mk_whole_load(&mut g, &mut ops, &mask_view, &[1, cap]);
    let vind = g.next("vind");
    ops.push(Operation::new(Some(&vind), "math.exp", &[&mask_full]));
    let vind32 = g.next("vind32");
    ops.push(Operation::new(Some(&vind32), "arith.convertf", &[&vind]));
    let vzero = g.next("vzero");
    ops.push(const_f(&vzero, 0.0));
    let vinit = g.next("vinit");
    ops.push(mk_splat(&vinit, &vzero, &[1], "f32"));
    let vsum = g.next("vsum");
    ops.push(mk_reduce(&vsum, &vind32, &vinit, "arith.addf"));
    let vscalar = g.next("vsc");
    ops.push(Operation::new(Some(&vscalar), "tensor.extract", &[&vsum]));
    let vidx = g.next("vidx");
    ops.push(Operation::new(Some(&vidx), "arith.index_cast", &[&vscalar]));
    let ub = g.next("ub");
    ops.push(Operation::new(
        Some(&ub),
        "arith.ceildivui",
        &[&vidx, &blk_c],
    ));

    // ---- iter-arg inits: running max [m,1]=FLOOR, sum [m,1]=0, acc [m,d]=empty ----
    // The running max seeds a FINITE floor, NOT -inf: a fresh prefill's prefix
    // CONTEXT is empty, so every context KV block is fully mask-additive `-inf`.
    // With a -inf seed, `m_new = max(-inf,-inf) = -inf` and `exp(Sj - m_new) =
    // exp(-inf - -inf) = NaN`. A finite floor (well below any real scaled score,
    // within f16 range) makes a fully-masked block yield `exp(-inf - floor) = 0`
    // (contributes nothing, as it must), while any valid block's real max exceeds
    // the floor so its arithmetic is unchanged. The whole-tensor path never hit
    // this because its reduce_max spans the valid diagonal (a finite global max).
    let mfloor_c = g.next("mfloor");
    ops.push(const_f(&mfloor_c, -3.0e4));
    let m0 = g.next("m0");
    ops.push(mk_splat(&m0, &mfloor_c, &[m, 1], dt));
    let l0 = g.next("l0");
    ops.push(mk_splat(&l0, &zero_c, &[m, 1], dt));
    let acc0 = g.next("acc0");
    ops.push(mk_empty(&acc0, &[m, d], dt));

    // iter-arg body-visible names.
    let mi = "%fa_mi".to_string();
    let li = "%fa_li".to_string();
    let acci = "%fa_acci".to_string();
    let iv = "%fa_j".to_string();

    // ===================== CONTEXT KV-block loop body =====================
    let mut body: Vec<Operation> = Vec::new();
    // off = j * blk (the KV-block row offset into the [cap, kv_cols] view).
    let off = g.next("off");
    body.push(Operation::new(Some(&off), "arith.muli", &[&iv, &blk_c]));

    // Kj = Kc[off.., kvcol] -> [blk, d]; Vj = Vc[off.., kvcol] -> [blk, d].
    let kj = block_load(&mut g, &mut body, &kc_view, &off, &kvcol, blk, d);
    let vj = block_load(&mut g, &mut body, &vc_view, &off, &kvcol, blk, d);

    // Kjt = transpose(Kj) -> [d, blk]; Sj_raw = Q @ Kjt -> [m, blk].
    let kjt = mk_transpose(&mut g, &mut body, &kj, blk, d, dt);
    let sj_raw = mk_matmul(&mut g, &mut body, &q, &kjt, m, blk, dt);
    // scaled = Sj_raw * scale.
    let sjscl = g.next("sjscl");
    body.push(mk_splat(&sjscl, &scale_c, &[m, blk], dt));
    let sj_scaled = g.next("sjscaled");
    body.push(Operation::new(
        Some(&sj_scaled),
        "arith.mulf",
        &[&sj_raw, &sjscl],
    ));
    // maskj = Mask[0, off..] -> [1, blk], broadcast to [m, blk] (NOT a triangle).
    let maskj = block_load(&mut g, &mut body, &mask_view, &c0, &off, 1, blk);
    let maskj_init = g.next("mji");
    body.push(mk_empty(&maskj_init, &[m, blk], dt));
    let maskj_b = g.next("mjb");
    body.push(
        Operation::new(Some(&maskj_b), "linalg.broadcast", &[&maskj, &maskj_init])
            .with_attr("dimensions", Attr::IntList(vec![])),
    );
    let sj = g.next("sj");
    body.push(Operation::new(
        Some(&sj),
        "arith.addf",
        &[&sj_scaled, &maskj_b],
    ));

    // rmax = reduce_max(Sj, 1) -> [m] -> [m,1].
    let rmax_init = g.next("rmi");
    body.push(mk_splat(&rmax_init, &ninf_c, &[m], dt));
    let rmax = g.next("rmax");
    body.push(mk_reduce(&rmax, &sj, &rmax_init, "arith.maximumf"));
    let rmax2 = g.next("rmax2");
    body.push(reshape_to(&rmax2, &rmax, &[m, 1]));
    // m_new = max(m_i, rmax2) -> [m,1].
    let mnew = g.next("mnew");
    body.push(Operation::new(
        Some(&mnew),
        "arith.maximumf",
        &[&mi, &rmax2],
    ));

    // P = exp(Sj - m_new_bc[m,blk]).
    let mnew_b = broadcast_col_to(&mut g, &mut body, &mnew, m, blk, dt);
    let shifted = g.next("shift");
    body.push(Operation::new(
        Some(&shifted),
        "arith.subf",
        &[&sj, &mnew_b],
    ));
    let p = g.next("p");
    body.push(Operation::new(Some(&p), "math.exp", &[&shifted]));

    // alpha = exp(m_i - m_new) -> [m,1].
    let mdiff = g.next("mdiff");
    body.push(Operation::new(Some(&mdiff), "arith.subf", &[&mi, &mnew]));
    let alpha = g.next("alpha");
    body.push(Operation::new(Some(&alpha), "math.exp", &[&mdiff]));

    // rsum = reduce_sum(P, 1) -> [m] -> [m,1]; l_new = alpha*l_i + rsum.
    let rsum_init = g.next("rsi");
    body.push(mk_splat(&rsum_init, &zero_c, &[m], dt));
    let rsum = g.next("rsum");
    body.push(mk_reduce(&rsum, &p, &rsum_init, "arith.addf"));
    let rsum2 = g.next("rsum2");
    body.push(reshape_to(&rsum2, &rsum, &[m, 1]));
    let al = g.next("al");
    body.push(Operation::new(Some(&al), "arith.mulf", &[&alpha, &li]));
    let lnew = g.next("lnew");
    body.push(Operation::new(Some(&lnew), "arith.addf", &[&al, &rsum2]));

    // acc_new = alpha_bc[m,d]*acc + P @ Vj.
    let alpha_b = broadcast_col_to(&mut g, &mut body, &alpha, m, d, dt);
    let acc_scaled = g.next("accs");
    body.push(Operation::new(
        Some(&acc_scaled),
        "arith.mulf",
        &[&alpha_b, &acci],
    ));
    let pv = mk_matmul(&mut g, &mut body, &p, &vj, m, d, dt);
    let accnew = g.next("accnew");
    body.push(Operation::new(
        Some(&accnew),
        "arith.addf",
        &[&acc_scaled, &pv],
    ));

    body.push(Operation::new(None, "scf.yield", &[&mnew, &lnew, &accnew]));

    // ---- the scf.for over CONTEXT KV blocks ----
    let mc_f = g.next("mcf"); // running max [m,1] (un-normalized partial base).
    let lc_f = g.next("lcf"); // running sum [m,1].
    let acc_f = g.next("accf"); // running acc [m,d] (un-normalized, base mc_f).
    let mut forop = Operation::new(None, "scf.for", &[&lb, &ub, &step, &m0, &l0, &acc0])
        .with_attr("iter_var", Attr::Str(iv.clone()))
        .with_attr(
            "iter_args",
            Attr::StrList(vec![mi.clone(), li.clone(), acci.clone()]),
        )
        .with_attr(
            "result_names",
            Attr::StrList(vec![mc_f.clone(), lc_f.clone(), acc_f.clone()]),
        );
    forop.regions = vec![body];
    ops.push(forop);

    // mc_f is [m,1]; flatten to [m] for the diagonal-combine arith below.
    let mc_row = g.next("mcrow");
    ops.push(reshape_to(&mc_row, &mc_f, &[m]));
    let lc_row = g.next("lcrow");
    ops.push(reshape_to(&lc_row, &lc_f, &[m]));

    // ===================== DIAGONAL block (whole) =========================
    // Kd [m, d] at [0, kvcol] -> Kdt [d, m]; Sd = (Q @ Kdt)*scale + tri[m,m].
    let kd = block_load(&mut g, &mut ops, &kd_view, &c0, &kvcol, m, d);
    let kdt = mk_transpose(&mut g, &mut ops, &kd, m, d, dt);
    let sd_raw = mk_matmul(&mut g, &mut ops, &q, &kdt, m, m, dt);
    let sd_scl = g.next("sdscl");
    ops.push(mk_splat(&sd_scl, &scale_c, &[m, m], dt));
    let sd_scaled = g.next("sdscaled");
    ops.push(Operation::new(
        Some(&sd_scaled),
        "arith.mulf",
        &[&sd_raw, &sd_scl],
    ));
    let tri = g.next("tri");
    ops.push(causal_mask_mm(&tri, m, isl.ninf, dt));
    let sd = g.next("sd");
    ops.push(Operation::new(Some(&sd), "arith.addf", &[&sd_scaled, &tri]));
    // md = reduce_max(Sd, 1) -> [m].
    let md_init = g.next("mdi");
    ops.push(mk_splat(&md_init, &ninf_c, &[m], dt));
    let md = g.next("md");
    ops.push(mk_reduce(&md, &sd, &md_init, "arith.maximumf"));

    // ===================== COMBINE (global re-association) =================
    // gm = max(mc_f, md) [m].
    let gm = g.next("gm");
    ops.push(Operation::new(Some(&gm), "arith.maximumf", &[&mc_row, &md]));
    // cfac = exp(mc_f - gm) [m]  (re-base the CONTEXT partial onto the global max).
    let cdiff = g.next("cdiff");
    ops.push(Operation::new(Some(&cdiff), "arith.subf", &[&mc_row, &gm]));
    let cfac = g.next("cfac");
    ops.push(Operation::new(Some(&cfac), "math.exp", &[&cdiff]));
    // accC' = cfac_bc[m,d] * acc_f; lC' = cfac * lc_f.
    let cfac_bd = broadcast_row_to(&mut g, &mut ops, &cfac, m, d, dt);
    let acc_rb = g.next("accrb");
    ops.push(Operation::new(
        Some(&acc_rb),
        "arith.mulf",
        &[&cfac_bd, &acc_f],
    ));
    let lc_rb = g.next("lcrb");
    ops.push(Operation::new(
        Some(&lc_rb),
        "arith.mulf",
        &[&cfac, &lc_row],
    ));
    // Pd = exp(Sd - gm_bc[m,m]); sd_sum = reduce_sum(Pd,1) [m].
    let gm_bd = broadcast_row_to(&mut g, &mut ops, &gm, m, m, dt);
    let shd = g.next("shd");
    ops.push(Operation::new(Some(&shd), "arith.subf", &[&sd, &gm_bd]));
    let pd = g.next("pd");
    ops.push(Operation::new(Some(&pd), "math.exp", &[&shd]));
    let sds_init = g.next("sdsi");
    ops.push(mk_splat(&sds_init, &zero_c, &[m], dt));
    let sds = g.next("sds");
    ops.push(mk_reduce(&sds, &pd, &sds_init, "arith.addf"));
    // gs = lC' + sd_sum [m].
    let gs = g.next("gs");
    ops.push(Operation::new(Some(&gs), "arith.addf", &[&lc_rb, &sds]));
    // Wd = Pd / gs_bc[m,m].
    let gs_bd = broadcast_row_to(&mut g, &mut ops, &gs, m, m, dt);
    let wd = g.next("wd");
    ops.push(Operation::new(Some(&wd), "arith.divf", &[&pd, &gs_bd]));

    // ovd = Wd @ Vd; Vd [m, d] at [0, kvcol].
    let vd = block_load(&mut g, &mut ops, &vd_view, &c0, &kvcol, m, d);
    let ovd = mk_matmul(&mut g, &mut ops, &wd, &vd, m, d, dt);
    // O = (accC' + ovd) / gs_bc[m,d].
    let num = g.next("num");
    ops.push(Operation::new(Some(&num), "arith.addf", &[&acc_rb, &ovd]));
    let gs_bcd = broadcast_row_to(&mut g, &mut ops, &gs, m, d, dt);
    let o = g.next("o");
    ops.push(Operation::new(Some(&o), "arith.divf", &[&num, &gs_bcd]));

    // store O [m, d] at [0, qcol].
    let o_at = g.next("oat");
    ops.push(
        Operation::new(
            Some(&o_at),
            "ktdp.construct_access_tile",
            &[&o_view, &c0, &qcol],
        )
        .with_attr("shape", Attr::IntList(vec![m, d])),
    );
    ops.push(Operation::new(None, "ktdp.store", &[&o, &o_at]));
    ops.push(Operation::new(None, "func.return", &[]));

    IRFunction {
        name: String::new(), // caller stamps the original name
        arguments: vec![
            (isl.q_arg.clone(), "index".into()),
            (isl.o_arg.clone(), "index".into()),
            (isl.mask_arg.clone(), "index".into()),
            (isl.kc_arg.clone(), "index".into()),
            (isl.kd_arg.clone(), "index".into()),
            (isl.vc_arg.clone(), "index".into()),
            (isl.vd_arg.clone(), "index".into()),
        ],
        operations: ops,
        grid: (isl.h as usize, 1, 1),
        return_type: None,
    }
}

// ---- small emit helpers shared by `tile_rerolled_attention` (mirroring the
//      head_rewrite emitters so the diagonal block is byte-identical) ----

/// `linalg.transpose ins(%x) outs(empty[cols,rows]) permutation=[1,0]`.
fn mk_transpose(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    x: &str,
    rows: i64,
    cols: i64,
    dt: &str,
) -> String {
    let init = g.next("tpi");
    ops.push(mk_empty(&init, &[cols, rows], dt));
    let res = g.next("tp");
    ops.push(
        Operation::new(Some(&res), "linalg.transpose", &[x, &init])
            .with_attr("permutation", Attr::IntList(vec![1, 0])),
    );
    res
}

/// `C = A @ B` with a zero `tensor.empty` outs init.
fn mk_matmul(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    a: &str,
    b: &str,
    rows: i64,
    cols: i64,
    dt: &str,
) -> String {
    let init = g.next("mmi");
    ops.push(mk_empty(&init, &[rows, cols], dt));
    let res = g.next("mm");
    ops.push(Operation::new(Some(&res), "linalg.matmul", &[a, b, &init]));
    res
}

/// Broadcast a `[m]` row-vector to `[m, cols]` (reshape `[m]`→`[m,1]` then
/// `linalg.broadcast` to the outs shape) — the head_rewrite combine convention.
fn broadcast_row_to(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    rowv: &str,
    m: i64,
    cols: i64,
    dt: &str,
) -> String {
    let r2 = g.next("rs");
    ops.push(reshape_to(&r2, rowv, &[m, 1]));
    let init = g.next("bri");
    ops.push(mk_empty(&init, &[m, cols], dt));
    let res = g.next("br");
    ops.push(
        Operation::new(Some(&res), "linalg.broadcast", &[&r2, &init])
            .with_attr("dimensions", Attr::IntList(vec![])),
    );
    res
}

/// The static `[m, m]` lower-triangular causal mask: `0` for `k ≤ r`, `ninf` for
/// `k > r` (the diagonal block's mask, kept whole — byte-identical to
/// `head_rewrite::causal_mask_mm`).
fn causal_mask_mm(res: &str, m: i64, ninf: f32, dt: &str) -> Operation {
    let mut vals = Vec::with_capacity((m * m) as usize);
    for r in 0..m {
        for k in 0..m {
            vals.push(if k <= r { 0.0 } else { ninf as f64 });
        }
    }
    Operation::new(Some(res), "arith.constant", &[])
        .with_attr("is_tensor", Attr::Bool(true))
        .with_attr("dense_list", Attr::Bool(true))
        .with_attr("shape", Attr::IntList(vec![m, m]))
        .with_attr("dtype", Attr::Str(dt.into()))
        .with_attr("value", Attr::FloatList(vals))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The canonical naive-attention builder reused by the recognizer unit tests
    // (the EXECUTION-equivalence golden lives in the ktir-cpu test crate, where
    // the interpreter is available).
    pub fn naive_attention(m: i64, cap: i64, d: i64, scale: f32, causal: bool) -> IRFunction {
        super::test_support::naive_attention(m, cap, d, scale, causal)
    }

    #[test]
    fn recognizes_canonical_attention() {
        let f = naive_attention(4, 8, 2, 0.5, true);
        let isl = recognize_attention(&f).expect("should recognize canonical attention");
        assert_eq!(isl.m, 4);
        assert_eq!(isl.cap, 8);
        assert_eq!(isl.d, 2);
        assert!((isl.scale - 0.5).abs() < 1e-6);
        assert!(isl.causal);
        assert_eq!(isl.q_arg, "%q_ptr");
        assert_eq!(isl.k_arg, "%k_ptr");
        assert_eq!(isl.v_arg, "%v_ptr");
        assert_eq!(isl.o_arg, "%o_ptr");
    }

    #[test]
    fn recognizes_noncausal() {
        let f = naive_attention(2, 4, 2, 0.25, false);
        let isl = recognize_attention(&f).expect("non-causal still recognized");
        assert!(!isl.causal);
        assert_eq!(isl.scores_bytes(), 2 * 4 * 2); // [2,4] f16
    }

    #[test]
    fn rejects_non_attention() {
        // A plain copy node: load -> exp -> store. No QKᵀ/softmax/AV.
        let f = IRFunction {
            name: "copy".into(),
            arguments: vec![
                ("%in".into(), "index".into()),
                ("%out".into(), "index".into()),
            ],
            grid: (1, 1, 1),
            return_type: None,
            operations: vec![
                mk_view("%vi", "%in", &[4, 4], "f16"),
                Operation::new(Some("%ti"), "ktdp.construct_access_tile", &["%vi"])
                    .with_attr("shape", Attr::IntList(vec![4, 4])),
                Operation::new(Some("%l"), "ktdp.load", &["%ti"]),
                Operation::new(Some("%y"), "math.exp", &["%l"]),
                mk_view("%vo", "%out", &[4, 4], "f16"),
                Operation::new(Some("%to"), "ktdp.construct_access_tile", &["%vo"])
                    .with_attr("shape", Attr::IntList(vec![4, 4])),
                Operation::new(None, "ktdp.store", &["%y", "%to"]),
                Operation::new(None, "func.return", &[]),
            ],
        };
        assert!(
            recognize_attention(&f).is_none(),
            "copy node must not be recognized"
        );
    }

    #[test]
    fn rejects_region_bearing() {
        // A function that already contains an scf.for is not the flat idiom.
        let mut f = naive_attention(2, 4, 2, 0.5, false);
        let mut forop = Operation::new(None, "scf.for", &["%x", "%y", "%z"]);
        forop.regions = vec![vec![Operation::new(None, "scf.yield", &[])]];
        f.operations.insert(0, forop);
        assert!(
            recognize_attention(&f).is_none(),
            "region-bearing func bails"
        );
    }

    #[test]
    fn rejects_multi_store() {
        // Two stores (the real unrolled per-query-row lowering) -> not canonical.
        let mut f = naive_attention(2, 4, 2, 0.5, false);
        // duplicate the store op.
        let store = f
            .operations
            .iter()
            .find(|o| o.op_type == "ktdp.store")
            .cloned()
            .unwrap();
        let idx = f.operations.len() - 1; // before func.return
        f.operations.insert(idx, store);
        assert!(recognize_attention(&f).is_none(), "multi-store bails");
    }

    #[test]
    fn tile_emits_scf_for_and_no_insert_slice() {
        let f = naive_attention(4, 256, 8, 0.125, true);
        let isl = recognize_attention(&f).unwrap();
        let tiled = tile_attention(&isl);

        // Exactly one scf.for at top level, carrying 3 iter-args (m, l, acc).
        let fors: Vec<&Operation> = tiled
            .operations
            .iter()
            .filter(|o| o.op_type == "scf.for")
            .collect();
        assert_eq!(fors.len(), 1, "one KV loop");
        let f0 = fors[0];
        match f0.attributes.get("iter_args") {
            Some(Attr::StrList(v)) => assert_eq!(v.len(), 3, "m, l, acc iter-args"),
            other => panic!("iter_args not a 3-list: {other:?}"),
        }
        match f0.attributes.get("result_names") {
            Some(Attr::StrList(v)) => assert_eq!(v.len(), 3),
            other => panic!("result_names not a 3-list: {other:?}"),
        }

        // NO tensor.insert_slice anywhere (it is UNREGISTERED in this emulator).
        fn has_insert(ops: &[Operation]) -> bool {
            ops.iter().any(|o| {
                o.op_type == "tensor.insert_slice" || o.regions.iter().any(|r| has_insert(r))
            })
        }
        assert!(!has_insert(&tiled.operations), "must not emit insert_slice");

        // The loop body reads each KV block via a `ktdp.construct_access_tile` at
        // the dynamic block offset + a `ktdp.load` (Kj and Vj) — the KTIR-native,
        // fusion-safe analogue of an extract_slice of the block.
        let body = &f0.regions[0];
        let block_tiles = body
            .iter()
            .filter(|o| o.op_type == "ktdp.construct_access_tile")
            .count();
        let block_loads = body.iter().filter(|o| o.op_type == "ktdp.load").count();
        assert_eq!(block_tiles, 2, "Kj and Vj access tiles per block");
        assert_eq!(block_loads, 2, "Kj and Vj loads per block");
        // No tensor.extract_slice in the body either (we use ktdp block loads).
        let slices = body
            .iter()
            .filter(|o| o.op_type == "tensor.extract_slice")
            .count();
        assert_eq!(
            slices, 0,
            "KV blocks read via ktdp access tile, not extract_slice"
        );

        // online-softmax kernels present: two matmuls (QKᵀ and P·V), an exp for P
        // and an exp for alpha.
        let matmuls = body.iter().filter(|o| o.op_type == "linalg.matmul").count();
        assert_eq!(matmuls, 2, "QKᵀ and P·V");
        let exps = body.iter().filter(|o| o.op_type == "math.exp").count();
        assert_eq!(exps, 2, "exp(P) and exp(alpha)");
    }

    #[test]
    fn tile_preserves_args_and_grid() {
        let f = naive_attention(2, 8, 4, 0.5, false);
        let isl = recognize_attention(&f).unwrap();
        let tiled = tile_attention(&isl);
        let names: Vec<&str> = tiled.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["%q_ptr", "%k_ptr", "%v_ptr", "%o_ptr"]);
        assert_eq!(
            tiled.grid,
            (1, 1, 1),
            "rewritten node runs single-grid generic"
        );
    }

    #[test]
    fn choose_block_divides_cap() {
        for &cap in &[8, 16, 64, 128, 256, 512, 1024, 2048, 4096] {
            let b = choose_block(cap);
            assert!(b >= 1 && b <= cap);
            assert_eq!(cap % b, 0, "block {b} must divide cap {cap}");
            assert!(b <= DEFAULT_KV_BLOCK || cap <= DEFAULT_KV_BLOCK);
        }
    }

    // ----- RE-ROLLED path (the REAL model node, post head_rewrite) -----

    use crate::head_rewrite::{HeadAttnIsland, rewrite_head_attention};

    /// A synthetic head island matching the smollm2 shape (H=9, m=8, gqac=3, d=64,
    /// cap=64). `rewrite_head_attention` turns it into the EXACT re-rolled IR the
    /// real node produces, which `recognize_rerolled_attention` must match.
    fn smollm_head_island(cap: i64) -> HeadAttnIsland {
        HeadAttnIsland {
            q_arg: "%q".into(),
            o_arg: "%o".into(),
            mask_arg: "%mask".into(),
            kc_arg: "%kc".into(),
            kd_arg: "%kd".into(),
            vc_arg: "%vc".into(),
            vd_arg: "%vd".into(),
            q_cols: 576,
            kv_cols: 192,
            m: 8,
            cap,
            d: 64,
            gqac: 3,
            hdc: 64,
            h: 9,
            scale: 0.125,
            ninf: -1.0e38,
            dtype: "f16".into(),
        }
    }

    #[test]
    fn recognizes_rerolled_head_output() {
        // The re-rolled output of a head island IS the structural idiom the new
        // recognizer must match (this is exactly what head_rewrite emits for the
        // real node111). All fields must round-trip.
        let head = smollm_head_island(64);
        let mut rerolled = rewrite_head_attention(&head);
        rerolled.name = "attn".into();
        let isl = recognize_rerolled_attention(&rerolled)
            .expect("re-rolled head output must be recognized");
        assert_eq!(isl.m, head.m);
        assert_eq!(isl.cap, head.cap);
        assert_eq!(isl.d, head.d);
        assert_eq!(isl.h, head.h);
        assert_eq!(isl.gqac, head.gqac);
        assert_eq!(isl.q_cols, head.q_cols);
        assert_eq!(isl.kv_cols, head.kv_cols);
        assert!((isl.scale - head.scale).abs() < 1e-6);
        assert_eq!(isl.q_arg, "%q");
        assert_eq!(isl.kc_arg, "%kc");
        assert_eq!(isl.kd_arg, "%kd");
        // scores_bytes must EQUAL HeadAttnIsland::scores_bytes (disjoint partition).
        assert_eq!(isl.scores_bytes(), head.scores_bytes());
        assert_eq!(isl.scores_bytes(), 8 * 64 * 2);
    }

    #[test]
    fn rerolled_recognizer_rejects_single_core() {
        let head = smollm_head_island(64);
        let mut rerolled = rewrite_head_attention(&head);
        rerolled.grid = (1, 1, 1); // not head-parallel
        assert!(recognize_rerolled_attention(&rerolled).is_none());
    }

    #[test]
    fn rerolled_recognizer_rejects_region_bearing() {
        // A body that already contains an scf.for is the already-tiled form.
        let head = smollm_head_island(64);
        let mut rerolled = rewrite_head_attention(&head);
        let mut forop = Operation::new(None, "scf.for", &["%x", "%y", "%z"]);
        forop.regions = vec![vec![Operation::new(None, "scf.yield", &[])]];
        rerolled.operations.insert(0, forop);
        assert!(recognize_rerolled_attention(&rerolled).is_none());
    }

    #[test]
    fn rerolled_recognizer_rejects_single_block_naive() {
        // The single-block canonical idiom is NOT the two-block re-rolled form.
        let naive = test_support::naive_attention(4, 8, 2, 0.5, true);
        assert!(recognize_rerolled_attention(&naive).is_none());
    }

    #[test]
    fn tile_rerolled_emits_one_scf_for_no_insert_slice() {
        let head = smollm_head_island(256); // cap=256 so real tiling happens
        let mut rerolled = rewrite_head_attention(&head);
        rerolled.name = "attn".into();
        let isl = recognize_rerolled_attention(&rerolled).unwrap();
        let blk = choose_block_budgeted(isl.m, isl.cap, isl.d, 2, &|sb| sb >= 1024);
        let tiled = tile_rerolled_attention(&isl, blk);

        // Grid + args preserved (7 args, [H,1,1]).
        assert_eq!(tiled.grid, (9, 1, 1));
        let names: Vec<&str> = tiled.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["%q", "%o", "%mask", "%kc", "%kd", "%vc", "%vd"]);

        // Exactly one scf.for, 3 iter-args (mC, lC, accC).
        let fors: Vec<&Operation> = tiled
            .operations
            .iter()
            .filter(|o| o.op_type == "scf.for")
            .collect();
        assert_eq!(fors.len(), 1, "one CONTEXT KV loop");
        match fors[0].attributes.get("iter_args") {
            Some(Attr::StrList(v)) => assert_eq!(v.len(), 3),
            other => panic!("iter_args not a 3-list: {other:?}"),
        }

        // NO tensor.insert_slice / extract_slice anywhere.
        fn has_bad(ops: &[Operation]) -> bool {
            ops.iter().any(|o| {
                matches!(
                    o.op_type.as_str(),
                    "tensor.insert_slice" | "tensor.extract_slice"
                ) || o.regions.iter().any(|r| has_bad(r))
            })
        }
        assert!(
            !has_bad(&tiled.operations),
            "must not emit insert/extract_slice"
        );

        // Per-head selection arithmetic preserved.
        assert!(
            tiled
                .operations
                .iter()
                .any(|o| o.op_type == "ktdp.get_compute_tile_id")
        );
        assert!(tiled.operations.iter().any(|o| o.op_type == "arith.divui"));
    }

    #[test]
    fn rerolled_per_block_tile_fits_budget_and_shrinks() {
        // The actual long-context fix: the per-block CONTEXT scores tile [m, blk]
        // is BELOW the forced budget AND strictly smaller than the full [m, cap]
        // tile — proven for a long cap (cap=512).
        let head = smollm_head_island(512);
        let mut rerolled = rewrite_head_attention(&head);
        rerolled.name = "attn".into();
        let isl = recognize_rerolled_attention(&rerolled).unwrap();
        let bytes = 2usize;
        let budget = 4096usize; // full tile 8*512*2=8192 overflows; sub-blocks fit.
        let needs = |sb: usize| sb.saturating_mul(8) >= budget.saturating_mul(7);
        assert!(
            needs(isl.scores_bytes()),
            "full tile must overflow at this budget"
        );
        let blk = choose_block_budgeted(isl.m, isl.cap, isl.d, bytes, &needs);
        assert!(blk < isl.cap, "must tile: blk {blk} < cap {}", isl.cap);
        assert_eq!(isl.cap % blk, 0, "blk divides cap");
        let per_block = (isl.m as usize) * (blk as usize) * bytes;
        assert!(
            !needs(per_block),
            "per-block tile {per_block} must fit budget"
        );
        assert!(per_block < isl.scores_bytes(), "per-block < full");

        // Walk the emitted scf.for body: every 2-D static-shape tile op is [m, blk]
        // or smaller in the cap axis (never the full [m, cap]).
        let tiled = tile_rerolled_attention(&isl, blk);
        let forop = tiled
            .operations
            .iter()
            .find(|o| o.op_type == "scf.for")
            .unwrap();
        for op in &forop.regions[0] {
            if let Some(Attr::IntList(s)) = op.attributes.get("shape")
                && s.len() == 2
                && s[0] == isl.m
            {
                // any [m, c] tile in the loop must have c <= blk (never == cap).
                assert!(
                    s[1] <= blk,
                    "loop tile [{},{}] exceeds blk {blk}",
                    s[0],
                    s[1]
                );
            }
        }
    }
}

// Shared synthetic-IR builder used by both the in-crate unit tests above and the
// ktir-cpu execution-equivalence golden (which re-declares the same structure).
#[doc(hidden)]
pub mod test_support {
    use super::*;

    /// Build a canonical NAIVE attention `IRFunction` over args
    /// `%q_ptr, %k_ptr, %v_ptr, %o_ptr` with Q[m,d], K[cap,d], V[cap,d], O[m,d].
    /// This is the EXACT idiom `recognize_attention` matches; the golden runs it
    /// on the interpreter and compares against the tiled rewrite.
    pub fn naive_attention(m: i64, cap: i64, d: i64, scale: f32, causal: bool) -> IRFunction {
        let dt = "f16";
        let mut g = NameGen::new("nv");
        let mut ops: Vec<Operation> = Vec::new();

        // load Q, K, V whole.
        let qv = g.next("qv");
        ops.push(mk_view(&qv, "%q_ptr", &[m, d], dt));
        let q = mk_whole_load(&mut g, &mut ops, &qv, &[m, d]);
        let kv = g.next("kv");
        ops.push(mk_view(&kv, "%k_ptr", &[cap, d], dt));
        let k = mk_whole_load(&mut g, &mut ops, &kv, &[cap, d]);
        let vv = g.next("vv");
        ops.push(mk_view(&vv, "%v_ptr", &[cap, d], dt));
        let v = mk_whole_load(&mut g, &mut ops, &vv, &[cap, d]);

        // Kt = transpose(K) -> [d, cap]
        let kti = g.next("kti");
        ops.push(mk_empty(&kti, &[d, cap], dt));
        let kt = g.next("kt");
        ops.push(
            Operation::new(Some(&kt), "linalg.transpose", &[&k, &kti])
                .with_attr("permutation", Attr::IntList(vec![1, 0])),
        );
        // raw = Q @ Kt -> [m, cap]
        let rawi = g.next("rawi");
        ops.push(mk_empty(&rawi, &[m, cap], dt));
        let raw = g.next("raw");
        ops.push(Operation::new(
            Some(&raw),
            "linalg.matmul",
            &[&q, &kt, &rawi],
        ));
        // scaled = raw * scale
        let sc = g.next("sc");
        ops.push(const_f(&sc, scale as f64));
        let sct = g.next("sct");
        ops.push(mk_splat(&sct, &sc, &[m, cap], dt));
        let scaled = g.next("scaled");
        ops.push(Operation::new(Some(&scaled), "arith.mulf", &[&raw, &sct]));

        // sm = scaled (+ causal mask)
        let sm = if causal {
            let mask = g.next("mask");
            // full [m,cap] causal mask via tensor.generate.
            ops.push(causal_mask_full(&mask, m, cap, dt));
            let masked = g.next("smm");
            ops.push(Operation::new(
                Some(&masked),
                "arith.addf",
                &[&scaled, &mask],
            ));
            masked
        } else {
            scaled
        };

        // mx = reduce_max(sm, 1) -> [m]
        let ninf = g.next("ninf");
        ops.push(const_f(&ninf, -1.0e30));
        let mxi = g.next("mxi");
        ops.push(mk_splat(&mxi, &ninf, &[m], dt));
        let mx = g.next("mx");
        ops.push(mk_reduce(&mx, &sm, &mxi, "arith.maximumf"));
        // mx_b broadcast to [m,cap]
        let mxr = g.next("mxr");
        ops.push(reshape_to(&mxr, &mx, &[m, 1]));
        let mxb = broadcast_col_to(&mut g, &mut ops, &mxr, m, cap, dt);
        // shifted = sm - mx_b
        let sh = g.next("sh");
        ops.push(Operation::new(Some(&sh), "arith.subf", &[&sm, &mxb]));
        // P = exp(shifted)
        let p = g.next("p");
        ops.push(Operation::new(Some(&p), "math.exp", &[&sh]));
        // l = reduce_sum(P, 1) -> [m]
        let zero = g.next("zero");
        ops.push(const_f(&zero, 0.0));
        let li = g.next("li");
        ops.push(mk_splat(&li, &zero, &[m], dt));
        let l = g.next("l");
        ops.push(mk_reduce(&l, &p, &li, "arith.addf"));
        let lr = g.next("lr");
        ops.push(reshape_to(&lr, &l, &[m, 1]));
        let lb = broadcast_col_to(&mut g, &mut ops, &lr, m, cap, dt);
        // W = P / l_b
        let w = g.next("w");
        ops.push(Operation::new(Some(&w), "arith.divf", &[&p, &lb]));

        // O = W @ V -> [m, d]
        let oi = g.next("oi");
        ops.push(mk_empty(&oi, &[m, d], dt));
        let o = g.next("o");
        ops.push(Operation::new(Some(&o), "linalg.matmul", &[&w, &v, &oi]));

        // store O
        let ov = g.next("ov");
        ops.push(mk_view(&ov, "%o_ptr", &[m, d], dt));
        let oat = g.next("oat");
        ops.push(
            Operation::new(Some(&oat), "ktdp.construct_access_tile", &[&ov])
                .with_attr("shape", Attr::IntList(vec![m, d])),
        );
        ops.push(Operation::new(None, "ktdp.store", &[&o, &oat]));
        ops.push(Operation::new(None, "func.return", &[]));

        IRFunction {
            name: "naive_attn".into(),
            arguments: vec![
                ("%q_ptr".into(), "index".into()),
                ("%k_ptr".into(), "index".into()),
                ("%v_ptr".into(), "index".into()),
                ("%o_ptr".into(), "index".into()),
            ],
            operations: ops,
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// A full `[m, cap]` causal mask: visible (`0`) where absolute key
    /// `<= cap - m + qr`, else `-inf`. The whole-matrix analogue of the per-block
    /// `causal_mask_tensor`.
    fn causal_mask_full(res: &str, m: i64, cap: i64, dt: &str) -> Operation {
        let bb0 = Operation::new(None, "region.bb0_args", &[])
            .with_attr("names", Attr::StrList(vec!["%qr".into(), "%kc".into()]));
        let base = Operation::new(Some("%fm_base"), "arith.constant", &[])
            .with_attr("value", Attr::Int(cap - m));
        let aq = Operation::new(Some("%fm_aq"), "arith.addi", &["%fm_base", "%qr"]);
        let cmp = Operation::new(Some("%fm_vis"), "arith.cmpi", &["%kc", "%fm_aq"])
            .with_attr("predicate", Attr::Str("sle".into()));
        let zero = Operation::new(Some("%fm_zero"), "arith.constant", &[])
            .with_attr("value", Attr::Float(0.0));
        let ninf = Operation::new(Some("%fm_ninf"), "arith.constant", &[])
            .with_attr("value", Attr::Float(-1.0e30));
        let sel = Operation::new(
            Some("%fm_v"),
            "arith.select",
            &["%fm_vis", "%fm_zero", "%fm_ninf"],
        );
        let yld = Operation::new(None, "tensor.yield", &["%fm_v"]);
        let mut gen_op = Operation::new(Some(res), "tensor.generate", &[])
            .with_attr("shape", Attr::IntList(vec![m, cap]))
            .with_attr("dtype", Attr::Str(dt.into()));
        gen_op.regions = vec![vec![bb0, base, aq, cmp, zero, ninf, sel, yld]];
        gen_op
    }
}
