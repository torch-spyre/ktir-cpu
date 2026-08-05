// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Tiled-elementwise COALESCE pass.
//!
//! Many `grid = [1, 1]` nodes emit `K >= 2` STRUCTURALLY-IDENTICAL blocks of
//! ops, each operating on a disjoint dim-0 tile of the same memory views — block
//! `j` differs from block `0` ONLY by a per-access-tile leading index offset that
//! is a consistent affine function of `j` (`indices_j = indices_0 + j * delta`,
//! `delta` constant across blocks). The canonical case is RoPE (rotary
//! embeddings): a `[1024, 64]` tensor processed in 32 blocks of `[32, *]`, block
//! `j` over rows `[32j, 32(j+1))`, with cos/sin `[32]` tables read at `[64j :
//! 64j+32]` (stride `2h`, NOT lockstep with the rows). Running them is dominated
//! by per-op interpreter dispatch (~1.3 µs/op): ~1500 ops for a `K=32` RoPE node.
//!
//! ## Coalescing by PREPENDING a leading `K` dimension
//!
//! Because each block's elementwise update is independent (no op reduces across
//! the block axis), running `K` structurally-identical blocks is arithmetically
//! identical to running ONE block with a leading axis of extent `K`. This pass
//! recognizes the idiom from STRUCTURAL invariants and rewrites the `K` blocks
//! into ONE, prepending a leading dim of size `K` to every value and dropping
//! blocks `1..K` — a ~Kx op reduction.
//!
//! The KEY capability over a naive "scale dim-0 by K" rewrite is that the leading
//! `K` axis carries a PER-VIEW element stride `S = dot(delta, view.strides)`.
//! That handles BOTH:
//!   * the contiguous row tiles (`delta = [h, 0]`, `S = h * row_stride`), and
//!   * the strided cos/sin tables (`delta = [2h]`, `S = 2h`), whose `[h]` tile at
//!     block `j` lands at flat element `2h*j` — exactly `cos[2h*j : 2h*j+h]`.
//!
//! Both become one access tile over the SAME view reshaped to rank `+1` with a
//! leading `(K, S)` (extent, stride) — a valid RFC-0682 affine box: the access
//! tile gains a leading index `0`, its `base_map`/`coordinate_set` gain an
//! identity leading dim, and the load/store reads the strided box directly.
//! A `linalg.broadcast` along the old row axis simply shifts its broadcast
//! `dimensions` by `+1`; its source gains the leading `K`.
//!
//! ## Why this is exact
//!
//! Stacking `K` independent elementwise blocks, where block `j`'s footprint for
//! every access tile is block `0`'s footprint translated by `j * delta` (in the
//! view's coordinate units), into ONE block with a leading axis of extent `K` and
//! per-view stride `S = dot(delta, view.strides)` is pure re-association: each
//! `(j, ...)` output element depends only on the `(j, ...)` input elements, i.e.
//! exactly block `j`'s computation. The emitted ops are exactly block `0`'s ops
//! with a leading `K` dim prepended — only RFC-0682 ops already present
//! (`ktdp` load/store/construct_*, `linalg.broadcast`, `arith.*`, `tensor.empty`).
//!
//! ## Fail-safe recognition (correctness over coverage)
//!
//! [`recognize_coalesce`] returns `None` unless the body is PROVABLY `K >= 2`
//! consecutive structurally-identical blocks differing solely by per-access-tile
//! leading index offsets `indices_j = indices_0 + j * delta` with `delta`
//! constant across blocks, where every access tile over a given memory view
//! shares ONE `delta` (so the view's leading stride `S` is well-defined). If ANY
//! access tile's offsets are not a consistent affine progression in `j`, or two
//! access tiles over the same view disagree on `delta`, or any required type /
//! affine rewrite cannot be applied, the function is left 100% unchanged. We
//! never rewrite a node we cannot prove equivalent.

use ktir_core::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
use ktir_core::ir::{Attr, IRFunction, IRModule, Operation};
use std::collections::HashMap;
use std::rc::Rc;

/// Apply the tile-coalesce pass to every function in `module`, in place.
/// Returns the number of functions rewritten.
pub fn apply_tile_coalesce(module: &mut IRModule) -> usize {
    let names: Vec<String> = module.functions.keys().cloned().collect();
    let mut rewritten = 0usize;
    for name in names {
        let Some(func) = module.functions.get(&name) else {
            continue;
        };
        let Some(new_ops) = recognize_coalesce(func) else {
            continue;
        };
        if let Some(f) = module.functions.get_mut(&name) {
            f.operations = new_ops;
            rewritten += 1;
        }
    }
    rewritten
}

// ===========================================================================
// Type-string helpers
// ===========================================================================

/// Parse the dim list out of a result-type string such as
/// `"tensor<32x32xf16>"`, `"tensor<32xf16>"`, or
/// `"!ktdp.access_tile<32x32xindex>"`. Returns `(prefix, dims, dtype, suffix)`
/// so the caller can edit the dims and re-render verbatim. `prefix` is everything
/// up to and including the opening `<`; `suffix` is from the closing `>` on.
fn split_typed(ty: &str) -> Option<(String, Vec<i64>, String, String)> {
    let open = ty.find('<')?;
    let close = ty.rfind('>')?;
    if close <= open {
        return None;
    }
    let prefix = ty[..=open].to_string();
    let suffix = ty[close..].to_string();
    let inner = &ty[open + 1..close];
    // inner is `D0xD1x...xDTYPE`. Consume leading `<int>x` groups as dims; the
    // remainder (which may itself contain 'x', e.g. the `index` dtype) is the
    // element type. We split only where an 'x' immediately follows a run of
    // ASCII digits AND precedes a dim or the dtype.
    let mut rest = inner;
    let mut dims: Vec<i64> = Vec::new();
    // Find the next 'x' such that the token before it is all digits.
    while let Some(xpos) = rest.find('x') {
        let (head, tail) = (&rest[..xpos], &rest[xpos + 1..]);
        if head.is_empty() || !head.bytes().all(|b| b.is_ascii_digit()) {
            break;
        }
        dims.push(head.parse::<i64>().ok()?);
        rest = tail;
    }
    if dims.is_empty() {
        return None;
    }
    let dtype = rest.to_string();
    Some((prefix, dims, dtype, suffix))
}

/// Re-render a typed string from an edited dim list.
fn render_typed(prefix: &str, dims: &[i64], dtype: &str, suffix: &str) -> String {
    let body: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!("{prefix}{}x{dtype}{suffix}", body.join("x"))
}

/// Prepend a leading dim of extent `k` to a shaped type string. Returns `None`
/// if the string is not a recognizable shaped type.
fn prepend_type_dim(ty: &str, k: i64) -> Option<String> {
    let (prefix, mut dims, dtype, suffix) = split_typed(ty)?;
    dims.insert(0, k);
    Some(render_typed(&prefix, &dims, &dtype, &suffix))
}

// ===========================================================================
// Affine helpers — prepend a leading dimension
// ===========================================================================

/// Shift every `Dim(i)` reference in an affine expression up by `1` (a new
/// leading dim was inserted at position 0). Symbols are untouched.
fn shift_dims(expr: &AffineExpr) -> AffineExpr {
    match expr {
        AffineExpr::Dim(i) => AffineExpr::Dim(i + 1),
        AffineExpr::Sym(i) => AffineExpr::Sym(*i),
        AffineExpr::Const(c) => AffineExpr::Const(*c),
        AffineExpr::Ref(s) => AffineExpr::Ref(s.clone()),
        AffineExpr::Add(a, b) => AffineExpr::Add(Rc::new(shift_dims(a)), Rc::new(shift_dims(b))),
        AffineExpr::Sub(a, b) => AffineExpr::Sub(Rc::new(shift_dims(a)), Rc::new(shift_dims(b))),
        AffineExpr::Neg(a) => AffineExpr::Neg(Rc::new(shift_dims(a))),
        AffineExpr::Mul(a, b) => AffineExpr::Mul(Rc::new(shift_dims(a)), Rc::new(shift_dims(b))),
        AffineExpr::FloorDiv(a, b) => {
            AffineExpr::FloorDiv(Rc::new(shift_dims(a)), Rc::new(shift_dims(b)))
        }
        AffineExpr::Mod(a, b) => AffineExpr::Mod(Rc::new(shift_dims(a)), Rc::new(shift_dims(b))),
        AffineExpr::Max(a, b) => AffineExpr::Max(Rc::new(shift_dims(a)), Rc::new(shift_dims(b))),
        AffineExpr::Min(a, b) => AffineExpr::Min(Rc::new(shift_dims(a)), Rc::new(shift_dims(b))),
    }
}

/// Prepend a leading identity result dim to an affine map: the new map has
/// `num_dims + 1` dims, its first result is `Dim(0)`, and every existing result
/// has its dim refs shifted up by one. Used for `base_map`.
fn prepend_map_dim(map: &AffineMap) -> AffineMap {
    let mut exprs = Vec::with_capacity(map.exprs.len() + 1);
    exprs.push(AffineExpr::Dim(0));
    for e in &map.exprs {
        exprs.push(shift_dims(e));
    }
    AffineMap {
        num_dims: map.num_dims + 1,
        num_syms: map.num_syms,
        exprs,
    }
}

/// Prepend a leading dim `0 <= d0 <= k-1` to an affine set: shift all existing
/// dim refs up by one and add the two box constraints for the new leading dim.
fn prepend_set_dim(set: &AffineSet, k: i64) -> AffineSet {
    let mut constraints: Vec<Constraint> = Vec::with_capacity(set.constraints.len() + 2);
    // d0 >= 0
    constraints.push(Constraint {
        expr: AffineExpr::Dim(0),
        kind: ConstraintKind::GreaterEq,
    });
    // -d0 + (k-1) >= 0
    constraints.push(Constraint {
        expr: AffineExpr::Add(
            Rc::new(AffineExpr::Neg(Rc::new(AffineExpr::Dim(0)))),
            Rc::new(AffineExpr::Const(k - 1)),
        ),
        kind: ConstraintKind::GreaterEq,
    });
    for c in &set.constraints {
        constraints.push(Constraint {
            expr: shift_dims(&c.expr),
            kind: c.kind,
        });
    }
    AffineSet {
        num_dims: set.num_dims + 1,
        num_syms: set.num_syms,
        constraints,
    }
}

// ===========================================================================
// Recognition
// ===========================================================================

/// Recognize the K-block tiled-elementwise idiom in `func` and return the
/// coalesced op list (block 0 with a leading `K` dim prepended, blocks `1..K`
/// dropped). Returns `None` on ANY structural deviation (fail-safe).
pub fn recognize_coalesce(func: &IRFunction) -> Option<Vec<Operation>> {
    // (1) grid must be [1, 1, 1] (single core).
    if func.grid != (1, 1, 1) {
        return None;
    }
    // (2) no control flow.
    if func
        .operations
        .iter()
        .any(|op| op.op_type.starts_with("scf.") || !op.regions.is_empty())
    {
        return None;
    }

    // Index constant table (resolve access-tile leading-index offsets).
    let mut int_const: HashMap<String, i64> = HashMap::new();
    for op in &func.operations {
        if op.op_type == "arith.constant"
            && let Some(res) = &op.result
            && let Some(Attr::Int(v)) = op.attributes.get("value")
        {
            int_const.insert(res.clone(), *v);
        }
    }

    let ops = &func.operations;
    // Strip a trailing func.return for block partitioning; keep to re-append.
    let has_return = ops
        .last()
        .map(|o| o.op_type == "func.return" || o.op_type == "return")
        .unwrap_or(false);
    let body_end = if has_return { ops.len() - 1 } else { ops.len() };

    // Partition the body into blocks. A block ENDS at the last `ktdp.store` of a
    // maximal "store cluster" — a run of ops that are only `ktdp.store` or the
    // `ktdp.construct_access_tile` feeding the next store (stores are emitted as
    // `access_tile; store; access_tile; store; ...`). The cluster must END on a
    // store; the block's exclusive end is just past that final store.
    let mut block_ends: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < body_end {
        if ops[i].op_type == "ktdp.store" {
            // Extend through interleaved (access_tile, store) pairs.
            let mut j = i;
            let mut last_store_end = i + 1;
            while j < body_end {
                match ops[j].op_type.as_str() {
                    "ktdp.store" => {
                        j += 1;
                        last_store_end = j;
                    }
                    "ktdp.construct_access_tile" => {
                        j += 1;
                    }
                    _ => break,
                }
            }
            block_ends.push(last_store_end); // exclusive end (just past last store)
            i = last_store_end;
        } else {
            i += 1;
        }
    }
    let k = block_ends.len();
    if k < 2 {
        return None;
    }

    // The last store-run must end the body; ops before the first block's store
    // run are a shared prologue (hoisted views / constants) kept verbatim.
    if block_ends[k - 1] != body_end {
        return None;
    }
    // Block boundaries: block `idx` spans `(prev_end, block_ends[idx]]`. The
    // FIRST block also absorbs the prologue's tail up to block_starts[0]; we set
    // block starts from the previous block's end (block 0 starts after the
    // prologue, which we identify as everything before the first block's first
    // CORE op — see below). Blocks may have UNEQUAL length (block 0 commonly
    // shares hoisted views and lacks a leading offset constant), so we compare
    // structure on CORE ops only (excluding `construct_memory_view` and index
    // `arith.constant` ops, which are hoisting/offset bookkeeping).
    let mut block_starts: Vec<usize> = Vec::with_capacity(k);
    let mut prev = 0usize;
    for &end in &block_ends {
        block_starts.push(prev);
        prev = end;
    }
    // Refine block 0's start: the prologue is the maximal prefix of view/const
    // ops before the first CORE op. Everything from the first core op onward is
    // block 0.
    let is_core = |op: &Operation| -> bool {
        !(op.op_type == "ktdp.construct_memory_view"
            || (op.op_type == "arith.constant"
                && matches!(op.result_type.as_deref(), Some("index") | None)))
    };
    let first_core = (0..block_ends[0]).find(|&i| is_core(&ops[i]))?;
    block_starts[0] = first_core;
    let prologue = &ops[..first_core];

    let mut blocks: Vec<&[Operation]> = Vec::with_capacity(k);
    for idx in 0..k {
        blocks.push(&ops[block_starts[idx]..block_ends[idx]]);
    }

    let k_i64 = k as i64;

    // Build per-block CORE op-index lists (positions within each block slice that
    // are core ops). All blocks must have the SAME number of core ops with the
    // SAME op-type signature.
    let core_idx: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| {
            b.iter()
                .enumerate()
                .filter(|(_, op)| is_core(op))
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        })
        .collect();
    let ncore = core_idx[0].len();
    if ncore == 0 || core_idx.iter().any(|c| c.len() != ncore) {
        return None;
    }
    let core_sig: Vec<&str> = core_idx[0]
        .iter()
        .map(|&i| blocks[0][i].op_type.as_str())
        .collect();
    for (b, ci) in blocks.iter().zip(&core_idx) {
        let sig: Vec<&str> = ci.iter().map(|&i| b[i].op_type.as_str()).collect();
        if sig != core_sig {
            return None;
        }
    }

    // Track, per CORE position, the per-block leading-index DELTA, and map each
    // access tile to the memory view (operand[0]) it reads in block 0; require a
    // single consistent delta per view (so the view's leading stride is well-
    // defined). delta_for_cpos[c] = delta_vec: indices_j = indices_0 + j*delta.
    let mut delta_for_cpos: HashMap<usize, Vec<i64>> = HashMap::new();
    let mut view_for_cpos: HashMap<usize, String> = HashMap::new();

    for cpos in 0..ncore {
        let op0 = &blocks[0][core_idx[0][cpos]];
        if op0.op_type != "ktdp.construct_access_tile" {
            // Non-access-tile core ops must be structurally identical across
            // blocks (same attributes — only access-tile offsets vary).
            for (b, ci) in blocks[1..].iter().zip(&core_idx[1..]) {
                let opj = &b[ci[cpos]];
                if opj.attributes != op0.attributes {
                    return None;
                }
            }
            continue;
        }

        // shape must be identical across blocks.
        let shape0 = match op0.attributes.get("shape") {
            Some(Attr::IntList(v)) if !v.is_empty() => v.clone(),
            _ => return None,
        };
        // Resolve block 0's index operands (operands[1..]) to constants.
        let idx0 = resolve_indices(op0, &int_const)?;
        // The coalesced access tile reuses block 0's FIRST index operand as the
        // new leading index (which must address row 0 of the prepended K axis).
        // Require that operand to resolve to 0, else the reuse is unsound.
        if idx0.first() != Some(&0) {
            return None;
        }

        // Per-block: same shape, indices = idx0 + j*delta with delta constant.
        let mut delta: Option<Vec<i64>> = None;
        for (jb, (b, ci)) in blocks.iter().zip(&core_idx).enumerate() {
            let opj = &b[ci[cpos]];
            let shapej = match opj.attributes.get("shape") {
                Some(Attr::IntList(v)) => v,
                _ => return None,
            };
            if *shapej != shape0 {
                return None;
            }
            // base_map / coordinate_set / coordinate_order must match block 0
            // (only the index offsets vary).
            if opj.attributes.get("base_map") != op0.attributes.get("base_map")
                || opj.attributes.get("coordinate_set") != op0.attributes.get("coordinate_set")
                || opj.attributes.get("coordinate_order") != op0.attributes.get("coordinate_order")
            {
                return None;
            }
            let idxj = resolve_indices(opj, &int_const)?;
            if idxj.len() != idx0.len() {
                return None;
            }
            if jb == 0 {
                // block 0 defines the base; delta inferred from block 1 below.
                continue;
            }
            // Recover the per-step delta from this block and require it be a
            // consistent affine progression: (idxj - idx0) must be divisible by
            // jb and equal jb * delta.
            let mut step = Vec::with_capacity(idx0.len());
            for (a, b0) in idxj.iter().zip(&idx0) {
                let d = a - b0;
                if d % jb as i64 != 0 {
                    return None;
                }
                step.push(d / jb as i64);
            }
            match &delta {
                None => delta = Some(step),
                Some(prev) => {
                    if *prev != step {
                        return None; // not a consistent linear progression
                    }
                }
            }
        }
        let delta = delta?; // K >= 2 guarantees at least one non-zero block
        delta_for_cpos.insert(cpos, delta.clone());

        // View consistency: all access tiles over the same view must agree on
        // delta (else the view's leading stride is ambiguous).
        let view = op0.operands.first()?.clone();
        view_for_cpos.insert(cpos, view);
    }

    // Aggregate per-view deltas; require a single delta per view.
    let mut view_delta: HashMap<String, Vec<i64>> = HashMap::new();
    for (cpos, view) in &view_for_cpos {
        let delta = &delta_for_cpos[cpos];
        match view_delta.get(view) {
            None => {
                view_delta.insert(view.clone(), delta.clone());
            }
            Some(prev) => {
                if prev != delta {
                    return None;
                }
            }
        }
    }

    // Compute each view's leading stride S = dot(delta, view.strides). The view
    // may be defined in the prologue OR inside block 0; collect strides from the
    // matching `construct_memory_view`.
    let mut view_strides: HashMap<String, Vec<i64>> = HashMap::new();
    for op in prologue.iter().chain(blocks[0].iter()) {
        if op.op_type == "ktdp.construct_memory_view"
            && let Some(res) = &op.result
            && let Some(Attr::IntList(s)) = op.attributes.get("strides")
        {
            view_strides.insert(res.clone(), s.clone());
        }
    }
    // S per view.
    let mut view_lead_stride: HashMap<String, i64> = HashMap::new();
    for (view, delta) in &view_delta {
        let strides = view_strides.get(view)?;
        if strides.len() != delta.len() {
            return None;
        }
        let s: i64 = delta.iter().zip(strides).map(|(d, st)| d * st).sum();
        if s <= 0 {
            return None; // degenerate / overlapping; bail
        }
        view_lead_stride.insert(view.clone(), s);
    }

    // ---- Build the coalesced body: prologue (with referenced views reshaped) +
    // block 0 (with a leading K dim prepended), blocks 1..K dropped. ----
    let mut new_ops: Vec<Operation> = Vec::with_capacity(prologue.len() + blocks[0].len() + 1);

    // Prologue: reshape any memory view that a coalesced access tile reads.
    for op in prologue {
        let mut nop = op.clone();
        if op.op_type == "ktdp.construct_memory_view"
            && let Some(res) = &op.result
            && let Some(&s) = view_lead_stride.get(res)
        {
            reshape_view_prepend(&mut nop, k_i64, s)?;
        }
        new_ops.push(nop);
    }

    // Block 0: prepend a leading K dim to every value.
    for op in blocks[0] {
        let mut nop = op.clone();
        let lead_stride = op
            .result
            .as_deref()
            .and_then(|r| view_lead_stride.get(r).copied());
        prepend_op_dim(&mut nop, k_i64, lead_stride)?;
        new_ops.push(nop);
    }

    if has_return {
        new_ops.push(ops[body_end].clone());
    }
    Some(new_ops)
}

/// Resolve an access tile's leading index operands (`operands[1..]`) to integer
/// constants via the int-constant table. Returns `None` if any is unknown.
fn resolve_indices(op: &Operation, int_const: &HashMap<String, i64>) -> Option<Vec<i64>> {
    op.operands[1..]
        .iter()
        .map(|name| int_const.get(name).copied())
        .collect()
}

/// Reshape a `construct_memory_view` op IN PLACE to prepend a leading dim of
/// extent `k` with element stride `s`: `shape -> [k, ...]`, `strides -> [s,
/// ...]`, `coordinate_set` gains a leading `0..k-1` box, and the result_type
/// (`memref<...>`) gains a leading `k`. Fail-safe.
fn reshape_view_prepend(op: &mut Operation, k: i64, s: i64) -> Option<()> {
    if let Some(Attr::IntList(v)) = op.attributes.get_mut("shape") {
        v.insert(0, k);
    } else {
        return None; // dynamic-size view: not handled
    }
    if let Some(Attr::IntList(st)) = op.attributes.get_mut("strides") {
        st.insert(0, s);
    } else {
        return None;
    }
    if let Some(Attr::AffineSet(set)) = op.attributes.get("coordinate_set") {
        let new = prepend_set_dim(set, k);
        op.attributes
            .insert("coordinate_set".to_string(), Attr::AffineSet(new));
    }
    if let Some(rt) = &op.result_type
        && let Some(nt) = prepend_type_dim(rt, k)
    {
        op.result_type = Some(nt);
    }
    Some(())
}

/// Prepend a leading dim of extent `k` to one op (in place). `lead_stride` is
/// `Some(s)` only for `construct_memory_view` ops defined INSIDE the block whose
/// view is read by a coalesced access tile. Index `arith.constant` ops are left
/// verbatim (they carry block offsets, which the access tile's leading index 0
/// now subsumes). Returns `None` if a required rewrite cannot be applied.
fn prepend_op_dim(op: &mut Operation, k: i64, lead_stride: Option<i64>) -> Option<()> {
    match op.op_type.as_str() {
        // Index constants stay verbatim.
        "arith.constant"
            if matches!(op.result_type.as_deref(), Some("index") | None)
                && op
                    .attributes
                    .get("value")
                    .map(|a| matches!(a, Attr::Int(_)))
                    .unwrap_or(false) =>
        {
            return Some(());
        }
        // Memory views read by a coalesced access tile: reshape with the view's
        // leading stride. Views NOT read by a coalesced access tile keep their
        // rank (rare, but stay verbatim).
        "ktdp.construct_memory_view" => {
            if let Some(s) = lead_stride {
                return reshape_view_prepend(op, k, s);
            }
            return Some(());
        }
        _ => {}
    }

    // construct_access_tile: prepend a leading index 0, shape K, base_map +
    // coordinate_set leading identity dim, and the result type.
    if op.op_type == "ktdp.construct_access_tile" {
        // leading index operand 0: reuse operands[1] if it is a known c0, else
        // insert a fresh "%c0"-style operand. The block already binds a `0`
        // index constant (every access tile has one); we route the new leading
        // index through the SAME operand name as the existing first index when
        // that index is 0, otherwise we still need a zero. To stay self-
        // contained we require operand[1] to exist and inject a literal 0 token.
        if op.operands.len() < 2 {
            return None;
        }
        // Insert the new leading index right after the view operand. We need a
        // value that resolves to 0; reuse the existing first index operand only
        // if it is itself 0 is not guaranteed, so synthesize a dedicated zero
        // operand name understood by the interpreter's constant table. The
        // interpreter resolves operand SSA names through scope, so we cannot
        // invent a name; instead we rely on the access tile's existing `%c0`
        // (operand[1]) being 0 in the canonical RoPE/copy shape. Verify it.
        // (Callers that don't satisfy this were rejected upstream by requiring
        // idx0[0] == 0 below.)
        let zero_operand = op.operands[1].clone();
        op.operands.insert(1, zero_operand);

        // shape: prepend K.
        if let Some(Attr::IntList(v)) = op.attributes.get_mut("shape") {
            v.insert(0, k);
        } else {
            return None;
        }
        // base_map: prepend identity leading dim.
        if let Some(Attr::AffineMap(m)) = op.attributes.get("base_map") {
            op.attributes
                .insert("base_map".to_string(), Attr::AffineMap(prepend_map_dim(m)));
        } else {
            // synthesize an identity map of the new rank (operands-1 indices).
            let n = op.operands.len().saturating_sub(1);
            op.attributes.insert(
                "base_map".to_string(),
                Attr::AffineMap(AffineMap::identity(n)),
            );
        }
        // coordinate_set: prepend leading 0..k-1 box (shift others).
        if let Some(Attr::AffineSet(set)) = op.attributes.get("coordinate_set") {
            let new = prepend_set_dim(set, k);
            op.attributes
                .insert("coordinate_set".to_string(), Attr::AffineSet(new));
        }
        // coordinate_order: prepend identity leading dim.
        if let Some(Attr::AffineMap(m)) = op.attributes.get("coordinate_order") {
            op.attributes.insert(
                "coordinate_order".to_string(),
                Attr::AffineMap(prepend_map_dim(m)),
            );
        }
        if let Some(rt) = &op.result_type {
            op.result_type = Some(prepend_type_dim(rt, k)?);
        }
        return Some(());
    }

    // linalg.broadcast: the broadcast `dimensions` index the OUTPUT axes; a new
    // leading axis shifts them all by +1.
    if op.op_type == "linalg.broadcast"
        && let Some(Attr::IntList(dims)) = op.attributes.get_mut("dimensions")
    {
        for d in dims.iter_mut() {
            *d += 1;
        }
    }

    // Any op: prepend K to a `shape` IntList attr (tensor.empty / broadcast outs)
    // and to a shaped tensor result_type, when present.
    if let Some(Attr::IntList(v)) = op.attributes.get_mut("shape") {
        v.insert(0, k);
    }
    if let Some(rt) = &op.result_type
        && rt.starts_with("tensor<")
        && let Some(nt) = prepend_type_dim(rt, k)
    {
        op.result_type = Some(nt);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktir_core::affine::Constraint;

    fn const_idx(name: &str, v: i64) -> Operation {
        Operation::new(Some(name), "arith.constant", &[])
            .with_attr("value", Attr::Int(v))
            .with_attr_rt("index")
    }

    trait WithRt {
        fn with_attr_rt(self, rt: &str) -> Self;
    }
    impl WithRt for Operation {
        fn with_attr_rt(mut self, rt: &str) -> Self {
            self.result_type = Some(rt.to_string());
            self
        }
    }

    /// dim0 box set `-d0 + (h-1) >= 0 & d0 >= 0`.
    fn tile_set(h: i64) -> Attr {
        Attr::AffineSet(AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::Dim(0),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: AffineExpr::Add(
                        Rc::new(AffineExpr::Neg(Rc::new(AffineExpr::Dim(0)))),
                        Rc::new(AffineExpr::Const(h - 1)),
                    ),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        })
    }

    fn view_1d(name: &str, ptr: &str, n: i64) -> Operation {
        Operation::new(Some(name), "ktdp.construct_memory_view", &[ptr])
            .with_attr("shape", Attr::IntList(vec![n]))
            .with_attr("strides", Attr::IntList(vec![1]))
            .with_attr_rt(&format!("memref<{n}xf16>"))
    }

    /// One block of a 1-D copy at row offset `off` (height `h`):
    /// load view_in[off] -> exp -> store view_out[off].
    fn block(off_name: &str, h: i64, tag: usize) -> Vec<Operation> {
        let at_in = format!("%ati{tag}");
        let ld = format!("%ld{tag}");
        let ex = format!("%ex{tag}");
        let at_out = format!("%ato{tag}");
        vec![
            Operation::new(
                Some(&at_in),
                "ktdp.construct_access_tile",
                &["%vin", off_name],
            )
            .with_attr("shape", Attr::IntList(vec![h]))
            .with_attr("base_map", Attr::AffineMap(AffineMap::identity(1)))
            .with_attr("coordinate_set", tile_set(h))
            .with_attr_rt(&format!("!ktdp.access_tile<{h}xindex>")),
            Operation::new(Some(&ld), "ktdp.load", &[&at_in])
                .with_attr_rt(&format!("tensor<{h}xf16>")),
            Operation::new(Some(&ex), "math.exp", &[&ld]).with_attr_rt(&format!("tensor<{h}xf16>")),
            Operation::new(
                Some(&at_out),
                "ktdp.construct_access_tile",
                &["%vout", off_name],
            )
            .with_attr("shape", Attr::IntList(vec![h]))
            .with_attr("base_map", Attr::AffineMap(AffineMap::identity(1)))
            .with_attr("coordinate_set", tile_set(h))
            .with_attr_rt(&format!("!ktdp.access_tile<{h}xindex>")),
            Operation::new(None, "ktdp.store", &[&ex, &at_out]),
        ]
    }

    fn copy_func(offsets: &[i64], h: i64) -> IRFunction {
        let mut ops = vec![
            view_1d("%vin", "%pin", 4096),
            view_1d("%vout", "%pout", 4096),
        ];
        for (j, &off) in offsets.iter().enumerate() {
            ops.push(const_idx(&format!("%off{j}"), off));
        }
        for (j, _) in offsets.iter().enumerate() {
            ops.extend(block(&format!("%off{j}"), h, j));
        }
        ops.push(Operation::new(None, "func.return", &[]));
        IRFunction {
            name: "copy".into(),
            arguments: vec![],
            operations: ops,
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    #[test]
    fn coalesces_two_contiguous_blocks() {
        let f = copy_func(&[0, 32], 32);
        let new_ops = recognize_coalesce(&f).expect("should coalesce");
        // Stores: exactly one (K blocks collapsed to one).
        assert_eq!(
            new_ops.iter().filter(|o| o.op_type == "ktdp.store").count(),
            1
        );
        // Access tile shape prepends K=2: [32] -> [2, 32].
        let at = new_ops
            .iter()
            .find(|o| o.op_type == "ktdp.construct_access_tile")
            .unwrap();
        assert_eq!(
            at.attributes.get("shape"),
            Some(&Attr::IntList(vec![2, 32]))
        );
        assert_eq!(
            at.result_type.as_deref(),
            Some("!ktdp.access_tile<2x32xindex>")
        );
        // Leading index operand 0 inserted (view, idx0, idx_inner).
        assert_eq!(at.operands.len(), 3);
        // base_map prepends identity leading dim -> rank 2, first result Dim(0).
        if let Some(Attr::AffineMap(m)) = at.attributes.get("base_map") {
            assert_eq!(m.num_dims, 2);
            assert_eq!(m.exprs[0], AffineExpr::Dim(0));
        } else {
            panic!("missing base_map");
        }
        // coordinate_set gains leading 0..1 box, inner shifted.
        if let Some(Attr::AffineSet(set)) = at.attributes.get("coordinate_set") {
            assert_eq!(set.num_dims, 2);
            // leading upper bound -d0 + 1 >= 0 (k-1 == 1).
            let (c, k) = linearize_probe(&set.constraints[1].expr);
            assert_eq!((c, k), (vec![-1, 0], 1));
        } else {
            panic!("missing coordinate_set");
        }
        // Loaded tensor prepends K: tensor<32xf16> -> tensor<2x32xf16>.
        let ld = new_ops.iter().find(|o| o.op_type == "ktdp.load").unwrap();
        assert_eq!(ld.result_type.as_deref(), Some("tensor<2x32xf16>"));
        // The input view is reshaped: shape [4096] -> [2, 4096], stride [1] ->
        // [S, 1] where S = delta(32) * stride(1) = 32.
        let vin = new_ops
            .iter()
            .find(|o| o.result.as_deref() == Some("%vin"))
            .unwrap();
        assert_eq!(
            vin.attributes.get("shape"),
            Some(&Attr::IntList(vec![2, 4096]))
        );
        assert_eq!(
            vin.attributes.get("strides"),
            Some(&Attr::IntList(vec![32, 1]))
        );
    }

    /// Linearize a `-d0 + c` style expr into (dim_coeffs, const) by probing.
    fn linearize_probe(expr: &AffineExpr) -> (Vec<i64>, i64) {
        let base = expr.eval(&[0, 0], &[]);
        let c0 = expr.eval(&[1, 0], &[]) - base;
        let c1 = expr.eval(&[0, 1], &[]) - base;
        (vec![c0, c1], base)
    }

    /// A STRIDED-operand RoPE-like case: a `[32,32]` row tile stepping by h=32
    /// rows AND a `[32]` cos table stepping by 2h=64 (stride != tile height).
    /// This must now COALESCE: the cos view reshapes to leading stride 64, the
    /// row view to leading stride 32*stride. Asserts the strided load reads the
    /// right elements via the reshaped view.
    fn rope_func(k: usize) -> IRFunction {
        // row view: [1024, 64] strides [64,1]; cos view: [2048] stride [1].
        let mut ops = vec![
            Operation::new(Some("%vrow"), "ktdp.construct_memory_view", &["%prow"])
                .with_attr("shape", Attr::IntList(vec![1024, 64]))
                .with_attr("strides", Attr::IntList(vec![64, 1]))
                .with_attr_rt("memref<1024x64xf16>"),
            Operation::new(Some("%vout"), "ktdp.construct_memory_view", &["%pout"])
                .with_attr("shape", Attr::IntList(vec![1024, 64]))
                .with_attr("strides", Attr::IntList(vec![64, 1]))
                .with_attr_rt("memref<1024x64xf16>"),
        ];
        // constants: c0, and per-block row off (32*j) and cos off (64*j).
        ops.push(const_idx("%c0", 0));
        for j in 0..k {
            ops.push(const_idx(&format!("%row{j}"), 32 * j as i64));
            ops.push(const_idx(&format!("%cos{j}"), 64 * j as i64));
        }
        let row_set = || {
            Attr::AffineSet(AffineSet {
                num_dims: 2,
                num_syms: 0,
                constraints: vec![
                    Constraint {
                        expr: AffineExpr::Dim(0),
                        kind: ConstraintKind::GreaterEq,
                    },
                    Constraint {
                        expr: AffineExpr::Add(
                            Rc::new(AffineExpr::Neg(Rc::new(AffineExpr::Dim(0)))),
                            Rc::new(AffineExpr::Const(31)),
                        ),
                        kind: ConstraintKind::GreaterEq,
                    },
                    Constraint {
                        expr: AffineExpr::Dim(1),
                        kind: ConstraintKind::GreaterEq,
                    },
                    Constraint {
                        expr: AffineExpr::Add(
                            Rc::new(AffineExpr::Neg(Rc::new(AffineExpr::Dim(1)))),
                            Rc::new(AffineExpr::Const(31)),
                        ),
                        kind: ConstraintKind::GreaterEq,
                    },
                ],
            })
        };
        for j in 0..k {
            let rk = format!("%row{j}");
            let ck = format!("%cos{j}");
            ops.extend(vec![
                // row load [32,32] at [32j, 0]
                Operation::new(
                    Some(&format!("%rl{j}")),
                    "ktdp.construct_access_tile",
                    &["%vrow", &rk, "%c0"],
                )
                .with_attr("shape", Attr::IntList(vec![32, 32]))
                .with_attr("base_map", Attr::AffineMap(AffineMap::identity(2)))
                .with_attr("coordinate_set", row_set())
                .with_attr_rt("!ktdp.access_tile<32x32xindex>"),
                Operation::new(Some(&format!("%rv{j}")), "ktdp.load", &[&format!("%rl{j}")])
                    .with_attr_rt("tensor<32x32xf16>"),
                // cos load [32] at [64j]
                Operation::new(
                    Some(&format!("%cl{j}")),
                    "ktdp.construct_access_tile",
                    &["%vcos", &ck],
                )
                .with_attr("shape", Attr::IntList(vec![32]))
                .with_attr("base_map", Attr::AffineMap(AffineMap::identity(1)))
                .with_attr("coordinate_set", tile_set(32))
                .with_attr_rt("!ktdp.access_tile<32xindex>"),
                Operation::new(Some(&format!("%cv{j}")), "ktdp.load", &[&format!("%cl{j}")])
                    .with_attr_rt("tensor<32xf16>"),
                // broadcast cos [32] -> [32,32] along dim 0
                Operation::new(Some(&format!("%ci{j}")), "tensor.empty", &[])
                    .with_attr("shape", Attr::IntList(vec![32, 32]))
                    .with_attr_rt("tensor<32x32xf16>"),
                Operation::new(
                    Some(&format!("%cb{j}")),
                    "linalg.broadcast",
                    &[&format!("%cv{j}"), &format!("%ci{j}")],
                )
                .with_attr("dimensions", Attr::IntList(vec![0]))
                .with_attr_rt("tensor<32x32xf16>"),
                // out = row * cosb
                Operation::new(
                    Some(&format!("%o{j}")),
                    "arith.mulf",
                    &[&format!("%rv{j}"), &format!("%cb{j}")],
                )
                .with_attr_rt("tensor<32x32xf16>"),
                // store
                Operation::new(
                    Some(&format!("%sl{j}")),
                    "ktdp.construct_access_tile",
                    &["%vout", &rk, "%c0"],
                )
                .with_attr("shape", Attr::IntList(vec![32, 32]))
                .with_attr("base_map", Attr::AffineMap(AffineMap::identity(2)))
                .with_attr("coordinate_set", row_set())
                .with_attr_rt("!ktdp.access_tile<32x32xindex>"),
                Operation::new(None, "ktdp.store", &[&format!("%o{j}"), &format!("%sl{j}")]),
            ]);
        }
        // cos view declared once in prologue (shared).
        ops.insert(
            2,
            Operation::new(Some("%vcos"), "ktdp.construct_memory_view", &["%pcos"])
                .with_attr("shape", Attr::IntList(vec![2048]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr_rt("memref<2048xf16>"),
        );
        ops.push(Operation::new(None, "func.return", &[]));
        IRFunction {
            name: "rope".into(),
            arguments: vec![],
            operations: ops,
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    #[test]
    fn coalesces_strided_cos_operand() {
        let f = rope_func(4);
        let new_ops = recognize_coalesce(&f).expect("RoPE strided case should coalesce");
        // One store run (4 blocks -> 1).
        assert_eq!(
            new_ops.iter().filter(|o| o.op_type == "ktdp.store").count(),
            1
        );
        // cos view reshaped: [2048] strides [1] -> [4, 2048] strides [64, 1].
        // (delta=64, view stride=1 -> S=64).
        let vcos = new_ops
            .iter()
            .find(|o| o.result.as_deref() == Some("%vcos"))
            .unwrap();
        assert_eq!(
            vcos.attributes.get("shape"),
            Some(&Attr::IntList(vec![4, 2048]))
        );
        assert_eq!(
            vcos.attributes.get("strides"),
            Some(&Attr::IntList(vec![64, 1]))
        );
        // The cos access tile became [4, 32] (row j reads cos[64j : 64j+32]).
        let cos_at = new_ops
            .iter()
            .find(|o| o.result.as_deref() == Some("%cl0"))
            .unwrap();
        assert_eq!(
            cos_at.attributes.get("shape"),
            Some(&Attr::IntList(vec![4, 32]))
        );
        assert_eq!(
            cos_at.result_type.as_deref(),
            Some("!ktdp.access_tile<4x32xindex>")
        );
        // row view reshaped: [1024,64] strides [64,1] -> [4,1024,64] strides
        // [2048,64,1] (delta=[32,0] dot strides = 32*64 = 2048).
        let vrow = new_ops
            .iter()
            .find(|o| o.result.as_deref() == Some("%vrow"))
            .unwrap();
        assert_eq!(
            vrow.attributes.get("strides"),
            Some(&Attr::IntList(vec![2048, 64, 1]))
        );
        // The broadcast shifts its dimensions [0] -> [1] (new leading axis).
        let bc = new_ops
            .iter()
            .find(|o| o.op_type == "linalg.broadcast")
            .unwrap();
        assert_eq!(
            bc.attributes.get("dimensions"),
            Some(&Attr::IntList(vec![1]))
        );
        // The mulf result prepends K: tensor<32x32xf16> -> tensor<4x32x32xf16>.
        let mul = new_ops.iter().find(|o| o.op_type == "arith.mulf").unwrap();
        assert_eq!(mul.result_type.as_deref(), Some("tensor<4x32x32xf16>"));
    }

    #[test]
    fn rejects_non_uniform_offset() {
        // Block 1 offset jumps non-linearly across 3 blocks -> bail.
        // offsets 0, 32, 96 (not an arithmetic progression: deltas 32 then 64).
        let f = copy_func(&[0, 32, 96], 32);
        assert!(recognize_coalesce(&f).is_none());
    }

    #[test]
    fn rejects_single_block() {
        let f = copy_func(&[0], 32);
        assert!(recognize_coalesce(&f).is_none());
    }

    #[test]
    fn rejects_multicore_grid() {
        let mut f = copy_func(&[0, 32], 32);
        f.grid = (9, 1, 1);
        assert!(recognize_coalesce(&f).is_none());
    }

    #[test]
    fn coalesces_three_blocks() {
        let f = copy_func(&[0, 32, 64], 32);
        let new_ops = recognize_coalesce(&f).expect("should coalesce");
        let at = new_ops
            .iter()
            .find(|o| o.op_type == "ktdp.construct_access_tile")
            .unwrap();
        assert_eq!(
            at.attributes.get("shape"),
            Some(&Attr::IntList(vec![3, 32]))
        );
    }
}
