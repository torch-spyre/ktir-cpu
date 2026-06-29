// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `linalg` dialect handlers — Rust port of `ktir_emulator/dialects/linalg_ops.py`.
//!
//! Ports the structured-op family: `matmul`, `batch_matmul`, `generic`,
//! `reduce`, `transpose`, `broadcast`, `fill`, `index`, and the `yield`
//! terminator. `generic` and `reduce` are *zero-cost orchestrators*: the cost
//! lives in the ops of their combiner region, which we execute via
//! `execute_region` exactly as Python executes them through `env.execute_region`.
//!
//! ## Region / yield handling
//!
//! The locked `interpreter::execute_region` returns `()`, not a value — so this
//! module threads the yielded value through a per-scope sentinel SSA binding,
//! [`YIELD_KEY`]. `linalg.yield %v` binds `%v` under that key in the current
//! scope; [`run_region`] reads it back out before the caller pops the scope.
//! This mirrors Python's `_YieldResult` / `unwrap_yield` plumbing, kept local to
//! linalg since the shared scf yield seam is not yet in the Rust tree.
//!
//! ## N-dimensional tiles
//!
//! `Tile` stores a flat `Vec<f32>` + a `shape`; this module carries the
//! row-major index arithmetic NumPy gives for free in Python (strides, broadcast,
//! transpose, axis reductions) as small local helpers.

use super::{Dispatch, LatencyCategory};
use crate::affine::AffineMap;
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::interpreter::execute_region;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::tile::Tile;

/// Row-major `C(m×k·k×n)` for the emulator, on the highest-performance backend
/// available. With the `metal` feature this dispatches through the size-gated
/// NAX-or-Accelerate selector ([`crate::metal::metal_gemm_or_blas`]):
/// large GEMMs run on the M5 NAX tensor engine (bf16, ~2× Accelerate), small
/// ones on Accelerate (f32) — so unit-test-scale matmuls keep exact f32 parity
/// while production-scale ones get the GPU. Without `metal`, it's the BLAS path
/// (Accelerate on macOS, naive elsewhere).
fn gemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    // M == 1 is a matrix-VECTOR product (decode): route to a real GEMV instead of
    // the tiled GEMM / NAX `matmul2d`, which is built for M ≥ 8 and at M=1 leaves
    // ~15/16 of every matrix tile idle. Same math, same f32-accumulate, so golden
    // parity holds (the caller still rounds to f16 via `Tile::compute`); only the
    // BLAS/GPU routine differs. M > 1 keeps the GEMM path untouched.
    if m == 1 {
        return gemv(k, n, a, b);
    }
    #[cfg(metal)]
    {
        crate::metal::metal_gemm_or_blas(m, k, n, a, b)
    }
    #[cfg(not(metal))]
    {
        crate::blas::sgemm_rowmajor(m, k, n, a, b)
    }
}

/// The m=1 case of [`gemm`]: `y(n) = a(k) · B(k×n)`. Selects the GPU GEMV (when
/// the Metal backend is on and the op is large enough to win) or the CPU
/// `sgemv_rowmajor` (AMX/OpenBLAS), mirroring `gemm`'s size-gated dispatch.
fn gemv(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(metal)]
    {
        crate::metal::metal_gemv_or_blas(k, n, a, b)
    }
    #[cfg(not(metal))]
    {
        crate::blas::sgemv_rowmajor(k, n, a, b)
    }
}

/// The m=1 transpose-B case: `y(n) = a(k) · B(n×k)ᵀ`, B stored `[n,k]`. GPU GEMV
/// (transpose-B) or CPU `sgemv_rowmajor_bt` — the matrix-VECTOR analogue of
/// `matmul2d_bt`'s GEMM dispatch.
fn gemv_bt(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(metal)]
    {
        crate::metal::metal_gemv_or_blas_bt(k, n, a, b)
    }
    #[cfg(not(metal))]
    {
        crate::blas::sgemv_rowmajor_bt(k, n, a, b)
    }
}

/// Sentinel scope key under which `linalg.yield` parks its yielded value so the
/// region driver can recover it after `execute_region` (which itself returns
/// `()`). Chosen to never collide with a real SSA name.
const YIELD_KEY: &str = "__linalg_yield__";

/// Sentinel scope key holding the current `linalg.generic` iteration shape, so
/// `linalg.index` can build its broadcasting index array. Mirrors the Python
/// `__linalg_shape__` binding.
const SHAPE_KEY: &str = "__linalg_shape__";

pub fn register(d: &mut Dispatch) {
    // generic/matmul carry real float compute cost in Python (LC.COMPUTE_FLOAT /
    // LC.COMPUTE_MATMUL); map both onto ComputeFloat, the closest present
    // variant. reduce is LC.ZERO (cost lives in its region's ops).
    d.register("linalg.matmul", LatencyCategory::ComputeFloat, matmul);
    d.register(
        "linalg.batch_matmul",
        LatencyCategory::ComputeFloat,
        batch_matmul,
    );
    d.register("linalg.generic", LatencyCategory::ComputeFloat, generic);
    d.register("linalg.reduce", LatencyCategory::Zero, reduce);
    d.register("linalg.transpose", LatencyCategory::Zero, transpose);
    d.register("linalg.broadcast", LatencyCategory::Zero, broadcast);
    d.register("linalg.fill", LatencyCategory::Zero, fill);
    d.register("linalg.index", LatencyCategory::Zero, index);
    d.register("linalg.yield", LatencyCategory::Zero, yield_op);
    // Elementwise named ops: `linalg.add/sub/mul/div/max/min ins(%a, %b) outs(%c)`.
    d.register("linalg.add", LatencyCategory::ComputeFloat, |o, c, _| {
        elementwise(o, c, |a, b| a + b)
    });
    d.register("linalg.sub", LatencyCategory::ComputeFloat, |o, c, _| {
        elementwise(o, c, |a, b| a - b)
    });
    d.register("linalg.mul", LatencyCategory::ComputeFloat, |o, c, _| {
        elementwise(o, c, |a, b| a * b)
    });
    d.register("linalg.div", LatencyCategory::ComputeFloat, |o, c, _| {
        elementwise(o, c, |a, b| a / b)
    });
    d.register("linalg.max", LatencyCategory::ComputeFloat, |o, c, _| {
        elementwise(o, c, f32::max)
    });
    d.register("linalg.min", LatencyCategory::ComputeFloat, |o, c, _| {
        elementwise(o, c, f32::min)
    });
}

/// `%r = linalg.<op> ins(%a, %b) outs(%c)` — element-wise binary named op over
/// two tiles of equal shape. (The `outs` operand only supplies the result
/// shape/dtype; named elementwise ops overwrite, not accumulate.)
fn elementwise(
    op: &Operation,
    ctx: &mut CoreContext,
    f: fn(f32, f32) -> f32,
) -> Result<Option<Value>, String> {
    let a = expect_tile(ctx.get_value(&op.operands[0])?, "linalg elementwise A")?;
    let b = expect_tile(ctx.get_value(&op.operands[1])?, "linalg elementwise B")?;
    if a.shape != b.shape {
        return Err(format!(
            "linalg.{}: shape mismatch {:?} vs {:?}",
            op.op_type, a.shape, b.shape
        ));
    }
    let data: Vec<f32> = a
        .as_f32()
        .iter()
        .zip(b.as_f32().iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Ok(Some(Value::Tile(Tile::compute(
        data,
        a.dtype,
        a.shape.clone(),
    ))))
}

// ===========================================================================
// fill / broadcast / transpose
// ===========================================================================

/// `%r = linalg.fill ins(%scalar) outs(%init)` — fill a tile with a scalar.
fn fill(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let scalar = ctx.get_value(&op.operands[0])?;
    let scalar_val = as_f32(scalar, "linalg.fill scalar")?;
    let out = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.fill outs")?;
    let data = vec![scalar_val; out.len()];
    Ok(Some(Value::Tile(Tile::compute(
        data,
        out.dtype,
        out.shape.clone(),
    ))))
}

/// `%r = linalg.broadcast ins(%x) outs(%init) dimensions = [...]`.
///
/// Expands `dimensions` on the input then broadcasts to the outs shape. Mirrors
/// `np.expand_dims` over sorted dims followed by `np.broadcast_to`.
fn broadcast(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let inp = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.broadcast ins")?.clone();
    let out = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.broadcast outs")?;
    let out_shape = out.shape.clone();
    let out_dtype = inp.dtype;

    let mut dims = int_list_attr(op, "dimensions").cloned().unwrap_or_default();
    dims.sort_unstable();

    // Build the input's expanded shape: start from inp.shape, insert size-1 axes
    // at each broadcast dimension (sorted, so earlier inserts don't shift later).
    let mut shape: Vec<usize> = inp.shape.clone();
    for &d in &dims {
        let d = d as usize;
        if d > shape.len() {
            return Err(format!(
                "linalg.broadcast: dim {d} out of range for shape {shape:?}"
            ));
        }
        shape.insert(d, 1);
    }

    let data = broadcast_to(&inp.as_f32(), &shape, &out_shape)
        .ok_or_else(|| format!("linalg.broadcast: cannot broadcast {shape:?} to {out_shape:?}"))?;
    // Broadcast only REPLICATES the input's (already on-grid) values — no
    // arithmetic — so the result is on `out_dtype`'s grid too; skip `Tile::compute`'s
    // redundant `round_to_dtype` pass over the whole output via `from_decoded`.
    // Bit-identical.
    Ok(Some(Value::Tile(Tile::from_decoded(
        data, out_dtype, out_shape, None, None,
    ))))
}

/// `%r = linalg.transpose ins(%x) outs(%y) permutation = [...]`.
fn transpose(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let inp = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.transpose ins")?.clone();
    let perm = int_list_attr(op, "permutation")
        .ok_or("linalg.transpose: missing permutation attribute")?
        .iter()
        .map(|&p| p as usize)
        .collect::<Vec<_>>();
    if perm.len() != inp.shape.len() {
        return Err(format!(
            "linalg.transpose: permutation rank {} != input rank {}",
            perm.len(),
            inp.shape.len()
        ));
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| inp.shape[p]).collect();
    let in_strides = row_major_strides(&inp.shape);
    // Input stride to advance for a +1 step along each OUTPUT axis k (output axis
    // k maps to input axis perm[k]). Walking the output in row-major order, we
    // then maintain the source offset incrementally via an odometer — no
    // per-element `unravel` (which allocated two Vecs per output element and
    // dominated the decode host profile).
    let out_src_strides: Vec<usize> = perm.iter().map(|&p| in_strides[p]).collect();
    let inp_data = inp.as_f32();
    let mut data = vec![0.0f32; inp_data.len()];
    let rank = new_shape.len();
    let mut out_idx = vec![0usize; rank];
    let mut src = 0usize;
    for slot in data.iter_mut() {
        *slot = inp_data[src];
        // Advance the output multi-index like an odometer (rightmost = innermost),
        // updating `src` by the corresponding input stride on each carry.
        let mut k = rank;
        while k > 0 {
            k -= 1;
            out_idx[k] += 1;
            src += out_src_strides[k];
            if out_idx[k] < new_shape[k] {
                break;
            }
            out_idx[k] = 0;
            src -= out_src_strides[k] * new_shape[k];
        }
    }
    // Transpose only PERMUTES the input's (already on-grid) values — no arithmetic —
    // so skip `Tile::compute`'s redundant round via `from_decoded`. Bit-identical.
    Ok(Some(Value::Tile(Tile::from_decoded(
        data, inp.dtype, new_shape, None, None,
    ))))
}

// ===========================================================================
// matmul / batch_matmul
// ===========================================================================

/// `%r = linalg.matmul ins(%A, %B) outs(%C)` -> `C + Aᵀ?·Bᵀ?`.
///
/// Per upstream MLIR (the linalg `matmul_transpose_{a,b}` named ops were
/// removed in favour of this), transpose semantics ride on an optional
/// `indexing_maps` affine-map list rather than a dedicated op. We read those
/// maps and route by *layout*, not op name: a transposed B (map `(n, k)`)
/// reaches the zero-copy `Bᵀ` kernels. Absent `indexing_maps`, it's a plain
/// `A·B` — the default linalg.matmul contract.
fn matmul(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let (transpose_a, transpose_b) = matmul_transpose_flags(op)?;
    matmul_dispatch(op, ctx, transpose_a, transpose_b, "linalg.matmul")
}

/// Resolve `(transpose_a, transpose_b)` for a `linalg.matmul` from its optional
/// `indexing_maps`. The **single source of truth** shared by the scalar
/// interpreter dispatch ([`matmul`]) and the Metal/NAX K-loop offload
/// recognizer (`metal::recognize_matmul_loop`), so the transpose layout is
/// decided one way everywhere — from the affine maps, never the op name.
///
/// `indexing_maps` present -> per [`classify_matmul_maps`]; absent ->
/// `(false, false)` (plain `A·B`); present but not an affine-map list -> `Err`
/// (don't guess).
pub(crate) fn matmul_transpose_flags(op: &Operation) -> Result<(bool, bool), String> {
    match op.attributes.get("indexing_maps") {
        Some(Attr::AffineMapList(maps)) => classify_matmul_maps(maps),
        None => Ok((false, false)),
        Some(other) => Err(format!(
            "linalg.matmul: indexing_maps must be an affine-map list, got {other:?}"
        )),
    }
}

/// Shared 2-D matmul body. `transpose_b` reads `B` as `[n, k]` and contracts the
/// last axis of both operands — the zero-copy `Bᵀ` path ([`matmul2d_bt`]);
/// otherwise plain `A·B` ([`matmul2d`]). `outs` (operands[2]) accumulates.
fn matmul_dispatch(
    op: &Operation,
    ctx: &mut CoreContext,
    transpose_a: bool,
    transpose_b: bool,
    name: &str,
) -> Result<Option<Value>, String> {
    let a = expect_tile(ctx.get_value(&op.operands[0])?, name)?.clone();
    let b = expect_tile(ctx.get_value(&op.operands[1])?, name)?.clone();
    let result = match (transpose_a, transpose_b) {
        (false, false) => matmul2d(&a, &b)?,
        (false, true) => matmul2d_bt(&a, &b, name)?,
        // Transpose-A (`indexing_maps` A map `(k, m)`) is valid upstream but not
        // emitted here; reject loudly rather than silently transpose-B it.
        (true, _) => {
            return Err(format!(
                "{name}: transpose-A indexing_maps (A = (k, m)) not yet supported"
            ));
        }
    };
    let result = accumulate_outs(op, ctx, result, name)?;
    Ok(Some(Value::Tile(result)))
}

/// Classify a `linalg.matmul` `indexing_maps` list `[A, B, C]` over iteration
/// dims `(m, n, k) = (d0, d1, d2)` into `(transpose_a, transpose_b)`.
///
/// The four canonical operand maps:
/// - A: `(d0, d2)` = `[m, k]` (normal) or `(d2, d0)` = `[k, m]` (transpose-A)
/// - B: `(d2, d1)` = `[k, n]` (normal) or `(d1, d2)` = `[n, k]` (transpose-B)
/// - C: `(d0, d1)` = `[m, n]` (output; fixed)
///
/// Errors on any non-canonical map (wrong arity, broadcast, reduction in the
/// wrong place) rather than guessing.
fn classify_matmul_maps(maps: &[AffineMap]) -> Result<(bool, bool), String> {
    if maps.len() != 3 {
        return Err(format!(
            "linalg.matmul: indexing_maps must list 3 maps (A, B, C), got {}",
            maps.len()
        ));
    }
    let dims: Vec<Option<Vec<usize>>> = maps.iter().map(|m| m.result_dims()).collect();
    let transpose_a = match dims[0].as_deref() {
        Some([0, 2]) => false,
        Some([2, 0]) => true,
        _ => {
            return Err(format!(
                "linalg.matmul: A indexing_map must be (m, k) or (k, m), got {:?}",
                maps[0]
            ));
        }
    };
    let transpose_b = match dims[1].as_deref() {
        Some([2, 1]) => false,
        Some([1, 2]) => true,
        _ => {
            return Err(format!(
                "linalg.matmul: B indexing_map must be (k, n) or (n, k), got {:?}",
                maps[1]
            ));
        }
    };
    if dims[2].as_deref() != Some(&[0, 1]) {
        return Err(format!(
            "linalg.matmul: C indexing_map must be (m, n), got {:?}",
            maps[2]
        ));
    }
    Ok((transpose_a, transpose_b))
}

/// `%r = linalg.batch_matmul ins(%A, %B) outs(%C)` over the leading batch dim.
fn batch_matmul(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let a = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.batch_matmul A")?.clone();
    let b = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.batch_matmul B")?.clone();
    if a.shape.len() != 3 || b.shape.len() != 3 {
        return Err(format!(
            "linalg.batch_matmul: expected 3-D operands, got {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
    if b.shape[0] != batch || b.shape[1] != k {
        return Err(format!(
            "linalg.batch_matmul: incompatible shapes {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let n = b.shape[2];
    let a_data = a.as_f32();
    let b_data = b.as_f32();
    let mut data = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a_slice = &a_data[bi * m * k..(bi + 1) * m * k];
        let b_slice = &b_data[bi * k * n..(bi + 1) * k * n];
        let c = gemm(m, k, n, a_slice, b_slice);
        data[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c);
    }
    let result = Tile::compute(data, a.dtype, vec![batch, m, n]);

    let result = accumulate_outs(op, ctx, result, "linalg.batch_matmul")?;
    Ok(Some(Value::Tile(result)))
}

/// 2-D matmul `A @ B` keeping A's dtype. A is [M, K], B is [K, N].
fn matmul2d(a: &Tile, b: &Tile) -> Result<Tile, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "linalg.matmul: expected 2-D operands, got {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    if b.shape[0] != k {
        return Err(format!(
            "linalg.matmul: inner dims disagree: {:?} @ {:?}",
            a.shape, b.shape
        ));
    }
    let n = b.shape[1];
    let data = gemm(m, k, n, &a.as_f32(), &b.as_f32());
    Ok(Tile::compute(data, a.dtype, vec![m, n]))
}

/// Apply the optional `outs` accumulator (operands[2]) of a matmul-family op:
/// `result = C + A·B`, written in the accumulator's dtype. This is the single
/// place the matmul accumulate lives. Reads C through [`Tile::as_f32`] and rounds
/// the sum once via [`Tile::compute`] (matching the Python reference, which builds
/// `Tile(acc.data + result.data, acc.dtype)`).
fn accumulate_outs(
    op: &Operation,
    ctx: &CoreContext,
    result: Tile,
    op_name: &str,
) -> Result<Tile, String> {
    if op.operands.len() > 2
        && let Value::Tile(c) = ctx.get_value(&op.operands[2])?
    {
        if c.shape != result.shape {
            return Err(format!(
                "{op_name}: outs shape {:?} != product shape {:?}",
                c.shape, result.shape
            ));
        }
        let dtype = c.dtype;
        let shape = result.shape.clone();
        let sum: Vec<f32> = result
            .as_f32()
            .iter()
            .zip(c.as_f32().iter())
            .map(|(&r, &cv)| r + cv)
            .collect();
        return Ok(Tile::compute(sum, dtype, shape));
    }
    Ok(result)
}

/// `A[m,k] · B[n,k]ᵀ -> [m,n]`, contracting the last axis of both (transpose-B).
/// B is stored `[n, k]`; no data is transposed — `sgemm_rowmajor_bt` reads B's
/// rows directly (contiguous), so this is as fast as a plain GEMM.
fn matmul2d_bt(a: &Tile, b: &Tile, op_name: &str) -> Result<Tile, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "{op_name}: expected 2-D operands, got {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    if b.shape[1] != k {
        return Err(format!(
            "{op_name}: contraction dims disagree: {:?} · {:?}ᵀ",
            a.shape, b.shape
        ));
    }
    let n = b.shape[0];
    // M == 1 (decode) is a matrix-VECTOR transpose-B product — route to the GEMV
    // fast path; M > 1 keeps the tiled transpose-B GEMM.
    let data = if m == 1 {
        gemv_bt(k, n, &a.as_f32(), &b.as_f32())
    } else {
        crate::blas::sgemm_rowmajor_bt(m, k, n, &a.as_f32(), &b.as_f32())
    };
    Ok(Tile::compute(data, a.dtype, vec![m, n]))
}

// ===========================================================================
// reduce
// ===========================================================================

/// `%r = linalg.reduce ins(%x) outs(%init) dimensions = [d] { <combiner> }`.
///
/// Zero-cost orchestrator: the cost belongs to the combiner region's ops, not
/// the reduce. Both surface forms feed a pairwise tree fold of the combiner
/// region (`tree_fold`); shorthand (`{ arith.addf }`) synthesizes a one-op
/// region so it takes the identical path. Relies on the combiner being
/// associative (MLIR's `linalg.reduce` legalization already guarantees this).
fn reduce(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let tile = match ctx.get_value(&op.operands[0])? {
        Value::Tile(t) => t.clone(),
        // Already a scalar — nothing to reduce, pass it through.
        other => return Ok(Some(other.clone())),
    };

    // Resolve the combiner region (capturing bb0 arg names).
    let (mut bb0_names, mut body_ops) = resolve_region_body(op);

    // Combiner op name: explicit form has it as the region's first non-yield op;
    // shorthand stores it in `reduce_fn`. Default to arith.addf.
    let reduce_fn = match op.attributes.get("reduce_fn") {
        Some(Attr::Str(s)) => Some(s.clone()),
        _ => None,
    }
    .or_else(|| {
        body_ops
            .iter()
            .find(|o| o.op_type != "linalg.yield")
            .map(|o| o.op_type.clone())
    })
    .unwrap_or_else(|| "arith.addf".to_string());

    // Shorthand has no region — synthesize the explicit-form block:
    //   (%in, %out) { %s = <reduce_fn> %in, %out; linalg.yield %s }
    if body_ops.is_empty() {
        bb0_names = vec!["__reduce_in__".to_string(), "__reduce_acc__".to_string()];
        body_ops = vec![
            Operation::new(
                Some("__reduce_combined__"),
                &reduce_fn,
                &["__reduce_in__", "__reduce_acc__"],
            ),
            Operation::new(None, "linalg.yield", &["__reduce_combined__"]),
        ];
    }

    // Axes to reduce. MLIR text carries them as `dimensions = [d0, d1, ...]`
    // (IntList); the programmatic form uses `dim` (single Int). Semantics, mirroring
    // Python `linalg__reduce` after #106:
    //   * absent (`None`)  -> collapse ALL axes to a scalar (flatten then fold);
    //   * `[]` (empty)     -> reduce ZERO axes: identity (shape & values unchanged);
    //   * `[d0, d1, ...]`  -> fold each listed axis, rightmost (fastest-moving)
    //                         first, then squeeze every reduced axis.
    let dims: Option<Vec<usize>> = match op.attributes.get("dim") {
        Some(Attr::Int(d)) => Some(vec![*d as usize]),
        _ => match op.attributes.get("dimensions") {
            Some(Attr::IntList(v)) => Some(v.iter().map(|&d| d as usize).collect()),
            _ => None,
        },
    };

    // Fast path: a simple `(in, acc) { %s = <op> in, acc; yield %s }` body whose
    // op is a recognized commutative+associative combiner (the synthesized
    // shorthand form, and the only form the real model emits) folds directly in
    // the f32 buffer — no per-round region execution. The operand check ensures
    // the op combines exactly the two block args, not an external value.
    let fast_combine = if body_ops.len() == 2
        && body_ops[1].op_type == "linalg.yield"
        && body_ops[0].operands.len() == 2
        && body_ops[0].operands.iter().all(|o| {
            let o = o.trim_start_matches('%');
            bb0_names.iter().any(|n| n.trim_start_matches('%') == o)
        }) {
        reduce_combiner(&body_ops[0].op_type)
    } else {
        None
    };

    // Reduce. Tree-fold each axis independently, rightmost first, matching Python.
    // (This reorders element groupings vs. MLIR's left-associative scalar loop for
    // multi-axis f16 — the documented Python xfail `test_reduce_multi_axis_treefold_bug`
    // — but is the implemented behavior the oracle prescribes.)
    let (mut data, mut shape) = (tile.as_f32().to_vec(), tile.shape.clone());
    let mut reduced_value = match &dims {
        None => {
            // Collapse all: flatten then fold to a single scalar.
            let (folded, _) =
                tree_fold(&tile, None, &bb0_names, &body_ops, fast_combine, ctx, env)?;
            Value::Scalar(Scalar::F32(folded[0]))
        }
        Some(ds) if ds.is_empty() => {
            // Reduce zero axes — identity.
            Value::Tile(Tile::compute(data.clone(), tile.dtype, shape.clone()))
        }
        Some(ds) => {
            // Fold each axis (rightmost first), then squeeze the reduced axes.
            let mut sorted = ds.clone();
            sorted.sort_unstable();
            for &d in sorted.iter().rev() {
                let cur = Tile::compute(data.clone(), tile.dtype, shape.clone());
                let (folded, fshape) =
                    tree_fold(&cur, Some(d), &bb0_names, &body_ops, fast_combine, ctx, env)?;
                data = folded;
                shape = fshape;
            }
            // Squeeze reduced axes (rightmost first so earlier removals don't shift).
            for &d in sorted.iter().rev() {
                shape.remove(d);
            }
            if shape.is_empty() {
                Value::Scalar(Scalar::F32(data[0]))
            } else {
                Value::Tile(Tile::compute(data, tile.dtype, shape))
            }
        }
    };

    // Combine the reduced value with the `outs` initial accumulator, mirroring
    // Python's final `_run_combiner(reduced, outs_tile)`. MLIR `linalg.reduce`
    // semantics: `outs` is the INITIAL accumulator value, so the result is
    // `combiner(reduce(ins), outs)`. Python (`ktir_cpu`) folds it unconditionally
    // whenever the `outs` operand is a Tile of the reduced shape, and so do we —
    // there is NO identity-only guard. The `test_reduce_folds_outs_init` unit test
    // pins this: `sum([1,2,3,4])` with `outs` init `100` is `110`, not `10`.
    //
    // Every reduce the real model and the conformance suite emit splats an identity
    // accumulator (`0` for addf, `1` for mulf, `-inf` for max, `+inf` for min) via a
    // fresh `linalg.fill` / `tensor.splat`, so the fold is a no-op there
    // (`combiner(reduced, identity) == reduced`). The RESIDENT GPU executor
    // re-materializes that fill on every reduce against a freshly per-pass-zeroed
    // HBM (`zero_non_sources`) and a fresh per-execution value context, so the
    // accumulator it folds is always the identity the program wrote — never a stale
    // shared partial sum. Folding unconditionally is therefore both oracle-faithful
    // and golden-bit-exact on every path (fresh-context interpreter, harness, and
    // resident), with no special-casing.
    if let Some(Attr::Str(outs_var)) = op.attributes.get("outs_var")
        && let Ok(Value::Tile(outs_tile)) = ctx.get_value(outs_var)
    {
        let outs_tile = outs_tile.clone();
        let reduced_tile = match &reduced_value {
            Value::Tile(t) => t.clone(),
            Value::Scalar(Scalar::F32(s)) => Tile::compute(vec![*s], tile.dtype, vec![]),
            other => {
                return Err(format!(
                    "linalg.reduce: unexpected reduced value {other:?} for outs combine"
                ));
            }
        };
        if reduced_tile.shape == outs_tile.shape {
            let combined = run_combiner(&bb0_names, &body_ops, reduced_tile, outs_tile, ctx, env)?;
            reduced_value = match combined {
                Value::Tile(t) if t.shape.is_empty() => Value::Scalar(Scalar::F32(t.as_f32()[0])),
                other => other,
            };
        }
    }

    // MLIR writes the result back into the outs buffer; downstream ops may
    // reference it by the outs SSA name. Bind both so either reference resolves.
    if let Some(Attr::Str(outs_var)) = op.attributes.get("outs_var") {
        ctx.set_value(outs_var, reduced_value.clone());
    }

    Ok(Some(reduced_value))
}

/// f32 combiner for a recognized **commutative + associative** reduce op, so the
/// pairwise tree fold can combine the two halves directly instead of executing
/// the combiner region op-by-op (slice -> build two Tiles -> dispatch the op ->
/// extract the result, every round). Only order-insensitive ops qualify: the
/// fold pairs halves and the operand order within a pair must not matter. `subf`
/// and custom multi-op regions fall back to the faithful region path.
///
/// Each closure matches the corresponding `arith` handler exactly (incl. the
/// NaN-propagating `maximumf`/`minimumf` vs the `*numf` fmax/fmin variants); the
/// caller rounds the result to the tile dtype each round, mirroring how the
/// region path's `Tile::compute` rounds after every combine.
fn reduce_combiner(op_name: &str) -> Option<fn(f32, f32) -> f32> {
    Some(match op_name {
        "arith.addf" => |a, b| a + b,
        "arith.mulf" => |a, b| a * b,
        "arith.maxnumf" => f32::max,
        "arith.minnumf" => f32::min,
        "arith.maximumf" => |a: f32, b: f32| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a >= b {
                a
            } else {
                b
            }
        },
        "arith.minimumf" => |a: f32, b: f32| {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a <= b {
                a
            } else {
                b
            }
        },
        _ => return None,
    })
}

/// Reduce `tile` along `dim` by folding the combiner region pairwise.
///
/// Splits the reduced axis in half, combines the two halves with one
/// *vectorised* region call, and repeats — `ceil(log2(N))` region executions
/// rather than `N` sequential folds. Odd lengths carry the unpaired slice into
/// the next round. Returns `(data, shape)` with extent 1 along `dim`.
fn tree_fold(
    tile: &Tile,
    dim: Option<usize>,
    bb0_names: &[String],
    body_ops: &[Operation],
    fast_combine: Option<fn(f32, f32) -> f32>,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    // Reduce to scalar (no dim) -> flatten everything onto one axis first.
    let (mut acc, mut shape, axis) = match dim {
        None => (tile.as_f32().to_vec(), vec![tile.len()], 0usize),
        Some(d) => {
            if d >= tile.shape.len() {
                return Err(format!(
                    "linalg.reduce: dim {d} out of range for shape {:?}",
                    tile.shape
                ));
            }
            (tile.as_f32().to_vec(), tile.shape.clone(), d)
        }
    };

    // Fast path: a recognized commutative+associative combiner folds directly in
    // the f32 buffer with strided indexing — no per-round `slice_along` Tiles, no
    // region dispatch. Bit-identical to the region path below (same pairwise tree
    // order, same per-round dtype rounding).
    if let Some(f) = fast_combine {
        return Ok(fast_tree_fold(acc, shape, axis, tile.dtype, f));
    }

    let mut n = shape[axis];
    while n > 1 {
        let half = n / 2;
        let (left_data, left_shape) = slice_along(&acc, &shape, axis, 0, half);
        let (right_data, right_shape) = slice_along(&acc, &shape, axis, half, 2 * half);

        let combined = run_combiner(
            bb0_names,
            body_ops,
            Tile::compute(left_data, tile.dtype, left_shape.clone()),
            Tile::compute(right_data, tile.dtype, right_shape),
            ctx,
            env,
        )?;
        let mut combined_data = match combined {
            Value::Tile(t) => t.as_f32().to_vec(),
            Value::Scalar(s) => vec![as_f32(&Value::Scalar(s), "reduce combiner")?],
            other => {
                return Err(format!(
                    "linalg.reduce: combiner yielded {other:?}, expected tile/scalar"
                ));
            }
        };
        let mut combined_shape = left_shape;

        if n % 2 == 1 {
            // Odd: concatenate the leftover slice along the reduced axis.
            let (tail_data, _tail_shape) = slice_along(&acc, &shape, axis, 2 * half, n);
            combined_data = concat_along(&combined_data, &combined_shape, &tail_data, axis);
            combined_shape[axis] += 1;
        }

        acc = combined_data;
        shape = combined_shape;
        n = shape[axis];
    }

    Ok((acc, shape))
}

/// Direct strided implementation of the pairwise tree fold for a known
/// commutative+associative combiner `f`. Mirrors `tree_fold`'s region path
/// exactly — same halving, same odd-length carry, same per-round rounding to
/// `dtype` — but combines straight from the flat buffer (`acc[outer, i, inner]`
/// with `i+half`) instead of materializing two slice Tiles and dispatching the
/// combiner op each round. This is what makes `linalg.reduce` cheap on the hot
/// `addf`-sum path the real model emits.
fn fast_tree_fold(
    mut acc: Vec<f32>,
    mut shape: Vec<usize>,
    axis: usize,
    dtype: DType,
    f: fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<usize>) {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let mut n = shape[axis];
    while n > 1 {
        let half = n / 2;
        let new_n = n - half; // ceil(n/2): `half` combined pairs + (odd ? 1 carry : 0)
        let mut next = vec![0.0f32; outer * new_n * inner];
        for o in 0..outer {
            for i in 0..half {
                let lhs = (o * n + i) * inner;
                let rhs = (o * n + i + half) * inner;
                let dst = (o * new_n + i) * inner;
                for k in 0..inner {
                    next[dst + k] = f(acc[lhs + k], acc[rhs + k]);
                }
            }
            if n % 2 == 1 {
                // Carry the unpaired last axis-slice (index n-1 == 2*half) into
                // position `half`, matching the region path's concat.
                let src = (o * n + (n - 1)) * inner;
                let dst = (o * new_n + half) * inner;
                next[dst..dst + inner].copy_from_slice(&acc[src..src + inner]);
            }
        }
        crate::codec::round_to_dtype(&mut next, dtype);
        acc = next;
        n = new_n;
        shape[axis] = n;
    }
    (acc, shape)
}

/// Run the combiner region once on two equal-shaped operands, returning the
/// yielded value. Binds the bb0 args in an isolated scope and dispatches the
/// region via `execute_region` (so each combiner op fires through the normal
/// driver and is charged latency under its own category).
fn run_combiner(
    bb0_names: &[String],
    body_ops: &[Operation],
    lhs: Tile,
    rhs: Tile,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Value, String> {
    run_region(
        ctx,
        env,
        |ctx| {
            if let Some(name) = bb0_names.first() {
                ctx.set_value(name, Value::Tile(lhs.clone()));
            }
            if let Some(name) = bb0_names.get(1) {
                ctx.set_value(name, Value::Tile(rhs.clone()));
            }
            Ok(())
        },
        body_ops,
    )?
    .ok_or_else(|| "linalg.reduce: combiner region did not yield".to_string())
}

// ===========================================================================
// generic / index / yield
// ===========================================================================

/// `%r = linalg.generic ins(...) outs(%init) { ^bb0(...): <body> }`.
///
/// Broadcasts each input to the outs iteration space per its indexing map
/// (inserting size-1 axes for missing dims), binds the bb0 block-arg names, then
/// runs the region body once over the full arrays and broadcasts the yielded
/// value back to the outs shape.
fn generic(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let n_ins = match op.attributes.get("n_ins") {
        Some(Attr::Int(n)) => *n as usize,
        _ => 0,
    };
    let indexing_maps = indexing_maps_attr(op);

    // Snapshot input values (clone to drop the borrow on ctx before we mutate).
    let ins_vals: Vec<Value> = (0..n_ins)
        .map(|i| ctx.get_value(&op.operands[i]).cloned())
        .collect::<Result<_, _>>()?;
    let outs_val = expect_tile(ctx.get_value(&op.operands[n_ins])?, "linalg.generic outs")?.clone();
    let out_shape = outs_val.shape.clone();
    let out_ndim = out_shape.len();
    let out_dtype = outs_val.dtype;

    let (bb0_names, body_ops) = resolve_region_body(op);
    if bb0_names.is_empty() {
        return Err("linalg.generic: cannot determine bb0 argument names".into());
    }

    let result = run_region(
        ctx,
        env,
        |ctx| {
            // Store output shape so linalg.index can build index arrays.
            ctx.set_value(
                SHAPE_KEY,
                Value::Tuple(out_shape.iter().map(|&d| Value::Index(d as i64)).collect()),
            );

            // Broadcast each input to the iteration space and bind to its bb0 arg.
            for (i, val) in ins_vals.iter().enumerate() {
                let arg_val = match val {
                    Value::Tile(t) => {
                        let imap = indexing_maps.get(i).cloned().unwrap_or_default();
                        // With an explicit indexing map, insert size-1 axes for
                        // any output dim the map does not reference (Python's
                        // np.expand_dims loop). With no map, fall back to plain
                        // right-aligned NumPy broadcasting against out_shape.
                        let mut shape: Vec<usize> = t.shape.clone();
                        if !imap.is_empty() {
                            for d in 0..out_ndim {
                                if !imap.contains(&d) && d <= shape.len() {
                                    shape.insert(d, 1);
                                }
                            }
                        }
                        let data = broadcast_to(&t.as_f32(), &shape, &out_shape).ok_or_else(|| {
                            format!(
                                "linalg.generic: cannot broadcast input {i} {shape:?} to {out_shape:?}"
                            )
                        })?;
                        Value::Tile(Tile::compute(data, t.dtype, out_shape.clone()))
                    }
                    other => other.clone(),
                };
                if let Some(name) = bb0_names.get(i) {
                    ctx.set_value(name, arg_val);
                }
            }

            // Bind the outs bb0 arg — in MLIR semantics outs is the initial value
            // of the output block argument.
            if n_ins < bb0_names.len() {
                ctx.set_value(
                    &bb0_names[n_ins],
                    Value::Tile(Tile::compute(
                        outs_val.as_f32().to_vec(),
                        outs_val.dtype,
                        out_shape.clone(),
                    )),
                );
            }
            Ok(())
        },
        &body_ops,
    )?;

    // Broadcast the yielded value back to the outs shape.
    let out_tile = match result {
        Some(Value::Tile(t)) => {
            let data = broadcast_to(&t.as_f32(), &t.shape, &out_shape).ok_or_else(|| {
                format!(
                    "linalg.generic: yield shape {:?} not broadcastable to {out_shape:?}",
                    t.shape
                )
            })?;
            Tile::compute(data, out_dtype, out_shape)
        }
        Some(other) => {
            let v = as_f32(&other, "linalg.generic yield")?;
            Tile::compute(vec![v; out_shape.iter().product()], out_dtype, out_shape)
        }
        None => return Err("linalg.generic: region did not yield".into()),
    };
    Ok(Some(Value::Tile(out_tile)))
}

/// `%r = linalg.index <dim>` — a broadcasting index array for iteration `dim`.
fn index(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let dim = match op.attributes.get("dim") {
        Some(Attr::Int(d)) => *d as usize,
        _ => 0,
    };
    let out_shape = match ctx.get_value(SHAPE_KEY)? {
        Value::Tuple(items) => items
            .iter()
            .map(|v| match v {
                Value::Index(i) => Ok(*i as usize),
                other => Err(format!("linalg.index: bad shape entry {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        other => {
            return Err(format!(
                "linalg.index: {SHAPE_KEY} is {other:?}, expected shape tuple"
            ));
        }
    };
    if dim >= out_shape.len() {
        return Err(format!(
            "linalg.index: dim {dim} out of range for shape {out_shape:?}"
        ));
    }
    // arange(out_shape[dim]) reshaped to [1,...,out_shape[dim],...,1].
    let mut shape = vec![1usize; out_shape.len()];
    shape[dim] = out_shape[dim];
    let arange: Vec<f32> = (0..out_shape[dim]).map(|i| i as f32).collect();
    Ok(Some(Value::Tile(Tile::compute(arange, DType::I32, shape))))
}

/// `linalg.yield %v` — park the yielded value under [`YIELD_KEY`] in the current
/// scope so the enclosing region driver can recover it (see [`run_region`]).
fn yield_op(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if let Some(name) = op.operands.first() {
        let v = ctx.get_value(name)?.clone();
        ctx.set_value(YIELD_KEY, v);
    }
    Ok(None)
}

// ===========================================================================
// region helpers (yield threading)
// ===========================================================================

/// Run `body_ops` in a fresh scope after `bind` populates the block args, then
/// recover the value parked by `linalg.yield`. Owns the `push_scope` /
/// `pop_scope` pair so callers cannot leak a scope on error.
fn run_region(
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
    bind: impl FnOnce(&mut CoreContext) -> Result<(), String>,
    body_ops: &[Operation],
) -> Result<Option<Value>, String> {
    ctx.push_scope();
    let outcome = (|| {
        bind(ctx)?;
        execute_region(body_ops, ctx, env)?;
        // Recover the yielded value (if any) before the scope is torn down.
        Ok(if ctx.has_value(YIELD_KEY) {
            Some(ctx.get_value(YIELD_KEY)?.clone())
        } else {
            None
        })
    })();
    ctx.pop_scope();
    outcome
}

/// Resolve a linalg op's region into `(bb0_names, body_ops)`.
///
/// Block-argument names are found in priority order, mirroring Python:
///   1. a `bb0_names` string-list attribute (mlir_frontend / `^bb0(...)` path);
///   2. the operand names of the region's first non-yield op (inline-block form,
///      e.g. `linalg.reduce`'s `(%in, %out) { %s = addf %in, %out }`).
///
/// Returns `([], [])` when the op has no region (reduce shorthand synthesizes
/// one). A synthetic `region.bb0_args` op, if present, is dropped from the body.
fn resolve_region_body(op: &Operation) -> (Vec<String>, Vec<Operation>) {
    let region: &[Operation] = op.regions.first().map(|r| r.as_slice()).unwrap_or(&[]);
    let body_ops: Vec<Operation> = region
        .iter()
        .filter(|o| o.op_type != "region.bb0_args")
        .cloned()
        .collect();

    if let Some(Attr::StrList(names)) = op.attributes.get("bb0_names") {
        return (names.clone(), body_ops);
    }
    // The synthetic `region.bb0_args` op (parsed from the `^bb0(...)` label)
    // carries the canonical block-arg names — prefer it over guessing from the
    // first body op's operands (which is empty when the body opens with an
    // operand-less op like `linalg.index`).
    if let Some(Attr::StrList(names)) = region
        .iter()
        .find(|o| o.op_type == "region.bb0_args")
        .and_then(|o| o.attributes.get("names"))
    {
        return (names.clone(), body_ops);
    }
    if let Some(first) = body_ops.first() {
        return (first.operands.clone(), body_ops);
    }
    (Vec::new(), Vec::new())
}

// ===========================================================================
// shape / ndarray helpers (the NumPy ops Python gets for free)
// ===========================================================================

/// Row-major strides for `shape` (element strides, not bytes).
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a flat row-major index into a multi-index for `shape`.
/// Only the unit tests (cross-checking the block-copy slice/concat) still use
/// this — the hot paths walk contiguous spans, not per-element multi-indices.
#[cfg(test)]
fn unravel(mut lin: usize, shape: &[usize]) -> Vec<usize> {
    let strides = row_major_strides(shape);
    let mut idx = vec![0usize; shape.len()];
    for (k, &s) in strides.iter().enumerate() {
        idx[k] = lin / s;
        lin %= s;
    }
    idx
}

/// Broadcast `data` (logical `from_shape`) to `to_shape`, NumPy rules: right-
/// aligned by rank, each axis must match or be 1. Returns the expanded flat
/// data, or `None` if incompatible.
fn broadcast_to(data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> Option<Vec<f32>> {
    if from_shape.len() > to_shape.len() {
        return None;
    }
    // Right-align ranks by left-padding from_shape with leading 1s.
    let pad = to_shape.len() - from_shape.len();
    let mut src_shape = vec![1usize; pad];
    src_shape.extend_from_slice(from_shape);

    for (s, t) in src_shape.iter().zip(to_shape) {
        if *s != *t && *s != 1 {
            return None;
        }
    }

    // Fast path: already the exact shape.
    if src_shape == to_shape {
        return Some(data.to_vec());
    }

    let src_strides = row_major_strides(&src_shape);
    let total: usize = to_shape.iter().product();
    let mut out = vec![0.0f32; total];
    let rank = to_shape.len();
    // Source-offset increment for a +1 step along each output axis: the src
    // stride, or 0 where the src axis is broadcast (size 1). Walk the output in
    // row-major order, maintaining `src` via an odometer — no per-element
    // `unravel` (two Vec allocs each, the broadcast host hot spot).
    let steps: Vec<usize> = (0..rank)
        .map(|k| if src_shape[k] == 1 { 0 } else { src_strides[k] })
        .collect();
    let mut idx = vec![0usize; rank];
    let mut src = 0usize;
    for slot in out.iter_mut() {
        *slot = data[src];
        let mut k = rank;
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            src += steps[k];
            if idx[k] < to_shape[k] {
                break;
            }
            idx[k] = 0;
            src -= steps[k] * to_shape[k];
        }
    }
    Some(out)
}

/// Slice `data` (logical `shape`) along `axis` for `[lo, hi)`. Returns the
/// sliced flat data and its shape.
///
/// Row-major slicing along one axis is a sequence of contiguous block copies:
/// everything to the right of `axis` (the `inner` block) stays contiguous in the
/// source, so for each `(outer, j)` pair we `copy_from_slice` an `inner`-element
/// run rather than gathering element-by-element (no per-element `unravel` + dot).
fn slice_along(
    data: &[f32],
    shape: &[usize],
    axis: usize,
    lo: usize,
    hi: usize,
) -> (Vec<f32>, Vec<usize>) {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = hi - lo;
    let total: usize = out_shape.iter().product();

    let src_axis = shape[axis];
    let out_axis = hi - lo;
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();

    let mut out = vec![0.0f32; total];
    let src_row = src_axis * inner; // one outer-slab in the source
    let dst_row = out_axis * inner; // one outer-slab in the output
    for o in 0..outer {
        let src_base = o * src_row + lo * inner;
        let dst_base = o * dst_row;
        let span = out_axis * inner;
        out[dst_base..dst_base + span].copy_from_slice(&data[src_base..src_base + span]);
    }
    (out, out_shape)
}

/// Concatenate `a` and `b` along `axis`. `a_shape` is the shape of `a`; `b` is
/// assumed to share that shape except along `axis` (the leftover odd slice the
/// tree fold carries). Result extent along `axis` is `a_shape[axis] + b_extent`.
fn concat_along(a: &[f32], a_shape: &[usize], b: &[f32], axis: usize) -> Vec<f32> {
    // Recover b's extent along `axis` from its element count and a's other axes.
    let outer: usize = a_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .product();
    let b_extent = b.len().checked_div(outer).unwrap_or(0);

    let a_axis = a_shape[axis];
    let out_axis = a_axis + b_extent;
    let mut out_shape = a_shape.to_vec();
    out_shape[axis] = out_axis;
    let total: usize = out_shape.iter().product();

    // Row-major concat along one axis is a per-outer-slab interleave of two
    // contiguous blocks: a's `a_axis*inner` run followed by b's `b_extent*inner`
    // run. Block-copy each (no per-element `unravel` + dot).
    let inner: usize = a_shape[axis + 1..].iter().product();
    let outer: usize = a_shape[..axis].iter().product();
    let a_block = a_axis * inner;
    let b_block = b_extent * inner;
    let dst_row = out_axis * inner;

    let mut out = vec![0.0f32; total];
    for o in 0..outer {
        let dst_base = o * dst_row;
        let a_base = o * a_block;
        out[dst_base..dst_base + a_block].copy_from_slice(&a[a_base..a_base + a_block]);
        let b_base = o * b_block;
        out[dst_base + a_block..dst_base + a_block + b_block]
            .copy_from_slice(&b[b_base..b_base + b_block]);
    }
    out
}

// ===========================================================================
// value / attribute helpers
// ===========================================================================

fn expect_tile<'a>(v: &'a Value, ctx: &str) -> Result<&'a Tile, String> {
    match v {
        Value::Tile(t) => Ok(t),
        other => Err(format!("{ctx}: expected Tile, got {other:?}")),
    }
}

/// Coerce a scalar-ish value to f32. Mirrors Python's `float(scalar)`.
fn as_f32(v: &Value, ctx: &str) -> Result<f32, String> {
    match v {
        Value::Scalar(Scalar::F32(x)) => Ok(*x),
        Value::Scalar(Scalar::I32(x)) => Ok(*x as f32),
        Value::Scalar(Scalar::I64(x)) => Ok(*x as f32),
        Value::Scalar(Scalar::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Index(i) => Ok(*i as f32),
        other => Err(format!("{ctx}: expected scalar, got {other:?}")),
    }
}

fn int_list_attr<'a>(op: &'a Operation, key: &str) -> Option<&'a Vec<i64>> {
    match op.attributes.get(key) {
        Some(Attr::IntList(v)) => Some(v),
        _ => None,
    }
}

/// Read `indexing_maps`: the Python parser stores, per input, the list of output
/// dims its affine map references. The closed `Attr` enum has no nested-list
/// variant, so the integrator supplies these via `Attr::StrList` of
/// comma-separated dim lists (e.g. `"0,1"`) per input, or omits the attribute —
/// in which case handlers fall back to NumPy right-aligned broadcasting, which
/// covers the common elementwise / scalar-broadcast cases the Python tests use.
fn indexing_maps_attr(op: &Operation) -> Vec<Vec<usize>> {
    match op.attributes.get("indexing_maps") {
        // Parsed `[affine_map<...>, ...]` from MLIR text (`Attr::AffineMapList`):
        // take each map's referenced output dims. `result_dims` yields `None`
        // for non-pure-dim maps; we treat those as "no projection" here.
        Some(Attr::AffineMapList(maps)) => maps
            .iter()
            .map(|m| m.result_dims().unwrap_or_default())
            .collect(),
        // Legacy shorthand: `Attr::StrList` of comma-separated dim indices
        // (e.g. `"0,1"`) the Python integrator supplies directly — not MLIR
        // `affine_map<>` syntax, just the output-dim list per input.
        Some(Attr::StrList(maps)) => maps
            .iter()
            .map(|m| {
                m.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    fn tile(data: Vec<f32>, shape: Vec<usize>) -> Value {
        Value::Tile(Tile::compute(data, DType::F32, shape))
    }

    fn get_tile(ctx: &CoreContext, name: &str) -> Tile {
        match ctx.get_value(name).unwrap() {
            Value::Tile(t) => t.clone(),
            other => panic!("expected tile, got {other:?}"),
        }
    }

    // --- fill -------------------------------------------------------------

    #[test]
    fn fill_broadcasts_scalar() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Scalar(Scalar::F32(7.0)));
        ctx.set_value("%init", tile(vec![0.0; 6], vec![2, 3]));
        run(
            &[Operation::new(Some("%r"), "linalg.fill", &["%s", "%init"])],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.as_f32().to_vec(), vec![7.0; 6]);
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn fill_from_index_scalar() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Index(3));
        ctx.set_value("%init", tile(vec![0.0; 2], vec![2]));
        run(
            &[Operation::new(Some("%r"), "linalg.fill", &["%s", "%init"])],
            &mut ctx,
        )
        .unwrap();
        assert_eq!(get_tile(&ctx, "%r").as_f32().to_vec(), vec![3.0, 3.0]);
    }

    // --- transpose --------------------------------------------------------

    #[test]
    fn transpose_2d() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] -> transpose [1,0] -> [[1,4],[2,5],[3,6]]
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        ctx.set_value("%y", tile(vec![0.0; 6], vec![3, 2]));
        let op = Operation::new(Some("%r"), "linalg.transpose", &["%x", "%y"])
            .with_attr("permutation", Attr::IntList(vec![1, 0]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.as_f32().to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_identity_permutation() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%y", tile(vec![0.0; 4], vec![2, 2]));
        let op = Operation::new(Some("%r"), "linalg.transpose", &["%x", "%y"])
            .with_attr("permutation", Attr::IntList(vec![0, 1]));
        run(&[op], &mut ctx).unwrap();
        assert_eq!(
            get_tile(&ctx, "%r").as_f32().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn transpose_3d_permutation() {
        let mut ctx = single_core_context();
        // shape [2,1,3], permute [1,2,0] -> shape [1,3,2]; out[a,b,c]=in[c,a,b].
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        ctx.set_value("%x", tile(data, vec![2, 1, 3]));
        ctx.set_value("%y", tile(vec![0.0; 6], vec![1, 3, 2]));
        let op = Operation::new(Some("%r"), "linalg.transpose", &["%x", "%y"])
            .with_attr("permutation", Attr::IntList(vec![1, 2, 0]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![1, 3, 2]);
        // in[i,j,k] at i*3+k (j=0). out flat: (0,0,0)->in[0,0,0]=0 (0,0,1)->in[1,0,0]=3
        // (0,1,0)->in[0,0,1]=1 (0,1,1)->in[1,0,1]=4 (0,2,0)->in[0,0,2]=2 (0,2,1)->in[1,0,2]=5
        assert_eq!(t.as_f32().to_vec(), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    // --- broadcast --------------------------------------------------------

    #[test]
    fn broadcast_along_dim() {
        let mut ctx = single_core_context();
        // ins [3], broadcast dim 1 -> expand to [3,1] -> [3,4]: rows constant.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.set_value("%y", tile(vec![0.0; 12], vec![3, 4]));
        let op = Operation::new(Some("%r"), "linalg.broadcast", &["%x", "%y"])
            .with_attr("dimensions", Attr::IntList(vec![1]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(
            t.as_f32().to_vec(),
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn broadcast_leading_dim() {
        let mut ctx = single_core_context();
        // ins [4], broadcast dim 0 -> [1,4] -> [3,4]: each row identical.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        ctx.set_value("%y", tile(vec![0.0; 12], vec![3, 4]));
        let op = Operation::new(Some("%r"), "linalg.broadcast", &["%x", "%y"])
            .with_attr("dimensions", Attr::IntList(vec![0]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(
            t.as_f32().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    // --- matmul -----------------------------------------------------------

    #[test]
    fn matmul_plain() {
        let mut ctx = single_core_context();
        // A=[[1,2],[3,4]], B=[[5,6],[7,8]] -> [[19,22],[43,50]]
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]));
        run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.as_f32().to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_accumulates_outs() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]));
        ctx.set_value("%c", tile(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]));
        run(
            &[Operation::new(
                Some("%r"),
                "linalg.matmul",
                &["%a", "%b", "%c"],
            )],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.as_f32().to_vec(), vec![20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn matmul_nonsquare() {
        let mut ctx = single_core_context();
        // A [2x3], B [3x2] -> [2x2]
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        ctx.set_value(
            "%b",
            tile(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
        );
        run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2]);
        // row0: [58, 64], row1: [139, 154]
        assert_eq!(t.as_f32().to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0], vec![1, 2]));
        ctx.set_value("%b", tile(vec![1.0, 2.0, 3.0], vec![3, 1]));
        let err = run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])],
            &mut ctx,
        )
        .unwrap_err();
        assert!(err.contains("inner dims disagree"));
    }

    /// M=1 routes through the GEMV fast path (`gemm` -> `gemv`), and must produce
    /// the same result the GEMM would. A [1x3] · B [3x2] -> [1x2].
    #[test]
    fn matmul_m1_routes_through_gemv() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0], vec![1, 3]));
        ctx.set_value(
            "%b",
            tile(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
        );
        run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![1, 2]);
        // [1,2,3]·B = [1·7+2·9+3·11, 1·8+2·10+3·12] = [58, 64].
        assert_eq!(t.as_f32().to_vec(), vec![58.0, 64.0]);
    }

    /// M=1 transpose-B routes through the GEMV-bt fast path and matches the GEMM.
    /// a [1x2] · B[n,k]=[[5,7],[6,8]]ᵀ -> [1x2].
    #[test]
    fn matmul_bt_m1_routes_through_gemv() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0], vec![1, 2]));
        ctx.set_value("%b", tile(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]));
        run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])
                .with_attr("indexing_maps", tb_maps())],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![1, 2]);
        // y[j] = Σ_k a[k]·B[j,k]: [1·5+2·7, 1·6+2·8] = [19, 22].
        assert_eq!(t.as_f32().to_vec(), vec![19.0, 22.0]);
    }

    #[test]
    fn matmul_bt_basic() {
        let mut ctx = single_core_context();
        // A=[[1,2],[3,4]]. B stored [n,k]=[[5,7],[6,8]] (= Bᵀ of [[5,6],[7,8]]).
        // A·Bᵀ contracts the LAST axis: C[m,n]=Σ_k A[m,k]·B[n,k] = [[19,22],[43,50]].
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]));
        run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])
                .with_attr("indexing_maps", tb_maps())],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.as_f32().to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_bt_nonsquare_matches_plain() {
        let mut ctx = single_core_context();
        // A [2x3]; B stored [n,k]=[2x3]=[[7,9,11],[8,10,12]] (= Bᵀ of the [3x2] in
        // matmul_nonsquare). A·Bᵀ must equal that plain result [58,64,139,154].
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        ctx.set_value(
            "%b",
            tile(vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0], vec![2, 3]),
        );
        run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])
                .with_attr("indexing_maps", tb_maps())],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.as_f32().to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_bt_accumulates_outs() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]));
        ctx.set_value("%c", tile(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]));
        run(
            &[
                Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b", "%c"])
                    .with_attr("indexing_maps", tb_maps()),
            ],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.as_f32().to_vec(), vec![20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn matmul_bt_rejects_contraction_mismatch() {
        let mut ctx = single_core_context();
        // A [1x2] (k=2), B [n,k]=[1x3] (k=3) — last axes disagree.
        ctx.set_value("%a", tile(vec![1.0, 2.0], vec![1, 2]));
        ctx.set_value("%b", tile(vec![1.0, 2.0, 3.0], vec![1, 3]));
        let err = run(
            &[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])
                .with_attr("indexing_maps", tb_maps())],
            &mut ctx,
        )
        .unwrap_err();
        assert!(err.contains("contraction dims disagree"));
    }

    // --- linalg.matmul + indexing_maps (upstream transpose encoding) ---------

    /// Build a matmul `indexing_maps` AffineMapList over `(m, n, k)`.
    fn imaps(a: &str, b: &str, c: &str) -> Attr {
        let p = |s: &str| crate::parser_ast::parse_affine_map(s).unwrap();
        Attr::AffineMapList(vec![p(a), p(b), p(c)])
    }

    /// Transpose-B `indexing_maps`: A `[m,k]`, B `[n,k]`, C `[m,n]`.
    fn tb_maps() -> Attr {
        imaps(
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d1, d2)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        )
    }

    /// Identity-layout `indexing_maps` (B map `(d2, d1)`) is a plain `A·B`.
    #[test]
    fn matmul_indexing_maps_normal_is_plain() {
        let mut ctx = single_core_context();
        // A=[[1,2,3]] [1x3], B=[[7,8],[9,10],[11,12]] [3x2] -> [58, 64].
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0], vec![1, 3]));
        ctx.set_value(
            "%b",
            tile(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]),
        );
        let op = Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"]).with_attr(
            "indexing_maps",
            imaps(
                "affine_map<(d0, d1, d2) -> (d0, d2)>", // A: [m, k]
                "affine_map<(d0, d1, d2) -> (d2, d1)>", // B: [k, n]  (normal)
                "affine_map<(d0, d1, d2) -> (d0, d1)>", // C: [m, n]
            ),
        );
        run(&[op], &mut ctx).unwrap();
        assert_eq!(get_tile(&ctx, "%r").as_f32().to_vec(), vec![58.0, 64.0]);
    }

    /// Transpose-A `indexing_maps` (A map `(d2, d0)`) is rejected, not guessed.
    #[test]
    fn matmul_indexing_maps_rejects_transpose_a() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]));
        let op = Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"]).with_attr(
            "indexing_maps",
            imaps(
                "affine_map<(d0, d1, d2) -> (d2, d0)>", // A: [k, m]  (transpose-A)
                "affine_map<(d0, d1, d2) -> (d2, d1)>",
                "affine_map<(d0, d1, d2) -> (d0, d1)>",
            ),
        );
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("transpose-A"), "got: {err}");
    }

    #[test]
    fn classify_matmul_maps_detects_layouts() {
        let p = |s: &str| crate::parser_ast::parse_affine_map(s).unwrap();
        let a = p("affine_map<(d0, d1, d2) -> (d0, d2)>");
        let at = p("affine_map<(d0, d1, d2) -> (d2, d0)>");
        let b = p("affine_map<(d0, d1, d2) -> (d2, d1)>");
        let bt = p("affine_map<(d0, d1, d2) -> (d1, d2)>");
        let c = p("affine_map<(d0, d1, d2) -> (d0, d1)>");
        assert_eq!(
            classify_matmul_maps(&[a.clone(), b.clone(), c.clone()]),
            Ok((false, false))
        );
        assert_eq!(
            classify_matmul_maps(&[a.clone(), bt.clone(), c.clone()]),
            Ok((false, true))
        );
        assert_eq!(
            classify_matmul_maps(&[at.clone(), b.clone(), c.clone()]),
            Ok((true, false))
        );
        // Wrong arity and a bad C map both error.
        assert!(classify_matmul_maps(&[a.clone(), b.clone()]).is_err());
        assert!(classify_matmul_maps(&[a, bt, p("affine_map<(d0, d1, d2) -> (d1, d0)>")]).is_err());
    }

    /// A map referencing an out-of-range dim (`d3` with only 3 dims) must error
    /// gracefully via classify, not panic in the affine linearizer.
    #[test]
    fn classify_matmul_maps_out_of_range_dim_errors_not_panics() {
        use crate::affine::{AffineExpr, AffineMap};
        let bad = AffineMap {
            num_dims: 3,
            num_syms: 0,
            exprs: vec![AffineExpr::Dim(3), AffineExpr::Dim(0)],
        };
        let c =
            crate::parser_ast::parse_affine_map("affine_map<(d0, d1, d2) -> (d0, d1)>").unwrap();
        let b =
            crate::parser_ast::parse_affine_map("affine_map<(d0, d1, d2) -> (d2, d1)>").unwrap();
        assert_eq!(bad.result_dims(), None); // no panic
        assert!(classify_matmul_maps(&[bad, b, c]).is_err());
    }

    /// `indexing_maps` present but not an affine-map list (e.g. the legacy
    /// StrList shorthand) is rejected, not silently treated as plain `A·B`.
    #[test]
    fn matmul_rejects_non_affine_indexing_maps() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]));
        let op = Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"]).with_attr(
            "indexing_maps",
            Attr::StrList(vec!["0,2".into(), "1,2".into(), "0,1".into()]),
        );
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("affine-map list"), "got: {err}");
    }

    #[test]
    fn batch_matmul_two_batches() {
        let mut ctx = single_core_context();
        // batch0: [[1,2],[3,4]] @ I = same. batch1: I @ [[5,6],[7,8]] = same.
        ctx.set_value(
            "%a",
            tile(vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], vec![2, 2, 2]),
        );
        ctx.set_value(
            "%b",
            tile(vec![1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]),
        );
        run(
            &[Operation::new(
                Some("%r"),
                "linalg.batch_matmul",
                &["%a", "%b"],
            )],
            &mut ctx,
        )
        .unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2, 2]);
        assert_eq!(
            t.as_f32().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    // --- reduce -----------------------------------------------------------

    fn addf_combiner_region() -> Vec<Operation> {
        // (%in, %out) { %s = arith.addf %in, %out ; linalg.yield %s }
        vec![
            Operation::new(Some("%s"), "arith.addf", &["%in", "%out"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]
    }

    #[test]
    fn reduce_all_to_scalar_explicit_region() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]);
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected scalar 10.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_all_odd_length() {
        let mut ctx = single_core_context();
        // 5 elements exercises the odd-carry path in the tree fold.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]);
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 15.0),
            other => panic!("expected scalar 15.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_along_dim_keeps_other_axis() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] reduce dim=1 -> [6, 15]
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        let mut op =
            Operation::new(Some("%r"), "linalg.reduce", &["%x"]).with_attr("dim", Attr::Int(1));
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.as_f32().to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_along_dim0() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] reduce dim=0 -> [5,7,9]
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        let mut op =
            Operation::new(Some("%r"), "linalg.reduce", &["%x"]).with_attr("dim", Attr::Int(0));
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.as_f32().to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_dim1_odd_extent() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] dim=1 odd extent 3 -> [6,15] exercises odd carry on a 2-D fold.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        let mut op =
            Operation::new(Some("%r"), "linalg.reduce", &["%x"]).with_attr("dim", Attr::Int(1));
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        assert_eq!(get_tile(&ctx, "%r").as_f32().to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_shorthand_synthesizes_region() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![2.0, 4.0, 6.0, 8.0], vec![4]));
        // Shorthand: reduce_fn attribute, no region.
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()));
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 20.0),
            other => panic!("expected 20.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_mul_combiner() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.mulf".into()));
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 24.0),
            other => panic!("expected 24.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_binds_outs_var() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()))
            .with_attr("outs_var", Attr::Str("%acc".into()));
        run(&[op], &mut ctx).unwrap();
        // Both %r and %acc resolve to the reduced scalar.
        match ctx.get_value("%acc").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 6.0),
            other => panic!("expected 6.0 via outs_var, got {other:?}"),
        }
    }

    #[test]
    fn reduce_folds_outs_init() {
        // MLIR semantics: `outs` is the INITIAL accumulator. sum([1,2,3,4]) with a
        // non-identity `outs` init of 100 is 110, not 10 — the Rust port of the
        // Python `test_reduce_folds_outs_init` (tests/test_dialects_exec.py). Folds
        // a GENUINELY non-identity outs (no identity-only guard), matching the oracle.
        let mut ctx = single_core_context();
        // [[1,2,3,4]] f16 reduced along dim=1 → 10, then + outs init 100 → 110.
        ctx.set_value(
            "%x",
            Value::Tile(Tile::compute(
                vec![1.0, 2.0, 3.0, 4.0],
                DType::F16,
                vec![1, 4],
            )),
        );
        ctx.set_value(
            "%init",
            Value::Tile(Tile::compute(vec![100.0], DType::F16, vec![1])),
        );
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()))
            .with_attr("dim", Attr::Int(1))
            .with_attr("outs_var", Attr::Str("%init".into()));
        run(&[op], &mut ctx).unwrap();
        // dim=1 reduce of a [1,4] tile keeps the leading axis → shape [1], value 110.
        let val = match ctx.get_value("%r").unwrap() {
            Value::Tile(t) => t.as_f32()[0],
            Value::Scalar(Scalar::F32(v)) => *v,
            other => panic!("expected 110.0, got {other:?}"),
        };
        assert!((val - 110.0).abs() < 1e-1, "expected ~110.0, got {val}");
        // Bound back to outs_var too.
        match ctx.get_value("%init").unwrap() {
            Value::Tile(t) => assert!((t.as_f32()[0] - 110.0).abs() < 1e-1),
            Value::Scalar(Scalar::F32(v)) => assert!((*v - 110.0).abs() < 1e-1),
            other => panic!("expected outs_var bound to 110.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_scalar_input_passthrough() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Scalar(Scalar::F32(42.0)));
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()));
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 42.0),
            other => panic!("expected 42.0 passthrough, got {other:?}"),
        }
    }

    // --- generic ----------------------------------------------------------

    #[test]
    fn generic_elementwise_add() {
        let mut ctx = single_core_context();
        // ^bb0(%a, %b, %out): %s = addf %a, %b ; yield %s
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.set_value("%y", tile(vec![10.0, 20.0, 30.0], vec![3]));
        ctx.set_value("%init", tile(vec![0.0; 3], vec![3]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%y", "%init"])
            .with_attr("n_ins", Attr::Int(2))
            .with_attr(
                "bb0_names",
                Attr::StrList(vec!["%a".into(), "%b".into(), "%out".into()]),
            );
        op.regions.push(vec![
            Operation::new(Some("%s"), "arith.addf", &["%a", "%b"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]);
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.as_f32().to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn generic_uses_outs_block_arg() {
        let mut ctx = single_core_context();
        // ^bb0(%a, %out): %s = addf %a, %out ; yield %s  — accumulate into outs.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.set_value("%init", tile(vec![100.0, 200.0, 300.0], vec![3]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%out".into()]));
        op.regions.push(vec![
            Operation::new(Some("%s"), "arith.addf", &["%a", "%out"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]);
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.as_f32().to_vec(), vec![101.0, 202.0, 303.0]);
    }

    #[test]
    fn generic_broadcasts_input_via_indexing_map() {
        let mut ctx = single_core_context();
        // out [2,3]. input %x shape [3] maps to dim 1 only (indexing_maps "1"),
        // so it broadcasts across rows. addf with the [2,3] outs (all zero).
        ctx.set_value("%x", tile(vec![10.0, 20.0, 30.0], vec![3]));
        ctx.set_value("%init", tile(vec![0.0; 6], vec![2, 3]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%out".into()]))
            .with_attr(
                "indexing_maps",
                Attr::StrList(vec!["1".into(), "0,1".into()]),
            );
        op.regions.push(vec![
            Operation::new(Some("%s"), "arith.addf", &["%a", "%out"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]);
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 3]);
        // Each row is [10,20,30].
        assert_eq!(
            t.as_f32().to_vec(),
            vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn generic_yield_passthrough() {
        let mut ctx = single_core_context();
        // body just yields the input arg unchanged.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%init", tile(vec![0.0; 4], vec![2, 2]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%out".into()]));
        op.regions
            .push(vec![Operation::new(None, "linalg.yield", &["%a"])]);
        run(&[op], &mut ctx).unwrap();
        assert_eq!(
            get_tile(&ctx, "%r").as_f32().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn generic_requires_bb0_names() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0], vec![1]));
        ctx.set_value("%init", tile(vec![0.0], vec![1]));
        let op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1));
        // No region, no bb0_names -> error.
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("cannot determine bb0"));
    }

    // --- index ------------------------------------------------------------

    #[test]
    fn index_builds_arange() {
        let mut ctx = single_core_context();
        ctx.set_value(
            SHAPE_KEY,
            Value::Tuple(vec![Value::Index(2), Value::Index(3)]),
        );
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let op = Operation::new(Some("%i"), "linalg.index", &[]).with_attr("dim", Attr::Int(1));
        let v = super::index(&op, &mut ctx, &env).unwrap().unwrap();
        match v {
            Value::Tile(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.as_f32().to_vec(), vec![0.0, 1.0, 2.0]);
                assert_eq!(t.dtype, DType::I32);
            }
            other => panic!("expected index tile, got {other:?}"),
        }
    }

    #[test]
    fn index_dim0() {
        let mut ctx = single_core_context();
        ctx.set_value(
            SHAPE_KEY,
            Value::Tuple(vec![Value::Index(4), Value::Index(2)]),
        );
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let op = Operation::new(Some("%i"), "linalg.index", &[]).with_attr("dim", Attr::Int(0));
        let v = super::index(&op, &mut ctx, &env).unwrap().unwrap();
        match v {
            Value::Tile(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                assert_eq!(t.as_f32().to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("got {other:?}"),
        }
    }

    // --- helper unit tests ------------------------------------------------

    #[test]
    fn broadcast_to_rules() {
        // [1,3] -> [2,3]
        assert_eq!(
            broadcast_to(&[1.0, 2.0, 3.0], &[1, 3], &[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        // [3,1] -> [3,2]
        assert_eq!(
            broadcast_to(&[1.0, 2.0, 3.0], &[3, 1], &[3, 2]).unwrap(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
        // rank-extend: [3] -> [2,3]
        assert_eq!(
            broadcast_to(&[1.0, 2.0, 3.0], &[3], &[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        // incompatible
        assert!(broadcast_to(&[1.0, 2.0], &[2], &[3]).is_none());
    }

    #[test]
    fn slice_and_concat_roundtrip() {
        let shape = vec![2, 4];
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let (left, ls) = slice_along(&data, &shape, 1, 0, 2);
        let (right, _rs) = slice_along(&data, &shape, 1, 2, 4);
        assert_eq!(ls, vec![2, 2]);
        assert_eq!(left, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(right, vec![2.0, 3.0, 6.0, 7.0]);
        let cat = concat_along(&left, &ls, &right, 1);
        assert_eq!(cat, data);
    }

    #[test]
    fn strides_and_unravel() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(unravel(7, &[2, 4]), vec![1, 3]);
        assert_eq!(unravel(0, &[2, 3]), vec![0, 0]);
    }
}
