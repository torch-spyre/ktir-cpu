// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `tensor` dialect handlers — Rust port of `ktir_emulator/dialects/tensor_ops.py`.
//!
//! Ports every tensor op the Python source defines: `empty`, `splat`,
//! `extract`, `expand_shape`, `collapse_shape`, `reshape`, `from_elements`,
//! `generate` (region-bodied), and `yield`. The shape/reshape ops are pure
//! reinterpretations of a flat element buffer; index semantics follow NumPy's
//! C (row-major) order, exactly as the Python handlers do via `np.reshape` /
//! tuple indexing.
//!
//! TILE STORAGE NOTE: matching `tile.rs`, every tile is a flat `Vec<f32>`
//! widened from its declared dtype. The Python source carries integer / index
//! tiles as `np.int32` arrays; here we keep the values in `f32` and tag the
//! `DType`, mirroring the slice-1 storage decision. Integer index grids
//! (`tensor.generate` block args) are therefore exact up to 2^24, which covers
//! every shape the interpreter actually builds.

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::interpreter::execute_op;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::tile::Tile;

/// Register every handler this module owns. Called by `Dispatch::new`.
pub fn register(d: &mut Dispatch) {
    d.register("tensor.empty", LatencyCategory::Zero, empty);
    d.register("tensor.splat", LatencyCategory::Zero, splat);
    d.register("tensor.extract", LatencyCategory::Zero, extract);
    d.register("tensor.extract_slice", LatencyCategory::Zero, extract_slice);
    d.register("tensor.expand_shape", LatencyCategory::Zero, expand_shape);
    d.register(
        "tensor.collapse_shape",
        LatencyCategory::Zero,
        collapse_shape,
    );
    d.register("tensor.reshape", LatencyCategory::Zero, reshape);
    d.register("tensor.from_elements", LatencyCategory::Zero, from_elements);
    d.register("tensor.generate", LatencyCategory::Zero, generate);
    d.register("tensor.yield", LatencyCategory::Zero, yield_op);
}

/// `%t = tensor.empty() : tensor<...>` — uninitialized (zero-filled) tensor.
///
/// Mirrors `tensor__empty`: shape/dtype come from the result-type attributes
/// (`shape` defaults to `(1,)`, `dtype` to `f16`); data is `np.zeros`.
fn empty(
    op: &Operation,
    _ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let shape = shape_attr(op).unwrap_or_else(|| vec![1]);
    let dtype = dtype_attr_or(op, DType::F16)?;
    let n: usize = shape.iter().product();
    let data = vec![0.0f32; n];
    Ok(Some(Value::Tile(Tile::compute(data, dtype, shape))))
}

/// `%t = tensor.splat %scalar : ... -> tensor<...>` — broadcast a scalar to a
/// full tensor. Mirrors `tensor__splat`.
///
/// The Python heuristics are reproduced in order:
///   1. If the operand is itself a Tile, take its first (flat) element.
///   2. Target shape from the `shape` attribute (parser-synthesized from the
///      result type); otherwise fall back to `_infer_splat_shape` (the largest
///      tile already in scope), otherwise `(1,)`.
///   3. Integer scalars force an `i32` result tensor (NumPy `np.int32`);
///      everything else uses the declared dtype.
fn splat(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("tensor.splat: missing scalar operand".into());
    }
    let operand = ctx.get_value(&op.operands[0])?.clone();

    // (1) a Tile operand contributes its first flat element.
    let (scalar, is_int) = match &operand {
        Value::Tile(t) => {
            let v = t.as_f32().first().copied().unwrap_or(0.0);
            (
                v,
                t.dtype == DType::I32 || t.dtype == DType::I64 || t.dtype == DType::Bool,
            )
        }
        Value::Scalar(Scalar::F32(v)) => (*v, false),
        Value::Scalar(Scalar::I32(v)) => (*v as f32, true),
        Value::Scalar(Scalar::I64(v)) => (*v as f32, true),
        Value::Scalar(Scalar::Bool(b)) => (if *b { 1.0 } else { 0.0 }, true),
        Value::Index(i) => (*i as f32, true),
        other => {
            return Err(format!(
                "tensor.splat: unsupported scalar operand {other:?}"
            ));
        }
    };

    let mut dtype = dtype_attr_or(op, DType::F16)?;

    // (2) resolve target shape: attr -> infer-largest -> (1,)
    let shape = shape_attr(op)
        .or_else(|| infer_splat_shape(ctx))
        .unwrap_or_else(|| vec![1]);

    // (3) integer scalars override to an i32 tensor (NumPy np.int32).
    if is_int {
        dtype = DType::I32;
    }

    let n: usize = shape.iter().product();
    let data = vec![scalar; n];
    Ok(Some(Value::Tile(Tile::compute(data, dtype, shape))))
}

/// `%s = tensor.extract %t[%i, %j, ...]` — read a single element. Mirrors
/// `tensor__extract`.
///
/// With no indices the source is treated as a 0-D tensor and its single
/// element is returned. A non-Tile operand is passed through unchanged (the
/// Python "already a scalar" branch). The extracted element is returned as a
/// scalar whose flavor matches the tile dtype (float vs int/index).
fn extract(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("tensor.extract: missing source operand".into());
    }
    let src = ctx.get_value(&op.operands[0])?.clone();

    let tile = match src {
        Value::Tile(t) => t,
        // Already a scalar/index — pass through.
        other => return Ok(Some(other)),
    };

    let indices: Vec<i64> = op.operands[1..]
        .iter()
        .map(|name| {
            ctx.get_value(name)
                .and_then(|v| as_i64(v, "tensor.extract index"))
        })
        .collect::<Result<_, _>>()?;

    let flat = if indices.is_empty() {
        // 0-D tensor: src.data.flat[0]
        0
    } else {
        ravel_index(&indices, &tile.shape, "tensor.extract")?
    };
    let tile_data = tile.as_f32();
    let elem = *tile_data
        .get(flat)
        .ok_or_else(|| format!("tensor.extract: flat index {flat} out of bounds"))?;

    Ok(Some(scalar_for_dtype(elem, tile.dtype)))
}

/// `%slice = tensor.extract_slice %src[offsets][sizes][strides] : T to U` —
/// a strided rectangular sub-view of `src`, materialized as a fresh tile.
///
/// The parser captured the three offset-size-stride lists as `StrList` token
/// attributes (`slice_offsets` / `slice_sizes` / `slice_strides`); each token is
/// either a static integer or a dynamic `%ssa` resolved here against the value
/// table (the tiled K-loop edge passes its induction variable as a dynamic
/// offset). For output element at multi-index `c` (row-major over `sizes`), the
/// source element is at `offset[k] + c[k] * stride[k]` per axis `k`, flattened
/// row-major over the source shape. Result dtype follows the source tile.
///
/// Rank-reduced results (where `sizes` has fewer entries than the source rank,
/// MLIR's unit-dim drop) are not produced by our fusion path and are rejected
/// rather than guessed.
fn extract_slice(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("tensor.extract_slice: missing source operand".into());
    }
    let src = ctx.get_value(&op.operands[0])?.clone();
    let tile = match src {
        Value::Tile(t) => t,
        other => {
            return Err(format!(
                "tensor.extract_slice: source must be a tensor, got {other:?}"
            ));
        }
    };

    let offsets = resolve_slice_list(op, ctx, "slice_offsets")?;
    let sizes = resolve_slice_list(op, ctx, "slice_sizes")?;
    let strides = resolve_slice_list(op, ctx, "slice_strides")?;
    let rank = tile.shape.len();
    if offsets.len() != rank || sizes.len() != rank || strides.len() != rank {
        return Err(format!(
            "tensor.extract_slice: offsets/sizes/strides ranks {}/{}/{} must equal source rank {rank}",
            offsets.len(),
            sizes.len(),
            strides.len()
        ));
    }

    // Row-major strides of the source buffer (elements per step along each axis).
    let mut src_strides = vec![1i64; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        src_strides[k] = src_strides[k + 1] * tile.shape[k + 1] as i64;
    }

    let out_n: usize = sizes.iter().map(|&s| s.max(0) as usize).product();
    let mut out = Vec::with_capacity(out_n);
    let mut coord = vec![0i64; rank]; // current output multi-index
    let tile_data = tile.as_f32();
    for _ in 0..out_n {
        let mut flat = 0i64;
        for k in 0..rank {
            let s = offsets[k] + coord[k] * strides[k];
            if s < 0 || s >= tile.shape[k] as i64 {
                return Err(format!(
                    "tensor.extract_slice: source index {s} out of bounds on axis {k} (size {})",
                    tile.shape[k]
                ));
            }
            flat += s * src_strides[k];
        }
        out.push(tile_data[flat as usize]);
        // increment row-major over `sizes` (rightmost axis fastest).
        for k in (0..rank).rev() {
            coord[k] += 1;
            if coord[k] < sizes[k] {
                break;
            }
            coord[k] = 0;
        }
    }

    let shape: Vec<usize> = sizes.iter().map(|&s| s as usize).collect();
    Ok(Some(Value::Tile(Tile::compute(out, tile.dtype, shape))))
}

/// Resolve a `slice_offsets`/`slice_sizes`/`slice_strides` token list to i64s:
/// `%ssa` tokens read the value table (dynamic dims), the rest parse as ints.
fn resolve_slice_list(
    op: &Operation,
    ctx: &mut CoreContext,
    key: &str,
) -> Result<Vec<i64>, String> {
    let toks = match op.attributes.get(key) {
        Some(Attr::StrList(v)) => v,
        _ => return Err(format!("tensor.extract_slice: missing '{key}' attribute")),
    };
    toks.iter()
        .map(|tok| {
            if tok.starts_with('%') {
                as_i64(ctx.get_value(tok)?, key)
            } else {
                tok.parse::<i64>()
                    .map_err(|_| format!("tensor.extract_slice: non-integer {key} token {tok:?}"))
            }
        })
        .collect()
}

/// `%t = tensor.expand_shape %src ... into tensor<...>` — reinterpret under a
/// larger-rank shape. Mirrors `tensor__expand_shape`.
fn expand_shape(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    reshape_via_target(op, ctx, "tensor.expand_shape")
}

/// `%t = tensor.collapse_shape %src ... into tensor<...>` — reinterpret under a
/// smaller-rank shape. Mirrors `tensor__collapse_shape` (identical body).
fn collapse_shape(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    reshape_via_target(op, ctx, "tensor.collapse_shape")
}

/// `%out = tensor.reshape %src(%shape) -> tensor<...>` — reinterpret the same
/// element count under a new shape. Mirrors `tensor__reshape`.
///
/// As in Python, the target shape is read from the (parser-synthesized,
/// result-type-pinned) `target_shape` attribute — never the runtime shape
/// operand — so the second operand is ignored here. A non-Tile source passes
/// through unchanged; a missing `target_shape` is a hard error.
fn reshape(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let target = target_shape_attr(op).ok_or_else(|| {
        format!(
            "tensor.reshape: missing 'target_shape' attribute on op {}",
            op.op_type
        )
    })?;
    if op.operands.is_empty() {
        return Err("tensor.reshape: missing source operand".into());
    }
    let src = ctx.get_value(&op.operands[0])?.clone();
    let tile = match src {
        Value::Tile(t) => t,
        other => return Ok(Some(other)),
    };
    Ok(Some(Value::Tile(reshaped(
        &tile,
        target,
        "tensor.reshape",
    )?)))
}

/// `%shape = tensor.from_elements %d0, %d1, ... : tensor<NxT>` — build a 1-D
/// (or reshaped) tensor from N scalar operands. Mirrors `tensor__from_elements`.
///
/// Each operand is coerced to a scalar (a Tile operand contributes its first
/// flat element). The values are stacked then reshaped into the declared shape.
fn from_elements(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let shape = shape_attr(op).ok_or_else(|| {
        format!(
            "tensor.from_elements: missing 'shape' attribute on op {}",
            op.op_type
        )
    })?;
    let dtype = dtype_attr(op).ok_or_else(|| {
        format!(
            "tensor.from_elements: missing 'dtype' attribute on op {}",
            op.op_type
        )
    })??;

    let mut values: Vec<f32> = Vec::with_capacity(op.operands.len());
    for name in &op.operands {
        let v = ctx.get_value(name)?;
        let s = match v {
            Value::Tile(t) => t.as_f32().first().copied().unwrap_or(0.0),
            Value::Scalar(Scalar::F32(x)) => *x,
            Value::Scalar(Scalar::I32(x)) => *x as f32,
            Value::Scalar(Scalar::I64(x)) => *x as f32,
            Value::Scalar(Scalar::Bool(b)) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            Value::Index(i) => *i as f32,
            other => {
                return Err(format!(
                    "tensor.from_elements: unsupported operand {other:?}"
                ));
            }
        };
        values.push(s);
    }

    let n: usize = shape.iter().product();
    if values.len() != n {
        return Err(format!(
            "tensor.from_elements: {} elements cannot fill shape {:?} ({n} elements)",
            values.len(),
            shape
        ));
    }
    Ok(Some(Value::Tile(Tile::compute(values, dtype, shape))))
}

/// `%t = tensor.generate { ^bb0(%i, %j): ...; tensor.yield %v } : tensor<...>`
/// — build a tensor by evaluating a region body over an index grid. Mirrors
/// `tensor__generate`.
///
/// Faithful to the Python *vectorized* execution: rather than re-running the
/// body once per element, the block args are bound to full index grids (as
/// index-typed Tiles) and the body runs a single time. Compute ops already act
/// element-wise on Tiles, so one pass produces the whole output. The grids use
/// `np.meshgrid(..., indexing='ij')` semantics: arg `k` varies along axis `k`.
///
/// The yielded value (captured from the body's `tensor.yield`) is cast to the
/// declared dtype; a scalar yield is broadcast to the full shape.
fn generate(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let shape = shape_attr(op).unwrap_or_default();
    let dtype = dtype_attr_or(op, DType::F16)?;
    let n: usize = shape.iter().product();

    let region: &[Operation] = op.regions.first().map(|r| r.as_slice()).unwrap_or(&[]);

    // The ^bb0 label is parsed into a synthetic `region.bb0_args` op carrying
    // the block-arg names; the real body is everything else.
    let bb0 = region.iter().find(|o| o.op_type == "region.bb0_args");
    let block_args: Vec<String> = match bb0.and_then(|o| o.attributes.get("names")) {
        Some(Attr::StrList(names)) => names.clone(),
        _ => Vec::new(),
    };
    let body: Vec<&Operation> = region
        .iter()
        .filter(|o| o.op_type != "region.bb0_args")
        .collect();

    // Build one index grid per block arg (meshgrid 'ij' indexing).
    let grids = meshgrid_ij(&shape);

    ctx.push_scope();
    let result = (|| -> Result<Value, String> {
        for (arg_name, grid) in block_args.iter().zip(grids) {
            ctx.set_value(
                arg_name,
                Value::Tile(Tile::compute(grid, DType::I32, shape.clone())),
            );
        }
        // Execute the body; `tensor.yield` returns the produced value, which is
        // the region result (single-value yield, mirroring scf.yield).
        let mut yielded: Option<Value> = None;
        for o in &body {
            let produced = execute_op(o, ctx, env)?;
            if o.op_type == "tensor.yield" {
                yielded = produced;
            }
        }
        yielded.ok_or_else(|| "tensor.generate: body did not yield a value".to_string())
    })();
    // Always restore the scope, even on error.
    ctx.pop_scope();
    let result = result?;

    // Cast the body result to the declared dtype; broadcast a scalar yield.
    let data = match result {
        Value::Tile(t) => {
            if t.len() != n {
                return Err(format!(
                    "tensor.generate: body produced {} elements, expected {n} for shape {:?}",
                    t.len(),
                    shape
                ));
            }
            t.as_f32().to_vec()
        }
        other => {
            let s = as_f32(&other, "tensor.generate yield")?;
            vec![s; n]
        }
    };

    Ok(Some(Value::Tile(Tile::compute(data, dtype, shape))))
}

/// `tensor.yield %v` — terminate a `tensor.generate` body. Same semantics as
/// `scf.yield`: returns the (single) yielded operand value. Mirrors
/// `tensor__yield` (which wraps a single value in a `YieldSignal`).
fn yield_op(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    match op.operands.first() {
        Some(name) => Ok(Some(ctx.get_value(name)?.clone())),
        None => Ok(None),
    }
}

// --- shared reshape logic ------------------------------------------------

/// Body shared by `expand_shape` / `collapse_shape`: reinterpret `src` under
/// the `target_shape` attribute, keeping the source dtype. A non-Tile source
/// (or a missing target shape) passes through unchanged, mirroring the Python
/// guard `isinstance(src, Tile) and target_shape`.
fn reshape_via_target(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err(format!("{name}: missing source operand"));
    }
    let src = ctx.get_value(&op.operands[0])?.clone();
    let target = match target_shape_attr(op) {
        Some(s) => s,
        None => return Ok(Some(src)), // no target: pass through
    };
    match src {
        Value::Tile(t) => Ok(Some(Value::Tile(reshaped(&t, target, name)?))),
        other => Ok(Some(other)),
    }
}

/// Reinterpret a tile's flat (row-major) buffer under `target`, preserving the
/// source dtype. Errors if the element count changes — matching `np.reshape`,
/// which raises rather than silently truncating.
fn reshaped(tile: &Tile, target: Vec<usize>, name: &str) -> Result<Tile, String> {
    let want: usize = target.iter().product();
    if want != tile.len() {
        return Err(format!(
            "{name}: cannot reshape {} elements into {:?} ({want} elements)",
            tile.len(),
            target
        ));
    }
    Ok(Tile::compute(tile.as_f32().to_vec(), tile.dtype, target))
}

// --- index / shape helpers -----------------------------------------------

/// Ravel a multi-index into a flat (row-major / C-order) offset. Mirrors NumPy
/// tuple indexing `data[(i, j, ...)]`. Bounds-checks each axis.
fn ravel_index(indices: &[i64], shape: &[usize], name: &str) -> Result<usize, String> {
    if indices.len() != shape.len() {
        return Err(format!(
            "{name}: {} indices for rank-{} tensor",
            indices.len(),
            shape.len()
        ));
    }
    let mut flat = 0usize;
    for (k, (&idx, &dim)) in indices.iter().zip(shape).enumerate() {
        if idx < 0 || idx as usize >= dim {
            return Err(format!(
                "{name}: index {idx} out of bounds for axis {k} (size {dim})"
            ));
        }
        flat = flat * dim + idx as usize;
    }
    Ok(flat)
}

/// `np.meshgrid(*(arange(s) for s in shape), indexing='ij')` flattened to
/// row-major buffers — one grid per axis. Grid `k` holds, at each flat
/// position, the value of index `k` for that position.
fn meshgrid_ij(shape: &[usize]) -> Vec<Vec<f32>> {
    let n: usize = shape.iter().product();
    let mut grids: Vec<Vec<f32>> = vec![Vec::with_capacity(n); shape.len()];
    for flat in 0..n {
        let mut rem = flat;
        // Decompose flat -> (c0, c1, ...) in row-major order.
        let mut coords = vec![0usize; shape.len()];
        for axis in (0..shape.len()).rev() {
            let dim = shape[axis];
            coords[axis] = rem % dim;
            rem /= dim;
        }
        for (axis, grid) in grids.iter_mut().enumerate() {
            grid.push(coords[axis] as f32);
        }
    }
    grids
}

/// Largest tile already in scope, by element count — port of
/// `_infer_splat_shape`. `CoreContext` does not expose its scope stack, so this
/// is a no-op fallback: the parser almost always supplies the `shape`
/// attribute, and the Python heuristic only fires when the result type was
/// unparseable. See contract_notes.
fn infer_splat_shape(_ctx: &CoreContext) -> Option<Vec<usize>> {
    None
}

/// Wrap a raw element value in the scalar variant matching `dtype`.
fn scalar_for_dtype(v: f32, dtype: DType) -> Value {
    match dtype {
        DType::F16 | DType::F32 => Value::Scalar(Scalar::F32(v)),
        DType::I32 | DType::I64 => Value::Index(v as i64),
        DType::Bool => Value::Scalar(Scalar::Bool(v != 0.0)),
    }
}

// --- attribute helpers ---------------------------------------------------

/// Read the `shape` attribute as a usize vector, if present and well-formed.
fn shape_attr(op: &Operation) -> Option<Vec<usize>> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) => Some(v.iter().map(|&n| n as usize).collect()),
        Some(Attr::Int(n)) => Some(vec![*n as usize]),
        _ => None,
    }
}

/// Read the `target_shape` attribute as a usize vector, if present.
fn target_shape_attr(op: &Operation) -> Option<Vec<usize>> {
    match op.attributes.get("target_shape") {
        Some(Attr::IntList(v)) => Some(v.iter().map(|&n| n as usize).collect()),
        Some(Attr::Int(n)) => Some(vec![*n as usize]),
        _ => None,
    }
}

/// Read the `dtype` attribute (string or `Dtype`), if present.
fn dtype_attr(op: &Operation) -> Option<Result<DType, String>> {
    match op.attributes.get("dtype") {
        Some(Attr::Dtype(d)) => Some(Ok(*d)),
        Some(Attr::Str(s)) => Some(DType::parse(s)),
        _ => None,
    }
}

/// Read the `dtype` attribute, falling back to `default` when absent.
fn dtype_attr_or(op: &Operation, default: DType) -> Result<DType, String> {
    match dtype_attr(op) {
        Some(r) => r,
        None => Ok(default),
    }
}

fn as_i64(v: &Value, name: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(Scalar::I32(i)) => Ok(*i as i64),
        Value::Scalar(Scalar::I64(i)) => Ok(*i),
        Value::Scalar(Scalar::F32(f)) => Ok(*f as i64),
        Value::Tile(t) => Ok(t.as_f32().first().copied().unwrap_or(0.0) as i64),
        other => Err(format!("{name}: expected index/int, got {other:?}")),
    }
}

fn as_f32(v: &Value, name: &str) -> Result<f32, String> {
    match v {
        Value::Scalar(Scalar::F32(f)) => Ok(*f),
        Value::Scalar(Scalar::I32(i)) => Ok(*i as f32),
        Value::Scalar(Scalar::I64(i)) => Ok(*i as f32),
        Value::Scalar(Scalar::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Index(i) => Ok(*i as f32),
        Value::Tile(t) => Ok(t.as_f32().first().copied().unwrap_or(0.0)),
        other => Err(format!("{name}: expected scalar, got {other:?}")),
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

    fn tile(ctx: &CoreContext, name: &str) -> Tile {
        match ctx.get_value(name).unwrap() {
            Value::Tile(t) => t.clone(),
            other => panic!("expected tile for {name}, got {other:?}"),
        }
    }

    // --- empty -----------------------------------------------------------

    #[test]
    fn empty_zeros_with_shape_and_dtype() {
        let mut ctx = single_core_context();
        let op = Operation::new(Some("%t"), "tensor.empty", &[])
            .with_attr("shape", Attr::IntList(vec![2, 3]))
            .with_attr("dtype", Attr::Str("f32".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.as_f32().to_vec(), vec![0.0; 6]);
    }

    #[test]
    fn empty_defaults_to_unit_shape_f16() {
        let mut ctx = single_core_context();
        let op = Operation::new(Some("%t"), "tensor.empty", &[]);
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.shape, vec![1]);
        assert_eq!(t.dtype, DType::F16);
        assert_eq!(t.as_f32().to_vec(), vec![0.0]);
    }

    // --- splat -----------------------------------------------------------

    #[test]
    fn splat_broadcasts_float_scalar() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Scalar(Scalar::F32(2.5)));
        let op = Operation::new(Some("%t"), "tensor.splat", &["%s"])
            .with_attr("shape", Attr::IntList(vec![1, 4]))
            .with_attr("dtype", Attr::Str("f16".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.shape, vec![1, 4]);
        assert_eq!(t.as_f32().to_vec(), vec![2.5; 4]);
        assert_eq!(t.dtype, DType::F16);
    }

    #[test]
    fn splat_integer_scalar_forces_i32() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Index(7));
        let op = Operation::new(Some("%t"), "tensor.splat", &["%s"])
            .with_attr("shape", Attr::IntList(vec![3]))
            .with_attr("dtype", Attr::Str("f16".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        // integer scalar overrides dtype to i32 (mirrors np.int32 branch)
        assert_eq!(t.dtype, DType::I32);
        assert_eq!(t.as_f32().to_vec(), vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn splat_tile_operand_takes_first_element() {
        let mut ctx = single_core_context();
        ctx.set_value(
            "%src",
            Value::Tile(Tile::compute(vec![9.0, 1.0, 2.0], DType::F32, vec![3])),
        );
        let op = Operation::new(Some("%t"), "tensor.splat", &["%src"])
            .with_attr("shape", Attr::IntList(vec![2]))
            .with_attr("dtype", Attr::Str("f32".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.as_f32().to_vec(), vec![9.0, 9.0]);
    }

    #[test]
    fn splat_no_shape_defaults_to_unit() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Scalar(Scalar::F32(4.0)));
        let op = Operation::new(Some("%t"), "tensor.splat", &["%s"]);
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.shape, vec![1]);
        assert_eq!(t.as_f32().to_vec(), vec![4.0]);
    }

    // --- extract ---------------------------------------------------------

    #[test]
    fn extract_reads_row_major_element() {
        let mut ctx = single_core_context();
        // 2x3 tile: [[0,1,2],[3,4,5]]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        ctx.set_value(
            "%t",
            Value::Tile(Tile::compute(data, DType::F32, vec![2, 3])),
        );
        ctx.set_value("%i", Value::Index(1));
        ctx.set_value("%j", Value::Index(2));
        let op = Operation::new(Some("%s"), "tensor.extract", &["%t", "%i", "%j"]);
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%s").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 5.0), // [1][2]
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn extract_zero_d_returns_only_element() {
        let mut ctx = single_core_context();
        ctx.set_value(
            "%t",
            Value::Tile(Tile::compute(vec![42.0], DType::F32, vec![1])),
        );
        let op = Operation::new(Some("%s"), "tensor.extract", &["%t"]);
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%s").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 42.0),
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn extract_index_tile_returns_index_scalar() {
        let mut ctx = single_core_context();
        ctx.set_value(
            "%t",
            Value::Tile(Tile::compute(vec![3.0, 8.0], DType::I32, vec![2])),
        );
        ctx.set_value("%i", Value::Index(1));
        let op = Operation::new(Some("%s"), "tensor.extract", &["%t", "%i"]);
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%s").unwrap() {
            Value::Index(i) => assert_eq!(*i, 8),
            other => panic!("expected Index, got {other:?}"),
        }
    }

    #[test]
    fn extract_passthrough_non_tile() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Scalar(Scalar::F32(1.5)));
        let op = Operation::new(Some("%out"), "tensor.extract", &["%s"]);
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%out").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 1.5),
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn extract_out_of_bounds_errors() {
        let mut ctx = single_core_context();
        ctx.set_value(
            "%t",
            Value::Tile(Tile::compute(vec![0.0, 1.0], DType::F32, vec![2])),
        );
        ctx.set_value("%i", Value::Index(5));
        let op = Operation::new(Some("%s"), "tensor.extract", &["%t", "%i"]);
        assert!(run(&[op], &mut ctx).is_err());
    }

    // --- extract_slice ---------------------------------------------------

    fn slice_op(src: &str, offsets: &[&str], sizes: &[i64], strides: &[i64]) -> Operation {
        let strs = |xs: &[&str]| Attr::StrList(xs.iter().map(|s| s.to_string()).collect());
        let ints = |xs: &[i64]| Attr::StrList(xs.iter().map(|n| n.to_string()).collect());
        Operation::new(Some("%slice"), "tensor.extract_slice", &[src])
            .with_attr("slice_offsets", strs(offsets))
            .with_attr("slice_sizes", ints(sizes))
            .with_attr("slice_strides", ints(strides))
    }

    #[test]
    fn extract_slice_static_2d_block() {
        let mut ctx = single_core_context();
        // 4x4 with values 0..16, take [1,1][2,2][1,1] -> rows 1..2, cols 1..2.
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        ctx.set_value(
            "%t",
            Value::Tile(Tile::compute(data, DType::F32, vec![4, 4])),
        );
        let op = slice_op("%t", &["1", "1"], &[2, 2], &[1, 1]);
        run(&[op], &mut ctx).unwrap();
        let s = tile(&ctx, "%slice");
        assert_eq!(s.shape, vec![2, 2]);
        // row1 = [4,5,6,7], row2 = [8,9,10,11] -> cols 1,2 -> [5,6,9,10]
        assert_eq!(s.as_f32().to_vec(), vec![5.0, 6.0, 9.0, 10.0]);
        assert_eq!(s.dtype, DType::F32);
    }

    #[test]
    fn extract_slice_strided_1d() {
        let mut ctx = single_core_context();
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        ctx.set_value("%t", Value::Tile(Tile::compute(data, DType::F32, vec![8])));
        // offset 1, size 3, stride 2 -> elements 1,3,5
        let op = slice_op("%t", &["1"], &[3], &[2]);
        run(&[op], &mut ctx).unwrap();
        let s = tile(&ctx, "%slice");
        assert_eq!(s.as_f32().to_vec(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn extract_slice_dynamic_offset_row() {
        let mut ctx = single_core_context();
        // 4x4; the tiled K-loop edge passes its induction var as a dynamic row
        // offset and reads a 1x4 sub-tile.
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        ctx.set_value(
            "%t",
            Value::Tile(Tile::compute(data, DType::F32, vec![4, 4])),
        );
        ctx.set_value("%k", Value::Index(2));
        let op = slice_op("%t", &["%k", "0"], &[1, 4], &[1, 1]);
        run(&[op], &mut ctx).unwrap();
        let s = tile(&ctx, "%slice");
        assert_eq!(s.shape, vec![1, 4]);
        assert_eq!(s.as_f32().to_vec(), vec![8.0, 9.0, 10.0, 11.0]); // row 2
    }

    #[test]
    fn extract_slice_out_of_bounds_errors() {
        let mut ctx = single_core_context();
        let data: Vec<f32> = (0..4).map(|x| x as f32).collect();
        ctx.set_value("%t", Value::Tile(Tile::compute(data, DType::F32, vec![4])));
        // offset 3, size 2, stride 1 -> would read index 4 (out of bounds).
        let op = slice_op("%t", &["3"], &[2], &[1]);
        assert!(run(&[op], &mut ctx).is_err());
    }

    // --- reshape / expand / collapse -------------------------------------

    #[test]
    fn reshape_reinterprets_row_major() {
        let mut ctx = single_core_context();
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        ctx.set_value(
            "%src",
            Value::Tile(Tile::compute(data.clone(), DType::F32, vec![6])),
        );
        // shape operand is ignored; target_shape attr drives the result
        ctx.set_value(
            "%shape",
            Value::Tile(Tile::compute(vec![2.0, 3.0], DType::I32, vec![2])),
        );
        let op = Operation::new(Some("%out"), "tensor.reshape", &["%src", "%shape"])
            .with_attr("target_shape", Attr::IntList(vec![2, 3]))
            .with_attr("dtype", Attr::Str("f32".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%out");
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.as_f32().to_vec(), data); // same flat buffer, row-major
    }

    #[test]
    fn reshape_missing_target_errors() {
        let mut ctx = single_core_context();
        ctx.set_value(
            "%src",
            Value::Tile(Tile::compute(vec![1.0], DType::F32, vec![1])),
        );
        let op = Operation::new(Some("%out"), "tensor.reshape", &["%src", "%shape"]);
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("target_shape"));
    }

    #[test]
    fn reshape_wrong_count_errors() {
        let mut ctx = single_core_context();
        ctx.set_value(
            "%src",
            Value::Tile(Tile::compute(vec![1.0, 2.0], DType::F32, vec![2])),
        );
        let op = Operation::new(Some("%out"), "tensor.reshape", &["%src"])
            .with_attr("target_shape", Attr::IntList(vec![3]));
        assert!(run(&[op], &mut ctx).is_err());
    }

    #[test]
    fn expand_shape_keeps_dtype_and_data() {
        let mut ctx = single_core_context();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        ctx.set_value(
            "%src",
            Value::Tile(Tile::compute(data.clone(), DType::F16, vec![4])),
        );
        let op = Operation::new(Some("%out"), "tensor.expand_shape", &["%src"])
            .with_attr("target_shape", Attr::IntList(vec![2, 2]));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%out");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.dtype, DType::F16); // source dtype preserved
        assert_eq!(t.as_f32().to_vec(), data);
    }

    #[test]
    fn collapse_shape_flattens() {
        let mut ctx = single_core_context();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        ctx.set_value(
            "%src",
            Value::Tile(Tile::compute(data.clone(), DType::F32, vec![2, 3])),
        );
        let op = Operation::new(Some("%out"), "tensor.collapse_shape", &["%src"])
            .with_attr("target_shape", Attr::IntList(vec![6]));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%out");
        assert_eq!(t.shape, vec![6]);
        assert_eq!(t.as_f32().to_vec(), data);
    }

    #[test]
    fn reshape_passthrough_non_tile() {
        let mut ctx = single_core_context();
        ctx.set_value("%src", Value::Index(9));
        let op = Operation::new(Some("%out"), "tensor.reshape", &["%src"])
            .with_attr("target_shape", Attr::IntList(vec![1]));
        run(&[op], &mut ctx).unwrap();
        assert!(matches!(ctx.get_value("%out").unwrap(), Value::Index(9)));
    }

    // --- from_elements ---------------------------------------------------

    #[test]
    fn from_elements_stacks_scalars() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::Index(16));
        ctx.set_value("%b", Value::Index(32));
        let op = Operation::new(Some("%shape"), "tensor.from_elements", &["%a", "%b"])
            .with_attr("shape", Attr::IntList(vec![2]))
            .with_attr("dtype", Attr::Str("index".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%shape");
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.dtype, DType::I32); // index lowers to i32
        assert_eq!(t.as_f32().to_vec(), vec![16.0, 32.0]);
    }

    #[test]
    fn from_elements_reshapes_to_declared_shape() {
        let mut ctx = single_core_context();
        for (n, v) in ["%a", "%b", "%c", "%d"].iter().zip([1.0, 2.0, 3.0, 4.0]) {
            ctx.set_value(n, Value::Scalar(Scalar::F32(v)));
        }
        let op = Operation::new(
            Some("%t"),
            "tensor.from_elements",
            &["%a", "%b", "%c", "%d"],
        )
        .with_attr("shape", Attr::IntList(vec![2, 2]))
        .with_attr("dtype", Attr::Str("f32".into()));
        run(&[op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.as_f32().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn from_elements_count_mismatch_errors() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::Index(1));
        let op = Operation::new(Some("%t"), "tensor.from_elements", &["%a"])
            .with_attr("shape", Attr::IntList(vec![2]))
            .with_attr("dtype", Attr::Str("index".into()));
        assert!(run(&[op], &mut ctx).is_err());
    }

    // --- meshgrid helper -------------------------------------------------

    #[test]
    fn meshgrid_ij_matches_numpy() {
        // shape (3,3): grids[0] varies along axis 0, grids[1] along axis 1.
        let grids = meshgrid_ij(&[3, 3]);
        assert_eq!(
            grids[0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0] // %i
        );
        assert_eq!(
            grids[1],
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0] // %j
        );
    }

    #[test]
    fn meshgrid_ij_rectangular() {
        // shape (2,3)
        let grids = meshgrid_ij(&[2, 3]);
        assert_eq!(grids[0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(grids[1], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    // --- generate (region body) ------------------------------------------

    #[test]
    fn generate_yields_index_grid_sum() {
        // %t = tensor.generate { ^bb0(%i,%j): %s = addf %i,%j; yield %s } : 2x2
        // addf works element-wise on the index grids -> i + j at each position.
        let mut ctx = single_core_context();
        let bb0 = Operation::new(None, "region.bb0_args", &[])
            .with_attr("names", Attr::StrList(vec!["%i".into(), "%j".into()]));
        let add = Operation::new(Some("%s"), "arith.addf", &["%i", "%j"]);
        let yld = Operation::new(None, "tensor.yield", &["%s"]);
        let mut gen_op = Operation::new(Some("%t"), "tensor.generate", &[])
            .with_attr("shape", Attr::IntList(vec![2, 2]))
            .with_attr("dtype", Attr::Str("f32".into()));
        gen_op.regions = vec![vec![bb0, add, yld]];
        run(&[gen_op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.shape, vec![2, 2]);
        // i+j over (i,j) in 2x2: [[0,1],[1,2]]
        assert_eq!(t.as_f32().to_vec(), vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(t.dtype, DType::F32);
    }

    #[test]
    fn generate_scalar_yield_broadcasts() {
        // body yields a constant scalar -> full tensor of that value.
        let mut ctx = single_core_context();
        let bb0 = Operation::new(None, "region.bb0_args", &[])
            .with_attr("names", Attr::StrList(vec!["%i".into()]));
        let c =
            Operation::new(Some("%v"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "tensor.yield", &["%v"]);
        let mut gen_op = Operation::new(Some("%t"), "tensor.generate", &[])
            .with_attr("shape", Attr::IntList(vec![3]))
            .with_attr("dtype", Attr::Str("f32".into()));
        gen_op.regions = vec![vec![bb0, c, yld]];
        run(&[gen_op], &mut ctx).unwrap();
        let t = tile(&ctx, "%t");
        assert_eq!(t.as_f32().to_vec(), vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn generate_body_scope_is_popped() {
        // Block-arg bindings must not leak past the generate op.
        let mut ctx = single_core_context();
        let bb0 = Operation::new(None, "region.bb0_args", &[])
            .with_attr("names", Attr::StrList(vec!["%i".into()]));
        let yld = Operation::new(None, "tensor.yield", &["%i"]);
        let mut gen_op = Operation::new(Some("%t"), "tensor.generate", &[])
            .with_attr("shape", Attr::IntList(vec![2]))
            .with_attr("dtype", Attr::Str("index".into()));
        gen_op.regions = vec![vec![bb0, yld]];
        run(&[gen_op], &mut ctx).unwrap();
        assert!(!ctx.has_value("%i")); // popped
        let t = tile(&ctx, "%t");
        assert_eq!(t.as_f32().to_vec(), vec![0.0, 1.0]); // identity index grid
    }

    // --- yield -----------------------------------------------------------

    #[test]
    fn yield_returns_operand() {
        let mut ctx = single_core_context();
        ctx.set_value("%v", Value::Scalar(Scalar::F32(3.0)));
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let op = Operation::new(None, "tensor.yield", &["%v"]);
        let out = execute_op(&op, &mut ctx, &env).unwrap();
        match out {
            Some(Value::Scalar(Scalar::F32(v))) => assert_eq!(v, 3.0),
            other => panic!("expected F32, got {other:?}"),
        }
    }
}
