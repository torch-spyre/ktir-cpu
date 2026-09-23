// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `arith` dialect handlers — full port of `ktir_emulator/dialects/arith_ops.py`
//! (which dispatches into `ktir_emulator/ops/arith_ops.py::ArithOps`).
//!
//! Every op works on scalars *and* element-wise on tiles, exactly where the
//! Python source does. Tile storage in this crate is a flat `Vec<f32>` (see
//! `tile.rs`), so integer / bitwise / shift ops round-trip element values
//! through `i64` at the op boundary — f32 exactly represents integers up to
//! 2^24, which covers the index-arithmetic these ops perform. Scalars carry
//! their `Scalar` variant: float ops yield `F32` (Python yields f16 — we widen
//! at the boundary), integer ops yield `I64`, comparisons yield `Bool`.

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::tile::Tile;

/// Register every handler this module owns. Called by `Dispatch::new`.
pub fn register(d: &mut Dispatch) {
    // Float binary
    d.register("arith.addf", LatencyCategory::ComputeFloat, addf);
    d.register("arith.subf", LatencyCategory::ComputeFloat, subf);
    d.register("arith.mulf", LatencyCategory::ComputeFloat, mulf);
    d.register("arith.divf", LatencyCategory::ComputeFloat, divf);
    d.register("arith.remf", LatencyCategory::ComputeFloat, remf);

    // Float unary
    d.register("arith.negf", LatencyCategory::ComputeFloat, negf);
    d.register("arith.absf", LatencyCategory::ComputeFloat, absf);

    // Float min/max — note `maxf`/`maximumf` and `minf`/`minimumf` are aliases.
    d.register("arith.maxf", LatencyCategory::ComputeFloat, maxf);
    d.register("arith.maximumf", LatencyCategory::ComputeFloat, maxf);
    d.register("arith.maxnumf", LatencyCategory::ComputeFloat, maxnumf);
    d.register("arith.minf", LatencyCategory::ComputeFloat, minf);
    d.register("arith.minimumf", LatencyCategory::ComputeFloat, minf);
    d.register("arith.minnumf", LatencyCategory::ComputeFloat, minnumf);

    // Float comparison
    d.register("arith.cmpf", LatencyCategory::ComputeFloat, cmpf);

    // Integer binary
    d.register("arith.addi", LatencyCategory::ComputeInt, addi);
    d.register("arith.subi", LatencyCategory::ComputeInt, subi);
    d.register("arith.muli", LatencyCategory::ComputeInt, muli);
    d.register("arith.divsi", LatencyCategory::ComputeInt, divsi);
    d.register("arith.divui", LatencyCategory::ComputeInt, divui);
    d.register("arith.floordivsi", LatencyCategory::ComputeInt, floordivsi);
    d.register("arith.ceildivsi", LatencyCategory::ComputeInt, ceildivsi);
    d.register("arith.ceildivui", LatencyCategory::ComputeInt, ceildivui);
    d.register("arith.remsi", LatencyCategory::ComputeInt, remsi);
    d.register("arith.remui", LatencyCategory::ComputeInt, remui);
    d.register("arith.minsi", LatencyCategory::ComputeInt, minsi);
    d.register("arith.maxsi", LatencyCategory::ComputeInt, maxsi);
    d.register("arith.minui", LatencyCategory::ComputeInt, minui);
    d.register("arith.maxui", LatencyCategory::ComputeInt, maxui);

    // Integer bitwise / shift
    d.register("arith.andi", LatencyCategory::ComputeInt, andi);
    d.register("arith.ori", LatencyCategory::ComputeInt, ori);
    d.register("arith.xori", LatencyCategory::ComputeInt, xori);
    d.register("arith.shli", LatencyCategory::ComputeInt, shli);
    d.register("arith.shrsi", LatencyCategory::ComputeInt, shrsi);
    d.register("arith.shrui", LatencyCategory::ComputeInt, shrui);

    // Integer comparison (Python uses COMPUTE_FLOAT for cmpi too).
    d.register("arith.cmpi", LatencyCategory::ComputeFloat, cmpi);

    // Select
    d.register("arith.select", LatencyCategory::ComputeFloat, select);

    // Constants & casts (latency-free).
    d.register("arith.constant", LatencyCategory::Zero, constant);
    d.register("arith.extf", LatencyCategory::Zero, extf);
    d.register("arith.truncf", LatencyCategory::Zero, truncf);
    d.register("arith.extsi", LatencyCategory::Zero, extsi);
    d.register("arith.extui", LatencyCategory::Zero, extui);
    d.register("arith.trunci", LatencyCategory::Zero, trunci);
    d.register("arith.sitofp", LatencyCategory::Zero, sitofp);
    d.register("arith.uitofp", LatencyCategory::Zero, uitofp);
    d.register("arith.fptosi", LatencyCategory::Zero, fptosi);
    d.register("arith.fptoui", LatencyCategory::Zero, fptoui);
    d.register("arith.index_cast", LatencyCategory::Zero, index_cast);
    d.register("arith.index_castui", LatencyCategory::Zero, index_cast);
    d.register("arith.convertf", LatencyCategory::Zero, convertf);
    d.register("arith.bitcast", LatencyCategory::Zero, bitcast);
}

// ===========================================================================
// Float binary ops
// ===========================================================================

fn addf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_float(op, ctx, "arith.addf", |a, b| a + b)
}

fn subf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_float(op, ctx, "arith.subf", |a, b| a - b)
}

fn mulf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_float(op, ctx, "arith.mulf", |a, b| a * b)
}

fn divf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_float(op, ctx, "arith.divf", |a, b| a / b)
}

fn remf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // operator.mod on floats: numpy/Python `%` — result takes divisor's sign.
    binary_float(op, ctx, "arith.remf", py_fmod)
}

// ===========================================================================
// Float unary ops
// ===========================================================================

fn negf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary_float(op, ctx, "arith.negf", |x| -x)
}

fn absf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary_float(op, ctx, "arith.absf", f32::abs)
}

// ===========================================================================
// Float min/max
// ===========================================================================

fn maxf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // np.maximum — NaN-propagating.
    binary_float(op, ctx, "arith.maxf", |a, b| {
        if a.is_nan() || b.is_nan() {
            f32::NAN
        } else if a >= b {
            a
        } else {
            b
        }
    })
}

fn maxnumf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // np.fmax — NaN non-propagating.
    binary_float(op, ctx, "arith.maxnumf", f32::max)
}

fn minf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_float(op, ctx, "arith.minf", |a, b| {
        if a.is_nan() || b.is_nan() {
            f32::NAN
        } else if a <= b {
            a
        } else {
            b
        }
    })
}

fn minnumf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_float(op, ctx, "arith.minnumf", f32::min)
}

// ===========================================================================
// Float comparison
// ===========================================================================

fn cmpf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let (a, b) = two_operands(op, ctx, "arith.cmpf")?;
    let pred = predicate(op, "arith.cmpf")?;
    let f = cmpf_fn(&pred)?;
    compare(a, b, "arith.cmpf", |x, y| f(x as f64, y as f64))
}

/// Resolve a cmpf predicate string to a comparator over (lhs, rhs) as f64.
/// Ordered (`o*`) follow NaN-false IEEE semantics; unordered (`u*`) OR with
/// "either NaN"; plus the constant `true`/`false` and `ord`/`uno`.
fn cmpf_fn(pred: &str) -> Result<fn(f64, f64) -> bool, String> {
    Ok(match pred {
        "false" => |_a, _b| false,
        "oeq" => |a, b| a == b,
        "ogt" => |a, b| a > b,
        "oge" => |a, b| a >= b,
        "olt" => |a, b| a < b,
        "ole" => |a, b| a <= b,
        "one" => |a: f64, b: f64| a != b && !(a.is_nan() || b.is_nan()),
        "ord" => |a: f64, b: f64| !(a.is_nan() || b.is_nan()),
        "ueq" => |a: f64, b: f64| a == b || a.is_nan() || b.is_nan(),
        "ugt" => |a: f64, b: f64| a > b || a.is_nan() || b.is_nan(),
        "uge" => |a: f64, b: f64| a >= b || a.is_nan() || b.is_nan(),
        "ult" => |a: f64, b: f64| a < b || a.is_nan() || b.is_nan(),
        "ule" => |a: f64, b: f64| a <= b || a.is_nan() || b.is_nan(),
        "une" => |a, b| a != b,
        "uno" => |a: f64, b: f64| a.is_nan() || b.is_nan(),
        "true" => |_a, _b| true,
        other => return Err(format!("arith.cmpf: unsupported predicate '{other}'")),
    })
}

// ===========================================================================
// Integer binary ops
// ===========================================================================

fn addi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.addi", |a, b| a.wrapping_add(b))
}

fn subi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.subi", |a, b| a.wrapping_sub(b))
}

fn muli(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.muli", |a, b| a.wrapping_mul(b))
}

fn divsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // MLIR divsi truncates toward zero (Rust `/` already does).
    binary_int(op, ctx, "arith.divsi", |a, b| a / b)
}

fn divui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Python uses floordiv; for the index ranges here operands are non-negative.
    binary_int(op, ctx, "arith.divui", py_floordiv)
}

fn floordivsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Python `//` floors toward -inf.
    binary_int(op, ctx, "arith.floordivsi", py_floordiv)
}

fn remsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // remsi is remainder after truncating division: a - trunc(a/b)*b (Rust `%`).
    binary_int(op, ctx, "arith.remsi", |a, b| a % b)
}

fn remui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Python `%` floors toward -inf (sign follows divisor).
    binary_int(op, ctx, "arith.remui", py_mod)
}

fn ceildivsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.ceildivsi", ceil_div)
}

fn ceildivui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.ceildivui", ceil_div)
}

fn minsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.minsi", i64::min)
}

fn maxsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.maxsi", i64::max)
}

fn minui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Operands are non-negative index values; unsigned min == signed min here.
    binary_int(op, ctx, "arith.minui", i64::min)
}

fn maxui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.maxui", i64::max)
}

// ===========================================================================
// Integer bitwise / shift
// ===========================================================================

fn andi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.andi", |a, b| a & b)
}

fn ori(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.ori", |a, b| a | b)
}

fn xori(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.xori", |a, b| a ^ b)
}

fn shli(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    binary_int(op, ctx, "arith.shli", |a, b| a << b)
}

fn shrsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Arithmetic (sign-preserving) right shift — Rust `>>` on i64.
    binary_int(op, ctx, "arith.shrsi", |a, b| a >> b)
}

fn shrui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Logical right shift — reinterpret as unsigned 32-bit (Python uses uint32).
    binary_int(op, ctx, "arith.shrui", |a, b| {
        ((a as u32) >> (b as u32)) as i64
    })
}

// ===========================================================================
// Integer comparison
// ===========================================================================

fn cmpi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let (a, b) = two_operands(op, ctx, "arith.cmpi")?;
    let pred = predicate(op, "arith.cmpi")?;
    let f = cmpi_fn(&pred)?;
    // Compare element values as integers. Unsigned predicates use the same
    // comparison as signed: the interpreter operates on non-negative index
    // integers, so sign-bit reinterpretation never occurs (matches Python).
    compare(a, b, "arith.cmpi", move |x, y| {
        f(round_i64(x), round_i64(y))
    })
}

fn cmpi_fn(pred: &str) -> Result<fn(i64, i64) -> bool, String> {
    Ok(match pred {
        "eq" => |a, b| a == b,
        "ne" => |a, b| a != b,
        "slt" | "ult" => |a, b| a < b,
        "sle" | "ule" => |a, b| a <= b,
        "sgt" | "ugt" => |a, b| a > b,
        "sge" | "uge" => |a, b| a >= b,
        other => return Err(format!("arith.cmpi: unsupported predicate '{other}'")),
    })
}

// ===========================================================================
// Select
// ===========================================================================

/// `%r = arith.select %cond, %t, %f`. Scalar cond picks one operand whole;
/// tile cond does element-wise `np.where`, taking the result dtype/shape from
/// whichever of true/false is a tile (mirrors the Python handler).
fn select(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.len() != 3 {
        return Err(format!(
            "arith.select expects 3 operands, got {}",
            op.operands.len()
        ));
    }
    let cond = ctx.get_value(&op.operands[0])?.clone();
    let true_val = ctx.get_value(&op.operands[1])?.clone();
    let false_val = ctx.get_value(&op.operands[2])?.clone();

    match cond {
        Value::Tile(c) => {
            // Element-wise selection. Materialize true/false to per-element data,
            // broadcasting a scalar operand across the condition's shape.
            let c_data = c.as_f32();
            let n = c_data.len();
            let t = elementwise_data(&true_val, n, "arith.select true")?;
            let f = elementwise_data(&false_val, n, "arith.select false")?;
            let data: Vec<f32> = (0..n)
                .map(|i| if c_data[i] != 0.0 { t[i] } else { f[i] })
                .collect();
            // dtype/shape come from whichever of true/false is a tile, else f16.
            let (dtype, shape) = match (&true_val, &false_val) {
                (Value::Tile(tt), _) => (tt.dtype, tt.shape.clone()),
                (_, Value::Tile(ff)) => (ff.dtype, ff.shape.clone()),
                _ => (DType::F16, c.shape.clone()),
            };
            Ok(Some(Value::Tile(Tile::compute(data, dtype, shape))))
        }
        Value::Scalar(Scalar::Bool(b)) => Ok(Some(if b { true_val } else { false_val })),
        Value::Scalar(s) => {
            // Truthiness of a numeric scalar (non-zero is true).
            let truthy = match s {
                Scalar::F32(v) => v != 0.0,
                Scalar::I32(v) => v != 0,
                Scalar::I64(v) => v != 0,
                Scalar::Bool(v) => v,
            };
            Ok(Some(if truthy { true_val } else { false_val }))
        }
        Value::Index(i) => Ok(Some(if i != 0 { true_val } else { false_val })),
        other => Err(format!("arith.select: bad condition kind {other:?}")),
    }
}

// ===========================================================================
// Constants & casts
// ===========================================================================

/// `%c = arith.constant <value> : <type>`. Scalar form binds the carried value;
/// the tensor form (`is_tensor`) splats a scalar across `shape`, or — for the
/// `dense<[..]>` list form (`dense_list`) — lays the per-element list out
/// directly. Mirrors `arith__constant`.
fn constant(
    op: &Operation,
    _ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let is_tensor = matches!(op.attributes.get("is_tensor"), Some(Attr::Bool(true)));
    if is_tensor {
        let shape: Vec<usize> = match op.attributes.get("shape") {
            Some(Attr::IntList(xs)) => xs.iter().map(|&n| n as usize).collect(),
            _ => return Err("arith.constant tensor: missing 'shape' attribute".into()),
        };
        let dtype = match op.attributes.get("dtype") {
            Some(Attr::Str(s)) => DType::parse(s)?,
            _ => DType::F16,
        };
        let n: usize = shape.iter().product();
        let dense_list = matches!(op.attributes.get("dense_list"), Some(Attr::Bool(true)));
        let data: Vec<f32> = if dense_list {
            // dense<[v0, v1, ...]>: each element distinct.
            match op.attributes.get("value") {
                Some(Attr::FloatList(vs)) => vs.iter().map(|&v| v as f32).collect(),
                Some(Attr::IntList(vs)) => vs.iter().map(|&v| v as f32).collect(),
                other => {
                    return Err(format!(
                        "arith.constant dense_list: bad 'value' attr {other:?}"
                    ));
                }
            }
        } else {
            // Splat a single value across the shape.
            let v = scalar_attr_f32(op)?;
            vec![v; n]
        };
        if data.len() != n {
            return Err(format!(
                "arith.constant: data length {} != product of shape {n}",
                data.len()
            ));
        }
        return Ok(Some(Value::Tile(Tile::compute(data, dtype, shape))));
    }

    // Scalar constant.
    let v = op
        .attributes
        .get("value")
        .ok_or("arith.constant missing 'value' attribute")?;
    let val = match v {
        Attr::Float(f) => Value::Scalar(Scalar::F32(*f as f32)),
        Attr::Int(i) => Value::Scalar(Scalar::I64(*i)),
        Attr::Bool(b) => Value::Scalar(Scalar::Bool(*b)),
        other => return Err(format!("arith.constant: bad value attr {other:?}")),
    };
    Ok(Some(val))
}

fn extf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Widen float (e.g. f16 -> f32). Storage is already f32; relabel tiles.
    cast_to_float(op, ctx, "arith.extf", DType::F32)
}

fn truncf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Narrow float (e.g. f32 -> f16). Round element values through f16.
    let v = unary_operand(op, ctx, "arith.truncf")?;
    match v {
        Value::Tile(t) => {
            let data: Vec<f32> = t
                .as_f32()
                .iter()
                .map(|&x| widen_f16(narrow_f16(x)))
                .collect();
            Ok(Some(Value::Tile(Tile::compute(data, DType::F16, t.shape))))
        }
        Value::Scalar(s) => {
            let x = s.as_f32().ok_or("arith.truncf: non-float scalar")?;
            Ok(Some(Value::Scalar(Scalar::F32(widen_f16(narrow_f16(x))))))
        }
        other => Err(format!("arith.truncf: bad operand {other:?}")),
    }
}

fn extsi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    cast_to_int(op, ctx, "arith.extsi", DType::I64)
}

fn extui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    cast_to_int(op, ctx, "arith.extui", DType::I64)
}

fn trunci(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    cast_to_int(op, ctx, "arith.trunci", DType::I32)
}

fn sitofp(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Convert signed int -> float, target dtype from result_type (default f32).
    let dtype = op
        .result_type
        .as_deref()
        .and_then(|s| DType::parse(s).ok())
        .unwrap_or(DType::F32);
    let v = unary_operand(op, ctx, "arith.sitofp")?;
    match v {
        Value::Tile(t) => {
            let data: Vec<f32> = t.as_f32().iter().map(|&x| round_i64(x) as f32).collect();
            Ok(Some(Value::Tile(Tile::compute(data, dtype, t.shape))))
        }
        Value::Scalar(s) => {
            let x = s.as_i64().ok_or("arith.sitofp: non-int scalar")? as f32;
            Ok(Some(Value::Scalar(Scalar::F32(x))))
        }
        Value::Index(i) => Ok(Some(Value::Scalar(Scalar::F32(i as f32)))),
        other => Err(format!("arith.sitofp: bad operand {other:?}")),
    }
}

/// `%r = arith.bitcast %x : <src> to <dst>` — reinterpret the bits, no value
/// change. Scalar 32-bit pairs only (`i32`<->`f32`), which covers the ±inf/NaN
/// bit-pattern idiom (`0xFF800000 : i32` -> `-inf : f32`). Tile bitcasts need
/// the dtype-faithful storage fork (see tile.rs) and are rejected for now.
fn bitcast(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // result_type is "<src> to <dst>"; take the destination spelling.
    let dst = op
        .result_type
        .as_deref()
        .and_then(|s| s.rsplit(" to ").next())
        .map(str::trim)
        .ok_or("arith.bitcast: missing 'to <type>' result type")?;
    let dst = DType::parse(dst)?;
    let v = unary_operand(op, ctx, "arith.bitcast")?;
    match v {
        Value::Tile(_) => Err(
            "arith.bitcast: tile bitcasts require dtype-faithful storage (tile.rs fork); \
             not yet supported"
                .into(),
        ),
        scalar => {
            // Extract the 32-bit source pattern (int as-is, float via to_bits).
            let bits: u32 = match &scalar {
                Value::Scalar(Scalar::F32(f)) => f.to_bits(),
                Value::Scalar(s) => {
                    s.as_i64().ok_or("arith.bitcast: non-numeric scalar")? as i32 as u32
                }
                Value::Index(i) => *i as i32 as u32,
                other => return Err(format!("arith.bitcast: bad operand {other:?}")),
            };
            let out = match dst {
                DType::F32 => Value::Scalar(Scalar::F32(f32::from_bits(bits))),
                DType::I32 => Value::Scalar(Scalar::I64(bits as i32 as i64)),
                other => {
                    return Err(format!(
                        "arith.bitcast: unsupported scalar target {other} (32-bit i32/f32 only)"
                    ));
                }
            };
            Ok(Some(out))
        }
    }
}

fn uitofp(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Unsigned int -> f32.
    let v = unary_operand(op, ctx, "arith.uitofp")?;
    match v {
        Value::Tile(t) => {
            let data: Vec<f32> = t.as_f32().iter().map(|&x| round_i64(x) as f32).collect();
            Ok(Some(Value::Tile(Tile::compute(data, DType::F32, t.shape))))
        }
        Value::Scalar(s) => {
            let x = s.as_i64().ok_or("arith.uitofp: non-int scalar")? as f32;
            Ok(Some(Value::Scalar(Scalar::F32(x))))
        }
        Value::Index(i) => Ok(Some(Value::Scalar(Scalar::F32(i as f32)))),
        other => Err(format!("arith.uitofp: bad operand {other:?}")),
    }
}

fn fptosi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Float -> signed int (truncation toward zero); tiles become i32.
    let v = unary_operand(op, ctx, "arith.fptosi")?;
    match v {
        Value::Tile(t) => {
            let data: Vec<f32> = t.as_f32().iter().map(|&x| x.trunc()).collect();
            Ok(Some(Value::Tile(Tile::compute(data, DType::I32, t.shape))))
        }
        Value::Scalar(s) => {
            let x = s.as_f32().ok_or("arith.fptosi: non-float scalar")?;
            Ok(Some(Value::Scalar(Scalar::I64(x.trunc() as i64))))
        }
        other => Err(format!("arith.fptosi: bad operand {other:?}")),
    }
}

fn fptoui(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Float -> unsigned int (truncation toward zero); tiles become ui32 (-> i32 here).
    let v = unary_operand(op, ctx, "arith.fptoui")?;
    match v {
        Value::Tile(t) => {
            let data: Vec<f32> = t.as_f32().iter().map(|&x| x.trunc()).collect();
            Ok(Some(Value::Tile(Tile::compute(data, DType::I32, t.shape))))
        }
        Value::Scalar(s) => {
            let x = s.as_f32().ok_or("arith.fptoui: non-float scalar")?;
            Ok(Some(Value::Scalar(Scalar::I64(x.trunc() as i64))))
        }
        other => Err(format!("arith.fptoui: bad operand {other:?}")),
    }
}

/// `arith.index_cast` / `index_castui` — coerce a scalar to an integer index.
fn index_cast(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let v = unary_operand(op, ctx, "arith.index_cast")?;
    let i = match v {
        Value::Index(i) => i,
        Value::Scalar(Scalar::I32(x)) => x as i64,
        Value::Scalar(Scalar::I64(x)) => x,
        Value::Scalar(Scalar::Bool(b)) => b as i64,
        Value::Scalar(Scalar::F32(x)) => x as i64,
        other => return Err(format!("arith.index_cast: bad operand {other:?}")),
    };
    Ok(Some(Value::Index(i)))
}

/// `arith.convertf` — float-to-float conversion; direction inferred from the
/// input dtype (f16 widens to f32, otherwise narrow to f16). Mirrors `convertf`.
fn convertf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let v = unary_operand(op, ctx, "arith.convertf")?;
    match v {
        Value::Tile(t) => {
            if t.dtype == DType::F16 {
                Ok(Some(Value::Tile(Tile::compute(
                    t.as_f32().to_vec(),
                    DType::F32,
                    t.shape,
                ))))
            } else {
                let data: Vec<f32> = t
                    .as_f32()
                    .iter()
                    .map(|&x| widen_f16(narrow_f16(x)))
                    .collect();
                Ok(Some(Value::Tile(Tile::compute(data, DType::F16, t.shape))))
            }
        }
        // Scalars carry no f16/f32 distinction here; pass the value through.
        Value::Scalar(Scalar::F32(x)) => Ok(Some(Value::Scalar(Scalar::F32(x)))),
        other => Err(format!("arith.convertf: bad operand {other:?}")),
    }
}

// ===========================================================================
// Generic binary/unary engines
// ===========================================================================

fn two_operands<'s>(
    op: &Operation,
    ctx: &'s CoreContext,
    name: &str,
) -> Result<(&'s Value, &'s Value), String> {
    if op.operands.len() != 2 {
        return Err(format!(
            "{name} expects 2 operands, got {}",
            op.operands.len()
        ));
    }
    let a = ctx.get_value(&op.operands[0])?;
    let b = ctx.get_value(&op.operands[1])?;
    Ok((a, b))
}

fn unary_operand(op: &Operation, ctx: &CoreContext, name: &str) -> Result<Value, String> {
    if op.operands.len() != 1 {
        return Err(format!(
            "{name} expects 1 operand, got {}",
            op.operands.len()
        ));
    }
    Ok(ctx.get_value(&op.operands[0])?.clone())
}

/// Float binary op accepting scalar/scalar, tile/tile, or mixed (scalar
/// broadcast across the tile). Mirrors `_float_binop`.
fn binary_float(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
    f: fn(f32, f32) -> f32,
) -> Result<Option<Value>, String> {
    let (a, b) = two_operands(op, ctx, name)?;
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => {
            let x = x
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar"))?;
            let y = y
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar"))?;
            Ok(Some(Value::Scalar(Scalar::F32(f(x, y)))))
        }
        (Value::Tile(x), Value::Tile(y)) => {
            let (lhs, rhs, out_shape) =
                broadcast_pair(&x.as_f32(), &x.shape, &y.as_f32(), &y.shape).ok_or_else(|| {
                    format!("{name}: shape mismatch {:?} vs {:?}", x.shape, y.shape)
                })?;
            let data: Vec<f32> = lhs.iter().zip(&rhs).map(|(&p, &q)| f(p, q)).collect();
            let dtype = result_float_dtype(x.dtype, y.dtype);
            Ok(Some(Value::Tile(Tile::compute(data, dtype, out_shape))))
        }
        (Value::Tile(x), Value::Scalar(y)) => {
            let y = y
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar"))?;
            let data: Vec<f32> = x.as_f32().iter().map(|&p| f(p, y)).collect();
            Ok(Some(Value::Tile(Tile::compute(
                data,
                x.dtype,
                x.shape.clone(),
            ))))
        }
        (Value::Scalar(x), Value::Tile(y)) => {
            let x = x
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar"))?;
            let data: Vec<f32> = y.as_f32().iter().map(|&q| f(x, q)).collect();
            Ok(Some(Value::Tile(Tile::compute(
                data,
                y.dtype,
                y.shape.clone(),
            ))))
        }
        _ => Err(format!("{name}: operand kinds not float-compatible")),
    }
}

fn unary_float(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
    f: fn(f32) -> f32,
) -> Result<Option<Value>, String> {
    let v = unary_operand(op, ctx, name)?;
    match v {
        Value::Scalar(s) => {
            let x = s
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar"))?;
            Ok(Some(Value::Scalar(Scalar::F32(f(x)))))
        }
        Value::Tile(t) => {
            let data: Vec<f32> = t.as_f32().iter().map(|&x| f(x)).collect();
            Ok(Some(Value::Tile(Tile::compute(data, t.dtype, t.shape))))
        }
        other => Err(format!("{name}: bad operand {other:?}")),
    }
}

/// Integer binary op accepting scalar/scalar, tile/tile, or mixed (scalar
/// broadcast). Mirrors `_int_binop` / the `ArithOps.*` scalar+Tile branches.
/// Element values round-trip through `i64`. Result dtype is the tile's dtype
/// (or `I64` for scalar/scalar, since Python returns a Python int).
fn binary_int(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
    f: fn(i64, i64) -> i64,
) -> Result<Option<Value>, String> {
    let (a, b) = two_operands(op, ctx, name)?;
    match (a, b) {
        (Value::Tile(x), Value::Tile(y)) => {
            let (lhs, rhs, out_shape) =
                broadcast_pair(&x.as_f32(), &x.shape, &y.as_f32(), &y.shape).ok_or_else(|| {
                    format!("{name}: shape mismatch {:?} vs {:?}", x.shape, y.shape)
                })?;
            let data: Vec<f32> = lhs
                .iter()
                .zip(&rhs)
                .map(|(&p, &q)| f(round_i64(p), round_i64(q)) as f32)
                .collect();
            Ok(Some(Value::Tile(Tile::compute(data, x.dtype, out_shape))))
        }
        (Value::Tile(x), _) => {
            let s = scalar_i64(b, name)?;
            let data: Vec<f32> = x
                .as_f32()
                .iter()
                .map(|&p| f(round_i64(p), s) as f32)
                .collect();
            Ok(Some(Value::Tile(Tile::compute(
                data,
                x.dtype,
                x.shape.clone(),
            ))))
        }
        (_, Value::Tile(y)) => {
            let s = scalar_i64(a, name)?;
            let data: Vec<f32> = y
                .as_f32()
                .iter()
                .map(|&q| f(s, round_i64(q)) as f32)
                .collect();
            Ok(Some(Value::Tile(Tile::compute(
                data,
                y.dtype,
                y.shape.clone(),
            ))))
        }
        _ => {
            let (x, y) = (scalar_i64(a, name)?, scalar_i64(b, name)?);
            Ok(Some(Value::Scalar(Scalar::I64(f(x, y)))))
        }
    }
}

/// Comparison engine for cmpi/cmpf. `cmp(lhs, rhs) -> bool` per element.
/// Scalar/scalar -> `Bool` scalar; any tile involved -> i1 tile (booleans
/// stored as 0.0/1.0), with the scalar operand broadcast.
fn compare(
    a: &Value,
    b: &Value,
    name: &str,
    cmp: impl Fn(f32, f32) -> bool,
) -> Result<Option<Value>, String> {
    let is_tile = matches!(a, Value::Tile(_)) || matches!(b, Value::Tile(_));
    if !is_tile {
        let x = scalar_f32_any(a, name)?;
        let y = scalar_f32_any(b, name)?;
        return Ok(Some(Value::Scalar(Scalar::Bool(cmp(x, y)))));
    }
    // Tile path. Two tiles broadcast NumPy-style to their common shape; a
    // tile/scalar pair broadcasts the scalar across the tile's shape.
    let (lhs, rhs, shape) = match (a, b) {
        (Value::Tile(x), Value::Tile(y)) => {
            broadcast_pair(&x.as_f32(), &x.shape, &y.as_f32(), &y.shape)
                .ok_or_else(|| format!("{name}: shape mismatch {:?} vs {:?}", x.shape, y.shape))?
        }
        (Value::Tile(t), _) => {
            let n = t.len();
            (
                t.as_f32().to_vec(),
                elementwise_data(b, n, name)?,
                t.shape.clone(),
            )
        }
        (_, Value::Tile(t)) => {
            let n = t.len();
            (
                elementwise_data(a, n, name)?,
                t.as_f32().to_vec(),
                t.shape.clone(),
            )
        }
        _ => unreachable!(),
    };
    let data: Vec<f32> = lhs
        .iter()
        .zip(&rhs)
        .map(|(&p, &q)| if cmp(p, q) { 1.0 } else { 0.0 })
        .collect();
    Ok(Some(Value::Tile(Tile::compute(data, DType::Bool, shape))))
}

/// NumPy-style broadcast of two tiles (`(data, shape)` each) to a common shape.
/// Right-aligns ranks; each axis must match or be 1. Returns the two expanded
/// row-major buffers and the broadcast shape, or `None` if incompatible.
///
/// This is what makes `arith.*` element-wise ops over two differently-shaped
/// broadcast tiles work — e.g. paged_attention's causal mask compares a row
/// index tile `[8,1]` against a column index tile `[1,16]`, which numpy fuses to
/// `[8,16]`. Without it the tile/tile arms required identical shapes.
fn broadcast_pair(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Option<(Vec<f32>, Vec<f32>, Vec<usize>)> {
    let rank = a_shape.len().max(b_shape.len());
    // Left-pad each shape with 1s to the common rank.
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1usize; rank - s.len()];
        v.extend_from_slice(s);
        v
    };
    let as_ = pad(a_shape);
    let bs = pad(b_shape);
    let mut out = vec![0usize; rank];
    for d in 0..rank {
        out[d] = match (as_[d], bs[d]) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    let expand = |data: &[f32], src: &[usize]| -> Vec<f32> {
        let total: usize = out.iter().product();
        let out_strides = row_major_strides(&out);
        let src_strides = row_major_strides(src);
        let mut res = vec![0.0f32; total];
        for (lin, slot) in res.iter_mut().enumerate() {
            let mut src_off = 0usize;
            for d in 0..rank {
                let coord = (lin / out_strides[d]) % out[d];
                // Broadcast axis (src extent 1) contributes index 0.
                let sc = if src[d] == 1 { 0 } else { coord };
                src_off += sc * src_strides[d];
            }
            *slot = data[src_off];
        }
        res
    };
    Some((expand(a, &as_), expand(b, &bs), out))
}

/// Row-major (C-order) element strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Materialize a value as `n` per-element f32s: a tile's data verbatim, or a
/// scalar broadcast to length `n`.
fn elementwise_data(v: &Value, n: usize, name: &str) -> Result<Vec<f32>, String> {
    match v {
        Value::Tile(t) => {
            if t.len() != n {
                return Err(format!(
                    "{name}: tile length {} != broadcast length {n}",
                    t.len()
                ));
            }
            Ok(t.as_f32().to_vec())
        }
        Value::Scalar(_) | Value::Index(_) => Ok(vec![scalar_f32_any(v, name)?; n]),
        other => Err(format!("{name}: bad operand {other:?}")),
    }
}

// ===========================================================================
// Cast helpers
// ===========================================================================

fn cast_to_float(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
    dtype: DType,
) -> Result<Option<Value>, String> {
    let v = unary_operand(op, ctx, name)?;
    match v {
        Value::Tile(t) => Ok(Some(Value::Tile(Tile::compute(
            t.as_f32().to_vec(),
            dtype,
            t.shape,
        )))),
        Value::Scalar(s) => {
            let x = s
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar"))?;
            Ok(Some(Value::Scalar(Scalar::F32(x))))
        }
        other => Err(format!("{name}: bad operand {other:?}")),
    }
}

fn cast_to_int(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
    dtype: DType,
) -> Result<Option<Value>, String> {
    let v = unary_operand(op, ctx, name)?;
    match v {
        Value::Tile(t) => {
            let data: Vec<f32> = t.as_f32().iter().map(|&x| round_i64(x) as f32).collect();
            Ok(Some(Value::Tile(Tile::compute(data, dtype, t.shape))))
        }
        Value::Scalar(s) => {
            let i = s
                .as_i64()
                .ok_or_else(|| format!("{name}: non-int scalar"))?;
            Ok(Some(Value::Scalar(Scalar::I64(i))))
        }
        Value::Index(i) => Ok(Some(Value::Scalar(Scalar::I64(i)))),
        other => Err(format!("{name}: bad operand {other:?}")),
    }
}

// ===========================================================================
// Small numeric helpers
// ===========================================================================

fn scalar_i64(v: &Value, name: &str) -> Result<i64, String> {
    match v {
        Value::Scalar(s) => s.as_i64().ok_or_else(|| format!("{name}: non-int scalar")),
        Value::Index(i) => Ok(*i),
        _ => Err(format!("{name}: expected scalar/index operand")),
    }
}

/// Coerce any numeric scalar/index to f32 (for comparison operands).
fn scalar_f32_any(v: &Value, name: &str) -> Result<f32, String> {
    match v {
        Value::Scalar(Scalar::F32(x)) => Ok(*x),
        Value::Scalar(Scalar::I32(x)) => Ok(*x as f32),
        Value::Scalar(Scalar::I64(x)) => Ok(*x as f32),
        Value::Scalar(Scalar::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Index(i) => Ok(*i as f32),
        other => Err(format!("{name}: expected numeric scalar, got {other:?}")),
    }
}

/// Read the scalar `value` attr of a constant as f32 (splat fill).
fn scalar_attr_f32(op: &Operation) -> Result<f32, String> {
    match op.attributes.get("value") {
        Some(Attr::Float(f)) => Ok(*f as f32),
        Some(Attr::Int(i)) => Ok(*i as f32),
        Some(Attr::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        None => Ok(0.0), // Python defaults the missing value attr to 0.
        other => Err(format!("arith.constant: bad scalar value attr {other:?}")),
    }
}

fn predicate(op: &Operation, name: &str) -> Result<String, String> {
    match op.attributes.get("predicate") {
        Some(Attr::Str(s)) => Ok(s.clone()),
        _ => Err(format!("{name}: missing 'predicate' attribute")),
    }
}

fn result_float_dtype(a: DType, b: DType) -> DType {
    if a == DType::F16 || b == DType::F16 {
        DType::F16
    } else {
        DType::F32
    }
}

/// Round a stored f32 element to its integer value (storage is f32; element
/// values for integer tiles are exact integers).
fn round_i64(x: f32) -> i64 {
    x.round() as i64
}

/// Python `//` floor division (rounds toward -inf), avoiding /0 wrap.
fn py_floordiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Python `%` modulo (sign follows the divisor).
fn py_mod(a: i64, b: i64) -> i64 {
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        r + b
    } else {
        r
    }
}

/// Ceiling division for signed/unsigned integers (np.ceil(a / b)).
fn ceil_div(a: i64, b: i64) -> i64 {
    (a as f64 / b as f64).ceil() as i64
}

/// Python float `%` (numpy `np.mod`): result takes the divisor's sign.
fn py_fmod(a: f32, b: f32) -> f32 {
    let r = a % b;
    if r != 0.0 && ((r < 0.0) != (b < 0.0)) {
        r + b
    } else {
        r
    }
}

/// Standard f32 -> f16 conversion (round-to-nearest-even), returning the f16
/// bit pattern. Delegates to the crate codec so truncf/convertf-to-f16 round
/// IDENTICALLY to how `ktdp.store` / `ktdp.load` encode/decode f16 (the codec is
/// the single source of truth, incl. correct subnormal handling — the previous
/// hand-rolled version mis-normalized subnormals, halving values at the
/// normal/subnormal boundary, e.g. f16 bits 1022 ≈ 6.09e-5 read back as 3.05e-5).
fn narrow_f16(x: f32) -> u16 {
    crate::codec::f32_to_f16_bits(x)
}

/// Convert an f16 bit pattern back to f32 — the codec's table-backed widen.
fn widen_f16(h: u16) -> f32 {
    crate::codec::f16_bits_to_f32(h)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::single_core_context;
    use std::collections::HashMap;

    /// Build an env + ctx, seed operands, run one op, return its produced value.
    fn run_op(op: &Operation, seed: &[(&str, Value)]) -> Value {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        for (n, v) in seed {
            ctx.set_value(n, v.clone());
        }
        let handler = dispatch.handler(&op.op_type).expect("handler registered");
        handler(op, &mut ctx, &env)
            .unwrap()
            .expect("op produced a value")
    }

    fn f32s(name: &str, op_ty: &str, ops: &[&str]) -> Operation {
        Operation::new(Some(name), op_ty, ops)
    }

    fn sf(x: f32) -> Value {
        Value::Scalar(Scalar::F32(x))
    }
    fn si(x: i64) -> Value {
        Value::Scalar(Scalar::I64(x))
    }
    fn tile(data: Vec<f32>, dt: DType, shape: Vec<usize>) -> Value {
        Value::Tile(Tile::compute(data, dt, shape))
    }

    fn expect_f32(v: &Value) -> f32 {
        match v {
            Value::Scalar(Scalar::F32(x)) => *x,
            other => panic!("expected F32, got {other:?}"),
        }
    }
    fn expect_i64(v: &Value) -> i64 {
        match v {
            Value::Scalar(Scalar::I64(x)) => *x,
            other => panic!("expected I64, got {other:?}"),
        }
    }
    fn expect_bool(v: &Value) -> bool {
        match v {
            Value::Scalar(Scalar::Bool(b)) => *b,
            other => panic!("expected Bool, got {other:?}"),
        }
    }
    fn expect_tile(v: &Value) -> &Tile {
        match v {
            Value::Tile(t) => t,
            other => panic!("expected Tile, got {other:?}"),
        }
    }

    // --- float binary ------------------------------------------------------

    #[test]
    fn float_binops_scalar() {
        let seed = [("%a", sf(6.0)), ("%b", sf(4.0))];
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.addf", &["%a", "%b"]), &seed)),
            10.0
        );
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.subf", &["%a", "%b"]), &seed)),
            2.0
        );
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.mulf", &["%a", "%b"]), &seed)),
            24.0
        );
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.divf", &["%a", "%b"]), &seed)),
            1.5
        );
    }

    #[test]
    fn remf_takes_divisor_sign() {
        let seed = [("%a", sf(-7.0)), ("%b", sf(3.0))];
        // numpy mod: -7 % 3 == 2.0
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.remf", &["%a", "%b"]), &seed)),
            2.0
        );
    }

    #[test]
    fn float_binop_elementwise_tile() {
        let seed = [
            ("%a", tile(vec![1.0, 2.0, 3.0], DType::F32, vec![3])),
            ("%b", tile(vec![10.0, 20.0, 30.0], DType::F32, vec![3])),
        ];
        let r = run_op(&f32s("%r", "arith.addf", &["%a", "%b"]), &seed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn float_binop_mixed_scalar_tile_broadcasts() {
        let seed = [
            ("%a", tile(vec![1.0, 2.0, 3.0], DType::F32, vec![3])),
            ("%b", sf(10.0)),
        ];
        let r = run_op(&f32s("%r", "arith.addf", &["%a", "%b"]), &seed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![11.0, 12.0, 13.0]);
        // scalar-on-left broadcasts too
        let seed2 = [
            ("%a", sf(10.0)),
            ("%b", tile(vec![1.0, 2.0], DType::F32, vec![2])),
        ];
        let r2 = run_op(&f32s("%r", "arith.subf", &["%a", "%b"]), &seed2);
        assert_eq!(expect_tile(&r2).as_f32().to_vec(), vec![9.0, 8.0]);
    }

    // --- float unary -------------------------------------------------------

    #[test]
    fn negf_and_absf() {
        let seed = [("%a", sf(-3.5))];
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.negf", &["%a"]), &seed)),
            3.5
        );
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.absf", &["%a"]), &seed)),
            3.5
        );
        let tseed = [("%a", tile(vec![-1.0, 2.0, -3.0], DType::F32, vec![3]))];
        let r = run_op(&f32s("%r", "arith.absf", &["%a"]), &tseed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    // --- min/max -----------------------------------------------------------

    #[test]
    fn maxf_minf_propagate_nan() {
        let seed = [("%a", sf(f32::NAN)), ("%b", sf(1.0))];
        assert!(expect_f32(&run_op(&f32s("%r", "arith.maximumf", &["%a", "%b"]), &seed)).is_nan());
        assert!(expect_f32(&run_op(&f32s("%r", "arith.minimumf", &["%a", "%b"]), &seed)).is_nan());
        // numf variants ignore NaN
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.maxnumf", &["%a", "%b"]), &seed)),
            1.0
        );
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.minnumf", &["%a", "%b"]), &seed)),
            1.0
        );
    }

    #[test]
    fn maxf_minf_pick_extreme() {
        let seed = [("%a", sf(2.0)), ("%b", sf(5.0))];
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.maxf", &["%a", "%b"]), &seed)),
            5.0
        );
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.minf", &["%a", "%b"]), &seed)),
            2.0
        );
    }

    // --- cmpf --------------------------------------------------------------

    fn cmpf_op(pred: &str, ops: &[&str]) -> Operation {
        Operation::new(Some("%r"), "arith.cmpf", ops).with_attr("predicate", Attr::Str(pred.into()))
    }

    #[test]
    fn cmpf_ordered_predicates() {
        let seed = [("%a", sf(1.0)), ("%b", sf(2.0))];
        assert!(expect_bool(&run_op(&cmpf_op("olt", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(&cmpf_op("ogt", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(&cmpf_op("oeq", &["%a", "%b"]), &seed)));
        assert!(expect_bool(&run_op(&cmpf_op("one", &["%a", "%b"]), &seed)));
    }

    #[test]
    fn cmpf_nan_ordered_vs_unordered() {
        let seed = [("%a", sf(f32::NAN)), ("%b", sf(1.0))];
        // ordered comparisons with NaN are false
        assert!(!expect_bool(&run_op(&cmpf_op("oeq", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(&cmpf_op("olt", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(&cmpf_op("one", &["%a", "%b"]), &seed)));
        // unordered comparisons with NaN are true
        assert!(expect_bool(&run_op(&cmpf_op("ult", &["%a", "%b"]), &seed)));
        assert!(expect_bool(&run_op(&cmpf_op("ueq", &["%a", "%b"]), &seed)));
        assert!(expect_bool(&run_op(&cmpf_op("uno", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(&cmpf_op("ord", &["%a", "%b"]), &seed)));
        // une is true even with NaN
        assert!(expect_bool(&run_op(&cmpf_op("une", &["%a", "%b"]), &seed)));
    }

    #[test]
    fn cmpf_true_false_constants() {
        let seed = [("%a", sf(1.0)), ("%b", sf(2.0))];
        assert!(expect_bool(&run_op(&cmpf_op("true", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(
            &cmpf_op("false", &["%a", "%b"]),
            &seed
        )));
    }

    #[test]
    fn cmpf_tile_produces_i1_tile() {
        let seed = [
            ("%a", tile(vec![1.0, 5.0, 3.0], DType::F32, vec![3])),
            ("%b", tile(vec![2.0, 2.0, 3.0], DType::F32, vec![3])),
        ];
        let r = run_op(&cmpf_op("olt", &["%a", "%b"]), &seed);
        let t = expect_tile(&r);
        assert_eq!(t.dtype, DType::Bool);
        assert_eq!(t.as_f32().to_vec(), vec![1.0, 0.0, 0.0]);
    }

    // --- integer binary ----------------------------------------------------

    #[test]
    fn int_binops_scalar() {
        let seed = [("%a", si(17)), ("%b", si(5))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.addi", &["%a", "%b"]), &seed)),
            22
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.subi", &["%a", "%b"]), &seed)),
            12
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.muli", &["%a", "%b"]), &seed)),
            85
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.divsi", &["%a", "%b"]), &seed)),
            3
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.remsi", &["%a", "%b"]), &seed)),
            2
        );
    }

    #[test]
    fn divsi_truncates_toward_zero_remsi_matches() {
        // -7 / 2: divsi truncates -> -3 ; remsi = -7 - (-3*2) = -1
        let seed = [("%a", si(-7)), ("%b", si(2))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.divsi", &["%a", "%b"]), &seed)),
            -3
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.remsi", &["%a", "%b"]), &seed)),
            -1
        );
    }

    #[test]
    fn floordivsi_floors_toward_neg_inf() {
        // -7 // 2 floors -> -4
        let seed = [("%a", si(-7)), ("%b", si(2))];
        assert_eq!(
            expect_i64(&run_op(
                &f32s("%r", "arith.floordivsi", &["%a", "%b"]),
                &seed
            )),
            -4
        );
    }

    #[test]
    fn divui_remui_nonneg() {
        let seed = [("%a", si(17)), ("%b", si(5))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.divui", &["%a", "%b"]), &seed)),
            3
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.remui", &["%a", "%b"]), &seed)),
            2
        );
    }

    #[test]
    fn ceildiv_rounds_up() {
        let seed = [("%a", si(7)), ("%b", si(2))];
        assert_eq!(
            expect_i64(&run_op(
                &f32s("%r", "arith.ceildivsi", &["%a", "%b"]),
                &seed
            )),
            4
        );
        assert_eq!(
            expect_i64(&run_op(
                &f32s("%r", "arith.ceildivui", &["%a", "%b"]),
                &seed
            )),
            4
        );
    }

    #[test]
    fn int_min_max() {
        let seed = [("%a", si(3)), ("%b", si(8))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.minsi", &["%a", "%b"]), &seed)),
            3
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.maxsi", &["%a", "%b"]), &seed)),
            8
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.minui", &["%a", "%b"]), &seed)),
            3
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.maxui", &["%a", "%b"]), &seed)),
            8
        );
    }

    #[test]
    fn int_binop_elementwise_and_broadcast() {
        let seed = [
            ("%a", tile(vec![1.0, 2.0, 3.0], DType::I32, vec![3])),
            ("%b", tile(vec![4.0, 5.0, 6.0], DType::I32, vec![3])),
        ];
        let r = run_op(&f32s("%r", "arith.addi", &["%a", "%b"]), &seed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![5.0, 7.0, 9.0]);
        // scalar broadcast
        let seed2 = [
            ("%a", tile(vec![1.0, 2.0, 3.0], DType::I32, vec![3])),
            ("%b", si(10)),
        ];
        let r2 = run_op(&f32s("%r", "arith.muli", &["%a", "%b"]), &seed2);
        assert_eq!(expect_tile(&r2).as_f32().to_vec(), vec![10.0, 20.0, 30.0]);
    }

    // --- bitwise / shift ---------------------------------------------------

    #[test]
    fn bitwise_ops() {
        let seed = [("%a", si(0b1100)), ("%b", si(0b1010))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.andi", &["%a", "%b"]), &seed)),
            0b1000
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.ori", &["%a", "%b"]), &seed)),
            0b1110
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.xori", &["%a", "%b"]), &seed)),
            0b0110
        );
    }

    #[test]
    fn shift_ops() {
        let seed = [("%a", si(1)), ("%b", si(4))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.shli", &["%a", "%b"]), &seed)),
            16
        );
        let seed2 = [("%a", si(256)), ("%b", si(2))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.shrsi", &["%a", "%b"]), &seed2)),
            64
        );
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.shrui", &["%a", "%b"]), &seed2)),
            64
        );
    }

    // --- cmpi --------------------------------------------------------------

    fn cmpi_op(pred: &str, ops: &[&str]) -> Operation {
        Operation::new(Some("%r"), "arith.cmpi", ops).with_attr("predicate", Attr::Str(pred.into()))
    }

    #[test]
    fn cmpi_predicates_scalar() {
        let seed = [("%a", si(3)), ("%b", si(5))];
        assert!(expect_bool(&run_op(&cmpi_op("slt", &["%a", "%b"]), &seed)));
        assert!(expect_bool(&run_op(&cmpi_op("ult", &["%a", "%b"]), &seed)));
        assert!(!expect_bool(&run_op(&cmpi_op("sge", &["%a", "%b"]), &seed)));
        assert!(expect_bool(&run_op(&cmpi_op("ne", &["%a", "%b"]), &seed)));
        let eqseed = [("%a", si(5)), ("%b", si(5))];
        assert!(expect_bool(&run_op(&cmpi_op("eq", &["%a", "%b"]), &eqseed)));
        assert!(expect_bool(&run_op(
            &cmpi_op("sle", &["%a", "%b"]),
            &eqseed
        )));
    }

    #[test]
    fn cmpi_tile_produces_i1_tile() {
        let seed = [
            ("%a", tile(vec![1.0, 5.0, 3.0], DType::I32, vec![3])),
            ("%b", tile(vec![2.0, 2.0, 3.0], DType::I32, vec![3])),
        ];
        let r = run_op(&cmpi_op("sge", &["%a", "%b"]), &seed);
        let t = expect_tile(&r);
        assert_eq!(t.dtype, DType::Bool);
        assert_eq!(t.as_f32().to_vec(), vec![0.0, 1.0, 1.0]);
    }

    // --- select ------------------------------------------------------------

    #[test]
    fn select_scalar_cond() {
        let t = Operation::new(Some("%r"), "arith.select", &["%c", "%t", "%f"]);
        let seed_true = [
            ("%c", Value::Scalar(Scalar::Bool(true))),
            ("%t", si(1)),
            ("%f", si(2)),
        ];
        assert_eq!(expect_i64(&run_op(&t, &seed_true)), 1);
        let seed_false = [
            ("%c", Value::Scalar(Scalar::Bool(false))),
            ("%t", si(1)),
            ("%f", si(2)),
        ];
        assert_eq!(expect_i64(&run_op(&t, &seed_false)), 2);
    }

    #[test]
    fn select_tile_cond_elementwise() {
        let op = Operation::new(Some("%r"), "arith.select", &["%c", "%t", "%f"]);
        let seed = [
            ("%c", tile(vec![1.0, 0.0, 1.0], DType::Bool, vec![3])),
            ("%t", tile(vec![10.0, 20.0, 30.0], DType::F32, vec![3])),
            ("%f", tile(vec![-1.0, -2.0, -3.0], DType::F32, vec![3])),
        ];
        let r = run_op(&op, &seed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![10.0, -2.0, 30.0]);
    }

    #[test]
    fn select_tile_cond_scalar_branches_broadcast() {
        let op = Operation::new(Some("%r"), "arith.select", &["%c", "%t", "%f"]);
        let seed = [
            ("%c", tile(vec![1.0, 0.0], DType::Bool, vec![2])),
            ("%t", sf(7.0)),
            ("%f", sf(9.0)),
        ];
        let r = run_op(&op, &seed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![7.0, 9.0]);
    }

    // --- constant ----------------------------------------------------------

    #[test]
    fn constant_scalar_forms() {
        let cf =
            Operation::new(Some("%r"), "arith.constant", &[]).with_attr("value", Attr::Float(2.5));
        assert_eq!(expect_f32(&run_op(&cf, &[])), 2.5);
        let ci =
            Operation::new(Some("%r"), "arith.constant", &[]).with_attr("value", Attr::Int(42));
        assert_eq!(expect_i64(&run_op(&ci, &[])), 42);
        let cb =
            Operation::new(Some("%r"), "arith.constant", &[]).with_attr("value", Attr::Bool(true));
        assert!(expect_bool(&run_op(&cb, &[])));
    }

    #[test]
    fn constant_splat_tensor() {
        let mut attrs = HashMap::new();
        attrs.insert("value".to_string(), Attr::Float(3.0));
        attrs.insert("is_tensor".to_string(), Attr::Bool(true));
        attrs.insert("shape".to_string(), Attr::IntList(vec![4]));
        attrs.insert("dtype".to_string(), Attr::Str("f16".into()));
        let op = Operation {
            result: Some("%r".into()),
            op_type: "arith.constant".into(),
            operands: vec![],
            attributes: attrs,
            result_type: None,
            regions: vec![],
        };
        let r = run_op(&op, &[]);
        let t = expect_tile(&r);
        assert_eq!(t.as_f32().to_vec(), vec![3.0, 3.0, 3.0, 3.0]);
        assert_eq!(t.dtype, DType::F16);
        assert_eq!(t.shape, vec![4]);
    }

    #[test]
    fn constant_dense_list_tensor() {
        let mut attrs = HashMap::new();
        attrs.insert("value".to_string(), Attr::IntList(vec![16, 32]));
        attrs.insert("is_tensor".to_string(), Attr::Bool(true));
        attrs.insert("dense_list".to_string(), Attr::Bool(true));
        attrs.insert("shape".to_string(), Attr::IntList(vec![2]));
        attrs.insert("dtype".to_string(), Attr::Str("index".into()));
        let op = Operation {
            result: Some("%r".into()),
            op_type: "arith.constant".into(),
            operands: vec![],
            attributes: attrs,
            result_type: None,
            regions: vec![],
        };
        let t = run_op(&op, &[]);
        assert_eq!(expect_tile(&t).as_f32().to_vec(), vec![16.0, 32.0]);
    }

    // --- casts -------------------------------------------------------------

    #[test]
    fn extf_truncf_roundtrip() {
        // extf scalar passes value through (widening is a no-op on f32 storage).
        let seed = [("%a", sf(1.5))];
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.extf", &["%a"]), &seed)),
            1.5
        );
        // truncf on a representable f16 value is exact.
        assert_eq!(
            expect_f32(&run_op(&f32s("%r", "arith.truncf", &["%a"]), &seed)),
            1.5
        );
    }

    #[test]
    fn truncf_rounds_to_f16_precision() {
        // 1 + 1/2048 is the exact midpoint between 1.0 and 1+1/1024 and ties to
        // even -> 1.0; 1 + 1/1024 is exactly representable.
        let seed = [("%a", sf(1.0 + 1.0 / 2048.0))];
        let r = expect_f32(&run_op(&f32s("%r", "arith.truncf", &["%a"]), &seed));
        assert_eq!(r, 1.0);
        let seed2 = [("%a", sf(1.0 + 1.0 / 1024.0))];
        let r2 = expect_f32(&run_op(&f32s("%r", "arith.truncf", &["%a"]), &seed2));
        assert_eq!(r2, 1.0 + 1.0 / 1024.0);
    }

    #[test]
    fn extsi_extui_trunci_tiles() {
        let seed = [("%a", tile(vec![1.0, 2.0, 3.0], DType::I32, vec![3]))];
        let r = run_op(&f32s("%r", "arith.extsi", &["%a"]), &seed);
        assert_eq!(expect_tile(&r).dtype, DType::I64);
        let r2 = run_op(&f32s("%r", "arith.trunci", &["%a"]), &seed);
        assert_eq!(expect_tile(&r2).dtype, DType::I32);
        assert_eq!(expect_tile(&r2).as_f32().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn sitofp_with_result_type() {
        let op = Operation {
            result: Some("%r".into()),
            op_type: "arith.sitofp".into(),
            operands: vec!["%a".into()],
            attributes: HashMap::new(),
            result_type: Some("f32".into()),
            regions: vec![],
        };
        let seed = [("%a", tile(vec![5.0, 7.0], DType::I32, vec![2]))];
        let r = run_op(&op, &seed);
        assert_eq!(expect_tile(&r).dtype, DType::F32);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![5.0, 7.0]);
        // scalar path
        let sseed = [("%a", si(9))];
        assert_eq!(expect_f32(&run_op(&op, &sseed)), 9.0);
    }

    #[test]
    fn fptosi_truncates_toward_zero() {
        let seed = [("%a", sf(-2.7))];
        assert_eq!(
            expect_i64(&run_op(&f32s("%r", "arith.fptosi", &["%a"]), &seed)),
            -2
        );
        let tseed = [("%a", tile(vec![1.9, -1.9, 2.5], DType::F32, vec![3]))];
        let r = run_op(&f32s("%r", "arith.fptosi", &["%a"]), &tseed);
        assert_eq!(expect_tile(&r).as_f32().to_vec(), vec![1.0, -1.0, 2.0]);
        assert_eq!(expect_tile(&r).dtype, DType::I32);
    }

    #[test]
    fn index_cast_coerces_to_index() {
        let seed = [("%a", si(7))];
        match run_op(&f32s("%r", "arith.index_cast", &["%a"]), &seed) {
            Value::Index(7) => {}
            other => panic!("expected Index(7), got {other:?}"),
        }
    }

    #[test]
    fn convertf_tile_direction() {
        // f16 tile widens to f32
        let seed = [("%a", tile(vec![1.0, 2.0], DType::F16, vec![2]))];
        let r = run_op(&f32s("%r", "arith.convertf", &["%a"]), &seed);
        assert_eq!(expect_tile(&r).dtype, DType::F32);
        // f32 tile narrows to f16
        let seed2 = [("%a", tile(vec![1.0, 2.0], DType::F32, vec![2]))];
        let r2 = run_op(&f32s("%r", "arith.convertf", &["%a"]), &seed2);
        assert_eq!(expect_tile(&r2).dtype, DType::F16);
    }

    // --- f16 round-trip sanity --------------------------------------------

    #[test]
    fn f16_roundtrip_exact_values() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 2.0, 1024.0, -0.25] {
            assert_eq!(widen_f16(narrow_f16(v)), v, "roundtrip failed for {v}");
        }
    }
}
