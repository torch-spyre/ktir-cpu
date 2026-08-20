// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `math` dialect handlers — port of `ktir_emulator/dialects/math_ops.py` and the
//! `MathOps` compute helpers in `ktir_emulator/ops/math_ops.py`.
//!
//! Every op is element-wise: a unary (or binary/ternary for `powf`/`fma`) math
//! function applied across a whole tile, or to a single scalar. This mirrors how
//! the Python `_unary` helper accepts either a NumPy array (Tile) or a Python
//! scalar and dispatches to `tile_fn` / `scalar_fn`.
//!
//! Transcendental ops (`exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`,
//! `erf`, `powf`, …) register under `LatencyCategory::ComputeTranscendental`;
//! the cheaper rounding/abs ops (`absf`, `absi`, `floor`, `ceil`, `fma`) use
//! `LatencyCategory::ComputeFloat`, exactly matching the Python latency tags on
//! the `@register(...)` decorators.
//!
//! STORAGE NOTE: tile data is `Vec<f32>` (the slice-1 decision in `tile.rs`).
//! The Python code computes in float32 then rounds back to the tile's dtype
//! (`.astype(tile.data.dtype)`); for f16 tiles we reproduce that round-trip by
//! rounding each result through IEEE-754 binary16 at the f16 boundary so parity
//! holds where f16 rounding bites.

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Scalar, Value};
use crate::tile::Tile;

/// Register every handler this module owns. Called by `Dispatch::new`.
///
/// Latency categories are kept in lockstep with the `@register(...)` decorators
/// in `ktir_emulator/dialects/math_ops.py`.
pub fn register(d: &mut Dispatch) {
    // Transcendental — the expensive special functions.
    d.register("math.exp", LatencyCategory::ComputeTranscendental, exp);
    d.register("math.sqrt", LatencyCategory::ComputeTranscendental, sqrt);
    d.register("math.rsqrt", LatencyCategory::ComputeTranscendental, rsqrt);
    d.register("math.log", LatencyCategory::ComputeTranscendental, log);
    d.register("math.log2", LatencyCategory::ComputeTranscendental, log2);
    d.register("math.log1p", LatencyCategory::ComputeTranscendental, log1p);
    d.register("math.tanh", LatencyCategory::ComputeTranscendental, tanh);
    d.register("math.sin", LatencyCategory::ComputeTranscendental, sin);
    d.register("math.cos", LatencyCategory::ComputeTranscendental, cos);
    d.register("math.erf", LatencyCategory::ComputeTranscendental, erf);
    d.register("math.powf", LatencyCategory::ComputeTranscendental, powf);
    // Cheap float ops — abs / rounding / fused multiply-add.
    d.register("math.absf", LatencyCategory::ComputeFloat, absf);
    d.register("math.absi", LatencyCategory::ComputeFloat, absi);
    d.register("math.ceil", LatencyCategory::ComputeFloat, ceil);
    d.register("math.floor", LatencyCategory::ComputeFloat, floor);
    d.register("math.fma", LatencyCategory::ComputeFloat, fma);
}

// --- unary handlers -------------------------------------------------------
//
// Each is a thin wrapper that names the op (for error messages) and hands a
// pure `f32 -> f32` kernel to `unary`. The kernels are the exact element-wise
// functions `MathOps.exp` / `MathOps.exp_scalar` etc. apply via NumPy.

fn exp(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.exp", |x| x.exp())
}

fn sqrt(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.sqrt", |x| x.sqrt())
}

fn rsqrt(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // 1.0 / sqrt(x), matching `MathOps.rsqrt`.
    unary(op, ctx, "math.rsqrt", |x| 1.0 / x.sqrt())
}

fn log(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.log", |x| x.ln())
}

fn log2(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.log2", |x| x.log2())
}

fn log1p(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // log(1 + x), matching `np.log1p`.
    unary(op, ctx, "math.log1p", |x| x.ln_1p())
}

fn tanh(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.tanh", |x| x.tanh())
}

fn sin(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.sin", |x| x.sin())
}

fn cos(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.cos", |x| x.cos())
}

fn absf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // `MathOps.absf` applies `np.abs` directly to the stored data without the
    // float32 round-trip, so abs is exact regardless of dtype.
    unary(op, ctx, "math.absf", |x| x.abs())
}

fn absi(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Integer absolute value. Tile data is f32-backed in slice-1, but the values
    // are whole numbers; `MathOps.absi` is also a plain `np.abs`. For genuine
    // integer scalars we keep the integer variant exact.
    let v = one_operand(op, ctx, "math.absi")?;
    match v {
        Value::Scalar(Scalar::I32(i)) => Ok(Some(Value::Scalar(Scalar::I32(i.abs())))),
        Value::Scalar(Scalar::I64(i)) => Ok(Some(Value::Scalar(Scalar::I64(i.abs())))),
        Value::Index(i) => Ok(Some(Value::Index(i.abs()))),
        _ => unary(op, ctx, "math.absi", |x| x.abs()),
    }
}

fn ceil(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.ceil", |x| x.ceil())
}

fn floor(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    unary(op, ctx, "math.floor", |x| x.floor())
}

fn erf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // Abramowitz & Stegun 7.1.26 polynomial — see `MathOps._erf_f32`.
    unary(op, ctx, "math.erf", erf_f32)
}

// --- multi-operand handlers ----------------------------------------------

/// `math.powf %base, %exp` — element-wise `base ** exp`. Both operands must be
/// the same kind (tile/tile or scalar/scalar), mirroring `MathOps.powf` /
/// `powf_scalar` which read `base`'s type to decide the dispatch.
fn powf(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.len() != 2 {
        return Err(format!(
            "math.powf expects 2 operands, got {}",
            op.operands.len()
        ));
    }
    let base = ctx.get_value(&op.operands[0])?.clone();
    let exponent = ctx.get_value(&op.operands[1])?.clone();
    match (&base, &exponent) {
        (Value::Tile(b), Value::Tile(e)) => {
            if b.shape != e.shape {
                return Err(format!(
                    "math.powf: shape mismatch {:?} vs {:?}",
                    b.shape, e.shape
                ));
            }
            let data: Vec<f32> = b
                .as_f32()
                .iter()
                .zip(e.as_f32().iter())
                .map(|(&x, &y)| round_to(x.powf(y), b.dtype))
                .collect();
            Ok(Some(Value::Tile(Tile::compute(
                data,
                b.dtype,
                b.shape.clone(),
            ))))
        }
        (Value::Scalar(b), Value::Scalar(e)) => {
            let x = b.as_f32().ok_or("math.powf: non-float base scalar")?;
            let y = e.as_f32().ok_or("math.powf: non-float exponent scalar")?;
            Ok(Some(Value::Scalar(Scalar::F32(x.powf(y)))))
        }
        _ => Err("math.powf: base and exponent must both be tiles or both scalars".into()),
    }
}

/// `math.fma %a, %b, %c` — fused multiply-add `a * b + c`, element-wise.
/// Mirrors `MathOps.fma` / `fma_scalar`; dispatch keys on whether `a` is a tile.
fn fma(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.len() != 3 {
        return Err(format!(
            "math.fma expects 3 operands, got {}",
            op.operands.len()
        ));
    }
    let a = ctx.get_value(&op.operands[0])?.clone();
    let b = ctx.get_value(&op.operands[1])?.clone();
    let c = ctx.get_value(&op.operands[2])?.clone();
    match (&a, &b, &c) {
        (Value::Tile(ta), Value::Tile(tb), Value::Tile(tc)) => {
            if ta.shape != tb.shape || ta.shape != tc.shape {
                return Err(format!(
                    "math.fma: shape mismatch {:?} / {:?} / {:?}",
                    ta.shape, tb.shape, tc.shape
                ));
            }
            let ta_data = ta.as_f32();
            let tb_data = tb.as_f32();
            let tc_data = tc.as_f32();
            let data: Vec<f32> = (0..ta_data.len())
                .map(|i| round_to(ta_data[i] * tb_data[i] + tc_data[i], ta.dtype))
                .collect();
            Ok(Some(Value::Tile(Tile::compute(
                data,
                ta.dtype,
                ta.shape.clone(),
            ))))
        }
        (Value::Scalar(sa), Value::Scalar(sb), Value::Scalar(sc)) => {
            let x = sa.as_f32().ok_or("math.fma: non-float scalar")?;
            let y = sb.as_f32().ok_or("math.fma: non-float scalar")?;
            let z = sc.as_f32().ok_or("math.fma: non-float scalar")?;
            Ok(Some(Value::Scalar(Scalar::F32(x * y + z))))
        }
        _ => Err("math.fma: operands must be all tiles or all scalars".into()),
    }
}

// --- helpers --------------------------------------------------------------

fn one_operand<'s>(op: &Operation, ctx: &'s CoreContext, name: &str) -> Result<&'s Value, String> {
    if op.operands.len() != 1 {
        return Err(format!(
            "{name} expects 1 operand, got {}",
            op.operands.len()
        ));
    }
    ctx.get_value(&op.operands[0])
}

/// Apply a pure `f32 -> f32` kernel element-wise to a tile, or to a single
/// scalar — the Rust shape of Python's `_unary(op, ctx, tile_fn, scalar_fn)`.
///
/// For tiles, results are rounded back into the tile's dtype just like the
/// Python `.astype(tile.data.dtype)` round-trip, so f16 parity holds.
fn unary(
    op: &Operation,
    ctx: &mut CoreContext,
    name: &str,
    f: fn(f32) -> f32,
) -> Result<Option<Value>, String> {
    let v = one_operand(op, ctx, name)?;
    match v {
        Value::Tile(t) => {
            let dtype = t.dtype;
            let shape = t.shape.clone();
            let data: Vec<f32> = t.as_f32().iter().map(|&x| round_to(f(x), dtype)).collect();
            Ok(Some(Value::Tile(Tile::compute(data, dtype, shape))))
        }
        Value::Scalar(s) => {
            let x = s
                .as_f32()
                .ok_or_else(|| format!("{name}: non-float scalar operand"))?;
            // Scalars in the Python path stay f16 (`np.float16`) where they came
            // from f16; here scalars are f32-typed, so we keep f32 precision.
            Ok(Some(Value::Scalar(Scalar::F32(f(x)))))
        }
        other => Err(format!(
            "{name}: expected tile or scalar operand, got {other:?}"
        )),
    }
}

/// Round an f32 result into the tile's element type. For f16 we emulate NumPy's
/// `.astype(float16)` (round-to-nearest-even), then widen back to f32 for the
/// `Vec<f32>` storage. All other float-capable dtypes keep full f32 precision.
fn round_to(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => f16_round(x),
        _ => x,
    }
}

/// Round-trip an f32 through IEEE-754 binary16 and back, reproducing NumPy's
/// `np.float16(x)` rounding (round-to-nearest, ties-to-even). Implemented
/// inline to avoid pulling in the `half` crate at the slice-1 boundary.
fn f16_round(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN: preserve, forcing a quiet-NaN bit when the payload is set.
        let half = sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
        return f16_bits_to_f32(half as u16);
    }

    // Unbias the f32 exponent (127) and rebias to f16 (15).
    let unbiased = exp - 127 + 15;
    let half: u16 = if unbiased >= 0x1f {
        // Overflow to infinity.
        (sign | 0x7c00) as u16
    } else if unbiased <= 0 {
        // Subnormal or underflow to zero.
        if unbiased < -10 {
            sign as u16
        } else {
            // Restore the implicit leading 1, then shift into subnormal range.
            let m = mant | 0x0080_0000;
            let shift = (14 - unbiased) as u32;
            let mut sub = m >> shift;
            // Round to nearest even on the bits shifted out.
            let rem = m & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            if rem > halfway || (rem == halfway && (sub & 1) == 1) {
                sub += 1;
            }
            (sign | sub) as u16
        }
    } else {
        // Normal range: take top 10 mantissa bits, round to nearest even.
        let mut h = (sign | ((unbiased as u32) << 10) | (mant >> 13)) as u16;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
            h += 1; // a mantissa carry ripples naturally into the exponent field
        }
        h
    };
    f16_bits_to_f32(half)
}

/// Expand IEEE-754 binary16 bits to an f32 value.
fn f16_bits_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exp = ((half >> 10) & 0x1f) as u32;
    let mant = (half & 0x03ff) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign // signed zero
        } else {
            // Subnormal: normalize into the f32 normal range.
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            sign | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN.
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let f32_exp = exp + (127 - 15);
        sign | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Scalar erf kernel — Abramowitz & Stegun 7.1.26 (max error < 1.5e-7), a
/// faithful transcription of `MathOps._erf_f32`. Avoids a libm `erf` / scipy
/// dependency so results match the Python implementation in f32.
fn erf_f32(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let sign = if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0 // np.sign(0) == 0, matching the Python `np.sign(x)` factor
    };
    sign * (1.0 - poly * (-a * a).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::single_core_context;
    use crate::ir::{Operation, Scalar, Value};

    /// Run a single op through the real dispatch table, binding `inputs` first.
    fn run(op: &Operation, inputs: &[(&str, Value)]) -> Result<Value, String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        for (name, v) in inputs {
            ctx.set_value(name, v.clone());
        }
        let handler = env
            .dispatch
            .handler(&op.op_type)
            .expect("handler registered");
        handler(op, &mut ctx, &env).map(|o| o.expect("op produces a result"))
    }

    fn ok(op: &Operation, inputs: &[(&str, Value)]) -> Value {
        run(op, inputs).unwrap()
    }

    fn tile(data: Vec<f32>) -> Value {
        let n = data.len();
        Value::Tile(Tile::compute(data, DType::F32, vec![n]))
    }

    fn f16_tile(data: Vec<f32>) -> Value {
        let n = data.len();
        Value::Tile(Tile::compute(data, DType::F16, vec![n]))
    }

    fn as_tile(v: &Value) -> &Tile {
        match v {
            Value::Tile(t) => t,
            other => panic!("expected tile, got {other:?}"),
        }
    }

    fn f32_scalar(v: &Value) -> f32 {
        match v {
            Value::Scalar(s) => s.as_f32().unwrap(),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }

    // --- registration ----------------------------------------------------

    #[test]
    fn all_ops_register_with_expected_latency() {
        let d = Dispatch::new();
        for name in [
            "math.exp",
            "math.sqrt",
            "math.rsqrt",
            "math.log",
            "math.log2",
            "math.log1p",
            "math.tanh",
            "math.sin",
            "math.cos",
            "math.erf",
            "math.powf",
        ] {
            assert!(d.handler(name).is_some(), "{name} missing");
            assert_eq!(
                d.latency_category(name),
                LatencyCategory::ComputeTranscendental,
                "{name}"
            );
        }
        for name in [
            "math.absf",
            "math.absi",
            "math.ceil",
            "math.floor",
            "math.fma",
        ] {
            assert!(d.handler(name).is_some(), "{name} missing");
            assert_eq!(
                d.latency_category(name),
                LatencyCategory::ComputeFloat,
                "{name}"
            );
        }
    }

    // --- exp -------------------------------------------------------------

    #[test]
    fn exp_tile_elementwise() {
        let op = Operation::new(Some("%r"), "math.exp", &["%x"]);
        let r = ok(&op, &[("%x", tile(vec![0.0, 1.0, 2.0]))]);
        let t = as_tile(&r);
        close(t.as_f32()[0], 1.0);
        close(t.as_f32()[1], std::f32::consts::E);
        close(t.as_f32()[2], (2.0f32).exp());
    }

    #[test]
    fn exp_scalar_preserves_kind() {
        let op = Operation::new(Some("%r"), "math.exp", &["%x"]);
        let r = ok(&op, &[("%x", Value::Scalar(Scalar::F32(1.0)))]);
        close(f32_scalar(&r), std::f32::consts::E);
    }

    // --- sqrt / rsqrt ----------------------------------------------------

    #[test]
    fn sqrt_tile() {
        let op = Operation::new(Some("%r"), "math.sqrt", &["%x"]);
        let r = ok(&op, &[("%x", tile(vec![4.0, 9.0, 16.0]))]);
        let t = as_tile(&r);
        close(t.as_f32()[0], 2.0);
        close(t.as_f32()[1], 3.0);
        close(t.as_f32()[2], 4.0);
    }

    #[test]
    fn rsqrt_tile_is_reciprocal_sqrt() {
        let op = Operation::new(Some("%r"), "math.rsqrt", &["%x"]);
        let r = ok(&op, &[("%x", tile(vec![4.0, 16.0]))]);
        let t = as_tile(&r);
        close(t.as_f32()[0], 0.5);
        close(t.as_f32()[1], 0.25);
    }

    #[test]
    fn sqrt_scalar() {
        let op = Operation::new(Some("%r"), "math.sqrt", &["%x"]);
        let r = ok(&op, &[("%x", Value::Scalar(Scalar::F32(4.0)))]);
        close(f32_scalar(&r), 2.0);
    }

    // --- logs ------------------------------------------------------------

    #[test]
    fn log_family() {
        let e = std::f32::consts::E;
        let r = ok(
            &Operation::new(Some("%r"), "math.log", &["%x"]),
            &[("%x", tile(vec![e]))],
        );
        close(as_tile(&r).as_f32()[0], 1.0);

        let r = ok(
            &Operation::new(Some("%r"), "math.log2", &["%x"]),
            &[("%x", tile(vec![8.0]))],
        );
        close(as_tile(&r).as_f32()[0], 3.0);

        let r = ok(
            &Operation::new(Some("%r"), "math.log1p", &["%x"]),
            &[("%x", tile(vec![0.0]))],
        );
        close(as_tile(&r).as_f32()[0], 0.0);
    }

    // --- trig / tanh -----------------------------------------------------

    #[test]
    fn trig_and_tanh() {
        let pi = std::f32::consts::PI;
        let r = ok(
            &Operation::new(Some("%r"), "math.sin", &["%x"]),
            &[("%x", tile(vec![0.0, pi / 2.0]))],
        );
        let t = as_tile(&r);
        close(t.as_f32()[0], 0.0);
        close(t.as_f32()[1], 1.0);

        let r = ok(
            &Operation::new(Some("%r"), "math.cos", &["%x"]),
            &[("%x", tile(vec![0.0, pi]))],
        );
        let t = as_tile(&r);
        close(t.as_f32()[0], 1.0);
        close(t.as_f32()[1], -1.0);

        let r = ok(
            &Operation::new(Some("%r"), "math.tanh", &["%x"]),
            &[("%x", tile(vec![0.0]))],
        );
        close(as_tile(&r).as_f32()[0], 0.0);
    }

    // --- abs / rounding --------------------------------------------------

    #[test]
    fn absf_tile() {
        let op = Operation::new(Some("%r"), "math.absf", &["%x"]);
        let r = ok(&op, &[("%x", tile(vec![-1.5, 2.0, -0.0]))]);
        let t = as_tile(&r);
        close(t.as_f32()[0], 1.5);
        close(t.as_f32()[1], 2.0);
        close(t.as_f32()[2], 0.0);
    }

    #[test]
    fn absi_scalar_keeps_integer_kind() {
        let op = Operation::new(Some("%r"), "math.absi", &["%x"]);
        let r = ok(&op, &[("%x", Value::Scalar(Scalar::I64(-7)))]);
        assert!(matches!(r, Value::Scalar(Scalar::I64(7))));

        let r = ok(&op, &[("%x", Value::Index(-3))]);
        assert!(matches!(r, Value::Index(3)));
    }

    #[test]
    fn floor_and_ceil() {
        let r = ok(
            &Operation::new(Some("%r"), "math.floor", &["%x"]),
            &[("%x", tile(vec![1.7, -1.2]))],
        );
        let t = as_tile(&r);
        close(t.as_f32()[0], 1.0);
        close(t.as_f32()[1], -2.0);

        let r = ok(
            &Operation::new(Some("%r"), "math.ceil", &["%x"]),
            &[("%x", tile(vec![1.2, -1.7]))],
        );
        let t = as_tile(&r);
        close(t.as_f32()[0], 2.0);
        close(t.as_f32()[1], -1.0);
    }

    // --- erf -------------------------------------------------------------

    #[test]
    fn erf_matches_known_values() {
        let op = Operation::new(Some("%r"), "math.erf", &["%x"]);
        let r = ok(&op, &[("%x", tile(vec![0.0, 1.0, -1.0]))]);
        let t = as_tile(&r);
        close(t.as_f32()[0], 0.0);
        // erf(1) ≈ 0.8427007
        close(t.as_f32()[1], 0.8427007);
        // erf is odd: erf(-1) = -erf(1)
        close(t.as_f32()[2], -0.8427007);
    }

    #[test]
    fn erf_scalar() {
        let op = Operation::new(Some("%r"), "math.erf", &["%x"]);
        let r = ok(&op, &[("%x", Value::Scalar(Scalar::F32(1.0)))]);
        close(f32_scalar(&r), 0.8427007);
    }

    // --- powf ------------------------------------------------------------

    #[test]
    fn powf_tile() {
        let op = Operation::new(Some("%r"), "math.powf", &["%b", "%e"]);
        let r = ok(
            &op,
            &[
                ("%b", tile(vec![2.0, 3.0, 4.0])),
                ("%e", tile(vec![2.0, 2.0, 0.5])),
            ],
        );
        let t = as_tile(&r);
        close(t.as_f32()[0], 4.0);
        close(t.as_f32()[1], 9.0);
        close(t.as_f32()[2], 2.0);
    }

    #[test]
    fn powf_scalar() {
        let op = Operation::new(Some("%r"), "math.powf", &["%b", "%e"]);
        let r = ok(
            &op,
            &[
                ("%b", Value::Scalar(Scalar::F32(2.0))),
                ("%e", Value::Scalar(Scalar::F32(10.0))),
            ],
        );
        close(f32_scalar(&r), 1024.0);
    }

    // --- fma -------------------------------------------------------------

    #[test]
    fn fma_tile() {
        let op = Operation::new(Some("%r"), "math.fma", &["%a", "%b", "%c"]);
        let r = ok(
            &op,
            &[
                ("%a", tile(vec![2.0, 3.0])),
                ("%b", tile(vec![4.0, 5.0])),
                ("%c", tile(vec![1.0, 1.0])),
            ],
        );
        let t = as_tile(&r);
        close(t.as_f32()[0], 9.0); // 2*4 + 1
        close(t.as_f32()[1], 16.0); // 3*5 + 1
    }

    #[test]
    fn fma_scalar() {
        let op = Operation::new(Some("%r"), "math.fma", &["%a", "%b", "%c"]);
        let r = ok(
            &op,
            &[
                ("%a", Value::Scalar(Scalar::F32(2.0))),
                ("%b", Value::Scalar(Scalar::F32(3.0))),
                ("%c", Value::Scalar(Scalar::F32(4.0))),
            ],
        );
        close(f32_scalar(&r), 10.0);
    }

    // --- f16 rounding boundary ------------------------------------------

    #[test]
    fn f16_round_trip_is_exact_for_representable() {
        // These values are exactly representable in f16.
        for v in [1.0f32, 0.5, 2.0, -3.0, 0.0, 0.25, 100.0] {
            assert_eq!(f16_round(v), v, "{v}");
        }
    }

    #[test]
    fn f16_round_is_idempotent() {
        // Rounding an already-f16 value again must be a no-op.
        for v in [1.0f32, 1.5, std::f32::consts::E, 0.1, -7.25] {
            let once = f16_round(v);
            assert_eq!(f16_round(once), once, "{v}");
        }
    }

    #[test]
    fn f16_tile_results_are_rounded_and_typed() {
        // A transcendental result on an f16 tile lands on an f16 grid point and
        // keeps the f16 dtype, mirroring `.astype(tile.data.dtype)`.
        let op = Operation::new(Some("%r"), "math.exp", &["%x"]);
        let r = ok(&op, &[("%x", f16_tile(vec![1.0]))]);
        let t = as_tile(&r);
        assert_eq!(t.dtype, DType::F16);
        assert_eq!(f16_round(t.as_f32()[0]), t.as_f32()[0]);
        // Still numerically close to e, within f16 resolution (~1e-2 near 2.7).
        assert!(
            (t.as_f32()[0] - std::f32::consts::E).abs() < 1e-2,
            "{}",
            t.as_f32()[0]
        );
    }

    // --- error paths -----------------------------------------------------

    #[test]
    fn powf_mixed_kinds_errors() {
        let op = Operation::new(Some("%r"), "math.powf", &["%b", "%e"]);
        let err = run(
            &op,
            &[
                ("%b", tile(vec![2.0])),
                ("%e", Value::Scalar(Scalar::F32(2.0))),
            ],
        );
        assert!(err.is_err());
    }

    #[test]
    fn unary_wrong_arity_errors() {
        let op = Operation::new(Some("%r"), "math.exp", &["%x", "%y"]);
        let err = run(&op, &[("%x", tile(vec![1.0])), ("%y", tile(vec![1.0]))]);
        assert!(err.is_err());
    }

    #[test]
    fn powf_shape_mismatch_errors() {
        let op = Operation::new(Some("%r"), "math.powf", &["%b", "%e"]);
        let err = run(
            &op,
            &[("%b", tile(vec![2.0, 3.0])), ("%e", tile(vec![2.0]))],
        );
        assert!(err.is_err());
    }
}
