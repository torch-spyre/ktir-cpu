// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `ktdp` dialect handlers — partial port of `ktir_emulator/dialects/ktdp_ops.py`.
//!
//! This slice ports the two construct ops that build the memory-view types,
//! single-allocation path only:
//!   * `construct_memory_view`  -> `MemRef`   (logical view; does NOT allocate)
//!   * `construct_access_tile`  -> `AccessTile` over a `TileRef`
//!
//! The distributed path (`construct_distributed_memory_view`,
//! `distributed_tile_access`) and `load`/`store` follow once `memory.rs` grows
//! a real `HBMSimulator`. Symbolic access-tile sets are rejected here, matching
//! the Python handler's `NotImplementedError`.

use super::{Dispatch, LatencyCategory};
use crate::affine::AffineMap;
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::memref::{AccessTile, DistributedMemRef, MemRef, MemorySpace, ParentRef, TileRef};
use crate::ops_memory::distributed_tile_access;

pub fn register(d: &mut Dispatch) {
    d.register(
        "ktdp.construct_memory_view",
        LatencyCategory::Zero,
        construct_memory_view,
    );
    d.register(
        "ktdp.construct_access_tile",
        LatencyCategory::Zero,
        construct_access_tile,
    );
}

/// `%v = ktdp.construct_memory_view %ptr {shape, strides, memory_space, dtype, ...}`
///
/// Builds a logical `MemRef`. Mirrors `tile_view`. Slice limitation: shape /
/// strides must be static (the Python parser also stores dynamic dims as SSA
/// names resolved at runtime — that resolution lands with grid/scope support).
fn construct_memory_view(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("construct_memory_view: missing pointer operand".into());
    }
    let base_ptr = scalar_i64(ctx.get_value(&op.operands[0])?, "construct_memory_view ptr")?;

    // Static `shape` (IntList) or dynamic `sizes_dyn` (StrList of `%ssa`/literal
    // tokens) resolved from scope now. Mirrors the runtime SSA-size resolution
    // in `ktdp__construct_memory_view`.
    let shape: Vec<usize> = match op.attributes.get("shape") {
        Some(Attr::IntList(v)) => v.iter().map(|&n| n as usize).collect(),
        _ => match op.attributes.get("sizes_dyn") {
            Some(Attr::StrList(tokens)) => tokens
                .iter()
                .map(|t| {
                    if t.starts_with('%') {
                        scalar_i64(ctx.get_value(t)?, "construct_memory_view size")
                            .map(|n| n as usize)
                    } else {
                        t.parse::<usize>()
                            .map_err(|_| format!("construct_memory_view: bad size token {t:?}"))
                    }
                })
                .collect::<Result<_, String>>()?,
            _ => return Err("construct_memory_view: missing required attribute 'shape'".into()),
        },
    };
    let strides = int_list(op, "strides")?.clone();

    let space_str = str_attr(op, "memory_space")?;
    let core_id = match op.attributes.get("lx_core_id") {
        Some(Attr::Int(n)) => Some(*n as u32),
        _ => None,
    };
    let space = MemorySpace::parse(space_str, core_id)?;

    let dtype = dtype_attr(op, "dtype")?;

    let coordinate_set = match op.attributes.get("coordinate_set") {
        Some(Attr::AffineSet(s)) => Some(s.clone()),
        _ => None,
    };

    Ok(Some(Value::MemRef(MemRef {
        base_ptr,
        shape,
        strides,
        space,
        dtype,
        coordinate_set,
    })))
}

/// `%t = ktdp.construct_access_tile %view, %i, %j {shape, base_map, ...}`
///
/// Single-allocation path: evaluate `base_map` at the indices to get base
/// coords, fold them through the parent strides into a byte offset, and wrap
/// the resulting `TileRef` in an `AccessTile`. Mirrors `tile_access`.
fn construct_access_tile(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("construct_access_tile: missing parent operand".into());
    }
    // Parent is a single-allocation MemRef or a distributed view; clone the
    // relevant one so we can drop the borrow before reading the index operands.
    enum Parent {
        Single(MemRef),
        Dist(DistributedMemRef),
    }
    let parent = match ctx.get_value(&op.operands[0])? {
        Value::MemRef(m) => Parent::Single(m.clone()),
        Value::DistMemRef(d) => Parent::Dist(d.clone()),
        other => {
            return Err(format!(
                "construct_access_tile: parent is {other:?}, expected MemRef"
            ));
        }
    };

    let indices: Vec<i64> = op.operands[1..]
        .iter()
        .map(|name| {
            ctx.get_value(name)
                .and_then(|v| scalar_i64(v, "construct_access_tile index"))
        })
        .collect::<Result<_, _>>()?;

    let access_shape = int_list(op, "shape")?
        .iter()
        .map(|&n| n as usize)
        .collect::<Vec<_>>();

    // base_map is always present (synthesized as identity upstream if absent).
    let base_map = match op.attributes.get("base_map") {
        Some(Attr::AffineMap(m)) => m.clone(),
        _ => AffineMap::identity(indices.len()),
    };

    let coordinate_set = match op.attributes.get("coordinate_set") {
        Some(Attr::AffineSet(s)) => Some(s.clone()),
        _ => None,
    };

    // Single allocation -> direct tile_access; distributed view -> resolve
    // partition routing now via distributed_tile_access (mirrors ktdp__construct_access_tile).
    let parent_ref = match parent {
        Parent::Single(m) => {
            ParentRef::Tile(tile_access(m, &indices, access_shape.clone(), &base_map))
        }
        Parent::Dist(d) => {
            let dist = distributed_tile_access(
                &d,
                &access_shape,
                &base_map,
                &indices,
                coordinate_set.as_ref(),
            )?;
            ParentRef::Dist(dist)
        }
    };

    Ok(Some(Value::AccessTile(AccessTile {
        parent_ref,
        shape: access_shape,
        base_map,
        coordinate_set,
        coordinate_order: None, // access_tile_order parsing lands with the parser slice
    })))
}

/// Port of `MemoryOps.tile_access`: indices -> base coords (via base_map) ->
/// byte offset (via parent strides) -> byte-addressed `TileRef`.
fn tile_access(
    parent: MemRef,
    indices: &[i64],
    access_shape: Vec<usize>,
    base_map: &AffineMap,
) -> TileRef {
    let base_coords = base_map.eval(indices, &[]);
    let bpe = parent.dtype.bytes_per_elem() as i64;
    let offset_elems: i64 = base_coords
        .iter()
        .zip(&parent.strides)
        .map(|(coord, stride)| coord * stride)
        .sum();
    let byte_pos = parent.byte_address() + offset_elems * bpe;

    // Take parent's fields, then MOVE it into the box — no second clone of the
    // MemRef (and its affine `coordinate_set`), which `construct_access_tile`
    // already paid once. Halves the per-access affine clone/drop churn.
    let strides = parent.strides.clone();
    let dtype = parent.dtype;
    TileRef {
        base_ptr: byte_pos,
        shape: access_shape,
        strides,
        dtype,
        memref: Box::new(parent),
        coordinate_set: None,
        partition_origin: None,
    }
}

// --- attribute helpers ---------------------------------------------------

fn int_list<'a>(op: &'a Operation, key: &str) -> Result<&'a Vec<i64>, String> {
    match op.attributes.get(key) {
        Some(Attr::IntList(v)) => Ok(v),
        Some(other) => Err(format!(
            "{}: attr '{key}' is {other:?}, expected IntList",
            op.op_type
        )),
        None => Err(format!(
            "{}: missing required attribute '{key}'",
            op.op_type
        )),
    }
}

fn str_attr<'a>(op: &'a Operation, key: &str) -> Result<&'a str, String> {
    match op.attributes.get(key) {
        Some(Attr::Str(s)) => Ok(s),
        _ => Err(format!(
            "{}: missing/invalid string attribute '{key}'",
            op.op_type
        )),
    }
}

fn dtype_attr(op: &Operation, key: &str) -> Result<DType, String> {
    match op.attributes.get(key) {
        Some(Attr::Dtype(d)) => Ok(*d),
        Some(Attr::Str(s)) => DType::parse(s),
        _ => Err(format!(
            "{}: missing/invalid dtype attribute '{key}'",
            op.op_type
        )),
    }
}

fn scalar_i64(v: &Value, ctx: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(Scalar::I32(i)) => Ok(*i as i64),
        Value::Scalar(Scalar::I64(i)) => Ok(*i),
        other => Err(format!("{ctx}: expected index/int, got {other:?}")),
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

    fn build_view() -> Operation {
        // %v = construct_memory_view %p {shape=[64,32], strides=[32,1], HBM, f16}
        Operation::new(Some("%v"), "ktdp.construct_memory_view", &["%p"])
            .with_attr("shape", Attr::IntList(vec![64, 32]))
            .with_attr("strides", Attr::IntList(vec![32, 1]))
            .with_attr("memory_space", Attr::Str("HBM".into()))
            .with_attr("dtype", Attr::Str("f16".into()))
    }

    #[test]
    fn construct_view_builds_memref() {
        let mut ctx = single_core_context();
        ctx.set_value("%p", Value::Index(4)); // element index 4 (base_ptr is an element index)
        run(&[build_view()], &mut ctx).unwrap();
        match ctx.get_value("%v").unwrap() {
            Value::MemRef(m) => {
                assert_eq!(m.shape, vec![64, 32]);
                // base_ptr=4 element index at f16 (2 bytes) -> byte 8.
                assert_eq!(m.byte_address(), 4 * 2);
                assert_eq!(m.dtype, DType::F16);
            }
            other => panic!("expected MemRef, got {other:?}"),
        }
    }

    #[test]
    fn access_tile_offset_via_base_map() {
        let mut ctx = single_core_context();
        ctx.set_value("%p", Value::Index(0)); // base at byte 0 for a clean offset check
        ctx.set_value("%i", Value::Index(2));
        ctx.set_value("%j", Value::Index(3));
        // identity base_map over (i, j); offset = (2*32 + 3*1) elems * 2 bytes
        let at = Operation::new(
            Some("%t"),
            "ktdp.construct_access_tile",
            &["%v", "%i", "%j"],
        )
        .with_attr("shape", Attr::IntList(vec![1, 1]))
        .with_attr("base_map", Attr::AffineMap(AffineMap::identity(2)));
        run(&[build_view(), at], &mut ctx).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::AccessTile(a) => match &a.parent_ref {
                ParentRef::Tile(tr) => assert_eq!(tr.base_ptr, (2 * 32 + 3) * 2),
                _ => panic!("expected single-allocation TileRef parent"),
            },
            other => panic!("expected AccessTile, got {other:?}"),
        }
    }

    #[test]
    fn distributed_parent_is_flagged_unported() {
        let mut ctx = single_core_context();
        // assert the single-allocation path rejects a non-memref parent.
        ctx.set_value("%v", Value::Index(7));
        let at = Operation::new(Some("%t"), "ktdp.construct_access_tile", &["%v"])
            .with_attr("shape", Attr::IntList(vec![1]))
            .with_attr("base_map", Attr::AffineMap(AffineMap::identity(0)));
        let err = run(&[at], &mut ctx).unwrap_err();
        assert!(err.contains("expected MemRef"));
    }
}
