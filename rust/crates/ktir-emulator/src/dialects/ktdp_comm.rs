// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Inter-tile collective ops — port of the `ktdp.inter_tile_produce` /
//! `ktdp.yield_partial` / `ktdp.yield_reduced` handlers from
//! `ktir_cpu/dialects/ktdp_ops.py`.
//!
//! These three are ordinary (synchronous) handlers: the **producer** materialises
//! this core's partial by running its `^bb0(%gid): yield_partial` region and
//! stashes it on a per-core [`TileFuture`]; the two `yield_*` ops park their value
//! for the enclosing region driver to recover (the same pattern as `linalg.yield`).
//!
//! The cross-core data movement is owned by `ktdp.inter_tile_reduce`, which is a
//! **comm op** driven by the top-level scheduler (the ring all-reduce in
//! `comm_sched.rs`), not a handler here — comm only happens at the top level.

use super::{Dispatch, LatencyCategory};
use crate::affine::AffineSet;
use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::interpreter::execute_region;
use crate::ir::{Attr, Operation, TileFuture, Value};
use crate::tile::Tile;

/// Scope key the `yield_partial` / `yield_reduced` terminators park their value
/// under, so the enclosing region driver (`run_produce_region` here, and the
/// reduce combiner in `comm_sched.rs`) can recover it. Distinct from the linalg
/// yield key — these regions never nest inside a linalg combiner.
pub const COMM_YIELD_KEY: &str = "__ktdp_comm_yield__";

pub fn register(d: &mut Dispatch) {
    d.register(
        "ktdp.inter_tile_produce",
        LatencyCategory::Zero,
        inter_tile_produce,
    );
    d.register("ktdp.yield_partial", LatencyCategory::Zero, yield_partial);
    d.register("ktdp.yield_reduced", LatencyCategory::Zero, yield_reduced);
}

/// `ktdp.yield_partial %v` / `ktdp.yield_reduced %v` — park `%v` under
/// [`COMM_YIELD_KEY`] in the current scope. Mirrors `ktdp__yield_partial` /
/// `ktdp__yield_reduced`.
fn yield_partial(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    park_yield(op, ctx)
}

fn yield_reduced(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    park_yield(op, ctx)
}

fn park_yield(op: &Operation, ctx: &mut CoreContext) -> Result<Option<Value>, String> {
    if let Some(name) = op.operands.first() {
        let v = ctx.get_value(name)?.clone();
        ctx.set_value(COMM_YIELD_KEY, v);
    }
    Ok(None)
}

/// Resolve the unique group index `g` whose membership set contains `tile_id`.
/// `producer_set` is the family `(d)[g]`; `groups_set` is the 1-D key domain.
/// Enumerates `groups_set` over `[0, num_cores)` and keeps keys `g` for which
/// `producer_set.contains([tile_id], [g])`. Mirrors `_find_tile_group` /
/// `enumerate_membership_keys`: exactly one match is required (the disjointness
/// invariant).
pub fn find_tile_group(
    tile_id: i64,
    producer_set: &AffineSet,
    groups_set: &AffineSet,
    num_cores: usize,
) -> Result<i64, String> {
    let matches: Vec<i64> = groups_set
        .enumerate(&[num_cores], &[])
        .into_iter()
        .map(|pt| pt[0])
        .filter(|&g| producer_set.contains(&[tile_id], &[g]))
        .collect();
    match matches.as_slice() {
        [g] => Ok(*g),
        [] => Err(format!(
            "tile {tile_id} is not contained in any producer group"
        )),
        many => Err(format!(
            "tile {tile_id} matched multiple groups {many:?} — violates the disjointness invariant"
        )),
    }
}

/// `%fut = ktdp.inter_tile_produce ... { ^bb0(%gid): yield_partial %p }`
///
/// Resolves this core's group index, runs the producer region with `%gid` bound,
/// captures the `yield_partial` tile as the local partial, and returns a
/// [`TileFuture`] carrying the partial plus the producer/groups sets and group
/// index for the consume-side ring plan. Mirrors `ktdp__inter_tile_produce`.
fn inter_tile_produce(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let producer_set = match op.attributes.get("producer_tiles_per_group") {
        Some(Attr::AffineSet(s)) => s.clone(),
        _ => return Err("ktdp.inter_tile_produce: missing producer_tiles_per_group".into()),
    };
    let groups_set = match op.attributes.get("groups") {
        Some(Attr::AffineSet(s)) => s.clone(),
        _ => return Err("ktdp.inter_tile_produce: missing groups".into()),
    };

    let tile_id = ctx.core_id as i64;
    let gid = find_tile_group(tile_id, &producer_set, &groups_set, env.grid.num_cores)?;

    // Run the producer region with %gid bound, recovering the yielded partial.
    let region: &[Operation] = op.regions.first().map(|r| r.as_slice()).unwrap_or(&[]);
    let gid_name = region
        .iter()
        .find(|o| o.op_type == "region.bb0_args")
        .and_then(|o| match o.attributes.get("names") {
            Some(Attr::StrList(names)) => names.first().cloned(),
            _ => None,
        });
    let body: Vec<Operation> = region
        .iter()
        .filter(|o| o.op_type != "region.bb0_args")
        .cloned()
        .collect();

    ctx.push_scope();
    let local_partial = (|| {
        if let Some(name) = &gid_name {
            ctx.set_value(name, Value::Index(gid));
        }
        execute_region(&body, ctx, env)?;
        if ctx.has_value(COMM_YIELD_KEY) {
            match ctx.get_value(COMM_YIELD_KEY)?.clone() {
                Value::Tile(t) => Ok(Some(t)),
                other => Err(format!(
                    "ktdp.inter_tile_produce: yield_partial must yield a Tile, got {other:?}"
                )),
            }
        } else {
            Ok(None)
        }
    })();
    ctx.pop_scope();
    let local_partial: Option<Tile> = local_partial?;

    Ok(Some(Value::TileFuture(Box::new(TileFuture {
        local_partial,
        producer_set,
        groups_set,
        group_idx: gid,
    }))))
}
