// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `scf` dialect handlers — port of `ktir_emulator/dialects/scf_ops.py` plus the
//! `ControlOps` loop/conditional helpers in `ktir_emulator/ops/control_ops.py`.
//!
//! Covers `scf.for` (induction var + iter_args + yield, with per-iteration
//! `push_scope`/`pop_scope` and iter_arg rebinding in the *parent* scope),
//! `scf.if` (then/else regions), and `scf.yield`.
//!
//! REGION/YIELD SEAM. The Python `execute_region` returns whatever the body's
//! last op produced; `scf.yield` returns a `_YieldResult` sentinel the loop
//! driver unwraps. The Rust contract's [`interpreter::execute_region`] returns
//! `Result<(), String>` and discards op results, so it cannot carry a yield out
//! of the body. We therefore run region bodies through a thin local executor
//! that drives [`interpreter::execute_op`] op-by-op and captures the value the
//! terminating `scf.yield` produces. Semantics are identical: the handler still
//! owns `push_scope`/`pop_scope`, and comm ops cannot appear in regions (so the
//! body never suspends), matching the spec.
//!
//! `scf.yield` is modeled as returning a `Value::Tuple(values)` (the
//! `_YieldResult` analogue). It has no SSA result name, so the value is not
//! bound into scope — it is observed only by the enclosing for/if driver.

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::interpreter::execute_op;
use crate::ir::{Attr, Operation, Scalar, Value};

pub fn register(d: &mut Dispatch) {
    d.register("scf.for", LatencyCategory::Zero, scf_for);
    d.register("scf.if", LatencyCategory::Zero, scf_if);
    d.register("scf.yield", LatencyCategory::Zero, scf_yield);
    // The synthetic `region.bb0_args` op (parsed from a `^bb0(...)` block label)
    // is a no-op at execution time — the enclosing op handler (linalg.generic /
    // linalg.reduce / tensor.generate) binds the block-arg names to its values.
    // Mirrors Python `region__bb0_args`. A registered no-op keeps it out of the
    // dispatch-coverage "no handler" set when a region body runs op-by-op.
    d.register("region.bb0_args", LatencyCategory::Zero, region_bb0_args);
}

/// No-op handler for the synthetic `region.bb0_args` op. See [`register`].
fn region_bb0_args(
    _op: &Operation,
    _ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    Ok(None)
}

// ---------------------------------------------------------------------------
// scf.yield
// ---------------------------------------------------------------------------

/// `scf.yield %a, %b, ...` — gather the operand values and hand them back to
/// the enclosing loop/conditional driver. Mirrors `ControlOps.yield_op`: the
/// returned `Value::Tuple` is the `_YieldResult` sentinel analogue.
fn scf_yield(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let values: Vec<Value> = op
        .operands
        .iter()
        .map(|name| ctx.get_value(name).cloned())
        .collect::<Result<_, _>>()?;
    Ok(Some(Value::Tuple(values)))
}

// ---------------------------------------------------------------------------
// scf.if
// ---------------------------------------------------------------------------

/// `scf.if %cond { then } else { else }` — execute the selected branch in its
/// own scope. Mirrors `ControlOps.if_op`.
///
/// The branch body gets its own scope; body-local LX is freed on `pop_scope`.
/// If the branch yields Tile values, their LX is freed by `pop_scope` too — the
/// driver in `execute_op` re-tracks the bound result afterward.
fn scf_if(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("scf.if: missing condition operand".into());
    }
    let condition = as_bool(ctx.get_value(&op.operands[0])?, "scf.if")?;

    let region: &[Operation] = if condition {
        op.regions.first().map(Vec::as_slice).unwrap_or(&[])
    } else {
        op.regions.get(1).map(Vec::as_slice).unwrap_or(&[])
    };

    if region.is_empty() {
        return Ok(None);
    }

    // Branch body gets its own scope; body-local LX is freed on pop.
    ctx.push_scope();
    let result = run_region(region, ctx, env);
    ctx.pop_scope();
    let yielded = result?;

    // Mirror `unwrap_yield`: a single yielded value passes through bare; a
    // multi-value yield stays a tuple; no yield -> None.
    Ok(unwrap_yield(yielded))
}

// ---------------------------------------------------------------------------
// scf.for
// ---------------------------------------------------------------------------

/// `%r = scf.for %i = %lb to %ub step %step iter_args(%a = %init, ...) { body }`
///
/// Counted loop with optional loop-carried state. Mirrors `ControlOps.for_op`
/// and the `scf__for` handler glue: iter_args are bound in the *parent* scope
/// (they persist across iterations); each iteration body runs in a fresh scope
/// whose body-local LX is freed on `pop_scope`. Yielded values are fed back as
/// the next iteration's iter_arg bindings, with LX untracked/retracked across
/// the rebinding.
///
/// Returns the final iter_arg value (single) or a `Value::Tuple` (multiple);
/// `None` when there are no iter_args.
fn scf_for(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.len() < 3 {
        return Err(format!(
            "scf.for expects at least 3 operands (lb, ub, step), got {}",
            op.operands.len()
        ));
    }
    let lb = as_i64(ctx.get_value(&op.operands[0])?, "scf.for lb")?;
    let ub = as_i64(ctx.get_value(&op.operands[1])?, "scf.for ub")?;
    let step = as_i64(ctx.get_value(&op.operands[2])?, "scf.for step")?;

    let iter_var = match op.attributes.get("iter_var") {
        Some(Attr::Str(s)) => s.clone(),
        _ => "%i".to_string(),
    };

    let body_region: &[Operation] = op.regions.first().map(Vec::as_slice).unwrap_or(&[]);

    let iter_arg_names: Vec<String> = match op.attributes.get("iter_args") {
        Some(Attr::StrList(v)) => v.clone(),
        _ => Vec::new(),
    };
    let iter_init_operands = &op.operands[3..];
    let iter_init_values: Vec<Value> = iter_init_operands
        .iter()
        .map(|name| ctx.get_value(name).cloned())
        .collect::<Result<_, _>>()?;

    let result = for_op(
        ctx,
        lb,
        ub,
        step,
        &iter_var,
        body_region,
        env,
        &iter_arg_names,
        iter_init_values,
    )?;

    // for_op returns a Vec of final iter_arg values; unwrap when there is
    // exactly one (the common case for a single result var). Mirrors `scf__for`.
    match result {
        None => Ok(None),
        Some(mut vals) => {
            if vals.len() == 1 {
                Ok(Some(vals.pop().unwrap()))
            } else if vals.len() == iter_arg_names.len() {
                Ok(Some(Value::Tuple(vals)))
            } else {
                Err(format!(
                    "scf.for: expected {} results, got {}",
                    iter_arg_names.len(),
                    vals.len()
                ))
            }
        }
    }
}

/// Port of `ControlOps.for_op`. Returns the list of final iter_arg values, or
/// `None` when there are no iter_args.
#[allow(clippy::too_many_arguments)]
fn for_op(
    ctx: &mut CoreContext,
    lower_bound: i64,
    upper_bound: i64,
    step: i64,
    iter_var_name: &str,
    body_region: &[Operation],
    env: &ExecutionEnv,
    iter_arg_names: &[String],
    iter_init_values: Vec<Value>,
) -> Result<Option<Vec<Value>>, String> {
    // Bind initial iter_arg values in the *parent* scope. These persist across
    // iterations; body-local values do not.
    //
    // The iter_arg is an ALIAS of the init value, which is typically already
    // charged LX under its definition name (e.g. `%acc_zero` loaded before the
    // loop). `track_lx_tile` dedups by the tile's backing allocation, so binding
    // the alias bumps the shared refcount and charges 0 extra bytes — removing the
    // double-count the old per-name `track_lx` introduced (#118).
    let mut current_values = iter_init_values;
    for (name, val) in iter_arg_names.iter().zip(current_values.iter()) {
        ctx.set_value(name, val.clone());
        if let Value::Tile(t) = val {
            ctx.track_lx_tile(name, t)?;
        }
    }

    // `max(step, 1)` mirrors the Python guard against non-positive steps.
    let step = step.max(1);
    let mut i = lower_bound;
    while i < upper_bound {
        // New scope for this iteration's body-local values; pop frees their LX.
        ctx.push_scope();

        // Bind the iteration variable (a plain index, like Python's `int`).
        ctx.set_value(iter_var_name, Value::Index(i));

        // Execute the body, capturing the terminating yield (if any).
        let result = run_region(body_region, ctx, env);

        // Save yielded values before pop_scope() discards them.
        let yielded_values: Option<Vec<Value>> = match &result {
            Ok(Some(Value::Tuple(vals))) if !iter_arg_names.is_empty() => Some(vals.clone()),
            _ => None,
        };

        // Pop body scope — frees LX for all body-local Tiles, including any
        // Tiles that were yielded (they lived in this scope).
        ctx.pop_scope();
        result?; // surface any body error after the scope is cleaned up.

        // Re-bind yielded values as iter_args in the parent scope. `track_lx_tile`
        // releases this id's prior allocation reference (freeing the old carry
        // tile when its refcount hits 0) and charges the new one — alias-aware, so
        // an unchanged carry (yield == iter_arg) doesn't double-charge (#118).
        if let Some(yielded) = yielded_values {
            for (name, val) in iter_arg_names.iter().zip(yielded.iter()) {
                ctx.set_value(name, val.clone());
                if let Value::Tile(t) = val {
                    ctx.track_lx_tile(name, t)?;
                }
            }
            current_values = yielded;
        }

        i += step;
    }

    if current_values.is_empty() {
        Ok(None)
    } else {
        Ok(Some(current_values))
    }
}

// ---------------------------------------------------------------------------
// region execution + yield plumbing
// ---------------------------------------------------------------------------

/// Drive a region body op-by-op, returning the value produced by its
/// terminating `scf.yield` (a `Value::Tuple`), or `None` if it does not yield.
///
/// This is the contract's `execute_region` with one addition: it threads the
/// terminator's value back out. Comm ops cannot appear in regions, so this
/// never suspends — matching `interpreter::execute_region`'s guarantee. The
/// caller owns `push_scope`/`pop_scope`.
fn run_region(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    // GATED DESCEND of the forced map-offload into this region body (#A).
    //
    // The top-level map-window fusion planner (`comm_sched::StepFn::step` via
    // `metal::map_fusion_plan`) treats `scf.for` as a window boundary and never
    // descends into loop bodies, so the per-row elementwise maps of
    // softmax/softmax_wide/layernorm (which wrap ALL their map ops inside an
    // `scf.for` body) stay on the interpreter. When `KTIR_FORCE_GPU_MAP` is set
    // (the conformance harness's ForceAllMetal mode), descend: plan THIS body's
    // map windows and offload each window's trigger to the Metal map kernel.
    //
    // STRICTLY GATED so the unforced (golden/production) path is byte-identical:
    //   * `env.tracker.is_none()` — never on the latency-tracking path (mirrors
    //     `comm_sched`'s `gpu_map_offload` gate exactly), and
    //   * `metal::force_gpu_map()` — only when the force flag is set, and
    //   * `KTIR_NO_GPU_MAP` unset.
    // A per-row map is core-local + elementwise, so offloading it per-iteration
    // is semantically identical to running its ops on the interpreter; the result
    // tile is written through `Tile::compute(.., out_dtype, ..)` so per-step f16
    // rounding matches the interpreter. When the gate is off, this whole block is
    // skipped and the body runs op-by-op exactly as before.
    #[cfg(metal)]
    if env.tracker.is_none()
        && crate::metal::force_gpu_map()
        && std::env::var_os("KTIR_NO_GPU_MAP").is_none()
        && let Some(plan) = cached_region_map_plan(ops)
    {
        return run_region_with_map_offload(ops, ctx, env, &plan);
    }
    let mut last_yield = None;
    for op in ops {
        let produced = execute_op(op, ctx, env)?;
        if op.op_type == "scf.yield" {
            last_yield = produced;
        }
    }
    Ok(last_yield)
}

/// Run a region body op-by-op, applying a precomputed map-window offload plan
/// (`trigger -> kernel`, `skip` set) — the descended analogue of the top-level
/// map-window handling in `comm_sched::StepFn::step`. At a window's TRIGGER op
/// the whole window runs as ONE fused Metal kernel; a non-trigger window op is
/// subsumed by that kernel and is NOT executed. Per-iteration LX for body-local
/// window outputs is reclaimed by the caller's `pop_scope`, so (unlike the
/// top-level loop) no `dies_at` bookkeeping is needed here. A trigger failure is
/// FATAL (the window's other ops were skipped — there is no interpreter result
/// to fall back to), surfacing a real kernel/planner bug rather than silently
/// diverging.
#[cfg(metal)]
fn run_region_with_map_offload(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
    plan: &RegionMapPlan,
) -> Result<Option<Value>, String> {
    let (triggers, skip) = (&plan.plan.0, &plan.plan.1);
    let mut last_yield = None;
    for (i, op) in ops.iter().enumerate() {
        if let Some(mrk) = triggers.get(&i) {
            // `run_map_region_gpu` consumes the window's dead (single-use,
            // current-generation) live-ins before charging the output's LX —
            // mirroring `execute_op`'s #134 consume-on-last-use order — so a wide
            // per-row window doesn't hold inputs + output simultaneously.
            crate::metal::run_map_region_gpu(mrk, ctx)?;
        } else if skip.contains(&i) {
            // Subsumed by the trigger's fused kernel — not executed.
        } else {
            let produced = execute_op(op, ctx, env)?;
            if op.op_type == "scf.yield" {
                last_yield = produced;
            }
        }
        // PER-OP LIVENESS RECLAIM (the body analogue of the top-level loop's
        // `dies_at` reclaim). A `tensor.splat`/`arith.constant` that feeds ONLY a
        // SKIPPED window op is materialized by `execute_op` here but FOLDED away by
        // the fused kernel (which reads the splat's scalar operand, not the splat
        // tile), so its only "use" never runs and `consume_if_last_use` (single-use
        // only) can't free it — without this it leaks 512 KB/row and softmax_wide's
        // wide rows overflow LX. Free every body-local name whose LAST body use is
        // op `i` (matches the interpreter's per-iteration peak; pop_scope would
        // otherwise free it only at iteration end).
        if let Some(dead) = plan.dies_at.get(&i) {
            for name in dead {
                ctx.forget(name);
            }
        }
    }
    Ok(last_yield)
}

/// A region body's offload plan: the fused-map windows ([`crate::metal::map_fusion_plan`])
/// plus a body-local liveness map (`op index -> names whose LAST use in the body
/// is that op`) so the offload loop can reclaim folded-away plumbing tiles
/// per-iteration (see [`run_region_with_map_offload`]).
#[cfg(metal)]
struct RegionMapPlan {
    plan: crate::metal::MapRegionPlan,
    dies_at: std::collections::HashMap<usize, Vec<String>>,
}

/// Body-local last-use map: `op index -> SSA names whose last operand use in this
/// body is that op` (counting names embedded in string attrs, like the top-level
/// `comm_sched::compute_dies_at`). Only names DEFINED in this body are tracked
/// (an outer-scope value read here is freed by the outer driver, not us). Used to
/// reclaim a fused window's folded-away plumbing producers per iteration.
///
/// A name whose last use is a SKIPPED window op actually dies at that window's
/// TRIGGER (the fused kernel reads it there, after the skipped op's index), so its
/// death is REMAPPED to the trigger — freeing it at the skipped index would race
/// the not-yet-run kernel that still needs it as a live-in.
#[cfg(metal)]
fn body_dies_at(
    ops: &[Operation],
    plan: &crate::metal::MapRegionPlan,
) -> std::collections::HashMap<usize, Vec<String>> {
    use std::collections::{HashMap, HashSet};
    let (triggers, skip) = (&plan.0, &plan.1);
    // For a skipped index, the window TRIGGER that subsumes it = the smallest
    // trigger index >= that skipped index (windows are contiguous runs ending at
    // their trigger). Used to defer a folded op's operand deaths to the kernel.
    let trigger_for = |j: usize| -> usize {
        triggers
            .keys()
            .copied()
            .filter(|&t| t >= j)
            .min()
            .unwrap_or(j)
    };
    // Names defined by a body op (only these are body-local; reclaiming an
    // outer-scope name here would free it before the outer driver is done).
    let mut defined: HashSet<String> = HashSet::new();
    for op in ops {
        if let Some(r) = &op.result {
            defined.insert(r.clone());
        }
        if let Some(crate::ir::Attr::StrList(names)) = op.attributes.get("result_names") {
            for n in names {
                defined.insert(n.clone());
            }
        }
    }
    // Highest body index at which each name is used (operands + SSA string attrs).
    let mut last_use: HashMap<String, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        for operand in &op.operands {
            if operand.starts_with('%') {
                last_use.insert(operand.clone(), i);
            }
        }
        for attr in op.attributes.values() {
            match attr {
                crate::ir::Attr::Str(s) if s.starts_with('%') => {
                    last_use.insert(s.clone(), i);
                }
                crate::ir::Attr::StrList(xs) => {
                    for x in xs {
                        if x.starts_with('%') {
                            last_use.insert(x.clone(), i);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut dies: HashMap<usize, Vec<String>> = HashMap::new();
    for (name, idx) in last_use {
        if !defined.contains(&name) {
            continue;
        }
        // If the last use is a skipped (folded) op, the fused kernel reads it at
        // the trigger — defer the death there. A trigger op is itself the window's
        // last op (already the right index), so only non-trigger skips remap.
        let death = if skip.contains(&idx) && !triggers.contains_key(&idx) {
            trigger_for(idx)
        } else {
            idx
        };
        dies.entry(death).or_default().push(name);
    }
    dies
}

/// Memoized region-body offload plan, keyed by the body's STRUCTURAL fingerprint
/// ([`crate::comm_sched::plan_key`]). A content key (not the body's pointer) avoids
/// the ABA hazard where a freed body slice's address is reused by a different
/// program's body — which would apply the wrong program's plan (the softmax_fwd
/// plan to softmax_wide, etc.). The IR is immutable for a run, so a body executed
/// thousands of times (e.g. a 4096-row softmax `scf.for`) plans its windows ONCE.
/// Returns `None` when the body has no offloadable window (the caller then takes
/// the plain op-by-op path with zero per-iteration overhead).
#[cfg(metal)]
fn cached_region_map_plan(ops: &[Operation]) -> Option<std::rc::Rc<RegionMapPlan>> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<u64, Option<std::rc::Rc<RegionMapPlan>>>> =
            RefCell::new(HashMap::new());
    }
    let key = crate::comm_sched::plan_key(ops);
    CACHE.with(|c| {
        c.borrow_mut()
            .entry(key)
            .or_insert_with(|| {
                // A body containing a `linalg.matmul` is a GEMM K-loop: it is
                // reconstructed as ONE whole-M GEMM by the matmul-loop offload
                // (`comm_sched` / `run_matmul_loop_gpu`), NOT map-descended. Its
                // accumulate `arith.addf` is part of that reconstruction, so
                // offloading it here as a standalone map window would race the
                // GEMM offload (and the loop-carried accumulator binding differs).
                // Leave such bodies entirely to the existing GEMM path.
                if ops.iter().any(|o| o.op_type == "linalg.matmul") {
                    return None;
                }
                let plan = crate::metal::map_fusion_plan(ops);
                if plan.0.is_empty() {
                    None
                } else {
                    Some(std::rc::Rc::new(RegionMapPlan {
                        dies_at: body_dies_at(ops, &plan),
                        plan,
                    }))
                }
            })
            .clone()
    })
}

/// Mirror `_helpers.unwrap_yield`: a single yielded value passes through bare,
/// a multi-value yield stays a tuple, and a non-yield (None / empty) is `None`.
fn unwrap_yield(result: Option<Value>) -> Option<Value> {
    match result {
        Some(Value::Tuple(mut vals)) => match vals.len() {
            0 => None,
            1 => Some(vals.pop().unwrap()),
            _ => Some(Value::Tuple(vals)),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// operand coercion helpers
// ---------------------------------------------------------------------------

fn as_i64(v: &Value, name: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(s) => s.as_i64().ok_or_else(|| format!("{name}: non-int scalar")),
        other => Err(format!("{name}: expected index/int, got {other:?}")),
    }
}

fn as_bool(v: &Value, name: &str) -> Result<bool, String> {
    match v {
        Value::Scalar(Scalar::Bool(b)) => Ok(*b),
        Value::Scalar(Scalar::I32(i)) => Ok(*i != 0),
        Value::Scalar(Scalar::I64(i)) => Ok(*i != 0),
        Value::Index(i) => Ok(*i != 0),
        other => Err(format!("{name}: expected boolean condition, got {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::dtypes::DType;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::tile::Tile;

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    /// A `scf.for` op with a body region (and optional iter_args).
    #[allow(clippy::too_many_arguments)]
    fn for_op_ir(
        result: Option<&str>,
        lb: &str,
        ub: &str,
        step: &str,
        iter_var: &str,
        iter_inits: &[&str],
        iter_args: &[&str],
        body: Vec<Operation>,
    ) -> Operation {
        let mut operands = vec![lb, ub, step];
        operands.extend_from_slice(iter_inits);
        let mut op = Operation::new(result, "scf.for", &operands)
            .with_attr("iter_var", Attr::Str(iter_var.into()));
        if !iter_args.is_empty() {
            op = op.with_attr(
                "iter_args",
                Attr::StrList(iter_args.iter().map(|s| s.to_string()).collect()),
            );
        }
        op.regions = vec![body];
        op
    }

    // --- scf.for: counting ------------------------------------------------

    #[test]
    fn for_counts_iterations_via_iter_arg() {
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(5));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        // body: %s = addi %acc %one ; yield %s   (counts iterations)
        let one =
            Operation::new(Some("%one"), "arith.constant", &[]).with_attr("value", Attr::Int(1));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%one"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let body = vec![one, add, yld];
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            body,
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 5), // 5 iterations
            other => panic!("expected I64(5), got {other:?}"),
        }
    }

    #[test]
    fn for_step_2_visits_three_times() {
        // 0..6 step 2 -> 3 iterations.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(6));
        ctx.set_value("%step", Value::Index(2));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        let one =
            Operation::new(Some("%one"), "arith.constant", &[]).with_attr("value", Attr::Int(1));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%one"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![one, add, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 3),
            other => panic!("expected I64(3), got {other:?}"),
        }
    }

    #[test]
    fn for_induction_var_is_visible_in_body() {
        // running sum of the induction variable: acc += i over 0..4.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(4));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        // %s = addi %acc %i ; yield %s   (i is an Index, addi accepts it)
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%i"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![add, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            // 0 + (0+1+2+3) = 6
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 6),
            other => panic!("expected I64(6), got {other:?}"),
        }
    }

    // --- scf.for: iter_args -----------------------------------------------

    #[test]
    fn for_iter_args_running_sum() {
        // Mirrors test_for_op_iter_args_running_sum: acc starts 0, += i, 0..4.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(4));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%i"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![add, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 6),
            other => panic!("expected I64(6), got {other:?}"),
        }
    }

    #[test]
    fn for_no_iters_returns_none_and_leaves_no_result() {
        // ub == lb: zero iterations, no iter_args, no result binding.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(3));
        ctx.set_value("%ub", Value::Index(3));
        ctx.set_value("%step", Value::Index(1));
        let body = vec![Operation::new(None, "scf.yield", &[])];
        let f = for_op_ir(None, "%lb", "%ub", "%step", "%i", &[], &[], body);
        run(&[f], &mut ctx).unwrap();
        assert!(ctx.get_value("%i").is_err()); // induction var scope is gone
    }

    #[test]
    fn for_multi_iter_args_yields_tuple() {
        // Two scalar accumulators advanced independently.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(3));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%a0", Value::Scalar(Scalar::I64(0)));
        ctx.set_value("%b0", Value::Scalar(Scalar::I64(10)));
        let one =
            Operation::new(Some("%one"), "arith.constant", &[]).with_attr("value", Attr::Int(1));
        let na = Operation::new(Some("%na"), "arith.addi", &["%a", "%one"]);
        let nb = Operation::new(Some("%nb"), "arith.addi", &["%b", "%one"]);
        let yld = Operation::new(None, "scf.yield", &["%na", "%nb"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%a0", "%b0"],
            &["%a", "%b"],
            vec![one, na, nb, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Tuple(vals) => {
                assert_eq!(vals.len(), 2);
                assert!(matches!(vals[0], Value::Scalar(Scalar::I64(3)))); // 0+3
                assert!(matches!(vals[1], Value::Scalar(Scalar::I64(13)))); // 10+3
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn for_tile_iter_arg_lx_is_conserved() {
        // A Tile iter_arg: LX usage after the loop equals exactly one tile's
        // worth — the per-iteration body tile and old iter_arg tiles are freed.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(3));
        ctx.set_value("%step", Value::Index(1));
        // init tile: 4 x f32 = 16 bytes. Charge via the alias-aware `track_lx_tile`
        // (the #118 path: the iter_arg binding is an ALIAS of this allocation, so it
        // must NOT charge a second 16 bytes).
        let init = Tile::compute(vec![0.0; 4], DType::F32, vec![4]);
        ctx.set_value("%init", Value::Tile(init.clone()));
        ctx.track_lx_tile("%init", &init).unwrap();
        let one = Operation::new(Some("%one"), "arith.constant", &[])
            .with_attr("value", Attr::Float(1.0));
        // each iteration yields a fresh tile via addf %acc %acc.
        let add = Operation::new(Some("%s"), "arith.addf", &["%acc", "%acc"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![one, add, yld],
        );
        let used_before = ctx.lx.borrow().used;
        assert_eq!(used_before, 16); // only %init is tracked
        run(&[f], &mut ctx).unwrap();
        // Conservation: LX does not grow with the iteration count. After the loop
        // the live distinct tile allocations are %init (still bound) and the final
        // carry tile (bound to %acc, the loop result %r, AND the last %s — all
        // aliases of one allocation, charged ONCE under alias-dedup). The init
        // iter_arg alias never double-charged (#118), and each iteration's body
        // tile is freed on pop_scope. Two distinct allocations × 16 bytes = 32.
        assert_eq!(ctx.lx.borrow().used, 32);
    }

    // --- scf.if -----------------------------------------------------------

    #[test]
    fn if_true_runs_then_branch() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(true)));
        let c =
            Operation::new(Some("%t"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "scf.yield", &["%t"]);
        let e =
            Operation::new(Some("%f"), "arith.constant", &[]).with_attr("value", Attr::Float(9.0));
        let eyld = Operation::new(None, "scf.yield", &["%f"]);
        let mut iff = Operation::new(Some("%r"), "scf.if", &["%cond"]);
        iff.regions = vec![vec![c, yld], vec![e, eyld]];
        run(&[iff], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 7.0),
            other => panic!("expected F32(7.0), got {other:?}"),
        }
    }

    #[test]
    fn if_false_runs_else_branch() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(false)));
        let c =
            Operation::new(Some("%t"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "scf.yield", &["%t"]);
        let e =
            Operation::new(Some("%f"), "arith.constant", &[]).with_attr("value", Attr::Float(9.0));
        let eyld = Operation::new(None, "scf.yield", &["%f"]);
        let mut iff = Operation::new(Some("%r"), "scf.if", &["%cond"]);
        iff.regions = vec![vec![c, yld], vec![e, eyld]];
        run(&[iff], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 9.0),
            other => panic!("expected F32(9.0), got {other:?}"),
        }
    }

    #[test]
    fn if_empty_branch_returns_none() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(false)));
        // then has a body, else is empty -> condition false selects empty -> None.
        let c =
            Operation::new(Some("%t"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "scf.yield", &["%t"]);
        // no result name: op produces None, nothing bound.
        let mut iff = Operation::new(None, "scf.if", &["%cond"]);
        iff.regions = vec![vec![c, yld]]; // only a then-region
        run(&[iff], &mut ctx).unwrap();
        // no panic, and no stray binding leaked from the (unrun) then-branch.
        assert!(ctx.get_value("%t").is_err());
    }

    #[test]
    fn if_branch_local_lx_is_freed() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(true)));
        ctx.set_value(
            "%x",
            Value::Tile(Tile::compute(vec![1.0, 2.0], DType::F32, vec![2])),
        );
        // then: %y = addf %x %x ; yield nothing (no result) -> body-local tile freed.
        let add = Operation::new(Some("%y"), "arith.addf", &["%x", "%x"]);
        let yld = Operation::new(None, "scf.yield", &[]);
        let mut iff = Operation::new(None, "scf.if", &["%cond"]);
        iff.regions = vec![vec![add, yld]];
        let before = ctx.lx.borrow().used;
        run(&[iff], &mut ctx).unwrap();
        // %y (body-local) was freed on pop_scope; LX usage unchanged.
        assert_eq!(ctx.lx.borrow().used, before);
        assert!(ctx.get_value("%y").is_err());
    }

    #[test]
    fn yield_gathers_multiple_operands() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::Scalar(Scalar::I64(1)));
        ctx.set_value("%b", Value::Scalar(Scalar::I64(2)));
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let op = Operation::new(None, "scf.yield", &["%a", "%b"]);
        let out = scf_yield(&op, &mut ctx, &env).unwrap();
        match out {
            Some(Value::Tuple(vals)) => {
                assert_eq!(vals.len(), 2);
                assert!(matches!(vals[0], Value::Scalar(Scalar::I64(1))));
                assert!(matches!(vals[1], Value::Scalar(Scalar::I64(2))));
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn nested_for_inside_if() {
        // if(true) { %r = for ... acc += i ; yield acc } yield %r
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(true)));
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(4));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%i"]);
        let fyld = Operation::new(None, "scf.yield", &["%s"]);
        let inner_for = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![add, fyld],
        );
        let oyld = Operation::new(None, "scf.yield", &["%r"]);
        let mut iff = Operation::new(Some("%out"), "scf.if", &["%cond"]);
        iff.regions = vec![vec![inner_for, oyld]];
        run(&[iff], &mut ctx).unwrap();
        match ctx.get_value("%out").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 6),
            other => panic!("expected I64(6), got {other:?}"),
        }
    }
}
