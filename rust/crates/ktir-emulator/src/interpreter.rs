// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Execution orchestrator — port of `ktir_emulator/interpreter.py` + the per-op
//! driver from `grid.py`.
//!
//! This slice locks the execution contract every dialect handler builds
//! against: the handler signature `(op, &mut CoreContext, &ExecutionEnv)`, the
//! single-op driver `execute_op` (binds results, tracks Tile LX usage), and the
//! synchronous `execute_region` callback handlers use for scf bodies. The
//! multi-core comm scheduler (top-level only — see `comm.rs`) and HBM
//! input/output marshalling in `execute_function` are implement-phase fills
//! against these locked seams.

use std::collections::HashMap;
use std::rc::Rc;

use crate::codec;
use crate::context::CoreContext;
use crate::dialects::Dispatch;
use crate::dtypes::DType;
use crate::env::{ExecutionEnv, GridExecutor};
use crate::ir::{IRModule, Operation, Scalar, Value};
use crate::memory::{STICK_BYTES, SpyreMemoryHierarchy};

// ---- Host wall-clock profiler (opt-in via KTIR_PROFILE) --------------------
// Buckets each op handler's HOST wall-clock by op class, so a real forward shows
// where time actually goes (matmul vs elementwise vs load/store vs ...) — this is
// separate from the Spyre *latency model* (record_op) below. Zero cost when off.
fn profile_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KTIR_PROFILE").is_some())
}

fn profile_class(op_type: &str) -> &'static str {
    match op_type {
        "linalg.matmul" | "linalg.batch_matmul" => "matmul",
        t if t.starts_with("linalg.") => "linalg-other",
        t if t.starts_with("arith.") => "arith",
        t if t.starts_with("math.") => "math",
        "ktdp.load" | "ktdp.store" => "load/store",
        t if t.starts_with("ktdp.") => "ktdp-other",
        t if t.starts_with("scf.") => "scf",
        t if t.starts_with("tensor.") => "tensor",
        _ => "other",
    }
}

thread_local! {
    static PROFILE: std::cell::RefCell<std::collections::BTreeMap<&'static str, (std::time::Duration, u64)>> =
        const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
}

/// Format the accumulated per-op-class wall-clock and CLEAR it. Empty if profiling
/// is off or nothing ran. Call after a forward (e.g. from a test) to read the split.
pub fn profile_report() -> String {
    PROFILE.with(|p| {
        let mut rows: Vec<_> = p.borrow().iter().map(|(k, (d, n))| (*k, *d, *n)).collect();
        if rows.is_empty() {
            return String::new();
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));
        let total: f64 = rows.iter().map(|(_, d, _)| d.as_secs_f64()).sum();
        let mut s = format!("  [op-class profile — {total:.2}s in handlers]\n");
        for (k, d, n) in rows {
            let secs = d.as_secs_f64();
            s += &format!(
                "    {k:>12}  {secs:7.2}s  {:5.1}%  ({n} ops)\n",
                100.0 * secs / total
            );
        }
        p.borrow_mut().clear();
        s
    })
}

/// Execute one operation: dispatch, then bind its result (tracking LX for
/// Tiles). Mirrors `_execute_op`. Comm ops (which suspend) are driven by the
/// top-level scheduler, not here — see `comm.rs`.
pub fn execute_op(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let handler = env
        .dispatch
        .handler(&op.op_type)
        .ok_or_else(|| format!("no handler registered for op '{}'", op.op_type))?;
    let produced = if profile_on() {
        let cls = profile_class(&op.op_type);
        let t0 = std::time::Instant::now();
        let r = handler(op, ctx, env);
        let dt = t0.elapsed();
        PROFILE.with(|p| {
            let mut m = p.borrow_mut();
            let e = m.entry(cls).or_insert((std::time::Duration::ZERO, 0));
            e.0 += dt;
            e.1 += 1;
        });
        r?
    } else {
        handler(op, ctx, env)?
    };

    // Latency: record this op's cost before binding its result (operands are
    // still bound in scope; the result is not yet). Mirrors `_execute_op`.
    if let Some(tracker) = env.tracker {
        let lat_t0 = if profile_on() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let operands: Vec<Option<Value>> = op
            .operands
            .iter()
            .map(|n| ctx.get_value(n).ok().cloned())
            .collect();
        let category = env.dispatch.latency_category(&op.op_type);
        tracker
            .borrow_mut()
            .record_op(ctx.core_id, &op.op_type, category, &produced, &operands);
        if let Some(t0) = lat_t0 {
            let dt = t0.elapsed();
            PROFILE.with(|p| {
                let mut m = p.borrow_mut();
                let e = m
                    .entry("__latency__")
                    .or_insert((std::time::Duration::ZERO, 0));
                e.0 += dt;
                e.1 += 1;
            });
        }
    }

    // Consume-on-last-use: free any operand tile whose LAST use is THIS op (and
    // which lives in the active scope) BEFORE charging this op's result, so the
    // result can reuse the freed LX at no net increase — the peak-LX reduction
    // from #134. Handlers (and latency pre-resolution above) have already read
    // the operands, so dropping them now is safe. A single pass over distinct
    // operand names (an operand used twice by this op has use_count >= 2, so it
    // isn't consumed). Mirrors Python's consume in `get_value`, relocated to the
    // post-handler point where it is LX-peak-equivalent.
    if ctx.consume_last_use_enabled() {
        for operand in &op.operands {
            if operand.starts_with('%') {
                ctx.consume_if_last_use(operand);
            }
        }
    }

    // Multi-result op (`%a, %b = ...`): bind each name to one tuple element.
    // Mirrors Python `if isinstance(op.result, list) and isinstance(result, tuple)`.
    if let Some(crate::ir::Attr::StrList(names)) = op.attributes.get("result_names")
        && names.len() > 1
    {
        let Some(Value::Tuple(vals)) = &produced else {
            return Err(format!(
                "op '{}' has {} result names but did not produce a tuple",
                op.op_type,
                names.len()
            ));
        };
        if vals.len() != names.len() {
            return Err(format!(
                "op '{}': {} result names but produced {} values",
                op.op_type,
                names.len(),
                vals.len()
            ));
        }
        for (name, val) in names.iter().zip(vals) {
            if let Value::Tile(t) = val {
                ctx.track_lx_tile(name, t)?;
            }
            ctx.set_value(name, val.clone());
        }
        return Ok(produced);
    }

    if let Some(name) = &op.result {
        match produced {
            Some(val) => {
                // Tiles occupy LX; bookkeeping values (TileRef, index, ...) don't.
                if let Value::Tile(t) = &val {
                    ctx.track_lx_tile(name, t)?;
                }
                ctx.set_value(name, val.clone());
                return Ok(Some(val));
            }
            None => {
                return Err(format!(
                    "op '{}' has result {name} but produced no value",
                    op.op_type
                ));
            }
        }
    }
    Ok(produced)
}

/// Run a straight-line list of operations against a context. The function-body
/// driver and the body of `execute_region`.
pub fn execute_ops(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<(), String> {
    let mut i = 0;
    while i < ops.len() {
        // Peephole: fold `matmul` + a following elementwise op into one fused
        // NAX kernel (no host elementwise pass, one GPU dispatch). Only fires
        // when there's no latency tracker — the analytical model must still see
        // each op individually — so it speeds up pure execution / validation
        // without changing the latency report.
        #[cfg(metal)]
        if env.tracker.is_none()
            && let Some(advance) = try_fuse_matmul_epilogue(ops, i, ctx)?
        {
            i += advance;
            continue;
        }
        execute_op(&ops[i], ctx, env)?;
        i += 1;
    }
    Ok(())
}

/// Try to fuse `ops[i]` (a 2-operand `linalg.matmul` producing `%c`) with the
/// immediately following elementwise op that consumes `%c` (`linalg.add/mul/
/// sub/max/min`), running both as one fused NAX kernel and binding the
/// elementwise result. Returns `Some(2)` on a fuse, `None` to fall through to
/// normal op-by-op execution. Conservative: only fuses when `%c` is used by
/// nothing but that consumer, the shapes line up, and the size gate picks NAX.
#[cfg(metal)]
fn try_fuse_matmul_epilogue(
    ops: &[Operation],
    i: usize,
    ctx: &mut CoreContext,
) -> Result<Option<usize>, String> {
    use crate::metal::Epilogue;

    let mm = &ops[i];
    if mm.op_type != "linalg.matmul" || mm.operands.len() != 2 {
        return Ok(None);
    }
    let Some(cname) = mm.result.as_deref() else {
        return Ok(None);
    };
    let Some(ep) = ops.get(i + 1) else {
        return Ok(None);
    };
    let Some(dname) = ep.result.as_deref() else {
        return Ok(None);
    };

    // The consumer must be a fusable binary elementwise op with `%c` as one
    // operand; `%e` is the other. For non-commutative ops the kernel computes
    // `c BINOP e`, so `%c` must be the FIRST operand.
    let Some(epi) = Epilogue::from_binary_op(&ep.op_type) else {
        return Ok(None);
    };
    if ep.operands.len() != 2 {
        return Ok(None);
    }
    let commutative = matches!(
        epi,
        Epilogue::ADD | Epilogue::MUL | Epilogue::MAX | Epilogue::MIN
    );
    let ename = if ep.operands[0] == cname {
        ep.operands[1].as_str()
    } else if ep.operands[1] == cname && commutative {
        ep.operands[0].as_str()
    } else {
        return Ok(None);
    };

    // `%c` must be dead after the consumer (else we'd still have to materialize
    // it). Reject if it reappears later or is used twice by the consumer itself.
    if ename == cname {
        return Ok(None);
    }
    let reused = ops[i + 2..]
        .iter()
        .any(|o| o.operands.iter().any(|x| x == cname))
        || ops[i + 1]
            .operands
            .iter()
            .filter(|x| x.as_str() == cname)
            .count()
            > 1;
    if reused {
        return Ok(None);
    }

    // Pull A, B, E tiles; check 2-D, compatible inner dim, and E matching C.
    let (a, b, e) = (
        as_tile(ctx, &mm.operands[0])?,
        as_tile(ctx, &mm.operands[1])?,
        as_tile(ctx, ename)?,
    );
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return Ok(None);
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    if e.shape != [m, n] {
        return Ok(None);
    }
    let (a_data, b_data, e_data, dtype) = (
        a.as_f32().into_owned(),
        b.as_f32().into_owned(),
        e.as_f32().into_owned(),
        a.dtype,
    );

    // Fuse only if the gate picks NAX and the kernel runs; else fall through.
    let Some(out) = crate::metal::metal_gemm_fused(m, k, n, &a_data, &b_data, &e_data, epi) else {
        return Ok(None);
    };
    let tile = crate::tile::Tile::compute(out, dtype, vec![m, n]);
    ctx.track_lx(dname, tile.size_bytes() as i64)?;
    ctx.set_value(dname, crate::ir::Value::Tile(tile));
    Ok(Some(2))
}

/// Borrow an SSA value as a `Tile`, or `Err` if it isn't one.
#[cfg(metal)]
fn as_tile<'a>(ctx: &'a CoreContext, name: &str) -> Result<&'a crate::tile::Tile, String> {
    match ctx.get_value(name)? {
        crate::ir::Value::Tile(t) => Ok(t),
        _ => Err(format!("fuse: {name} is not a tile")),
    }
}

/// Synchronous nested-region executor — the callback handlers use for scf.for
/// bodies / scf.if branches. Mirrors `execute_region`. Per the spec, comm ops
/// cannot appear in nested regions, so this never suspends. The caller (the op
/// handler) owns `push_scope`/`pop_scope`.
pub fn execute_region(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<(), String> {
    execute_ops(ops, ctx, env)
}

/// Build a single-core context for `grid_pos`/`core_id` over a fresh memory
/// hierarchy — the common setup for executing a `grid = [1]` function or a unit
/// test. Returns `(context, dispatch)` ready for `execute_ops`.
pub fn single_core_context() -> CoreContext {
    let mem = SpyreMemoryHierarchy::new(1);
    CoreContext::new(
        0,
        (0, 0, 0),
        Rc::clone(&mem.hbm),
        mem.get_lx(0),
        mem.lx_scratchpads.clone(),
    )
}

/// A function argument: a tensor (marshalled into HBM) or a scalar (bound
/// directly). Mirrors the `np.ndarray` vs scalar split in `execute_function`.
#[derive(Clone, Debug)]
pub enum Arg {
    /// f32 host data, narrowed to `dtype` on the way into HBM. The dtype-agnostic
    /// oracle path — convenient, but for an all-f16 model it pays an f32→f16
    /// narrow per input (and f16→f32 widen per output) and 2× host memory.
    Tensor {
        data: Vec<f32>,
        shape: Vec<usize>,
        dtype: DType,
    },
    /// Pre-encoded typed bytes (already in `dtype` layout, e.g. f16), copied
    /// straight into HBM with no conversion — mirrors Spyre's typed host→AIU DMA.
    /// Use this to avoid the f32 round-trip for typed (f16/…) host buffers.
    TensorBytes {
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: DType,
    },
    /// bfloat16 host bytes, narrowed to the model dtype (f16) on the way into HBM.
    /// bf16 is NOT a Spyre/KTIR HBM dtype (the hardware is f16) — this is a
    /// convenience for ingesting stock bf16 checkpoints (Llama/SmolLM2) without
    /// the caller first doing the bf16→f16 narrow. The narrowing is paid ONCE at
    /// ingest (bf16→f32 is exact; f32→f16 rounds to nearest-even); f16 weights
    /// should use [`Arg::TensorBytes`] (zero conversion).
    TensorBf16 {
        data: Vec<u8>,
        shape: Vec<usize>,
    },
    Scalar(Scalar),
}

/// A tensor read back from HBM after execution.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    /// Values widened to f32 — the dtype-agnostic oracle view.
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// The raw `dtype`-encoded HBM bytes (e.g. f16), undecoded. Lets a typed
    /// host runner thread an f16 output straight into the next node's
    /// [`Arg::TensorBytes`] input with no f16→f32→f16 round-trip. `data` is
    /// `decode(raw)`; the two are equivalent.
    pub raw: Vec<u8>,
}

/// Execute a function with tensor + scalar arguments and return every tensor
/// argument read back from HBM. Port of `KTIRInterpreter.execute_function`
/// (latency tracking is an optional add-on, see `latency.rs`). Cores are driven
/// by the comm scheduler (`comm_sched`), so cross-core collectives work; cores
/// with no comm op just run their body to completion against shared HBM.
pub fn execute_function(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
) -> Result<HashMap<String, Output>, String> {
    execute_function_filtered(module, func_name, args, None)
}

/// Like [`execute_function`] but reads back ONLY the named tensor args, skipping
/// the (read-only) inputs. A whole-program-fused function carries hundreds of
/// weight pointers as args; reading them all back decodes ~hundreds of MB of
/// unchanged f16 for nothing. Pass just the real outputs (e.g. the result ptr)
/// to cut that waste. Names may include or omit the leading `%`.
pub fn execute_function_outputs(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
    outputs: &[&str],
) -> Result<HashMap<String, Output>, String> {
    let wanted: std::collections::HashSet<String> = outputs
        .iter()
        .map(|s| s.trim_start_matches('%').to_string())
        .collect();
    execute_function_filtered(module, func_name, args, Some(&wanted))
}

fn execute_function_filtered(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
    wanted: Option<&std::collections::HashSet<String>>,
) -> Result<HashMap<String, Output>, String> {
    // Fast path (NAX tensor engine): a multi-core, comm-free, straight-line SPMD
    // grid whose cores share a matmul weight runs lock-step on the GPU, combining
    // the grid's per-core matmul panels into one GEMM instead of one Accelerate
    // call per tile. `execute_function_gpu` returns Err whenever it doesn't apply
    // (single core, comm ops, region-bearing ops, or no NAX device), so this is a
    // pure accelerator with a transparent fall-through to the comm scheduler.
    #[cfg(metal)]
    if let Ok(out) = execute_function_gpu(module, func_name, args) {
        return Ok(out);
    }

    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;

    let mem = SpyreMemoryHierarchy::new(num_cores.max(1));
    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::shared();

    let probe = std::env::var_os("KTIR_TIME_PHASES").is_some();
    let t0 = std::time::Instant::now();
    let (input_ptrs, tensor_meta) = marshal_inputs(&mem, args);
    let t_marshal = t0.elapsed();

    // Drive all cores via the comm scheduler (cores with no comm op simply run
    // to completion; ring/collective ops suspend and resume through it).
    let t1 = std::time::Instant::now();
    crate::comm_sched::execute_with_communication(
        &grid,
        &mem,
        &func.operations,
        &input_ptrs,
        dispatch,
        None,
        None,
    )?;
    let t_run = t1.elapsed();

    let t2 = std::time::Instant::now();
    let out = read_back(&mem, tensor_meta, wanted);
    if probe {
        eprintln!(
            "  [phases] marshal {:.0}ms  run {:.0}ms  readback {:.0}ms",
            t_marshal.as_secs_f64() * 1e3,
            t_run.as_secs_f64() * 1e3,
            t2.elapsed().as_secs_f64() * 1e3,
        );
    }
    out
}

/// Run `func` against an EXTERNALLY-OWNED, already-populated memory hierarchy —
/// the resident-executor seam. The caller has placed every pointer arg's data in
/// `mem`'s HBM ONCE (weights stay across passes; the per-pass input activation is
/// rewritten in place) and supplies `input_ptrs` (arg name -> its HBM stick /
/// scalar) so the function's args resolve to the resident sticks WITHOUT a fresh
/// marshal. `read` lists `(out_name, stick, n_elems, shape, dtype)` for the
/// tensors to read back. No `SpyreMemoryHierarchy::new`, no `marshal_inputs` —
/// this is the path that eliminates the per-pass / per-segment weight re-marshal
/// the fresh-context `execute_function` pays.
///
/// `grid` is the function's grid (the caller threads native attention at its own
/// grid, fused segments at `[1,1]`). The GPU offloads (K-loop GEMM, map windows,
/// resident weight cache) ride along exactly as in `execute_function`.
pub fn execute_function_in(
    mem: &SpyreMemoryHierarchy,
    ops: &[Operation],
    grid: (usize, usize, usize),
    input_ptrs: &[(String, Value)],
    read: &[TensorMeta],
    plan_key: Option<u64>,
) -> Result<HashMap<String, Output>, String> {
    let grid_exec = GridExecutor::new(grid);
    let dispatch = Dispatch::shared();
    crate::comm_sched::execute_with_communication(
        &grid_exec, mem, ops, input_ptrs, dispatch, None, plan_key,
    )?;
    read_back(mem, read.to_vec(), None)
}

/// Like [`execute_function_in`] but WITHOUT the boundary read-back. The resident
/// executor's intermediate segments write their outputs to the persistent HBM,
/// where the next segment (and the single final read-back in `run_program`) read
/// them directly — so decoding each segment's outputs to host tiles every pass is
/// pure discarded work (the caller did `let _ =` on the result). Skipping it
/// removes ~one output-tile decode + alloc + copy per segment per pass.
pub fn execute_function_in_exec_only(
    mem: &SpyreMemoryHierarchy,
    ops: &[Operation],
    grid: (usize, usize, usize),
    input_ptrs: &[(String, Value)],
    plan_key: Option<u64>,
) -> Result<(), String> {
    let grid_exec = GridExecutor::new(grid);
    let dispatch = Dispatch::shared();
    crate::comm_sched::execute_with_communication(
        &grid_exec, mem, ops, input_ptrs, dispatch, None, plan_key,
    )
}

/// Like [`execute_function`], but records per-op latency and returns the report
/// alongside the outputs. Port of running `KTIRInterpreter` with a
/// `latency_config`. Every op (including region-nested ops, via the shared
/// `ExecutionEnv`) is metered; comm ops are charged by the scheduler.
pub fn execute_function_with_latency(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
    config: crate::latency::HardwareConfig,
) -> Result<(HashMap<String, Output>, crate::latency::LatencyReport), String> {
    use std::cell::RefCell;

    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;

    let mem = SpyreMemoryHierarchy::new(num_cores.max(1));
    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::shared();
    let tracker = RefCell::new(crate::latency::LatencyTracker::new(config));

    let (input_ptrs, tensor_meta) = marshal_inputs(&mem, args);

    crate::comm_sched::execute_with_communication(
        &grid,
        &mem,
        &func.operations,
        &input_ptrs,
        dispatch,
        Some(&tracker),
        None,
    )?;

    let outputs = read_back(&mem, tensor_meta, None)?;
    let report = tracker.borrow().report();
    Ok((outputs, report))
}

/// A memory region to seed before execution: raw `dtype`-encoded bytes. `elem`
/// is the ELEMENT-index base (the MLIR `construct_memory_view` constant, RFC
/// #110: `MemRef.base_ptr` is an element index). The byte address is
/// `elem * dtype.bytes_per_elem()`. When `lx_core` is `None`, written to HBM at
/// that byte address (decomposed into stick+intra); when `Some(core)`, written
/// to that core's LX scratchpad at that byte address. `next_ptr` (when `Some`)
/// advances that LX's allocation cursor past the seeded region (a BYTE pointer,
/// not an element index) so the kernel's own staging does not overwrite the seed.
#[derive(Clone, Debug)]
pub struct HbmSeed {
    pub elem: i64,
    pub dtype: DType,
    pub bytes: Vec<u8>,
    pub lx_core: Option<usize>,
    pub next_ptr: Option<i64>,
}

/// An HBM region to read back after execution: `n_elements` of `dtype` at the
/// ELEMENT-index base `elem` (byte address = `elem * dtype.bytes_per_elem()`).
#[derive(Clone, Debug)]
pub struct HbmRead {
    pub name: String,
    pub elem: i64,
    pub n_elements: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Execute a function whose tensor operands are NOT passed as marshalled ndarray
/// args but live at hardcoded HBM addresses (the RFC / ring-reduce fixtures:
/// `in_ptr`/`out_ptr` are stick addresses, and the `construct_memory_view` bases
/// are `arith.constant` stick indices). Seeds every `seeds` region into a fresh
/// HBM, binds `scalars` as input pointers, runs the grid through the comm
/// scheduler (so ring/collective ops work), then reads back every `reads` region.
///
/// This is the file-IO twin of the Python harness's `_prepare_execution` seeding:
/// both sides write byte-identical HBM, run the same function, and read back the
/// same stick region — so the diff is a true Python↔Rust head-to-head even for
/// programs with no ndarray arguments. Errors (e.g. an LX-overflow `MemoryError`
/// or an unmapped read) propagate as `Err`, letting the harness assert a
/// *matched failure* against Python's exception.
pub fn execute_function_seeded(
    module: &IRModule,
    func_name: &str,
    scalars: &[(String, Scalar)],
    seeds: &[HbmSeed],
    reads: &[HbmRead],
) -> Result<HashMap<String, Output>, String> {
    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;

    let mem = SpyreMemoryHierarchy::new(num_cores.max(1));

    // Seed HBM/LX exactly as Python's _prepare_execution hook. The seed base is
    // an ELEMENT index (the MLIR construct_memory_view constant, RFC #110), so
    // the byte address is elem*bytes_per_elem(dtype) for BOTH spaces. LX takes an
    // optional next_ptr bump (a byte pointer) so kernel staging won't trample it.
    {
        let hbm = mem.hbm.borrow_mut();
        for s in seeds {
            let byte_addr = s.elem * s.dtype.bytes_per_elem() as i64;
            match s.lx_core {
                None => hbm.write_bytes(byte_addr, &s.bytes),
                Some(core) => {
                    let lx = mem.get_lx(core);
                    let lx = lx.borrow_mut();
                    lx.write_bytes(byte_addr, &s.bytes);
                    if let Some(np) = s.next_ptr {
                        lx.next_ptr = np;
                    }
                }
            }
        }
    }

    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::shared();
    let input_ptrs: Vec<(String, Value)> = scalars
        .iter()
        .map(|(n, s)| (n.clone(), Value::Scalar(*s)))
        .collect();

    crate::comm_sched::execute_with_communication(
        &grid,
        &mem,
        &func.operations,
        &input_ptrs,
        dispatch,
        None,
        None,
    )?;

    // Read back each requested HBM region (byte addr = elem*bytes_per_elem),
    // decode to both raw bytes and f32 — same Output shape as the marshalled path.
    let hbm = mem.hbm.borrow();
    let mut out = HashMap::new();
    for r in reads {
        let nbytes = r.n_elements * r.dtype.bytes_per_elem();
        let raw = hbm.read_bytes(r.elem * r.dtype.bytes_per_elem() as i64, nbytes);
        let data = codec::decode(&raw, r.n_elements, r.dtype);
        out.insert(
            r.name.clone(),
            Output {
                data,
                shape: r.shape.clone(),
                dtype: r.dtype,
                raw,
            },
        );
    }
    Ok(out)
}

/// Tensor read-back metadata: `(name, stick, n_elements, shape, dtype)`.
pub type TensorMeta = (String, i64, usize, Vec<usize>, DType);

/// Marshal tensor args into HBM and return `(input_ptrs, tensor_meta)`.
/// Shared by the plain and latency-tracked execution paths.
fn marshal_inputs(
    mem: &SpyreMemoryHierarchy,
    args: &[(&str, Arg)],
) -> (Vec<(String, Value)>, Vec<TensorMeta>) {
    let mut input_ptrs: Vec<(String, Value)> = Vec::new();
    let mut tensor_meta: Vec<TensorMeta> = Vec::new();
    // Allocate an HBM stick, write `bytes`, and record the read-back metadata.
    fn place(
        mem: &SpyreMemoryHierarchy,
        input_ptrs: &mut Vec<(String, Value)>,
        tensor_meta: &mut Vec<TensorMeta>,
        name: &str,
        bytes: Vec<u8>,
        shape: &[usize],
        dtype: DType,
    ) {
        let stick = {
            let hbm = mem.hbm.borrow_mut();
            let stick = hbm.allocate(bytes.len() as i64);
            hbm.write_bytes(stick * STICK_BYTES, &bytes);
            stick
        };
        // MLIR pointer operands are element indices (MemRef.base_ptr contract),
        // not stick indices: elem_idx = stick * STICK_BYTES / bytes_per_elem.
        let elem_idx = stick * STICK_BYTES / dtype.bytes_per_elem() as i64;
        input_ptrs.push((name.to_string(), Value::Index(elem_idx)));
        tensor_meta.push((
            name.to_string(),
            stick,
            shape.iter().product(),
            shape.to_vec(),
            dtype,
        ));
    }
    for (name, arg) in args {
        match arg {
            // f32 host data: narrow to `dtype` on the way in.
            Arg::Tensor { data, shape, dtype } => place(
                mem,
                &mut input_ptrs,
                &mut tensor_meta,
                name,
                codec::encode(data, *dtype),
                shape,
                *dtype,
            ),
            // Pre-encoded typed bytes: straight to HBM, no conversion.
            Arg::TensorBytes { data, shape, dtype } => place(
                mem,
                &mut input_ptrs,
                &mut tensor_meta,
                name,
                data.clone(),
                shape,
                *dtype,
            ),
            // bf16 host bytes -> f16 HBM layout in ONE fused pass (no f32 buffer).
            Arg::TensorBf16 { data, shape } => place(
                mem,
                &mut input_ptrs,
                &mut tensor_meta,
                name,
                codec::bf16_to_f16(data, shape.iter().product()),
                shape,
                DType::F16,
            ),
            Arg::Scalar(s) => input_ptrs.push((name.to_string(), Value::Scalar(*s))),
        }
    }
    (input_ptrs, tensor_meta)
}

/// Opt-in GPU/Spyre-faithful (**f16**) execution of a pure-SPMD grid: step all
/// cores in lockstep and COMBINE their shared-weight `linalg.matmul`s into one
/// zero-copy NAX dispatch (the grid's many small matmuls become one tall one —
/// 1.2–2.6× over a serial AMX loop). Restricted to no-comm, straight-line
/// (region-free) functions on an M5; returns `Err` otherwise so the caller can
/// fall back to [`execute_function`].
///
/// The NAX kernel runs in **f16** — Spyre's matmul precision, and exactly what
/// the interpreter rounds every tile to (`f32` accumulate → f16). So results
/// match the f32/`execute_function` path to f16 tolerance (only the GEMM
/// accumulation order differs). Kept opt-in for now while the lockstep executor
/// is young; it is precision-faithful, not a lossy mode.
#[cfg(metal)]
pub fn execute_function_gpu(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
) -> Result<HashMap<String, Output>, String> {
    use crate::metal::NaxGemm;

    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;
    let ops = &func.operations;

    // Applicable only to a multi-core, comm-free, straight-line SPMD body.
    if num_cores <= 1
        || ops
            .iter()
            .any(|o| crate::comm_sched::is_comm_op(&o.op_type) || !o.regions.is_empty())
    {
        return Err("execute_function_gpu: not a pure-SPMD straight-line grid".into());
    }
    let gemm = NaxGemm::new()?; // Err on non-M5 / no device -> caller falls back

    let mem = SpyreMemoryHierarchy::new(num_cores);
    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::shared();
    let env = ExecutionEnv::new(dispatch, &grid);
    let (input_ptrs, tensor_meta) = marshal_inputs(&mem, args);

    let mut ctxs: Vec<CoreContext> = (0..num_cores)
        .map(|c| {
            let mut ctx = CoreContext::new(
                c,
                grid.linear_to_grid(c),
                Rc::clone(&mem.hbm),
                mem.get_lx(c),
                mem.lx_scratchpads.clone(),
            );
            for (name, val) in &input_ptrs {
                ctx.set_value(name, val.clone());
            }
            ctx
        })
        .collect();

    // GATED map-window offload (part B): when `KTIR_FORCE_GPU_MAP` is set, plan
    // this straight-line body's fused map windows and run each window's TRIGGER on
    // the Metal map kernel PER CORE (a per-element map is core-local, so offloading
    // it per core is identical to the interpreter). This is what puts the non-F16
    // elementwise programs (vector_add_dynamic f32, indexed_add i64-gather) onto
    // Metal on the GPU path — the all-F16 resident path cannot drive their dtypes,
    // but `execute_function`'s per-core path (this one) handles them. STRICTLY
    // gated: when the force flag is unset the plan is empty and the body runs
    // op-by-op exactly as before, so the unforced path is byte-identical.
    #[cfg(metal)]
    let (map_triggers, map_skip): (
        std::collections::HashMap<usize, crate::metal::MapRegionKernel>,
        std::collections::HashSet<usize>,
    ) = if crate::metal::force_gpu_map() && std::env::var_os("KTIR_NO_GPU_MAP").is_none() {
        crate::metal::map_fusion_plan(ops)
    } else {
        (
            std::collections::HashMap::new(),
            std::collections::HashSet::new(),
        )
    };

    for (i, op) in ops.iter().enumerate() {
        // Combine a shared-weight 2-operand matmul across all cores into one
        // dispatch; fall through to per-core execution if it doesn't apply.
        if op.op_type == "linalg.matmul"
            && op.operands.len() == 2
            && try_combine_matmul(op, &mut ctxs, &gemm)?
        {
            continue;
        }
        #[cfg(metal)]
        if let Some(mrk) = map_triggers.get(&i) {
            // Window trigger: run the fused kernel PER CORE. A failure is fatal
            // (the window's other ops were skipped — no interpreter fallback).
            for ctx in &mut ctxs {
                crate::metal::run_map_region_gpu(mrk, ctx)?;
            }
            continue;
        }
        #[cfg(metal)]
        if map_skip.contains(&i) {
            // Subsumed by the trigger's fused kernel — not executed on any core.
            continue;
        }
        for ctx in &mut ctxs {
            execute_op(op, ctx, &env)?;
        }
    }
    read_back(&mem, tensor_meta, None)
}

/// Combine `op` (a 2-operand `linalg.matmul`) across all cores when every core's
/// weight operand B is identical: stack the per-core A panels into one tall
/// GEMM, run it zero-copy on NAX, and scatter the row-blocks back. Returns
/// `Ok(true)` if combined, `Ok(false)` to fall back to per-core execution.
#[cfg(metal)]
fn try_combine_matmul(
    op: &Operation,
    ctxs: &mut [CoreContext],
    gemm: &crate::metal::NaxGemm,
) -> Result<bool, String> {
    use crate::metal::Epilogue;
    use crate::tile::Tile;

    let result = match op.result.as_deref() {
        Some(r) => r,
        None => return Ok(false),
    };
    // Read core 0's operands to fix the shapes and the shared weights.
    let (a0, b0) = (
        as_tile(&ctxs[0], &op.operands[0])?,
        as_tile(&ctxs[0], &op.operands[1])?,
    );
    if a0.shape.len() != 2 || b0.shape.len() != 2 || a0.shape[1] != b0.shape[0] {
        return Ok(false);
    }
    let (m, k, n) = (a0.shape[0], a0.shape[1], b0.shape[1]);
    let dtype = a0.dtype;
    let shared_b = b0.as_f32().into_owned();
    let a_shape = a0.shape.clone();
    let b_shape = b0.shape.clone();

    // Gather A panels; bail (fall back) unless every core shares B exactly.
    let mut a_stack = Vec::with_capacity(ctxs.len() * m * k);
    for ctx in ctxs.iter() {
        let a = as_tile(ctx, &op.operands[0])?;
        let b = as_tile(ctx, &op.operands[1])?;
        if a.shape != a_shape || b.shape != b_shape || *b.as_f32() != *shared_b {
            return Ok(false);
        }
        a_stack.extend_from_slice(&a.as_f32());
    }

    // One zero-copy NAX dispatch for the whole grid's matmul.
    let ua = gemm.unified_from(&a_stack)?;
    let ub = gemm.unified_from(&shared_b)?;
    let mut uc = gemm.unified(ctxs.len() * m * n)?;
    gemm.matmul_unified(
        ctxs.len() * m,
        k,
        n,
        &ua,
        &ub,
        &mut uc,
        None,
        Epilogue::NONE,
        false,
    )?;

    // Scatter each core's row-block back as its matmul result.
    let c = uc.as_slice();
    for (i, ctx) in ctxs.iter_mut().enumerate() {
        let block = c[i * m * n..(i + 1) * m * n].to_vec();
        let tile = Tile::compute(block, dtype, vec![m, n]);
        ctx.track_lx(result, tile.size_bytes() as i64)?;
        ctx.set_value(result, Value::Tile(tile));
    }
    Ok(true)
}

/// Read every tensor arg back out of HBM into an `Output`.
fn read_back(
    mem: &SpyreMemoryHierarchy,
    tensor_meta: Vec<TensorMeta>,
    wanted: Option<&std::collections::HashSet<String>>,
) -> Result<HashMap<String, Output>, String> {
    let mut outputs = HashMap::new();
    for (name, stick, n, shape, dtype) in tensor_meta {
        if let Some(w) = wanted
            && !w.contains(name.trim_start_matches('%'))
        {
            continue;
        }
        let nbytes = n * dtype.bytes_per_elem();
        let bytes = mem.hbm.borrow().read_bytes(stick * STICK_BYTES, nbytes);
        let data = codec::decode(&bytes, n, dtype);
        outputs.insert(
            name,
            Output {
                data,
                shape,
                dtype,
                raw: bytes,
            },
        );
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::dtypes::DType;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::ir::{Attr, Operation, Scalar};
    use crate::tile::Tile;

    fn run(ops: &[Operation]) -> CoreContext {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        execute_ops(ops, &mut ctx, &env).unwrap();
        ctx
    }

    #[test]
    fn scalar_constant_fold_chain() {
        let ops = vec![
            Operation::new(Some("%a"), "arith.constant", &[]).with_attr("value", Attr::Float(2.0)),
            Operation::new(Some("%b"), "arith.constant", &[]).with_attr("value", Attr::Float(3.0)),
            Operation::new(Some("%c"), "arith.addf", &["%a", "%b"]),
            Operation::new(Some("%d"), "arith.mulf", &["%c", "%a"]),
        ];
        let ctx = run(&ops);
        match ctx.get_value("%d").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected F32(10.0), got {other:?}"),
        }
    }

    #[test]
    fn elementwise_tile_add_tracks_lx() {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        ctx.set_value(
            "%x",
            Value::Tile(Tile::compute(vec![1.0, 2.0, 3.0], DType::F32, vec![3])),
        );
        ctx.set_value(
            "%y",
            Value::Tile(Tile::compute(vec![10.0, 20.0, 30.0], DType::F32, vec![3])),
        );
        let ops = vec![Operation::new(Some("%z"), "arith.addf", &["%x", "%y"])];
        execute_ops(&ops, &mut ctx, &env).unwrap();
        match ctx.get_value("%z").unwrap() {
            Value::Tile(t) => assert_eq!(t.as_f32().to_vec(), vec![11.0, 22.0, 33.0]),
            other => panic!("expected tile, got {other:?}"),
        }
        // the result Tile was tracked in LX (3 * f32 = 12 bytes)
        assert_eq!(ctx.lx.borrow().used, 12);
    }

    #[test]
    fn unknown_op_errors() {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        let ops = vec![Operation::new(Some("%z"), "ktdp.not_yet", &[])];
        let err = execute_ops(&ops, &mut ctx, &env).unwrap_err();
        assert!(err.contains("no handler registered"));
    }
}
