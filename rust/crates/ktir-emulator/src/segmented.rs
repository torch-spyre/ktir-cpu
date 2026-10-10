// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! PARTIAL-FUSION execution — the production serving path for a whole KTIR
//! program (a multi-function bundle whose nodes thread intermediates through
//! HBM).
//!
//! [`ktir_optimizer::fusion::plan_segments`] splits the program into ordered
//! segments: a [`Segment::Fused`] is a maximal run of consecutive
//! non-attention nodes collapsed into one `[1,1]` function (intra-run HBM edges
//! forwarded as SSA / `tensor.extract_slice`, all the GPU offloads — K-loop
//! GEMM reconstruction, fused map windows, the resident weight cache — riding
//! along on it), and a [`Segment::Native`] is a single head-parallel attention
//! node kept verbatim so it runs across its NATIVE multi-core grid
//! (`[32,1]`/`[9,1]` etc.). The head-parallel split matters: collapsed into a
//! `[1,1]` function the per-core `ktdp.get_compute_tile_id` head select returns
//! 0, so only head 0's slice would be computed; running the node at its native
//! grid drives every head's core (the verified multi-core SPMD path).
//!
//! [`execute_segmented`] is the real callable: it plans the segments, threads
//! one shared host buffer per logical tensor through HBM in program order, runs
//! each segment (fused via `execute_function_outputs` at `[1,1]`, native via
//! `execute_function` at its grid), and reads back the requested outputs. This
//! is the per-node threading the per-node oracle proves correct, with the
//! non-attention runs collapsed into fused GPU-accelerated segments.
//!
//! Tensor naming: `args` and `outputs` are keyed by the canonical tensor name
//! `t<id>` (the leading `%` / trailing `_ptr` of the fused pointer-arg form
//! `%t<id>_ptr` are tolerated, as is a bare `<id>`). Sources (weights, inputs,
//! the attention mask) are supplied in `args`; intermediates, the final result,
//! and any intra-segment scratch are sized from the IR and zero-initialized.

use crate::dtypes::DType;
use crate::interpreter::{Arg, Output, execute_function, execute_function_outputs};
use crate::ir::{Attr, IRModule, Operation};
use ktir_optimizer::fusion::{ProgramSpec, Segment, plan_segments_budgeted};
use std::collections::HashMap;

/// Recover the logical tensor id from a fused pointer-arg name `%t<id>_ptr`.
fn tensor_id_of_arg(arg: &str) -> Result<u64, String> {
    arg.trim_start_matches('%')
        .trim_start_matches('t')
        .trim_end_matches("_ptr")
        .parse()
        .map_err(|_| format!("unexpected fused pointer-arg name {arg:?}"))
}

/// Normalize a caller-supplied tensor key (`t<id>`, `%t<id>`, `%t<id>_ptr`, or a
/// bare `<id>`) to its numeric tensor id.
fn tensor_id_of_key(key: &str) -> Result<u64, String> {
    let s = key.trim_start_matches('%');
    let s = s.strip_prefix('t').unwrap_or(s);
    let s = s.strip_suffix("_ptr").unwrap_or(s);
    s.parse()
        .map_err(|_| format!("cannot parse tensor id from arg/output key {key:?}"))
}

/// The integer element-shape attribute on a `construct_memory_view` op.
fn view_shape_of(op: &Operation) -> Option<Vec<usize>> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) if !v.is_empty() => Some(v.iter().map(|&x| x as usize).collect()),
        _ => None,
    }
}

/// Derive `tensor_id -> element-shape` for EVERY logical tensor the program
/// touches, by scanning each node's `ktdp.construct_memory_view` ops (whose
/// `sizes:` attribute is the tensor's full shape) and mapping the view's pointer
/// operand to a tensor id via that node's bindings. A tensor may be viewed in
/// several nodes; any one view's shape is authoritative (they agree). This lets
/// the driver size the intermediate / result / scratch buffers with NO external
/// manifest — the shapes live in the IR.
fn derive_shapes(
    module: &IRModule,
    spec: &ProgramSpec,
) -> Result<HashMap<u64, Vec<usize>>, String> {
    let mut shapes: HashMap<u64, Vec<usize>> = HashMap::new();
    for node in &spec.nodes {
        let func = module.get_function(&node.func)?;
        let arg_to_tensor: HashMap<&str, u64> = node
            .bindings
            .iter()
            .map(|b| (b.arg.as_str(), b.tensor))
            .collect();
        collect_view_shapes(&func.operations, &arg_to_tensor, &mut shapes);
    }
    Ok(shapes)
}

/// Walk ops (recursing into regions) recording the shape of every memory view
/// whose pointer operand is a known node arg → tensor id.
fn collect_view_shapes(
    ops: &[Operation],
    arg_to_tensor: &HashMap<&str, u64>,
    shapes: &mut HashMap<u64, Vec<usize>>,
) {
    for op in ops {
        if op.op_type == "ktdp.construct_memory_view"
            && let Some(ptr) = op.operands.first()
            && let Some(&tid) = arg_to_tensor.get(ptr.as_str())
            && let Some(shape) = view_shape_of(op)
        {
            shapes.entry(tid).or_insert(shape);
        }
        for rg in &op.regions {
            collect_view_shapes(rg, arg_to_tensor, shapes);
        }
    }
}

/// Execute a whole KTIR program via PARTIAL FUSION — the real serving path.
///
/// Plans `spec` into ordered segments with [`plan_segments`] and runs them in
/// program order, threading one shared host buffer per logical tensor through
/// HBM:
///
/// * [`Segment::Fused`] — a `[1,1]` function over a run of non-attention nodes,
///   carrying the GPU offloads (matmul-loop GEMM reconstruction, fused map
///   windows, the resident weight cache). Every surviving pointer arg is one
///   of: a boundary INPUT (a source / earlier output, fed from the live
///   buffer), a boundary OUTPUT (zero-init, read back and threaded forward), or
///   internal SCRATCH (a non-forwardable intra-segment edge the fused body
///   writes then reads — zero-init, never threaded). Only the boundary outputs
///   are read back (selective readback via [`execute_function_outputs`]), so
///   the hundreds of resident weight pointers are not decoded for nothing.
/// * [`Segment::Native`] — a single head-parallel attention node run at its
///   native multi-core grid (every head's core driven), threading buffers
///   exactly like the per-node oracle.
///
/// `args` supplies the program's sources (weights, inputs, the attention mask),
/// keyed by tensor name `t<id>` (`%t<id>` / `%t<id>_ptr` / bare `<id>` are also
/// accepted). Intermediates, the final result, and intra-segment scratch are
/// sized from the IR (`construct_memory_view` shapes) and zero-initialized.
/// `outputs` names the tensors to return (same keying); the returned map is
/// keyed by `t<id>` for each requested output the program produced.
///
/// This is the production analogue of the per-node oracle: attention runs
/// head-parallel at its native grid, and the fused `[1,1]` segments carry the
/// map / GEMM / weight-cache GPU offloads.
/// Apply the attention IR-rewrite optimizer passes (head re-roll, then flash
/// cap-tiling) to `module` in place, under the LX scores budget Contract B uses.
///
/// Run at the EXECUTION ENTRY POINT (here + [`crate::resident::ResidentExecutor`])
/// rather than in one specific module-builder, so the optimizer is GUARANTEED to
/// run for *every* path that executes a module — the turnkey
/// [`crate::program::execute`], a [`crate::program::Session`], AND a caller that
/// built the module itself and calls [`execute_segmented`] directly (e.g. the
/// real-model e2e harness). Both passes fail-safe to a no-op on any non-matching
/// node and are idempotent (re-running on already-rewritten IR recognizes
/// nothing), so a redundant application is harmless. `KTIR_FLASH_ATTN_SCORES_BUDGET`
/// overrides the budget (a tiny value forces flash to own everything — the knob
/// the FA golden uses).
pub(crate) fn apply_attention_rewrites(module: &mut IRModule) {
    // Baseline knob: skip the rewrites entirely (used to A/B the e2e wall-clock
    // of the optimized vs unoptimized real prefill). Default OFF — production
    // always optimizes.
    if std::env::var_os("KTIR_NO_ATTENTION_REWRITE").is_some() {
        if std::env::var_os("KTIR_REWRITE_VERBOSE").is_some() {
            eprintln!("[ktir-optimizer] attention rewrites DISABLED (KTIR_NO_ATTENTION_REWRITE)");
        }
        return;
    }
    let budget = std::env::var("KTIR_FLASH_ATTN_SCORES_BUDGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(crate::memory::lx_fusion_budget);
    // HEAD RE-ROLL (below-cap, TODO #1): unrolled per-query-row head-parallel
    // attention -> whole-`[m,*]` tensor ops. FLASH (above-cap, TODO #2): cap-tile
    // the re-rolled `[m,cap]` context block with online softmax. Disjoint via the
    // shared `attention_needs_flash` predicate; head runs first so flash sees its
    // re-rolled form.
    let n_head = ktir_optimizer::head_rewrite::apply_head_rewrite(module, |scores_bytes| {
        ktir_optimizer::fusion::attention_needs_flash(scores_bytes, budget)
    });
    let n_flash = ktir_optimizer::flash_attn::apply_flash_attention(module, |scores_bytes| {
        ktir_optimizer::fusion::attention_needs_flash(scores_bytes, budget)
    });
    // TILE COALESCE: collapse K>=2 structurally-identical dim-0-tiled
    // elementwise blocks (e.g. RoPE) into one whole-height block. Fail-safe and
    // independent of the attention rewrites; default ON, skippable for A/B.
    let n_coalesce = if std::env::var_os("KTIR_NO_TILE_COALESCE").is_some() {
        if std::env::var_os("KTIR_REWRITE_VERBOSE").is_some() {
            eprintln!("[ktir-optimizer] tile coalesce DISABLED (KTIR_NO_TILE_COALESCE)");
        }
        0
    } else {
        ktir_optimizer::tile_coalesce::apply_tile_coalesce(module)
    };
    if std::env::var_os("KTIR_REWRITE_VERBOSE").is_some() {
        eprintln!(
            "[ktir-optimizer] attention rewrites @ execution entry: head re-roll fired on {n_head} node(s), flash cap-tiling fired on {n_flash} node(s) (budget={budget} bytes); tile coalesce fired on {n_coalesce} node(s)"
        );
    }
}

pub fn execute_segmented(
    module: &IRModule,
    spec: &ProgramSpec,
    args: &[(&str, Arg)],
    outputs: &[&str],
) -> Result<HashMap<String, Output>, String> {
    // Optimize at the execution entry (see `apply_attention_rewrites`): every
    // executed module gets the attention rewrites, regardless of how it was built.
    // We only borrow the caller's module and the rewrite mutates, so clone first.
    let prof = std::env::var_os("KTIR_PROFILE").is_some();
    let t_seg0 = std::time::Instant::now();
    let mut owned = module.clone();
    apply_attention_rewrites(&mut owned);
    let module = &owned;
    let shapes = derive_shapes(module, spec)?;
    // LX-budgeted segmentation (see `plan_segments_budgeted`): keep each fused
    // segment's co-resident `[m, *]` intermediates within the per-core LX.
    let tensor_bytes: HashMap<u64, usize> = shapes
        .iter()
        .map(|(&id, shp)| {
            (
                id,
                shp.iter().product::<usize>() * DType::F16.bytes_per_elem(),
            )
        })
        .collect();
    let segments = plan_segments_budgeted(
        module,
        spec,
        crate::memory::lx_fusion_budget(),
        &tensor_bytes,
    )?;
    let ms_setup = t_seg0.elapsed().as_secs_f64() * 1e3;

    // One shared host buffer per logical tensor (the emulated HBM). Sources are
    // seeded from `args`; intermediates / results / scratch are written as the
    // segments run. We thread as f32 host data + dtype, narrowing into HBM on
    // each segment call (the dtype-agnostic oracle marshalling).
    let mut buf: HashMap<u64, (Vec<f32>, DType)> = HashMap::new();
    for (key, arg) in args {
        let tid = tensor_id_of_key(key)?;
        let (data, dtype) = match arg {
            Arg::Tensor { data, dtype, .. } => (data.clone(), *dtype),
            Arg::TensorBytes { data, shape, dtype } => (
                crate::codec::decode(data, shape.iter().product(), *dtype),
                *dtype,
            ),
            // bf16 host bytes: widen to f32 (exact); threaded as the f16 model dtype.
            Arg::TensorBf16 { data, shape } => (
                crate::codec::bf16_to_f32(data, shape.iter().product()),
                DType::F16,
            ),
            Arg::Scalar(_) => {
                return Err(format!(
                    "scalar arg {key:?} unsupported in execute_segmented"
                ));
            }
        };
        buf.insert(tid, (data, dtype));
    }

    // A zero buffer + dtype + shape for a tensor id, sized from the derived IR
    // shape (F16 — the model dtype, as the per-node oracle threads it).
    let zeroed = |tid: u64| -> Result<(Vec<f32>, DType, Vec<usize>), String> {
        let shape = shapes
            .get(&tid)
            .cloned()
            .ok_or_else(|| format!("no shape derivable for tensor t{tid}"))?;
        Ok((vec![0.0f32; shape.iter().product()], DType::F16, shape))
    };

    let diag = std::env::var_os("KTIR_SEG_DIAG").is_some();
    let mut t_fused = 0.0f64;
    let mut t_native = 0.0f64;
    let mut n_fused = 0usize;
    let mut n_native = 0usize;
    let ms_premarshal = t_seg0.elapsed().as_secs_f64() * 1e3;
    for seg in &segments {
        let seg_t0 = std::time::Instant::now();
        match seg {
            // A fused segment: marshal every surviving pointer arg, run the
            // `[1,1]` fused function, and copy back every boundary OUTPUT.
            Segment::Fused(fs) => {
                let mut call_args: Vec<(String, Arg)> = Vec::new();
                let mut want: Vec<String> = Vec::new();
                for (arg_name, _) in &fs.func.arguments {
                    let tid = tensor_id_of_arg(arg_name)?;
                    let bare = arg_name.trim_start_matches('%').to_string();
                    let (data, dtype, shape) = if fs.inputs.contains(&tid) {
                        // Boundary input: must already be resident.
                        let (data, dtype) = buf.get(&tid).cloned().ok_or_else(|| {
                            format!("fused segment input t{tid} not produced before it ran")
                        })?;
                        let shape = shapes
                            .get(&tid)
                            .cloned()
                            .ok_or_else(|| format!("no shape for fused input t{tid}"))?;
                        (data, dtype, shape)
                    } else {
                        // Boundary output OR internal scratch: zero-init. Only
                        // boundary outputs are read back and threaded forward.
                        if fs.outputs.contains(&tid) {
                            want.push(bare.clone());
                        }
                        zeroed(tid)?
                    };
                    call_args.push((bare, Arg::Tensor { data, shape, dtype }));
                }
                let refs: Vec<(&str, Arg)> = call_args
                    .iter()
                    .map(|(n, a)| (n.as_str(), a.clone()))
                    .collect();
                let want_refs: Vec<&str> = want.iter().map(|s| s.as_str()).collect();
                let mut seg_module = IRModule::default();
                seg_module.add_function(fs.func.clone());
                let out = execute_function_outputs(&seg_module, &fs.func.name, &refs, &want_refs)?;
                for name in &want {
                    let tid = tensor_id_of_arg(name)?;
                    let o = out
                        .get(name)
                        .ok_or_else(|| format!("fused segment output {name} not read back"))?;
                    buf.insert(tid, (o.data.clone(), o.dtype));
                }
            }
            // A native attention node: run at its OWN grid (multi-core SPMD over
            // heads), threading buffers exactly like the per-node oracle.
            Segment::Native(node) => {
                let mut call_args: Vec<(String, Arg)> = Vec::new();
                let mut out_ids: Vec<(String, u64)> = Vec::new();
                for bind in &node.bindings {
                    let tid = bind.tensor;
                    let name = bind.arg.trim_start_matches('%').to_string();
                    let (data, dtype, shape) = if bind.is_output {
                        out_ids.push((name.clone(), tid));
                        zeroed(tid)?
                    } else {
                        let (data, dtype) = buf.get(&tid).cloned().ok_or_else(|| {
                            format!("native attn input t{tid} not produced before it ran")
                        })?;
                        let shape = shapes
                            .get(&tid)
                            .cloned()
                            .ok_or_else(|| format!("no shape for native attn input t{tid}"))?;
                        (data, dtype, shape)
                    };
                    call_args.push((name, Arg::Tensor { data, shape, dtype }));
                }
                let refs: Vec<(&str, Arg)> = call_args
                    .iter()
                    .map(|(n, a)| (n.as_str(), a.clone()))
                    .collect();
                let out = execute_function(module, &node.func, &refs)?;
                for (name, tid) in &out_ids {
                    let o = out
                        .get(name)
                        .ok_or_else(|| format!("native attn output {name} not read back"))?;
                    buf.insert(*tid, (o.data.clone(), o.dtype));
                }
            }
        }
        if diag {
            let dt = seg_t0.elapsed().as_secs_f64() * 1e3;
            match seg {
                Segment::Fused(_) => {
                    t_fused += dt;
                    n_fused += 1;
                }
                Segment::Native(_) => {
                    t_native += dt;
                    n_native += 1;
                }
            }
        }
    }
    if diag {
        eprintln!(
            "  [seg-diag] {n_fused} fused {t_fused:.1}ms | {n_native} native {t_native:.1}ms"
        );
    }
    if prof {
        let ms_loop = t_seg0.elapsed().as_secs_f64() * 1e3 - ms_premarshal;
        eprintln!(
            "  [seg-phases] setup(clone+rewrite+plan) {ms_setup:.0}ms | args-decode(f16->f32) {:.0}ms | loop(marshal+run+readback) {ms_loop:.0}ms | {} segments",
            ms_premarshal - ms_setup,
            segments.len()
        );
    }

    // Return the requested outputs, keyed by canonical `t<id>`.
    let mut result: HashMap<String, Output> = HashMap::new();
    for key in outputs {
        let tid = tensor_id_of_key(key)?;
        let (data, dtype) = buf
            .get(&tid)
            .cloned()
            .ok_or_else(|| format!("requested output t{tid} was not produced"))?;
        let shape = shapes
            .get(&tid)
            .cloned()
            .unwrap_or_else(|| vec![data.len()]);
        let raw = crate::codec::encode(&data, dtype);
        result.insert(
            format!("t{tid}"),
            Output {
                data,
                shape,
                dtype,
                raw,
            },
        );
    }
    Ok(result)
}
