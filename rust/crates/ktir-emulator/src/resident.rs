// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! RESIDENT GPU execution — the production serving path that keeps weights
//! resident across passes and segments.
//!
//! The prior [`crate::segmented::execute_segmented`] is correct but pays, on
//! EVERY pass, a full weight MARSHAL: each segment call allocates a fresh
//! [`SpyreMemoryHierarchy`] and re-encodes + re-writes every weight tensor's f16
//! bytes into a brand-new HBM (and on a multi-segment program — Llama prefill's
//! interleaved attention nodes — that marshal runs once per segment, many times
//! per pass). On Llama-1B (~2 GB of f16 weights) that is 2–4 s/pass of pure
//! data movement plus the alloc/free churn of 2 GB per pass — the regression
//! that made fused-Metal decode 8× SLOWER than CPU.
//!
//! [`ResidentExecutor`] fixes that structurally:
//!
//! * ONE persistent [`SpyreMemoryHierarchy`]. Every logical tensor is allocated
//!   an HBM stick ONCE, at construction, so the pointer the IR binds for a weight
//!   is STABLE across passes. Sources (weights / the attention mask / the input)
//!   are written once in [`ResidentExecutor::set_source`]; they are never
//!   re-marshaled.
//! * The GPU side is already resident: the thread-local `WEIGHT_CACHE` in
//!   `metal` decodes+uploads each GEMM weight to a [`UnifiedBuffer`] at
//!   most once and serves every subsequent pass from it (keyed by a content
//!   fingerprint, so it can never serve a stale weight). With the HBM bytes now
//!   stable too, the whole weight working set is uploaded exactly once.
//! * PER PASS [`ResidentExecutor::run`] only (a) rewrites the changing program
//!   INPUT activation(s) in place, (b) zeroes the result / intermediate / scratch
//!   sticks (small — kB, not GB), and (c) runs each planned segment against the
//!   SAME persistent HBM via [`crate::interpreter::execute_function_in`] — no
//!   `SpyreMemoryHierarchy::new`, no `marshal_inputs`, no fresh 2 GB alloc.
//!   Intermediates stay in the one persistent HBM across segments; the fused
//!   K-loop GEMMs (NaxGemm on resident weight buffers), fused map-window MSL
//!   kernels, and reductions chain on it exactly as before, but with zero weight
//!   re-marshal in the loop.
//!
//! The attention islands (native segments) run at their native head-parallel
//! grid against the SAME persistent HBM — so an attention segment does NOT
//! trigger a full per-segment weight re-marshal either (the previous segmented
//! path re-marshaled every attention input into a fresh HBM per node).
//!
//! Correctness is identical to `execute_segmented`: the same segment plan, the
//! same per-op handlers and GPU offloads, the same f16 precision. The only
//! difference is WHERE the bytes live (one resident HBM vs a fresh one per call)
//! — so the golden results are unchanged.

use crate::dtypes::DType;
use crate::interpreter::{Arg, Output, TensorMeta, execute_function_in_exec_only};
use crate::ir::{Attr, IRModule, Operation, Value};
use crate::memory::{STICK_BYTES, SpyreMemoryHierarchy};
use ktir_optimizer::fusion::{NodeSpec, ProgramSpec, Segment, plan_segments_budgeted};
use std::collections::HashMap;

/// One planned program before it becomes a [`ResidentProgram`]: the (rewritten)
/// module, its execution segments, its result tensor ids, and the raw grid=[H,1]
/// nodes the compute-tile dataflow executor runs per tile.
type PlannedProgram = (
    IRModule,
    Vec<Segment>,
    std::collections::HashSet<u64>,
    Vec<NodeSpec>,
);

/// For KTIR_SEG_PROF: scan a fused function's ops and return (label, n_matmuls)
/// where label is the SHAPE of its largest `linalg.matmul` (by k·n), recovered from
/// the operand/result tensor types (e.g. `1x8192` ins → `2048x8192` weight). The
/// scf.for K-loop's per-step matmul carries `tensor<MxKBxf16>` / `tensor<NxKBxf16>`;
/// the full weight view shape is on the `construct_memory_view`. We approximate the
/// GEMM size by the largest 2-D memory-view (the weight) and count matmul ops.
fn dominant_gemm(ops: &[Operation]) -> (String, usize) {
    fn walk(ops: &[Operation], best: &mut (usize, usize, usize), nmm: &mut usize) {
        for op in ops {
            if op.op_type == "linalg.matmul" {
                *nmm += 1;
            }
            if op.op_type == "ktdp.construct_memory_view"
                && let Some(Attr::IntList(v)) = op.attributes.get("shape")
                && v.len() == 2
            {
                let (a, b) = (v[0] as usize, v[1] as usize);
                if a * b > best.0 * best.1 {
                    *best = (a, b, a * b);
                }
            }
            for rg in &op.regions {
                walk(rg, best, nmm);
            }
        }
    }
    let mut best = (0usize, 0usize, 0usize);
    let mut nmm = 0usize;
    walk(ops, &mut best, &mut nmm);
    (format!("{}x{}", best.0, best.1), nmm)
}

/// Compact op-type histogram (e.g. "linalg.reduce×2 linalg.broadcast×2 arith.mulf×4")
/// over a function's ops, recursing into regions — for diagnosing no-matmul segments.
fn op_type_histogram(ops: &[Operation]) -> String {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    fn walk(ops: &[Operation], counts: &mut std::collections::BTreeMap<String, usize>) {
        for op in ops {
            *counts.entry(op.op_type.clone()).or_default() += 1;
            for rg in &op.regions {
                walk(rg, counts);
            }
        }
    }
    walk(ops, &mut counts);
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    v.iter()
        .take(8)
        .map(|(k, n)| format!("{k}×{n}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Recover the logical tensor id from a fused pointer-arg name `%t<id>_ptr`.
fn tensor_id_of_arg(arg: &str) -> Result<u64, String> {
    arg.trim_start_matches('%')
        .trim_start_matches('t')
        .trim_end_matches("_ptr")
        .parse()
        .map_err(|_| format!("unexpected fused pointer-arg name {arg:?}"))
}

/// Normalize a caller key (`t<id>`, `%t<id>`, `%t<id>_ptr`, bare `<id>`) -> id.
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

/// Walk ops (recursing into regions) recording every memory view's shape keyed by
/// the tensor id its pointer operand binds. Mirrors `segmented::collect_view_shapes`.
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

/// Derive `tensor_id -> element-shape` for every logical tensor the program
/// touches (from each node's `construct_memory_view` shapes). Same derivation the
/// segmented executor uses — the shapes live in the IR, no external manifest.
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

/// Restores the GPU weight-cache "trusted" flag to its prior value on drop, so a
/// resident `run` enables it only for the duration of that call.
#[cfg(metal)]
struct TrustedWeightsGuard(bool);
#[cfg(metal)]
impl Drop for TrustedWeightsGuard {
    fn drop(&mut self) {
        crate::metal::set_trusted_weights(self.0);
    }
}

/// Marks the resident run as PARALLEL-SAFE (every stick is pre-allocated at
/// construction, so the multi-core attention cores never call `hbm.allocate()`
/// during the parallel section — the race that corrupted the heap on the
/// `execute_function` path). Restores the prior value on drop.
struct ParallelSafeGuard(bool);
impl Drop for ParallelSafeGuard {
    fn drop(&mut self) {
        crate::comm_sched::set_parallel_safe(self.0);
    }
}

/// A persistent resident-execution context for one KTIR program.
///
/// Build once with [`ResidentExecutor::new`], write the source weights once with
/// [`ResidentExecutor::set_source`] (or [`ResidentExecutor::set_sources`]), then
/// call [`ResidentExecutor::run`] per pass. Weights are NEVER re-marshaled.
/// One program (e.g. prefill or decode) sharing the executor's resident weights.
/// Holds its OWN module + planned segments + result tensor ids; binds the SHARED
/// sticks by tensor id at run time, so the weights it reads were uploaded once.
struct ResidentProgram {
    /// OWNED (not borrowed) so the executor holds its ENTIRE `Rc` graph
    /// exclusively — see the `unsafe impl Send` below.
    module: IRModule,
    segments: Vec<Segment>,
    /// The RAW grid=[H,1] nodes (pre-fusion), in program order. The compute-tile
    /// dataflow executor runs these per-tile (the fused segments collapse the grid,
    /// so they can't be run tile-major).
    nodes: Vec<NodeSpec>,
    /// This program's final result tensor ids (for default readback).
    results: std::collections::HashSet<u64>,
}

pub struct ResidentExecutor {
    /// The programs sharing this executor's resident HBM. One entry for a single
    /// program (`new`); prefill + decode share weights via `new_multi` so the
    /// weight set is uploaded ONCE and both run against the same sticks.
    programs: Vec<ResidentProgram>,
    shapes: HashMap<u64, Vec<usize>>,
    /// The one persistent HBM (and per-core LX). Sticks are allocated once and
    /// reused across every pass.
    mem: SpyreMemoryHierarchy,
    /// tensor id -> its fixed HBM stick. Stable across passes (so the IR's weight
    /// pointers don't move).
    stick: HashMap<u64, i64>,
    /// tensor id -> element count (product of its derived shape).
    numel: HashMap<u64, usize>,
    /// Which tensor ids are program SOURCES (weights / mask / input) — written
    /// once via `set_source`, NOT zeroed per pass. Union across all programs.
    sources: std::collections::HashSet<u64>,
    /// Tensor ids written by some forward NODE (a node output) — MUTABLE across
    /// passes (e.g. the KV-cache prefixes the decode forward grows). A cached GPU
    /// weight buffer keyed on such a tid can go stale, so it is dropped on every
    /// `set_sources` — only tids written by NEITHER the forward NOR the current
    /// `set_sources` (the immutable model weights) stay resident. Read only on the
    /// Metal weight-cache path (`retain_resident_weights`); the off-Metal build has
    /// no GPU weight cache, so the field is unused there.
    #[cfg_attr(not(metal), allow(dead_code))]
    forward_written: std::collections::HashSet<u64>,
    /// The model dtype the per-node oracle threads (F16). All sticks are sized and
    /// read back at this dtype.
    dtype: DType,
    /// Per-segment plan-cache key (`comm_sched::plan_key`), memoized by the ops
    /// slice address. The deep ops-tree hash that keys the scheduler's Metal/
    /// liveness plan caches is otherwise recomputed every forward pass (~7% of a
    /// real decode flamegraph). This executor owns its segments for its whole
    /// lifetime, so each segment's ops slice has a STABLE address and an immutable
    /// structure — keying by address is safe HERE (the cache lives and dies with
    /// the executor, so it never sees another program's freed pointers, which is
    /// what made a process-global pointer cache unsound). Hashed once per segment.
    seg_keys: std::cell::RefCell<HashMap<usize, u64>>,
    /// LAST-TOKEN-ONLY mode. When set, the final result-producing GEMM segment is
    /// rewritten (per program) to compute ONLY output row `m-1` — the only logits
    /// autoregressive generation needs. Default `false` ⇒ every row is computed
    /// (the default golden path is byte-for-byte unchanged). See
    /// [`Self::set_last_token_only`].
    last_token_only: bool,
    /// Per-program LAST-TOKEN rewrite of the result segment, built lazily the first
    /// time `last_token_only` is enabled for a `run_program(idx)`. `[idx]` holds the
    /// rewritten `(segments, result-segment-index)` so the rewrite is done once.
    last_token_segs: std::cell::RefCell<HashMap<usize, Vec<Segment>>>,
    /// Run isolated decode (m=1) attention segments via the fused CPU GEMV/softmax
    /// path instead of the decomposed op storm. Read once at construction (the
    /// planner gates the segment isolation on the same env var, so this only needs
    /// to recognize+compute the now-Native node). Default `true` — the fused path
    /// is a measured win (llama decode ~1.33x, smollm2 ~2.0x) and golden-faithful
    /// (max-abs identical to the decomposed oracle). Set `KTIR_NO_FUSE_ATTN` to
    /// opt out (falls back to the decomposed oracle path).
    fuse_attn: bool,
    /// Keep the multi-core grid SERIAL (no worker-pool parallelism). Set by
    /// [`new_native`](Self::new_native): every node runs at its native grid, so a
    /// non-attention multi-tile kernel dispatches per-core Metal GEMV concurrently —
    /// which the shared Metal device/queue/dispatch-cache cannot do safely. Serial
    /// execution matches the single-threaded per-op GPU diff regime. Default `false`
    /// (the parallel head-grid attention fast path).
    serial_cores: bool,
}

// SAFETY: `ResidentExecutor` owns its ENTIRE object graph exclusively. The
// `IRModule` (with its `Rc<AffineExpr>`s) is moved in and never shared; the
// `SpyreMemoryHierarchy`'s `Rc<RefCell<..>>`s are created and held only here; the
// per-core contexts that clone those `Rc`s during `run()` are created AND dropped
// inside that one call, on the calling thread. No `Rc` clone of any of these
// allocations ever exists outside the executor, so moving the whole executor to
// another thread transfers every `Rc` together — no non-atomic refcount is ever
// touched from two threads at once. We impl `Send` (move between threads) but
// deliberately NOT `Sync`: the executor is internally single-threaded
// (`Rc`/`RefCell`) and must never be shared by `&` across threads. A serving
// worker owns one and calls `run()` serially — exactly this contract. (This is
// why the module is OWNED, not borrowed: a borrowed `&IRModule` shared by two
// executors on two threads could race its `Rc<AffineExpr>` refcounts.)
unsafe impl Send for ResidentExecutor {}

impl ResidentExecutor {
    /// Plan `spec` into segments, derive every tensor's shape from the IR, and
    /// allocate ONE persistent HBM stick per tensor (stable address across
    /// passes). Sources are not yet written — call [`set_source`](Self::set_source)
    /// / [`set_sources`](Self::set_sources) before [`run`](Self::run).
    pub fn new(module: IRModule, spec: &ProgramSpec) -> Result<Self, String> {
        Self::new_multi(vec![(module, spec)])
    }

    /// Build an executor holding MULTIPLE programs that SHARE one resident weight
    /// set — e.g. prefill + decode. Every program's segments bind the SAME HBM
    /// sticks by tensor id, so the weights are uploaded ONCE (one `set_sources`)
    /// and serve every program — no second load for the second program. Sticks are
    /// allocated for the UNION of all programs' tensors; the HBM is sized for the
    /// largest grid any program needs. `run_program(i, ..)` runs program `i` (in
    /// declaration order); `run(..)` runs program 0.
    pub fn new_multi(programs: Vec<(IRModule, &ProgramSpec)>) -> Result<Self, String> {
        Self::build(programs, false)
    }

    /// Like [`new`](Self::new), but plan EVERY node as a [`Segment::Native`] run at
    /// its OWN grid (no fusion to `[1,1]`). This is the faithful way to run a
    /// hand-written SPMD-tiled example kernel — whose multi-tile grid (`[2,16]`
    /// matmul, `[32,1]` softmax/layernorm/vector_add) has each compute-tile write a
    /// DISJOINT output slice selected by `ktdp.get_compute_tile_id`. Collapsed to a
    /// `[1,1]` fused segment only compute-tile 0's slice would be written (the rest
    /// of the output stays its seeded value); run at the native grid every core's
    /// slice is computed. The same resident HBM, weight cache, per-segment seg-plan
    /// (K-loop GEMM reconstruction where recognizable), and per-op Metal offloads
    /// (`metal_gemm_or_blas` / fused map windows) ride along — this is the resident
    /// executor's Native arm, just applied to every node.
    pub fn new_native(module: IRModule, spec: &ProgramSpec) -> Result<Self, String> {
        Self::build(vec![(module, spec)], true)
    }

    fn build(programs: Vec<(IRModule, &ProgramSpec)>, force_native: bool) -> Result<Self, String> {
        if programs.is_empty() {
            return Err("resident: new_multi needs at least one program".into());
        }
        let dtype = DType::F16;
        let bpe = dtype.bytes_per_elem();

        // Union of shapes + sources across programs; per-program planned segments.
        // A stick is allocated for EVERY referenced tensor id so its address is
        // fixed for the executor's lifetime, shared by all programs.
        let mut shapes: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut sources: std::collections::HashSet<u64> = std::collections::HashSet::new();
        // Tensor ids WRITTEN by some forward node (a node output). These are MUTABLE
        // across passes — notably the KV-cache prefixes the decode forward grows each
        // step. A GPU weight buffer cached against such a tid can go stale, so it is
        // never kept resident across a `set_sources` (see `set_sources`).
        let mut forward_written: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut ids: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        let mut planned: Vec<PlannedProgram> = Vec::with_capacity(programs.len());

        for (module, spec) in programs {
            // Optimize at the execution entry (see
            // `crate::segmented::apply_attention_rewrites`): every resident program
            // gets the attention IR rewrites, applied ONCE here before planning.
            let mut module = module;
            crate::segmented::apply_attention_rewrites(&mut module);
            let prog_shapes = derive_shapes(&module, spec)?;
            for (&id, shp) in &prog_shapes {
                shapes.entry(id).or_insert_with(|| shp.clone());
            }
            // Plan segments under the LX live-set budget (per program).
            let tensor_bytes: HashMap<u64, usize> = prog_shapes
                .iter()
                .map(|(&id, shp)| (id, shp.iter().product::<usize>() * bpe))
                .collect();
            let segments = if force_native {
                // Force every node to run at its native grid (no [1,1] fusion):
                // the SPMD-tiled example kernels need every compute-tile's slice.
                spec.nodes
                    .iter()
                    .map(|n| Segment::Native(n.clone()))
                    .collect::<Vec<_>>()
            } else {
                plan_segments_budgeted(
                    &module,
                    spec,
                    crate::memory::lx_fusion_budget(),
                    &tensor_bytes,
                )?
            };
            for seg in &segments {
                match seg {
                    Segment::Fused(fs) => {
                        for (arg, _) in &fs.func.arguments {
                            ids.insert(tensor_id_of_arg(arg)?);
                        }
                    }
                    Segment::Native(node) => {
                        for b in &node.bindings {
                            ids.insert(b.tensor);
                        }
                    }
                }
            }
            // The compute-tile dataflow runs the RAW nodes, which also touch the
            // fusion-INTERNAL intermediates the fused segments hide — allocate a
            // resident stick for every raw-node tensor too.
            for node in &spec.nodes {
                for b in &node.bindings {
                    ids.insert(b.tensor);
                    if b.is_output {
                        forward_written.insert(b.tensor);
                    }
                }
            }
            for &r in &spec.results {
                ids.insert(r);
            }
            sources.extend(spec.sources.iter().copied());
            planned.push((module, segments, spec.results.clone(), spec.nodes.clone()));
        }

        // Size the HBM (LX-per-core array) for the largest grid across ALL programs.
        let grid = planned
            .iter()
            .map(|(m, segs, _, _)| largest_grid(m, segs))
            .max()
            .unwrap_or(1);
        let mem = SpyreMemoryHierarchy::new(grid);

        let mut stick: HashMap<u64, i64> = HashMap::new();
        let mut numel: HashMap<u64, usize> = HashMap::new();
        {
            let hbm = mem.hbm.borrow_mut();
            for &tid in &ids {
                let shape = shapes
                    .get(&tid)
                    .cloned()
                    .ok_or_else(|| format!("no shape derivable for tensor t{tid}"))?;
                let n: usize = shape.iter().product();
                let s = hbm.allocate((n * bpe).max(bpe) as i64);
                stick.insert(tid, s);
                numel.insert(tid, n);
            }
        }

        let programs = planned
            .into_iter()
            .map(|(module, segments, results, nodes)| ResidentProgram {
                module,
                segments,
                nodes,
                results,
            })
            .collect();

        Ok(ResidentExecutor {
            programs,
            shapes,
            mem,
            stick,
            numel,
            sources,
            forward_written,
            dtype,
            seg_keys: std::cell::RefCell::new(HashMap::new()),
            last_token_only: false,
            last_token_segs: std::cell::RefCell::new(HashMap::new()),
            // Default ON (opt out via KTIR_NO_FUSE_ATTN). KTIR_FORCE_FUSE_ATTN
            // forces it ON even when KTIR_NO_FUSE_ATTN is set — the conformance
            // harness uses it to guarantee the decode (m=1) attention island takes
            // the fused GEMV/softmax/GEMV Metal path (proven by gemm_or_blas_gpu>0)
            // rather than the decomposed oracle. (The non-decode example attention
            // programs — sdpa m=32, paged_attention m=8 — are not recognized as the
            // decode island and run the decomposed grid, which fires the same
            // per-op gemm_or_blas_gpu offload anyway.)
            fuse_attn: std::env::var_os("KTIR_FORCE_FUSE_ATTN").is_some()
                || std::env::var_os("KTIR_NO_FUSE_ATTN").is_none(),
            // Native-grid example runs keep the grid serial (concurrent per-core
            // Metal GEMV is not thread-safe); the fused/attention path stays parallel.
            serial_cores: force_native,
        })
    }

    /// FUSED CPU m=1 ATTENTION. Resolve each island arg to its resident HBM stick
    /// (via `node.bindings`: arg name → tensor id → stick → byte addr), decode the
    /// f16 inputs to f32, run [`DecodeAttnIsland::compute_f32`], and write the f16
    /// output row back to the O tensor's stick. Reproduces the decomposed path's
    /// exact arithmetic per head with f32 accumulation (golden-faithful), at ~3·H
    /// primitives instead of the ~1500-op decomposed storm.
    fn run_fused_decode_attention(
        &self,
        node: &NodeSpec,
        island: &ktir_optimizer::head_rewrite::DecodeAttnIsland,
    ) -> Result<(), String> {
        // arg name (e.g. "%t339_ptr") -> tensor id, from the node bindings.
        let arg_tid: HashMap<&str, u64> = node
            .bindings
            .iter()
            .map(|b| (b.arg.as_str(), b.tensor))
            .collect();
        let read_arg = |arg: &str, n: usize| -> Result<Vec<f32>, String> {
            let tid = *arg_tid
                .get(arg)
                .ok_or_else(|| format!("fused attn: arg {arg} not bound"))?;
            let s = *self
                .stick
                .get(&tid)
                .ok_or_else(|| format!("fused attn: tensor t{tid} has no resident stick"))?;
            let nbytes = n * self.dtype.bytes_per_elem();
            let bytes = self.mem.hbm.borrow().read_bytes(s * STICK_BYTES, nbytes);
            Ok(crate::codec::decode(&bytes, n, self.dtype))
        };

        let q_cols = island.q_cols as usize;
        let cap = island.cap as usize;
        let kv_cols = island.kv_cols as usize;
        let q = read_arg(&island.q_arg, q_cols)?;
        let mask = read_arg(&island.mask_arg, cap)?;
        let kc = read_arg(&island.kc_arg, cap * kv_cols)?;
        let kd = read_arg(&island.kd_arg, kv_cols)?;
        let vc = read_arg(&island.vc_arg, cap * kv_cols)?;
        let vd = read_arg(&island.vd_arg, kv_cols)?;

        let mut o = vec![0.0f32; q_cols];
        island.compute_f32(&q, &mask, &kc, &kd, &vc, &vd, &mut o);

        // Write the output row back to the O tensor's stick as f16.
        let o_tid = *arg_tid
            .get(island.o_arg.as_str())
            .ok_or_else(|| format!("fused attn: output arg {} not bound", island.o_arg))?;
        let o_stick = *self
            .stick
            .get(&o_tid)
            .ok_or_else(|| format!("fused attn: output t{o_tid} has no resident stick"))?;
        let obytes = crate::codec::encode(&o, self.dtype);
        self.mem
            .hbm
            .borrow_mut()
            .write_bytes(o_stick * STICK_BYTES, &obytes);
        Ok(())
    }

    /// Enable/disable LAST-TOKEN-ONLY mode. When ON, the final result-producing
    /// projection (the segment whose store writes a program RESULT tensor — e.g. a
    /// transformer's `lm_head`) computes ONLY the last output row (`m-1`) instead
    /// of all `m` rows. For autoregressive generation only that row's logits pick
    /// the next token, so the other `m-1` rows are pure waste; on Llama-1B prefill
    /// (m=32, vocab=128256) this turns the ~31 ms [32,128256] GEMM into a ~1 ms
    /// [1,128256] one. The result tensor is still read back at its full
    /// `[m, vocab]` shape — only row `m-1` is populated (the rest stay zeroed),
    /// and that row equals the all-rows path's last row (it reads the SAME last
    /// activation row through the SAME weight). Default OFF ⇒ identical to today.
    ///
    /// Structural (NOT model-specific): keys off the result-producing GEMM segment
    /// and the general "only the last output row is needed" contract.
    pub fn set_last_token_only(&mut self, on: bool) {
        self.last_token_only = on;
    }

    /// The scheduler's plan-cache key for `ops`, memoized by slice address (see
    /// [`Self::seg_keys`]). Computes the deep hash on first sight of a segment,
    /// then returns it directly on every subsequent forward pass.
    fn seg_plan_key(&self, ops: &[Operation]) -> u64 {
        let ptr = ops.as_ptr() as usize;
        if let Some(&k) = self.seg_keys.borrow().get(&ptr) {
            return k;
        }
        let k = crate::comm_sched::plan_key(ops);
        self.seg_keys.borrow_mut().insert(ptr, k);
        k
    }

    /// Write one SOURCE tensor's f32 data into its resident HBM stick ONCE
    /// (encoded to the model dtype). Call this for every weight / the mask / the
    /// input before the first [`run`](Self::run). The bytes persist for the
    /// executor's lifetime — the per-pass loop never rewrites a source unless you
    /// explicitly do so (e.g. a changing decode input via [`set_input`](Self::set_input)).
    pub fn set_source(&mut self, tensor: u64, data: &[f32]) -> Result<(), String> {
        let s = *self
            .stick
            .get(&tensor)
            .ok_or_else(|| format!("set_source: t{tensor} is not a tensor this program uses"))?;
        let bytes = crate::codec::encode(data, self.dtype);
        self.mem
            .hbm
            .borrow_mut()
            .write_bytes(s * STICK_BYTES, &bytes);
        Ok(())
    }

    /// Write one SOURCE tensor's already-typed bytes straight into its resident
    /// HBM stick ONCE — the f32-free fast path. When `dtype` matches the model
    /// dtype (the stick layout), the bytes are `write_bytes`-copied verbatim: no
    /// `Vec<f32>`, no decode, no encode (mirrors Spyre's typed host→AIU DMA). This
    /// is what a memory-mapped f16 safetensor wants — hand it the tensor's byte
    /// slice and it lands in HBM with one copy. A mismatched `dtype` (e.g. an f32
    /// host buffer, or a future widened source) falls back through f32: decode to
    /// f32, re-encode to the stick dtype.
    pub fn set_source_bytes(
        &mut self,
        tensor: u64,
        bytes: &[u8],
        dtype: DType,
    ) -> Result<(), String> {
        let s = *self.stick.get(&tensor).ok_or_else(|| {
            format!("set_source_bytes: t{tensor} is not a tensor this program uses")
        })?;
        if dtype == self.dtype {
            // Verbatim: typed bytes already match the stick layout.
            self.mem
                .hbm
                .borrow_mut()
                .write_bytes(s * STICK_BYTES, bytes);
            Ok(())
        } else {
            // dtype crossing (e.g. f32 bytes into an f16 stick): go through f32.
            let n = self
                .numel
                .get(&tensor)
                .copied()
                .unwrap_or(bytes.len() / dtype.bytes_per_elem());
            let data = crate::codec::decode(bytes, n, dtype);
            self.set_source(tensor, &data)
        }
    }

    /// Write many sources at once (keyed by the canonical `t<id>` / `%t<id>` /
    /// `%t<id>_ptr` / bare `<id>` name). Unknown keys (a tensor this program does
    /// not reference) are skipped — the caller can hand the whole weight set.
    ///
    /// [`Arg::TensorBytes`] takes the f32-free byte path ([`set_source_bytes`]) —
    /// for an all-f16 model the weights land in HBM with a single copy each, never
    /// touching `Vec<f32>`. [`Arg::Tensor`] (host f32) still narrows on the way in.
    pub fn set_sources(&mut self, args: &[(&str, Arg)]) -> Result<(), String> {
        #[cfg(metal)]
        let mut just_set: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for (key, arg) in args {
            let tid = tensor_id_of_key(key)?;
            if !self.stick.contains_key(&tid) {
                continue;
            }
            #[cfg(metal)]
            just_set.insert(tid);
            match arg {
                Arg::TensorBytes { data, dtype, .. } => {
                    self.set_source_bytes(tid, data, *dtype)?;
                }
                Arg::Tensor { data, .. } => {
                    // Host f32: narrowed to the stick dtype on the way in.
                    self.set_source(tid, data)?;
                }
                Arg::TensorBf16 { data, shape } => {
                    // bf16 host bytes -> f16 stick layout in ONE fused pass (no f32
                    // intermediate), written straight into HBM like the f16 path.
                    let f16 = crate::codec::bf16_to_f16(data, shape.iter().product());
                    self.set_source_bytes(tid, &f16, DType::F16)?;
                }
                Arg::Scalar(_) => return Err("resident: scalar args unsupported".into()),
            }
        }
        // Keep ONLY the provably-immutable model weights resident; drop every cached
        // buffer whose tid the forward pass writes (KV cache) or that this call just
        // re-set. Both cover every way an HBM tensor's bytes change, so a kept buffer
        // can never be stale — the decode loop no longer re-decodes the constant ~2 GB
        // of weights every token, while the KV cache is correctly re-read each step.
        #[cfg(metal)]
        crate::metal::retain_resident_weights(&self.forward_written, &just_set);
        Ok(())
    }

    /// Rewrite a changing INPUT activation in place (decode threads a new token
    /// each pass). Identical to [`set_source`](Self::set_source) but named for the
    /// per-pass intent. The stick is unchanged, so the resident weights are
    /// untouched.
    pub fn set_input(&mut self, tensor: u64, data: &[f32]) -> Result<(), String> {
        self.set_source(tensor, data)
    }

    /// Run ONE forward pass against the resident HBM and read back `outputs`
    /// (canonical `t<id>` keys; empty = the program's declared result tensors).
    ///
    /// Zeroes every NON-source stick first (results / intermediates / scratch) so
    /// a pass never reads a stale value from the previous pass, then runs each
    /// segment in order: a fused segment at grid `[1,1]` (carrying the GPU K-loop
    /// GEMM / map-window / resident-weight-cache offloads), a native attention
    /// node at its native head-parallel grid — both against the SAME persistent
    /// HBM, so intermediates flow segment-to-segment with no marshal.
    pub fn run(&mut self, outputs: &[&str]) -> Result<HashMap<String, Output>, String> {
        self.run_program(0, outputs)
    }

    /// Run program `idx` (declaration order in [`new_multi`]) for one forward pass.
    /// All programs share the resident weights, so switching between prefill and
    /// decode re-uploads NOTHING — only the per-pass input/mask change via
    /// [`set_sources`](Self::set_sources) / [`set_input`](Self::set_input).
    pub fn run_program(
        &mut self,
        idx: usize,
        outputs: &[&str],
    ) -> Result<HashMap<String, Output>, String> {
        if idx >= self.programs.len() {
            return Err(format!(
                "resident: program {idx} out of range (have {})",
                self.programs.len()
            ));
        }
        self.zero_non_sources();

        // Resident weights are uploaded once (and the weight cache is cleared on
        // every `set_sources`), so they're immutable for the run — let the GPU
        // weight cache skip its per-pass content fingerprint. Restored on return.
        #[cfg(metal)]
        let _trust = TrustedWeightsGuard(crate::metal::set_trusted_weights(true));

        // Every stick is pre-allocated at construction, so the multi-core attention
        // cores only write disjoint pre-existing sticks — no concurrent allocation,
        // so the worker-pool parallel grid is sound here (unlike `execute_function`'s
        // fresh-HBM lazy-allocation path). Enables the ~3.5x head-parallel attention.
        //
        // EXCEPT in `serial_cores` mode (the native-grid example runner, see
        // `new_native`): there EVERY node runs at its native grid, and a non-attention
        // multi-tile kernel's per-core `linalg.matmul`/GEMV tiles each dispatch to the
        // Metal engine — concurrently across worker-pool threads. The shared Metal
        // device/queue/dispatch-cache is NOT thread-safe under that concurrent GEMV
        // dispatch (intermittent Bus/Trap/Abort), so those runs keep the grid SERIAL
        // (the same single-threaded regime the per-op GPU diff path already uses). The
        // attention fast path, which set this true, is unaffected (it stays parallel).
        let _par = ParallelSafeGuard(crate::comm_sched::set_parallel_safe(!self.serial_cores));

        // KTIR_SEG_DIAG: accumulate fused (GPU GEMM/map) vs native (CPU-interpreter
        // attention) wall-time per pass — to see how much of e2e is the attention
        // islands still on the interpreter.
        let diag = std::env::var_os("KTIR_SEG_DIAG").is_some();
        // Compute-tile dataflow (KTIR_TILE_DATAFLOW): run the raw grid=[H,1] nodes
        // TILE-major (each token-row streams its whole per-row chain, syncing only
        // at attention), bypassing the GPU-fused segments. Default: the segment loop.
        let tile_dataflow = std::env::var_os("KTIR_TILE_DATAFLOW").is_some();
        if tile_dataflow {
            self.run_compute_tile_dataflow(idx)?;
        }
        let (mut t_fused, mut t_native, mut n_fused, mut n_native) =
            (0.0f64, 0.0f64, 0usize, 0usize);
        // KTIR_SEG_PROF: per-segment wall-time + dominant GEMM shape (weights already
        // resident — no upload artifact), to split true compute from per-dispatch
        // overhead. Each entry: (label, ms, n_matmuls).
        let seg_prof = std::env::var_os("KTIR_SEG_PROF").is_some();
        let mut prof_rows: Vec<(String, f64, usize)> = Vec::new();
        // LAST-TOKEN-ONLY: run a rewritten segment list whose final result GEMM
        // computes only row m-1 (built once per program, then cached). Default
        // mode runs the program's unmodified segments.
        if self.last_token_only && !self.last_token_segs.borrow().contains_key(&idx) {
            let rewritten = self.build_last_token_segments(idx)?;
            self.last_token_segs.borrow_mut().insert(idx, rewritten);
        }
        let lt = self.last_token_only;
        let lt_borrow = self.last_token_segs.borrow();
        let segments: &[Segment] = if lt {
            lt_borrow.get(&idx).unwrap()
        } else {
            &self.programs[idx].segments
        };
        for (seg_i, seg) in segments.iter().enumerate() {
            if tile_dataflow {
                break;
            }
            // Reset every core's LX scratchpad before each segment run. The
            // persistent `mem` reuses the SAME LX across segments/passes, but each
            // function run is a self-contained SPMD execution that bump-allocates
            // LX from empty (and whose `used` watermark must start at 0). Without
            // this, `used` accumulates the live-out tracking of every prior
            // segment and eventually trips the LX capacity guard. (A fresh-`mem`
            // `execute_function` got this for free; the resident `mem` must do it
            // explicitly.) HBM is NOT cleared — that's the resident weight set.
            for lx in &self.mem.lx_scratchpads {
                lx.borrow_mut().clear();
            }
            let seg_t0 = std::time::Instant::now();
            match seg {
                Segment::Fused(fs) => {
                    // Bind every pointer arg to its resident stick, and collect
                    // the boundary OUTPUTs to read back (so intermediates flow via
                    // HBM, not host).
                    let mut input_ptrs: Vec<(String, Value)> =
                        Vec::with_capacity(fs.func.arguments.len());
                    for (arg_name, _) in &fs.func.arguments {
                        let tid = tensor_id_of_arg(arg_name)?;
                        let bare = arg_name.trim_start_matches('%').to_string();
                        let s = *self
                            .stick
                            .get(&tid)
                            .ok_or_else(|| format!("fused arg t{tid} has no resident stick"))?;
                        // base_ptr is an ELEMENT index (RFC #110): the view's
                        // byte_address = base_ptr*bpe must land on the resident
                        // stick (byte s*STICK_BYTES), so bind elem = s*STICK_BYTES/bpe.
                        let elem = s * STICK_BYTES / self.dtype.bytes_per_elem() as i64;
                        input_ptrs.push((bare, Value::Index(elem)));
                    }
                    // Run against the persistent HBM, NO per-segment read-back: the
                    // outputs are already resident in HBM for the next segment, and
                    // the single final read-back below decodes the requested
                    // results. Decoding every segment's outputs here was discarded
                    // work (`let _ =`).
                    let key = self.seg_plan_key(&fs.func.operations);
                    execute_function_in_exec_only(
                        &self.mem,
                        &fs.func.operations,
                        (1, 1, 1),
                        &input_ptrs,
                        Some(key),
                    )?;
                }
                Segment::Native(node) => {
                    let func = self.programs[idx].module.get_function(&node.func)?;
                    // FUSED CPU m=1 ATTENTION (default ON; opt out via
                    // KTIR_NO_FUSE_ATTN): when the native segment recognizes as the
                    // decode (m=1) attention island, compute it directly with
                    // BLAS-style GEMV + softmax + GEMV per head against resident HBM
                    // — collapsing the ~1500-op decomposed storm to ~3·H primitives.
                    // Default-on: the planner isolates this node into a Native
                    // segment unless KTIR_NO_FUSE_ATTN is set (then it stays folded
                    // in a Fused segment, decomposed — the oracle path), so
                    // recognition here is the steady-state path for an isolated m=1
                    // attention node. If recognition fails (a non-decode native
                    // node), fall through to the decomposed grid run below.
                    let fused = if self.fuse_attn {
                        if let Some(island) =
                            ktir_optimizer::head_rewrite::recognize_head_attention_decode(func)
                        {
                            self.run_fused_decode_attention(node, &island)?;
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if fused {
                        // computed in fused path; outputs already resident in HBM.
                    } else {
                        let grid = func.grid;
                        let mut input_ptrs: Vec<(String, Value)> =
                            Vec::with_capacity(node.bindings.len());
                        for b in &node.bindings {
                            let name = b.arg.trim_start_matches('%').to_string();
                            let s = *self.stick.get(&b.tensor).ok_or_else(|| {
                                format!("native attn arg t{} has no resident stick", b.tensor)
                            })?;
                            // base_ptr is an ELEMENT index (RFC #110): bind
                            // elem = s*STICK_BYTES/bpe so byte_address lands on stick s.
                            let elem = s * STICK_BYTES / self.dtype.bytes_per_elem() as i64;
                            input_ptrs.push((name, Value::Index(elem)));
                        }
                        // No per-segment read-back (outputs flow via HBM; see the
                        // Fused arm) — the final read-back below decodes the results.
                        let key = self.seg_plan_key(&func.operations);
                        execute_function_in_exec_only(
                            &self.mem,
                            &func.operations,
                            grid,
                            &input_ptrs,
                            Some(key),
                        )?;
                    }
                }
            }
            if diag || seg_prof {
                let dt = seg_t0.elapsed().as_secs_f64() * 1e3;
                match seg {
                    Segment::Fused(fs) => {
                        t_fused += dt;
                        n_fused += 1;
                        if seg_prof {
                            let (mkn, nmm) = dominant_gemm(&fs.func.operations);
                            let extra = if nmm == 0 {
                                format!("  ops:[{}]", op_type_histogram(&fs.func.operations))
                            } else {
                                String::new()
                            };
                            prof_rows.push((format!("seg{seg_i:>3} fused {mkn}{extra}"), dt, nmm));
                        }
                    }
                    Segment::Native(_) => {
                        t_native += dt;
                        n_native += 1;
                        if seg_prof {
                            prof_rows.push((format!("seg{seg_i:>3} native attn"), dt, 0));
                        }
                    }
                }
            }
        }
        if diag {
            eprintln!(
                "  [resident-seg-diag] {n_fused} fused {t_fused:.1}ms (GPU GEMM/map) | \
                 {n_native} native {t_native:.1}ms (CPU-interp attention)"
            );
        }
        if seg_prof {
            prof_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let total: f64 = prof_rows.iter().map(|r| r.1).sum();
            let floor = 0.30; // measured fixed per-dispatch floor (gpu_dispatch_floor)
            let est_overhead: f64 = prof_rows.iter().map(|r| floor * r.2.max(1) as f64).sum();
            eprintln!(
                "  [resident-seg-prof] {} segs, {total:.1}ms total; est fixed dispatch \
                 overhead ≈ {est_overhead:.1}ms ({:.0}%) at {floor}ms/GEMM × ΣGEMMs",
                prof_rows.len(),
                100.0 * est_overhead / total
            );
            for (label, ms, nmm) in prof_rows.iter().take(20) {
                eprintln!("    {label:<28} {ms:6.3} ms  ({nmm} matmul)");
            }
            // Weight-cache hit/miss tally: proves resident weights (incl. the lm_head
            // N-tiles) are decoded+uploaded ONCE and then served from cache (0 misses
            // on the 2nd+ pass). A nonzero steady-state miss count would flag a weight
            // re-upload regression — the cost candidate #1 was meant to eliminate.
            #[cfg(metal)]
            eprintln!(
                "    [weight-cache] hits={} misses={} (steady-state misses should be 0)",
                crate::metal::WEIGHT_CACHE_HITS.swap(0, std::sync::atomic::Ordering::Relaxed),
                crate::metal::WEIGHT_CACHE_MISSES.swap(0, std::sync::atomic::Ordering::Relaxed),
            );
        }

        // Read back the requested outputs (default: the program results) from the
        // resident HBM.
        let want: Vec<u64> = if outputs.is_empty() {
            let mut v: Vec<u64> = self.programs[idx].results.iter().copied().collect();
            v.sort_unstable();
            v
        } else {
            outputs
                .iter()
                .map(|k| tensor_id_of_key(k))
                .collect::<Result<_, _>>()?
        };
        let read: Vec<TensorMeta> = want
            .iter()
            .map(|&tid| self.meta_for(tid))
            .collect::<Result<_, _>>()?;
        // One readback pass (decode the wanted sticks to host f32).
        let mut result = HashMap::new();
        for (name, stick, n, shape, dtype) in read {
            let nbytes = n * dtype.bytes_per_elem();
            let bytes = self
                .mem
                .hbm
                .borrow()
                .read_bytes(stick * STICK_BYTES, nbytes);
            let data = crate::codec::decode(&bytes, n, dtype);
            result.insert(
                name,
                Output {
                    data,
                    shape,
                    dtype,
                    raw: bytes,
                },
            );
        }
        Ok(result)
    }

    /// COMPUTE-TILE DATAFLOW — the general grid-parallel executor. The raw nodes
    /// are grid=[H,1]: H independent compute-tiles. Run them TILE-major instead of
    /// node-major — each tile streams through its whole per-row chain with NO
    /// per-node barrier — syncing only at attention, where a tile reads ALL tiles'
    /// K/V. So the program splits into phases at each attention node (the only
    /// cross-tile dependency); within a phase every tile runs every node. Bypasses
    /// the GPU-fused segments: each tile's per-row work is a CPU/AMX GEMV, not a
    /// batched GPU dispatch. Single-threaded; the worker pool runs tiles concurrently.
    fn run_compute_tile_dataflow(&self, idx: usize) -> Result<(), String> {
        let prog = &self.programs[idx];
        let module = &prog.module;
        let dispatch = crate::dialects::Dispatch::shared();
        // Per-node compute-tile partition dims, and the dim each cross-node tensor
        // is WRITTEN on by its producer. A producer→consumer edge is tile-major-safe
        // only if both partition the shared tensor on the SAME dim AND a width-1
        // (own-tile) slice — i.e. consumer tile k reads exactly what producer tile k
        // wrote. Otherwise it's a re-tiling (or full-axis) barrier.
        let pdims: Vec<ktir_optimizer::fusion::PartitionDims> = prog
            .nodes
            .iter()
            .map(|n| {
                let f = module.get_function(&n.func)?;
                Ok(ktir_optimizer::fusion::node_partition_dims(f, n))
            })
            .collect::<Result<_, String>>()?;
        let mut write_dim: std::collections::HashMap<u64, Option<usize>> =
            std::collections::HashMap::new();
        for pd in &pdims {
            for &(t, d) in &pd.writes {
                write_dim.insert(t, d);
            }
        }
        // Phases: a node is a BARRIER when, for some cross-node tensor it reads, its
        // partition dim differs from the producer's write dim (a re-tiling, e.g.
        // attention writes head-tiled then o-proj reads token-tiled), or it reads
        // the tensor full-axis (None). A barrier runs in its OWN phase: every tile
        // finishes the prior phase (so the re-tiled input is fully materialized),
        // all tiles run the barrier together, then later nodes proceed.
        let is_barrier = |i: usize| -> bool {
            let g = module.get_function(&prog.nodes[i].func).map(|f| f.grid);
            if let Ok((gx, gy, gz)) = g
                && gx * gy * gz <= 1
            {
                return false;
            }
            pdims[i].reads.iter().any(|&(t, rd)| {
                // Only inter-node edges matter; a tensor no node writes is a source.
                match write_dim.get(&t) {
                    Some(&wd) => rd != wd, // re-tile (or full-axis read) ⇒ barrier
                    None => false,
                }
            })
        };
        let mut phases: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        for i in 0..prog.nodes.len() {
            if is_barrier(i) {
                if !cur.is_empty() {
                    phases.push(std::mem::take(&mut cur)); // close before
                }
                phases.push(vec![i]); // the barrier runs alone
            } else {
                cur.push(i);
            }
        }
        if !cur.is_empty() {
            phases.push(cur);
        }
        // Bisection harness: cap phase length to N extra barriers (KTIR_TILE_MAXPHASE).
        // N=1 ⇒ every phase is one node ⇒ tile-major degenerates to node-major. The
        // largest N that still passes golden localizes the missed cross-tile coupler.
        if let Some(n) = std::env::var("KTIR_TILE_MAXPHASE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            let n = n.max(1);
            let mut capped: Vec<Vec<usize>> = Vec::new();
            for ph in phases {
                for chunk in ph.chunks(n) {
                    capped.push(chunk.to_vec());
                }
            }
            phases = capped;
        }
        // Surgical bisection: run EVERYTHING node-major (singleton phases) except a
        // single adjacent pair (P, P+1) fused into one tile-major phase. Scanning P
        // pinpoints the first pair whose tile-major execution diverges.
        if let Some(p) = std::env::var("KTIR_TILE_PAIR")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            let flat: Vec<usize> = phases.into_iter().flatten().collect();
            let mut rebuilt: Vec<Vec<usize>> = Vec::new();
            let mut i = 0;
            while i < flat.len() {
                if flat[i] == p && i + 1 < flat.len() {
                    rebuilt.push(vec![flat[i], flat[i + 1]]);
                    i += 2;
                } else {
                    rebuilt.push(vec![flat[i]]);
                    i += 1;
                }
            }
            phases = rebuilt;
        }
        if std::env::var_os("KTIR_TILE_DIAG").is_some() {
            let barriers: Vec<usize> = (0..prog.nodes.len()).filter(|&i| is_barrier(i)).collect();
            eprintln!(
                "[tile-diag] {} nodes, {} phases, {} barrier nodes: {:?}",
                prog.nodes.len(),
                phases.len(),
                barriers.len(),
                barriers
            );
        }
        // NODE-MAJOR SEQUENTIAL mode (KTIR_TILE_SEQ): for each node, run every tile
        // before moving to the next node — i.e. the SAME order the grid executor uses,
        // just driven per-tile through the single-tile interpreter. This isolates the
        // per-tile interpreter path from the tile-major reorder: if this passes golden
        // but tile-major doesn't, the reorder/phases are the bug, not the per-tile path.
        let node_major = std::env::var_os("KTIR_TILE_SEQ").is_some();
        for phase in &phases {
            let mut num_tiles = 1usize;
            for &ni in phase {
                let g = module.get_function(&prog.nodes[ni].func)?.grid;
                num_tiles = num_tiles.max(g.0 * g.1 * g.2);
            }
            if node_major {
                // Node-major: complete each node across all tiles before the next node.
                for &ni in phase {
                    for tile in 0..num_tiles {
                        self.run_one_node_tile(idx, ni, tile, dispatch)?;
                    }
                }
            } else {
                // Tile-major: each tile runs the phase's whole node chain independently.
                for tile in 0..num_tiles {
                    for &ni in phase {
                        self.run_one_node_tile(idx, ni, tile, dispatch)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Run a single (node, compute-tile) pair through the single-tile interpreter.
    /// Skips tiles outside the node's grid. Shared by the tile-major and node-major
    /// dataflow drivers so both exercise the identical per-tile execution path.
    fn run_one_node_tile(
        &self,
        idx: usize,
        ni: usize,
        tile: usize,
        dispatch: &crate::dialects::Dispatch,
    ) -> Result<(), String> {
        let prog = &self.programs[idx];
        let module = &prog.module;
        let node = &prog.nodes[ni];
        let func = module.get_function(&node.func)?;
        let g = func.grid;
        if tile >= (g.0 * g.1 * g.2) {
            return Ok(());
        }
        let mut input_ptrs: Vec<(String, Value)> = Vec::with_capacity(node.bindings.len());
        for b in &node.bindings {
            let name = b.arg.trim_start_matches('%').to_string();
            let s = *self
                .stick
                .get(&b.tensor)
                .ok_or_else(|| format!("tile-dataflow: t{} has no resident stick", b.tensor))?;
            // base_ptr is an ELEMENT index (RFC #110): bind elem = s*STICK_BYTES/bpe.
            let elem = s * STICK_BYTES / self.dtype.bytes_per_elem() as i64;
            input_ptrs.push((name, Value::Index(elem)));
        }
        self.mem.get_lx(tile).borrow_mut().clear();
        let grid = crate::env::GridExecutor::new(g);
        let key = self.seg_plan_key(&func.operations);
        crate::comm_sched::execute_function_single_tile(
            &grid,
            &self.mem,
            &func.operations,
            &input_ptrs,
            dispatch,
            tile,
            Some(key),
        )?;
        Ok(())
    }

    /// Zero every NON-source stick (results, intermediates, scratch) so a pass
    /// starts clean. Sources (weights / mask / input) keep their resident bytes.
    /// Cheap: these are activations (kB) not weights (GB).
    fn zero_non_sources(&mut self) {
        let hbm = self.mem.hbm.borrow_mut();
        let bpe = self.dtype.bytes_per_elem();
        for (&tid, &s) in &self.stick {
            if self.sources.contains(&tid) {
                continue;
            }
            let n = *self.numel.get(&tid).unwrap_or(&0);
            if n == 0 {
                continue;
            }
            // Zero the existing backing bytes IN PLACE — every non-source stick is
            // already allocated (the stick base is its exact allocation base), so
            // fill its bytes rather than allocating a fresh zero `Vec` +
            // `copy_from_slice` per stick every pass.
            if let Some((buf, off)) = hbm.allocation_at_mut(s * STICK_BYTES) {
                let end = (off + n * bpe).min(buf.len());
                buf[off..end].fill(0);
            } else {
                hbm.write_bytes(s * STICK_BYTES, &vec![0u8; n * bpe]);
            }
        }
    }

    /// Build the LAST-TOKEN-ONLY segment list for program `idx`: clone its
    /// segments and rewrite the FINAL fused segment that writes a program RESULT
    /// tensor so its result GEMM computes only output row `m-1`.
    ///
    /// The rewrite (in `rewrite_last_token_func`) keys off structure: it finds the
    /// result view (root binds a result tensor) to recover the token count `m`,
    /// pins every result-store and matching-`m` activation access tile's leading
    /// (row) index to the constant `m-1`, and shrinks the activation view's first
    /// dim to 1. The GPU GEMM recognizer then reconstructs a single-row
    /// `[1,k]@[k,n]` GEMM reading activation row `m-1`, writing result row `m-1`.
    fn build_last_token_segments(&self, idx: usize) -> Result<Vec<Segment>, String> {
        let results = &self.programs[idx].results;
        let mut segs = self.programs[idx].segments.clone();
        // The LAST fused segment writing a result tensor is the final projection.
        let target = segs.iter().enumerate().rev().find_map(|(i, s)| match s {
            Segment::Fused(fs) if fs.outputs.iter().any(|t| results.contains(t)) => Some(i),
            _ => None,
        });
        let Some(ti) = target else {
            return Err(
                "last-token-only: no fused segment writes a program result (cannot isolate \
                 the final projection)"
                    .into(),
            );
        };
        if let Segment::Fused(fs) = &mut segs[ti] {
            // Map pointer-arg name -> tensor id, to spot the result view's root.
            let arg_to_tensor: HashMap<String, u64> = fs
                .func
                .arguments
                .iter()
                .map(|(a, _)| Ok((a.trim_start_matches('%').to_string(), tensor_id_of_arg(a)?)))
                .collect::<Result<_, String>>()?;
            rewrite_last_token_func(&mut fs.func.operations, &arg_to_tensor, results)?;
        }
        Ok(segs)
    }

    /// Build the `(name, stick, numel, shape, dtype)` readback tuple for a tensor,
    /// keyed by the canonical `t<id>` name.
    fn meta_for(&self, tid: u64) -> Result<TensorMeta, String> {
        let s = *self
            .stick
            .get(&tid)
            .ok_or_else(|| format!("t{tid} has no resident stick"))?;
        let n = *self.numel.get(&tid).unwrap_or(&0);
        let shape = self.shapes.get(&tid).cloned().unwrap_or_else(|| vec![n]);
        Ok((format!("t{tid}"), s, n, shape, self.dtype))
    }
}

/// LAST-TOKEN-ONLY rewrite of one fused result function: make the final
/// projection compute only output row `m-1`. Pure IR rewrite, structurally keyed:
///
///  1. Find the RESULT view (a `construct_memory_view` whose pointer root binds a
///     program result tensor). Its first dim is the token count `m`.
///  2. Find ACTIVATION views: 2-D input views with first dim == `m` that are NOT a
///     result view and NOT a weight (weights have a first dim ≫ m — the vocab/
///     hidden axis). Shrink each activation view's first dim to 1 (so the GPU GEMM
///     recognizer reconstructs a single-row A).
///  3. Pin every access tile built on a result view or a shrunk activation view to
///     read/write ROW `m-1`: replace its leading (row) index operand with a fresh
///     `arith.constant m-1 : index`. The recognizer then reads the activation's
///     row `m-1` (`m_row_off`) and the store lands the [1,n] result in result row
///     `m-1`. Every other row stays zeroed (the per-pass zero-init) — exactly the
///     all-rows path sliced to its last row.
///
/// Returns Err if no result view (or `m <= 1`, where the rewrite is a no-op).
fn rewrite_last_token_func(
    ops: &mut Vec<Operation>,
    arg_to_tensor: &HashMap<String, u64>,
    results: &std::collections::HashSet<u64>,
) -> Result<(), String> {
    // Pass 1: classify memory views by their SSA result name.
    // result_views: views whose root binds a result tensor; act_views: activation
    // views to shrink (first dim == m). Recover m from the result view.
    let mut m: Option<usize> = None;
    // Collect (view_ssa, is_result) and the activation candidates' first dims.
    let mut result_views: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut view_shape: HashMap<String, Vec<usize>> = HashMap::new();
    fn scan_views(
        ops: &[Operation],
        arg_to_tensor: &HashMap<String, u64>,
        results: &std::collections::HashSet<u64>,
        m: &mut Option<usize>,
        result_views: &mut std::collections::HashSet<String>,
        view_shape: &mut HashMap<String, Vec<usize>>,
    ) {
        for op in ops {
            if op.op_type == "ktdp.construct_memory_view"
                && let (Some(res), Some(root)) = (op.result.as_deref(), op.operands.first())
                && let Some(Attr::IntList(shape)) = op.attributes.get("shape")
                && shape.len() == 2
            {
                let shp: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let root_bare = root.trim_start_matches('%');
                view_shape.insert(res.to_string(), shp.clone());
                if let Some(&tid) = arg_to_tensor.get(root_bare)
                    && results.contains(&tid)
                {
                    result_views.insert(res.to_string());
                    *m = Some(shp[0]);
                }
            }
            for rg in &op.regions {
                scan_views(rg, arg_to_tensor, results, m, result_views, view_shape);
            }
        }
    }
    scan_views(
        ops,
        arg_to_tensor,
        results,
        &mut m,
        &mut result_views,
        &mut view_shape,
    );
    let Some(m) = m else {
        return Err("last-token-only: no result memory view in the final segment".into());
    };
    if m <= 1 {
        return Ok(()); // single row already — nothing to prune
    }
    // Activation views: first dim == m, not a result view (weights have first dim
    // == vocab/hidden ≫ m, so they're excluded by the == m test).
    let act_views: std::collections::HashSet<String> = view_shape
        .iter()
        .filter(|(name, shp)| shp[0] == m && !result_views.contains(*name))
        .map(|(name, _)| name.clone())
        .collect();

    // Pass 2: rewrite. Shrink activation views to [1, k]; pin row index to m-1 on
    // every access tile over a result OR activation view. The constant is inserted
    // once per region just before its first use (a unique SSA per region).
    let row_const = format!("%lt_last_row_{}", m - 1);
    fn rewrite(
        ops: &mut Vec<Operation>,
        m: usize,
        act_views: &std::collections::HashSet<String>,
        result_views: &std::collections::HashSet<String>,
        row_const: &str,
    ) {
        let mut need_const = false;
        for op in ops.iter_mut() {
            if op.op_type == "ktdp.construct_memory_view"
                && let Some(res) = op.result.as_deref()
                && act_views.contains(res)
                && let Some(Attr::IntList(shape)) = op.attributes.get_mut("shape")
                && shape.len() == 2
            {
                shape[0] = 1; // activation A becomes a single row
            }
            if op.op_type == "ktdp.construct_access_tile"
                && let Some(view) = op.operands.first()
            {
                let view = view.clone();
                if (act_views.contains(&view) || result_views.contains(&view))
                    && op.operands.len() >= 2
                    && op.operands[1] != row_const
                {
                    op.operands[1] = row_const.to_string();
                    need_const = true;
                }
            }
            for rg in &mut op.regions {
                rewrite(rg, m, act_views, result_views, row_const);
            }
        }
        if need_const {
            let mut c = Operation::new(Some(row_const), "arith.constant", &[]);
            c.attributes
                .insert("value".into(), Attr::Int((m - 1) as i64));
            c.result_type = Some("index".into());
            ops.insert(0, c);
        }
    }
    rewrite(ops, m, &act_views, &result_views, &row_const);
    Ok(())
}

/// The largest core count any segment runs at — sizes the persistent LX array so
/// a native attention node's grid has an LX per core. Fused segments are `[1,1]`;
/// native attention nodes run at their own grid.
fn largest_grid(module: &IRModule, segments: &[Segment]) -> usize {
    let mut n = 1usize;
    for seg in segments {
        if let Segment::Native(node) = seg
            && let Ok(f) = module.get_function(&node.func)
        {
            let (gx, gy, gz) = f.grid;
            n = n.max(gx * gy * gz);
        }
    }
    n.max(1)
}

/// Execute a whole KTIR program with RESIDENT weights — the convenience
/// single-shot analogue of [`crate::segmented::execute_segmented`].
///
/// Builds a [`ResidentExecutor`], writes every source from `args` ONCE, runs one
/// pass, and reads back `outputs`. For a multi-pass loop (decode), construct a
/// [`ResidentExecutor`] directly and call [`ResidentExecutor::run`] per pass so
/// the weights are uploaded exactly once across all passes.
///
/// `args` keys are the canonical tensor names (`t<id>` / `%t<id>` / `%t<id>_ptr`
/// / bare `<id>`); `outputs` names the tensors to return (empty = the program's
/// declared results). The returned map is keyed by `t<id>`.
pub fn execute_resident(
    module: IRModule,
    spec: &ProgramSpec,
    args: &[(&str, Arg)],
    outputs: &[&str],
) -> Result<HashMap<String, Output>, String> {
    let mut exec = ResidentExecutor::new(module, spec)?;
    exec.set_sources(args)?;
    // PRODUCTION DEFAULT: only the last token's logits are computed — autoregressive
    // generation samples the next token from the final position, so the other rows
    // are dead output. Validated against the real model by the last-token golden test
    // (`real_forward_golden_last_token`): identical next-token prediction, ~1/m the
    // lm_head work. Callers needing ALL rows use `ResidentExecutor` with
    // `set_last_token_only(false)`.
    exec.set_last_token_only(true);
    exec.run(outputs)
}
