// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Cross-function fusion: collapse a multi-function KTIR program whose nodes
//! thread intermediates through HBM into a single function, forwarding each
//! whole-tensor producer→consumer edge as an SSA value (dropping the
//! `store`/`load` pair). This is a KTIR→KTIR transform — it removes real HBM
//! traffic the hardware would otherwise pay (RFC 0682 §"further optimizations
//! within that decomposition"; intermediates "reused across producer-consumer
//! operations").
//!
//! INCREMENT 1 (this file): only **whole-tensor** edges are forwarded — where
//! the consumer's `ktdp.load` reads the entire producer tensor (access-tile
//! shape == memory-view shape). Tiled consumers (a sub-tile read inside a loop)
//! are left as HBM `store`/`load` for now; they need `tensor.extract_slice`
//! against the producer value, which is increment 2.
//!
//! What's eliminated regardless of fusing an edge: the *inter-function* boundary
//! itself. The fused single function holds all intermediates in one resident
//! context, so even a non-forwarded edge no longer pays the per-call
//! marshal/read-back the multi-call runner imposed.

use ktir_core::ir::{Attr, IRFunction, IRModule, Operation};
use std::collections::{HashMap, HashSet};

/// One function argument's binding to a logical tensor and its direction.
#[derive(Clone, Debug)]
pub struct Binding {
    /// The function arg name (e.g. `%t335_ptr`).
    pub arg: String,
    /// Logical tensor id the arg points at.
    pub tensor: u64,
    /// True if the node writes this tensor (an output), false if it reads it.
    pub is_output: bool,
}

/// One node in the program: a function name + how its args bind to tensors.
#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub func: String,
    pub bindings: Vec<Binding>,
}

/// A whole multi-function program to fuse, in execution order.
#[derive(Clone, Debug)]
pub struct ProgramSpec {
    pub nodes: Vec<NodeSpec>,
    /// Tensors provided from outside (weights / inputs) — stay HBM args.
    pub sources: HashSet<u64>,
    /// Final result tensors — stay HBM (stored, read back by the caller).
    pub results: HashSet<u64>,
}

/// One execution unit produced by [`plan_segments`]: either a fused run of
/// consecutive non-attention nodes (run once at grid `[1,1]`, with intra-segment
/// HBM edges forwarded as SSA) or a single attention node kept verbatim so the
/// executor runs it across its NATIVE head-parallel grid.
///
/// Why the split: an attention node selects its head with
/// `ktdp.get_compute_tile_id` against a grid like `[32,1]`/`[9,1]`. Collapsed
/// into a single `[1,1]` function that primitive returns 0, so only head 0's
/// slice is computed — the rest of the output rows stay whatever the input was.
/// Running that node SEPARATELY at its native grid drives every head's core
/// (the per-node multi-core SPMD path, verified correct), and threading the
/// inter-segment tensors through HBM stitches the segments back together in
/// program order.
#[derive(Clone, Debug)]
pub enum Segment {
    /// A fused function over a maximal run of consecutive non-attention nodes.
    /// Its `func.grid` is `[1,1]`; the GPU GEMM reconstruction handles the
    /// token-parallel (`[8,1]`) matmul nodes folded in here by ignoring the SPMD
    /// grid and rebuilding the whole GEMM. Pointer args are named `%t<id>_ptr`.
    Fused(FusedSegment),
    /// A single attention node, run at its native multi-core grid. The
    /// `bindings` carry the original arg→tensor mapping the executor marshals.
    Native(NodeSpec),
}

/// A fused segment: the fused `[1,1]` function plus, for each pointer arg, the
/// tensor id it binds and whether the segment WRITES it (a boundary output the
/// caller must copy forward) or only READS it (a source / boundary input the
/// caller must already have resident). The runner marshals from this directly
/// instead of guessing direction from buffer presence.
#[derive(Clone, Debug)]
pub struct FusedSegment {
    pub func: IRFunction,
    /// Pointer-arg tensor ids written by this segment (boundary outputs).
    pub outputs: HashSet<u64>,
    /// Pointer-arg tensor ids only read by this segment (sources / inputs).
    pub inputs: HashSet<u64>,
}

/// True when `func` is a head-parallel ATTENTION node that must run at its native
/// grid (NOT be collapsed into a single-grid fused function).
///
/// Discriminator: a non-trivial grid (`> [1,1]`, so it has per-core heads) PLUS
/// the attention op signature — a `linalg.transpose` (the K transpose) and the
/// softmax `linalg.reduce { arith.maximumf }`. The signature excludes the
/// `[8,1]` token-parallel pure-matmul nodes (matmul but no transpose/softmax),
/// which the GPU GEMM reconstruction already runs correctly at grid `[1,1]`. In
/// DECODE the attention nodes are themselves grid `[1,1]` (single token), so the
/// grid clause keeps them fused (decode is correct single-grid).
pub fn is_attention_node(func: &IRFunction) -> bool {
    let (gx, gy, gz) = func.grid;
    if gx * gy * gz <= 1 {
        return false;
    }
    let mut has_transpose = false;
    let mut has_softmax_reduce = false;
    fn scan(ops: &[Operation], has_transpose: &mut bool, has_softmax_reduce: &mut bool) {
        for op in ops {
            if op.op_type == "linalg.transpose" {
                *has_transpose = true;
            }
            // The softmax max-reduce: a `linalg.reduce` whose combiner is
            // `arith.maximumf`. The parser lifts the `{ arith.maximumf }`
            // shorthand into a `reduce_fn` attribute; the explicit form keeps
            // the combiner as a region op. Match either.
            if op.op_type == "linalg.reduce"
                && (matches!(op.attributes.get("reduce_fn"), Some(Attr::Str(s)) if s == "arith.maximumf")
                    || region_has_op(&op.regions, "arith.maximumf"))
            {
                *has_softmax_reduce = true;
            }
            for rg in &op.regions {
                scan(rg, has_transpose, has_softmax_reduce);
            }
        }
    }
    scan(
        &func.operations,
        &mut has_transpose,
        &mut has_softmax_reduce,
    );
    has_transpose && has_softmax_reduce
}

/// True when `func` is a DECODE (m=1) attention node — the single-token form
/// whose grid is `[1,1]` (so [`is_attention_node`]'s grid clause excludes it) but
/// whose body is the unrolled per-head two-block online softmax. Discriminator:
/// grid `[1,1]`, a `linalg.transpose` (the K transpose) AND the softmax
/// `linalg.reduce { arith.maximumf }`. Structural, not name/shape based. Used —
/// by default, unless `KTIR_NO_FUSE_ATTN` is set — to ISOLATE the node into its
/// own segment so the resident executor runs the fused CPU path, not the op storm.
pub fn is_decode_attention_node(func: &IRFunction) -> bool {
    let (gx, gy, gz) = func.grid;
    if gx * gy * gz != 1 {
        return false;
    }
    let mut has_transpose = false;
    let mut has_softmax_reduce = false;
    fn scan(ops: &[Operation], has_transpose: &mut bool, has_softmax_reduce: &mut bool) {
        for op in ops {
            if op.op_type == "linalg.transpose" {
                *has_transpose = true;
            }
            if op.op_type == "linalg.reduce"
                && (matches!(op.attributes.get("reduce_fn"), Some(Attr::Str(s)) if s == "arith.maximumf")
                    || region_has_op(&op.regions, "arith.maximumf"))
            {
                *has_softmax_reduce = true;
            }
            for rg in &op.regions {
                scan(rg, has_transpose, has_softmax_reduce);
            }
        }
    }
    scan(
        &func.operations,
        &mut has_transpose,
        &mut has_softmax_reduce,
    );
    has_transpose && has_softmax_reduce
}

/// ROW-LOCALITY — is every cross-node tensor this node touches accessed only at
/// the node's OWN compute-tile row? A node is grid=[H,1]: compute-tile `k` runs
/// the body with `ktdp.get_compute_tile_id` ⇒ `k`. The node is **row-local** iff,
/// for every `ktdp.load`/`ktdp.store` of a tensor that some OTHER node produces
/// (a cross-node activation — NOT a weight/source, which no node writes), the
/// access tile's LEADING (row) index is exactly that compute-tile id. Then
/// compute-tile `k` reads and writes only row `k` of every inter-node tensor, so
/// the tiles are mutually independent and may stream in tile-major order with no
/// barrier between them.
///
/// A node is NOT row-local when it reads a cross-node activation across rows it
/// doesn't own — attention (reads ALL rows' K/V), a transpose, a cross-row
/// reduce. Such a node must run only AFTER every tile has finished the prior
/// nodes (a phase barrier). This generalizes [`is_attention_node`]: attention
/// reads cross-node K/V at a non-`pid` leading index, so it is caught here too,
/// along with any other cross-tile coupler the attention-only heuristic missed —
/// which is exactly why tile-major dataflow that split phases ONLY at attention
/// computed wrong results.
///
/// CONSERVATIVE: weights/sources (full-axis reads, never produced by a node) are
/// ignored; anything we cannot resolve to a provably row-local access makes the
/// node a barrier (correctness over parallelism). `produced` is the set of tensor
/// ids written by some node in the program.
pub fn node_is_row_local(func: &IRFunction, node: &NodeSpec, produced: &HashSet<u64>) -> bool {
    let pd = node_partition_dims(func, node);
    // Row-local ⟺ every cross-node tensor read sits on the compute-tile axis
    // (Some(dim)); a full-axis read (None) means it reads rows it doesn't own.
    // (Dim-agnostic legacy view — the phase builder uses the richer edge check.)
    let (gx, gy, gz) = func.grid;
    if gx * gy * gz <= 1 {
        return true;
    }
    pd.reads
        .iter()
        .all(|(t, d)| !produced.contains(t) || d.is_some())
}

/// Per-node compute-tile partition analysis. For each cross-/inter-node tensor the
/// node touches, recover WHICH tensor dimension carries the compute-tile id (`pid`)
/// in that access — its *partition dim*. `Some(d)` = the access selects only tile
/// `k`'s slice along dim `d`; `None` = the index on every dim is `pid`-free (a
/// full-axis / cross-row read, e.g. a weight, or attention reading all K/V rows).
///
/// This is the substrate for re-tiling detection: tile-major streaming is correct
/// across a producer→consumer edge ONLY if both partition the shared tensor on the
/// SAME dim (so consumer tile `k` reads exactly what producer tile `k` wrote). A
/// node that writes a tensor on dim 1 (e.g. attention, head-tiled) feeding a node
/// that reads it on dim 0 (token-tiled) is a re-tiling barrier even though BOTH
/// look "row-local" in isolation — the pid axis means a different thing on each
/// side. `reads`/`writes` carry `(tensor_id, partition_dim)` for inter-node edges.
pub struct PartitionDims {
    pub reads: Vec<(u64, Option<usize>)>,
    pub writes: Vec<(u64, Option<usize>)>,
}

pub fn node_partition_dims(func: &IRFunction, node: &NodeSpec) -> PartitionDims {
    // arg SSA name (e.g. "%t181_ptr") -> logical tensor id, from the bindings.
    let mut arg_tensor: HashMap<&str, u64> = HashMap::new();
    for b in &node.bindings {
        arg_tensor.insert(b.arg.as_str(), b.tensor);
    }
    let mut pid: Option<&str> = None;
    let mut view_arg: HashMap<&str, &str> = HashMap::new();
    // access-tile SSA -> (parent view SSA, [index SSA per dim]).
    let mut acc: HashMap<&str, (&str, Vec<&str>)> = HashMap::new();
    let mut reads: Vec<(u64, Option<usize>)> = Vec::new();
    let mut writes: Vec<(u64, Option<usize>)> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn walk<'a>(
        ops: &'a [Operation],
        arg_tensor: &HashMap<&str, u64>,
        pid: &mut Option<&'a str>,
        view_arg: &mut HashMap<&'a str, &'a str>,
        acc: &mut HashMap<&'a str, (&'a str, Vec<&'a str>)>,
        reads: &mut Vec<(u64, Option<usize>)>,
        writes: &mut Vec<(u64, Option<usize>)>,
    ) {
        for op in ops {
            match op.op_type.as_str() {
                "ktdp.get_compute_tile_id" => {
                    if let Some(r) = op.result.as_deref() {
                        *pid = Some(r);
                    }
                }
                "ktdp.construct_memory_view" => {
                    if let (Some(r), Some(ptr)) = (
                        op.result.as_deref(),
                        op.operands.first().map(|s| s.as_str()),
                    ) {
                        view_arg.insert(r, ptr);
                    }
                }
                "ktdp.construct_access_tile" => {
                    if let (Some(r), Some(view)) = (
                        op.result.as_deref(),
                        op.operands.first().map(|s| s.as_str()),
                    ) {
                        let idx: Vec<&str> = op.operands[1..].iter().map(|s| s.as_str()).collect();
                        acc.insert(r, (view, idx));
                    }
                }
                "ktdp.load" | "ktdp.store" => {
                    let is_store = op.op_type == "ktdp.store";
                    if let Some((_, (view, idx))) = op
                        .operands
                        .iter()
                        .find_map(|o| acc.get_key_value(o.as_str()))
                        && let Some(&tensor) = view_arg.get(view).and_then(|a| arg_tensor.get(a))
                    {
                        // Which dim's index is the compute-tile id?
                        let dim = pid.and_then(|p| idx.iter().position(|x| *x == p));
                        if is_store {
                            writes.push((tensor, dim));
                        } else {
                            reads.push((tensor, dim));
                        }
                    }
                }
                _ => {}
            }
            for rg in &op.regions {
                walk(rg, arg_tensor, pid, view_arg, acc, reads, writes);
            }
        }
    }
    walk(
        &func.operations,
        &arg_tensor,
        &mut pid,
        &mut view_arg,
        &mut acc,
        &mut reads,
        &mut writes,
    );
    PartitionDims { reads, writes }
}

/// CONTRACT (B) — the single source of truth that partitions the cap (KV-length)
/// axis between the project's two attention optimizations, so that **each
/// attention node receives EXACTLY ONE transform** and the region-free
/// batched-executor gate (`interpreter.rs`, the `!regions.is_empty()` clause in
/// `execute_function_gpu`) is never violated:
///
/// * **scores tile FITS LX** (below the cap) → leave attention **naive**. The node
///   stays a region-free [`Segment::Native`] and is eligible for **head-batching**
///   on the GPU multi-core batched executor (the multi-core-GPU TODO). Head dim is
///   tiled across cores.
/// * **scores tile OVERFLOWS LX** (above the cap) → the **flash-attention pass**
///   (the FA-rewrite TODO) rewrites the node into a tiled `scf.for` online-softmax
///   form that fits LX. The cap/KV dim is tiled. That node is now *region-bearing*,
///   so it runs on the generic interpreter (the batched executor's region-free gate
///   makes it `Err` → fall back, which is the *intended* path above the cap —
///   head-batching cannot help a node whose scores already overflow LX).
///
/// The two are orthogonal (head dim vs cap dim) and compose only via a later
/// region-aware INC; **neither fleet edits the region-free gate line** (reserved
/// for that post-merge step). Because the predicate is monotone in `scores_bytes`
/// and exhaustively partitions the axis, there is no overlap (no double-transform)
/// and no gap (no silently-unhandled regime).
///
/// `scores_bytes` is the byte footprint of the attention scores tile `[m, cap]`
/// (numel × storage-dtype bytes), as recovered by the FA recognizer; `lx_budget`
/// is the per-core LX byte budget the segmenter already uses
/// (`KTIR_LX_FUSION_BUDGET`, default 7/8 of 2 MB). Threshold mirrors that 7/8
/// convention. **Fail-safe:** callers that cannot prove the scores footprint must
/// pass a value that keeps this `false` (stay naive) — never force an FA path we
/// cannot prove correct.
pub fn attention_needs_flash(scores_bytes: usize, lx_budget: usize) -> bool {
    // scores_bytes ≥ 7/8 · lx_budget  ⟺  the scores tile would overflow the LX
    // fusion budget and must be cap-tiled (flash attention). Saturating math so a
    // pathological huge footprint can't wrap.
    lx_budget != 0 && scores_bytes.saturating_mul(8) >= lx_budget.saturating_mul(7)
}

/// Recover the tensor id from a fused pointer-arg name `%t<id>_ptr`.
fn tensor_id_of_arg(arg: &str) -> u64 {
    arg.trim_start_matches('%')
        .trim_start_matches('t')
        .trim_end_matches("_ptr")
        .parse()
        .unwrap_or_else(|_| panic!("unexpected fused arg name {arg:?}"))
}

/// True if any op at any region depth in `regions` has `op_type`.
fn region_has_op(regions: &[Vec<Operation>], op_type: &str) -> bool {
    regions.iter().any(|rg| {
        rg.iter()
            .any(|op| op.op_type == op_type || region_has_op(&op.regions, op_type))
    })
}

/// Partition `spec` into ordered execution segments: maximal runs of consecutive
/// non-attention nodes fused into one `[1,1]` function each, with every
/// attention node kept as its own [`Segment::Native`] to run at its native grid.
///
/// Each fused segment is fused with a segment-LOCAL `ProgramSpec` whose
/// `sources`/`results` are widened to pin every tensor that crosses the
/// segment's boundary (read from another segment / a true source, or written
/// for another segment / a true result) as an HBM pointer arg. Only edges
/// internal to the run forward as SSA / `extract_slice`; boundary edges stay HBM
/// so the caller can thread them between segments and the native attention nodes.
///
/// The returned segments execute in order; the caller marshals one HBM buffer
/// per tensor id, runs each segment (fused via the interpreter at `[1,1]`,
/// native at its grid), and copies every output buffer forward — exactly the
/// proven per-node threading, just with non-attention runs collapsed.
pub fn plan_segments(module: &IRModule, spec: &ProgramSpec) -> Result<Vec<Segment>, String> {
    // No LX budget: maximal-fuse every non-attention run (the historical behavior;
    // the optimizer's own unit tests use this).
    plan_segments_budgeted(module, spec, usize::MAX, &HashMap::new())
}

/// Like [`plan_segments`], but bounds each FUSED segment's peak LX live-set to
/// `lx_budget` bytes — splitting a maximal non-attention run into several fused
/// segments when its co-resident `[m, *]` intermediates would overflow LX.
/// `tensor_bytes[id]` is a tensor's LX footprint (numel × storage-dtype bytes).
///
/// Without this a whole transformer MLP (gate/up/silu·up/down + norms) fuses into
/// one `[1,1]` segment whose wide intermediates are live at once — llama m=32:
/// gate+up+product = 3×[32,8192] + residual > 2 MB LX. The per-op `dies_at`
/// reclaim cannot free genuinely-live tensors, so the fix is to not over-group
/// them. Edges the split introduces fall back to HBM, like any segment boundary.
pub fn plan_segments_budgeted(
    module: &IRModule,
    spec: &ProgramSpec,
    lx_budget: usize,
    tensor_bytes: &HashMap<u64, usize>,
) -> Result<Vec<Segment>, String> {
    // Per-node attention classification. Nodes flagged here are kept as their own
    // `Segment::Native` (run at their native grid, NOT collapsed into a fused
    // [1,1] function). The head-parallel prefill form (grid > 1) is always
    // isolated. The DECODE (m=1, grid [1,1]) form is ALSO isolated by default — so
    // the resident executor can run the fused CPU attention for it (a measured
    // decode win, golden-faithful). Set `KTIR_NO_FUSE_ATTN` to opt out: decode
    // attention then stays folded into the fused segment (the decomposed oracle
    // path), so the suite stays byte-identical to the pre-fusion baseline.
    let fuse_decode_attn = std::env::var_os("KTIR_NO_FUSE_ATTN").is_none();
    let attn: Vec<bool> = spec
        .nodes
        .iter()
        .map(|n| {
            module
                .get_function(&n.func)
                .map(|f| is_attention_node(f) || (fuse_decode_attn && is_decode_attention_node(f)))
        })
        .collect::<Result<_, _>>()?;

    // For widening segment-local sources/results: which node indices produce /
    // consume each tensor, across the WHOLE program.
    let mut produced_at: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut consumed_at: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, node) in spec.nodes.iter().enumerate() {
        for b in &node.bindings {
            if b.is_output {
                produced_at.entry(b.tensor).or_default().push(i);
            } else {
                consumed_at.entry(b.tensor).or_default().push(i);
            }
        }
    }
    // Whole-program last-touch index per tensor — the LX split uses it to tell
    // when a tensor crosses a sub-run boundary (and so must persist to it).
    let mut global_last: HashMap<u64, usize> = HashMap::new();
    for (i, node) in spec.nodes.iter().enumerate() {
        for b in &node.bindings {
            global_last.insert(b.tensor, i);
        }
    }

    let mut segments: Vec<Segment> = Vec::new();
    let mut i = 0;
    while i < spec.nodes.len() {
        if attn[i] {
            segments.push(Segment::Native(spec.nodes[i].clone()));
            i += 1;
            continue;
        }
        // Maximal run [start, j) of consecutive non-attention nodes.
        let start = i;
        let mut j = i;
        while j < spec.nodes.len() && !attn[j] {
            j += 1;
        }
        // Split the maximal run into sub-runs that each fit the LX live-set
        // budget (the whole run, unsplit, when lx_budget is usize::MAX), each
        // becoming its own fused segment.
        for run in split_run(&spec.nodes, start, j, tensor_bytes, &global_last, lx_budget) {
            segments.push(build_fused_segment(
                module,
                spec,
                run,
                &produced_at,
                &consumed_at,
            )?);
        }
        i = j;
    }
    Ok(segments)
}

/// Build ONE fused segment from node sub-range `run`, widening its segment-local
/// sources/results so any boundary-crossing edge stays an HBM pointer (never
/// forwarded as SSA across a segment break).
fn build_fused_segment(
    module: &IRModule,
    spec: &ProgramSpec,
    run: std::ops::Range<usize>,
    produced_at: &HashMap<u64, Vec<usize>>,
    consumed_at: &HashMap<u64, Vec<usize>>,
) -> Result<Segment, String> {
    let in_run = |k: usize| run.contains(&k);
    // Segment-local sources/results: widen so any boundary-crossing edge stays an
    // HBM pointer (never forwarded as SSA across a segment break).
    let mut seg_sources: HashSet<u64> = HashSet::new();
    let mut seg_results: HashSet<u64> = HashSet::new();
    for k in run.clone() {
        for b in &spec.nodes[k].bindings {
            if b.is_output {
                let consumed_outside = consumed_at
                    .get(&b.tensor)
                    .is_some_and(|cs| cs.iter().any(|&c| !in_run(c)));
                if spec.results.contains(&b.tensor) || consumed_outside {
                    seg_results.insert(b.tensor);
                }
            } else {
                let produced_outside = produced_at
                    .get(&b.tensor)
                    .is_some_and(|ps| ps.iter().any(|&p| !in_run(p)));
                if spec.sources.contains(&b.tensor) || produced_outside {
                    seg_sources.insert(b.tensor);
                }
            }
        }
    }
    let seg_spec = ProgramSpec {
        nodes: spec.nodes[run.clone()].to_vec(),
        sources: seg_sources.clone(),
        results: seg_results.clone(),
    };
    let mut func = fuse_program(module, &seg_spec)?;
    // Force the fused segment to grid [1,1] (single core). `fuse_program` stamps
    // the grid from the run's FIRST node, which can be a token-parallel [8,1]
    // matmul node — but the whole point of folding those in is that the GPU GEMM
    // reconstruction (and the single-core K-loop offload it rides on) rebuilds the
    // full M at grid [1,1], ignoring the Spyre SPMD grid. A residual [8,1] grid
    // would (a) re-tile the GEMM across cores so the single-core offload never
    // fires, and (b) make each core recompute the whole reconstructed GEMM.
    // Collapsing to [1,1] is the correct + fast path.
    func.grid = (1, 1, 1);
    // Classify the fused function's surviving pointer args by direction against
    // the boundary sets: a `%t<id>_ptr` is a boundary OUTPUT iff `id ∈ seg_results`,
    // a boundary INPUT iff `id ∈ seg_sources`. Anything else is an INTERNAL SCRATCH
    // arg (an intra-segment edge fusion could NOT forward as SSA — resident HBM the
    // fused fn writes then reads in its own body; the runner zero-inits it).
    let mut outputs: HashSet<u64> = HashSet::new();
    let mut inputs: HashSet<u64> = HashSet::new();
    for (arg, _) in &func.arguments {
        let id = tensor_id_of_arg(arg);
        if seg_results.contains(&id) {
            outputs.insert(id);
        } else if seg_sources.contains(&id) {
            inputs.insert(id);
        }
    }
    Ok(Segment::Fused(FusedSegment {
        func,
        outputs,
        inputs,
    }))
}

/// Split node range `[start, j)` into consecutive sub-ranges whose fused LX
/// live-set each fits `budget`. Greedy: grow a sub-run until adding the next node
/// would push the peak co-resident bytes over budget, then start a new one. A lone
/// node over budget is kept alone (that is node-level tiling, not fusion's job).
/// `budget == usize::MAX` (or empty `tensor_bytes`) ⇒ the whole run, unsplit.
fn split_run(
    nodes: &[NodeSpec],
    start: usize,
    j: usize,
    tensor_bytes: &HashMap<u64, usize>,
    global_last: &HashMap<u64, usize>,
    budget: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut subs: Vec<std::ops::Range<usize>> = Vec::new();
    if budget == usize::MAX || tensor_bytes.is_empty() {
        subs.push(start..j); // whole run, unsplit
        return subs;
    }
    let mut s = start;
    while s < j {
        // Grow `e` (exclusive) while including node `e` keeps [s, e] within budget;
        // always include at least node `s`.
        let mut e = s + 1;
        while e < j && peak_live_bytes(nodes, s, e + 1, tensor_bytes, global_last) <= budget {
            e += 1;
        }
        subs.push(s..e);
        s = e;
    }
    subs
}

/// Peak co-resident LX bytes over node range `[s, e)` (exclusive `e`), at
/// NODE-output granularity: each tensor a node touches is live from its first
/// touch in the window to its last touch in the window — or to the window end if
/// it is also touched later in the program (it then crosses the sub-run boundary
/// and must persist to be stored). This captures exactly the wide intermediates
/// the per-op reclaim cannot free (e.g. an MLP's gate/up/product held at once);
/// intra-node temporaries are GPU/scratch-side, not large LX tiles.
fn peak_live_bytes(
    nodes: &[NodeSpec],
    s: usize,
    e: usize,
    tensor_bytes: &HashMap<u64, usize>,
    global_last: &HashMap<u64, usize>,
) -> usize {
    let mut win_last: HashMap<u64, usize> = HashMap::new();
    for (k, node) in nodes.iter().enumerate().take(e).skip(s) {
        for b in &node.bindings {
            win_last.insert(b.tensor, k);
        }
    }
    let mut live: HashMap<u64, usize> = HashMap::new();
    let mut peak = 0usize;
    for (k, node) in nodes.iter().enumerate().take(e).skip(s) {
        for b in &node.bindings {
            live.insert(b.tensor, tensor_bytes.get(&b.tensor).copied().unwrap_or(0));
        }
        peak = peak.max(live.values().sum());
        // Free tensors whose last in-window use is this node AND that are not
        // touched after the window (those persist to the sub-run boundary).
        live.retain(|tid, _| {
            win_last.get(tid).copied().unwrap_or(k) > k
                || global_last.get(tid).copied().unwrap_or(0) >= e
        });
    }
    peak
}

/// Fuse `spec`'s nodes (functions in `module`) into a single `IRFunction`.
///
/// The fused function's args are the source + result tensor pointers (one per
/// distinct tensor, named `%t<id>_ptr`); intermediates produced and consumed
/// whole-tensor are forwarded as SSA and need no pointer.
pub fn fuse_program(module: &IRModule, spec: &ProgramSpec) -> Result<IRFunction, String> {
    // Tensors that are produced by some node AND consumed by another, and are
    // neither a source nor a final result: candidates for SSA forwarding.
    let mut produced_by: HashMap<u64, usize> = HashMap::new();
    let mut consumed: HashSet<u64> = HashSet::new();
    for (i, node) in spec.nodes.iter().enumerate() {
        for b in &node.bindings {
            if b.is_output {
                produced_by.insert(b.tensor, i);
            } else {
                consumed.insert(b.tensor);
            }
        }
    }
    let is_intermediate = |t: u64| {
        produced_by.contains_key(&t)
            && consumed.contains(&t)
            && !spec.sources.contains(&t)
            && !spec.results.contains(&t)
    };

    // Analyze every node once (region-aware) and cache — `all_consumers_*`
    // would otherwise re-walk every consumer per producer (O(nodes²)).
    let analyses: Vec<Analysis> = spec
        .nodes
        .iter()
        .map(|n| module.get_function(&n.func).map(analyze))
        .collect::<Result<_, _>>()?;

    // `produced[T]` = (fused-function SSA value holding tensor T, its full shape),
    // recorded once its producing node is inlined and its whole-tensor store
    // forwarded. The shape pins the layout a tiled consumer slices into.
    let mut produced: HashMap<u64, (String, Vec<i64>)> = HashMap::new();
    // Pointer args the fused function still needs (sources, results, and any
    // intermediate edge we could not forward), keyed by tensor id.
    let mut needed_args: Vec<(u64, String)> = Vec::new();
    let mut have_arg: HashSet<u64> = HashSet::new();
    let mut body: Vec<Operation> = Vec::new();

    for (ni, node) in spec.nodes.iter().enumerate() {
        let an = &analyses[ni];
        let arg_to_tensor: HashMap<&str, &Binding> =
            node.bindings.iter().map(|b| (b.arg.as_str(), b)).collect();

        // ----- decide which of this node's pointer args get forwarded -----
        // An INPUT arg is forwarded iff its producer is resident AND *every*
        // load through it reads the producer's full-shape layout in a way we can
        // model — whole-tensor (alias) or a contiguous sub-tile (extract_slice).
        let mut forwarded_args: HashSet<String> = HashSet::new();
        let mut loads_by_arg: HashMap<&str, Vec<&LoadChain>> = HashMap::new();
        for ld in &an.loads {
            loads_by_arg.entry(ld.arg.as_str()).or_default().push(ld);
        }
        for (arg, lds) in &loads_by_arg {
            let Some(b) = arg_to_tensor.get(*arg) else {
                continue;
            };
            if b.is_output || !is_intermediate(b.tensor) {
                continue;
            }
            let Some((_, pshape)) = produced.get(&b.tensor) else {
                continue; // producer not resident -> keep HBM load
            };
            if lds
                .iter()
                .all(|l| &l.view_shape == pshape && (l.whole_tensor || l.sliceable))
            {
                forwarded_args.insert((*arg).to_string());
            }
        }
        // An OUTPUT arg is forwarded iff the producer writes the whole tensor and
        // every consuming node can forward it (same full-shape + whole/sliceable).
        // Record the resident SSA so later nodes can forward off it.
        for st in &an.stores {
            let Some(b) = arg_to_tensor.get(st.arg.as_str()) else {
                continue;
            };
            if b.is_output
                && st.whole_tensor
                && is_intermediate(b.tensor)
                && all_consumers_forwardable(spec, &analyses, b.tensor, &st.view_shape)
            {
                forwarded_args.insert(st.arg.clone());
                produced.insert(b.tensor, (prefixed(ni, &st.stored), st.view_shape.clone()));
            }
        }

        // ----- turn the forwarding decision into concrete drop/rename ops -----
        let mut rename: HashMap<String, String> = HashMap::new();
        // Op result SSAs to drop entirely (views/tiles on forwarded args, and
        // whole-tensor loads whose value is aliased to the producer).
        let mut drop_results: HashSet<String> = HashSet::new();
        // ktdp.store ops whose tile operand is in here are dropped.
        let mut drop_store_tiles: HashSet<String> = HashSet::new();
        // load result SSA -> the extract_slice that replaces it (tiled forward).
        let mut slice_at_load: HashMap<String, SliceForward> = HashMap::new();

        // Drop the construct_memory_view of every forwarded arg (its HBM pointer
        // is gone), then the access tiles built on those views.
        for (vssa, (arg, _)) in &an.views {
            if forwarded_args.contains(arg) {
                drop_results.insert(vssa.clone());
            }
        }
        for (tssa, ti) in &an.tiles {
            if drop_results.contains(&ti.view) {
                drop_results.insert(tssa.clone());
            }
        }
        // Loads on dropped tiles: alias (whole) or slice (tiled).
        for ld in &an.loads {
            if !drop_results.contains(&ld.tile) {
                continue;
            }
            let Some(b) = arg_to_tensor.get(ld.arg.as_str()) else {
                continue;
            };
            let Some((val, _)) = produced.get(&b.tensor) else {
                continue;
            };
            if ld.whole_tensor {
                rename.insert(ld.loaded.clone(), val.clone());
                drop_results.insert(ld.loaded.clone());
            } else {
                slice_at_load.insert(
                    ld.loaded.clone(),
                    SliceForward {
                        source: val.clone(),
                        loaded: ld.loaded.clone(),
                        offsets: ld.offsets.clone(),
                        sizes: ld.tile_shape.clone(),
                    },
                );
            }
        }
        // Stores on dropped tiles: drop the store itself.
        for st in &an.stores {
            if drop_results.contains(&st.tile) {
                drop_store_tiles.insert(st.tile.clone());
            }
        }

        // Non-forwarded args keep an HBM pointer, shared by tensor id under the
        // canonical name; map this node's arg name onto it.
        for b in &node.bindings {
            if forwarded_args.contains(&b.arg) {
                continue;
            }
            let canon = format!("%t{}_ptr", b.tensor);
            rename.insert(b.arg.clone(), canon.clone());
            if have_arg.insert(b.tensor) {
                needed_args.push((b.tensor, canon));
            }
        }

        // Emit the node's ops (recursively, into regions), renamed, dropping the
        // forwarded chains and substituting tiled loads with their extract_slice.
        body.extend(emit_ops(
            &module.get_function(&node.func)?.operations,
            ni,
            &rename,
            &drop_results,
            &drop_store_tiles,
            &slice_at_load,
        ));
    }

    // Fused function args, in a deterministic order: sources, then results.
    needed_args.sort_by_key(|(t, _)| {
        let cls = if spec.sources.contains(t) {
            0
        } else if spec.results.contains(t) {
            2
        } else {
            1
        };
        (cls, *t)
    });
    let args: Vec<(String, String)> = needed_args
        .into_iter()
        .map(|(_, name)| (name, "index".to_string()))
        .collect();
    body.push(Operation::new(None, "func.return", &[]));

    Ok(IRFunction {
        name: "fused".to_string(),
        arguments: args,
        operations: body,
        grid: spec
            .nodes
            .first()
            .map(|n| {
                module
                    .get_function(&n.func)
                    .map(|f| f.grid)
                    .unwrap_or((1, 1, 1))
            })
            .unwrap_or((1, 1, 1)),
        return_type: None,
    })
}

/// True if every node consuming `tensor` reads it in a way we can forward off a
/// resident SSA value of shape `pshape`: every load through that arg must read
/// the producer's full-shape layout (`view_shape == pshape`) as a whole tensor
/// or a contiguous sub-tile. Any other read (a different view shape, a
/// non-identity base_map, indirect/gather access, or no load at all) keeps the
/// producer store and that consumer's HBM load. `analyses[i]` is node `i`'s
/// cached analysis.
fn all_consumers_forwardable(
    spec: &ProgramSpec,
    analyses: &[Analysis],
    tensor: u64,
    pshape: &[i64],
) -> bool {
    for (i, node) in spec.nodes.iter().enumerate() {
        for b in &node.bindings {
            if !b.is_output && b.tensor == tensor {
                let lds: Vec<&LoadChain> = analyses[i]
                    .loads
                    .iter()
                    .filter(|l| l.arg == b.arg)
                    .collect();
                if lds.is_empty() {
                    return false; // consumed but no recognizable load -> can't forward
                }
                if !lds
                    .iter()
                    .all(|l| l.view_shape == pshape && (l.whole_tensor || l.sliceable))
                {
                    return false;
                }
            }
        }
    }
    true
}

/// A tiled forwarded load rewritten as a `tensor.extract_slice` of the
/// producer's resident SSA value. Built at emit time so its offset operands
/// resolve through the node's final rename map; the `source` is already in the
/// fused namespace (the producer node prefixed it) and is emitted verbatim.
struct SliceForward {
    source: String,
    loaded: String,
    offsets: Vec<String>,
    sizes: Vec<i64>,
}

impl SliceForward {
    fn build(&self, ni: usize, rename: &HashMap<String, String>) -> Operation {
        let res = resolve(ni, &self.loaded, rename);
        let offsets: Vec<String> = self
            .offsets
            .iter()
            .map(|o| resolve(ni, o, rename))
            .collect();
        let sizes: Vec<String> = self.sizes.iter().map(|n| n.to_string()).collect();
        let strides: Vec<String> = self.sizes.iter().map(|_| "1".to_string()).collect();
        Operation::new(Some(&res), "tensor.extract_slice", &[self.source.as_str()])
            .with_attr("slice_offsets", Attr::StrList(offsets))
            .with_attr("slice_sizes", Attr::StrList(sizes))
            .with_attr("slice_strides", Attr::StrList(strides))
    }
}

// --- per-function analysis (region-aware) ----------------------------------

struct LoadChain {
    arg: String,
    loaded: String,
    /// View SSA the access tile is built on (dropped when the arg is forwarded).
    #[allow(dead_code)]
    view: String,
    /// Access-tile SSA — identifies the tile op to drop and the load to rewrite.
    tile: String,
    whole_tensor: bool,
    /// The access tile's index operands (`construct_access_tile %view[%i, %j]`).
    /// With an identity `base_map` these are the slice's per-axis start offsets.
    offsets: Vec<String>,
    /// The access tile's logical shape — the slice sizes for a tiled forward.
    tile_shape: Vec<i64>,
    /// The memory-view's shape — must equal the producer's stored shape for the
    /// forward to index the right layout.
    view_shape: Vec<i64>,
    /// True when the access tile reads a contiguous box at `offsets` (identity
    /// `base_map`, no reordering) — the only shape a plain `extract_slice` models.
    sliceable: bool,
}
struct StoreChain {
    arg: String,
    stored: String,
    tile: String,
    whole_tensor: bool,
    view_shape: Vec<i64>,
}

/// A construct_access_tile's decoded fields.
struct TileInfo {
    view: String,
    offsets: Vec<String>,
    shape: Vec<i64>,
    base_identity: bool,
    has_order: bool,
}

#[derive(Default)]
struct Analysis {
    loads: Vec<LoadChain>,
    stores: Vec<StoreChain>,
    /// view SSA -> (arg pointer it interprets, view shape).
    views: HashMap<String, (String, Vec<i64>)>,
    /// access-tile SSA -> decoded tile.
    tiles: HashMap<String, TileInfo>,
}

/// Trace every `ktdp.load`/`ktdp.store` — at any region depth — back through its
/// access tile and memory view to the function arg pointer it touches. The real
/// model issues its tiled loads INSIDE an `scf.for`, so the walk must recurse
/// into op regions; views/tiles are collected across all depths first (a tile in
/// a loop body is built on a view declared at function top level).
fn analyze(func: &IRFunction) -> Analysis {
    let mut a = Analysis::default();
    collect_views_tiles(&func.operations, &mut a);
    // Borrow-split: read views/tiles while pushing into loads/stores.
    let Analysis {
        views,
        tiles,
        loads,
        stores,
    } = &mut a;
    collect_loads_stores(&func.operations, views, tiles, loads, stores);
    a
}

fn shape_attr_of(op: &Operation) -> Vec<i64> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) => v.clone(),
        _ => Vec::new(),
    }
}

fn collect_views_tiles(ops: &[Operation], a: &mut Analysis) {
    for op in ops {
        match op.op_type.as_str() {
            "ktdp.construct_memory_view" => {
                if let (Some(res), Some(arg)) = (&op.result, op.operands.first()) {
                    a.views
                        .insert(res.clone(), (arg.clone(), shape_attr_of(op)));
                }
            }
            "ktdp.construct_access_tile" => {
                if let (Some(res), Some(view)) = (&op.result, op.operands.first()) {
                    a.tiles.insert(
                        res.clone(),
                        TileInfo {
                            view: view.clone(),
                            offsets: op.operands[1..].to_vec(),
                            shape: shape_attr_of(op),
                            base_identity: base_map_is_identity(op),
                            has_order: op.attributes.contains_key("coordinate_order"),
                        },
                    );
                }
            }
            _ => {}
        }
        for rg in &op.regions {
            collect_views_tiles(rg, a);
        }
    }
}

fn collect_loads_stores(
    ops: &[Operation],
    views: &HashMap<String, (String, Vec<i64>)>,
    tiles: &HashMap<String, TileInfo>,
    loads: &mut Vec<LoadChain>,
    stores: &mut Vec<StoreChain>,
) {
    for op in ops {
        match op.op_type.as_str() {
            "ktdp.load" => {
                if let (Some(loaded), Some(tile_ssa)) = (&op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile_ssa)
                    && let Some((arg, vshape)) = views.get(&ti.view)
                {
                    let whole = !ti.shape.is_empty() && &ti.shape == vshape;
                    let sliceable = ti.base_identity
                        && !ti.has_order
                        && !ti.offsets.is_empty()
                        && ti.offsets.len() == ti.shape.len();
                    loads.push(LoadChain {
                        arg: arg.clone(),
                        loaded: loaded.clone(),
                        view: ti.view.clone(),
                        tile: tile_ssa.clone(),
                        whole_tensor: whole,
                        offsets: ti.offsets.clone(),
                        tile_shape: ti.shape.clone(),
                        view_shape: vshape.clone(),
                        sliceable,
                    });
                }
            }
            "ktdp.store" => {
                if let (Some(stored), Some(tile_ssa)) = (op.operands.first(), op.operands.get(1))
                    && let Some(ti) = tiles.get(tile_ssa)
                    && let Some((arg, vshape)) = views.get(&ti.view)
                {
                    let whole = !ti.shape.is_empty() && &ti.shape == vshape;
                    stores.push(StoreChain {
                        arg: arg.clone(),
                        stored: stored.clone(),
                        tile: tile_ssa.clone(),
                        whole_tensor: whole,
                        view_shape: vshape.clone(),
                    });
                }
            }
            _ => {}
        }
        for rg in &op.regions {
            collect_loads_stores(rg, views, tiles, loads, stores);
        }
    }
}

/// True when a `construct_access_tile` op's `base_map` is the identity (so the
/// access reads a contiguous box starting at its index operands). An absent
/// `base_map` is identity by construction (the emulator synthesizes one).
fn base_map_is_identity(tile_op: &Operation) -> bool {
    match tile_op.attributes.get("base_map") {
        Some(Attr::AffineMap(m)) => m.is_identity(),
        _ => true,
    }
}

// --- SSA renaming + recursive emit -----------------------------------------

/// `%foo` -> `%nN_foo` (node-local rename to avoid collisions across inlined nodes).
fn prefixed(ni: usize, ssa: &str) -> String {
    format!("%n{ni}_{}", ssa.trim_start_matches('%'))
}

/// Resolve an operand/result name through the rename map: an explicit mapping
/// wins (arg→canonical/forwarded); otherwise an SSA value gets the node prefix.
fn resolve(ni: usize, name: &str, rename: &HashMap<String, String>) -> String {
    if let Some(mapped) = rename.get(name) {
        return mapped.clone();
    }
    if name.starts_with('%') {
        prefixed(ni, name)
    } else {
        name.to_string() // non-SSA token (rare in operands)
    }
}

/// Some ops carry SSA names in ATTRIBUTES, not just operands — `scf.for`'s
/// induction variable (`iter_var`) and loop-carried names (`iter_args`), and any
/// op's multi-result `result_names` / a view's dynamic `sizes_dyn`. These must be
/// renamed in lockstep with the op stream, or a fused loop body would reference a
/// differently-prefixed induction variable than the one the loop binds.
fn rename_attrs(
    op: &Operation,
    ni: usize,
    rename: &HashMap<String, String>,
) -> std::collections::HashMap<String, Attr> {
    let mut attrs = op.attributes.clone();
    // `outs_var` holds the SSA name of an op's `outs` operand (e.g. a
    // `linalg.reduce`'s init accumulator splat). It must be prefixed like every
    // other SSA reference: each node defines its OWN `%sinit6 = tensor.splat 0.0`
    // (renamed to `%n<ni>_sinit6` here), and the reduce folds `outs` as the INITIAL
    // accumulator value. If `outs_var` is left UNprefixed, every node's reduce
    // reads/writes a single shared bare `%sinit6` slot — so node N's reduce folds
    // node N-1's stale partial sum instead of its own freshly-splat 0.0, corrupting
    // the reduction (the RMSNorm sum-of-squares grew monotonically across layers,
    // diverging the e2e golden by ~30 logits). Renaming it gives each reduce a
    // FRESH, per-node identity accumulator, so folding `outs` stays bit-exact.
    for key in [
        "iter_var",
        "iter_args",
        "result_names",
        "sizes_dyn",
        "outs_var",
    ] {
        match attrs.get(key) {
            Some(Attr::Str(s)) => {
                attrs.insert(key.to_string(), Attr::Str(resolve(ni, s, rename)));
            }
            Some(Attr::StrList(xs)) => {
                let mapped = xs.iter().map(|s| resolve(ni, s, rename)).collect();
                attrs.insert(key.to_string(), Attr::StrList(mapped));
            }
            _ => {}
        }
    }
    attrs
}

/// Emit a node's ops into the fused body, recursing into regions: rename every
/// SSA (operands, results, SSA-bearing attributes, nested regions), drop the
/// forwarded view/tile/load/store chains, and substitute each tiled forwarded
/// load with its `extract_slice`. Per-node `func.return`s are dropped (the fused
/// function gets a single trailing return).
fn emit_ops(
    ops: &[Operation],
    ni: usize,
    rename: &HashMap<String, String>,
    drop_results: &HashSet<String>,
    drop_store_tiles: &HashSet<String>,
    slice_at_load: &HashMap<String, SliceForward>,
) -> Vec<Operation> {
    let mut out = Vec::new();
    for op in ops {
        if op.op_type == "func.return" {
            continue;
        }
        if let Some(r) = &op.result
            && drop_results.contains(r)
        {
            continue;
        }
        if op.op_type == "ktdp.store"
            && let Some(tile) = op.operands.get(1)
            && drop_store_tiles.contains(tile)
        {
            continue;
        }
        if op.op_type == "ktdp.load"
            && let Some(r) = &op.result
            && let Some(sf) = slice_at_load.get(r)
        {
            out.push(sf.build(ni, rename));
            continue;
        }
        out.push(Operation {
            result: op.result.as_ref().map(|r| resolve(ni, r, rename)),
            op_type: op.op_type.clone(),
            operands: op.operands.iter().map(|o| resolve(ni, o, rename)).collect(),
            attributes: rename_attrs(op, ni, rename),
            result_type: op.result_type.clone(),
            regions: op
                .regions
                .iter()
                .map(|rg| {
                    emit_ops(
                        rg,
                        ni,
                        rename,
                        drop_results,
                        drop_store_tiles,
                        slice_at_load,
                    )
                })
                .collect(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktir_core::ir::Attr;

    /// Build a node function: load whole tensor from `in_arg`, "compute"
    /// (identity copy via op `%out = <op> %loaded`), store whole to `out_arg`.
    /// `whole` toggles whether the access-tile shape matches the view (full) or
    /// is a sub-tile (forces the HBM fallback).
    fn copy_node(name: &str, in_arg: &str, out_arg: &str, shape: i64, whole: bool) -> IRFunction {
        let tile_shape = if whole { shape } else { shape / 2 };
        let mk_view = |res: &str, arg: &str| {
            Operation::new(Some(res), "ktdp.construct_memory_view", &[arg])
                .with_attr("shape", Attr::IntList(vec![shape]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr("memory_space", Attr::Str("HBM".into()))
                .with_attr("dtype", Attr::Str("f16".into()))
        };
        let mk_tile = |res: &str, view: &str| {
            Operation::new(Some(res), "ktdp.construct_access_tile", &[view])
                .with_attr("shape", Attr::IntList(vec![tile_shape]))
        };
        IRFunction {
            name: name.to_string(),
            arguments: vec![
                (in_arg.to_string(), "index".into()),
                (out_arg.to_string(), "index".into()),
            ],
            grid: (1, 1, 1),
            return_type: None,
            operations: vec![
                mk_view("%vin", in_arg),
                mk_tile("%tin", "%vin"),
                Operation::new(Some("%loaded"), "ktdp.load", &["%tin"]),
                Operation::new(Some("%y"), "math.exp", &["%loaded"]),
                mk_view("%vout", out_arg),
                mk_tile("%tout", "%vout"),
                Operation::new(None, "ktdp.store", &["%y", "%tout"]),
                Operation::new(None, "func.return", &[]),
            ],
        }
    }

    /// Consumer that reads a contiguous sub-tile of `in_arg` at a dynamic offset
    /// (`construct_access_tile %vin[%c0]`, identity base_map) — the tiled edge
    /// increment 2 forwards via `tensor.extract_slice`. Produces a `tile`-sized
    /// result stored whole to `out_arg`.
    fn tiled_consumer(
        name: &str,
        in_arg: &str,
        out_arg: &str,
        shape: i64,
        tile: i64,
    ) -> IRFunction {
        IRFunction {
            name: name.to_string(),
            arguments: vec![
                (in_arg.to_string(), "index".into()),
                (out_arg.to_string(), "index".into()),
            ],
            grid: (1, 1, 1),
            return_type: None,
            operations: vec![
                Operation::new(Some("%c0"), "arith.constant", &[]).with_attr("value", Attr::Int(0)),
                Operation::new(Some("%vin"), "ktdp.construct_memory_view", &[in_arg])
                    .with_attr("shape", Attr::IntList(vec![shape]))
                    .with_attr("strides", Attr::IntList(vec![1]))
                    .with_attr("memory_space", Attr::Str("HBM".into()))
                    .with_attr("dtype", Attr::Str("f16".into())),
                // access tile at offset %c0, size `tile` (a sub-tile of the view).
                Operation::new(Some("%tin"), "ktdp.construct_access_tile", &["%vin", "%c0"])
                    .with_attr("shape", Attr::IntList(vec![tile])),
                Operation::new(Some("%loaded"), "ktdp.load", &["%tin"]),
                Operation::new(Some("%y"), "math.exp", &["%loaded"]),
                Operation::new(Some("%vout"), "ktdp.construct_memory_view", &[out_arg])
                    .with_attr("shape", Attr::IntList(vec![tile]))
                    .with_attr("strides", Attr::IntList(vec![1]))
                    .with_attr("memory_space", Attr::Str("HBM".into()))
                    .with_attr("dtype", Attr::Str("f16".into())),
                Operation::new(Some("%tout"), "ktdp.construct_access_tile", &["%vout"])
                    .with_attr("shape", Attr::IntList(vec![tile])),
                Operation::new(None, "ktdp.store", &["%y", "%tout"]),
                Operation::new(None, "func.return", &[]),
            ],
        }
    }

    fn module(funcs: Vec<IRFunction>) -> IRModule {
        let mut m = IRModule::default();
        for f in funcs {
            m.add_function(f);
        }
        m
    }

    /// a: src(1) -> t(2);  b: t(2) -> result(3).  t is a whole-tensor edge.
    fn two_node_spec() -> ProgramSpec {
        ProgramSpec {
            nodes: vec![
                NodeSpec {
                    func: "a".into(),
                    bindings: vec![
                        Binding {
                            arg: "%in".into(),
                            tensor: 1,
                            is_output: false,
                        },
                        Binding {
                            arg: "%out".into(),
                            tensor: 2,
                            is_output: true,
                        },
                    ],
                },
                NodeSpec {
                    func: "b".into(),
                    bindings: vec![
                        Binding {
                            arg: "%in".into(),
                            tensor: 2,
                            is_output: false,
                        },
                        Binding {
                            arg: "%out".into(),
                            tensor: 3,
                            is_output: true,
                        },
                    ],
                },
            ],
            sources: HashSet::from([1]),
            results: HashSet::from([3]),
        }
    }

    #[test]
    fn whole_tensor_edge_is_forwarded_no_hbm() {
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, true),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();

        // The intermediate t2's store AND load are gone: no HBM round-trip.
        let loads = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.load")
            .count();
        let stores = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.store")
            .count();
        assert_eq!(loads, 1, "only the source load survives");
        assert_eq!(stores, 1, "only the result store survives");

        // The fused function only needs the source (t1) + result (t3) pointers.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            arg_names,
            vec!["%t1_ptr", "%t3_ptr"],
            "no pointer for intermediate t2"
        );

        // b's exp consumes a's exp result directly (SSA forwarded).
        let b_exp = fused
            .operations
            .iter()
            .find(|o| o.op_type == "math.exp" && o.result.as_deref() == Some("%n1_y"))
            .expect("b's exp present");
        assert_eq!(
            b_exp.operands,
            vec!["%n0_y"],
            "b's exp reads a's stored SSA value"
        );
    }

    #[test]
    fn unsliceable_tiled_edge_falls_back_to_hbm() {
        // b reads a sub-tile with NO index operands (offsets empty) — not a
        // contiguous extract_slice we can place, so it stays an HBM round-trip.
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, false),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();
        let loads = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.load")
            .count();
        let stores = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.store")
            .count();
        let slices = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "tensor.extract_slice")
            .count();
        // a still stores t2, b still loads it (resident HBM within the fused fn).
        assert_eq!(loads, 2, "source + tiled intermediate load both kept");
        assert_eq!(stores, 2, "intermediate + result stores both kept");
        assert_eq!(
            slices, 0,
            "no extract_slice emitted for the unsliceable edge"
        );
        // The intermediate pointer is still a fused-function arg.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            arg_names.contains(&"%t2_ptr"),
            "intermediate kept as HBM arg: {arg_names:?}"
        );
    }

    #[test]
    fn tiled_edge_forwards_via_extract_slice() {
        // a writes t2 whole; b reads a contiguous sub-tile of t2 at offset %c0.
        // The edge forwards: a's store and b's load are gone, replaced by a
        // tensor.extract_slice of a's resident SSA value — no HBM round-trip.
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            tiled_consumer("b", "%in", "%out", 16, 8),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();

        // Only the source load (a) and the result store (b) survive.
        let loads = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.load")
            .count();
        let stores = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.store")
            .count();
        assert_eq!(loads, 1, "intermediate load replaced by extract_slice");
        assert_eq!(stores, 1, "intermediate store dropped (producer resident)");

        // The extract_slice reads a's stored value at the tile offset/size.
        let slice = fused
            .operations
            .iter()
            .find(|o| o.op_type == "tensor.extract_slice")
            .expect("extract_slice emitted for the tiled edge");
        assert_eq!(
            slice.operands,
            vec!["%n0_y"],
            "slices a's resident producer SSA"
        );
        assert_eq!(slice.result.as_deref(), Some("%n1_loaded"));
        assert_eq!(
            slice.attributes.get("slice_offsets"),
            Some(&Attr::StrList(vec!["%n1_c0".into()])),
            "offset is b's renamed index operand"
        );
        assert_eq!(
            slice.attributes.get("slice_sizes"),
            Some(&Attr::StrList(vec!["8".into()]))
        );
        assert_eq!(
            slice.attributes.get("slice_strides"),
            Some(&Attr::StrList(vec!["1".into()]))
        );

        // b's exp consumes the slice (downstream SSA lines up).
        let b_exp = fused
            .operations
            .iter()
            .find(|o| o.op_type == "math.exp" && o.result.as_deref() == Some("%n1_y"))
            .expect("b's exp present");
        assert_eq!(b_exp.operands, vec!["%n1_loaded"]);

        // No HBM pointer for the forwarded intermediate t2.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            arg_names,
            vec!["%t1_ptr", "%t3_ptr"],
            "no t2 pointer: {arg_names:?}"
        );
    }

    #[test]
    fn ssa_renaming_avoids_collisions() {
        // Both nodes use identical internal SSA names (%loaded, %y); after fusion
        // they must be distinct (prefixed).
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, true),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();
        let exps: Vec<&str> = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "math.exp")
            .filter_map(|o| o.result.as_deref())
            .collect();
        assert_eq!(exps, vec!["%n0_y", "%n1_y"], "node-prefixed, no collision");
    }

    // --- partial fusion: segment plan keeps attention nodes native ----------

    /// A head-parallel attention node: a multi-head grid plus the attention op
    /// signature (a `linalg.transpose` and the softmax `linalg.reduce {
    /// arith.maximumf }`). Reads `in_arg`, writes `out_arg`. Mirrors the model's
    /// `get_compute_tile_id` head select; the body is just enough to trip the
    /// detector.
    fn attn_node(name: &str, in_arg: &str, out_arg: &str, heads: usize) -> IRFunction {
        IRFunction {
            name: name.to_string(),
            arguments: vec![
                (in_arg.to_string(), "index".into()),
                (out_arg.to_string(), "index".into()),
            ],
            grid: (heads, 1, 1),
            return_type: None,
            operations: vec![
                Operation::new(Some("%hpid"), "ktdp.get_compute_tile_id", &[]),
                Operation::new(Some("%vin"), "ktdp.construct_memory_view", &[in_arg])
                    .with_attr("shape", Attr::IntList(vec![16]))
                    .with_attr("dtype", Attr::Str("f16".into())),
                Operation::new(Some("%tin"), "ktdp.construct_access_tile", &["%vin"])
                    .with_attr("shape", Attr::IntList(vec![16])),
                Operation::new(Some("%loaded"), "ktdp.load", &["%tin"]),
                Operation::new(Some("%kt"), "linalg.transpose", &["%loaded"]),
                Operation::new(Some("%mx"), "linalg.reduce", &["%kt"])
                    .with_attr("reduce_fn", Attr::Str("arith.maximumf".into())),
                Operation::new(Some("%vout"), "ktdp.construct_memory_view", &[out_arg])
                    .with_attr("shape", Attr::IntList(vec![16]))
                    .with_attr("dtype", Attr::Str("f16".into())),
                Operation::new(Some("%tout"), "ktdp.construct_access_tile", &["%vout"])
                    .with_attr("shape", Attr::IntList(vec![16])),
                Operation::new(None, "ktdp.store", &["%mx", "%tout"]),
                Operation::new(None, "func.return", &[]),
            ],
        }
    }

    /// A token-parallel matmul node: a multi-core grid but NO transpose/softmax —
    /// the GPU GEMM reconstruction runs it correctly at grid [1,1], so it must
    /// NOT be treated as attention.
    fn matmul_node(name: &str, in_arg: &str, out_arg: &str, cores: usize) -> IRFunction {
        let mut f = copy_node(name, in_arg, out_arg, 16, true);
        f.grid = (cores, 1, 1);
        f.operations.insert(
            0,
            Operation::new(Some("%pid"), "ktdp.get_compute_tile_id", &[]),
        );
        // Replace the math.exp with a linalg.matmul-shaped op (no softmax).
        for op in &mut f.operations {
            if op.op_type == "math.exp" {
                op.op_type = "linalg.matmul".to_string();
            }
        }
        f
    }

    #[test]
    fn detects_head_parallel_attention_node() {
        // Multi-head grid + transpose + softmax reduce = attention.
        assert!(is_attention_node(&attn_node("a", "%in", "%out", 9)));
        // Multi-core matmul (no transpose/softmax) = NOT attention.
        assert!(!is_attention_node(&matmul_node("m", "%in", "%out", 8)));
        // Plain elementwise copy at grid [1,1] = NOT attention.
        assert!(!is_attention_node(&copy_node("c", "%in", "%out", 16, true)));
        // Even WITH the attention op signature, a [1,1] grid (decode attention,
        // single token) stays fused — grid clause gates it out.
        let mut decode_attn = attn_node("d", "%in", "%out", 1);
        decode_attn.grid = (1, 1, 1);
        assert!(!is_attention_node(&decode_attn));
    }

    /// Program: src(1) -[copy a]-> t(2) -[attn b]-> t(3) -[copy c]-> result(4).
    /// The attention node sits between two non-attention nodes.
    fn three_node_attn_spec() -> ProgramSpec {
        ProgramSpec {
            nodes: vec![
                NodeSpec {
                    func: "a".into(),
                    bindings: vec![
                        Binding {
                            arg: "%in".into(),
                            tensor: 1,
                            is_output: false,
                        },
                        Binding {
                            arg: "%out".into(),
                            tensor: 2,
                            is_output: true,
                        },
                    ],
                },
                NodeSpec {
                    func: "b".into(),
                    bindings: vec![
                        Binding {
                            arg: "%in".into(),
                            tensor: 2,
                            is_output: false,
                        },
                        Binding {
                            arg: "%out".into(),
                            tensor: 3,
                            is_output: true,
                        },
                    ],
                },
                NodeSpec {
                    func: "c".into(),
                    bindings: vec![
                        Binding {
                            arg: "%in".into(),
                            tensor: 3,
                            is_output: false,
                        },
                        Binding {
                            arg: "%out".into(),
                            tensor: 4,
                            is_output: true,
                        },
                    ],
                },
            ],
            sources: HashSet::from([1]),
            results: HashSet::from([4]),
        }
    }

    #[test]
    fn plan_isolates_attention_into_native_segment() {
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            attn_node("b", "%in", "%out", 9),
            copy_node("c", "%in", "%out", 16, true),
        ]);
        let segs = plan_segments(&m, &three_node_attn_spec()).unwrap();
        // Three segments: [fused a], [native b], [fused c].
        assert_eq!(
            segs.len(),
            3,
            "one fused segment per non-attention run + native attn"
        );
        assert!(matches!(segs[0], Segment::Fused(_)), "node a fused");
        match &segs[1] {
            Segment::Native(n) => assert_eq!(n.func, "b", "attention node b stays native"),
            _ => panic!("expected native attention segment"),
        }
        assert!(matches!(segs[2], Segment::Fused(_)), "node c fused");

        // The boundary edges (t2 into attn, t3 out of attn) must remain HBM
        // pointer args on the adjacent fused segments — NOT forwarded as SSA.
        let Segment::Fused(seg_a) = &segs[0] else {
            unreachable!()
        };
        let a_args: Vec<&str> = seg_a
            .func
            .arguments
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(
            a_args.contains(&"%t2_ptr"),
            "t2 stays HBM out of segment a: {a_args:?}"
        );
        // t2 is a's boundary OUTPUT (consumed by the native attn node).
        assert!(
            seg_a.outputs.contains(&2),
            "t2 classified as segment a output"
        );
        assert!(
            seg_a.inputs.contains(&1),
            "t1 classified as segment a input"
        );
        let Segment::Fused(seg_c) = &segs[2] else {
            unreachable!()
        };
        let c_args: Vec<&str> = seg_c
            .func
            .arguments
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        assert!(
            c_args.contains(&"%t3_ptr"),
            "t3 stays HBM into segment c: {c_args:?}"
        );
        // t3 is c's boundary INPUT (produced by the native attn node); t4 output.
        assert!(
            seg_c.inputs.contains(&3),
            "t3 classified as segment c input"
        );
        assert!(
            seg_c.outputs.contains(&4),
            "t4 classified as segment c output"
        );
        // Each fused segment still keeps its own load/store (no cross-segment
        // SSA forwarding); the attention output round-trips HBM.
        assert!(
            seg_c.func.grid == (1, 1, 1),
            "fused segment runs at grid [1,1]"
        );
    }

    #[test]
    fn consecutive_non_attention_nodes_fuse_into_one_segment() {
        // a -> b -> c all non-attention: a single fused segment, with the
        // intermediate edges forwarded as SSA (no t2/t3 HBM pointers).
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, true),
            copy_node("c", "%in", "%out", 16, true),
        ]);
        let segs = plan_segments(&m, &three_node_attn_spec()).unwrap();
        assert_eq!(
            segs.len(),
            1,
            "one fused segment for the whole non-attention run"
        );
        let Segment::Fused(seg) = &segs[0] else {
            panic!("expected fused")
        };
        let args: Vec<&str> = seg.func.arguments.iter().map(|(n, _)| n.as_str()).collect();
        // Only the true source (t1) and result (t4) survive as HBM pointers; the
        // intra-segment edges t2/t3 forward as SSA.
        assert_eq!(
            args,
            vec!["%t1_ptr", "%t4_ptr"],
            "intra-run edges forwarded: {args:?}"
        );
    }

    // Contract (B): the shared cap-threshold predicate must exhaustively and
    // disjointly partition the cap axis at 7/8 of the LX budget, and fail safe
    // (stay naive) at a zero budget.
    #[test]
    fn attention_needs_flash_partitions_at_seven_eighths() {
        let lx = 2 * 1024 * 1024; // 2 MB
        let thresh = lx * 7 / 8;
        // Below the 7/8 cap → naive (head-batchable); at/above → flash.
        assert!(
            !attention_needs_flash(thresh - 1, lx),
            "just below cap stays naive"
        );
        assert!(attention_needs_flash(thresh, lx), "at cap flips to flash");
        assert!(attention_needs_flash(thresh + 1, lx), "above cap is flash");
        assert!(
            attention_needs_flash(usize::MAX, lx),
            "huge footprint can't wrap"
        );
        // Fail-safe: an unknown/zero budget never forces an FA path.
        assert!(
            !attention_needs_flash(usize::MAX, 0),
            "zero budget fails safe to naive"
        );
        assert!(!attention_needs_flash(0, lx), "empty scores never flash");
    }
}
