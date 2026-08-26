// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Experimental KTIR -> Metal Shading Language backend (`--features metal`).
//!
//! This is the *codegen* half of a GPU execution backend: it lowers a KTIR
//! `IRModule` to an MSL kernel string. It is pure string emission — no GPU, no
//! Metal bindings — so it builds and unit-tests anywhere, and the emitted shader
//! can be diffed/inspected directly. The runtime half (compile the MSL, dispatch
//! on a `MTLDevice`, read back, and validate against `interpreter::execute_function`
//! as the golden oracle) is a later slice that needs a real Metal device.
//!
//! SCOPE (slice 1): the per-tile **element-wise** pattern — the shape of a
//! Triton-style `vector_add`: load N tiles, apply one element-wise compute op,
//! store the result. Each GPU thread handles one element; the Spyre grid +
//! BLOCK_SIZE collapse into a flat `thread_position_in_grid`. This is the GPU
//! "hello world" and proves the IR->MSL pipeline end to end.
//!
//! NOT YET (the roadmap): multi-op fusion, `linalg.matmul` (-> MPS), reductions
//! (threadgroup memory), cross-core comm (the global-sync problem), distributed
//! and indirect access. Each is its own slice.

use std::collections::{HashMap, HashSet};

use crate::dtypes::DType;
use crate::ir::{Attr, IRFunction, IRModule, Operation};

// =========================================================================
// Matmul acceleration tier — the GPU analogue of the BLAS auto-select.
//
// On Apple Silicon a GEMM can run three ways, best-first:
//   * Nax       — `mpp::tensor_ops::matmul2d` (Metal Performance Primitives),
//                 which drives the M5+ Neural Accelerators. NOT engaged
//                 automatically by MPS — it must be written in the shader.
//   * Simdgroup — `simdgroup_matrix<T,8,8>` + `simdgroup_multiply_accumulate`,
//                 the matrix instructions on Apple7+ (M1..M4) GPUs.
//   * Naive     — a plain per-element loop. The portable floor (also non-Apple).
//
// We pick the highest tier the device supports (capability), capped by the
// highest tier whose kernel we actually emit today (`HIGHEST_IMPLEMENTED`), so
// the backend degrades gracefully as the accelerated kernel slices land —
// exactly like the BLAS providers degrade to the naive matmul.
// =========================================================================

/// GPU matmul acceleration tier, ordered worst -> best.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatmulTier {
    Naive,
    Simdgroup,
    Nax,
}

/// The highest tier whose kernel we actually emit today. The general tiled NAX
/// GEMM ([`NaxGemm`] / [`run_nax_matmul`], `mpp::tensor_ops::matmul2d`) is
/// implemented and validated on the M5, so this is `Nax`. Note the tiers are
/// NOT all implemented in order — `Simdgroup` has no kernel yet — so
/// [`effective_matmul_tier`] selects the best *implemented* tier a device
/// supports rather than a simple linear cap (see [`tier_implemented`]).
pub const HIGHEST_IMPLEMENTED: MatmulTier = MatmulTier::Nax;

/// Whether a tier's GPU kernel is emitted today. `Naive` (the CPU/BLAS floor)
/// and `Nax` (the M5 tensor engine) are implemented; the pre-NAX `Simdgroup`
/// (`simdgroup_matrix`) kernel is a future slice, so a non-NAX Apple GPU falls
/// back to `Naive` rather than claiming a tier we can't run.
pub fn tier_implemented(tier: MatmulTier) -> bool {
    // All three are implemented now: Naive (CPU/BLAS floor), Simdgroup
    // (simdgroup_float8x8, M1–M4), and Nax (matmul2d, M5+).
    matches!(
        tier,
        MatmulTier::Naive | MatmulTier::Simdgroup | MatmulTier::Nax
    )
}

/// The matmul tier a Metal device *supports*, parsed from its name (mirrors
/// scratchy's `detect_device` name-parse → `AppleSiliconGen` → `is_nax_capable`):
///   * Apple `M5`+  -> Nax (Apple9 gen 17+, first with the Neural Accelerator)
///   * any other Apple GPU (M1..M4, Apple7+) -> Simdgroup
///   * non-Apple / unknown -> Naive
pub fn device_matmul_tier(device_name: &str) -> MatmulTier {
    if let Some(generation) = apple_m_generation(device_name) {
        return if generation >= 5 {
            MatmulTier::Nax
        } else {
            MatmulTier::Simdgroup
        };
    }
    if device_name.contains("Apple") {
        // An Apple GPU we couldn't pin to an M-number — assume Apple7+ matrix units.
        return MatmulTier::Simdgroup;
    }
    MatmulTier::Naive
}

/// The tier actually used for a device: the highest *implemented* tier that
/// the device's capability supports. Because `Simdgroup` isn't implemented yet,
/// a pre-NAX Apple GPU (capability `Simdgroup`) resolves to `Naive`, while an
/// M5 (capability `Nax`) resolves to `Nax`.
pub fn effective_matmul_tier(device_name: &str) -> MatmulTier {
    let cap = device_matmul_tier(device_name);
    [MatmulTier::Nax, MatmulTier::Simdgroup, MatmulTier::Naive]
        .into_iter()
        .find(|&t| t <= cap && tier_implemented(t))
        .unwrap_or(MatmulTier::Naive)
}

/// Which matmul implementation to dispatch for a given problem on a given
/// device. NAX is the M5 GPU tensor engine (f16); Accelerate is Apple's AMX
/// matrix coprocessor (f32). See [`choose_matmul_backend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulBackend {
    /// Apple Accelerate `cblas_sgemm` (AMX, f32). Used for tiny GEMMs and on
    /// non-Apple GPUs.
    Accelerate,
    /// The `simdgroup_float8x8` matrix GEMM — the GPU path on M1–M4 (pre-NAX).
    Simdgroup,
    /// The NAX `matmul2d` tensor-engine GEMM (f16) — the GPU path on M5+.
    Nax,
}

impl MatmulBackend {
    /// Whether this backend runs on the Metal GPU (`NaxGemm` context) vs the CPU.
    pub fn is_gpu(self) -> bool {
        matches!(self, MatmulBackend::Nax | MatmulBackend::Simdgroup)
    }
}

/// Minimum 128×256 output blocks before routing a matmul to the GPU. Measured
/// wall-clock crossover on the M5 is ~1024³ (= 32 blocks): NAX *compute* matches
/// or beats AMX from ~512³ (2739 vs 1498 GFLOP/s), but every GPU dispatch pays a
/// ~300 µs command-buffer submission round-trip that a single AMX call doesn't,
/// so only GEMMs large enough to dwarf that latency win. Below the crossover —
/// including every LX-sized tile (≤418³) a real KTIR program produces —
/// Accelerate is faster, so the gate routes there. (Batching many matmuls into
/// one submission, via [`NaxGemm::run_chain`], is how small GEMMs would win.)
pub const NAX_MIN_BLOCKS: usize = 32;
/// Minimum K depth before routing to the GPU. Same calibration.
pub const NAX_MIN_K: usize = 256;

/// Choose the matmul backend for `C(m×k·k×n)` on the named device.
///
/// NAX only exists on M5+ *and* only helps at scale: the M5 throughput sweep
/// shows small GEMMs are far slower on the GPU than on Accelerate's AMX (e.g.
/// 256³ ≈ 0.19 TFLOP/s vs AMX's ~2), because too few output blocks leave the
/// cores idle. So we route to NAX only when there are enough blocks to fill the
/// GPU ([`NAX_MIN_BLOCKS`]) and K is deep enough to amortize ([`NAX_MIN_K`]);
/// otherwise, and on every non-NAX device, we use Accelerate.
///
/// Note the backends differ in precision (NAX f16 vs Accelerate f32), so this
/// gate belongs to the experimental Metal/Spyre-faithful execution path, NOT
/// the f32 parity interpreter (which always uses Accelerate via `blas.rs`).
/// The threshold assumes GPU-resident operands; a one-shot host call pays
/// copy/readback that pushes the crossover higher (fusion keeps data resident
/// and lowers it back down).
pub fn choose_matmul_backend(device_name: &str, m: usize, k: usize, n: usize) -> MatmulBackend {
    // Only the NAX tensor engine on M5+, and only for GEMMs large enough to
    // beat the GPU submission latency, goes to the GPU. The pre-M5
    // `simdgroup_float8x8` path has lower compute throughput than AMX AND pays
    // the same submission latency, so it never wins wall-clock — M1–M4 (and any
    // smaller GEMM) use Accelerate, which is genuinely the fastest matmul there.
    //
    // GATE OVERRIDE (`KTIR_FORCE_GPU_GEMM` / `KTIR_NAX_MIN_BLOCKS` /
    // `KTIR_NAX_MIN_K`): the wall-clock size gate is a PERFORMANCE choice, not a
    // correctness one. The differential GPU-conformance harness must route the
    // small TILED example matmuls (e.g. 32×512×128) onto the Metal GEMM to check
    // NAX/simdgroup numerics against the Python interpreter — those tiles are
    // below the production crossover and would otherwise silently run on AMX.
    // `KTIR_FORCE_GPU_GEMM=1` forces the GPU branch for ANY GEMM on a GPU-capable
    // device (the best implemented tier the device supports — NAX on M5+,
    // simdgroup on M1–M4); the two `KTIR_NAX_MIN_*` env vars lower the thresholds
    // without fully bypassing them. None of these change the math, only WHICH
    // engine runs it.
    let tier = effective_matmul_tier(device_name);
    if force_gpu_gemm() {
        return match tier {
            MatmulTier::Nax => MatmulBackend::Nax,
            MatmulTier::Simdgroup => MatmulBackend::Simdgroup,
            MatmulTier::Naive => MatmulBackend::Accelerate,
        };
    }
    let blocks = m.div_ceil(128) * n.div_ceil(256);
    let big_enough = blocks >= nax_min_blocks() && k >= nax_min_k();
    match tier {
        MatmulTier::Nax if big_enough => MatmulBackend::Nax,
        _ => MatmulBackend::Accelerate,
    }
}

/// Whether `KTIR_FORCE_GPU_GEMM` is set (any non-empty value): force EVERY GEMM
/// onto the Metal GPU engine on a GPU-capable device, bypassing the wall-clock
/// size gate. Used by the GPU differential-conformance harness to exercise the
/// NAX/simdgroup matmul kernel on the small tiled example matmuls; never on by
/// default (it is slower for tiny GEMMs — purely a correctness-check override).
pub fn force_gpu_gemm() -> bool {
    std::env::var_os("KTIR_FORCE_GPU_GEMM").is_some_and(|v| !v.is_empty())
}

/// The 128×256-block GPU routing threshold, env-overridable via
/// `KTIR_NAX_MIN_BLOCKS` (else [`NAX_MIN_BLOCKS`]). Lets the harness lower the
/// gate to push small tiled matmuls onto the GPU without the all-or-nothing
/// [`force_gpu_gemm`] flag.
pub fn nax_min_blocks() -> usize {
    std::env::var("KTIR_NAX_MIN_BLOCKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(NAX_MIN_BLOCKS)
}

/// The K-depth GPU routing threshold, env-overridable via `KTIR_NAX_MIN_K` (else
/// [`NAX_MIN_K`]). See [`nax_min_blocks`].
pub fn nax_min_k() -> usize {
    std::env::var("KTIR_NAX_MIN_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(NAX_MIN_K)
}

/// Parse the `M<n>` generation from an Apple GPU name like `"Apple M5 Pro"`.
/// Returns `None` for non-Apple-Silicon names. Forward-compatible: an `M6`
/// reads as 6 (>= 5 -> Nax), unlike scratchy's fixed M1..M5 match.
fn apple_m_generation(name: &str) -> Option<u32> {
    let rest = name.split('M').nth(1)?; // text after the first 'M'
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    (!digits.is_empty() && name.contains("Apple"))
        .then(|| digits.parse().ok())
        .flatten()
}

/// One kernel-argument buffer: the KTIR pointer-arg name, whether it's written,
/// and its element dtype. The order of [`MslKernel::buffers`] is the MSL
/// `[[buffer(i)]]` binding order — the runtime must supply data in this order.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferBinding {
    pub name: String,
    pub is_output: bool,
    pub dtype: DType,
}

/// A lowered Metal kernel: the MSL source, the kernel name, and its buffer
/// bindings in `[[buffer(i)]]` order.
#[derive(Clone, Debug)]
pub struct MslKernel {
    pub source: String,
    pub name: String,
    pub buffers: Vec<BufferBinding>,
}

/// Lower `func_name` to an MSL kernel string. Errors if the function isn't the
/// supported element-wise shape (with a message pointing at what tripped it).
pub fn emit_msl(module: &IRModule, func_name: &str) -> Result<String, String> {
    Ok(emit_kernel(module, func_name)?.source)
}

// =========================================================================
// Kernel scheduling — partition a fused function's op stream into kernels.
//
// The ktir-optimizer fuses a whole program into one SSA function (no
// inter-function HBM). MLX-style, we then carve that op stream into kernels:
// maximal windows of fusable "map" ops (one fused MSL kernel each), with
// reductions and matmuls as their own kernels (they need different templates /
// the GEMM path). Windows are capped at MAX_KERNEL_WINDOW so a single kernel
// never grows unbounded (register pressure), matching MLX's bounded subgraphs.
// =========================================================================

/// Max map ops fused into one kernel before forcing a new window. MLX-style
/// bounded subgraphs — keeps register pressure and compile time in check.
pub const MAX_KERNEL_WINDOW: usize = 40;

/// One scheduled unit of a fused function's top-level op stream. Indices point
/// into the function body; the loads/stores feeding a region are traced by the
/// emitter (they are plumbing, not separately scheduled).
#[derive(Debug, PartialEq, Eq)]
pub enum KernelRegion {
    /// A maximal run of fusable map ops -> one fused MSL kernel.
    Map(Vec<usize>),
    /// A single reduction op -> its own (reduction) kernel.
    Reduce(usize),
    /// A single matmul -> a GEMM dispatch.
    Matmul(usize),
    /// An `scf.for` K-loop recognized as a single GEMM (the Spyre K-tiling and
    /// grid/M decomposition collapse into one full-shape matmul). Covers decode
    /// (M=1) and prefill (M=8) uniformly — the M comes from the operand's full
    /// view/producer shape, not the per-iteration tile.
    MatmulLoop(MatmulLoopInfo),
}

/// A K-loop matmul collapsed to one GEMM: `out[m,n] = A[m,k] @ B[k,n]`, with the
/// resident-buffer SSA roots of the full A and B tensors. `a_root`/`b_root` are
/// the SSA/pointer names the executor looks up (A is typically a forwarded
/// `tensor.extract_slice` source — the resident activation; B a weight load).
///
/// `n` is the OUTPUT width this loop computes (the matmul's per-iteration N). When
/// the program tiles the output N dimension across several sequential K-loops
/// (Llama-1B's lm_head: `[m,2048]@[2048,128256]` split into 16384-wide column
/// tiles), `n_off`/`b_stride` describe B as a COLUMN SLICE of a wider weight:
/// this loop's B is `B_full[k, n_off : n_off+n]`, where `B_full` has row stride
/// `b_stride` (the full weight width). For the common case (no N-tiling) `n_off=0`
/// and `b_stride=n` (B is contiguous and the slice is the whole tensor).
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MatmulLoopInfo {
    pub m: i64,
    pub k: i64,
    pub n: i64,
    pub a_root: String,
    pub b_root: String,
    pub out_ssa: String,
    /// Column offset of this loop's B slice within the full weight (0 = no tiling).
    pub n_off: i64,
    /// Row stride of the full weight B occupies (== `n` when B is contiguous).
    pub b_stride: i64,
    /// transpose-B: B is stored `[n, k]` (on-disk PyTorch `Linear`
    /// `[out, in]`), contracted over its LAST axis (`xWᵀ`). The weight binds
    /// VERBATIM — no transpose, no gather, no copy beyond the normal weight-cache
    /// upload. The GEMM reads `[n,k]` directly via the backend's native transpose-B
    /// (AMX `cblas` `CblasTrans`; NAX/simdgroup via the B-staging variant), chosen
    /// by the same `use_nax` gate. Plain `linalg.matmul` (B `[k, n]`) is `false`.
    pub transpose_b: bool,
    /// Row offset into the FULL activation A for the reconstructed `[m,k]` read
    /// (0 = read from A's stick base, the default). Set non-zero only when the A
    /// operand's access tile carries a STATIC constant row index (not the grid
    /// `pid`) — the "last-token-only" rewrite pins the row to `m-1` so the offload
    /// reconstructs a single-row GEMM over exactly that activation row. The offload
    /// reads A from `base + m_row_off * k` elements; default 0 leaves every other
    /// GEMM's full-M reconstruction (read from the stick base) untouched.
    pub m_row_off: i64,
}

/// How an op participates in scheduling.
enum OpClass {
    /// Fuses into a map kernel (elementwise / cast / broadcast).
    Map,
    /// A reduction — its own kernel, and a window boundary.
    Reduce,
    /// A matmul — its own GEMM dispatch, and a window boundary.
    Matmul,
    /// Dataflow/init plumbing the emitter traces through: never its own kernel,
    /// never a window output, does not break a window (loads, views, tiles,
    /// stores, constants, splat/empty inits, returns).
    Plumbing,
    /// Can't fuse (scf.*, comm, anything unrecognized) — forces a wholesale
    /// interpreter fallback for the function.
    Boundary,
}

fn classify(op: &Operation) -> OpClass {
    match op.op_type.as_str() {
        "arith.addf" | "arith.subf" | "arith.mulf" | "arith.divf" | "arith.maximumf"
        | "arith.maxf" | "arith.minimumf" | "arith.minf" | "arith.negf" | "arith.absf"
        | "math.absf" | "math.exp" | "math.log" | "math.sqrt" | "math.sin" | "math.cos"
        | "math.tanh" | "arith.extf" | "arith.truncf" | "linalg.add" | "linalg.mul"
        | "linalg.sub" | "linalg.broadcast" => OpClass::Map,
        "linalg.reduce" => OpClass::Reduce,
        "linalg.matmul" => OpClass::Matmul,
        // Folded/plumbing: constants & splat fold into expressions; empty is an
        // init shape hint; the ktdp.* memory ops + return are traced, not scheduled.
        "arith.constant" | "tensor.splat" | "tensor.empty" | "func.return" => OpClass::Plumbing,
        s if s.starts_with("ktdp.") => OpClass::Plumbing,
        _ => OpClass::Boundary,
    }
}

/// Partition a fused function's top-level ops into a kernel schedule. Greedily
/// grows a map window until a boundary (reduce/matmul) flushes it or it hits the
/// size cap; reductions and matmuls become their own regions. Returns `Err` if
/// the function contains an unfusable op (scf.for, comm, …) so the caller falls
/// back to the interpreter wholesale — the resident-buffer GPU path handles only
/// fully map/reduce/matmul functions for now.
pub fn plan_kernels(ops: &[Operation]) -> Result<Vec<KernelRegion>, String> {
    let defs = def_map_all(ops);
    let mut regions: Vec<KernelRegion> = Vec::new();
    let mut window: Vec<usize> = Vec::new();
    let flush = |w: &mut Vec<usize>, r: &mut Vec<KernelRegion>| {
        if !w.is_empty() {
            r.push(KernelRegion::Map(std::mem::take(w)));
        }
    };
    for (i, op) in ops.iter().enumerate() {
        // An scf.for is fusable ONLY if it's a recognizable matmul K-loop;
        // otherwise it's a hard boundary (the function falls back).
        if op.op_type == "scf.for" {
            match recognize_matmul_loop(op, &defs) {
                Some(info) => {
                    flush(&mut window, &mut regions);
                    regions.push(KernelRegion::MatmulLoop(info));
                    continue;
                }
                None => {
                    return Err(
                        "metal: scf.for is not a recognizable matmul K-loop — function \
                         falls back to the interpreter"
                            .to_string(),
                    );
                }
            }
        }
        match classify(op) {
            OpClass::Map => {
                window.push(i);
                if window.len() >= MAX_KERNEL_WINDOW {
                    flush(&mut window, &mut regions);
                }
            }
            OpClass::Reduce => {
                flush(&mut window, &mut regions);
                regions.push(KernelRegion::Reduce(i));
            }
            OpClass::Matmul => {
                flush(&mut window, &mut regions);
                regions.push(KernelRegion::Matmul(i));
            }
            OpClass::Plumbing => {} // traced by the emitter; doesn't break a window
            OpClass::Boundary => {
                return Err(format!(
                    "metal: op '{}' is not fusable — function falls back to the interpreter",
                    op.op_type
                ));
            }
        }
    }
    flush(&mut window, &mut regions);
    Ok(regions)
}

/// Map of `scf.for` result SSA (as written, with `%`) -> the GEMM it collapses
/// to. Computed once per fused function; the interpreter consults it to offload
/// each recognized K-loop to one GPU GEMM instead of running the loop.
pub fn matmul_loop_schedule(ops: &[Operation]) -> HashMap<String, MatmulLoopInfo> {
    let defs = def_map_all(ops);
    let mut sched = HashMap::new();
    for op in ops {
        if op.op_type == "scf.for"
            && let Some(info) = recognize_matmul_loop(op, &defs)
        {
            sched.insert(info.out_ssa.clone(), info);
        }
    }
    sched
}

#[cfg(metal)]
thread_local! {
    /// One GEMM engine per scheduler thread (the cooperative core scheduler is
    /// single-threaded), compiled once and reused across all K-loops.
    static GEMM_ENGINE: std::cell::OnceCell<Option<NaxGemm>> = const { std::cell::OnceCell::new() };

    /// Resident WEIGHT cache: a GEMM's constant weight operand (an HBM pointer)
    /// decoded f16->f32 and uploaded to a [`UnifiedBuffer`] EXACTLY ONCE, then
    /// reused across every pass. Weights are identical across the autoregressive
    /// decode loop / the bench loop, so re-decoding+re-uploading them each pass
    /// (e.g. the lm_head [576,49152] = 113 MB) was pure repeated work — the
    /// data-movement bottleneck this cache eliminates.
    ///
    /// SAFETY (correctness): keyed by [`WeightKey`] = `(root SSA name, element
    /// count, content fingerprint)`, NOT by name alone. The fingerprint is a hash
    /// of a fixed strided SAMPLE of the operand's raw HBM bytes (see
    /// [`weight_fingerprint`]), so different weight *data* bound to the same SSA
    /// name (a different model reusing `%t..._ptr`, or weights mutated in place)
    /// yields a different key and forces a refresh. This makes the cache immune to
    /// the stale-weight hazard a name-only cache would have. Only resolves for
    /// HBM-pointer operands (constant weights); resident activation tiles change
    /// every pass and are NEVER cached.
    ///
    /// `Rc` so a hit hands out a cheap clone (the buffer stays owned by the cache
    /// and outlives the GEMM dispatch). Bounded by [`WEIGHT_CACHE_MAX`] entries.
    static WEIGHT_CACHE: std::cell::RefCell<HashMap<WeightKey, std::rc::Rc<UnifiedBuffer>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Identity of a cached resident weight buffer. A match on all three fields means
/// the SAME data (same name, same length, same content sample) — safe to reuse.
/// A mismatch on ANY field (notably the content fingerprint) is a different
/// weight and forces a decode+upload refresh, so a stale weight can never be
/// served. See [`WEIGHT_CACHE`].
#[cfg(metal)]
#[derive(Clone, PartialEq, Eq, Hash)]
struct WeightKey {
    root: String,
    len: usize,
    fingerprint: u64,
    /// Column offset of the cached slice within the full weight (0 = whole tensor /
    /// contiguous). Distinguishes the N-tile slices of a single N-tiled weight (the
    /// lm_head's 8 column tiles share `root` but differ here) so they cache apart.
    col_off: i64,
    /// Element type of the cached buffer: `true` for an f16 (half) weight buffer
    /// (the `KTIR_F16_WEIGHTS` path), `false` for f32. Keeps the two encodings of
    /// the same weight from colliding when a process A/B-toggles the flag.
    f16: bool,
}

/// Max distinct weight buffers held resident at once. SmolLM2-135M has on the
/// order of ~100 weight tensors; this bounds memory if a long-running process
/// cycles through many distinct weights. On overflow the cache is cleared (a
/// simple, correct eviction — the next pass repopulates the working set).
#[cfg(metal)]
const WEIGHT_CACHE_MAX: usize = 512;

/// Whether to store the resident GPU WEIGHT (B) operand as f16 (half) rather than
/// f32 — the weight-streaming win. ON by default; `KTIR_NO_F16_WEIGHTS` disables it.
/// Only ACTUALLY used when the engine compiled the f16-B pipelines (NAX devices) —
/// see the `has_f16_b_pipelines()` guard at the call site, which keeps non-NAX
/// Metal devices on f32 (no missing-pipeline panic). AMX/simdgroup always use f32 B.
#[cfg(metal)]
pub fn f16_weights_enabled() -> bool {
    std::env::var_os("KTIR_NO_F16_WEIGHTS").is_none()
}

/// Number of f32 samples taken to fingerprint a weight operand. Enough spread
/// (first, last, and strided interior elements) that two different weight tensors
/// of the same shape collide only with astronomically low probability, while
/// staying O(SAMPLES) — negligible vs the full decode+upload it guards.
#[cfg(metal)]
const WEIGHT_FP_SAMPLES: usize = 64;

/// Content fingerprint of an HBM weight operand: hash a fixed strided sample of
/// its raw backing bytes (NOT a full decode). We sample the first and last
/// elements plus [`WEIGHT_FP_SAMPLES`] evenly-spaced interior elements, reading
/// each element's raw `dtype` bytes straight from HBM and folding them into a
/// `DefaultHasher` together with `n` and the dtype size. Hashing raw bytes (vs
/// decoded f32) avoids decoding the whole tensor just to key it, yet still
/// distinguishes any two operands whose data differs at a sampled position — the
/// staleness guard. (A weight that differs ONLY at unsampled positions is the
/// pathological miss; the dense, spread-out sampling makes that vanishingly
/// unlikely for real tensors, and the `len` term catches any shape change.)
#[cfg(metal)]
fn weight_fingerprint(
    hbm: &crate::memory::HBMSimulator,
    byte_addr: i64,
    n: usize,
    dtype: DType,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let bpe = dtype.bytes_per_elem();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    n.hash(&mut h);
    bpe.hash(&mut h);
    // Trusted (resident) weights are immutable, so the content sample can't change
    // pass to pass — skip it. The WeightKey's name+len+col_off already identify a
    // resident weight uniquely, and the cache is cleared whenever sources change.
    if TRUSTED_WEIGHTS.with(std::cell::Cell::get) {
        return h.finish();
    }
    if n == 0 {
        return h.finish();
    }
    // Element indices to sample: 0, last, and WEIGHT_FP_SAMPLES strided interior
    // points. Stepping at least 1 so a small tensor still terminates.
    let last = n - 1;
    let step = (n / WEIGHT_FP_SAMPLES.max(1)).max(1);
    // Resolve the weight's backing allocation ONCE, then index each sample within
    // it. The samples are offsets INTO the weight (never an allocation base), so a
    // per-sample lookup would hit `find_allocation`'s O(num-weights) linear scan —
    // ×~66 samples ×~113 GEMMs every pass. `(buf, base_off)` borrows the buffer.
    let region = hbm.allocation_at(byte_addr);
    let mut idx = 0usize;
    loop {
        let elem = idx.min(last);
        // Hash this element's raw bytes (zero past the allocation end, matching
        // `read_bytes`'s zero-pad), read straight from the borrowed buffer.
        if let Some((buf, base_off)) = region {
            let off = base_off + elem * bpe;
            for j in 0..bpe {
                buf.get(off + j).copied().unwrap_or(0).hash(&mut h);
            }
        } else {
            // Address in no allocation: all-zero bytes (same as read_bytes).
            for _ in 0..bpe {
                0u8.hash(&mut h);
            }
        }
        if elem == last {
            break;
        }
        idx += step;
    }
    h.finish()
}

/// Count of K-loops offloaded to a **NAX** (GPU) GEMM — test/telemetry proof the
/// fused path used the tensor engine, not a silent interpreter fallback.
#[cfg(metal)]
pub static MATMUL_LOOP_GPU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Count of `linalg.matmul`/GEMV ops dispatched to the Metal GPU GEMM through the
/// per-op [`metal_gemm_or_blas`] / [`metal_gemv_or_blas`] selector (the
/// `execute_function` per-tile path the example kernels take). PROOF for the GPU
/// differential-conformance harness that a tiled example matmul actually ran on
/// NAX/simdgroup, not the AMX fallback — a false pass is a program that secretly
/// stayed on Accelerate. Incremented only when the GPU branch is taken AND the
/// engine returned a result; an engine failure that falls through to BLAS does
/// NOT bump it. See [`gemm_or_blas_gpu_count`] / [`reset_gemm_or_blas_gpu_count`].
#[cfg(metal)]
pub static GEMM_OR_BLAS_GPU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Read the [`GEMM_OR_BLAS_GPU_COUNT`] proof counter.
#[cfg(metal)]
pub fn gemm_or_blas_gpu_count() -> usize {
    GEMM_OR_BLAS_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Reset the [`GEMM_OR_BLAS_GPU_COUNT`] proof counter (per-case harness hook so
/// each program can assert its own GEMMs ran on the GPU).
#[cfg(metal)]
pub fn reset_gemm_or_blas_gpu_count() {
    GEMM_OR_BLAS_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// A per-case snapshot of every Metal OFFLOAD proof counter, so the resident /
/// segmented differential harness can prove WHICH offload(s) a program actually
/// fired on this GPU — not just "a GEMM ran", but the full breakdown:
///
/// * `matmul_loop_gpu` — K-loop GEMMs reconstructed onto **NAX** (the resident /
///   segmented fused path's full-M GEMM, the production matmul offload).
/// * `matmul_loop_amx` — full-M K-loop GEMMs the size gate routed to **AMX**
///   (Accelerate) instead of NAX: still a resident offload (NOT the interpreter
///   scf.for fallback), just CPU-side. Counted so a "GEMM offload fired" assertion
///   does not falsely fail when the small example GEMM legitimately picks AMX.
/// * `gemm_or_blas_gpu` — per-op `linalg.matmul`/GEMV dispatched through the
///   [`metal_gemm_or_blas`] selector (the `execute_function` per-tile path).
/// * `map_region_gpu` — fused Map-window elementwise kernels run on the GPU.
///
/// "A GEMM/attention program hit a Metal offload" == `matmul_loop_gpu +
/// matmul_loop_amx + gemm_or_blas_gpu > 0`; a fused-attention program additionally
/// fires GEMMs via these same counters (the GEMV·softmax·GEMV it expands to).
#[cfg(metal)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OffloadProof {
    pub matmul_loop_gpu: usize,
    pub matmul_loop_amx: usize,
    pub gemm_or_blas_gpu: usize,
    pub map_region_gpu: usize,
}

/// Zero EVERY offload proof counter (per-case harness hook). Call before a case;
/// read [`offload_proof`] after to attribute the offloads to THAT case.
#[cfg(metal)]
pub fn reset_offload_proof() {
    use std::sync::atomic::Ordering::Relaxed;
    MATMUL_LOOP_GPU_COUNT.store(0, Relaxed);
    MATMUL_LOOP_AMX_COUNT.store(0, Relaxed);
    GEMM_OR_BLAS_GPU_COUNT.store(0, Relaxed);
    MAP_REGION_GPU_COUNT.store(0, Relaxed);
}

/// Snapshot every offload proof counter (see [`OffloadProof`]).
#[cfg(metal)]
pub fn offload_proof() -> OffloadProof {
    use std::sync::atomic::Ordering::Relaxed;
    OffloadProof {
        matmul_loop_gpu: MATMUL_LOOP_GPU_COUNT.load(Relaxed),
        matmul_loop_amx: MATMUL_LOOP_AMX_COUNT.load(Relaxed),
        gemm_or_blas_gpu: GEMM_OR_BLAS_GPU_COUNT.load(Relaxed),
        map_region_gpu: MAP_REGION_GPU_COUNT.load(Relaxed),
    }
}

/// Count of recognized full-M K-loops run on the **AMX** (Accelerate) backend
/// instead of NAX — the size-gated alternative for small GEMMs (low `k·n`) that
/// would only underfill the GPU. These are still full-M resident offloads (NOT
/// the interpreter scf.for fallback); they read the SAME resident f32 operands as
/// the NAX path. Decode small GEMMs (m==1) still use the interpreter K-loop and
/// are counted in neither.
#[cfg(metal)]
pub static MATMUL_LOOP_AMX_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Number of resident-weight-cache HITS — a weight operand served from a cached
/// `UnifiedBuffer` instead of re-decoded+re-uploaded. Test/telemetry proof the
/// cache is doing work (the 2nd+ pass of a multi-pass run should be nearly all
/// hits). Paired with [`WEIGHT_CACHE_MISSES`].
#[cfg(metal)]
pub static WEIGHT_CACHE_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Number of resident-weight-cache MISSES — a weight decoded+uploaded fresh
/// (first sight, changed data, or an evicted entry). See [`WEIGHT_CACHE_HITS`].
#[cfg(metal)]
pub static WEIGHT_CACHE_MISSES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Clear the resident weight cache (test hook / memory-pressure relief). The next
/// pass repopulates whatever weights it actually touches.
#[cfg(metal)]
pub fn clear_weight_cache() {
    WEIGHT_CACHE.with(|c| c.borrow_mut().clear());
}

/// Parse a GEMM-operand SSA root (`%t<id>_ptr`, `t<id>`, `%t<id>`) to its tensor id.
/// STRICT: the whole remainder after `t` (minus an optional `_ptr`) must be digits,
/// so a non-`t<id>` root (e.g. `%view0`) returns `None` rather than a wrong id — a
/// wrong id could mis-classify a mutable operand as immutable and serve a stale
/// buffer. `None` is always safe (the entry is dropped, just re-decoded).
#[cfg(metal)]
fn weight_root_tid(root: &str) -> Option<u64> {
    let s = root.trim_start_matches('%').strip_prefix('t')?;
    let s = s.strip_suffix("_ptr").unwrap_or(s);
    s.parse().ok()
}

/// Keep ONLY the cached GPU weight buffers that are provably IMMUTABLE — keyed on a
/// tid that is written by NEITHER the forward pass (`forward_written`) NOR this
/// `set_sources` call (`just_set`). Every HBM mutation goes through exactly one of
/// those two paths, so a kept buffer's bytes cannot have changed since it was cached
/// — it can never be stale. Everything else (KV cache, re-set inputs, and any root
/// we can't parse to a tid) is dropped and re-decoded. This keeps the ~2 GB of
/// constant model weights resident across decode steps (fixing the per-token
/// re-decode) without ever serving a stale weight.
#[cfg(metal)]
pub fn retain_resident_weights(
    forward_written: &std::collections::HashSet<u64>,
    just_set: &std::collections::HashSet<u64>,
) {
    WEIGHT_CACHE.with(|c| {
        c.borrow_mut().retain(|key, _| {
            weight_root_tid(&key.root)
                .is_some_and(|tid| !forward_written.contains(&tid) && !just_set.contains(&tid))
        })
    });
}

#[cfg(metal)]
thread_local! {
    /// When set, [`weight_fingerprint`] skips its per-pass content sampling and the
    /// weight cache keys on name+len+col_off ALONE. The resident executor sets this
    /// during a forward pass: its weights are uploaded ONCE (and the cache is
    /// cleared on each `set_sources`), so the content fingerprint — which exists to
    /// catch a weight whose bytes changed under the same name — can never differ and
    /// is pure per-pass overhead. Off by default (the general path still verifies).
    static TRUSTED_WEIGHTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Enable/disable trusting resident weights as immutable (skips the weight-cache
/// content fingerprint). Returns the previous value so callers can restore it.
#[cfg(metal)]
pub fn set_trusted_weights(on: bool) -> bool {
    TRUSTED_WEIGHTS.with(|c| c.replace(on))
}

/// Default minimum GEMM WEIGHT size (`k·n` elements) to run a recognized full-M
/// K-loop on **NAX** (the GPU tensor engine) rather than **AMX** (Accelerate).
///
/// Both backends compute the SAME reconstructed full-M GEMM over the SAME resident
/// f32 operands — this is purely a per-GEMM speed choice, not a correctness one.
/// The gate is on `k·n` (the weight footprint) rather than `m·k·n` (total MACs):
/// NAX pays a fixed ~300 µs command-buffer dispatch, so it only wins once the
/// weight is big enough to amortize it; below that the GEMM underfills the tensor
/// engine and AMX (no dispatch, runs on the already-resident f32) is faster. `k·n`
/// cleanly separates the measured models (which `m·k·n` could not — smollm2's M=8
/// prefill GEMM and llama's M=1 decode GEMM have similar MACs but very different
/// weights):
///   * smollm2 layer GEMMs: `k·n` ≈ 576·576 .. 1536·576 ≈ 0.3–0.9M  → AMX
///   * llama  layer GEMMs: `k·n` ≈ 2048·2048 .. 2048·8192 ≈ 4.2–16.8M → NAX
///     (llama's GQA k/v projections ≈ 2048·512 ≈ 1.0M land on AMX)
///   * both lm_heads:       `k·n` ≫ 28M                                 → NAX
///
/// 3M splits them. Override with `KTIR_GEMM_GPU_MIN_KN` (0 = always NAX).
#[cfg(metal)]
pub const GEMM_GPU_MIN_KN: u64 = 3_000_000;

/// The NAX-vs-AMX `k·n` threshold (env `KTIR_GEMM_GPU_MIN_KN`, else
/// [`GEMM_GPU_MIN_KN`]). Read once per call by [`matmul_loop_offload`] /
/// [`matmul_loop_use_nax`].
#[cfg(metal)]
pub fn matmul_min_kn() -> u64 {
    std::env::var("KTIR_GEMM_GPU_MIN_KN")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(GEMM_GPU_MIN_KN)
}

/// Whether a recognized `m×k×n` K-loop should be OFFLOADED here as a full-M GEMM
/// (on NAX or AMX — see [`matmul_loop_use_nax`]) rather than fall through to the
/// interpreter running the loop's `scf.for`.
///
/// CORRECTNESS FIRST: the interpreter fallback runs the body at the fused
/// segment's grid `[1,1]`, which reconstructs the GEMM ONLY when `m == 1` (decode):
/// the Spyre SPMD K-loop tiles its output across the grid, so at `[1,1]` it
/// computes exactly the single M-row grid position 0 owns. For `m > 1` (prefill,
/// token-parallel M=8/M=32) the full-M reconstruction lives ONLY in the offload —
/// the `[1,1]` loop would compute just row 0 and silently drop the rest (it broke
/// prefill golden by ~0.05 in testing). So `m > 1` is ALWAYS offloaded here.
///
/// For a PLAIN `m == 1` GEMM we offload only when the weight `k·n` clears
/// [`GEMM_GPU_MIN_KN`]; smaller decode GEMMs run on the interpreter's tiled
/// Accelerate K-loop (faster at that scale for a CONTIGUOUS `[k,n]` weight, and the
/// per-node golden oracle uses the same path).
///
/// `transpose_b` GEMMs (B stored on-disk `[n,k]`) ALWAYS offload, regardless of
/// `k·n`. The interpreter K-loop is a trap for them: its per-K-step B panel is a
/// `[n,BK]` window of the `[n,k]` weight, which fails `is_contiguous` (leftmost
/// stride `k` ≫ `BK`) and so `ktdp.load` takes the slow strided-gather path —
/// reading a span ≈`k/BK`× larger than the data, EVERY K-step, ×layers ×tokens
/// (the measured decode 0.28→0.68 regression). Offloading collapses the whole
/// K-loop to ONE backend call over B read verbatim-contiguous `[n,k]`
/// ([`resolve_gemm_bt_operand`]) — NAX if `k·n` clears the gate, else AMX
/// `sgemm_rowmajor_bt` (`cblas` `CblasTrans`, free-to-faster at real decode `k`).
/// Golden-safe: `m == 1` is trivially full-M, and the AMX-bt branch does the
/// identical contraction the `[1,1]` loop did, just in one call.
#[cfg(metal)]
pub fn matmul_loop_offload(m: usize, k: usize, n: usize, transpose_b: bool) -> bool {
    m > 1 || transpose_b || (k as u64) * (n as u64) >= matmul_min_kn()
}

/// Of the OFFLOADED full-M GEMMs ([`matmul_loop_offload`]), whether to run this one
/// on NAX (`k·n` ≥ the gate) or AMX (below it). Both are full-M-correct and read
/// the same resident operands; this only picks the faster engine for the shape.
#[cfg(metal)]
pub fn matmul_loop_use_nax(k: usize, n: usize) -> bool {
    (k as u64) * (n as u64) >= matmul_min_kn()
}

/// Run a recognized matmul K-loop as ONE full-M GEMM (on NAX or AMX), binding the
/// loop's result tensor in `ctx`. Operands are resolved from the value table: a
/// forwarded activation is already a resident `Tile` (f32) and is re-uploaded each
/// pass; a constant weight is an HBM pointer decoded+uploaded ONCE and then served
/// from the resident [`WEIGHT_CACHE`]. The interpreter then skips the loop body
/// entirely.
///
/// Backend ([`matmul_loop_use_nax`]): both branches compute the SAME full-M GEMM
/// over the SAME resident host-visible f32 operands — large `k·n` runs on NAX (the
/// dispatch amortizes), small `k·n` runs on AMX (`blas::sgemm_rowmajor`, no GPU
/// dispatch, reads the resident buffers in place; this is what wins small-M prefill
/// while staying resident). AMX is f32-multiply (NAX is f16-operand); both round to
/// f16 at write-back, so golden parity holds and AMX is if anything more accurate.
///
/// Returns `Err` only when the loop should NOT be offloaded here (an `m == 1`
/// decode GEMM below the gate — the caller's interpreter K-loop is correct and
/// faster) or when a genuine resource is missing (no device, shape mismatch). For
/// `m > 1` the caller MUST treat `Err` as fatal, never the row-0 interpreter loop.
#[cfg(metal)]
pub fn run_matmul_loop_gpu(
    info: &MatmulLoopInfo,
    ctx: &mut crate::context::CoreContext,
) -> Result<(), String> {
    let (m, k, n) = (info.m as usize, info.k as usize, info.n as usize);
    // OFFLOAD GATE: a PLAIN `m == 1` decode GEMM below the work gate falls through
    // (Err) to the interpreter's tiled Accelerate K-loop, faster on tiny contiguous
    // weights and golden-faithful at [1,1] (m==1). `m > 1` and ALL transpose-B
    // GEMMs are ALWAYS offloaded full-M here (transpose-B's per-K-step interpreter
    // B load is a slow strided gather — see `matmul_loop_offload`).
    if !matmul_loop_offload(m, k, n, info.transpose_b) {
        return Err("metal: decode GEMM below the work gate — interpreter K-loop".into());
    }
    // BACKEND: NAX if the weight is big enough to amortize the GPU dispatch, else
    // AMX. Decided on `k·n` alone, so it's independent of M (both branches are
    // full-M-correct). An `m == 1` GEMM that passed the gate above is by definition
    // large, so decode never reaches the AMX branch — decode routing is unchanged.
    //
    // TRANSPOSE-B (B stored on-disk [n,k], contracted over its last axis) flows
    // through the SAME gate: big k·n → NAX/simdgroup via the `KTIR_TRANSPOSE_B`
    // pipeline (kernel reads [n,k] verbatim); small k·n → AMX `cblas` `CblasTrans`.
    // Both are native, zero-copy. The gate is identical to plain matmul.
    let use_nax = matmul_loop_use_nax(k, n);
    let c = GEMM_ENGINE.with(|cell| -> Result<Vec<f32>, String> {
        let engine = cell.get_or_init(|| NaxGemm::new().ok());
        let engine = engine.as_ref().ok_or("metal: no NaxGemm device")?;
        // A and B each resolve to a resident UnifiedBuffer: a constant weight
        // (HBM pointer) comes from the cache (decoded+uploaded at most once); a
        // resident activation tile is uploaded fresh (it changes every pass and
        // is NEVER cached).
        // B (weight) is f16 only on the NAX GPU path, when the flag is enabled. The
        // AMX branches read `ub.as_slice()` (f32) in place, so they must keep f32 B.
        // f16 B only when the flag is on AND this engine actually compiled the f16-B
        // pipelines (NAX devices only). On a non-NAX Metal device (e.g. CI macOS
        // runners) the engine has no f16 pipeline, so stay f32 — never produce an
        // f16 buffer the kernel can't consume (that was the metal.rs unwrap panic).
        let want_b_f16 = use_nax && f16_weights_enabled() && engine.has_f16_b_pipelines();
        let ua = resolve_gemm_operand_unified_off(
            &info.a_root,
            m,
            k,
            info.m_row_off as usize,
            ctx,
            engine,
            false, // A (activation) is always f32 in
        )?;
        // B operand, resolved VERBATIM (no transpose, no gather):
        //   * transpose-B: the [n,k] weight, or its CONTIGUOUS row-slice for an
        //     N-tile (rows [n_off, n_off+n) — a contiguous block, not a gather).
        //   * plain: the [k,n] weight (contiguous) or a strided column slice.
        let ub = if info.transpose_b {
            resolve_gemm_bt_operand(&info.b_root, n, k, info.n_off, ctx, engine, want_b_f16)?
        } else if info.n_off == 0 && info.b_stride == info.n {
            resolve_gemm_operand_unified(&info.b_root, k, n, ctx, engine, want_b_f16)?
        } else {
            resolve_gemm_weight_slice(
                &info.b_root,
                k,
                n,
                info.n_off,
                info.b_stride,
                ctx,
                engine,
                want_b_f16,
            )?
        };
        let out = if use_nax {
            // NAX / simdgroup GPU GEMM. `transpose_b` selects the [n,k]-staging
            // pipeline; B (`ub`) is [n,k] for transpose-B, [k,n] otherwise — same
            // length. `&ua`/`&ub` deref-coerce to &UnifiedBuffer.
            let mut uc = engine.unified(m * n)?;
            engine.matmul_unified(
                m,
                k,
                n,
                &ua,
                &ub,
                &mut uc,
                None,
                Epilogue::NONE,
                info.transpose_b,
            )?;
            uc.as_slice().to_vec()
        } else if info.transpose_b {
            // AMX native transpose-B: B is [n,k], contract the last axis (`CblasTrans`,
            // zero-copy). Full-M, no GPU dispatch, reads the resident f32 in place.
            debug_assert_eq!(ua.as_slice().len(), m * k, "AMX A operand length");
            debug_assert_eq!(ub.as_slice().len(), n * k, "AMX transpose-B operand length");
            crate::blas::sgemm_rowmajor_bt(m, k, n, ua.as_slice(), ub.as_slice())
        } else {
            // AMX/Accelerate over the SAME resident f32 operands — full-M, no GPU
            // dispatch, no per-pass re-decode (ua/ub are host-visible f32 already).
            // matmul_unified's length asserts don't run on this path, so guard the
            // operand sizes here (they're sized m*k / k*n by the upstream resolvers).
            debug_assert_eq!(ua.as_slice().len(), m * k, "AMX A operand length");
            debug_assert_eq!(ub.as_slice().len(), k * n, "AMX B operand length");
            crate::blas::sgemm_rowmajor(m, k, n, ua.as_slice(), ub.as_slice())
        };
        // Diagnostic: cross-check the chosen backend against a CPU sgemm on the SAME
        // operands, matching the op's contraction (transB vs plain). For NAX a diff
        // >> f16 noise pinpoints a NaxGemm shape bug; for AMX it's the same cblas
        // primitive, so the diff is ~0 (a useful self-check that out is real).
        // Skip the CPU cross-check when B is an f16 buffer (the f32 `as_slice()`
        // view is invalid); the f16-B path is validated by the golden e2e tests.
        if std::env::var_os("KTIR_GEMM_CHECK").is_some() && !ub.is_f16() {
            let cpu = if info.transpose_b {
                crate::blas::sgemm_rowmajor_bt(m, k, n, ua.as_slice(), ub.as_slice())
            } else {
                crate::blas::sgemm_rowmajor(m, k, n, ua.as_slice(), ub.as_slice())
            };
            let d = out
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if d > 0.05 {
                let be = if use_nax {
                    "NAX"
                } else if info.transpose_b {
                    "AMX-bt"
                } else {
                    "AMX"
                };
                eprintln!("  [gemm-check] m={m} k={k} n={n}  {be} vs CPU max diff {d:.4}");
            }
        }
        Ok(out)
    })?;
    // The K-loop's result tensor is f16 (matmul outs dtype); this f16 rounding at
    // write-back is what keeps NAX and AMX golden-equivalent (both quantize the
    // f32 result identically), matching the interpreter's matmul precision.
    let tile = crate::tile::Tile::compute(c, DType::F16, vec![m, n]);
    let bytes = tile.size_bytes() as i64;
    ctx.set_value(&info.out_ssa, crate::ir::Value::Tile(tile));
    ctx.track_lx(&info.out_ssa, bytes)?;
    let counter = if use_nax {
        &MATMUL_LOOP_GPU_COUNT
    } else {
        &MATMUL_LOOP_AMX_COUNT
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Resolve a GEMM operand to a resident [`UnifiedBuffer`].
///
///   * A resident `Tile` (a forwarded activation) is uploaded to a FRESH buffer
///     each call — it changes every pass, so caching it would be incorrect.
///   * An HBM pointer (`Value::Index`, a constant weight) is served from
///     [`WEIGHT_CACHE`]: on a key match (same name + len + content fingerprint)
///     the cached `Rc<UnifiedBuffer>` is cloned (no decode, no upload); on a miss
///     it is decoded f16->f32, uploaded once, and inserted. This is where the
///     per-pass weight re-upload cost is eliminated.
#[cfg(metal)]
fn resolve_gemm_operand_unified(
    root: &str,
    rows: usize,
    cols: usize,
    ctx: &crate::context::CoreContext,
    engine: &NaxGemm,
    want_f16: bool,
) -> Result<std::rc::Rc<UnifiedBuffer>, String> {
    resolve_gemm_operand_unified_off(root, rows, cols, 0, ctx, engine, want_f16)
}

/// [`resolve_gemm_operand_unified`] with a leading ROW offset: read the `rows×cols`
/// operand starting at row `row_off` of the full tensor (offset `row_off*cols`
/// elements). `row_off=0` is the default full-tensor read. Used by the
/// last-token-only rewrite to reconstruct a single activation row (`m_row_off`).
#[cfg(metal)]
fn resolve_gemm_operand_unified_off(
    root: &str,
    rows: usize,
    cols: usize,
    row_off: usize,
    ctx: &crate::context::CoreContext,
    engine: &NaxGemm,
    // f16 applies only to a WEIGHT (HBM pointer); a forwarded activation TILE is
    // always f32 (it is the A operand, kept f32 in).
    want_f16: bool,
) -> Result<std::rc::Rc<UnifiedBuffer>, String> {
    let n = rows * cols;
    let elem_off = row_off * cols;
    match ctx.get_value(root)? {
        // Activations are resident already and CHANGE every pass — upload fresh,
        // never cache. (Caching one would serve a stale activation next pass.)
        crate::ir::Value::Tile(t) => {
            let full = t.as_f32();
            if elem_off + n > full.len() {
                return Err(format!(
                    "metal: GEMM operand {root} resident tile has {} elems, need {} at row_off {row_off}",
                    full.len(),
                    elem_off + n
                ));
            }
            Ok(std::rc::Rc::new(
                engine.unified_from(&full[elem_off..elem_off + n])?,
            ))
        }
        // Constant weight in HBM: cache by (name, len, content fingerprint).
        crate::ir::Value::Index(elem) => {
            // The pointer SSA value is an ELEMENT index (RFC #110): byte address
            // is elem*bytes_per_elem (f16 weight), NOT elem*STICK_BYTES.
            let addr = (elem + elem_off as i64) * DType::F16.bytes_per_elem() as i64;
            // Build the resident weight buffer: f16 (raw HBM bytes, no f32 expansion,
            // half the streamed bytes) when `want_f16`, else f32 (decoded).
            let build = || -> Result<UnifiedBuffer, String> {
                let hbm = ctx.hbm.borrow();
                if want_f16 {
                    let raw = hbm.read_bytes(addr, n * DType::F16.bytes_per_elem());
                    engine.unified_f16_from_raw(&raw)
                } else {
                    let decoded = hbm.read_decoded(addr, n, DType::F16);
                    engine.unified_from(&decoded)
                }
            };
            // Diagnostic: bypass the cache entirely (always decode+upload fresh).
            if std::env::var_os("KTIR_NO_WEIGHT_CACHE").is_some() {
                return Ok(std::rc::Rc::new(build()?));
            }
            let fingerprint = {
                let hbm = ctx.hbm.borrow();
                weight_fingerprint(hbm, addr, n, DType::F16)
            };
            let key = WeightKey {
                root: root.to_string(),
                len: n,
                fingerprint,
                col_off: 0,
                f16: want_f16,
            };
            // Fast path: a hit returns the cached buffer with no further HBM work.
            if let Some(buf) = WEIGHT_CACHE.with(|c| c.borrow().get(&key).cloned()) {
                WEIGHT_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(buf);
            }
            // Miss: read this weight from HBM and upload it once.
            let buf = std::rc::Rc::new(build()?);
            WEIGHT_CACHE.with(|c| {
                let mut cache = c.borrow_mut();
                // Bound memory: a simple clear-on-overflow eviction. Correct (the
                // next pass repopulates the working set); rare in practice.
                if cache.len() >= WEIGHT_CACHE_MAX {
                    cache.clear();
                }
                cache.insert(key, buf.clone());
            });
            WEIGHT_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(buf)
        }
        other => Err(format!(
            "metal: GEMM operand {root} is {other:?}, want tile/ptr"
        )),
    }
}

/// Resolve an N-TILED GEMM weight operand B to a resident `[k, n]` [`UnifiedBuffer`]
/// holding the COLUMN SLICE `B_full[:, col_off : col_off+n]`, where `B_full` is the
/// HBM weight with row stride `b_stride`. Used for the lm_head's column tiles (the
/// only N-tiled GEMMs in these bundles): the K-loop computes one 16384-wide output
/// tile from a strided window of the [2048,128256] weight, and reconstructing that
/// exact window (vs the whole tensor) is what makes the offload correct.
///
/// Only weights (HBM pointers) reach here (the recognizer rejects activation
/// N-tiles). The slice is decoded f16->f32 row-by-row (each row is `n` contiguous
/// elements at `col_off`) and cached by [`WeightKey`] including `col_off`, so the 8
/// tiles of one weight cache independently and are decoded+uploaded at most once.
#[cfg(metal)]
#[allow(clippy::too_many_arguments)]
fn resolve_gemm_weight_slice(
    root: &str,
    k: usize,
    n: usize,
    col_off: i64,
    b_stride: i64,
    ctx: &crate::context::CoreContext,
    engine: &NaxGemm,
    want_f16: bool,
) -> Result<std::rc::Rc<UnifiedBuffer>, String> {
    let elem = match ctx.get_value(root)? {
        crate::ir::Value::Index(s) => *s,
        other => {
            return Err(format!(
                "metal: N-tiled GEMM weight {root} is {other:?}, want an HBM pointer"
            ));
        }
    };
    let bpe = DType::F16.bytes_per_elem() as i64;
    // The pointer SSA value is an ELEMENT index (RFC #110): byte base = elem*bpe.
    let base = elem * bpe;
    // Build the [k,n] resident slice. f16: gather raw f16 bytes (half the bytes, no
    // f32 expansion). f32: decode each strided row to f32.
    let build = || -> Result<UnifiedBuffer, String> {
        let hbm = ctx.hbm.borrow();
        if want_f16 {
            let mut raw = Vec::with_capacity(k * n * bpe as usize);
            for r in 0..k as i64 {
                let row_addr = base + (r * b_stride + col_off) * bpe;
                raw.extend_from_slice(&hbm.read_bytes(row_addr, n * bpe as usize));
            }
            engine.unified_f16_from_raw(&raw)
        } else {
            let mut out = Vec::with_capacity(k * n);
            for r in 0..k as i64 {
                let row_addr = base + (r * b_stride + col_off) * bpe;
                out.extend_from_slice(&hbm.read_decoded(row_addr, n, DType::F16));
            }
            engine.unified_from(&out)
        }
    };
    if std::env::var_os("KTIR_NO_WEIGHT_CACHE").is_some() {
        return Ok(std::rc::Rc::new(build()?));
    }
    // Fingerprint the FULL weight (root identity); col_off keys the slice apart.
    let fingerprint = {
        let hbm = ctx.hbm.borrow();
        weight_fingerprint(hbm, base, k * b_stride as usize, DType::F16)
    };
    let key = WeightKey {
        root: root.to_string(),
        len: k * n,
        fingerprint,
        col_off,
        f16: want_f16,
    };
    if let Some(buf) = WEIGHT_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        WEIGHT_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(buf);
    }
    let buf = std::rc::Rc::new(build()?);
    WEIGHT_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if cache.len() >= WEIGHT_CACHE_MAX {
            cache.clear();
        }
        cache.insert(key, buf.clone());
    });
    WEIGHT_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(buf)
}

/// Resolve a TRANSPOSE-B weight operand B to a resident `[n, k]` f32
/// [`UnifiedBuffer`] — its on-disk PyTorch `Linear` `[out, in]` layout, uploaded
/// VERBATIM (no transpose). The GEMM then contracts the last axis via `transB`.
/// For an N-tile, the slice is rows `[n_off, n_off+n)` of the weight, which is a
/// CONTIGUOUS block (`n*k` elements at `n_off*k`) — so this is a plain contiguous
/// read either way. Cached once per process (keyed by `col_off = n_off`).
#[cfg(metal)]
fn resolve_gemm_bt_operand(
    root: &str,
    n: usize,
    k: usize,
    n_off: i64,
    ctx: &crate::context::CoreContext,
    engine: &NaxGemm,
    want_f16: bool,
) -> Result<std::rc::Rc<UnifiedBuffer>, String> {
    let elem = match ctx.get_value(root)? {
        crate::ir::Value::Index(s) => *s,
        other => {
            return Err(format!(
                "metal: transpose-B weight {root} is {other:?}, want an HBM pointer"
            ));
        }
    };
    let bpe = DType::F16.bytes_per_elem() as i64;
    // Contiguous [n,k] block: the N-tile is just rows [n_off, n_off+n) on disk.
    let elem_off = n_off * k as i64;
    // The pointer SSA value is an ELEMENT index (RFC #110): byte addr = elem*bpe.
    let addr = (elem + elem_off) * bpe;
    let count = n * k;
    // f16: copy the contiguous raw f16 block verbatim (half the bytes). f32: decode.
    let build = || -> Result<UnifiedBuffer, String> {
        let hbm = ctx.hbm.borrow();
        if want_f16 {
            let raw = hbm.read_bytes(addr, count * bpe as usize);
            engine.unified_f16_from_raw(&raw)
        } else {
            let decoded = hbm.read_decoded(addr, count, DType::F16);
            engine.unified_from(&decoded)
        }
    };
    if std::env::var_os("KTIR_NO_WEIGHT_CACHE").is_some() {
        return Ok(std::rc::Rc::new(build()?));
    }
    let fingerprint = {
        let hbm = ctx.hbm.borrow();
        weight_fingerprint(hbm, addr, count, DType::F16)
    };
    let key = WeightKey {
        root: root.to_string(),
        len: count,
        fingerprint,
        col_off: n_off,
        f16: want_f16,
    };
    if let Some(buf) = WEIGHT_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        WEIGHT_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Ok(buf);
    }
    let buf = std::rc::Rc::new(build()?);
    WEIGHT_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if cache.len() >= WEIGHT_CACHE_MAX {
            cache.clear();
        }
        cache.insert(key, buf.clone());
    });
    WEIGHT_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(buf)
}

/// Diagnostic: `(top-level scf.for count, of which recognized as matmul K-loops)`.
/// Lets a test confirm every K-loop in a fused function collapses to a GEMM —
/// the prefill-readiness check — without standing up the full executor.
pub fn count_matmul_loops(ops: &[Operation]) -> (usize, usize) {
    let defs = def_map_all(ops);
    let mut total = 0;
    let mut recognized = 0;
    for op in ops {
        if op.op_type == "scf.for" {
            total += 1;
            if recognize_matmul_loop(op, &defs).is_some() {
                recognized += 1;
            }
        }
    }
    (total, recognized)
}

// =========================================================================
// Attention-island offloads — move the heavy compute of the (unrolled, NO
// scf.for) attention nodes off the interpreter onto the GPU. These are PLAIN
// `linalg.matmul` (QK^T / A·V), `linalg.reduce dimensions=[1]` (softmax
// row-max / row-sum), and `linalg.transpose`. Each reads its operands from the
// value table as resident f32 Tiles, runs a GPU kernel, and binds the result
// back — the interpreter stays the coherence medium exactly like the K-loop and
// map-window offloads. Tiny index math / extracts / splats stay on the CPU.
//
// Each offload is gated by its own KTIR_NO_GPU_* toggle (for A/B measurement)
// under the same single-core/tracker-free conditions as the existing offloads,
// and on any failure falls through to the interpreter (correctness preserved).
// =========================================================================

/// Count of PLAIN `linalg.matmul` ops offloaded to a GPU GEMM (telemetry/proof
/// the attention QK^T / A·V GEMMs actually ran on Metal).
#[cfg(metal)]
pub static PLAIN_MATMUL_GPU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Count of `linalg.reduce` ops offloaded to a GPU reduction kernel.
#[cfg(metal)]
pub static REDUCE_GPU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Count of `linalg.transpose` ops offloaded to a GPU kernel.
#[cfg(metal)]
pub static TRANSPOSE_GPU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Run a PLAIN (not inside an scf.for) `linalg.matmul` as one GPU GEMM, binding
/// its result in `ctx`. Reads A = ins[0], B = ins[1] from the value table as
/// resident f32 Tiles, derives m/k/n from their shapes, runs the NaxGemm engine,
/// then folds in the `outs` accumulator (ins[2], `C = A@B + C`) on the host — the
/// attention matmuls all init `outs` to `dense<0.0>`, but folding it keeps the
/// op's exact `C + A@B` semantics for any init. The result dtype mirrors the
/// interpreter's `matmul`: the `outs` dtype if present, else A's dtype.
///
/// Returns `Err` (no device, operand not resident, shape mismatch) so the caller
/// falls back to the interpreter — never a wrong answer.
#[cfg(metal)]
pub fn run_plain_matmul_gpu(
    op: &Operation,
    ctx: &mut crate::context::CoreContext,
) -> Result<(), String> {
    let out_ssa = op
        .result
        .as_deref()
        .ok_or("metal: plain matmul has no result SSA")?;
    // A, B as resident f32 tiles (clone the shapes/data we need, drop borrows
    // before we touch the engine / mutate ctx).
    let a = expect_resident_tile(ctx, &op.operands[0], "plain matmul A")?;
    let b = expect_resident_tile(ctx, &op.operands[1], "plain matmul B")?;
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "metal: plain matmul wants 2-D operands, got {:?} @ {:?}",
            a.shape, b.shape
        ));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    if b.shape[0] != k {
        return Err(format!(
            "metal: plain matmul inner dims disagree: {:?} @ {:?}",
            a.shape, b.shape
        ));
    }
    let n = b.shape[1];
    // The `outs` accumulator + its dtype (the interpreter keeps `outs`'s dtype
    // for the result when present, else A's).
    let (acc, result_dtype) = if op.operands.len() > 2 {
        match ctx.get_value(&op.operands[2]) {
            Ok(crate::ir::Value::Tile(c)) if c.len() == m * n => {
                (Some(c.as_f32().to_vec()), c.dtype)
            }
            _ => (None, a.dtype),
        }
    } else {
        (None, a.dtype)
    };

    let mut out = GEMM_ENGINE.with(|cell| -> Result<Vec<f32>, String> {
        let engine = cell.get_or_init(|| NaxGemm::new().ok());
        let engine = engine.as_ref().ok_or("metal: no NaxGemm device")?;
        let ua = engine.unified_from(&a.as_f32())?;
        let ub = engine.unified_from(&b.as_f32())?;
        let mut uc = engine.unified(m * n)?;
        engine.matmul_unified(m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE, false)?;
        Ok(uc.as_slice().to_vec())
    })?;
    if let Some(acc) = acc {
        for (o, c) in out.iter_mut().zip(acc.iter()) {
            *o += c;
        }
    }
    let tile = crate::tile::Tile::compute(out, result_dtype, vec![m, n]);
    let bytes = tile.size_bytes() as i64;
    ctx.set_value(out_ssa, crate::ir::Value::Tile(tile));
    ctx.track_lx(out_ssa, bytes)?;
    PLAIN_MATMUL_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// A recognized `linalg.reduce` combiner: the per-element fold and its identity
/// (the value a row's accumulator starts at). Only the order-insensitive sum/max
/// the softmax uses are supported; anything else returns `None` (interpreter).
#[cfg(metal)]
#[derive(Clone, Copy)]
struct ReduceCombiner {
    /// MSL infix for `acc = acc <op> x` — `"+"` for sum, but max needs a call,
    /// so we carry an enum-ish tag instead and render in [`run_reduce_gpu`].
    is_max: bool,
    identity: f32,
}

/// Recognize a `linalg.reduce`'s combiner (sum -> +, init 0; max -> max, init
/// -inf). Reads `reduce_fn` (the shorthand the parser lifts) or the region's
/// single non-yield op. `None` for any other combiner.
#[cfg(metal)]
fn recognize_reduce_combiner(op: &Operation) -> Option<ReduceCombiner> {
    let name = match op.attributes.get("reduce_fn") {
        Some(Attr::Str(s)) => s.clone(),
        _ => op
            .regions
            .iter()
            .flatten()
            .find(|o| o.op_type != "linalg.yield")
            .map(|o| o.op_type.clone())?,
    };
    match name.as_str() {
        "arith.addf" => Some(ReduceCombiner {
            is_max: false,
            identity: 0.0,
        }),
        "arith.maximumf" | "arith.maxf" => Some(ReduceCombiner {
            is_max: true,
            identity: f32::NEG_INFINITY,
        }),
        _ => None,
    }
}

/// Run a `linalg.reduce ins(%x) dimensions=[1]` over the last axis of a 2-D
/// tensor `[rows, cols]` as a GPU reduction (one threadgroup row → one output
/// element). Binds the reduced `[rows]` tensor (or a scalar if `rows==1` AND the
/// input was 1-D — never here) under the op's result SSA. Mirrors the
/// interpreter's `reduce`: f16 input → f32 fold → round to the input dtype, and
/// the result is `Tile([rows])` (shape with the reduced axis removed).
///
/// Returns `Err` (unsupported combiner / shape / no device) so the caller falls
/// back to the interpreter.
#[cfg(metal)]
pub fn run_reduce_gpu(op: &Operation, ctx: &mut crate::context::CoreContext) -> Result<(), String> {
    // Only `dimensions = [1]` over a 2-D input is handled (the softmax pattern).
    let dims = int_list_attr_vec(op, "dimensions").unwrap_or_default();
    if dims.as_slice() != [1] {
        return Err(format!(
            "metal: reduce dimensions {dims:?} != [1] — interpreter"
        ));
    }
    let combiner =
        recognize_reduce_combiner(op).ok_or("metal: unsupported reduce combiner — interpreter")?;
    let out_ssa = op
        .result
        .as_deref()
        .ok_or("metal: reduce has no result SSA")?;
    let x = expect_resident_tile(ctx, &op.operands[0], "reduce ins")?;
    if x.shape.len() != 2 {
        return Err(format!("metal: reduce wants 2-D input, got {:?}", x.shape));
    }
    let (rows, cols) = (x.shape[0], x.shape[1]);
    if rows * cols != x.len() {
        return Err("metal: reduce input shape/data mismatch".into());
    }
    let dtype = x.dtype;
    let kernel = reduce_kernel(combiner, dtype);
    // One output element per row; the kernel folds `cols` along the row.
    let out = run_reduce_kernel(&kernel, &x.as_f32(), rows, cols, combiner.identity)?;
    // Result shape = input shape with axis 1 removed -> [rows]. (rows>=1; the
    // interpreter only collapses to a scalar when the remaining shape is empty,
    // which can't happen for a 2-D input.)
    let tile = crate::tile::Tile::compute(out, dtype, vec![rows]);
    let bytes = tile.size_bytes() as i64;
    ctx.set_value(out_ssa, crate::ir::Value::Tile(tile.clone()));
    ctx.track_lx(out_ssa, bytes)?;
    // MLIR may also reference the result by the `outs` SSA name (the interpreter
    // binds `outs_var` too) — mirror that if present.
    if let Some(Attr::Str(outs_var)) = op.attributes.get("outs_var") {
        ctx.set_value(outs_var, crate::ir::Value::Tile(tile));
    }
    REDUCE_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// MSL for a row reduction: each thread folds one row of `cols` elements with the
/// combiner (sum or max), seeded from `identity` (passed as a buffer so the same
/// kernel serves both). `rows` is the dispatch width.
#[cfg(metal)]
fn reduce_kernel(combiner: ReduceCombiner, dtype: DType) -> MslKernel {
    let ty = msl_type(dtype);
    // `arith.maximumf` is NaN-propagating (see the interpreter's `reduce_combiner`),
    // unlike MSL `max`/`fmax` which return the non-NaN argument. Match the
    // interpreter exactly so a NaN score reduces to NaN, not the finite operand.
    let acc_fold = if combiner.is_max {
        "acc = (isnan(acc) || isnan(xv)) ? NAN : (acc >= xv ? acc : xv)"
    } else {
        "acc = acc + xv"
    };
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         kernel void row_reduce(\n\
         \x20   device const {ty}* x [[buffer(0)]],\n\
         \x20   device {ty}* out [[buffer(1)]],\n\
         \x20   constant uint& cols [[buffer(2)]],\n\
         \x20   constant float& identity [[buffer(3)]],\n\
         \x20   uint row [[thread_position_in_grid]]\n\
         ) {{\n\
         \x20   float acc = identity;\n\
         \x20   for (uint c = 0; c < cols; c++) {{ float xv = float(x[row * cols + c]); {acc_fold}; }}\n\
         \x20   out[row] = ({ty})acc;\n\
         }}\n",
        ty = ty,
        acc_fold = acc_fold,
    );
    MslKernel {
        source,
        name: "row_reduce".to_string(),
        buffers: vec![
            BufferBinding {
                name: "x".into(),
                is_output: false,
                dtype,
            },
            BufferBinding {
                name: "out".into(),
                is_output: true,
                dtype,
            },
        ],
    }
}

/// Dispatch the row-reduce kernel: upload `x` (rows*cols, dtype-encoded), pass
/// `cols`/`identity` as inline bytes, dispatch `rows` threads, read back `rows`
/// f32. Uses the shared device/queue/pipeline cache (`cached_dispatch`).
#[cfg(metal)]
fn run_reduce_kernel(
    kernel: &MslKernel,
    x: &[f32],
    rows: usize,
    cols: usize,
    identity: f32,
) -> Result<Vec<f32>, String> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    let (device, queue, pipeline) = cached_dispatch(kernel)?;
    let res = MTLResourceOptions::StorageModeShared;
    let in_dtype = kernel.buffers[0].dtype;
    let out_dtype = kernel.buffers[1].dtype;

    let in_bytes = crate::codec::encode(x, in_dtype);
    // SAFETY: `in_bytes` outlives the copy inside newBufferWithBytes.
    let in_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(in_bytes.as_ptr() as *mut c_void).unwrap(),
                in_bytes.len().max(1),
                res,
            )
            .ok_or("metal: reduce input buffer alloc failed")?
    };
    let out_buf = device
        .newBufferWithLength_options((rows * out_dtype.bytes_per_elem()).max(1), res)
        .ok_or("metal: reduce output buffer alloc failed")?;

    let cb = queue.commandBuffer().ok_or("metal: commandBuffer nil")?;
    let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
    enc.setComputePipelineState(&pipeline);
    let cols_u = cols as u32;
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&cols_u as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            2,
        );
        enc.setBytes_length_atIndex(
            NonNull::new(&identity as *const f32 as *mut c_void).unwrap(),
            std::mem::size_of::<f32>(),
            3,
        );
    }
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(rows).max(1);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let nbytes = rows * out_dtype.bytes_per_elem();
    let raw =
        unsafe { std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const u8, nbytes) }
            .to_vec();
    Ok(crate::codec::decode(&raw, rows, out_dtype))
}

/// Run a `linalg.transpose ins(%x) permutation=[...]` on the GPU as a gather:
/// one thread per output element, `out[o] = in[ source(o) ]` with the source
/// index computed from the permutation and the row-major strides. Binds the
/// transposed tensor under the op's result SSA. Mirrors the interpreter's
/// `transpose` exactly (same dtype, same index mapping). Supports any rank.
///
/// Returns `Err` (no permutation, rank mismatch, no device) → interpreter.
#[cfg(metal)]
pub fn run_transpose_gpu(
    op: &Operation,
    ctx: &mut crate::context::CoreContext,
) -> Result<(), String> {
    let perm = int_list_attr_vec(op, "permutation")
        .ok_or("metal: transpose missing permutation — interpreter")?;
    let out_ssa = op
        .result
        .as_deref()
        .ok_or("metal: transpose has no result SSA")?;
    let x = expect_resident_tile(ctx, &op.operands[0], "transpose ins")?;
    if perm.len() != x.shape.len() {
        return Err(format!(
            "metal: transpose permutation rank {} != input rank {}",
            perm.len(),
            x.shape.len()
        ));
    }
    let perm: Vec<usize> = perm.iter().map(|&p| p as usize).collect();
    if perm.iter().any(|&p| p >= x.shape.len()) {
        return Err("metal: transpose permutation out of range".into());
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let in_strides = {
        // row-major strides of the input shape
        let mut s = vec![1usize; x.shape.len()];
        for i in (0..x.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * x.shape[i + 1];
        }
        s
    };
    let out_strides = {
        let mut s = vec![1usize; out_shape.len()];
        for i in (0..out_shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1];
        }
        s
    };
    let out_len: usize = out_shape.iter().product();
    let dtype = x.dtype;
    let kernel = transpose_kernel(dtype);
    // Per-output-element source index, computed from out_strides/in_strides/perm
    // on the GPU. We pass the rank and the three index arrays as buffers.
    let src_in_strides: Vec<u32> = perm.iter().map(|&p| in_strides[p] as u32).collect();
    let out_strides_u: Vec<u32> = out_strides.iter().map(|&s| s as u32).collect();
    let out = run_transpose_kernel(
        &kernel,
        &x.as_f32(),
        out_len,
        &out_strides_u,
        &src_in_strides,
    )?;
    let tile = crate::tile::Tile::compute(out, dtype, out_shape);
    let bytes = tile.size_bytes() as i64;
    ctx.set_value(out_ssa, crate::ir::Value::Tile(tile));
    ctx.track_lx(out_ssa, bytes)?;
    TRANSPOSE_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// MSL for a permutation gather: thread `o` decomposes its linear output index
/// into per-axis coordinates via `out_strides`, then recombines through
/// `src_in_strides[axis] = in_strides[perm[axis]]` to read the source element.
/// `rank` and the two stride arrays are passed as buffers.
#[cfg(metal)]
fn transpose_kernel(dtype: DType) -> MslKernel {
    let ty = msl_type(dtype);
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         kernel void transpose_gather(\n\
         \x20   device const {ty}* x [[buffer(0)]],\n\
         \x20   device {ty}* out [[buffer(1)]],\n\
         \x20   constant uint& rank [[buffer(2)]],\n\
         \x20   device const uint* out_strides [[buffer(3)]],\n\
         \x20   device const uint* src_in_strides [[buffer(4)]],\n\
         \x20   uint o [[thread_position_in_grid]]\n\
         ) {{\n\
         \x20   uint rem = o;\n\
         \x20   uint src = 0;\n\
         \x20   for (uint d = 0; d < rank; d++) {{\n\
         \x20       uint coord = rem / out_strides[d];\n\
         \x20       rem = rem % out_strides[d];\n\
         \x20       src += coord * src_in_strides[d];\n\
         \x20   }}\n\
         \x20   out[o] = x[src];\n\
         }}\n",
        ty = ty,
    );
    MslKernel {
        source,
        name: "transpose_gather".to_string(),
        buffers: vec![
            BufferBinding {
                name: "x".into(),
                is_output: false,
                dtype,
            },
            BufferBinding {
                name: "out".into(),
                is_output: true,
                dtype,
            },
        ],
    }
}

/// Dispatch the transpose gather: upload `x` (dtype-encoded), the rank + two
/// stride arrays, dispatch `out_len` threads, read back `out_len` f32.
#[cfg(metal)]
fn run_transpose_kernel(
    kernel: &MslKernel,
    x: &[f32],
    out_len: usize,
    out_strides: &[u32],
    src_in_strides: &[u32],
) -> Result<Vec<f32>, String> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    let (device, queue, pipeline) = cached_dispatch(kernel)?;
    let res = MTLResourceOptions::StorageModeShared;
    let in_dtype = kernel.buffers[0].dtype;
    let out_dtype = kernel.buffers[1].dtype;

    let in_bytes = crate::codec::encode(x, in_dtype);
    // SAFETY: each `*_bytes`/array outlives the copy inside newBufferWithBytes.
    let in_buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(in_bytes.as_ptr() as *mut c_void).unwrap(),
                in_bytes.len().max(1),
                res,
            )
            .ok_or("metal: transpose input buffer alloc failed")?
    };
    let out_buf = device
        .newBufferWithLength_options((out_len * out_dtype.bytes_per_elem()).max(1), res)
        .ok_or("metal: transpose output buffer alloc failed")?;
    let stride_buf = |arr: &[u32]| -> Result<_, String> {
        let nbytes = std::mem::size_of_val(arr).max(4);
        // SAFETY: `arr` outlives the copy.
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(arr.as_ptr() as *mut c_void).unwrap(),
                    nbytes,
                    res,
                )
                .ok_or_else(|| "metal: transpose stride buffer alloc failed".to_string())
        }
    };
    let out_strides_buf = stride_buf(out_strides)?;
    let src_in_strides_buf = stride_buf(src_in_strides)?;

    let cb = queue.commandBuffer().ok_or("metal: commandBuffer nil")?;
    let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
    enc.setComputePipelineState(&pipeline);
    let rank = out_strides.len() as u32;
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 1);
        enc.setBytes_length_atIndex(
            NonNull::new(&rank as *const u32 as *mut c_void).unwrap(),
            std::mem::size_of::<u32>(),
            2,
        );
        enc.setBuffer_offset_atIndex(Some(&out_strides_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&src_in_strides_buf), 0, 4);
    }
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(out_len).max(1);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: out_len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let nbytes = out_len * out_dtype.bytes_per_elem();
    let raw =
        unsafe { std::slice::from_raw_parts(out_buf.contents().as_ptr() as *const u8, nbytes) }
            .to_vec();
    Ok(crate::codec::decode(&raw, out_len, out_dtype))
}

/// Resolve an operand to a resident f32 [`Tile`] (cloned), erroring if it is not
/// a `Value::Tile` — the offloads need materialized data, not a pointer/scalar.
#[cfg(metal)]
fn expect_resident_tile(
    ctx: &crate::context::CoreContext,
    name: &str,
    what: &str,
) -> Result<crate::tile::Tile, String> {
    match ctx.get_value(name)? {
        crate::ir::Value::Tile(t) => Ok(t.clone()),
        other => Err(format!(
            "metal: {what} {name} is {other:?}, want a resident tile"
        )),
    }
}

/// Result-SSA (stripped of `%`) -> defining op, recursively through regions.
/// A matmul K-loop references views/producers defined OUTSIDE the loop body and
/// loads/slices defined INSIDE it, so recognition needs a function-wide map.
fn def_map_all(ops: &[Operation]) -> HashMap<String, &Operation> {
    let mut m = HashMap::new();
    fn rec<'a>(ops: &'a [Operation], m: &mut HashMap<String, &'a Operation>) {
        for op in ops {
            if let Some(r) = &op.result {
                m.insert(strip(r).to_string(), op);
            }
            for region in &op.regions {
                rec(region, m);
            }
        }
    }
    rec(ops, &mut m);
    m
}

/// Recognize an `scf.for` as a single GEMM: body accumulates
/// `acc += A_tile @ B_tile` over the induction variable. Returns the FULL-shape
/// GEMM (M from the A operand's full view/producer, not the per-iter tile), so a
/// decode (M=1) and a prefill (M=8, grid/token-parallel) K-loop both collapse to
/// one matmul. Tolerant of plumbing ops in the body (the `outs` init constant).
fn recognize_matmul_loop(
    forop: &Operation,
    defs: &HashMap<String, &Operation>,
) -> Option<MatmulLoopInfo> {
    let body = forop.regions.first()?;
    // Exactly one matmul in the body.
    let mut mms = body.iter().filter(|o| o.op_type == "linalg.matmul");
    let mm = mms.next()?;
    if mms.next().is_some() {
        return None;
    }
    // Transpose layout: read from the op name OR its `indexing_maps` (the same
    // source of truth the scalar dispatch uses) — never re-derive from the op
    // name alone, or `linalg.matmul` + transpose-B `indexing_maps` would offload
    // as a plain `A·B` and silently compute the wrong contraction. transpose-A
    // is not supported on the offload path: skip (fall back to the interpreter).
    let (transpose_a, transpose_b) = crate::dialects::linalg::matmul_transpose_flags(mm).ok()?;
    if transpose_a {
        return None;
    }
    let mm_res = mm.result.as_deref()?;
    // Single loop-carried accumulator.
    let iter_args = match forop.attributes.get("iter_args") {
        Some(Attr::StrList(v)) if v.len() == 1 => v,
        _ => return None,
    };
    let acc = iter_args[0].as_str();
    // The accumulate: addf(acc, matmul_result) (either operand order).
    let addf = body.iter().find(|o| {
        o.op_type == "arith.addf"
            && o.operands.iter().any(|x| x == acc)
            && o.operands.iter().any(|x| x == mm_res)
    })?;
    let addf_res = addf.result.as_deref()?;
    // The loop yields the accumulate.
    let yld = body.iter().find(|o| o.op_type == "scf.yield")?;
    if yld.operands.first().map(String::as_str) != Some(addf_res) {
        return None;
    }
    // A = ins[0], B = ins[1]; resolve each to its FULL tensor + resident root.
    let (a_root, a_shape) = matmul_operand_full(mm.operands.first()?, defs)?;
    let (b_root, b_shape) = matmul_operand_full(mm.operands.get(1)?, defs)?;
    // Contraction axis: plain `matmul` is A[m,k]·B[k,n] (B's FIRST axis = k);
    // transpose-B is A[m,k]·B[n,k]ᵀ (B's LAST axis = k, FIRST axis = n).
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return None;
    }
    let b_full_n = if transpose_b { b_shape[0] } else { b_shape[1] };
    let b_k = if transpose_b { b_shape[1] } else { b_shape[0] };
    if a_shape[1] != b_k {
        return None;
    }
    // N-TILING. The loop tiles only the K dimension; the matmul's per-iteration
    // output then spans this loop's OUTPUT N (its last dim). For a plain K-loop
    // that equals B's full view width (the whole output row, summed over K-blocks).
    // But scratchy also tiles the OUTPUT N dimension sequentially when the output
    // is too wide for one tile: Llama-1B's lm_head `[m,2048]@[2048,128256]` is
    // split into 16384-wide COLUMN tiles, several K-loops, each computing B's
    // COLUMN SLICE `B_full[k, n_off : n_off+16384]` at a constant offset `n_off`
    // carried in the B access tile's column index. `matmul_operand_full` resolves B
    // to its FULL view ([2048,128256]) and drops that column offset, so a naive
    // reconstruction would compute the whole [m,128256] for EVERY slice — a
    // correct-but-WRONG A@B. We instead reconstruct the exact column SLICE:
    //   * n   = this loop's output N (the matmul's last dim) = the tile width,
    //   * n_off = the B access tile's column-index constant (the slice start),
    //   * b_stride = B's full view width (the source row stride for the slice).
    // The executor then uploads `B_full[:, n_off : n_off+n]` (strided gather) and
    // runs an `[m,k]@[k,n]` GEMM, computing the full M for THIS column tile — so
    // prefill (M=8) writes ALL token rows (the interpreter fallback, run at grid
    // [1,1] in the fused function, would only write row 0). For the common case
    // (SmolLM2's lm_head, all projections) n_off=0 and n==b_stride==full width.
    //
    // M is intentionally NOT cross-checked: a grid-parallel prefill K-loop
    // (SmolLM2/Llama [8,1]) legitimately has matmul-out M=1 (one row per core)
    // while the reconstructed M=8 comes from the full activation view — that
    // M-from-grid reconstruction is exactly what this recognizer is for.
    let mm_out = shape_attr_vec(Some(mm))?;
    let n_tile = *mm_out.last()?;
    let (n, n_off, b_stride) = if n_tile == b_full_n {
        // Plain (untiled) output: B spans the whole N width.
        (b_full_n, 0, b_full_n)
    } else if transpose_b {
        // N-tiled transpose-B: the N axis is B's FIRST (row) axis, so the slice is
        // a CONTIGUOUS block of `n_tile` rows of the `[n,k]` weight at the N-offset
        // carried in the access tile's FIRST index.
        let n_off = matmul_b_axis_offset(mm.operands.get(1)?, defs, /*last=*/ false)?;
        if n_off < 0 || n_off + n_tile > b_full_n {
            return None;
        }
        (n_tile, n_off, b_full_n)
    } else {
        // N-tiled plain: B is a column slice of a wider weight; the offset is the
        // access tile's last (column) index. A non-weight N-tile can't be strided
        // here, so reject (interpreter).
        let n_off = matmul_b_axis_offset(mm.operands.get(1)?, defs, /*last=*/ true)?;
        if n_off < 0 || n_off + n_tile > b_full_n {
            return None; // offset/width out of the weight — refuse to guess
        }
        (n_tile, n_off, b_full_n)
    };
    // A ROW OFFSET. Normally A's access tile carries the grid `pid` as its first
    // (row) index, so the offload reconstructs all M rows from the stick base
    // (m_row_off=0). The last-token-only rewrite pins that index to a static
    // `arith.constant` (m-1); when so, read the offset and reconstruct a single
    // row at it. `matmul_a_row_offset` returns None for the default `%pid` index.
    let m_row_off = matmul_a_row_offset(mm.operands.first()?, defs).unwrap_or(0);
    Some(MatmulLoopInfo {
        m: a_shape[0],
        k: a_shape[1],
        n,
        a_root,
        b_root,
        out_ssa: forop.result.clone()?,
        n_off,
        b_stride,
        transpose_b,
        m_row_off,
    })
}

/// The constant ROW-axis (first index) offset of a matmul A operand's access tile.
/// `name` is the matmul's A operand (a `ktdp.load` of `construct_access_tile
/// %view[%row, %k]`). Returns the row index's `arith.constant` value, or `None`
/// when it isn't a static constant (the default: A's row index is the grid `pid`,
/// so the offload reconstructs all M rows from the stick base — offset 0).
fn matmul_a_row_offset(name: &str, defs: &HashMap<String, &Operation>) -> Option<i64> {
    let d = defs.get(strip(name))?;
    if d.op_type != "ktdp.load" {
        return None;
    }
    let tile = d.operands.first()?;
    let tile_op = defs.get(strip(tile))?;
    // construct_access_tile operands: [view, row_idx, k_idx]. The row index is the
    // first index operand (operand 1).
    let idx = tile_op.operands.get(1)?;
    let cd = defs.get(strip(idx))?;
    if cd.op_type != "arith.constant" {
        return None;
    }
    match cd.attributes.get("value") {
        Some(Attr::Int(i)) => Some(*i),
        _ => None,
    }
}

/// The constant N-axis offset of a matmul B operand's access tile, for an N-tiled
/// weight load. `name` is the matmul's B operand (a `ktdp.load`); its access tile
/// `construct_access_tile %view, %i0, %i1` carries the offset as one of its index
/// operands. `last=true` reads the LAST index (plain `matmul`, B `[k,n]` — N is the
/// column axis); `last=false` reads the FIRST index (transpose-B, B `[n,k]`
/// — N is the row axis). Returns that index's `arith.constant` value, or `None` if
/// B isn't a weight load or the index isn't a static constant.
fn matmul_b_axis_offset(name: &str, defs: &HashMap<String, &Operation>, last: bool) -> Option<i64> {
    let d = defs.get(strip(name))?;
    if d.op_type != "ktdp.load" {
        return None; // forwarded activation N-tile: not handled
    }
    let tile = d.operands.first()?;
    let tile_op = defs.get(strip(tile))?;
    // construct_access_tile operands: [view, idx0, idx1, ...]. The N-axis index is
    // the last operand for a `[k,n]` view, the first index operand (after `view`)
    // for a transposed `[n,k]` view.
    let idx = if last {
        tile_op.operands.last()?
    } else {
        tile_op.operands.get(1)?
    };
    let cd = defs.get(strip(idx))?;
    if cd.op_type != "arith.constant" {
        return None;
    }
    match cd.attributes.get("value") {
        Some(Attr::Int(i)) => Some(*i),
        _ => None,
    }
}

/// Resolve a matmul operand to (resident-root SSA/ptr name, FULL 2-D shape).
/// A forwarded activation is `tensor.extract_slice %src[..]` -> the full src
/// tensor; a weight is `ktdp.load` of an access tile -> its memory view's full
/// shape. The per-iteration tile (the [1,64] slice) is intentionally ignored —
/// we reconstruct the whole GEMM.
fn matmul_operand_full(
    name: &str,
    defs: &HashMap<String, &Operation>,
) -> Option<(String, Vec<i64>)> {
    let d = defs.get(strip(name))?;
    match d.op_type.as_str() {
        "tensor.extract_slice" => {
            let src = d.operands.first()?;
            let shape = shape_attr_vec(defs.get(strip(src)).copied())?;
            Some((src.clone(), shape))
        }
        "ktdp.load" => {
            let tile = d.operands.first()?;
            let view = defs.get(strip(tile))?.operands.first()?;
            let vd = defs.get(strip(view))?;
            let root = vd.operands.first()?.clone();
            let shape = shape_attr_vec(Some(vd))?;
            Some((root, shape))
        }
        _ => None,
    }
}

// =========================================================================
// Runtime map-region fusion — wire the MLX-style elementwise codegen into the
// EXECUTION path. `plan_kernels` carves a fused function's op stream into Map
// windows; here each Map window is compiled to ONE fused MSL kernel and run on
// the GPU at runtime (the analogue of the matmul-loop offload), instead of the
// interpreter running its ops one-by-one. A window with !=1 live-out (or any op
// we can't lower) returns Err so those ops stay on the interpreter.
// =========================================================================

/// A Map window compiled to one fused GPU kernel: the MSL, the window's external
/// inputs (the buffers the kernel reads, in `[[buffer(i)]]` order — original SSA
/// names with `%`), the single window result it produces, and that result's
/// shape/dtype for the output tile.
#[derive(Clone, Debug)]
pub struct MapRegionKernel {
    pub kernel: MslKernel,
    /// External inputs in buffer order: load results, prior-region outputs (e.g.
    /// the reduce sum), or any value defined outside the window. Original SSA
    /// (with `%`) so the runtime resolves them from the value table.
    pub live_ins: Vec<String>,
    /// The single window result consumed outside the window (original SSA, `%`).
    pub live_out: String,
    pub out_shape: Vec<usize>,
    pub out_dtype: DType,
}

/// Compile a Map window (the `window` op indices into `ops`, from `plan_kernels`)
/// into one fused MSL kernel. Computes the window's single LIVE-OUT (the result
/// used by any op OUTSIDE the window), then lowers from its defining op into one
/// MSL expression over `gid`, collecting the external LIVE-INS as buffer leaves.
/// `Err` (≠1 live-out, or an unlowerable op) leaves the window on the interpreter.
pub fn emit_map_region_kernel(
    ops: &[Operation],
    window: &[usize],
) -> Result<MapRegionKernel, String> {
    // Standalone entry: build the function-wide def map and use map once, then
    // delegate. `map_fusion_plan` shares one prebuilt pair across all windows so
    // the (linear-in-ops) analysis isn't redone per window.
    let defs = def_map_all(ops);
    let uses = build_uses(ops);
    emit_map_region_kernel_with(ops, window, &defs, &uses)
}

/// `emit_map_region_kernel` with the function-wide def map and use map supplied
/// by the caller (so a whole-function plan builds them once, not per window).
/// `uses[name]` = the set of TOP-LEVEL op indices that reference `name` (operands
/// or SSA string attrs, counting uses nested in that op's regions).
fn emit_map_region_kernel_with(
    ops: &[Operation],
    window: &[usize],
    defs: &HashMap<String, &Operation>,
    uses: &HashMap<String, HashSet<usize>>,
) -> Result<MapRegionKernel, String> {
    if window.is_empty() {
        return Err("metal: empty map window".into());
    }
    let win_set: HashSet<usize> = window.iter().copied().collect();

    // LIVE-OUT: a window result used by any op OUTSIDE the window. With the
    // prebuilt `uses` map this is a per-result lookup (not a scan of all ops).
    let mut live_outs: Vec<String> = Vec::new();
    for &i in window {
        let Some(r) = ops[i].result.as_deref() else {
            continue;
        };
        let name = strip(r);
        if let Some(idxs) = uses.get(name)
            && idxs.iter().any(|u| !win_set.contains(u))
        {
            live_outs.push(name.to_string());
        }
    }
    if live_outs.len() != 1 {
        return Err(format!(
            "metal: map window has {} live-outs (need exactly 1) — stays on interpreter",
            live_outs.len()
        ));
    }
    // Result SSA (stripped) produced by an op in this window — the recursion
    // boundary for lowering (an operand in this set is in-window).
    let in_window: HashSet<String> = window
        .iter()
        .filter_map(|&i| ops[i].result.as_deref().map(|r| strip(r).to_string()))
        .collect();
    let live_out_name = live_outs.into_iter().next().unwrap();
    let root = *defs
        .get(live_out_name.as_str())
        .ok_or("metal: map window live-out has no defining op")?;

    // Lower the live-out's defining op into one MSL expression, accumulating the
    // external live-in buffers (original SSA, first-seen order).
    let mut live_ins: Vec<String> = Vec::new();
    let expr = lower_map_compute(root, defs, &in_window, &mut live_ins, 0)?;

    // out shape/dtype from the live-out op's attrs (default f16, the KTIR tile dtype).
    let out_shape: Vec<usize> = shape_attr_vec(Some(root))
        .ok_or("metal: map window live-out has no shape attribute")?
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let out_dtype = match root.attributes.get("dtype") {
        Some(Attr::Str(dt)) => DType::parse(dt).unwrap_or(DType::F16),
        _ => DType::F16,
    };

    // One f32 buffer per live-in (in order), then the f32 output. We read/write
    // f32 throughout (the kernel computes in float and the resident tiles are
    // f32-backed), so encode/decode are no-ops and there is no half rounding in
    // the I/O — the per-step f16 rounding the oracle does is captured by writing
    // the result Tile via `Tile::compute(.., out_dtype, ..)` in `run_map_region_gpu`.
    let mut buffers: Vec<BufferBinding> = Vec::with_capacity(live_ins.len() + 1);
    for name in &live_ins {
        buffers.push(BufferBinding {
            name: strip(name).to_string(),
            is_output: false,
            dtype: DType::F32,
        });
    }
    let kname = format!("map_region_{}", strip(&live_out_name));
    buffers.push(BufferBinding {
        name: strip(&live_out_name).to_string(),
        is_output: true,
        dtype: DType::F32,
    });
    let source = render_kernel(&kname, &buffers, &expr);
    Ok(MapRegionKernel {
        kernel: MslKernel {
            source,
            name: kname,
            buffers,
        },
        live_ins,
        live_out: live_out_name,
        out_shape,
        out_dtype,
    })
}

/// Build `name (no %) -> set of TOP-LEVEL op indices that reference it`, in one
/// linear pass over the function. A name is "referenced" by a top-level op if it
/// appears in that op's operands or SSA-bearing string attributes, OR in any op
/// nested in its regions (so a value consumed only inside an `scf.for` body
/// counts as used by the loop's top-level index). This is the prebuilt index the
/// per-window live-out check consults — mirrors the use-counting in
/// `comm_sched::compute_dies_at`, just keyed name -> indices instead of last-use.
fn build_uses(ops: &[Operation]) -> HashMap<String, HashSet<usize>> {
    fn note(op: &Operation, top_idx: usize, uses: &mut HashMap<String, HashSet<usize>>) {
        for operand in &op.operands {
            if operand.starts_with('%') {
                uses.entry(strip(operand).to_string())
                    .or_default()
                    .insert(top_idx);
            }
        }
        for attr in op.attributes.values() {
            match attr {
                Attr::Str(s) if s.starts_with('%') => {
                    uses.entry(strip(s).to_string())
                        .or_default()
                        .insert(top_idx);
                }
                Attr::StrList(xs) => {
                    for x in xs {
                        if x.starts_with('%') {
                            uses.entry(strip(x).to_string())
                                .or_default()
                                .insert(top_idx);
                        }
                    }
                }
                _ => {}
            }
        }
        for region in &op.regions {
            for inner in region {
                note(inner, top_idx, uses);
            }
        }
    }
    let mut uses: HashMap<String, HashSet<usize>> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        note(op, i, &mut uses);
    }
    uses
}

/// Lower an in-window compute op into an MSL expression over `gid`, recursing
/// through in-window operands and turning external operands into live-in buffer
/// leaves. The map-region twin of [`lower_compute_depth`]: same operator table
/// (via [`compose_compute_expr`]), different leaf rule — the leaf is decided by
/// window membership, not by "is it a load".
fn lower_map_compute(
    op: &Operation,
    defs: &HashMap<String, &Operation>,
    in_window: &HashSet<String>,
    live_ins: &mut Vec<String>,
    depth: usize,
) -> Result<String, String> {
    compose_compute_expr(op, &mut |i: usize| -> Result<String, String> {
        let name = op
            .operands
            .get(i)
            .ok_or_else(|| format!("metal: {} missing operand {i}", op.op_type))?;
        lower_map_value(name, defs, in_window, live_ins, depth)
    })
}

/// Resolve one operand SSA name to its MSL sub-expression inside a map window.
///   * an `arith.constant` -> folded literal,
///   * a `tensor.splat` -> transparent (lower its scalar operand),
///   * a `linalg.broadcast` whose input is external/a load -> buffer leaf with the
///     broadcast index expr; if its input is an in-window scalar -> recurse,
///   * any other in-window compute op -> recurse,
///   * anything else (a load, a value from outside the window, the reduce sum)
///     -> a LIVE-IN buffer leaf, read `buf[0]` if scalar else `buf[gid]`.
fn lower_map_value(
    name: &str,
    defs: &HashMap<String, &Operation>,
    in_window: &HashSet<String>,
    live_ins: &mut Vec<String>,
    depth: usize,
) -> Result<String, String> {
    if depth > MAX_FUSE_DEPTH {
        return Err("metal: fused map expression exceeds max depth".into());
    }
    let key = strip(name);
    let in_win = in_window.contains(key);
    match defs.get(key) {
        // Constants and splats fold/are transparent regardless of window
        // membership (they're scheduling plumbing, never window members).
        Some(d) if d.op_type == "arith.constant" => constant_literal(d),
        Some(d) if d.op_type == "tensor.splat" => {
            // Splat is transparent (every lane reads the same scalar sub-expr);
            // parenthesize so a compound scalar keeps precedence in its parent.
            let inner = d
                .operands
                .first()
                .ok_or("metal: tensor.splat missing operand")?;
            Ok(format!(
                "({})",
                lower_map_value(inner, defs, in_window, live_ins, depth + 1)?
            ))
        }
        Some(d) if d.op_type == "linalg.broadcast" => {
            let input = d
                .operands
                .first()
                .ok_or("metal: linalg.broadcast missing ins operand")?;
            // Broadcasting a scalar constant (directly, or via a splat) is
            // transparent — every output element is that scalar. Fold it so the
            // constant never becomes a (would-be-Scalar-at-runtime) live-in.
            if let Some(lit) = try_fold_scalar(input, defs) {
                return Ok(lit);
            }
            let input_in_win = in_window.contains(strip(input));
            let input_is_load = defs
                .get(strip(input))
                .is_some_and(|x| x.op_type == "ktdp.load");
            // The input is SCALAR (shape product == 1) iff broadcasting it is a
            // pure splat: every output lane reads the one value. Only then is
            // recursing into an in-window input correct — the recursed expression
            // reads its leaves at fixed (scalar) indices, valid for every `gid`.
            let input_scalar = shape_attr_vec(defs.get(strip(input)).copied())
                .map(|s| s.iter().product::<i64>() == 1)
                .unwrap_or(false);
            if input_in_win && !input_is_load {
                // Recursing into an in-window scalar input gives the MLX scalar-tail
                // fusion (e.g. RMSNorm's `1/rms` folded into the final multiply).
                // Parenthesized to preserve precedence in the parent expression.
                if input_scalar {
                    Ok(format!(
                        "({})",
                        lower_map_value(input, defs, in_window, live_ins, depth + 1)?
                    ))
                } else {
                    // A rank-reducing broadcast of an in-window NON-scalar value
                    // (e.g. prefill's per-row `inv[8]` broadcast to `[8,576]`)
                    // can't be inlined: the recursed expression would index the
                    // lower-rank value by the output `gid` (out of bounds). The
                    // value would have to be materialized first, which it isn't in
                    // this window — so fail and leave the window to the interpreter.
                    Err(format!(
                        "metal: rank-reducing broadcast of in-window value {input} \
                         (not scalar) — window stays on interpreter"
                    ))
                }
            } else {
                // An external (load / prior-region) input: read it through the
                // broadcast index expression (a materialized buffer leaf).
                lower_map_broadcast(d, input, defs, live_ins)
            }
        }
        // An in-window compute op: descend, PARENTHESIZED so the inlined
        // sub-expression keeps its precedence inside the parent op (e.g. SiLU's
        // `v2 / (1 + exp(-v2))` must not flatten to `v2 / 1 + exp(-v2)`). Mirrors
        // the parenthesizing in `lower_value`.
        Some(d) if in_win => Ok(format!(
            "({})",
            lower_map_compute(d, defs, in_window, live_ins, depth + 1)?
        )),
        // A scalar constant (directly or via splat) reached as a plain operand
        // folds to its literal — it would be a `Value::Scalar` at runtime, not a
        // resident tile, so it must never become a live-in buffer.
        _ if try_fold_scalar(name, defs).is_some() => Ok(try_fold_scalar(name, defs).unwrap()),
        // A non-constant SCALAR value (e.g. an `arith.maximumf : f16` from an
        // attention softmax's scalar max/sum reduction, or any non-tensor op
        // result) is a `Value::Scalar` at runtime — it can't be bound as a tile
        // buffer and we can't fold it. Fail the window so it stays on the
        // interpreter (correctness over fusion).
        Some(d) if !is_tensor_valued(d) => Err(format!(
            "metal: map window needs scalar value {name} (def {}) — not a resident tile; \
             window stays on interpreter",
            d.op_type
        )),
        // Any other value (a load result, a value produced outside the window —
        // e.g. a prior region's output or the reduce sum) is an external input.
        _ => Ok(map_live_in_leaf(name, defs, live_ins)),
    }
}

/// Whether an op produces a tensor value (a resident `Tile` at runtime), vs a
/// scalar (`Value::Scalar`). The parser attaches a `shape` attribute exactly for
/// tensor/memref result types, so its presence is the tensor test. A would-be
/// live-in without a shape is a scalar that can't be bound as a kernel buffer.
fn is_tensor_valued(op: &Operation) -> bool {
    op.attributes.contains_key("shape")
}

/// If `name` is a scalar `arith.constant` (folded to its MSL literal) or a
/// `tensor.splat` of one (recursively), return that literal. `None` otherwise.
/// A scalar constant is a `Value::Scalar` at runtime, never a resident tile, so
/// it must be folded into the expression rather than bound as a live-in buffer.
fn try_fold_scalar(name: &str, defs: &HashMap<String, &Operation>) -> Option<String> {
    let d = defs.get(strip(name))?;
    match d.op_type.as_str() {
        "arith.constant" => constant_literal(d).ok(),
        "tensor.splat" => try_fold_scalar(d.operands.first()?, defs),
        _ => None,
    }
}

/// Emit a live-in buffer leaf for `name`: register it (dedup, original SSA) and
/// return `buf[0]` if it's a scalar (shape product == 1, e.g. the reduce sum),
/// else `buf[gid]`. `buf` is the sanitized identifier the buffer binding uses.
fn map_live_in_leaf(
    name: &str,
    defs: &HashMap<String, &Operation>,
    live_ins: &mut Vec<String>,
) -> String {
    let orig = if name.starts_with('%') {
        name.to_string()
    } else {
        format!("%{name}")
    };
    if !live_ins.contains(&orig) {
        live_ins.push(orig);
    }
    let buf = strip(name).to_string();
    let scalar = shape_attr_vec(defs.get(strip(name)).copied())
        .map(|s| s.iter().product::<i64>() == 1)
        .unwrap_or(false);
    if scalar {
        format!("{buf}[0]")
    } else {
        format!("{buf}[gid]")
    }
}

/// Lower a `linalg.broadcast` whose input is an external buffer to `buf[idx]`,
/// where `idx = broadcast_index_expr(out_shape, expanded_in)` maps the output
/// `gid` to the input element. Registers `input` as a live-in (the buffer named
/// `buf`). Mirrors [`lower_broadcast`] but takes the input as a live-in rather
/// than tracing it to a pointer argument.
fn lower_map_broadcast(
    op: &Operation,
    input: &str,
    defs: &HashMap<String, &Operation>,
    live_ins: &mut Vec<String>,
) -> Result<String, String> {
    let in_shape = shape_attr_vec(defs.get(strip(input)).copied())
        .ok_or("metal: broadcast input has no shape")?;
    let out_shape = shape_attr_vec(Some(op)).ok_or("metal: broadcast has no output shape")?;
    let mut dims = int_list_attr_vec(op, "dimensions").unwrap_or_default();
    dims.sort_unstable();
    let mut expanded = in_shape;
    for &d in &dims {
        let d = d as usize;
        if d > expanded.len() {
            return Err(format!("metal: broadcast dim {d} out of range"));
        }
        expanded.insert(d, 1);
    }
    // Register the input buffer (dedup, original SSA).
    let orig = if input.starts_with('%') {
        input.to_string()
    } else {
        format!("%{input}")
    };
    if !live_ins.contains(&orig) {
        live_ins.push(orig);
    }
    let buf = strip(input).to_string();
    Ok(format!(
        "{buf}[{}]",
        broadcast_index_expr(&out_shape, &expanded)
    ))
}

/// Count of Map windows successfully offloaded to a fused GPU kernel (test /
/// telemetry proof the fused map path actually used Metal). Mirrors
/// [`MATMUL_LOOP_GPU_COUNT`].
#[cfg(metal)]
pub static MAP_REGION_GPU_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Run a compiled Map window as one fused GPU kernel, binding its live-out tile
/// in `ctx`. Reads each live-in's f32 data from the value table (must be a
/// resident `Tile`), dispatches the kernel (one thread per output element), and
/// binds the result as a `Tile` of `out_dtype`/`out_shape`. Returns `Err` if a
/// live-in is missing/not a tile or the kernel can't run — the caller treats a
/// trigger failure as fatal (the window's ops were skipped on the interpreter).
#[cfg(metal)]
pub fn run_map_region_gpu(
    mrk: &MapRegionKernel,
    ctx: &mut crate::context::CoreContext,
) -> Result<(), String> {
    let out_len: usize = mrk.out_shape.iter().product();
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(mrk.live_ins.len());
    for name in &mrk.live_ins {
        match ctx.get_value(name)? {
            crate::ir::Value::Tile(t) => inputs.push(t.as_f32().to_vec()),
            // A SCALAR live-in. Two flavors reach here, both read as a broadcast:
            //   * `tensor.extract` of a reduce result (softmax/layernorm row max /
            //     mean) — a `Value::Scalar` even though MLIR's `extract :
            //     tensor<1xf16>` syntax tags it shape `[1]` (emitter reads `buf[0]`).
            //   * `arith.constant dense<0.0> : tensor<1x1024xf16>` — a tensor-typed
            //     accumulator init that the interpreter binds as a scalar; the
            //     emitter (seeing its multi-element shape) reads it as `buf[gid]`.
            // Filling the scalar to the FULL out_len is correct for BOTH reads
            // (`buf[0]` and every `buf[gid]` see the same value) and avoids an
            // out-of-bounds `buf[gid]` for the tensor-shaped constant case.
            crate::ir::Value::Scalar(s) => {
                let v = match *s {
                    crate::ir::Scalar::F32(v) => v,
                    crate::ir::Scalar::I32(v) => v as f32,
                    crate::ir::Scalar::I64(v) => v as f32,
                    crate::ir::Scalar::Bool(b) => b as i32 as f32,
                };
                inputs.push(vec![v; out_len.max(1)]);
            }
            crate::ir::Value::Index(i) => inputs.push(vec![*i as f32; out_len.max(1)]),
            other => {
                return Err(format!(
                    "metal: map-region live-in {name} is {other:?}, expected a resident tile or scalar"
                ));
            }
        }
    }
    let out = run_kernel(&mrk.kernel, &inputs, out_len)?;
    let tile = crate::tile::Tile::compute(out, mrk.out_dtype, mrk.out_shape.clone());
    let bytes = tile.size_bytes() as i64;
    // Consume-on-last-use for the window's inputs BEFORE charging the output —
    // the same order `execute_op` uses (#134): the fused kernel has read every
    // live-in above, so a single-use, current-generation live-in tile is freed
    // here so the output can reuse its LX at no net increase. Without this the
    // per-row window output is charged ALONGSIDE its inputs and a wide row
    // (softmax_wide's 512 KB rows) overflows the 2 MB LX budget. No-op for
    // multi-use / outer-scope live-ins (a value also read by a reduce), so it is
    // a strict subset of what the interpreter would free at this point.
    for name in &mrk.live_ins {
        ctx.consume_if_last_use(name);
    }
    ctx.set_value(&mrk.live_out, crate::ir::Value::Tile(tile));
    ctx.track_lx(&mrk.live_out, bytes)?;
    MAP_REGION_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Default minimum output element count to offload a fused map window to the GPU.
/// Below this the per-window GPU dispatch+sync + live-in upload costs more than
/// the interpreter's elementwise loop. Decode windows are M=1 (≤2048 elems) — a
/// net loss; prefill windows are M=8/32 (up to ~64k elems) — a win. 16384 splits
/// them. Override with `KTIR_MAP_GPU_MIN_ELEMS` (0 = offload every window, the old
/// always-GPU behavior).
#[cfg(metal)]
pub const MAP_GPU_MIN_ELEMS: usize = 16_384;

/// The map-window GPU offload size threshold (env-overridable). See
/// [`MAP_GPU_MIN_ELEMS`].
#[cfg(metal)]
pub fn map_gpu_min_elems() -> usize {
    std::env::var("KTIR_MAP_GPU_MIN_ELEMS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(MAP_GPU_MIN_ELEMS)
}

/// Whether `KTIR_FORCE_GPU_MAP` is set (any non-empty value): FORCE the fused
/// map-window GPU offload even for MULTI-CORE grids (which the scheduler's default
/// `gpu_offload` gate restricts to single-core functions, because a multi-core
/// matmul-loop reconstruction would be wrong — but a per-element elementwise MAP
/// is core-local and identical whatever the grid, so it is always safe to offload
/// per core). The differential conformance harness uses this to prove the
/// elementwise example programs (`softmax`/`layernorm`/`vector_add`, all native
/// grid `[32,1]`) actually run their maps on the Metal map kernel rather than the
/// interpreter. It does NOT enable the multi-core matmul-loop offload (still gated
/// on `num_cores == 1`); it ONLY lifts the gate for the map plan. Pair with
/// `KTIR_MAP_GPU_MIN_ELEMS=0` to also offload windows below the dispatch floor.
/// Never on by default (per-window dispatch is a net loss for tiny windows — purely
/// a conformance-check override). See [`crate::comm_sched`] `gpu_map_offload`.
#[cfg(metal)]
pub fn force_gpu_map() -> bool {
    std::env::var_os("KTIR_FORCE_GPU_MAP").is_some_and(|v| !v.is_empty())
}

/// Plan a fused function's Map windows for runtime GPU offload. Walks the op
/// stream with the SAME classification `plan_kernels` uses — Map ops accumulate
/// into a window; a Reduce / Matmul / Boundary / `scf.for` flushes it; plumbing
/// doesn't break a window — and for each window TRIES [`emit_map_region_kernel`].
/// On success (and if the window's output clears the [`MAP_GPU_MIN_ELEMS`] size
/// gate) the window registers its TRIGGER (the last op index — run the kernel
/// there) and all its op indices in the SKIP set; otherwise the window's ops are
/// left to the interpreter. Returns `(trigger -> kernel, skip set)`.
pub type MapRegionPlan = (HashMap<usize, MapRegionKernel>, HashSet<usize>);

pub fn map_fusion_plan(ops: &[Operation]) -> MapRegionPlan {
    // Build the function-wide def map and use map ONCE; share them across every
    // window's emit (the analysis is linear in ops, so per-window rebuilds would
    // make planning quadratic in a 35k-op fused model).
    let defs = def_map_all(ops);
    let uses = build_uses(ops);
    let mut triggers: HashMap<usize, MapRegionKernel> = HashMap::new();
    let mut skip: HashSet<usize> = HashSet::new();
    let mut window: Vec<usize> = Vec::new();
    let min_elems = map_gpu_min_elems();
    let try_flush = |window: &mut Vec<usize>,
                     triggers: &mut HashMap<usize, MapRegionKernel>,
                     skip: &mut HashSet<usize>| {
        if window.is_empty() {
            return;
        }
        let w = std::mem::take(window);
        if let Ok(mrk) = emit_map_region_kernel_with(ops, &w, &defs, &uses) {
            // SIZE GATE: a fused map kernel pays a GPU dispatch+sync round-trip
            // and uploads each live-in tile; below `min_elems` output elements the
            // interpreter's elementwise loop is faster (decode's M=1 windows are
            // ≤2048 elems — a net loss on GPU). Leaving the window OUT of the skip
            // set means the interpreter runs its ops normally (no fatal trigger).
            let out_len: usize = mrk.out_shape.iter().product();
            if out_len >= min_elems {
                let trigger = *w.last().unwrap();
                for &i in &w {
                    skip.insert(i);
                }
                triggers.insert(trigger, mrk);
            }
        }
    };
    for (i, op) in ops.iter().enumerate() {
        // scf.for: same boundary treatment as plan_kernels (a matmul K-loop or an
        // unfusable loop both flush the current map window).
        if op.op_type == "scf.for" {
            try_flush(&mut window, &mut triggers, &mut skip);
            continue;
        }
        match classify(op) {
            // A SCALAR map op (e.g. an attention softmax's `arith.maximumf : f16`
            // row-max, or a scalar `arith.addf`) produces a `Value::Scalar`, not a
            // resident tile — it can't be a buffer leaf and the kernel is a
            // per-element tensor map. Treat it as a boundary: flush the current
            // tensor window and let the scalar op run on the interpreter (so the
            // surrounding tensor windows still fuse, rather than the whole window
            // failing because one scalar op snuck in).
            OpClass::Map if is_tensor_valued(op) => {
                window.push(i);
                if window.len() >= MAX_KERNEL_WINDOW {
                    try_flush(&mut window, &mut triggers, &mut skip);
                }
            }
            OpClass::Map | OpClass::Reduce | OpClass::Matmul | OpClass::Boundary => {
                try_flush(&mut window, &mut triggers, &mut skip);
            }
            OpClass::Plumbing => {} // traced by the emitter; doesn't break a window
        }
    }
    try_flush(&mut window, &mut triggers, &mut skip);
    (triggers, skip)
}

/// Lower `func_name` to a full [`MslKernel`] (source + buffer bindings).
pub fn emit_kernel(module: &IRModule, func_name: &str) -> Result<MslKernel, String> {
    let f = module.get_function(func_name)?;
    let defs = def_map(f);

    // The kernel's "root" is its single store: `ktdp.store %value, %access_tile`.
    let store = f
        .operations
        .iter()
        .find(|o| o.op_type == "ktdp.store")
        .ok_or("metal: no ktdp.store — only element-wise store kernels are supported in slice 1")?;
    if store.operands.len() < 2 {
        return Err("metal: ktdp.store needs (value, access_tile) operands".into());
    }
    let out_buf = trace_buffer(&store.operands[1], &defs)
        .ok_or("metal: could not trace the store target back to a pointer argument")?;

    // The stored value must come from a single element-wise compute op whose
    // operands are loaded tiles.
    let compute = defs
        .get(strip(&store.operands[0]))
        .ok_or("metal: stored value has no defining op")?;
    let expr = lower_compute(compute, &defs)?;
    let dtype = buffer_dtype(&out_buf, f);

    // Inputs in first-seen order; the output buffer last. (De-dup: a buffer may
    // be both read and written, though vector_add's aren't.)
    let mut buffers: Vec<BufferBinding> = Vec::new();
    for b in collect_input_buffers(compute, &defs) {
        if !buffers.iter().any(|x| x.name == b) {
            let bdt = buffer_dtype(&b, f);
            buffers.push(BufferBinding {
                name: b,
                is_output: false,
                dtype: bdt,
            });
        }
    }
    buffers.push(BufferBinding {
        name: out_buf,
        is_output: true,
        dtype,
    });

    let source = render_kernel(func_name, &buffers, &expr);
    Ok(MslKernel {
        source,
        name: func_name.to_string(),
        buffers,
    })
}

// --- dataflow ------------------------------------------------------------

/// `result-name (no %) -> defining op`.
fn def_map(f: &IRFunction) -> HashMap<String, &Operation> {
    let mut m = HashMap::new();
    for op in &f.operations {
        if let Some(r) = &op.result {
            m.insert(strip(r).to_string(), op);
        }
    }
    m
}

fn strip(name: &str) -> &str {
    name.trim_start_matches('%')
}

/// Follow an SSA value back to the pointer-argument buffer it ultimately reads
/// or writes: `load`/`store` access tile -> `construct_access_tile` -> its view
/// -> `construct_memory_view` -> the `%ptr` argument. Returns the arg name.
fn trace_buffer(name: &str, defs: &HashMap<String, &Operation>) -> Option<String> {
    let mut cur = strip(name).to_string();
    // Walk defining ops until we hit a name with no def (a function argument).
    for _ in 0..16 {
        let Some(op) = defs.get(cur.as_str()) else {
            return Some(cur); // no def -> it's a function argument (the pointer)
        };
        // Each of these ops carries the thing-we-want as operand 0.
        match op.op_type.as_str() {
            "ktdp.construct_access_tile" | "ktdp.construct_memory_view" | "ktdp.load" => {
                cur = strip(&op.operands[0]).to_string();
            }
            // Any other defining op isn't part of a load/store->buffer chain.
            _ => return None,
        }
    }
    None
}

/// Every distinct input buffer feeding a fused elementwise expression tree, in
/// first-seen (DFS pre-order) order. Recurses through chained compute ops so a
/// fused kernel binds each loaded buffer once, no matter how deep in the
/// expression it appears.
fn collect_input_buffers(compute: &Operation, defs: &HashMap<String, &Operation>) -> Vec<String> {
    let mut out = Vec::new();
    collect_bufs(compute, defs, &mut out, 0);
    out
}

fn collect_bufs(
    op: &Operation,
    defs: &HashMap<String, &Operation>,
    out: &mut Vec<String>,
    depth: usize,
) {
    if depth > MAX_FUSE_DEPTH {
        return;
    }
    for operand in &op.operands {
        match defs.get(strip(operand)) {
            // A loaded tile is a leaf buffer.
            Some(d) if d.op_type == "ktdp.load" => {
                if let Some(b) = trace_buffer(operand, defs)
                    && !out.contains(&b)
                {
                    out.push(b);
                }
            }
            // A chained compute op: descend into its inputs.
            Some(d) => collect_bufs(d, defs, out, depth + 1),
            None => {}
        }
    }
}

/// Element dtype of a buffer, read from the `construct_memory_view` that
/// produced it. Defaults to `f16` — the common KTIR tile dtype.
fn buffer_dtype(buf: &str, f: &IRFunction) -> DType {
    for op in &f.operations {
        if op.op_type == "ktdp.construct_memory_view"
            && op.operands.first().map(|p| strip(p)) == Some(buf)
            && let Some(crate::ir::Attr::Str(dt)) = op.attributes.get("dtype")
            && let Ok(parsed) = DType::parse(dt)
        {
            return parsed;
        }
    }
    DType::F16
}

/// The MSL scalar type for a KTIR dtype.
fn msl_type(dt: DType) -> &'static str {
    match dt {
        DType::F16 => "half",
        DType::F32 => "float",
        DType::I32 => "int",
        DType::I64 => "long",
        DType::Bool => "bool",
    }
}

// --- compute lowering ----------------------------------------------------

/// Cap on fused-expression nesting — guards against pathological depth (and any
/// accidental cycle) while comfortably covering real elementwise chains.
const MAX_FUSE_DEPTH: usize = 256;

/// Lower an element-wise compute op into an MSL expression over `gid`, recursing
/// through chained compute operands so an entire elementwise DAG collapses into
/// ONE fused expression. Loaded tiles become `<buffer>[gid]` leaves; a chained
/// compute operand becomes a parenthesized sub-expression. This is the core of
/// MLX-style kernel fusion: `load,load,mul,exp,add -> store` lowers to a single
/// `exp(a[gid]*b[gid]) + c[gid]` kernel instead of three passes.
fn lower_compute(op: &Operation, defs: &HashMap<String, &Operation>) -> Result<String, String> {
    lower_compute_depth(op, defs, 0)
}

/// Resolve one operand SSA name to its MSL sub-expression: a loaded tile is a
/// `buf[gid]` leaf; anything else is recursively lowered as a compute op (which
/// errors if it isn't elementwise).
fn lower_value(
    name: &str,
    defs: &HashMap<String, &Operation>,
    depth: usize,
) -> Result<String, String> {
    if depth > MAX_FUSE_DEPTH {
        return Err("metal: fused expression exceeds max depth".into());
    }
    match defs.get(strip(name)) {
        None => Err(format!("metal: operand {name} has no defining op")),
        Some(d) if d.op_type == "ktdp.load" => {
            let buf = trace_buffer(name, defs)
                .ok_or_else(|| format!("metal: operand {name} is not a loaded buffer"))?;
            Ok(format!("{buf}[gid]"))
        }
        // A broadcast reads its (buffer) input at a gid-derived index that
        // repeats along the broadcast axes — `w[gid % N]` for a per-column
        // weight, `s[0]` for a scalar. The input must trace to a loaded buffer
        // (a computed value broadcast across a reduction is a separate kernel).
        Some(d) if d.op_type == "linalg.broadcast" => lower_broadcast(d, defs),
        Some(d) => Ok(format!("({})", lower_compute_depth(d, defs, depth + 1)?)),
    }
}

/// Lower `linalg.broadcast ins(%x) outs(%init) dimensions=[..]` to `buf[idx]`,
/// where `idx` maps the kernel's flat `gid` (over the broadcast's output shape)
/// to the input buffer's element, holding the broadcast axes constant.
fn lower_broadcast(op: &Operation, defs: &HashMap<String, &Operation>) -> Result<String, String> {
    let input = op
        .operands
        .first()
        .ok_or("metal: linalg.broadcast missing ins operand")?;
    let buf = trace_buffer(input, defs).ok_or(
        "metal: broadcast input must be a loaded buffer (a value broadcast across a \
         reduction is a separate kernel)",
    )?;
    let in_shape = shape_attr_vec(defs.get(strip(input)).copied())
        .ok_or("metal: broadcast input has no shape")?;
    let out_shape = shape_attr_vec(Some(op)).ok_or("metal: broadcast has no output shape")?;
    let mut dims = int_list_attr_vec(op, "dimensions").unwrap_or_default();
    dims.sort_unstable();

    // Expanded input shape = in_shape with a size-1 axis inserted at each
    // (sorted) broadcast dimension — rank now matches the output.
    let mut expanded = in_shape;
    for &d in &dims {
        let d = d as usize;
        if d > expanded.len() {
            return Err(format!("metal: broadcast dim {d} out of range"));
        }
        expanded.insert(d, 1);
    }
    Ok(format!(
        "{buf}[{}]",
        broadcast_index_expr(&out_shape, &expanded)
    ))
}

/// MSL index into a broadcast input: sum over axes whose expanded input size is
/// > 1 of `coord(axis) * input_stride`, where `coord(axis) = (gid / out_stride)
/// % out_dim`. Size-1 (broadcast) axes contribute nothing. Empty sum -> "0".
fn broadcast_index_expr(out_shape: &[i64], expanded_in: &[i64]) -> String {
    let r = out_shape.len();
    let mut terms: Vec<String> = Vec::new();
    for k in 0..r {
        if expanded_in.get(k).copied().unwrap_or(1) <= 1 {
            continue; // broadcast axis: contributes 0
        }
        let out_stride: i64 = out_shape[k + 1..].iter().product();
        let in_stride: i64 = expanded_in[k + 1..].iter().product();
        let coord = if out_stride == 1 {
            "gid".to_string()
        } else {
            format!("(gid / {out_stride})")
        };
        let coord = format!("({coord} % {})", out_shape[k]);
        terms.push(if in_stride == 1 {
            coord
        } else {
            format!("{coord} * {in_stride}")
        });
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// Read an op's `shape` attribute as an `i64` vector.
fn shape_attr_vec(op: Option<&Operation>) -> Option<Vec<i64>> {
    match op?.attributes.get("shape") {
        Some(crate::ir::Attr::IntList(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Read a named `IntList` attribute as an `i64` vector.
fn int_list_attr_vec(op: &Operation, key: &str) -> Option<Vec<i64>> {
    match op.attributes.get(key) {
        Some(crate::ir::Attr::IntList(v)) => Some(v.clone()),
        _ => None,
    }
}

fn lower_compute_depth(
    op: &Operation,
    defs: &HashMap<String, &Operation>,
    depth: usize,
) -> Result<String, String> {
    compose_compute_expr(op, &mut |i: usize| -> Result<String, String> {
        let name = op
            .operands
            .get(i)
            .ok_or_else(|| format!("metal: {} missing operand {i}", op.op_type))?;
        lower_value(name, defs, depth)
    })
}

/// The shared op-type -> MSL-expression table, parameterized over how an operand
/// resolves to its MSL sub-expression (`resolve(i)`). Both the store-rooted
/// elementwise lowering ([`lower_compute_depth`]) and the window-rooted map-region
/// lowering ([`lower_map_compute`]) compose through here, so the operator set —
/// and thus the precision/casting semantics — stays identical between them.
fn compose_compute_expr(
    op: &Operation,
    resolve: &mut dyn FnMut(usize) -> Result<String, String>,
) -> Result<String, String> {
    let operand = |i: usize, r: &mut dyn FnMut(usize) -> Result<String, String>| r(i);
    // Binary element-wise float ops -> infix operator.
    let binop =
        |sym: &str, r: &mut dyn FnMut(usize) -> Result<String, String>| -> Result<String, String> {
            Ok(format!("{} {} {}", operand(0, r)?, sym, operand(1, r)?))
        };
    // Unary math ops -> MSL intrinsic call.
    let unary = |func: &str,
                 r: &mut dyn FnMut(usize) -> Result<String, String>|
     -> Result<String, String> { Ok(format!("{func}({})", operand(0, r)?)) };

    match op.op_type.as_str() {
        "arith.addf" => binop("+", resolve),
        "arith.subf" => binop("-", resolve),
        "arith.mulf" => binop("*", resolve),
        "arith.divf" => binop("/", resolve),
        "arith.maximumf" | "arith.maxf" => Ok(format!(
            "max({}, {})",
            operand(0, resolve)?,
            operand(1, resolve)?
        )),
        "arith.minimumf" | "arith.minf" => Ok(format!(
            "min({}, {})",
            operand(0, resolve)?,
            operand(1, resolve)?
        )),
        "arith.negf" => Ok(format!("-{}", operand(0, resolve)?)),
        "arith.absf" | "math.absf" => unary("abs", resolve),
        "math.exp" => unary("exp", resolve),
        "math.log" => unary("log", resolve),
        "math.sqrt" => unary("sqrt", resolve),
        "math.sin" => unary("sin", resolve),
        "math.cos" => unary("cos", resolve),
        "math.tanh" => unary("tanh", resolve),
        "linalg.add" => binop("+", resolve),
        "linalg.mul" => binop("*", resolve),
        "linalg.sub" => binop("-", resolve),
        // A scalar constant folds into the expression as an MSL literal.
        "arith.constant" => constant_literal(op),
        // splat broadcasts a scalar to a tensor; in the per-element kernel it is
        // transparent — every lane reads the same scalar sub-expression.
        "tensor.splat" => operand(0, resolve),
        // dtype casts: compute in the wider type, narrow on store. Explicit so
        // an extf'd chain runs in float (matching the CPU oracle), not half.
        "arith.extf" => Ok(format!("float({})", operand(0, resolve)?)),
        "arith.truncf" => Ok(format!("half({})", operand(0, resolve)?)),
        other => Err(format!(
            "metal: compute op {other:?} not lowerable (element-wise / scalar only)"
        )),
    }
}

/// Render an `arith.constant`'s value as an MSL float literal. Only the
/// float/int scalar forms fold into a fused expression; anything else (a
/// `dense<>` tensor, a bool) is rejected so the caller can fall back.
fn constant_literal(op: &Operation) -> Result<String, String> {
    match op.attributes.get("value") {
        Some(crate::ir::Attr::Float(f)) => Ok(format!("{f:?}")),
        Some(crate::ir::Attr::Int(i)) => Ok(format!("{i}.0")),
        other => Err(format!(
            "metal: constant value {other:?} not lowerable as a scalar literal"
        )),
    }
}

// --- rendering -----------------------------------------------------------

fn render_kernel(name: &str, buffers: &[BufferBinding], expr: &str) -> String {
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    s.push_str(&format!("kernel void {name}(\n"));
    for (i, b) in buffers.iter().enumerate() {
        let qual = if b.is_output {
            "device"
        } else {
            "device const"
        };
        s.push_str(&format!(
            "    {qual} {}* {} [[buffer({i})]],\n",
            msl_type(b.dtype),
            b.name
        ));
    }
    s.push_str("    uint gid [[thread_position_in_grid]]\n) {\n");
    // The output buffer is the last entry.
    let out = &buffers.last().unwrap().name;
    s.push_str(&format!("    {out}[gid] = {expr};\n"));
    s.push_str("}\n");
    s
}

// =========================================================================
// Runtime dispatch (slice 2) — compile the MSL and run it on a Metal device.
// =========================================================================

/// Compile `kernel`'s MSL, upload `inputs` (in `kernel.buffers` non-output
/// order, as f32 — encoded to each buffer's dtype), dispatch one thread per
/// output element, and read `out_len` elements back as f32.
///
/// Returns `Err("no Metal device …")` when no GPU is available (e.g. headless
/// CI), so callers can skip gracefully.
/// Shared per-thread Metal device + queue + compiled-pipeline cache. Without
/// this, `run_kernel` compiled a fresh MTLLibrary+pipeline on EVERY dispatch —
/// fine for the old one-shot kernels, but with map-window fusion a single pass
/// dispatches ~900 kernels, and recompiling each one per pass is both slow and
/// exhausts GPU pipeline objects across many passes. Pipelines are keyed by MSL
/// source hash (the model's repeated layers share identical kernels), so each
/// distinct kernel compiles exactly once per thread.
#[cfg(metal)]
struct MetalDispatch {
    device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipelines: HashMap<
        u64,
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
}

#[cfg(metal)]
thread_local! {
    static METAL_DISPATCH: std::cell::RefCell<Option<MetalDispatch>> =
        const { std::cell::RefCell::new(None) };
}

/// Device + queue + the cached pipeline for `kernel` (compiled on first sight).
#[cfg(metal)]
#[allow(clippy::type_complexity)]
fn cached_dispatch(
    kernel: &MslKernel,
) -> Result<
    (
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    ),
    String,
> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};
    use std::hash::{Hash, Hasher};

    METAL_DISPATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
            let queue = device
                .newCommandQueue()
                .ok_or("metal: newCommandQueue nil")?;
            *slot = Some(MetalDispatch {
                device,
                queue,
                pipelines: HashMap::new(),
            });
        }
        let d = slot.as_mut().unwrap();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        kernel.source.hash(&mut h);
        let key = h.finish();
        if !d.pipelines.contains_key(&key) {
            let opts = objc2_metal::MTLCompileOptions::new();
            let src = NSString::from_str(&kernel.source);
            let library = d
                .device
                .newLibraryWithSource_options_error(&src, Some(&opts))
                .map_err(|e| format!("metal: MSL compile failed: {e:?}"))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(&kernel.name))
                .ok_or_else(|| format!("metal: kernel {:?} not found", kernel.name))?;
            let pipeline = d
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("metal: pipeline build failed: {e:?}"))?;
            d.pipelines.insert(key, pipeline);
        }
        Ok((d.device.clone(), d.queue.clone(), d.pipelines[&key].clone()))
    })
}

pub fn run_kernel(
    kernel: &MslKernel,
    inputs: &[Vec<f32>],
    out_len: usize,
) -> Result<Vec<f32>, String> {
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    // Cached device/queue/pipeline — compiled once per distinct MSL source.
    let (device, queue, pipeline) = cached_dispatch(kernel)?;

    let res = MTLResourceOptions::StorageModeShared;
    let mut gpu_buffers = Vec::with_capacity(kernel.buffers.len());
    let mut input_iter = inputs.iter();
    let mut out_dtype = DType::F16;
    for b in &kernel.buffers {
        let buf = if b.is_output {
            out_dtype = b.dtype;
            let len = (out_len * b.dtype.bytes_per_elem()).max(1);
            device
                .newBufferWithLength_options(len, res)
                .ok_or("metal: output buffer alloc failed")?
        } else {
            let data = input_iter
                .next()
                .ok_or("metal: too few inputs for kernel buffers")?;
            let bytes = crate::codec::encode(data, b.dtype);
            // SAFETY: `bytes` lives until the copy completes inside this call.
            unsafe {
                device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(1),
                        res,
                    )
                    .ok_or("metal: input buffer alloc failed")?
            }
        };
        gpu_buffers.push(buf);
    }

    let cb = queue
        .commandBuffer()
        .ok_or("metal: commandBuffer returned nil")?;
    let enc = cb
        .computeCommandEncoder()
        .ok_or("metal: computeCommandEncoder returned nil")?;
    enc.setComputePipelineState(&pipeline);
    for (i, buf) in gpu_buffers.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
    }
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(out_len).max(1);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: out_len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    // Read the output buffer (last) back and decode to f32.
    let out = gpu_buffers.last().unwrap();
    let nbytes = out_len * out_dtype.bytes_per_elem();
    let raw = unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const u8, nbytes) }
        .to_vec();
    Ok(crate::codec::decode(&raw, out_len, out_dtype))
}

/// Compile MSL source as **Metal 4** (`MTLLanguageVersion::Version4_0`,
/// `MathMode::Safe`) — the options Metal Performance Primitives (`mpp::tensor_ops`,
/// the M5 NAX path) require. Compiled from source at runtime because the offline
/// `xcrun metal` toolchain miscompiles MPP (per scratchy's findings). Returns
/// `Ok(())` if the source compiles on the system device, else the compiler error.
pub fn compile_metal4(source: &str) -> Result<(), String> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLMathMode};

    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
    let opts = objc2_metal::MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

// =========================================================================
// NAX (M5 Neural Accelerator) single-tile GEMM
// =========================================================================
//
// The M5's matmul tier. One simdgroup computes a fixed 16×32×16 output tile
// with `mpp::tensor_ops::matmul2d` — the Metal Performance Primitives op that
// dispatches to the NAX tensor engine. This is the irreducible NAX unit; a
// general GEMM tiles the problem into these (a later slice). It exists now to
// prove the engine produces correct results through our runtime and to measure
// the speedup, gating whether `HIGHEST_IMPLEMENTED` can rise to `Nax`.
//
// Inputs/outputs are host `f32` (row-major); A and B are converted to `half`
// in threadgroup memory inside the shader, so the host never touches f16. The
// op runs with `transpose_b`, so B (logical K×N) is consumed as its transpose
// Bᵀ (N×K) — the fill loop transposes while converting.

/// Fixed NAX tile dims: `C[M×N] = A[M×K] · B[K×N]`.
pub const NAX_TILE_M: usize = 16;
pub const NAX_TILE_N: usize = 32;
pub const NAX_TILE_K: usize = 16;

/// MSL for the single-tile NAX GEMM (pure MPP, no external headers): cooperative
/// fill of threadgroup A/B → load into `matmul2d` register cooperative tensors
/// via the BaseNAXFrag lane layout → `run` → store with the same layout. Mirrors
/// scratchy's proven register-fragment `mma` (the metal::tensor `run` overload
/// has a different, unvalidated output layout — see the kernel body).
const NAX_MATMUL_TILE_SRC: &str = "\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>
using namespace metal;

// C[16x32] = A[16x16] . B[16x32], all row-major device float.
[[kernel]] void nax_matmul_tile(
    device const float* a_in [[buffer(0)]],   // M x K = 16 x 16
    device const float* b_in [[buffer(1)]],   // K x N = 16 x 32
    device float* c_out      [[buffer(2)]],   // M x N = 16 x 32
    uint lid [[thread_index_in_simdgroup]])
{
    threadgroup half a_tg[16 * 16];   // [M, K] row-major
    threadgroup half b_tg[32 * 16];   // [N, K] = transpose(B), row-major
    // Cooperative fill across the 32 simdgroup lanes.
    for (uint i = lid; i < 16u * 16u; i += 32u) {
        a_tg[i] = half(a_in[i]);                  // A[m,k] at m*16+k
    }
    for (uint i = lid; i < 32u * 16u; i += 32u) {
        uint n = i / 16u;                           // 0..31
        uint k = i % 16u;                           // 0..15
        b_tg[n * 16u + k] = half(b_in[k * 32u + n]);   // Bt[n,k] = B[k,n]
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Register-fragment path (scratchy's proven `mma`): load A and B into the
    // input cooperative tensors via the validated BaseNAXFrag lane layout, run,
    // and store the destination with the SAME layout — internally consistent,
    // unlike the metal::tensor `run` overload whose output layout differs.
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = gemm_op.template
        get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // BaseNAXFrag lane→coord: within a 16x16 fragment, lane L element e maps to
    // (row fm + (e>>2)*8, col fn + e%4). N=32/M-as-two-frags pack as [.., 8+..].
    const short qid = (short)lid >> 2;
    const short fm = (qid & 4) | (((short)lid >> 1) & 3);
    const short fn = ((qid & 2) | ((short)lid & 1)) * 4;

    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        ct_a[e] = a_tg[r * 16 + c];               // A[M,K], 1 fragment
        ct_b[e]     = b_tg[r * 16 + c];           // B[N,K] n-frag 0 (n 0..15)
        ct_b[8 + e] = b_tg[(r + 16) * 16 + c];    // B[N,K] n-frag 1 (n 16..31)
        ct_c[e] = 0.0f;
        ct_c[8 + e] = 0.0f;
    }

    gemm_op.run(ct_a, ct_b, ct_c);

    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        c_out[r * 32 + c]      = ct_c[e];         // C[M,N] n 0..15
        c_out[r * 32 + c + 16] = ct_c[8 + e];     // C[M,N] n 16..31
    }
}
";

/// Run one NAX tile: `C[16×32] = A[16×16] · B[16×32]` on the M5 tensor engine.
/// `a` is row-major 16×16, `b` is row-major 16×32; returns row-major 16×32.
/// `Err("no Metal device …")` when no GPU is available, so callers can skip.
pub fn run_nax_matmul_tile(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
        MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    assert_eq!(a.len(), NAX_TILE_M * NAX_TILE_K, "A must be 16×16");
    assert_eq!(b.len(), NAX_TILE_K * NAX_TILE_N, "B must be 16×32");
    let out_len = NAX_TILE_M * NAX_TILE_N;

    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
    let opts = objc2_metal::MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(NAX_MATMUL_TILE_SRC), Some(&opts))
        .map_err(|e| format!("metal: NAX MSL compile failed: {e:?}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("nax_matmul_tile"))
        .ok_or("metal: kernel nax_matmul_tile not found")?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("metal: pipeline build failed: {e:?}"))?;
    let queue = device
        .newCommandQueue()
        .ok_or("metal: newCommandQueue returned nil")?;

    let res = MTLResourceOptions::StorageModeShared;
    let mk_in = |data: &[f32]| -> Result<_, String> {
        let bytes: &[u8] = bytemuck_cast(data);
        // SAFETY: `bytes` lives until the copy completes inside this call.
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                    bytes.len(),
                    res,
                )
                .ok_or_else(|| "metal: input buffer alloc failed".to_string())
        }
    };
    let a_buf = mk_in(a)?;
    let b_buf = mk_in(b)?;
    let c_buf = device
        .newBufferWithLength_options(out_len * 4, res)
        .ok_or("metal: output buffer alloc failed")?;

    let cb = queue
        .commandBuffer()
        .ok_or("metal: commandBuffer returned nil")?;
    let enc = cb
        .computeCommandEncoder()
        .ok_or("metal: computeCommandEncoder returned nil")?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
    }
    // One simdgroup (32 threads), one threadgroup.
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let raw =
        unsafe { std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len) };
    Ok(raw.to_vec())
}

/// Reinterpret an `&[f32]` as bytes without a dependency. (The runtime copies
/// it immediately into a Metal buffer.)
fn bytemuck_cast(data: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain-old-data; the returned slice covers exactly the same
    // bytes and borrows for the same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// Reinterpret a `&[u32]` as bytes (for small uniform buffers like dims/codes).
fn bytemuck_u32(data: &[u32]) -> &[u8] {
    // SAFETY: u32 is plain-old-data; same bytes, same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// One step of a batched matmul chain ([`NaxGemm::run_chain`]): multiply the
/// running result by `b` (k×n) and apply `epi` (with operand `e`, if any).
#[cfg(metal)]
pub struct ChainStep<'a> {
    pub k: usize,
    pub n: usize,
    pub b: &'a [f32],
    pub epi: Epilogue,
    pub e: Option<&'a [f32]>,
}

/// A fused matmul epilogue: `out = act(c BINOP e)`, where `e` is a per-element
/// operand (bias/residual/scale). The codes match the MSL `nax_epilogue` switch.
/// Lets the emulator fold a `matmul` and a following elementwise op (add, mul,
/// relu, tanh, …) into one GPU kernel — no readback, no second launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epilogue {
    /// Binary op with `e`: 0 none, 1 add, 2 mul, 3 sub, 4 max, 5 min.
    pub binop: u32,
    /// Activation: 0 none, 1 relu, 2 tanh, 3 exp, 4 sigmoid.
    pub act: u32,
}

impl Epilogue {
    /// No epilogue — a plain matmul.
    pub const NONE: Epilogue = Epilogue { binop: 0, act: 0 };
    pub const ADD: Epilogue = Epilogue { binop: 1, act: 0 };
    pub const MUL: Epilogue = Epilogue { binop: 2, act: 0 };
    pub const SUB: Epilogue = Epilogue { binop: 3, act: 0 };
    pub const MAX: Epilogue = Epilogue { binop: 4, act: 0 };
    pub const MIN: Epilogue = Epilogue { binop: 5, act: 0 };
    pub const RELU: Epilogue = Epilogue { binop: 0, act: 1 };
    pub const TANH: Epilogue = Epilogue { binop: 0, act: 2 };
    pub const EXP: Epilogue = Epilogue { binop: 0, act: 3 };
    pub const SIGMOID: Epilogue = Epilogue { binop: 0, act: 4 };

    /// Map a binary elementwise KTIR op name to its epilogue (with `e` the other
    /// operand), or `None` if it isn't a fusable binary op.
    pub fn from_binary_op(op_type: &str) -> Option<Epilogue> {
        Some(match op_type {
            "linalg.add" | "arith.addf" => Epilogue::ADD,
            "linalg.mul" | "arith.mulf" => Epilogue::MUL,
            "linalg.sub" | "arith.subf" => Epilogue::SUB,
            "linalg.max" | "arith.maximumf" | "arith.maxf" => Epilogue::MAX,
            "linalg.min" | "arith.minimumf" | "arith.minf" => Epilogue::MIN,
            _ => return None,
        })
    }

    /// Map a unary activation KTIR op name to its epilogue, or `None`.
    pub fn from_unary_op(op_type: &str) -> Option<Epilogue> {
        Some(match op_type {
            "math.tanh" => Epilogue::TANH,
            "math.exp" => Epilogue::EXP,
            _ => return None,
        })
    }
}

// =========================================================================
// General tiled NAX GEMM — arbitrary M, N, K
// =========================================================================
//
// One simdgroup per threadgroup computes one 16×32 output tile of C; the grid
// is ceil(M/16) × ceil(N/32) threadgroups. Each threadgroup walks K in steps of
// 16, staging A[16×16] and Bᵀ[32×16] sub-tiles into threadgroup memory (with
// bounds guards that zero-pad ragged edges), loading them into the `matmul2d`
// register cooperative tensors via the BaseNAXFrag layout, and accumulating
// into a persistent destination tensor across the K loop. The final tile is
// stored with per-element guards so partial M/N edges write only valid cells.
//
// This is the validated single-tile core (`run_nax_matmul_tile`) generalized:
// same fragment layout, now with a K-accumulation loop and edge handling.

/// MSL for the general NAX GEMM. `dims = (M, N, K)`.
///
/// Three levels of tiling. **Threadgroup**: SGS_M×SGS_N simdgroups (here 4×4 =
/// 16 simdgroups, 512 threads) cooperatively stage the A[128×16] and Bᵀ[256×16]
/// panels for a 128×256 output block — all threads share each device load.
/// **Register**: each simdgroup computes its 32×64 sub-block as a 2×2 grid of
/// 16×32 `matmul2d` tiles, loading 2 A row-fragments and 2 B column-fragment-
/// pairs per K-step and running all 4 products from them. **Pipeline**:
/// double-buffered panels — the next K-step's device loads are prefetched into
/// the other threadgroup half while the current panel feeds the matmuls, hiding
/// load latency behind compute. Ragged M/N/K are zero-padded on stage and
/// guarded on store.
const NAX_MATMUL_SRC: &str = include_str!("../shaders/nax_matmul.metal");

/// Pre-M5 GEMM via `simdgroup_float8x8` — the matrix path available on every
/// Apple7+ GPU (M1–M4), which lack the NAX tensor engine. Same fused epilogue
/// and same buffer layout as the NAX kernel (so the host dispatch is shared):
/// one simdgroup per 8×8 output tile accumulates over K in steps of 8 via
/// `simdgroup_multiply_accumulate`, then applies `act(c BINOP e)` on store.
#[cfg(metal)]
const SIMD_MATMUL_SRC: &str = include_str!("../shaders/simd_matmul.metal");

/// MSL for the matrix-VECTOR product — the `m == 1` decode fast path. One thread
/// per output column `n` dots the K-vector `x` (= the m=1 A row) with B's column
/// (plain `[k,n]`) or row (`transpose_b`, B `[n,k]`), accumulating in f32 over
/// f16 operands (so the result agrees with an f32 oracle to f16 tolerance, exactly
/// like the GEMM kernels). The buffer layout MATCHES the GEMM kernels
/// (`a,b,c,dims,e,epi`) so the host dispatch and the fused epilogue are shared:
/// `dims = (1, N, K)`, `c`/`e` are length N. B-staging keys off `KTIR_TRANSPOSE_B`
/// — the SAME `#define` the GEMM kernels use — so the [n,k] weight binds verbatim.
///
/// A GEMV is memory-bound (it streams B once, doing one MAC per element), so this
/// thread-per-column kernel — no tiling, no threadgroup staging — is the right
/// shape: at M=1 the tiled GEMM would launch ~16× the threads and leave 15/16 of
/// every matrix tile idle.
#[cfg(metal)]
const NAX_GEMV_SRC: &str = include_str!("../shaders/nax_gemv.metal");

/// A compiled, reusable Metal GEMM context — builds the device/pipeline/queue
/// once so repeated `run` calls (and benchmarks) exclude compile cost. Picks the
/// kernel by device: the NAX `matmul2d` engine on M5+, else the `simdgroup_*`
/// matrix path on M1–M4. Created with [`NaxGemm::new`]; `Err` if no Metal device
/// or the chosen kernel won't compile.
#[cfg(metal)]
type MtlBuf = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// Page-aligned host allocation, freed on drop. Backs a [`UnifiedBuffer`].
#[cfg(metal)]
struct AlignedAlloc {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}
#[cfg(metal)]
impl Drop for AlignedAlloc {
    fn drop(&mut self) {
        // SAFETY: ptr/layout came from the matching alloc in UnifiedBuffer::new.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// A **zero-copy** unified-memory tensor: page-aligned host memory wrapped as a
/// Metal buffer via `newBufferWithBytesNoCopy`. The CPU accesses it as `&[f32]`
/// and the GPU as an `MTLBuffer` — they share the *same bytes*, so a matmul over
/// `UnifiedBuffer`s has no host↔device fill or readback (the ~600 µs of copies
/// the host-`Vec` path pays). This is the right primitive for Apple's unified
/// memory; tile storage backed by these makes the whole compute path copy-free.
///
/// Field order matters: `mtl` is released before `alloc` frees the memory.
#[cfg(metal)]
pub struct UnifiedBuffer {
    mtl: MtlBuf,
    alloc: AlignedAlloc,
    len: usize,
    /// Element width in bytes: 4 (f32, the default) or 2 (f16 weight buffers, the
    /// `KTIR_F16_WEIGHTS` path). `len` is always the ELEMENT count, never bytes.
    elem_bytes: usize,
}

#[cfg(metal)]
impl UnifiedBuffer {
    /// Allocate `len` f32s of page-aligned, GPU-shared, zero-initialized memory.
    pub fn new(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        len: usize,
    ) -> Result<Self, String> {
        Self::new_sized(device, len, 4)
    }

    /// As [`new`](Self::new) but with an explicit element width (`elem_bytes`): 4
    /// for f32, 2 for an f16 (half) buffer. The f16 form is HALF the bytes — the
    /// `KTIR_F16_WEIGHTS` weight-streaming win — and is read by the matmul kernel's
    /// `half`-B variant. The backing memory is `len * elem_bytes` bytes.
    fn new_sized(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        len: usize,
        elem_bytes: usize,
    ) -> Result<Self, String> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        const PAGE: usize = 16 * 1024; // Apple Silicon page size
        let bytes = (len * elem_bytes).max(4).next_multiple_of(PAGE);
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).map_err(|e| e.to_string())?;
        // SAFETY: non-zero layout; zeroed so unused tail is defined.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("UnifiedBuffer: alloc failed".into());
        }
        // SAFETY: ptr is page-aligned and `bytes` long; deallocator None means we
        // (AlignedAlloc) own the memory and free it after the buffer is released.
        let mtl = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                std::ptr::NonNull::new(ptr as *mut std::ffi::c_void).unwrap(),
                bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or("UnifiedBuffer: newBufferWithBytesNoCopy returned nil")?;
        Ok(Self {
            mtl,
            alloc: AlignedAlloc { ptr, layout },
            len,
            elem_bytes,
        })
    }

    /// Build a unified buffer initialized from `data` (one copy in; thereafter
    /// the GPU reads it in place with no further copies).
    pub fn from_slice(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        data: &[f32],
    ) -> Result<Self, String> {
        let mut b = Self::new(device, data.len())?;
        b.as_mut_slice().copy_from_slice(data);
        Ok(b)
    }

    /// Build an f16 (half) unified buffer from raw little-endian f16 bytes — the
    /// HBM weight's native encoding. `raw` is `len` u16 halves (`2*len` bytes),
    /// copied verbatim (no f32 expansion): HALF the streamed bytes of [`from_slice`].
    /// Read by the matmul kernel's `half`-B variant (`KTIR_B_F16`).
    fn f16_from_raw(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        raw: &[u8],
    ) -> Result<Self, String> {
        debug_assert_eq!(raw.len() % 2, 0, "f16 raw must be an even byte count");
        let len = raw.len() / 2;
        let b = Self::new_sized(device, len, 2)?;
        // SAFETY: backing alloc holds >= len*2 live bytes; we own it exclusively here.
        unsafe {
            std::ptr::copy_nonoverlapping(raw.as_ptr(), b.alloc.ptr, raw.len());
        }
        Ok(b)
    }

    /// True for an f16 (half, 2-byte) buffer; false for the default f32 buffer.
    pub fn is_f16(&self) -> bool {
        self.elem_bytes == 2
    }

    pub fn as_slice(&self) -> &[f32] {
        debug_assert_eq!(self.elem_bytes, 4, "as_slice() on an f16 UnifiedBuffer");
        // SAFETY: alloc holds len f32s of live, aligned, initialized memory.
        unsafe { std::slice::from_raw_parts(self.alloc.ptr as *const f32, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        debug_assert_eq!(self.elem_bytes, 4, "as_mut_slice() on an f16 UnifiedBuffer");
        // SAFETY: as above; &mut self gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.alloc.ptr as *mut f32, self.len) }
    }
}

/// Persistent per-context scratch buffers, grown on demand and reused across
/// `run` calls so repeated matmuls pay no per-call allocation. Shared-storage
/// (unified memory), so the host fills/reads them via `contents()` directly.
#[cfg(metal)]
#[derive(Default)]
struct Scratch {
    a: Option<MtlBuf>,
    b: Option<MtlBuf>,
    e: Option<MtlBuf>,
    c: Option<MtlBuf>,
}

#[cfg(metal)]
pub struct NaxGemm {
    device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    /// Plain GEMM pipeline (B `[k,n]`). Covers BOTH active kernels: NAX `nax_matmul`
    /// (M5+) or the simdgroup `matmul` (pre-M5) — whichever the device selects.
    pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    /// Transpose-B pipeline: the SAME kernel compiled with `KTIR_TRANSPOSE_B=1`, so
    /// its B-staging reads the on-disk `[n,k]` weight verbatim (no copy/transpose).
    /// `matmul_unified(.., transpose_b=true)` selects it — covering NAX AND simdgroup.
    pipeline_bt: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    /// GEMV (matrix-vector, m=1) pipeline, plain B `[k,n]`. Plain MSL (no MPP), so
    /// it compiles on every Apple GPU regardless of tier. Selected by the
    /// `gemv_unified` / `gemv` entry points for the decode fast path.
    gemv_pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    /// GEMV transpose-B pipeline (`KTIR_TRANSPOSE_B=1`): B is the on-disk `[n,k]`
    /// weight read verbatim. `gemv_unified(.., transpose_b=true)` selects it.
    gemv_pipeline_bt: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    /// Small-M NAX GEMM variants (SGS_M=1 → 32-tall blocks), plain and transpose-B.
    /// Selected at dispatch when `m <= small_m` so a small-token-batch GEMM does not
    /// pad-and-compute the 96 phantom rows the full 128-tall kernel would. `None`
    /// on the simdgroup (pre-M5) path, whose 8-tall blocks have no such waste.
    pipeline_sm: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
    pipeline_sm_bt: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
    /// f16-B (`KTIR_B_F16`) variants — same kernels, B read as `half` (the
    /// `KTIR_F16_WEIGHTS` path). The full-block plain / transpose-B variants are
    /// built on BOTH tiers (so a pre-M5 device streams f16 weights too); the
    /// small-M `*_sm_*` variants are NAX-only (the simdgroup kernel has no small-M
    /// block). Selected by `matmul_unified` when the B `UnifiedBuffer` is f16.
    pipeline_f16b: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
    pipeline_bt_f16b: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
    pipeline_sm_f16b: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
    pipeline_sm_bt_f16b: Option<
        objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
        >,
    >,
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    scratch: std::cell::RefCell<Scratch>,
    /// Output block this kernel computes per threadgroup, and its thread count.
    block_m: usize,
    block_n: usize,
    threads: usize,
    /// Block height of the small-M variant (`KTIR_SGS_M * SG_M` = 32) and its
    /// thread count (`KTIR_SGS_M * SGS_N * 32` = 128). Unused when `pipeline_sm`
    /// is `None`.
    small_block_m: usize,
    small_threads: usize,
}

/// The embedded AOT metallib bytes for a variant `stem`, or `None` if this build
/// has no AOT (`cfg(metal_aot)` off) or the stem is unknown. `stem` matches the
/// build.rs output names exactly. The bytes are the offline-compiled
/// (`-mmacosx-version-min=26.2`, correct-K) metallibs, embedded into the binary so
/// the runtime never recompiles MSL when AOT is on.
#[cfg(all(metal, metal_aot))]
fn aot_metallib_bytes(stem: &str) -> Option<&'static [u8]> {
    macro_rules! lib {
        ($s:literal) => {
            include_bytes!(concat!(env!("OUT_DIR"), "/", $s, ".metallib")) as &'static [u8]
        };
    }
    Some(match stem {
        "nax_matmul__tb0_sm0_f16b0" => lib!("nax_matmul__tb0_sm0_f16b0"),
        "nax_matmul__tb0_sm0_f16b1" => lib!("nax_matmul__tb0_sm0_f16b1"),
        "nax_matmul__tb0_sm1_f16b0" => lib!("nax_matmul__tb0_sm1_f16b0"),
        "nax_matmul__tb0_sm1_f16b1" => lib!("nax_matmul__tb0_sm1_f16b1"),
        "nax_matmul__tb1_sm0_f16b0" => lib!("nax_matmul__tb1_sm0_f16b0"),
        "nax_matmul__tb1_sm0_f16b1" => lib!("nax_matmul__tb1_sm0_f16b1"),
        "nax_matmul__tb1_sm1_f16b0" => lib!("nax_matmul__tb1_sm1_f16b0"),
        "nax_matmul__tb1_sm1_f16b1" => lib!("nax_matmul__tb1_sm1_f16b1"),
        "nax_gemv__tb0" => lib!("nax_gemv__tb0"),
        "nax_gemv__tb1" => lib!("nax_gemv__tb1"),
        "simd_matmul__tb0_f16b0" => lib!("simd_matmul__tb0_f16b0"),
        "simd_matmul__tb1_f16b0" => lib!("simd_matmul__tb1_f16b0"),
        "simd_matmul__tb0_f16b1" => lib!("simd_matmul__tb0_f16b1"),
        "simd_matmul__tb1_f16b1" => lib!("simd_matmul__tb1_f16b1"),
        _ => return None,
    })
}

/// Build a compute pipeline from EMBEDDED AOT metallib `bytes` for kernel `kname`.
/// Loads via `MTLDevice::newLibraryWithURL` after staging the bytes to a temp file
/// (the `newLibraryWithData` binding needs the `dispatch2` feature; the file-URL
/// path needs no extra dep and the write is a one-time `metallib`-sized blob). The
/// staged file is content-stable per `stem` and only rewritten when missing or a
/// different size, so repeated process starts reuse it. Returns `Err` (so the
/// caller falls back to JIT) on any I/O / load / lookup failure — a bad metallib
/// never bricks the engine.
#[cfg(all(metal, metal_aot))]
fn aot_build_pipeline(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    stem: &str,
    kname: &str,
) -> Result<
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    String,
> {
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{MTLDevice, MTLLibrary};
    let bytes = aot_metallib_bytes(stem).ok_or_else(|| format!("aot: no metallib for {stem}"))?;
    // Stage the embedded bytes to a stable per-stem temp file once.
    let mut path = std::env::temp_dir();
    path.push(format!("ktir_aot_{stem}.metallib"));
    let need_write = match std::fs::metadata(&path) {
        Ok(m) => m.len() != bytes.len() as u64,
        Err(_) => true,
    };
    if need_write {
        // Write to a unique temp then rename, so concurrent starts don't tear.
        let tmp = path.with_extension(format!("metallib.{}", std::process::id()));
        std::fs::write(&tmp, bytes).map_err(|e| format!("aot: write {tmp:?} failed: {e}"))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("aot: rename failed: {e}"))?;
    }
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let library = device
        .newLibraryWithURL_error(&url)
        .map_err(|e| format!("aot: newLibraryWithURL {stem} failed: {e:?}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(kname))
        .ok_or_else(|| format!("aot: kernel {kname} not found in {stem}"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("aot: {kname} pipeline build failed ({stem}): {e:?}"))
}

#[cfg(metal)]
impl NaxGemm {
    /// Compile the best Metal GEMM kernel for the system default device: the NAX
    /// `matmul2d` engine on M5+, else the `simdgroup_float8x8` matrix path. Runs a
    /// one-time known-answer self-check ([`verify_full_k`]) so a miscompiled GEMM
    /// (e.g. the SDK-26.5 `matmul2d` half-K bug, were the runtime toolchain ever to
    /// regress the way the offline one does) fails LOUDLY instead of serving garbage.
    pub fn new() -> Result<Self, String> {
        let engine = Self::compile(None)?;
        engine.verify_full_k()?;
        Ok(engine)
    }

    /// True when this build embeds AOT-precompiled GEMM metallibs (`cfg(metal_aot)`,
    /// set by build.rs only when the offline `xcrun metal` toolchain can produce a
    /// correct-K NAX `matmul2d` with `-mmacosx-version-min=26.2`). When false, every
    /// pipeline is JIT-compiled from MSL at first use. The AOT load path always keeps
    /// the JIT as a per-variant fallback, so this reports the *build config*, not a
    /// guarantee that any given pipeline came from a metallib at runtime.
    pub fn aot_active() -> bool {
        cfg!(metal_aot)
    }

    /// Known-answer check that the GEMM reduces the FULL contraction axis: A=[2,128]
    /// and B=[128,2] of all ones give C[i,j] = Σ_k 1·1 = 128. The SDK-26.5 offline
    /// `matmul2d` miscompile reduces only half of K (C=64). Run once per engine; on
    /// failure the engine is rejected so the caller falls back to AMX/interpreter.
    fn verify_full_k(&self) -> Result<(), String> {
        use std::sync::atomic::{AtomicU8, Ordering};
        static STATE: AtomicU8 = AtomicU8::new(0); // 0 unchecked, 1 ok, 2 failed
        match STATE.load(Ordering::Relaxed) {
            1 => return Ok(()),
            2 => return Err("metal: matmul2d self-check previously FAILED (half-K)".into()),
            _ => {}
        }
        let (m, k, n) = (2usize, 128usize, 2usize);
        let c = self.run(m, k, n, &vec![1.0f32; m * k], &vec![1.0f32; k * n])?;
        let (got, want) = (c[0], k as f32);
        if (got - want).abs() > want * 0.1 {
            STATE.store(2, Ordering::Relaxed);
            return Err(format!(
                "metal: matmul2d self-check FAILED — C[0]={got}, want {want} (full-K reduction). \
                 Likely the SDK-26.5 MPP matmul2d half-K miscompile; refusing a broken GEMM."
            ));
        }
        STATE.store(1, Ordering::Relaxed);
        Ok(())
    }

    /// Force the `simdgroup_float8x8` (pre-M5) kernel regardless of device — used
    /// to validate that path on an M5 in tests.
    pub fn new_simdgroup() -> Result<Self, String> {
        Self::compile(Some(false))
    }

    /// Whether this engine compiled the f16-B GEMM pipelines. Built on both the NAX
    /// and simdgroup tiers (the full-block plain + transpose-B variants), so this is
    /// true on every supported device; callers check it before handing
    /// `matmul_unified` an f16 B buffer.
    pub fn has_f16_b_pipelines(&self) -> bool {
        self.pipeline_bt_f16b.is_some() && self.pipeline_f16b.is_some()
    }

    /// `force_nax`: `None` = auto by device, `Some(true)` = NAX, `Some(false)` =
    /// simdgroup.
    fn compile(force_nax: Option<bool>) -> Result<Self, String> {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
        };
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let is_nax = force_nax
            .unwrap_or_else(|| device_matmul_tier(&device.name().to_string()) == MatmulTier::Nax);

        let opts = objc2_metal::MTLCompileOptions::new();
        // These JIT `newLibraryWithSource` builds are the FALLBACK: when `metal_aot`
        // is on, the `aot` closure below loads the build.rs-embedded metallibs first
        // and only drops here if a variant fails to load. AOT LANDMINE (handled in
        // build.rs, repeated here): the NAX `matmul2d` metallibs MUST be compiled
        // `-mmacosx-version-min=26.2`. On SDK 26.5 the OFFLINE `xcrun metal` toolchain
        // miscompiles MPP `matmul2d` to reduce only HALF its K (the pre-26.2 headers
        // use a broken destination-tensor shim; 26.2 selects the MLX #3622 fix) —
        // REGARDLESS of MathMode/LanguageVersion4_0 — yielding ~95%-wrong GEMMs. The
        // RUNTIME JIT compiler (this path) is correct, so it's the safe fallback. The
        // `verify_full_k` self-check in `new` guards BOTH paths. See scratchy PR #56.
        let (src, kname, block_m, block_n, threads) = if is_nax {
            opts.setMathMode(MTLMathMode::Safe);
            opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
            (NAX_MATMUL_SRC, "nax_matmul", 128usize, 256usize, 512usize)
        } else {
            (SIMD_MATMUL_SRC, "matmul", 8usize, 8usize, 32usize)
        };
        // AOT-FIRST loader (only when `cfg(metal_aot)`): try the EMBEDDED metallib
        // for the variant `(stem_prefix, tb, sm, f16b)`; on success skip JIT. On ANY
        // failure return `None` so the caller JIT-compiles the same variant (a bad or
        // missing metallib never bricks the engine). When `cfg(metal_aot)` is off this
        // is a no-op that always returns `None`, so the JIT path runs unchanged.
        let aot = |stem_prefix: &str,
                   kname: &str,
                   tb: u32,
                   sm: u32,
                   f16b: u32|
         -> Option<
            objc2::rc::Retained<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            >,
        > {
            #[cfg(metal_aot)]
            {
                let stem = if stem_prefix == "nax_matmul" {
                    format!("nax_matmul__tb{tb}_sm{sm}_f16b{f16b}")
                } else if stem_prefix == "simd_matmul" {
                    // simd matmul varies by transpose_b and f16-B (no small-M block).
                    format!("simd_matmul__tb{tb}_f16b{f16b}")
                } else {
                    // gemv varies by transpose_b only.
                    format!("{stem_prefix}__tb{tb}")
                };
                // Ok(p) -> Some(p) (use the AOT pipeline); Err -> None (JIT fallback).
                aot_build_pipeline(&device, &stem, kname).ok()
            }
            #[cfg(not(metal_aot))]
            {
                let _ = (stem_prefix, kname, tb, sm, f16b);
                None
            }
        };
        // Build a pipeline from `src` with `KTIR_TRANSPOSE_B` defined to `tb` (0 =
        // plain B `[k,n]`, 1 = transpose-B reads B `[n,k]` verbatim). Prepending the
        // `#define` compiles the SAME kernel two ways — no source duplication, no
        // descriptor change (the B-staging index is the only difference).
        // Compile `src`'s `kname` kernel with `KTIR_TRANSPOSE_B` defined to `tb`.
        // Shared by the GEMM and GEMV kernels (both key B-staging off the define),
        // so each is built two ways from ONE source with no duplication.
        let build_src = |src: &str,
                         kname: &str,
                         tb: u32|
         -> Result<
            objc2::rc::Retained<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            >,
            String,
        > {
            let full = format!("#define KTIR_TRANSPOSE_B {tb}\n{src}");
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&full), Some(&opts))
                .map_err(|e| format!("metal: {kname} compile failed (tb={tb}): {e:?}"))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(kname))
                .ok_or_else(|| format!("metal: kernel {kname} not found"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("metal: {kname} pipeline build failed (tb={tb}): {e:?}"))
        };
        // f16-B (`KTIR_B_F16=1`) variant of `build_src`: B read as `half`.
        let build_f16b = |src: &str,
                          kname: &str,
                          tb: u32|
         -> Result<
            objc2::rc::Retained<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            >,
            String,
        > {
            let full = format!("#define KTIR_TRANSPOSE_B {tb}\n#define KTIR_B_F16 1\n{src}");
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&full), Some(&opts))
                .map_err(|e| format!("metal: {kname} f16b compile failed (tb={tb}): {e:?}"))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(kname))
                .ok_or_else(|| format!("metal: kernel {kname} not found"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("metal: {kname} f16b pipeline failed (tb={tb}): {e:?}"))
        };
        // Stem prefix for the active matmul kernel's AOT metallibs: the NAX kernel
        // (`nax_matmul`) is embedded as `nax_matmul__*`, the simdgroup kernel
        // (`matmul`) as `simd_matmul__*`.
        let mm_prefix = if is_nax { "nax_matmul" } else { "simd_matmul" };
        let pipeline = match aot(mm_prefix, kname, 0, 0, 0) {
            Some(p) => p,
            None => build_src(src, kname, 0)?,
        };
        let pipeline_bt = match aot(mm_prefix, kname, 1, 0, 0) {
            Some(p) => p,
            None => build_src(src, kname, 1)?,
        };
        // Small-M NAX variants (SGS_M=1 → 32-tall blocks) — only on the NAX path
        // (the simdgroup kernel's 8-tall blocks have no M-padding waste). Prepend
        // `KTIR_SGS_M 1` so the SAME source compiles a 32-row-block kernel.
        let build_small = |tb: u32| -> Result<
            objc2::rc::Retained<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            >,
            String,
        > {
            let full = format!("#define KTIR_TRANSPOSE_B {tb}\n#define KTIR_SGS_M 1\n{src}");
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&full), Some(&opts))
                .map_err(|e| format!("metal: {kname} small-M compile failed (tb={tb}): {e:?}"))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(kname))
                .ok_or_else(|| format!("metal: kernel {kname} not found"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("metal: {kname} small-M pipeline failed (tb={tb}): {e:?}"))
        };
        let (pipeline_sm, pipeline_sm_bt) = if is_nax {
            (
                Some(match aot(mm_prefix, kname, 0, 1, 0) {
                    Some(p) => p,
                    None => build_small(0)?,
                }),
                Some(match aot(mm_prefix, kname, 1, 1, 0) {
                    Some(p) => p,
                    None => build_small(1)?,
                }),
            )
        } else {
            (None, None)
        };
        // f16-B small-M variants (KTIR_SGS_M=1 + KTIR_B_F16=1). NAX-only.
        let build_small_f16b = |tb: u32| -> Result<
            objc2::rc::Retained<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
            >,
            String,
        > {
            let full = format!(
                "#define KTIR_TRANSPOSE_B {tb}\n#define KTIR_SGS_M 1\n#define KTIR_B_F16 1\n{src}"
            );
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&full), Some(&opts))
                .map_err(|e| {
                    format!("metal: {kname} small-M f16b compile failed (tb={tb}): {e:?}")
                })?;
            let function = library
                .newFunctionWithName(&NSString::from_str(kname))
                .ok_or_else(|| format!("metal: kernel {kname} not found"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| {
                    format!("metal: {kname} small-M f16b pipeline failed (tb={tb}): {e:?}")
                })
        };
        // f16-B GEMM variants — built on BOTH tiers so a pre-M5 device streams f16
        // weights (half the bytes) instead of f32. The full-block plain/transpose-B
        // variants exist on every tier; the small-M (`KTIR_SGS_M=1`) variants are
        // NAX-only (the simdgroup kernel has no small-M block, like the f32 set).
        let (pipeline_f16b, pipeline_bt_f16b) = (
            Some(match aot(mm_prefix, kname, 0, 0, 1) {
                Some(p) => p,
                None => build_f16b(src, kname, 0)?,
            }),
            Some(match aot(mm_prefix, kname, 1, 0, 1) {
                Some(p) => p,
                None => build_f16b(src, kname, 1)?,
            }),
        );
        let (pipeline_sm_f16b, pipeline_sm_bt_f16b) = if is_nax {
            (
                Some(match aot(mm_prefix, kname, 0, 1, 1) {
                    Some(p) => p,
                    None => build_small_f16b(0)?,
                }),
                Some(match aot(mm_prefix, kname, 1, 1, 1) {
                    Some(p) => p,
                    None => build_small_f16b(1)?,
                }),
            )
        } else {
            (None, None)
        };
        // GEMV kernel — plain MSL, device-tier-independent; built both B-layouts.
        let gemv_pipeline = match aot("nax_gemv", "nax_gemv", 0, 0, 0) {
            Some(p) => p,
            None => build_src(NAX_GEMV_SRC, "nax_gemv", 0)?,
        };
        let gemv_pipeline_bt = match aot("nax_gemv", "nax_gemv", 1, 0, 0) {
            Some(p) => p,
            None => build_src(NAX_GEMV_SRC, "nax_gemv", 1)?,
        };
        let queue = device
            .newCommandQueue()
            .ok_or("metal: newCommandQueue returned nil")?;
        Ok(Self {
            device,
            pipeline,
            pipeline_bt,
            gemv_pipeline,
            gemv_pipeline_bt,
            pipeline_sm,
            pipeline_sm_bt,
            pipeline_f16b,
            pipeline_bt_f16b,
            pipeline_sm_f16b,
            pipeline_sm_bt_f16b,
            queue,
            scratch: std::cell::RefCell::new(Scratch::default()),
            block_m,
            block_n,
            threads,
            // SG_M=32 → small block height; KTIR_SGS_M(1)*SGS_N(4)*32 = 128 threads.
            small_block_m: 32,
            small_threads: 128,
        })
    }

    /// `C(m×n) = A(m×k) · B(k×n)`, all row-major. A/B/C are f32 on the host;
    /// Whether to dispatch the small-M (32-tall-block) NAX variant for this GEMM.
    ///
    /// The full kernel computes a 128-tall output block per threadgroup; at m≤32 it
    /// pads and computes the 96 phantom rows anyway. The small-M variant skips that,
    /// but with 4× fewer threads per block — so it only WINS when there are enough
    /// N-blocks (block_n=256) to keep the GPU's cores busy with the smaller blocks.
    /// Microbench (M5, m=32): at n=8192 (32 N-blocks) gate 2.77→2.56 ms (+8%); at
    /// n=2048 (8 N-blocks) down 3.39→4.84 ms (−40%, under-occupied). So gate on a
    /// minimum N: only the wide-N MLP up/gate projections qualify, never down/qkv.
    ///
    /// `KTIR_NO_SMALL_M=1` forces the full kernel; `KTIR_SMALL_M=1` forces the small
    /// variant regardless of N (for the microbench / A-B testing). Default = the
    /// occupancy-aware rule below.
    #[cfg(metal)]
    fn pick_small_m(&self, m: usize, n: usize) -> bool {
        if self.pipeline_sm.is_none() || std::env::var_os("KTIR_NO_SMALL_M").is_some() {
            return false;
        }
        if m > self.small_block_m {
            return false;
        }
        // OFF by default. The isolated microbench shows a small-N-block win at
        // n≥4096 (gate/up: 2.77→2.56 ms, +8%), but it does NOT survive the full
        // prefill pipeline: with concurrent head-parallel attention on the CPU
        // contending for the GPU, the 32-tall block's reduced occupancy regresses
        // end-to-end (262 vs 231 ms/pass on M5). So the variant is kept as a
        // vetted, golden-exact, opt-in path only — enabled with KTIR_SMALL_M=1.
        let _ = n;
        std::env::var_os("KTIR_SMALL_M").is_some()
    }

    /// the kernel computes in f16 (the NAX engine's input precision), so the
    /// result agrees with an f32 oracle only to f16 tolerance.
    pub fn run(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.run_epi(m, k, n, a, b, None, Epilogue::NONE)
    }

    /// Fused matmul + elementwise epilogue in one kernel: `D = act(A·B BINOP E)`
    /// where `E` is the row-major m×n elementwise operand. No host readback of
    /// the matmul result and no second kernel launch — the activation/bias runs
    /// in the GEMM store. See [`Epilogue`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_fused(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        e: &[f32],
        epi: Epilogue,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(e.len(), m * n, "epilogue operand E must be m×n");
        self.run_epi(m, k, n, a, b, Some(e), epi)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_epi(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        e: Option<&[f32]>,
        epi: Epilogue,
    ) -> Result<Vec<f32>, String> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;

        assert_eq!(a.len(), m * k, "A must be m×k");
        assert_eq!(b.len(), k * n, "B must be k×n");
        let out_len = m * n;
        let res = MTLResourceOptions::StorageModeShared;

        // Grow `slot` to hold `cap` bytes if needed, then return the buffer.
        let ensure = |slot: &mut Option<MtlBuf>, cap: usize| -> Result<MtlBuf, String> {
            let need = cap.max(4);
            let ok = slot.as_ref().is_some_and(|b| b.length() >= need);
            if !ok {
                *slot = Some(
                    self.device
                        .newBufferWithLength_options(need, res)
                        .ok_or("metal: buffer alloc failed")?,
                );
            }
            Ok(slot.as_ref().unwrap().clone())
        };
        // Copy host floats into a shared buffer's contents (no realloc when reused).
        let fill = |buf: &MtlBuf, data: &[f32]| unsafe {
            let dst = buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        };

        let mut s = self.scratch.borrow_mut();
        let a_buf = ensure(&mut s.a, a.len() * 4)?;
        let b_buf = ensure(&mut s.b, b.len() * 4)?;
        let e_buf = ensure(&mut s.e, e.map_or(4, |e| e.len() * 4))?;
        let c_buf = ensure(&mut s.c, out_len * 4)?;
        fill(&a_buf, a);
        fill(&b_buf, b);
        if let Some(e) = e {
            fill(&e_buf, e);
        }

        let dims = [m as u32, n as u32, k as u32];
        let codes = [epi.binop, epi.act];

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer returned nil")?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or("metal: computeCommandEncoder returned nil")?;
        // SMALL-M selection (see pick_small_m): 32-tall block kernel for small-m,
        // wide-N GEMMs. run_epi is plain-B only, so never the transpose-B variant.
        let (pipe, blk_m, threads) = if self.pick_small_m(m, n) {
            (
                self.pipeline_sm.as_ref().unwrap(),
                self.small_block_m,
                self.small_threads,
            )
        } else {
            (&self.pipeline, self.block_m, self.threads)
        };
        enc.setComputePipelineState(pipe);
        // Small uniforms via setBytes — no per-call buffer allocation.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        // One threadgroup per output block (kernel-specific block + thread count).
        let m_blocks = m.div_ceil(blk_m);
        let n_blocks = n.div_ceil(self.block_n);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_blocks,
                height: m_blocks,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let raw =
            unsafe { std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len) };
        Ok(raw.to_vec())
    }

    /// Zero-copy matmul: `C = act(A·B BINOP E)` where A, B, C (and optional E)
    /// are [`UnifiedBuffer`]s already resident in shared memory. Encodes their
    /// buffers directly — no host↔device fill or readback. `c` must be sized
    /// `m·n`. This is the copy-free path unified memory makes possible.
    ///
    /// `transpose_b`: when true, B is the on-disk `[n,k]` weight and the GEMM
    /// contracts the last axis (`A·Bᵀ`) via the `KTIR_TRANSPOSE_B` pipeline — the
    /// kernel's B-staging reads `[n,k]` verbatim (no host copy/transpose/gather).
    /// `b.len` is `n·k` either way, so the size check is unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_unified(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &UnifiedBuffer,
        b: &UnifiedBuffer,
        c: &mut UnifiedBuffer,
        e: Option<&UnifiedBuffer>,
        epi: Epilogue,
        transpose_b: bool,
    ) -> Result<(), String> {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert_eq!(a.len, m * k, "A must be m×k");
        assert_eq!(
            b.len,
            k * n,
            "B must be k×n (n×k for transpose_b — same length)"
        );
        assert_eq!(c.len, m * n, "C must be m×n");

        let dims = [m as u32, n as u32, k as u32];
        let codes = [epi.binop, epi.act];
        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer nil")?;
        let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
        // SMALL-M selection (opt-in, KTIR_SMALL_M=1): when m fits the 32-tall block,
        // dispatch the small-M NAX variant — computes only the real rows instead of
        // padding to 128. Bit-identical output (same fragment math + edge guards).
        // OFF by default: at small-token prefill the GEMM is latency/occupancy-bound,
        // not M-compute-bound (padded rows are nearly free), so the 32-tall block's
        // 4× fewer threads can HURT GPU occupancy. Kept as a vetted, gated path.
        // f16-B selection: when the B (weight) buffer is f16, use the `KTIR_B_F16`
        // pipeline variant (B read as `half`). The full-block f16-B variants exist on
        // both tiers; only the small-M f16-B variants are NAX-only.
        let b_f16 = b.is_f16();
        let small = self.pick_small_m(m, n);
        // The f16-B and small-M variants are Option (None on tiers/configs that
        // didn't compile them); NEVER unwrap — a missing variant returns Err so the
        // caller falls back to the interpreter K-loop instead of aborting the process.
        let pipe = match (small, transpose_b, b_f16) {
            (true, true, true) => self
                .pipeline_sm_bt_f16b
                .as_ref()
                .ok_or("metal: small-M transpose-B f16 pipeline unavailable")?,
            (true, true, false) => self
                .pipeline_sm_bt
                .as_ref()
                .ok_or("metal: small-M transpose-B pipeline unavailable")?,
            (true, false, true) => self
                .pipeline_sm_f16b
                .as_ref()
                .ok_or("metal: small-M f16 pipeline unavailable")?,
            (true, false, false) => self
                .pipeline_sm
                .as_ref()
                .ok_or("metal: small-M pipeline unavailable")?,
            (false, true, true) => self
                .pipeline_bt_f16b
                .as_ref()
                .ok_or("metal: transpose-B f16 pipeline unavailable")?,
            (false, true, false) => &self.pipeline_bt,
            (false, false, true) => self
                .pipeline_f16b
                .as_ref()
                .ok_or("metal: f16 pipeline unavailable")?,
            (false, false, false) => &self.pipeline,
        };
        let (blk_m, threads) = if small {
            (self.small_block_m, self.small_threads)
        } else {
            (self.block_m, self.threads)
        };
        enc.setComputePipelineState(pipe);
        let e_mtl = e.unwrap_or(b); // dummy when binop==0 (never dereferenced)
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a.mtl), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b.mtl), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c.mtl), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_mtl.mtl), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(self.block_n),
                height: m.div_ceil(blk_m),
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// Zero-copy GEMV: `y(n) = act((x(k) · B) BINOP e)` where `x`, `B`, `y` (and
    /// optional `e`) are [`UnifiedBuffer`]s resident in shared memory — the m=1
    /// decode fast path, no host↔device fill or readback. `x` (= the m=1 A row)
    /// is length `k`, `y` is length `n`. With `transpose_b`, `B` is the on-disk
    /// `[n,k]` weight contracted over its last axis (`x·Bᵀ`), read verbatim.
    /// `b.len` is `k·n` either way. Dispatches one thread per output column.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_unified(
        &self,
        k: usize,
        n: usize,
        x: &UnifiedBuffer,
        b: &UnifiedBuffer,
        y: &mut UnifiedBuffer,
        e: Option<&UnifiedBuffer>,
        epi: Epilogue,
        transpose_b: bool,
    ) -> Result<(), String> {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
            MTLComputePipelineState, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert_eq!(x.len, k, "x must be length k");
        assert_eq!(
            b.len,
            k * n,
            "B must be k×n (n×k for transpose_b — same length)"
        );
        assert_eq!(y.len, n, "y must be length n");

        let dims = [1u32, n as u32, k as u32];
        let codes = [epi.binop, epi.act];
        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer nil")?;
        let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
        enc.setComputePipelineState(if transpose_b {
            &self.gemv_pipeline_bt
        } else {
            &self.gemv_pipeline
        });
        let e_mtl = e.unwrap_or(b); // dummy when binop==0 (never dereferenced)
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&x.mtl), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b.mtl), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&y.mtl), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_mtl.mtl), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        // One thread per output column (the kernel guards gid >= N).
        let tg = self
            .gemv_pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(n)
            .max(1);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: n,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// `y(n) = x(k) · B(k×n)` on the GPU, host f32 in/out — the copy-based GEMV
    /// convenience (uploads `x`/`B`, runs [`gemv_unified`], reads `y` back). For
    /// repeated calls over resident operands use [`gemv_unified`] directly.
    /// `transpose_b`: B is the on-disk `[n,k]` weight (`x·Bᵀ`).
    pub fn gemv(
        &self,
        k: usize,
        n: usize,
        x: &[f32],
        b: &[f32],
        transpose_b: bool,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(x.len(), k, "x must be length k");
        assert_eq!(b.len(), k * n, "B must be k×n (n×k for transpose_b)");
        let ux = self.unified_from(x)?;
        let ub = self.unified_from(b)?;
        let mut uy = self.unified(n)?;
        self.gemv_unified(k, n, &ux, &ub, &mut uy, None, Epilogue::NONE, transpose_b)?;
        Ok(uy.as_slice().to_vec())
    }

    /// Allocate a [`UnifiedBuffer`] on this context's device.
    pub fn unified(&self, len: usize) -> Result<UnifiedBuffer, String> {
        UnifiedBuffer::new(&self.device, len)
    }
    /// A [`UnifiedBuffer`] initialized from host data.
    pub fn unified_from(&self, data: &[f32]) -> Result<UnifiedBuffer, String> {
        UnifiedBuffer::from_slice(&self.device, data)
    }
    /// An f16 (half) [`UnifiedBuffer`] from raw little-endian f16 bytes (the HBM
    /// weight's native encoding) — half the streamed bytes of [`unified_from`], read
    /// by the matmul kernel's `half`-B variant. Used for the resident GPU WEIGHT
    /// operand under [`f16_weights_enabled`].
    fn unified_f16_from_raw(&self, raw: &[u8]) -> Result<UnifiedBuffer, String> {
        UnifiedBuffer::f16_from_raw(&self.device, raw)
    }
    /// Build an f16 (half) [`UnifiedBuffer`] from host f32 `data` by rounding each
    /// element to f16 — for benchmarks / tests that need an f16 B operand to feed
    /// [`matmul_unified`]. (The production path reads f16 raw bytes from HBM with no
    /// rounding via [`unified_f16_from_raw`].)
    pub fn unified_f16_from_f32(&self, data: &[f32]) -> Result<UnifiedBuffer, String> {
        let raw = crate::codec::encode(data, DType::F16);
        UnifiedBuffer::f16_from_raw(&self.device, &raw)
    }

    /// Run a chain of left-associated matmuls in ONE command buffer with a single
    /// GPU sync: `out₀ = a · steps[0].b`, then `outᵢ = outᵢ₋₁ · steps[i].b`, each
    /// with its fused epilogue. Intermediates stay in GPU buffers (never read back
    /// to the host), so the ~250 µs dispatch/sync latency is paid once for the
    /// whole chain instead of per matmul — the batching that makes GPU matmul win
    /// on the small LX-sized tiles. `a` is the host input (k0 = a.len()/m0);
    /// returns the final result `outₙ₋₁`.
    pub fn run_chain(
        &self,
        m0: usize,
        a: &[f32],
        steps: &[ChainStep<'_>],
    ) -> Result<Vec<f32>, String> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert!(!steps.is_empty(), "chain needs at least one step");

        let res = MTLResourceOptions::StorageModeShared;
        let rows = m0; // left-multiply: row count is fixed across the chain
        let alloc = |bytes: usize| -> Result<MtlBuf, String> {
            self.device
                .newBufferWithLength_options(bytes.max(4), res)
                .ok_or_else(|| "metal: chain alloc failed".to_string())
        };
        let fill = |buf: &MtlBuf, data: &[f32]| unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut f32,
                data.len(),
            );
        };

        // Allocate the pool ONCE (sized to the chain's maxima) and reuse it for
        // every step — no per-step allocation. Two ping-pong result buffers hold
        // the running product; A, B, and E are refilled in place.
        let max_b = steps.iter().map(|s| s.b.len()).max().unwrap_or(1);
        let max_e = steps
            .iter()
            .map(|s| s.e.map_or(1, <[f32]>::len))
            .max()
            .unwrap_or(1);
        let max_out = steps.iter().map(|s| rows * s.n).max().unwrap_or(1);
        let a_buf = alloc(a.len() * 4)?;
        fill(&a_buf, a);
        let ping = [alloc(max_out * 4)?, alloc(max_out * 4)?];
        let b_buf = alloc(max_b * 4)?;
        let e_buf = alloc(max_e * 4)?;

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer returned nil")?;
        let mut final_len = 0usize;
        for (i, s) in steps.iter().enumerate() {
            assert_eq!(s.b.len(), s.k * s.n, "chain step B must be k×n");
            let out_len = rows * s.n;
            let prev = if i == 0 { &a_buf } else { &ping[(i - 1) % 2] };
            let out = &ping[i % 2];
            fill(&b_buf, s.b);
            if let Some(e) = s.e {
                assert_eq!(e.len(), out_len, "chain step E must be m×n");
                fill(&e_buf, e);
            }
            let dims = [rows as u32, s.n as u32, s.k as u32];
            let codes = [s.epi.binop, s.epi.act];

            let enc = cb
                .computeCommandEncoder()
                .ok_or("metal: chain encoder nil")?;
            enc.setComputePipelineState(&self.pipeline);
            // Small uniforms go through setBytes (no buffer allocation).
            unsafe {
                enc.setBuffer_offset_atIndex(Some(prev), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(out), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(&dims),
                    3,
                );
                enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(&codes),
                    5,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: s.n.div_ceil(self.block_n),
                    height: rows.div_ceil(self.block_m),
                    depth: 1,
                },
                MTLSize {
                    width: self.threads,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            final_len = out_len;
        }

        cb.commit();
        cb.waitUntilCompleted();
        let last = &ping[(steps.len() - 1) % 2];
        let raw = unsafe {
            std::slice::from_raw_parts(last.contents().as_ptr() as *const f32, final_len)
        };
        Ok(raw.to_vec())
    }

    /// **Combine** many small matmuls that share the weight `b` into ONE tall
    /// matmul. `a_stack` is `count` row-panels (`count·m × k`, contiguous), `b`
    /// is the shared `k × n` weight; returns `count·m × n`. This is the real win
    /// for a SPMD KTIR grid (every core multiplies its rows by the same weights):
    /// stacking the panels into a single GEMM saturates the tensor engine, so
    /// the GPU beats a serial AMX loop by 1.25–1.74× once K ≳ 512 (measured) —
    /// unlike running them separately, where each small matmul underfills the
    /// engine and the GPU loses. Just a clarity wrapper over [`run`](Self::run)
    /// with `m' = count·m`.
    pub fn run_combined(
        &self,
        count: usize,
        m: usize,
        k: usize,
        n: usize,
        a_stack: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a_stack.len(), count * m * k, "stacked A must be count·m×k");
        assert_eq!(b.len(), k * n, "shared B must be k×n");
        self.run(count * m, k, n, a_stack, b)
    }

    /// `batch` independent same-shape GEMMs `Cᵢ = Aᵢ · Bᵢ` in ONE dispatch (one
    /// submission), run concurrently across the GPU. `a` is `batch·m·k` and `b`
    /// is `batch·k·n`, both row-major and contiguous per slice; returns
    /// `batch·m·n`. Use this when each matmul has its OWN B; when they share B,
    /// [`run_combined`](Self::run_combined) is faster (one saturating GEMM).
    pub fn run_batched(
        &self,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert_eq!(a.len(), batch * m * k, "A must be batch×m×k");
        assert_eq!(b.len(), batch * k * n, "B must be batch×k×n");
        let out_len = batch * m * n;
        let res = MTLResourceOptions::StorageModeShared;

        let upload = |data: &[f32]| -> Result<MtlBuf, String> {
            let bytes = bytemuck_cast(data);
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(4),
                        res,
                    )
                    .ok_or_else(|| "metal: batched upload failed".to_string())
            }
        };
        let a_buf = upload(a)?;
        let b_buf = upload(b)?;
        let e_buf = upload(&[0.0f32])?;
        let c_buf = self
            .device
            .newBufferWithLength_options((out_len * 4).max(4), res)
            .ok_or("metal: batched output alloc failed")?;
        let dims = [m as u32, n as u32, k as u32];
        let codes = [0u32, 0u32];

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer returned nil")?;
        let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
        enc.setComputePipelineState(&self.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        // Grid z = batch: all `batch` GEMMs dispatched together, run concurrently.
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(self.block_n),
                height: m.div_ceil(self.block_m),
                depth: batch,
            },
            MTLSize {
                width: self.threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        let raw =
            unsafe { std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len) };
        Ok(raw.to_vec())
    }

    /// Pure GPU kernel time (seconds) for one GEMM, from the command buffer's
    /// hardware timestamps — excludes buffer allocation, host→device copies,
    /// and readback. Buffers are allocated once and reused across `iters`
    /// dispatches (one command buffer), so this isolates kernel throughput from
    /// per-call CPU overhead. Returns the *total* GPU time over `iters`.
    pub fn gpu_time_seconds(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        iters: u32,
    ) -> Result<f64, String> {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
            MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;

        let res = MTLResourceOptions::StorageModeShared;
        let mk_in = |data: &[f32]| -> Result<_, String> {
            let bytes: &[u8] = bytemuck_cast(data);
            // SAFETY: `bytes` lives until the copy completes inside this call.
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(1),
                        res,
                    )
                    .ok_or_else(|| "alloc".to_string())
            }
        };
        let a_buf = mk_in(a)?;
        let b_buf = mk_in(b)?;
        let e_buf = mk_in(&[0.0f32])?;
        let c_buf = self
            .device
            .newBufferWithLength_options((m * n * 4).max(1), res)
            .ok_or("alloc")?;
        let small = |v: &[u32]| -> Result<_, String> {
            let bytes = bytemuck_u32(v);
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len(),
                        res,
                    )
                    .ok_or_else(|| "alloc".to_string())
            }
        };
        let dims_buf = small(&[m as u32, n as u32, k as u32])?;
        let codes_buf = small(&[0u32, 0u32])?;
        let m_blocks = m.div_ceil(self.block_m);
        let n_blocks = n.div_ceil(self.block_n);

        let cb = self.queue.commandBuffer().ok_or("cb")?;
        for _ in 0..iters {
            let enc = cb.computeCommandEncoder().ok_or("enc")?;
            enc.setComputePipelineState(&self.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&codes_buf), 0, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_blocks,
                    height: m_blocks,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        // Hardware GPU timestamps (CFTimeInterval seconds) for the whole buffer.
        Ok(cb.GPUEndTime() - cb.GPUStartTime())
    }
}

/// Convenience: compile + run a general NAX GEMM once. For repeated calls or
/// benchmarks build a [`NaxGemm`] and reuse it (compiles the kernel once).
#[cfg(metal)]
pub fn run_nax_matmul(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, String> {
    NaxGemm::new()?.run(m, k, n, a, b)
}

/// Convenience: compile + run the GPU GEMV once — `y(n) = x(k) · B`, the m=1
/// decode fast path. `transpose_b`: B is the on-disk `[n,k]` weight (`x·Bᵀ`).
/// For repeated calls build a [`NaxGemm`] and reuse `gemv` / `gemv_unified`.
#[cfg(metal)]
pub fn run_metal_gemv(
    k: usize,
    n: usize,
    x: &[f32],
    b: &[f32],
    transpose_b: bool,
) -> Result<Vec<f32>, String> {
    NaxGemm::new()?.gemv(k, n, x, b, transpose_b)
}

/// The system default Metal device's name (e.g. `"Apple M5 Pro"`), or `""` if
/// there is no device.
#[cfg(metal)]
pub fn device_name() -> String {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    MTLCreateSystemDefaultDevice()
        .map(|d| d.name().to_string())
        .unwrap_or_default()
}

#[cfg(metal)]
thread_local! {
    /// Cached device name (cheap, resolved once per thread).
    static GEMM_DEVICE: std::cell::OnceCell<String> = const { std::cell::OnceCell::new() };
    /// Lazily-compiled NAX GEMM context, built the first time the gate picks NAX
    /// (so we never pay the kernel compile when only Accelerate is used). `None`
    /// if compilation fails (e.g. a pre-M5 GPU without the tensor engine).
    static GEMM_NAX: std::cell::OnceCell<Option<NaxGemm>> = const { std::cell::OnceCell::new() };
}

/// The production GEMM entry point for the emulator: `C(m×k·k×n)` row-major,
/// dispatched to the highest-performance available backend.
///
/// On an M5, large GEMMs (per [`choose_matmul_backend`]) run on the NAX tensor
/// engine (f16, ~4 TFLOP/s, ~2× Accelerate); small ones and everything on
/// non-NAX devices run on Accelerate/`sgemm_rowmajor` (f32). The NAX context is
/// compiled once and cached per thread; if NAX is chosen but unavailable or
/// errors, it falls back to Accelerate. So this is always correct and never
/// slower than the BLAS path by more than one (cached) capability probe.
#[cfg(metal)]
pub fn metal_gemm_or_blas(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let name = GEMM_DEVICE.with(|c| c.get_or_init(device_name).clone());
    if choose_matmul_backend(&name, m, k, n).is_gpu()
        && let Some(mut out) = GEMM_NAX.with(|c| {
            c.get_or_init(|| NaxGemm::new().ok())
                .as_ref()
                .and_then(|g| g.run(m, k, n, a, b).ok())
        })
    {
        GEMM_OR_BLAS_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        inject_gpu_divergence(&mut out);
        return out;
    }
    crate::blas::sgemm_rowmajor(m, k, n, a, b)
}

/// TEST-ONLY fault injector for the GPU differential-conformance harness. When
/// `KTIR_DIFF_INJECT_DIVERGENCE` is set to a float, every Metal GEMM result is
/// scaled by `(1 + eps)` AFTER the NAX engine produces it — corrupting the GPU
/// output path while leaving the proof counter (the GPU branch DID run) intact.
///
/// This is the GPU analogue of the NaN-safe diff proof in the Python driver: it
/// lets CI demonstrate that the banded differential actually FAILS on a real
/// numeric divergence of the Metal fast path, rather than silently passing
/// garbage. It is NEVER set in a real run; the value is a relative perturbation
/// (e.g. `0.05` => +5%, well outside the principled bf16/f16 band) so a divergence
/// is unambiguous and not a borderline rounding artefact. No-op when unset/empty.
#[cfg(metal)]
fn inject_gpu_divergence(out: &mut [f32]) {
    if let Some(eps) = std::env::var("KTIR_DIFF_INJECT_DIVERGENCE")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<f32>().ok())
        && eps != 0.0
    {
        let scale = 1.0 + eps;
        for v in out.iter_mut() {
            *v *= scale;
        }
    }
}

/// The m=1 decode GEMV entry point: `y(n) = x(k) · B(k×n)` row-major, on the
/// highest-performance backend. Mirrors [`metal_gemm_or_blas`] but routes the
/// matrix-VECTOR case to a purpose-built GEMV instead of the tiled GEMM (which
/// wastes ~15/16 of every matrix tile at M=1). On an M5 a large-enough GEMV
/// (per [`choose_matmul_backend`] with m=1) runs the GPU `nax_gemv` kernel; small
/// ones and non-NAX devices use Accelerate `sgemv_rowmajor` (AMX, f32). The NAX
/// context is the SAME cached engine `metal_gemm_or_blas` uses. Always correct,
/// never slower than the CPU GEMV by more than one cached capability probe.
#[cfg(metal)]
pub fn metal_gemv_or_blas(k: usize, n: usize, x: &[f32], b: &[f32]) -> Vec<f32> {
    let name = GEMM_DEVICE.with(|c| c.get_or_init(device_name).clone());
    if choose_matmul_backend(&name, 1, k, n).is_gpu()
        && let Some(out) = GEMM_NAX.with(|c| {
            c.get_or_init(|| NaxGemm::new().ok())
                .as_ref()
                .and_then(|g| g.gemv(k, n, x, b, /*transpose_b=*/ false).ok())
        })
    {
        GEMM_OR_BLAS_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return out;
    }
    crate::blas::sgemv_rowmajor(k, n, x, b)
}

/// The m=1 transpose-B decode GEMV: `y(n) = x(k) · B(n×k)ᵀ`, B stored on-disk
/// `[n,k]` (PyTorch `Linear` `[out,in]`), read verbatim. The matrix-VECTOR
/// analogue of transpose-B's GEMM. GPU `nax_gemv` (transpose-B pipeline)
/// when the gate picks it, else Accelerate `sgemv_rowmajor_bt` (`cblas` `CblasNoTrans`
/// over the [n,k] rows — native, zero-copy). Same engine/gate as the plain form.
#[cfg(metal)]
pub fn metal_gemv_or_blas_bt(k: usize, n: usize, x: &[f32], b: &[f32]) -> Vec<f32> {
    let name = GEMM_DEVICE.with(|c| c.get_or_init(device_name).clone());
    if choose_matmul_backend(&name, 1, k, n).is_gpu()
        && let Some(out) = GEMM_NAX.with(|c| {
            c.get_or_init(|| NaxGemm::new().ok())
                .as_ref()
                .and_then(|g| g.gemv(k, n, x, b, /*transpose_b=*/ true).ok())
        })
    {
        GEMM_OR_BLAS_GPU_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return out;
    }
    crate::blas::sgemv_rowmajor_bt(k, n, x, b)
}

/// Fused `D = act(A·B BINOP E)` on the NAX engine, in one kernel — the matmul→
/// elementwise fusion the interpreter peephole uses. Returns `Some(D)` only when
/// the gate picks NAX (large enough to win) and the kernel runs; otherwise
/// `None`, so the caller falls back to running the matmul and elementwise op
/// separately. `e` is the row-major m×n elementwise operand.
#[cfg(metal)]
pub fn metal_gemm_fused(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    e: &[f32],
    epi: Epilogue,
) -> Option<Vec<f32>> {
    let name = GEMM_DEVICE.with(|c| c.get_or_init(device_name).clone());
    if !choose_matmul_backend(&name, m, k, n).is_gpu() {
        return None;
    }
    GEMM_NAX.with(|c| {
        c.get_or_init(|| NaxGemm::new().ok())
            .as_ref()
            .and_then(|g| g.run_fused(m, k, n, a, b, e, epi).ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;

    /// On a build that embedded the AOT metallibs (`cfg(metal_aot)`), assert the
    /// runtime reports AOT active AND that the engine still builds + reduces full K
    /// (the embedded NAX metallibs were compiled `-mmacosx-version-min=26.2`, so a
    /// half-K miscompile would trip `verify_full_k` inside `new`). On a build without
    /// AOT (toolchain/flag unavailable) the runtime falls back to JIT and this test
    /// simply records that AOT is off — never a failure.
    #[test]
    fn aot_active_matches_build_cfg() {
        assert_eq!(
            NaxGemm::aot_active(),
            cfg!(metal_aot),
            "aot_active() must mirror cfg(metal_aot)"
        );
        if NaxGemm::aot_active() {
            // AOT is on for THIS build — prove the embedded GEMM loads and is full-K.
            match NaxGemm::new() {
                Ok(_) => eprintln!("AOT active: embedded metallibs loaded, full-K verified"),
                Err(e) if e.contains("no Metal device") => {
                    eprintln!("no Metal device — skipping AOT load check");
                }
                Err(e) => panic!("AOT build but NaxGemm::new failed: {e}"),
            }
        } else {
            eprintln!("AOT not active on this build (JIT fallback) — cfg(metal_aot) off");
        }
    }

    /// Minimal Metal Performance Primitives probe — confirms the M5 NAX
    /// toolchain (`mpp::tensor_ops::matmul2d` + the MPP framework include)
    /// compiles through our `objc2-metal` runtime as Metal 4.
    const MPP_PROBE: &str = "\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
kernel void mpp_probe(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    (void)op;
    c[gid.y * 16 + gid.x] = a[gid.x] + b[gid.y];
}
";

    #[test]
    fn mpp_tensor_ops_compiles_as_metal4() {
        match compile_metal4(MPP_PROBE) {
            Ok(()) => eprintln!("MPP (mpp::tensor_ops) compiles as Metal 4 on this device ✓"),
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping MPP compile probe");
            }
            Err(e) => panic!("MPP shader failed to compile as Metal 4:\n{e}"),
        }
    }

    /// The NAX tensor engine produces a correct GEMM through our runtime.
    /// Small-integer inputs (exact in f16) let us assert *exact* equality with
    /// the naive oracle. Two identity probes pin the fragment layout: with
    /// A = I, `B[k,n] = n` must yield `C[m,n] = n` (column mapping) and
    /// `B[k,n] = k` must yield `C[m,n] = m` (row mapping) — together these catch
    /// any cooperative-tensor axis swap or scramble in the BaseNAXFrag layout.
    #[test]
    fn nax_matmul_tile_matches_oracle() {
        // The `matmul2d` MPP cooperative-tensor kernel only builds on M5+ (the
        // `Nax` tier); pre-M5 GPUs reject it at pipeline-build time. Skip there —
        // the simdgroup path is covered by the other `nax_matmul_*` tests.
        {
            use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
            match MTLCreateSystemDefaultDevice() {
                None => {
                    eprintln!("no Metal device — skipping NAX matmul tile test");
                    return;
                }
                Some(device)
                    if device_matmul_tier(&device.name().to_string()) < MatmulTier::Nax =>
                {
                    eprintln!(
                        "device {:?} is not NAX(matmul2d)-capable (needs M5+) — skipping",
                        device.name().to_string()
                    );
                    return;
                }
                Some(_) => {}
            }
        }
        // A[m,k] = (m + k) % 3, B[k,n] = (k + 2*n) % 4  — products ≤ 6, sums
        // over K=16 ≤ 96: all exact in f16 and f32, and distinct per (m,n).
        let a: Vec<f32> = (0..NAX_TILE_M * NAX_TILE_K)
            .map(|i| ((i / 16 + i % 16) % 3) as f32)
            .collect();
        let b: Vec<f32> = (0..NAX_TILE_K * NAX_TILE_N)
            .map(|i| ((i / 32 + 2 * (i % 32)) % 4) as f32)
            .collect();

        let got = match run_nax_matmul_tile(&a, &b) {
            Ok(v) => v,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping NAX matmul test");
                return;
            }
            Err(e) => panic!("NAX matmul failed: {e}"),
        };
        let want = crate::blas::naive_sgemm(NAX_TILE_M, NAX_TILE_K, NAX_TILE_N, &a, &b);
        assert_eq!(got, want, "NAX tile must match the naive oracle exactly");

        // Identity probes — A = I; column then row coordinate.
        let mut ai = vec![0.0f32; NAX_TILE_M * NAX_TILE_K];
        for d in 0..NAX_TILE_M {
            ai[d * NAX_TILE_K + d] = 1.0;
        }
        let col_probe = run_nax_matmul_tile(
            &ai,
            &(0..NAX_TILE_K * NAX_TILE_N)
                .map(|i| (i % NAX_TILE_N) as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let row_probe = run_nax_matmul_tile(
            &ai,
            &(0..NAX_TILE_K * NAX_TILE_N)
                .map(|i| (i / NAX_TILE_N) as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for m in 0..NAX_TILE_M {
            for n in 0..NAX_TILE_N {
                assert_eq!(
                    col_probe[m * NAX_TILE_N + n],
                    n as f32,
                    "column map at ({m},{n})"
                );
                assert_eq!(
                    row_probe[m * NAX_TILE_N + n],
                    m as f32,
                    "row map at ({m},{n})"
                );
            }
        }
        eprintln!("NAX matmul2d tile matches the oracle exactly (+ row/col layout) ✓");
    }

    /// The general tiled NAX GEMM is correct across shapes — including ragged
    /// M/N/K that exercise the zero-pad edge guards and multi-tile K
    /// accumulation. Small-integer inputs are exact in f16, so we assert exact
    /// equality with the naive oracle.
    #[test]
    fn nax_matmul_general_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping general NAX GEMM test");
                return;
            }
            Err(e) => panic!("NAX GEMM compile failed: {e}"),
        };
        // Exact tile (16,16,32); ragged in every dim; multi-K; K not /16; thin.
        let shapes = [
            (16usize, 16usize, 32usize),
            (1, 1, 1),
            (17, 33, 5),  // ragged M, N, K all
            (48, 16, 64), // multi-tile, clean
            (50, 20, 70), // multi-tile, ragged
            (7, 100, 3),  // wide N
            (100, 7, 3),  // tall M
        ];
        for (m, k, n) in shapes {
            // Small ints exact in f16: a in 0..3, b in 0..4. Sum over K stays
            // well under f16's 256 exact-integer limit for these K.
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let got = ctx.run(m, k, n, &a, &b).unwrap();
            let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
            assert_eq!(got, want, "NAX GEMM mismatch at shape ({m},{k},{n})");
        }
        eprintln!(
            "general NAX GEMM matches the oracle across {} shapes ✓",
            shapes.len()
        );
    }

    /// Shape sweep shared by the transpose-B oracle tests: exact tile, ragged in
    /// every dim, multi-K (K>16 and K not /16), wide N, tall M.
    #[cfg(metal)]
    const TRANSPOSE_B_SHAPES: [(usize, usize, usize); 7] = [
        (16, 16, 32),
        (1, 1, 1),
        (17, 33, 5),
        (48, 16, 64),
        (50, 20, 70),
        (7, 100, 3),
        (100, 7, 3),
    ];

    /// THE silent-wrong-answer guard for NAX native transpose-B: the
    /// `KTIR_TRANSPOSE_B` pipeline (B staged from on-disk `[n,k]`) must equal the
    /// `A·Bᵀ` oracle. A wrong staging index would compute a plausible-but-wrong
    /// product the GPU-vs-CPU self-check can't catch, so we pin it to the oracle.
    #[test]
    fn nax_matmul_unified_transpose_b_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping NAX transpose-B test");
                return;
            }
            Err(e) => panic!("NAX GEMM compile failed: {e}"),
        };
        for (m, k, n) in TRANSPOSE_B_SHAPES {
            // a in 0..3, b in 0..4 — exact in f16; f32 accumulation keeps the
            // integer dot products exact, so assert_eq is valid (as in the plain
            // oracle test). B is stored [n,k] (on-disk Linear [out,in]).
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..n * k).map(|i| (i % 4) as f32).collect();
            let ua = ctx.unified_from(&a).unwrap();
            let ub = ctx.unified_from(&b).unwrap();
            let mut uc = ctx.unified(m * n).unwrap();
            ctx.matmul_unified(
                m,
                k,
                n,
                &ua,
                &ub,
                &mut uc,
                None,
                Epilogue::NONE,
                /*transpose_b=*/ true,
            )
            .unwrap();
            let want = crate::blas::naive_sgemm_bt(m, k, n, &a, &b);
            assert_eq!(
                uc.as_slice(),
                want.as_slice(),
                "NAX transpose-B mismatch at ({m},{k},{n})"
            );
        }
        eprintln!(
            "NAX transpose-B (matmul_unified) matches the bt oracle across {} shapes ✓",
            TRANSPOSE_B_SHAPES.len()
        );
    }

    /// Same guard for the pre-NAX simdgroup "metal matmul" kernel (forced via
    /// `new_simdgroup`), so transpose-B is covered on non-M5 GPUs too.
    #[test]
    fn simdgroup_matmul_unified_transpose_b_matches_oracle() {
        let ctx = match NaxGemm::new_simdgroup() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping simdgroup transpose-B test");
                return;
            }
            Err(e) => panic!("simdgroup GEMM compile failed: {e}"),
        };
        for (m, k, n) in TRANSPOSE_B_SHAPES {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..n * k).map(|i| (i % 4) as f32).collect();
            let ua = ctx.unified_from(&a).unwrap();
            let ub = ctx.unified_from(&b).unwrap();
            let mut uc = ctx.unified(m * n).unwrap();
            ctx.matmul_unified(
                m,
                k,
                n,
                &ua,
                &ub,
                &mut uc,
                None,
                Epilogue::NONE,
                /*transpose_b=*/ true,
            )
            .unwrap();
            let want = crate::blas::naive_sgemm_bt(m, k, n, &a, &b);
            assert_eq!(
                uc.as_slice(),
                want.as_slice(),
                "simdgroup transpose-B mismatch at ({m},{k},{n})"
            );
        }
        eprintln!(
            "simdgroup transpose-B matches the bt oracle across {} shapes ✓",
            TRANSPOSE_B_SHAPES.len()
        );
    }

    /// f16-B on the pre-NAX simdgroup kernel: with the `KTIR_B_F16` read path the
    /// simdgroup tier now streams f16 weights too. Forcing `new_simdgroup` on this
    /// M5, an f16-B matmul (B read as `half`) must match the f32-B simdgroup result
    /// within f16 tolerance — both plain (`A·B`) and transpose-B (`A·Bᵀ`). Also
    /// pins `has_f16_b_pipelines()` true on the simdgroup tier (the gate the GEMM
    /// resolver checks before handing the kernel an f16 B buffer).
    #[test]
    fn simdgroup_matmul_unified_f16_b_matches_f32() {
        let ctx = match NaxGemm::new_simdgroup() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping simdgroup f16-B test");
                return;
            }
            Err(e) => panic!("simdgroup GEMM compile failed: {e}"),
        };
        assert!(
            ctx.has_f16_b_pipelines(),
            "simdgroup tier must compile the f16-B pipelines"
        );
        for &transpose_b in &[false, true] {
            for (m, k, n) in TRANSPOSE_B_SHAPES {
                // Signed, sub-unit values so the f16 rounding of B is exercised (not
                // exactly representable like small ints) — proves the half read path.
                let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
                let b: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
                let ua = ctx.unified_from(&a).unwrap();
                // f32-B reference (still the simdgroup kernel, B as f32).
                let ub_f32 = ctx.unified_from(&b).unwrap();
                let mut uc_f32 = ctx.unified(m * n).unwrap();
                ctx.matmul_unified(
                    m,
                    k,
                    n,
                    &ua,
                    &ub_f32,
                    &mut uc_f32,
                    None,
                    Epilogue::NONE,
                    transpose_b,
                )
                .unwrap();
                // f16-B: B read as `half` via the KTIR_B_F16 pipeline.
                let ub_f16 = ctx.unified_f16_from_f32(&b).unwrap();
                assert!(ub_f16.is_f16(), "B buffer must be f16");
                let mut uc_f16 = ctx.unified(m * n).unwrap();
                ctx.matmul_unified(
                    m,
                    k,
                    n,
                    &ua,
                    &ub_f16,
                    &mut uc_f16,
                    None,
                    Epilogue::NONE,
                    transpose_b,
                )
                .unwrap();
                let f32_res = uc_f32.as_slice();
                let f16_res = uc_f16.as_slice();
                let max_abs = f32_res
                    .iter()
                    .zip(f16_res)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs < 0.02,
                    "simdgroup f16-B vs f32-B mismatch at ({m},{k},{n}) tb={transpose_b}: \
                     max abs {max_abs} > f16 tol"
                );
            }
        }
        eprintln!(
            "simdgroup f16-B matches f32-B within f16 tol across {} shapes x {{plain, tb}} ✓",
            TRANSPOSE_B_SHAPES.len()
        );
    }

    /// The GPU GEMV (m=1 decode fast path) must match the naive oracle EXACTLY,
    /// both plain (`y = x·B`, B `[k,n]`) and transpose-B (`y = x·Bᵀ`, B `[n,k]`).
    /// Small-integer inputs (exact in f16, f32 accumulation keeps the dot products
    /// exact) make `assert_eq` valid — the same construction the `nax_matmul_*`
    /// oracle tests use. The GEMV kernel is plain MSL (no MPP/`matmul2d`), so it
    /// runs on EVERY Apple GPU tier — no NAX gate needed; only the no-device skip.
    #[test]
    fn metal_gemv_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping GPU GEMV oracle test");
                return;
            }
            Err(e) => panic!("GEMV GEMM compile failed: {e}"),
        };
        // (k, n) shapes: thin, exact-tile-ish, ragged N, deep K, wide N, n=1.
        let shapes = [
            (1usize, 1usize),
            (16, 32),
            (5, 17),
            (64, 33),
            (256, 7),
            (3, 100),
            (33, 1),
        ];
        for (k, n) in shapes {
            // x in 0..3, B in 0..4 — exact in f16; sums over K stay under f16's
            // 2048 exact-integer ceiling for these K.
            let x: Vec<f32> = (0..k).map(|i| (i % 3) as f32).collect();
            // Plain: B is [k, n].
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let got = ctx.gemv(k, n, &x, &b, /*transpose_b=*/ false).unwrap();
            let want = crate::blas::naive_sgemv(k, n, &x, &b);
            assert_eq!(got, want, "GPU GEMV mismatch at (k={k}, n={n})");
            // It must also equal the m=1 row of the naive GEMM (the path it replaces).
            assert_eq!(
                got,
                crate::blas::naive_sgemm(1, k, n, &x, &b),
                "GPU GEMV must equal naive_sgemm at m=1 (k={k}, n={n})"
            );

            // Transpose-B: B is [n, k]; y[j] = Σ_k x[k]·B[j,k].
            let b_nk: Vec<f32> = (0..n * k).map(|i| (i % 4) as f32).collect();
            let got_bt = ctx.gemv(k, n, &x, &b_nk, /*transpose_b=*/ true).unwrap();
            let want_bt = crate::blas::naive_sgemv_bt(k, n, &x, &b_nk);
            assert_eq!(
                got_bt, want_bt,
                "GPU GEMV transpose-B mismatch at (k={k}, n={n})"
            );
            assert_eq!(
                got_bt,
                crate::blas::naive_sgemm_bt(1, k, n, &x, &b_nk),
                "GPU GEMV-bt must equal naive_sgemm_bt at m=1 (k={k}, n={n})"
            );
        }
        // Fused epilogue path (bias add) over the zero-copy unified buffers: the
        // GEMV shares the GEMM's (binop, act) epilogue codes, so check one.
        {
            let (k, n) = (8usize, 12usize);
            let x: Vec<f32> = (0..k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let e: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
            let ux = ctx.unified_from(&x).unwrap();
            let ub = ctx.unified_from(&b).unwrap();
            let ue = ctx.unified_from(&e).unwrap();
            let mut uy = ctx.unified(n).unwrap();
            ctx.gemv_unified(k, n, &ux, &ub, &mut uy, Some(&ue), Epilogue::ADD, false)
                .unwrap();
            let base = crate::blas::naive_sgemv(k, n, &x, &b);
            let want: Vec<f32> = base.iter().zip(&e).map(|(v, ev)| v + ev).collect();
            assert_eq!(
                uy.as_slice(),
                want.as_slice(),
                "GPU GEMV fused-add mismatch"
            );
        }
        eprintln!("GPU GEMV matches the oracle (plain + transpose-B + fused) ✓");
    }

    /// A batched matmul chain (one command buffer, one sync) computes the same
    /// result as the matmuls run separately, and amortizes the per-dispatch
    /// latency: a chain of N small matmuls should be far faster than N calls.
    #[test]
    fn matmul_chain_matches_and_amortizes() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        // x (m×k0) · W1 (k0×k1) · W2 (k1×k2) · W3 (k2×k3), with a bias+relu epilogue.
        let (m, k0, k1, k2, k3) = (128usize, 128, 128, 128, 128);
        // Positive inputs: chained f16 matmuls don't cancel, so the f32 oracle
        // stays within f16 tolerance (signed inputs would cancel near zero and
        // blow up the *relative* error without any bug).
        let mk = |rows: usize, cols: usize, s: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i + s) % 7) as f32 * 0.03 + 0.01)
                .collect()
        };
        let x = mk(m, k0, 0);
        let (w1, w2, w3) = (mk(k0, k1, 1), mk(k1, k2, 2), mk(k2, k3, 3));
        let bias = mk(m, k3, 9);

        let steps = [
            ChainStep {
                k: k0,
                n: k1,
                b: &w1,
                epi: Epilogue::NONE,
                e: None,
            },
            ChainStep {
                k: k1,
                n: k2,
                b: &w2,
                epi: Epilogue::NONE,
                e: None,
            },
            ChainStep {
                k: k2,
                n: k3,
                b: &w3,
                epi: Epilogue { binop: 1, act: 1 },
                e: Some(&bias),
            },
        ];
        let got = ctx.run_chain(m, &x, &steps).unwrap();

        // Oracle: same chain on the CPU (f16 tolerance, since NAX is f16).
        let c1 = crate::blas::naive_sgemm(m, k0, k1, &x, &w1);
        let c2 = crate::blas::naive_sgemm(m, k1, k2, &c1, &w2);
        let c3 = crate::blas::naive_sgemm(m, k2, k3, &c2, &w3);
        let want: Vec<f32> = c3
            .iter()
            .zip(&bias)
            .map(|(&c, &b)| (c + b).max(0.0))
            .collect();
        let mut max_rel = 0.0f32;
        for (g, w) in got.iter().zip(&want) {
            max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
        }
        assert!(
            max_rel < 0.1,
            "chain result max rel err {max_rel} too large"
        );

        // Timing: the 3-matmul chain (one sync) vs three separate run() calls.
        let iters = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            ctx.run_chain(m, &x, &steps).unwrap();
        }
        let chained = t0.elapsed().as_secs_f64() / iters as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let a = ctx.run(m, k0, k1, &x, &w1).unwrap();
            let b = ctx.run(m, k1, k2, &a, &w2).unwrap();
            let _ = ctx
                .run_fused(m, k2, k3, &b, &w3, &bias, Epilogue { binop: 1, act: 1 })
                .unwrap();
        }
        let separate = t1.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "matmul chain: batched {:.1} µs vs separate {:.1} µs  ({:.2}× faster, one sync vs three)",
            chained * 1e6,
            separate * 1e6,
            separate / chained
        );
    }

    /// **Combining** many small same-weight matmuls into one tall GEMM is both
    /// correct (matches per-slice) AND faster than a serial AMX loop once the
    /// matmul is compute-bound (K ≳ 512) — the real way to exploit a SPMD grid's
    /// many small matmuls on the GPU. Measured speedups: ~1.25× at K=512 up to
    /// ~1.74× at K=2048 (see the module bench).
    #[test]
    fn combined_matmul_matches_and_wins() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        // 16 cores each multiply their 512 rows by the SAME 1024×1024 weights.
        let (count, m, k, n) = (16usize, 512usize, 1024usize, 1024usize);
        let a: Vec<f32> = (0..count * m * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect();

        let got = ctx.run_combined(count, m, k, n, &a, &b).unwrap();
        // Correctness: each core's rows match its own matmul (f16 tolerance).
        let mut max_rel = 0.0f32;
        for s in 0..count {
            let want = crate::blas::naive_sgemm(m, k, n, &a[s * m * k..(s + 1) * m * k], &b);
            for (g, w) in got[s * m * n..(s + 1) * m * n].iter().zip(&want) {
                max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
            }
        }
        assert!(max_rel < 0.05, "combined mismatch, max rel err {max_rel}");

        // Speed: combined GEMM vs the serial per-core AMX loop it replaces.
        let it = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            ctx.run_combined(count, m, k, n, &a, &b).unwrap();
        }
        let combined = t0.elapsed().as_secs_f64() / it as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..it {
            for s in 0..count {
                std::hint::black_box(crate::blas::sgemm_rowmajor(
                    m,
                    k,
                    n,
                    &a[s * m * k..(s + 1) * m * k],
                    &b,
                ));
            }
        }
        let amx_loop = t1.elapsed().as_secs_f64() / it as f64;
        eprintln!(
            "combine {count}×({m}×{k}×{n}): GPU one tall GEMM {:.0} µs vs serial AMX {:.0} µs  ({:.2}×)",
            combined * 1e6,
            amx_loop * 1e6,
            amx_loop / combined
        );
    }

    /// Zero-copy unified-memory matmul: correct, and free of the host↔device
    /// fill/readback the copy-based `run` pays (CPU and GPU share the bytes).
    #[test]
    fn unified_matmul_zero_copy_matches_and_is_faster() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (m, k, n) = (4096usize, 1024usize, 1024usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect();

        let ua = ctx.unified_from(&a).unwrap();
        let ub = ctx.unified_from(&b).unwrap();
        let mut uc = ctx.unified(m * n).unwrap();
        ctx.matmul_unified(m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE, false)
            .unwrap();

        // Correctness vs the copy-based path (same kernel, identical result).
        let want = ctx.run(m, k, n, &a, &b).unwrap();
        let mut max_abs = 0.0f32;
        for (g, w) in uc.as_slice().iter().zip(&want) {
            max_abs = max_abs.max((g - w).abs());
        }
        assert!(max_abs < 1e-3, "unified vs copy-path differ by {max_abs}");

        // Speed: zero-copy (operands already resident) vs run() which fills A,B
        // and reads C back every call.
        let it = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            ctx.matmul_unified(m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE, false)
                .unwrap();
        }
        let zc = t0.elapsed().as_secs_f64() / it as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..it {
            std::hint::black_box(ctx.run(m, k, n, &a, &b).unwrap());
        }
        let copied = t1.elapsed().as_secs_f64() / it as f64;
        eprintln!(
            "unified {m}×{k}×{n}: zero-copy {:.0} µs vs copy-path {:.0} µs  ({:.2}× faster, copies removed)",
            zc * 1e6,
            copied * 1e6,
            copied / zc
        );
    }

    /// A batched dispatch (independent same-shape GEMMs in one submission)
    /// matches running them separately. (For *shared*-weight matmuls,
    /// `run_combined` is the faster path — see `combined_matmul_matches_and_wins`.)
    #[test]
    fn batched_matmul_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (batch, m, k, n) = (64usize, 256usize, 256usize, 256usize);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
            .collect();

        let got = ctx.run_batched(batch, m, k, n, &a, &b).unwrap();
        let mut max_rel = 0.0f32;
        for s in 0..batch {
            let want = crate::blas::naive_sgemm(
                m,
                k,
                n,
                &a[s * m * k..(s + 1) * m * k],
                &b[s * k * n..(s + 1) * k * n],
            );
            for (g, w) in got[s * m * n..(s + 1) * m * n].iter().zip(&want) {
                max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
            }
        }
        assert!(max_rel < 0.05, "batched mismatch, max rel err {max_rel}");
    }

    /// The pre-M5 `simdgroup_float8x8` GEMM is correct across shapes (incl.
    /// ragged) and supports the same fused epilogue. Forced on the M5 so we can
    /// validate that code path here.
    #[test]
    fn simdgroup_matmul_matches_oracle() {
        let ctx = match NaxGemm::new_simdgroup() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("simdgroup compile failed: {e}"),
        };
        // Plain matmul across shapes (small ints exact in f32).
        for (m, k, n) in [
            (8usize, 8usize, 8usize),
            (17, 33, 5),
            (50, 20, 70),
            (100, 7, 3),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let got = ctx.run(m, k, n, &a, &b).unwrap();
            let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
            assert_eq!(got, want, "simdgroup GEMM mismatch at ({m},{k},{n})");
        }
        // Fused epilogue (add + relu) matches matmul-then-elementwise.
        let (m, k, n) = (40usize, 24usize, 56usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let e: Vec<f32> = (0..m * n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let got = ctx
            .run_fused(m, k, n, &a, &b, &e, Epilogue { binop: 1, act: 1 })
            .unwrap();
        let mm = crate::blas::naive_sgemm(m, k, n, &a, &b);
        for i in 0..m * n {
            let want = (mm[i] + e[i]).max(0.0);
            assert!(
                (got[i] - want).abs() < 1e-3,
                "simdgroup fused mismatch at {i}"
            );
        }
        eprintln!("simdgroup_float8x8 GEMM (+fused epilogue) matches the oracle ✓");
    }

    /// Fused matmul→elementwise epilogue (`D = act(A·B BINOP E)`) computed in one
    /// kernel matches doing the matmul then the elementwise op separately.
    #[test]
    fn nax_matmul_fused_epilogue_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (m, k, n) = (130usize, 40usize, 200usize); // ragged, multi-block
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let e: Vec<f32> = (0..m * n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let mm = crate::blas::naive_sgemm(m, k, n, &a, &b);

        type Case = (Epilogue, fn(f32, f32) -> f32);
        let cases: &[Case] = &[
            (Epilogue::ADD, |c, ev| c + ev),
            (Epilogue::MUL, |c, ev| c * ev),
            (Epilogue::SUB, |c, ev| c - ev),
            (Epilogue::MAX, |c, ev| c.max(ev)),
            (Epilogue::RELU, |c, _| c.max(0.0)),
            (Epilogue { binop: 1, act: 1 }, |c, ev| (c + ev).max(0.0)), // add + relu
            (Epilogue { binop: 1, act: 2 }, |c, ev| (c + ev).tanh()),   // add + tanh
        ];
        for &(epi, f) in cases {
            let got = ctx.run_fused(m, k, n, &a, &b, &e, epi).unwrap();
            let mut max_rel = 0.0f32;
            for i in 0..m * n {
                let want = f(mm[i], e[i]);
                max_rel = max_rel.max((got[i] - want).abs() / want.abs().max(1.0));
            }
            assert!(
                max_rel < 0.05,
                "fused {epi:?}: max rel err {max_rel} > f16 tol"
            );
        }
        eprintln!(
            "fused matmul→elementwise epilogue matches oracle across {} ops ✓",
            cases.len()
        );
    }

    /// Random (non-f16-exact) data: the NAX GEMM agrees with the f32 oracle to
    /// f16 tolerance. Documents the precision the engine actually delivers.
    #[test]
    fn nax_matmul_general_f16_tolerance() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (m, k, n) = (64usize, 48usize, 96usize);
        // Deterministic pseudo-random in [-1, 1].
        let prng = |i: usize| ((i.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0;
        let a: Vec<f32> = (0..m * k).map(prng).collect();
        let b: Vec<f32> = (0..k * n).map(|i| prng(i + 7)).collect();
        let got = ctx.run(m, k, n, &a, &b).unwrap();
        let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
        let mut max_rel = 0.0f32;
        for (g, w) in got.iter().zip(&want) {
            let denom = w.abs().max(1.0);
            max_rel = max_rel.max((g - w).abs() / denom);
        }
        // f16 has 8 mantissa bits; K=48 accumulation in f32 keeps error modest.
        assert!(
            max_rel < 0.05,
            "max relative error {max_rel} exceeds f16 tolerance"
        );
        eprintln!("NAX GEMM vs f32 oracle: max relative error {max_rel:.4} (f16) ✓");
    }

    /// Real benchmark: NAX vs naive vs the linked BLAS (Accelerate on macOS) on
    /// a sizeable GEMM. Prints GFLOP/s for each so the speedup is concrete.
    /// `--ignored` because it's a perf measurement, not a correctness gate.
    ///
    /// Observed on an M5 (numbers vary with thermals):
    ///
    /// - GPU-only kernel throughput climbs with size and plateaus ~4 TFLOP/s at
    ///   2048³+ (where #threadgroups finally fills the cores); at 1024³ it is
    ///   occupancy-bound (~32 threadgroups) and small sizes are far worse.
    /// - At its plateau the kernel is ~2× Apple Accelerate (AMX, ~2 TFLOP/s).
    /// - Per-call wall-clock is dominated by buffer alloc + host/device copy +
    ///   readback; `gpu_time_seconds` isolates the kernel from that overhead.
    ///
    /// The remaining gap to NAX's true peak is the per-K-step staging+barrier
    /// tax — a double-buffered kernel (overlap load with compute) is the next win.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_nax_vs_blas() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping benchmark: {e}");
                return;
            }
        };
        let prng = |i: usize| (i.wrapping_mul(2654435761) % 1000) as f32 / 1000.0;

        // GPU-only throughput sweep across sizes: diagnoses whether the kernel
        // is occupancy-bound (climbs as #threadgroups grows) or compute-bound
        // (plateaus). #threadgroups = ceil(s/128) * ceil(s/256).
        eprintln!("-- GPU-only throughput sweep (kernel time only) --");
        for &s in &[256usize, 512, 1024, 2048, 4096] {
            let a: Vec<f32> = (0..s * s).map(prng).collect();
            let b: Vec<f32> = (0..s * s).map(|i| prng(i + 3)).collect();
            let iters = if s <= 1024 { 50 } else { 10 };
            let g = ctx.gpu_time_seconds(s, s, s, &a, &b, iters).unwrap() / iters as f64;
            let tgs = s.div_ceil(128) * s.div_ceil(256);
            eprintln!(
                "  {s:>4}^3: {:7.3} ms   {:7.1} GFLOP/s   ({tgs} threadgroups)",
                g * 1e3,
                2.0 * (s as f64).powi(3) / g / 1e9
            );
        }

        let (m, k, n) = (1024usize, 1024usize, 1024usize);
        let a: Vec<f32> = (0..m * k).map(prng).collect();
        let b: Vec<f32> = (0..k * n).map(|i| prng(i + 3)).collect();
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        let bench = |label: &str, iters: u32, mut f: Box<dyn FnMut()>| {
            f(); // warm up
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!(
                "{label:>12}: {:7.2} ms   {:7.1} GFLOP/s",
                secs * 1e3,
                flops / secs / 1e9
            );
        };

        // GPU-only kernel time (excludes alloc/copy/readback): 50 dispatches on
        // reused buffers, timed by hardware timestamps. Isolates kernel speed.
        let gpu_total = ctx.gpu_time_seconds(m, k, n, &a, &b, 50).unwrap();
        let gpu_each = gpu_total / 50.0;
        eprintln!(
            "{:>12}: {:7.2} ms   {:7.1} GFLOP/s   (GPU kernel only)",
            "NAX-gpu",
            gpu_each * 1e3,
            flops / gpu_each / 1e9
        );

        let (a1, b1) = (a.clone(), b.clone());
        bench(
            "NAX-wall",
            20,
            Box::new(move || {
                ctx.run(m, k, n, &a1, &b1).unwrap();
            }),
        );
        let (a2, b2) = (a.clone(), b.clone());
        bench(
            "BLAS/accel",
            20,
            Box::new(move || {
                std::hint::black_box(crate::blas::sgemm_rowmajor(m, k, n, &a2, &b2));
            }),
        );
        let (a3, b3) = (a.clone(), b.clone());
        bench(
            "naive",
            1,
            Box::new(move || {
                std::hint::black_box(crate::blas::naive_sgemm(m, k, n, &a3, &b3));
            }),
        );
    }

    #[test]
    fn lowers_vector_add_to_msl() {
        let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
        let module = parse_module(src).unwrap();
        let msl = emit_msl(&module, "add_kernel").expect("emit MSL");

        // Structural checks on the emitted shader.
        assert!(msl.contains("#include <metal_stdlib>"));
        assert!(msl.contains("kernel void add_kernel("));
        assert!(msl.contains("thread_position_in_grid"));
        // Three f16 buffers: two read-only inputs, one writable output.
        assert!(msl.contains("device const half* x_ptr [[buffer(0)]]"));
        assert!(msl.contains("device const half* y_ptr [[buffer(1)]]"));
        assert!(msl.contains("device half* output_ptr [[buffer(2)]]"));
        // The element-wise add, with the output buffer on the LHS.
        assert!(
            msl.contains("output_ptr[gid] = x_ptr[gid] + y_ptr[gid];"),
            "unexpected body:\n{msl}"
        );
    }

    #[test]
    fn fuses_elementwise_chain_into_one_expression() {
        // exp(a * b) + c  -> a single fused kernel, not three passes.
        let src = r#"
module {
  func.func @chain(%a_ptr: index, %b_ptr: index, %c_ptr: index, %out_ptr: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %va = ktdp.construct_memory_view %a_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %vb = ktdp.construct_memory_view %b_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %vc = ktdp.construct_memory_view %c_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %ta = ktdp.construct_access_tile %va[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %tb = ktdp.construct_access_tile %vb[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %tc = ktdp.construct_access_tile %vc[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %la = ktdp.load %ta : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %lb = ktdp.load %tb : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %lc = ktdp.load %tc : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %ab = arith.mulf %la, %lb : tensor<8xf16>
    %e = math.exp %ab : tensor<8xf16>
    %r = arith.addf %e, %lc : tensor<8xf16>
    %vout = ktdp.construct_memory_view %out_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %tout = ktdp.construct_access_tile %vout[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    ktdp.store %r, %tout : tensor<8xf16>, !ktdp.access_tile<8xindex>
    return
  }
}
"#;
        let module = parse_module(src).unwrap();
        let kernel = emit_kernel(&module, "chain").expect("emit fused chain");
        // One kernel, three input buffers (a,b,c) + one output, deduped & ordered.
        let names: Vec<&str> = kernel.buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["a_ptr", "b_ptr", "c_ptr", "out_ptr"],
            "fused buffer set"
        );
        assert_eq!(kernel.buffers.iter().filter(|b| b.is_output).count(), 1);
        // The whole DAG collapses into one assignment: exp(a*b) + c.
        assert!(
            kernel
                .source
                .contains("out_ptr[gid] = (exp((a_ptr[gid] * b_ptr[gid]))) + c_ptr[gid];"),
            "expected one fused expression, got:\n{}",
            kernel.source
        );
    }

    #[test]
    fn fuses_constants_casts_and_splat() {
        // half a -> float, scale by a splat constant in f32, narrow back to half:
        //   out = half(float(a) * 2.0)
        let src = r#"
module {
  func.func @scale(%a_ptr: index, %out_ptr: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %va = ktdp.construct_memory_view %a_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %ta = ktdp.construct_access_tile %va[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %la = ktdp.load %ta : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %xf = arith.extf %la : tensor<8xf16> to tensor<8xf32>
    %c2 = arith.constant 2.0 : f32
    %s = tensor.splat %c2 : tensor<8xf32>
    %m = arith.mulf %xf, %s : tensor<8xf32>
    %t = arith.truncf %m : tensor<8xf32> to tensor<8xf16>
    %vout = ktdp.construct_memory_view %out_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %tout = ktdp.construct_access_tile %vout[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    ktdp.store %t, %tout : tensor<8xf16>, !ktdp.access_tile<8xindex>
    return
  }
}
"#;
        let module = parse_module(src).unwrap();
        let kernel = emit_kernel(&module, "scale").expect("emit scale chain");
        // Only `a` is a real buffer; the constant folded into the expression.
        let names: Vec<&str> = kernel.buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["a_ptr", "out_ptr"], "constant is not a buffer");
        assert!(
            kernel
                .source
                .contains("out_ptr[gid] = half(((float(a_ptr[gid])) * ((2.0))));"),
            "unexpected fused body:\n{}",
            kernel.source
        );
    }

    // --- kernel scheduling (partitioner) --------------------------------

    /// Build a bare op with just a type (the partitioner only reads op_type).
    fn op(ty: &str) -> Operation {
        Operation::new(Some("%r"), ty, &[])
    }

    #[test]
    fn partitions_rmsnorm_shape_into_map_reduce_map() {
        // load, [extf, mulf], reduce, [divf, sqrt], store, return
        // -> Map([1,2]), Reduce(3), Map([4,5])  (plumbing skipped, not boundaries)
        let ops = vec![
            op("ktdp.load"),     // 0 plumbing
            op("arith.extf"),    // 1 map
            op("arith.mulf"),    // 2 map
            op("linalg.reduce"), // 3 reduce
            op("arith.divf"),    // 4 map
            op("math.sqrt"),     // 5 map
            op("ktdp.store"),    // 6 plumbing
            op("func.return"),   // 7 plumbing
        ];
        let plan = plan_kernels(&ops).unwrap();
        assert_eq!(
            plan,
            vec![
                KernelRegion::Map(vec![1, 2]),
                KernelRegion::Reduce(3),
                KernelRegion::Map(vec![4, 5]),
            ]
        );
    }

    #[test]
    fn partitions_matmul_as_its_own_region() {
        // broadcast then matmul then add -> Map, Matmul, Map
        let ops = vec![
            op("linalg.broadcast"), // 0 map
            op("linalg.matmul"),    // 1 matmul
            op("arith.addf"),       // 2 map
        ];
        let plan = plan_kernels(&ops).unwrap();
        assert_eq!(
            plan,
            vec![
                KernelRegion::Map(vec![0]),
                KernelRegion::Matmul(1),
                KernelRegion::Map(vec![2]),
            ]
        );
    }

    #[test]
    fn unfusable_op_forces_fallback() {
        let ops = vec![op("arith.mulf"), op("scf.for"), op("arith.addf")];
        assert!(
            plan_kernels(&ops).is_err(),
            "bare scf.for must force a fallback"
        );
    }

    /// Build a K-loop matmul function: A is either a forwarded extract_slice of a
    /// `[m,k]` producer (prefill/decode forwarded activation) or a load of an
    /// `[m,k]` view; B is a load of a `[k,n]` weight view. Mirrors the real fused
    /// K-loop so recognition is exercised end to end.
    fn matmul_loop_fn(a_via_slice: bool, m: i64, k: i64, n: i64) -> Vec<Operation> {
        let il = |v: Vec<i64>| Attr::IntList(v);
        let mut top = vec![
            // A's full source / producer, shape [m,k]. Plumbing (tensor.empty)
            // so this focused test's plan is just the MatmulLoop; in the real
            // fused fn the source is a preceding map/reduce region's output.
            Operation::new(Some("%src"), "tensor.empty", &[]).with_attr("shape", il(vec![m, k])),
            // B weight view over %wptr, shape [k,n].
            Operation::new(Some("%vw"), "ktdp.construct_memory_view", &["%wptr"])
                .with_attr("shape", il(vec![k, n])),
        ];
        let mut body = Vec::new();
        if a_via_slice {
            body.push(
                Operation::new(Some("%a"), "tensor.extract_slice", &["%src"]).with_attr(
                    "slice_sizes",
                    Attr::StrList(vec!["1".into(), k.to_string()]),
                ),
            );
        } else {
            // A via a load of a [m,k] view over %aptr.
            top.push(
                Operation::new(Some("%va"), "ktdp.construct_memory_view", &["%aptr"])
                    .with_attr("shape", il(vec![m, k])),
            );
            body.push(
                Operation::new(
                    Some("%at"),
                    "ktdp.construct_access_tile",
                    &["%va", "%pid", "%kk"],
                )
                .with_attr("shape", il(vec![1, k])),
            );
            body.push(Operation::new(Some("%a"), "ktdp.load", &["%at"]));
        }
        body.push(
            Operation::new(
                Some("%bt"),
                "ktdp.construct_access_tile",
                &["%vw", "%kk", "%c0"],
            )
            .with_attr("shape", il(vec![k, n])),
        );
        body.push(Operation::new(Some("%b"), "ktdp.load", &["%bt"]));
        body.push(Operation::new(Some("%cinit"), "arith.constant", &[]));
        body.push(
            Operation::new(Some("%part"), "linalg.matmul", &["%a", "%b", "%cinit"])
                .with_attr("shape", il(vec![m, n])),
        );
        body.push(Operation::new(
            Some("%accnext"),
            "arith.addf",
            &["%acc", "%part"],
        ));
        body.push(Operation::new(None, "scf.yield", &["%accnext"]));
        let mut forop = Operation::new(Some("%mm"), "scf.for", &["%c0", "%K", "%KB", "%azero"])
            .with_attr("iter_var", Attr::Str("%kk".into()))
            .with_attr("iter_args", Attr::StrList(vec!["%acc".into()]));
        forop.regions = vec![body];
        top.push(forop);
        top
    }

    #[test]
    fn recognizes_prefill_matmul_kloop_as_m8_gemm() {
        // A = extract_slice of an [8,576] activation (the forwarded fused form);
        // B = [576,576] weight. The grid/K-tiling collapses to one M=8 GEMM.
        let ops = matmul_loop_fn(true, 8, 576, 576);
        let plan = plan_kernels(&ops).unwrap();
        assert_eq!(
            plan,
            vec![KernelRegion::MatmulLoop(MatmulLoopInfo {
                m: 8,
                k: 576,
                n: 576,
                a_root: "%src".into(),
                b_root: "%wptr".into(),
                out_ssa: "%mm".into(),
                n_off: 0,
                b_stride: 576,
                transpose_b: false,
                m_row_off: 0,
            })],
            "prefill K-loop must collapse to a single [8,576]@[576,576] GEMM"
        );
    }

    #[test]
    fn recognizes_decode_matmul_kloop_as_m1_gemm() {
        // A via a load of a [1,576] view (decode), B = [576,576]. Same recognizer,
        // M=1 from the full view shape.
        let ops = matmul_loop_fn(false, 1, 576, 576);
        let plan = plan_kernels(&ops).unwrap();
        assert_eq!(
            plan,
            vec![KernelRegion::MatmulLoop(MatmulLoopInfo {
                m: 1,
                k: 576,
                n: 576,
                a_root: "%aptr".into(),
                b_root: "%wptr".into(),
                out_ssa: "%mm".into(),
                n_off: 0,
                b_stride: 576,
                transpose_b: false,
                m_row_off: 0,
            })]
        );
    }

    #[test]
    fn recognizes_transpose_b_matmul_kloop() {
        // A transpose-B K-loop: the body `linalg.matmul` carries transpose-B
        // `indexing_maps` (B map (d0,d1,d2)->(d1,d2)) and B's view is [n,k]
        // (on-disk Linear [out,in]). The recognizer must set transpose_b=true,
        // derive n from B's FIRST axis and k from its LAST (== A's k), n_off=0.
        let (m, k, n) = (8, 576, 512);
        let mut ops = matmul_loop_fn(true, m, k, n);
        // Flip B's view shape [k,n] -> [n,k] (top-level %vw), and tag the matmul
        // op (which lives INSIDE the scf.for body region) with transpose-B maps.
        for op in ops.iter_mut() {
            if op.result.as_deref() == Some("%vw") {
                op.attributes
                    .insert("shape".into(), Attr::IntList(vec![n, k]));
            }
            if op.op_type == "scf.for" {
                for body_op in op.regions[0].iter_mut() {
                    if body_op.op_type == "linalg.matmul" {
                        let p = |s: &str| crate::parser_ast::parse_affine_map(s).unwrap();
                        body_op.attributes.insert(
                            "indexing_maps".into(),
                            Attr::AffineMapList(vec![
                                p("affine_map<(d0, d1, d2) -> (d0, d2)>"),
                                p("affine_map<(d0, d1, d2) -> (d1, d2)>"),
                                p("affine_map<(d0, d1, d2) -> (d0, d1)>"),
                            ]),
                        );
                    }
                }
            }
        }
        let plan = plan_kernels(&ops).unwrap();
        assert_eq!(
            plan,
            vec![KernelRegion::MatmulLoop(MatmulLoopInfo {
                m,
                k,
                n,
                a_root: "%src".into(),
                b_root: "%wptr".into(),
                out_ssa: "%mm".into(),
                n_off: 0,
                b_stride: n,
                transpose_b: true,
                m_row_off: 0,
            })],
            "transpose-B K-loop must be recognized with transpose_b=true and [n,k] B"
        );
    }

    #[test]
    fn non_matmul_scf_for_still_falls_back() {
        // A loop whose body is not the matmul-accumulate template -> Err.
        let mut body = vec![
            Operation::new(Some("%t"), "arith.mulf", &["%acc", "%acc"]),
            Operation::new(None, "scf.yield", &["%t"]),
        ];
        let mut forop = Operation::new(Some("%r"), "scf.for", &["%c0", "%K", "%KB", "%azero"])
            .with_attr("iter_args", Attr::StrList(vec!["%acc".into()]));
        forop.regions = vec![std::mem::take(&mut body)];
        assert!(
            plan_kernels(&[forop]).is_err(),
            "non-matmul loop must fall back"
        );
    }

    #[test]
    fn map_window_respects_size_cap() {
        // 2*CAP + 5 consecutive map ops -> windows of CAP, CAP, then 5.
        let n = MAX_KERNEL_WINDOW * 2 + 5;
        let ops: Vec<Operation> = (0..n).map(|_| op("arith.addf")).collect();
        let plan = plan_kernels(&ops).unwrap();
        let sizes: Vec<usize> = plan
            .iter()
            .map(|r| match r {
                KernelRegion::Map(v) => v.len(),
                _ => panic!("expected only map regions"),
            })
            .collect();
        assert_eq!(sizes, vec![MAX_KERNEL_WINDOW, MAX_KERNEL_WINDOW, 5]);
    }

    #[test]
    fn fuses_broadcast_per_column_weight() {
        // out[1,576] = a[1,576] * broadcast(w[576], dims=[0])  — RMSNorm's final
        // per-column gamma multiply. The weight indexes by column (gid % 576),
        // not gid, so this exercises shape-aware broadcast indexing.
        let src = r#"
module {
  func.func @scale(%a_ptr: index, %w_ptr: index, %out_ptr: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %va = ktdp.construct_memory_view %a_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %vw = ktdp.construct_memory_view %w_ptr, sizes: [576], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 575 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<576xf16>
    %ta = ktdp.construct_access_tile %va[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>, access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    %tw = ktdp.construct_access_tile %vw[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 575 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<576xf16> -> !ktdp.access_tile<576xindex>
    %la = ktdp.load %ta : !ktdp.access_tile<1x576xindex> -> tensor<1x576xf16>
    %lw = ktdp.load %tw : !ktdp.access_tile<576xindex> -> tensor<576xf16>
    %ginit = tensor.empty() : tensor<1x576xf16>
    %gb = linalg.broadcast ins(%lw : tensor<576xf16>) outs(%ginit : tensor<1x576xf16>) dimensions = [0]
    %y = arith.mulf %la, %gb : tensor<1x576xf16>
    %vout = ktdp.construct_memory_view %out_ptr, sizes: [1, 576], strides: [576, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x576xf16>
    %tout = ktdp.construct_access_tile %vout[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 575 >= 0)>, access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x576xf16> -> !ktdp.access_tile<1x576xindex>
    ktdp.store %y, %tout : tensor<1x576xf16>, !ktdp.access_tile<1x576xindex>
    return
  }
}
"#;
        let module = parse_module(src).unwrap();
        let kernel = emit_kernel(&module, "scale").expect("emit broadcast chain");
        let names: Vec<&str> = kernel.buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["a_ptr", "w_ptr", "out_ptr"]);
        assert!(
            kernel
                .source
                .contains("out_ptr[gid] = a_ptr[gid] * w_ptr[(gid % 576)];"),
            "unexpected broadcast body:\n{}",
            kernel.source
        );
    }

    #[test]
    fn gpu_matches_oracle_vector_add() {
        use crate::dtypes::DType;
        use crate::interpreter::{Arg, execute_function};
        use crate::ir::Scalar;

        let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
        let module = parse_module(src).unwrap();
        let kernel = emit_kernel(&module, "add_kernel").unwrap();

        let n = 4096usize;
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();

        let gpu = match run_kernel(&kernel, &[x.clone(), y.clone()], n) {
            Ok(g) => g,
            // No GPU in this environment (e.g. headless CI) — skip, don't fail.
            Err(e) if e.contains("no Metal device") => {
                eprintln!("skipping GPU validation: {e}");
                return;
            }
            Err(e) => panic!("GPU run failed: {e}"),
        };

        // Oracle: the same kernel through the CPU interpreter.
        let args = [
            (
                "x_ptr",
                Arg::Tensor {
                    data: x,
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            (
                "y_ptr",
                Arg::Tensor {
                    data: y,
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            (
                "output_ptr",
                Arg::Tensor {
                    data: vec![0.0; n],
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
        ];
        let oracle = execute_function(&module, "add_kernel", &args).unwrap();
        let oracle = &oracle.get("output_ptr").unwrap().data;

        assert_eq!(gpu.len(), n);
        for i in 0..n {
            assert!(
                (gpu[i] - oracle[i]).abs() < 1e-2,
                "GPU vs oracle mismatch at {i}: gpu={}, oracle={}",
                gpu[i],
                oracle[i]
            );
        }
        eprintln!("GPU output matches the interpreter oracle over {n} elements ✓");
    }

    #[test]
    fn rejects_non_elementwise() {
        // matmul_small has a linalg.matmul -> not lowerable in slice 1.
        let src = include_str!("../../../../examples/latency/matmul_small.mlir");
        if let Ok(module) = parse_module(src) {
            let name = module.functions.keys().next().unwrap().clone();
            assert!(emit_msl(&module, &name).is_err());
        }
    }

    #[test]
    fn matmul_tier_detection() {
        use super::MatmulTier::*;
        // M5+ -> NAX (Neural Accelerator).
        assert_eq!(device_matmul_tier("Apple M5"), Nax);
        assert_eq!(device_matmul_tier("Apple M5 Pro"), Nax);
        assert_eq!(device_matmul_tier("Apple M6 Max"), Nax); // forward-compatible
        // M1..M4 Apple GPUs -> simdgroup matrix units.
        assert_eq!(device_matmul_tier("Apple M1"), Simdgroup);
        assert_eq!(device_matmul_tier("Apple M3 Max"), Simdgroup);
        assert_eq!(device_matmul_tier("Apple M4"), Simdgroup);
        // An Apple GPU with no M-number still gets the matrix path.
        assert_eq!(device_matmul_tier("Apple Paravirtual device"), Simdgroup);
        // Non-Apple -> naive floor.
        assert_eq!(device_matmul_tier("Intel UHD Graphics 630"), Naive);
        assert_eq!(device_matmul_tier("AMD Radeon Pro 5500M"), Naive);
        // Effective tier is the best *implemented* tier the device supports —
        // and all three are implemented now.
        assert_eq!(effective_matmul_tier("Apple M5"), HIGHEST_IMPLEMENTED);
        assert_eq!(effective_matmul_tier("Apple M5"), Nax);
        // Pre-NAX Apple GPUs use the simdgroup_float8x8 GPU path.
        assert_eq!(effective_matmul_tier("Apple M4"), Simdgroup);
        assert_eq!(effective_matmul_tier("Apple M1"), Simdgroup);
        // Non-Apple stays at the naive floor.
        assert_eq!(effective_matmul_tier("Intel UHD Graphics 630"), Naive);
        assert!(tier_implemented(Simdgroup));
    }

    #[test]
    fn matmul_backend_gating() {
        use MatmulBackend::{Accelerate, Nax};
        // M5 sends large GEMMs (>= measured ~1024³ crossover) to the NAX engine.
        assert_eq!(choose_matmul_backend("Apple M5", 1024, 1024, 1024), Nax); // 32 blocks
        assert_eq!(choose_matmul_backend("Apple M5", 2048, 2048, 2048), Nax);
        // Smaller / LX-sized matmuls -> Accelerate (AMX), faster there.
        assert_eq!(choose_matmul_backend("Apple M5", 512, 512, 512), Accelerate); // 8 blocks < 32
        assert_eq!(choose_matmul_backend("Apple M5", 256, 256, 256), Accelerate);
        // Pre-M5: the simdgroup GPU path never beats AMX in wall-clock -> Accelerate.
        assert_eq!(
            choose_matmul_backend("Apple M4", 2048, 2048, 2048),
            Accelerate
        );
        assert_eq!(
            choose_matmul_backend("Apple M1", 4096, 4096, 4096),
            Accelerate
        );
        // Non-Apple GPUs -> Accelerate.
        assert_eq!(
            choose_matmul_backend("Intel UHD Graphics 630", 4096, 4096, 4096),
            Accelerate
        );
    }

    #[test]
    fn matmul_loop_gate_and_backend() {
        // Default 3M k·n threshold (env KTIR_GEMM_GPU_MIN_KN unset in test).
        assert_eq!(matmul_min_kn(), GEMM_GPU_MIN_KN);

        // OFFLOAD GATE (full-M offload here vs interpreter scf.for fallback):
        //   PLAIN m == 1 (decode): offload only if k·n clears the gate.
        assert!(!matmul_loop_offload(1, 576, 576, false)); // small decode GEMM -> interpreter
        assert!(matmul_loop_offload(1, 576, 49152, false)); // decode lm_head (28M) -> offload
        //   m > 1 (prefill): ALWAYS offloaded full-M, regardless of k·n — never the
        //   row-0 interpreter loop. THIS is what the AMX change relies on.
        assert!(matmul_loop_offload(8, 576, 576, false)); // small prefill GEMM still offloads
        assert!(matmul_loop_offload(32, 2048, 8192, false));
        //   transpose-B: ALWAYS offloaded, even tiny m=1 — the interpreter's
        //   per-K-step [n,k] B panel is a slow strided gather, so collapse it to one
        //   AMX sgemm_bt (B read contiguous [n,k]). This is the decode 0.28→0.68 fix.
        assert!(matmul_loop_offload(1, 576, 576, true)); // small decode transpose-B -> offload
        assert!(matmul_loop_offload(1, 2048, 512, true)); // llama decode q/k/v (1M) -> offload

        // BACKEND (of the offloaded GEMMs): NAX iff k·n >= gate, else AMX.
        // smollm2 layer GEMMs (k·n 0.33M..0.88M) -> AMX (the win at M=8).
        assert!(!matmul_loop_use_nax(576, 576)); // 0.33M
        assert!(!matmul_loop_use_nax(576, 1536)); // 0.88M
        assert!(!matmul_loop_use_nax(1536, 576)); // 0.88M down_proj
        // llama layer GEMMs (4.2M..16.8M) + GQA split + both lm_heads.
        assert!(matmul_loop_use_nax(2048, 2048)); // 4.2M -> NAX
        assert!(matmul_loop_use_nax(2048, 8192)); // 16.8M -> NAX
        assert!(!matmul_loop_use_nax(2048, 512)); // 1.05M GQA k/v -> AMX
        assert!(matmul_loop_use_nax(576, 49152)); // smollm2 lm_head 28M -> NAX
        assert!(matmul_loop_use_nax(2048, 128256)); // llama lm_head -> NAX
    }

    #[test]
    fn reports_device_tier_on_real_gpu() {
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("no Metal device — skipping live tier check");
            return;
        };
        let name = device.name().to_string();
        let cap = device_matmul_tier(&name);
        eprintln!(
            "device {name:?}: capability tier = {cap:?}, using = {:?}",
            effective_matmul_tier(&name)
        );
        // This machine is an Apple GPU, so it must be at least the simdgroup tier.
        assert!(
            cap >= MatmulTier::Simdgroup,
            "expected an Apple GPU, got {name:?}"
        );
    }
}
