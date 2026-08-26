// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Latency model — port of `ktir_emulator/latency.py`. This slice locks the
//! `LatencyCategory` enum (the dispatch table pairs every op with one) and the
//! `HardwareConfig` cost parameters with their computed roofline properties.
//! `LatencyTracker` / `LatencyReport` (per-op accounting + bottleneck
//! classification) are an implement-phase fill against these locked types.

/// Cost class assigned to each op at registration. Mirrors the `LatencyCategory`
/// StrEnum — all seven members.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyCategory {
    Zero,
    Memory,
    ComputeFloat,
    ComputeTranscendental,
    ComputeInt,
    ComputeMatmul,
    Comm,
}

impl LatencyCategory {
    /// The `StrEnum` string value, for parity with the Python registry.
    pub fn as_str(self) -> &'static str {
        match self {
            LatencyCategory::Zero => "zero",
            LatencyCategory::Memory => "memory",
            LatencyCategory::ComputeFloat => "compute_float",
            LatencyCategory::ComputeTranscendental => "compute_transcendental",
            LatencyCategory::ComputeInt => "compute_int",
            LatencyCategory::ComputeMatmul => "compute_matmul",
            LatencyCategory::Comm => "comm",
        }
    }
}

/// Hardware cost parameters. Mirrors `HardwareConfig` defaults exactly.
#[derive(Clone, Copy, Debug)]
pub struct HardwareConfig {
    pub num_cores: usize,
    pub clock_ghz: f64,
    pub hbm_bandwidth_tb_s: f64,
    pub ring_bandwidth_tb_s: f64,
    pub simd_elements_per_cycle: u32,
    pub systolic_flops_per_cycle: u64,
    pub transcendental_penalty: u32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        HardwareConfig {
            num_cores: 32,
            clock_ghz: 1.0,
            hbm_bandwidth_tb_s: 1.0,
            ring_bandwidth_tb_s: 4.0,
            simd_elements_per_cycle: 64,
            systolic_flops_per_cycle: 2 * 64 * 64 * 64, // 524288
            transcendental_penalty: 4,
        }
    }
}

impl HardwareConfig {
    /// `(hbm_bandwidth_tb_s * 1e12) / (clock_ghz * 1e9) / num_cores`.
    pub fn hbm_bytes_per_cycle_per_core(&self) -> f64 {
        (self.hbm_bandwidth_tb_s * 1e12) / (self.clock_ghz * 1e9) / self.num_cores as f64
    }

    /// `ring_bandwidth_tb_s * 1e12 / (clock_ghz * 1e9)`.
    pub fn ring_bytes_per_cycle(&self) -> f64 {
        self.ring_bandwidth_tb_s * 1e12 / (self.clock_ghz * 1e9)
    }
}

// ---------------------------------------------------------------------------
// Per-core latency counters
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;

use crate::ir::Value;
use crate::memory::STICK_BYTES;
use crate::memref::MemorySpace;
use crate::tile::Tile;

/// Coarse bucket a recorded op's cycles land in. Mirrors the Python
/// `_TraceEntry.category` / `CoreLatencyCounters.record` string ("compute",
/// "memory", "comm", "zero").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostBucket {
    Zero,
    Compute,
    Memory,
    Comm,
}

impl CostBucket {
    pub fn as_str(self) -> &'static str {
        match self {
            CostBucket::Zero => "zero",
            CostBucket::Compute => "compute",
            CostBucket::Memory => "memory",
            CostBucket::Comm => "comm",
        }
    }
}

/// A single operation trace entry. Mirrors `_TraceEntry`. Only populated when
/// tracing is enabled on the [`LatencyTracker`].
#[derive(Clone, Debug, PartialEq)]
pub struct TraceEntry {
    pub op_type: String,
    pub cycles: f64,
    pub bucket: CostBucket,
}

/// Per-core cycle counters. Mirrors `CoreLatencyCounters`.
#[derive(Clone, Debug, Default)]
pub struct CoreLatencyCounters {
    pub compute_cycles: f64,
    pub memory_cycles: f64,
    pub comm_cycles: f64,
    pub total_flops: f64,
    pub total_bytes: u64,
    /// `Some` when tracing is enabled (mirrors the `Optional[List]` field).
    pub trace: Option<Vec<TraceEntry>>,
}

impl CoreLatencyCounters {
    fn new(trace: bool) -> Self {
        CoreLatencyCounters {
            trace: if trace { Some(Vec::new()) } else { None },
            ..Default::default()
        }
    }

    /// `compute_cycles + memory_cycles + comm_cycles`.
    pub fn total_cycles(&self) -> f64 {
        self.compute_cycles + self.memory_cycles + self.comm_cycles
    }

    /// Accumulate one op's estimate. Mirrors `CoreLatencyCounters.record`.
    fn record(&mut self, bucket: CostBucket, cycles: f64, op_type: &str, flops: f64, nbytes: u64) {
        match bucket {
            CostBucket::Compute => self.compute_cycles += cycles,
            CostBucket::Memory => self.memory_cycles += cycles,
            CostBucket::Comm => self.comm_cycles += cycles,
            CostBucket::Zero => {}
        }
        self.total_flops += flops;
        self.total_bytes += nbytes;
        if let Some(trace) = &mut self.trace {
            trace.push(TraceEntry {
                op_type: op_type.to_string(),
                cycles,
                bucket,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Latency tracker
// ---------------------------------------------------------------------------

/// One op's cost estimate: `(bucket, cycles, flops, nbytes)`. Mirrors the
/// 4-tuple returned by `LatencyTracker._estimate`.
struct Estimate {
    bucket: CostBucket,
    cycles: f64,
    flops: f64,
    nbytes: u64,
}

/// Records per-operation cycle costs across all cores. Port of `LatencyTracker`.
///
/// Counters are created lazily on first `record_op` for each `core_id`, so the
/// tracker does not need to know the grid shape up front. The interpreter wires
/// this in optionally (an `Option<&mut LatencyTracker>` threaded through
/// `execute_op`): when no tracker is supplied, there is zero overhead and every
/// existing call site is unaffected.
pub struct LatencyTracker {
    pub config: HardwareConfig,
    trace: bool,
    counters: BTreeMap<usize, CoreLatencyCounters>,
}

impl LatencyTracker {
    pub fn new(config: HardwareConfig) -> Self {
        LatencyTracker {
            config,
            trace: false,
            counters: BTreeMap::new(),
        }
    }

    /// Enable per-op tracing (populates `CoreLatencyCounters.trace`).
    pub fn with_trace(config: HardwareConfig, trace: bool) -> Self {
        LatencyTracker {
            config,
            trace,
            counters: BTreeMap::new(),
        }
    }

    /// Clear all accumulated counters. Mirrors `reset`.
    pub fn reset(&mut self) {
        self.counters.clear();
    }

    /// Estimate and record the cycle cost of one operation.
    ///
    /// `category` is the dispatch table's latency class for `op_type`; `result`
    /// is the value the handler produced (`None` for result-less ops); and
    /// `operands` are the resolved operand values (`None` for any that could not
    /// be resolved). Mirrors `LatencyTracker.record_op`.
    pub fn record_op(
        &mut self,
        core_id: usize,
        op_type: &str,
        category: LatencyCategory,
        result: &Option<Value>,
        operands: &[Option<Value>],
    ) {
        let est = self.estimate(op_type, category, result, operands);
        let trace = self.trace;
        self.counters
            .entry(core_id)
            .or_insert_with(|| CoreLatencyCounters::new(trace))
            .record(est.bucket, est.cycles, op_type, est.flops, est.nbytes);
    }

    /// Build a [`LatencyReport`] from accumulated counters. Mirrors `report`.
    pub fn report(&self) -> LatencyReport {
        LatencyReport {
            config: self.config,
            counters: self.counters.clone(),
        }
    }

    /// Direct read-only access to the per-core counters (for tests / tooling).
    pub fn counters(&self) -> &BTreeMap<usize, CoreLatencyCounters> {
        &self.counters
    }

    // -- private helpers -----------------------------------------------------

    /// Return the estimate for a single op. Mirrors `LatencyTracker._estimate`.
    fn estimate(
        &self,
        op_type: &str,
        category: LatencyCategory,
        result: &Option<Value>,
        operands: &[Option<Value>],
    ) -> Estimate {
        let cfg = &self.config;
        match category {
            // Metadata-only ops (tensor.splat, scf.yield, …): no cycles.
            LatencyCategory::Zero => Estimate {
                bucket: CostBucket::Zero,
                cycles: 0.0,
                flops: 0.0,
                nbytes: 0,
            },

            LatencyCategory::Memory => {
                // LX (on-chip scratchpad) ops are free — the tile already lives
                // in LX as an SSA value, so no DMA occurs.
                if memory_space(operands) == SpaceKind::Lx {
                    return Estimate {
                        bucket: CostBucket::Memory,
                        cycles: 0.0,
                        flops: 0.0,
                        nbytes: 0,
                    };
                }
                // HBM load/store: cycles = bytes / per-core bandwidth.
                let nbytes = data_size(result, operands);
                let bw = cfg.hbm_bytes_per_cycle_per_core();
                let cycles = if bw > 0.0 { nbytes as f64 / bw } else { 0.0 };
                Estimate {
                    bucket: CostBucket::Memory,
                    cycles,
                    flops: 0.0,
                    nbytes,
                }
            }

            LatencyCategory::ComputeMatmul => {
                // Systolic matmul: 2*M*N*K FLOPs. No HBM traffic — operand tiles
                // are already in LX.
                let (m, n, k) = matmul_dims(operands);
                let flops = 2.0 * m as f64 * n as f64 * k as f64;
                let cycles = flops / cfg.systolic_flops_per_cycle as f64;
                Estimate {
                    bucket: CostBucket::Compute,
                    cycles,
                    flops,
                    nbytes: 0,
                }
            }

            LatencyCategory::ComputeTranscendental => {
                // Transcendentals: 1 FLOP per element, with a penalty multiplier
                // modelling the higher *latency* of the function unit — it does
                // not increase the FLOP count.
                let n_elems = num_elements(result, operands);
                let cycles = (n_elems as f64 / cfg.simd_elements_per_cycle as f64)
                    * cfg.transcendental_penalty as f64;
                Estimate {
                    bucket: CostBucket::Compute,
                    cycles,
                    flops: n_elems as f64,
                    nbytes: 0,
                }
            }

            LatencyCategory::ComputeFloat => {
                // Elementwise float: 1 FLOP per element, one SIMD-width per cycle.
                let n_elems = num_elements(result, operands);
                let cycles = n_elems as f64 / cfg.simd_elements_per_cycle as f64;
                Estimate {
                    bucket: CostBucket::Compute,
                    cycles,
                    flops: n_elems as f64,
                    nbytes: 0,
                }
            }

            LatencyCategory::ComputeInt => {
                // Integer ops: 1 FLOP per element. Scalar index arithmetic
                // (n_elems <= 1) is resolved at compile time — free.
                let n_elems = num_elements(result, operands);
                if n_elems <= 1 {
                    return Estimate {
                        bucket: CostBucket::Compute,
                        cycles: 0.0,
                        flops: 0.0,
                        nbytes: 0,
                    };
                }
                let cycles = n_elems as f64 / cfg.simd_elements_per_cycle as f64;
                Estimate {
                    bucket: CostBucket::Compute,
                    cycles,
                    flops: n_elems as f64,
                    nbytes: 0,
                }
            }

            LatencyCategory::Comm => {
                // Ring communication: bytes over ring bandwidth. No FLOPs.
                // Reduce requires ceil(log2(num_cores)) rounds.
                let nbytes = comm_size(operands);
                let bw = cfg.ring_bytes_per_cycle();
                let mut cycles = if bw > 0.0 { nbytes as f64 / bw } else { 0.0 };
                if op_type == "ktdp.reduce" {
                    let rounds = (cfg.num_cores as f64).log2().ceil().max(1.0);
                    cycles *= rounds;
                }
                Estimate {
                    bucket: CostBucket::Comm,
                    cycles,
                    flops: 0.0,
                    nbytes,
                }
            }
        }
    }
}

/// Memory-space classification of a memory op's operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpaceKind {
    Hbm,
    Lx,
}

fn space_of(space: MemorySpace) -> SpaceKind {
    match space {
        MemorySpace::Hbm => SpaceKind::Hbm,
        MemorySpace::Lx { .. } => SpaceKind::Lx,
    }
}

/// Return the memory space of the memory op's target. Mirrors `_memory_space`.
///
/// Returns `Hbm` when no view operand is found (e.g. pointer-based access that
/// always reads HBM).
fn memory_space(operands: &[Option<Value>]) -> SpaceKind {
    for v in operands.iter().flatten() {
        match v {
            Value::MemRef(m) => return space_of(m.space),
            Value::DistMemRef(d) => {
                if let Some(p) = d.partitions.first() {
                    return space_of(p.space);
                }
            }
            Value::TileRef(t) => return space_of(t.memref.space),
            Value::DistTileRef(d) => {
                if let Some(p) = d.partitions.first() {
                    return space_of(p.memref.space);
                }
            }
            Value::AccessTile(a) => {
                let mr = match &a.parent_ref {
                    crate::memref::ParentRef::Tile(t) => &t.memref,
                    crate::memref::ParentRef::Dist(d) => match d.partitions.first() {
                        Some(t) => &t.memref,
                        None => continue,
                    },
                };
                return space_of(mr.space);
            }
            Value::IndirectAccessTile(iat) => {
                let all_lx = space_of(iat.parent_ref.space) == SpaceKind::Lx
                    && iat
                        .index_views
                        .iter()
                        .all(|iv| space_of(iv.space) == SpaceKind::Lx);
                return if all_lx {
                    SpaceKind::Lx
                } else {
                    SpaceKind::Hbm
                };
            }
            _ => {}
        }
    }
    SpaceKind::Hbm
}

/// Estimate bytes transferred by a memory operation, charged at HBM stick
/// granularity (`unique_sticks * STICK_BYTES`). Mirrors `_data_size`.
///
/// * Loads stamp `unique_sticks` (and `index_unique_sticks` for IATs) on the
///   result [`Tile`]; read off the result here.
/// * Stores have no result Tile — the handler instead propagates the int
///   `unique_sticks` return as the op result (here a [`Value::Index`]). For an
///   indirect store the int already aggregates parent + idx sticks.
fn data_size(result: &Option<Value>, operands: &[Option<Value>]) -> u64 {
    // Store sideband: the handler propagated the int unique_sticks as the result.
    if let Some(Value::Index(sticks)) = result {
        return (*sticks).max(0) as u64 * STICK_BYTES as u64;
    }

    let mut total: u64 = 0;
    if let Some(Value::Tile(t)) = result {
        // On the HBM path a load must populate unique_sticks. Defensive: treat
        // a missing count as 0 (the Python path raises; the optional latency
        // hook must never abort execution).
        total += t.unique_sticks.unwrap_or(0) as u64 * STICK_BYTES as u64;
        if let Some(idx) = t.index_unique_sticks {
            total += idx as u64 * STICK_BYTES as u64;
        }
    }
    let _ = operands; // operand sticks already aggregated into the result Tile.
    total
}

/// Count number of data elements processed. Mirrors `_num_elements`.
fn num_elements(result: &Option<Value>, operands: &[Option<Value>]) -> usize {
    if let Some(Value::Tile(t)) = result {
        return t.shape.iter().product();
    }
    for v in operands.iter().flatten() {
        if let Value::Tile(t) = v {
            return t.shape.iter().product();
        }
    }
    1
}

/// Extract `(M, N, K)` from matmul operands. Mirrors `_matmul_dims`.
/// `a` is `(M, K)`, `b` is `(K, N)`.
fn matmul_dims(operands: &[Option<Value>]) -> (usize, usize, usize) {
    let tiles: Vec<&Tile> = operands
        .iter()
        .flatten()
        .filter_map(|v| {
            if let Value::Tile(t) = v {
                Some(t)
            } else {
                None
            }
        })
        .collect();
    if tiles.len() >= 2 {
        let a = tiles[0];
        let b = tiles[1];
        let m = if a.shape.len() >= 2 { a.shape[0] } else { 1 };
        let k = if a.shape.len() >= 2 {
            a.shape[1]
        } else {
            *a.shape.first().unwrap_or(&1)
        };
        let n = if b.shape.len() >= 2 { b.shape[1] } else { 1 };
        return (m, n, k);
    }
    (1, 1, 1)
}

/// Estimate bytes transferred by a communication op. Mirrors `_comm_size`
/// (`tile.data.nbytes`).
fn comm_size(operands: &[Option<Value>]) -> u64 {
    for v in operands.iter().flatten() {
        if let Value::Tile(t) = v {
            return t.size_bytes() as u64;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Latency report
// ---------------------------------------------------------------------------

/// A per-core breakdown row (mirrors one entry of `per_core_summary`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreSummary {
    pub core_id: usize,
    pub compute_cycles: f64,
    pub memory_cycles: f64,
    pub comm_cycles: f64,
    pub total_cycles: f64,
}

/// Roofline metrics for the critical-path core (mirrors the `roofline` dict).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roofline {
    pub arithmetic_intensity: f64,
    pub achieved_gflops: f64,
    pub peak_gflops: f64,
    pub peak_bw_gb_s: f64,
    pub ridge_point: f64,
    pub ceiling_gflops: f64,
    pub efficiency: f64,
}

/// Summary of estimated execution latency. Port of `LatencyReport`.
#[derive(Clone, Debug)]
pub struct LatencyReport {
    pub config: HardwareConfig,
    pub counters: BTreeMap<usize, CoreLatencyCounters>,
}

impl LatencyReport {
    /// The critical-path core (max total cycles). Ties resolve to the first
    /// (lowest core_id) — `BTreeMap` iteration is ordered.
    fn critical(&self) -> Option<&CoreLatencyCounters> {
        self.counters.values().reduce(|a, b| {
            if b.total_cycles() > a.total_cycles() {
                b
            } else {
                a
            }
        })
    }

    /// Kernel latency = max total cycles across all cores. Mirrors `kernel_cycles`.
    pub fn kernel_cycles(&self) -> f64 {
        self.critical().map(|c| c.total_cycles()).unwrap_or(0.0)
    }

    /// Kernel time in microseconds (`cycles / clock_ghz / 1e3`).
    pub fn kernel_time_us(&self) -> f64 {
        self.kernel_cycles() / (self.config.clock_ghz * 1e3)
    }

    /// Bottleneck category on the critical-path core. Mirrors `bottleneck`.
    /// One of "compute" / "memory" / "comm" / "none".
    pub fn bottleneck(&self) -> &'static str {
        match self.critical() {
            None => "none",
            Some(c) => {
                // ties keep the earliest (compute > memory > comm), matching
                // Python's `max(dict, key=...)` first-key-wins behaviour.
                let cats = [
                    ("compute", c.compute_cycles),
                    ("memory", c.memory_cycles),
                    ("comm", c.comm_cycles),
                ];
                let mut best = cats[0];
                for &(name, v) in &cats[1..] {
                    if v > best.1 {
                        best = (name, v);
                    }
                }
                best.0
            }
        }
    }

    /// Per-core breakdown, ordered by core_id. Mirrors `per_core_summary`.
    pub fn per_core_summary(&self) -> Vec<CoreSummary> {
        self.counters
            .iter()
            .map(|(&core_id, c)| CoreSummary {
                core_id,
                compute_cycles: c.compute_cycles,
                memory_cycles: c.memory_cycles,
                comm_cycles: c.comm_cycles,
                total_cycles: c.total_cycles(),
            })
            .collect()
    }

    /// Roofline metrics for the critical-path core. Mirrors `roofline`.
    /// Returns `None` when there are no counters (Python returns `{}`).
    pub fn roofline(&self) -> Option<Roofline> {
        let critical = self.critical()?;
        let clock = self.config.clock_ghz * 1e9;
        let peak_flops = self.config.simd_elements_per_cycle as f64 * clock;
        let peak_bw = self.config.hbm_bytes_per_cycle_per_core() * clock;
        let ridge_point = peak_flops / peak_bw;

        let elapsed_s = critical.total_cycles() / clock;
        let achieved_flops = if elapsed_s > 0.0 {
            critical.total_flops / elapsed_s
        } else {
            0.0
        };
        let ai = if critical.total_bytes > 0 {
            critical.total_flops / critical.total_bytes as f64
        } else {
            f64::INFINITY
        };
        let ceiling = peak_flops.min(peak_bw * ai);
        Some(Roofline {
            arithmetic_intensity: ai,
            achieved_gflops: achieved_flops / 1e9,
            peak_gflops: peak_flops / 1e9,
            peak_bw_gb_s: peak_bw / 1e9,
            ridge_point,
            ceiling_gflops: ceiling / 1e9,
            efficiency: if ceiling > 0.0 {
                achieved_flops / ceiling
            } else {
                0.0
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::DType;
    use crate::memref::{MemRef, TileRef};
    use crate::tile::Tile;

    fn hbm_memref(shape: Vec<usize>) -> MemRef {
        MemRef {
            base_ptr: 0,
            shape,
            strides: vec![1],
            space: MemorySpace::Hbm,
            dtype: DType::F16,
            coordinate_set: None,
        }
    }

    fn hbm_tileref() -> TileRef {
        hbm_memref(vec![8]).to_tile_ref()
    }

    fn lx_tileref() -> TileRef {
        MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F16,
            coordinate_set: None,
        }
        .to_tile_ref()
    }

    fn load_result(unique_sticks: usize, idx_sticks: Option<usize>) -> Option<Value> {
        let mut t = Tile::compute(vec![0.0; 8], DType::F16, vec![8]);
        t.unique_sticks = Some(unique_sticks);
        t.index_unique_sticks = idx_sticks;
        Some(Value::Tile(t))
    }

    #[test]
    fn config_defaults_and_roofline() {
        let c = HardwareConfig::default();
        assert_eq!(c.systolic_flops_per_cycle, 524288);
        // 1e12 / 1e9 / 32 = 1000/32
        assert!((c.hbm_bytes_per_cycle_per_core() - 1000.0 / 32.0).abs() < 1e-9);
        assert!((c.ring_bytes_per_cycle() - 4000.0).abs() < 1e-9);
    }

    #[test]
    fn category_strings_match_python() {
        assert_eq!(LatencyCategory::ComputeMatmul.as_str(), "compute_matmul");
        assert_eq!(LatencyCategory::Comm.as_str(), "comm");
    }

    // -- tracker / cost-formula tests ---------------------------------------

    #[test]
    fn zero_category_costs_nothing() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        t.record_op(0, "scf.yield", LatencyCategory::Zero, &None, &[]);
        let c = &t.counters()[&0];
        assert_eq!(c.total_cycles(), 0.0);
        assert_eq!(c.total_flops, 0.0);
        assert_eq!(c.total_bytes, 0);
    }

    #[test]
    fn hbm_load_charges_stick_traffic() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        // 4 sticks * 128 B = 512 B; bw = 1000/32 B/cycle.
        let operands = [Some(Value::TileRef(hbm_tileref()))];
        t.record_op(
            0,
            "ktdp.load",
            LatencyCategory::Memory,
            &load_result(4, None),
            &operands,
        );
        let c = &t.counters()[&0];
        let expect = 512.0 / (1000.0 / 32.0);
        assert!((c.memory_cycles - expect).abs() < 1e-9);
        assert_eq!(c.total_bytes, 512);
        assert_eq!(c.total_flops, 0.0);
    }

    #[test]
    fn lx_memory_op_is_free() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let operands = [Some(Value::TileRef(lx_tileref()))];
        t.record_op(
            0,
            "ktdp.load",
            LatencyCategory::Memory,
            &load_result(99, None),
            &operands,
        );
        let c = &t.counters()[&0];
        assert_eq!(c.memory_cycles, 0.0);
        assert_eq!(c.total_bytes, 0);
    }

    #[test]
    fn store_sideband_int_charges_sticks() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        // store handler returns unique_sticks as Value::Index; no view operand
        // means default HBM space.
        t.record_op(
            0,
            "ktdp.store",
            LatencyCategory::Memory,
            &Some(Value::Index(3)),
            &[],
        );
        let c = &t.counters()[&0];
        assert_eq!(c.total_bytes, 3 * 128);
        let expect = (3.0 * 128.0) / (1000.0 / 32.0);
        assert!((c.memory_cycles - expect).abs() < 1e-9);
    }

    #[test]
    fn indirect_load_adds_index_sticks() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let operands = [Some(Value::TileRef(hbm_tileref()))];
        // 2 data sticks + 5 idx sticks = 7 sticks * 128 B.
        t.record_op(
            0,
            "ktdp.load",
            LatencyCategory::Memory,
            &load_result(2, Some(5)),
            &operands,
        );
        assert_eq!(t.counters()[&0].total_bytes, 7 * 128);
    }

    #[test]
    fn compute_float_one_flop_per_elem() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let res = Some(Value::Tile(Tile::compute(
            vec![0.0; 128],
            DType::F32,
            vec![128],
        )));
        t.record_op(0, "arith.addf", LatencyCategory::ComputeFloat, &res, &[]);
        let c = &t.counters()[&0];
        assert_eq!(c.total_flops, 128.0);
        // 128 elems / 64 simd = 2 cycles.
        assert!((c.compute_cycles - 2.0).abs() < 1e-9);
    }

    #[test]
    fn transcendental_applies_penalty() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let res = Some(Value::Tile(Tile::compute(
            vec![0.0; 64],
            DType::F32,
            vec![64],
        )));
        t.record_op(
            0,
            "math.exp",
            LatencyCategory::ComputeTranscendental,
            &res,
            &[],
        );
        let c = &t.counters()[&0];
        // (64/64) * penalty(4) = 4 cycles; flops = 64 (penalty doesn't add flops).
        assert!((c.compute_cycles - 4.0).abs() < 1e-9);
        assert_eq!(c.total_flops, 64.0);
    }

    #[test]
    fn scalar_int_is_free() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        // no tile operands/result => n_elems == 1 => free.
        t.record_op(
            0,
            "arith.addi",
            LatencyCategory::ComputeInt,
            &Some(Value::Index(5)),
            &[],
        );
        let c = &t.counters()[&0];
        assert_eq!(c.compute_cycles, 0.0);
        assert_eq!(c.total_flops, 0.0);
    }

    #[test]
    fn vector_int_charges_elements() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let res = Some(Value::Tile(Tile::compute(
            vec![0.0; 128],
            DType::I32,
            vec![128],
        )));
        t.record_op(0, "arith.addi", LatencyCategory::ComputeInt, &res, &[]);
        let c = &t.counters()[&0];
        assert!((c.compute_cycles - 2.0).abs() < 1e-9);
        assert_eq!(c.total_flops, 128.0);
    }

    #[test]
    fn matmul_flops_two_mnk() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let a = Value::Tile(Tile::compute(vec![0.0; 64 * 64], DType::F16, vec![64, 64]));
        let b = Value::Tile(Tile::compute(vec![0.0; 64 * 64], DType::F16, vec![64, 64]));
        let operands = [Some(a), Some(b)];
        t.record_op(
            0,
            "linalg.matmul",
            LatencyCategory::ComputeMatmul,
            &None,
            &operands,
        );
        let c = &t.counters()[&0];
        let flops = 2.0 * 64.0 * 64.0 * 64.0;
        assert_eq!(c.total_flops, flops);
        // flops / systolic(524288) = exactly 1 cycle for a 64^3 matmul.
        assert!((c.compute_cycles - 1.0).abs() < 1e-9);
    }

    #[test]
    fn comm_charges_ring_bytes() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        // 64 f16 elems = 128 bytes; ring bw = 4000 B/cycle.
        let tile = Value::Tile(Tile::compute(vec![0.0; 64], DType::F16, vec![64]));
        t.record_op(
            0,
            "ktdp.allgather",
            LatencyCategory::Comm,
            &None,
            &[Some(tile)],
        );
        let c = &t.counters()[&0];
        assert_eq!(c.total_bytes, 128);
        assert!((c.comm_cycles - 128.0 / 4000.0).abs() < 1e-9);
    }

    #[test]
    fn reduce_multiplies_by_log2_rounds() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let tile = Value::Tile(Tile::compute(vec![0.0; 64], DType::F16, vec![64]));
        t.record_op(
            0,
            "ktdp.reduce",
            LatencyCategory::Comm,
            &None,
            &[Some(tile)],
        );
        let c = &t.counters()[&0];
        // ceil(log2(32)) = 5 rounds.
        let base = 128.0 / 4000.0;
        assert!((c.comm_cycles - base * 5.0).abs() < 1e-9);
    }

    #[test]
    fn report_kernel_cycles_is_max_and_bottleneck() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        // core 0: heavy compute. core 1: light.
        let big = Some(Value::Tile(Tile::compute(
            vec![0.0; 64 * 100],
            DType::F32,
            vec![6400],
        )));
        t.record_op(0, "arith.addf", LatencyCategory::ComputeFloat, &big, &[]);
        let small = Some(Value::Tile(Tile::compute(
            vec![0.0; 64],
            DType::F32,
            vec![64],
        )));
        t.record_op(1, "arith.addf", LatencyCategory::ComputeFloat, &small, &[]);
        let rep = t.report();
        assert!((rep.kernel_cycles() - 100.0).abs() < 1e-9);
        assert_eq!(rep.bottleneck(), "compute");
        assert_eq!(rep.per_core_summary().len(), 2);
    }

    #[test]
    fn memory_bound_kernel_roofline_classifies() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        // pure HBM load: bytes > 0, flops == 0 => AI 0 => memory bound.
        let operands = [Some(Value::TileRef(hbm_tileref()))];
        t.record_op(
            0,
            "ktdp.load",
            LatencyCategory::Memory,
            &load_result(8, None),
            &operands,
        );
        let rep = t.report();
        assert_eq!(rep.bottleneck(), "memory");
        let rf = rep.roofline().unwrap();
        assert_eq!(rf.arithmetic_intensity, 0.0);
        assert!(rf.peak_gflops > 0.0);
    }

    #[test]
    fn reset_clears_counters() {
        let mut t = LatencyTracker::new(HardwareConfig::default());
        let res = Some(Value::Tile(Tile::compute(
            vec![0.0; 64],
            DType::F32,
            vec![64],
        )));
        t.record_op(0, "arith.addf", LatencyCategory::ComputeFloat, &res, &[]);
        assert!(!t.counters().is_empty());
        t.reset();
        assert!(t.counters().is_empty());
        assert_eq!(t.report().kernel_cycles(), 0.0);
    }

    #[test]
    fn trace_records_per_op_entries() {
        let mut t = LatencyTracker::with_trace(HardwareConfig::default(), true);
        let res = Some(Value::Tile(Tile::compute(
            vec![0.0; 64],
            DType::F32,
            vec![64],
        )));
        t.record_op(0, "arith.addf", LatencyCategory::ComputeFloat, &res, &[]);
        let trace = t.counters()[&0].trace.as_ref().unwrap();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].op_type, "arith.addf");
        assert_eq!(trace[0].bucket, CostBucket::Compute);
    }
}
