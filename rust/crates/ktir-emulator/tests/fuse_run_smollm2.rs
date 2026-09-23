// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! REAL-MODEL fuse-then-run e2e (fusion increment 2): take the SmolLM2-135M KTIR
//! bundle scratchy emits, build one `ProgramSpec` from `manifest.json`, run
//! `ktir_optimizer::fusion::fuse_program` to collapse all 452 nodes into a single
//! function whose forwardable HBM intermediates become SSA / `tensor.extract_slice`,
//! then execute that fused function through the SAME interpreter and compare to
//! golden. This is the pressure test for increment 2: the model's tiled edges are
//! `construct_access_tile %view[%c0, %k7]` loads INSIDE an `scf.for` K-loop, so it
//! exercises the region-aware analysis, the nested extract_slice rewrite, and the
//! scf.for attribute renaming all at once.
//!
//! The bundle is machine-specific and not in the repo, so the test SKIPS when
//! absent. `--ignored` because it runs a whole model.
#![cfg(feature = "optimizer")] // this whole suite drives the optimizer/fusion path

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function, execute_function_outputs};
use ktir_emulator::ir::IRModule;
use ktir_emulator::parser::parse_module;
use ktir_optimizer::fusion::{
    Binding, NodeSpec, ProgramSpec, Segment, fuse_program, plan_segments,
};
// Only the cfg(metal) budgeted-split test uses this.
#[cfg(metal)]
use ktir_optimizer::fusion::plan_segments_budgeted;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn bundle_dir() -> Option<PathBuf> {
    bundle_dir_named("smollm2-135m")
}

fn bundle_dir_named(model: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home)
        .join(".cache/cudaforge/ktir")
        .join(model);
    dir.join("manifest.json").is_file().then_some(dir)
}

/// A whole bundle fused into one function, plus the metadata to run it.
struct Fused {
    func: ktir_emulator::ir::IRFunction,
    /// tensor id -> (rows, cols, is_source)
    shape: HashMap<u64, (usize, usize, bool)>,
    result_id: u64,
    mask_id: Option<u64>,
    n_nodes: usize,
}

/// A whole bundle parsed into a module + ProgramSpec, with the tensor metadata to
/// marshal it. The shared front-end of `fuse_bundle` (whole-program fuse) and
/// `plan_bundle` (partial fusion: fused segments + native attention nodes).
struct Bundle {
    module: IRModule,
    spec: ProgramSpec,
    /// tensor id -> (rows, cols, is_source)
    shape: HashMap<u64, (usize, usize, bool)>,
    result_id: u64,
    mask_id: Option<u64>,
    n_nodes: usize,
}

/// Load a bundle's manifest + per-node MLIR into a module + ProgramSpec.
fn load_bundle(dir: &std::path::Path) -> Bundle {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let mut shape: HashMap<u64, (usize, usize, bool)> = HashMap::new();
    let mut sources: HashSet<u64> = HashSet::new();
    for t in manifest["tensors"].as_array().unwrap() {
        let id = t["id"].as_u64().unwrap();
        let is_src = t["is_source"].as_bool().unwrap_or(false);
        shape.insert(
            id,
            (
                t["rows"].as_u64().unwrap() as usize,
                t["cols"].as_u64().unwrap() as usize,
                is_src,
            ),
        );
        if is_src {
            sources.insert(id);
        }
    }
    let result_id = manifest["result"].as_u64().unwrap();
    let mask_id = manifest["attn_mask"].as_u64();
    if let Some(m) = mask_id {
        sources.insert(m);
    }

    let mut module = IRModule::default();
    let mut nodes: Vec<NodeSpec> = Vec::new();
    for node in manifest["nodes"].as_array().unwrap() {
        let func = node["fn"].as_str().unwrap().to_string();
        let mlir = node["mlir"].as_str().unwrap();
        let src = std::fs::read_to_string(dir.join(mlir)).unwrap();
        let parsed = parse_module(&src).unwrap_or_else(|e| panic!("parse {mlir}: {e}"));
        for (_, f) in parsed.functions {
            module.add_function(f);
        }
        let bindings = node["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| Binding {
                arg: format!("%{}", a["name"].as_str().unwrap()),
                tensor: a["tensor"].as_u64().unwrap(),
                is_output: a["is_output"].as_bool().unwrap_or(false),
            })
            .collect();
        nodes.push(NodeSpec { func, bindings });
    }
    let n_nodes = nodes.len();
    let spec = ProgramSpec {
        nodes,
        sources,
        results: HashSet::from([result_id]),
    };
    Bundle {
        module,
        spec,
        shape,
        result_id,
        mask_id,
        n_nodes,
    }
}

/// Load a bundle's manifest + per-node MLIR, build the ProgramSpec, and fuse the
/// whole program into one function (decode or prefill — same path).
fn fuse_bundle(dir: &std::path::Path) -> Fused {
    let b = load_bundle(dir);
    let func = fuse_program(&b.module, &b.spec).expect("fuse bundle");
    Fused {
        func,
        shape: b.shape,
        result_id: b.result_id,
        mask_id: b.mask_id,
        n_nodes: b.n_nodes,
    }
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Parse `%t<id>_ptr` -> id. The fused function names every pointer arg this way.
fn tensor_id_of(arg: &str) -> u64 {
    arg.trim_start_matches('%')
        .trim_start_matches('t')
        .trim_end_matches("_ptr")
        .parse()
        .unwrap_or_else(|_| panic!("unexpected fused arg name {arg:?}"))
}

/// Run a bundle the PER-NODE way (each node executed with its OWN grid — the
/// proven-correct oracle that respects [8,1]/[9,1] SPMD), threading one host
/// buffer per tensor. Returns the result tensor. This is the apples-to-apples
/// reference for the fused path: if fused == per-node, the fused single-grid run
/// is correct regardless of golden's own generation noise.
fn run_per_node_result(dir: &std::path::Path) -> Vec<f32> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let mut shape: HashMap<u64, (usize, usize, bool)> = HashMap::new();
    for t in manifest["tensors"].as_array().unwrap() {
        let id = t["id"].as_u64().unwrap();
        shape.insert(
            id,
            (
                t["rows"].as_u64().unwrap() as usize,
                t["cols"].as_u64().unwrap() as usize,
                t["is_source"].as_bool().unwrap_or(false),
            ),
        );
    }
    let mut buf: HashMap<u64, Vec<f32>> = HashMap::new();
    for (&id, &(_, _, is_src)) in &shape {
        if is_src {
            buf.insert(id, read_f32(&dir.join(format!("t{id}.bin"))));
        }
    }
    if let Some(m) = manifest["attn_mask"].as_u64() {
        let (r, c, _) = shape[&m];
        buf.insert(m, vec![0.0f32; r * c]);
    }
    let mut cache: HashMap<String, IRModule> = HashMap::new();
    for node in manifest["nodes"].as_array().unwrap() {
        let func = node["fn"].as_str().unwrap();
        let mlir = node["mlir"].as_str().unwrap();
        let module = cache.entry(mlir.to_string()).or_insert_with(|| {
            parse_module(&std::fs::read_to_string(dir.join(mlir)).unwrap()).unwrap()
        });
        let mut arg_ids: Vec<(String, u64, bool)> = Vec::new();
        let mut args: Vec<(String, Arg)> = Vec::new();
        for a in node["args"].as_array().unwrap() {
            let name = a["name"].as_str().unwrap().to_string();
            let tid = a["tensor"].as_u64().unwrap();
            let is_out = a["is_output"].as_bool().unwrap_or(false);
            let (rows, cols, _) = shape[&tid];
            let data = if is_out {
                vec![0.0f32; rows * cols]
            } else {
                buf.get(&tid)
                    .cloned()
                    .unwrap_or_else(|| panic!("node input {tid} not produced"))
            };
            args.push((
                name.clone(),
                Arg::Tensor {
                    data,
                    shape: vec![rows, cols],
                    dtype: DType::F16,
                },
            ));
            arg_ids.push((name, tid, is_out));
        }
        let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
        let out = execute_function(module, func, &refs)
            .unwrap_or_else(|e| panic!("per-node {func}: {e}"));
        for (name, tid, is_out) in &arg_ids {
            if *is_out {
                buf.insert(*tid, out.get(name).expect("output").data.clone());
            }
        }
    }
    buf[&manifest["result"].as_u64().unwrap()].clone()
}

/// PARTIAL-FUSION run via the PRODUCTION API: build the program's source args
/// (weights/inputs from t{id}.bin, the attn mask zeroed) keyed by `t{id}`, then
/// call `ktir_emulator::segmented::execute_segmented` — the real serving path. It
/// plans the bundle into ordered segments (fused runs of non-attention nodes +
/// native attention nodes), threads one HBM host buffer per tensor id, runs each
/// fused segment at grid [1,1] (carrying the GPU offloads) and each head-parallel
/// attention node at its NATIVE grid (the proven-correct multi-core SPMD path),
/// and reads back the result. Returns the result tensor and the number of
/// (fused, native) segments (counted from `plan_segments` for the diagnostics).
// Used only by the cfg(metal) GPU-path tests; compiled (not gated) on non-metal
// so its callees stay live, but dead-code-allowed there.
#[cfg_attr(not(metal), allow(dead_code))]
fn run_segmented_result(dir: &std::path::Path) -> (Vec<f32>, usize, usize) {
    let b = load_bundle(dir);

    // Count the segments for the diagnostics the gates print (the production API
    // returns only the requested output tensors, not the plan shape).
    let segments = plan_segments(&b.module, &b.spec).expect("plan segments");
    let n_fused = segments
        .iter()
        .filter(|s| matches!(s, Segment::Fused(_)))
        .count();
    let n_native = segments
        .iter()
        .filter(|s| matches!(s, Segment::Native(_)))
        .count();

    // The program's SOURCES, keyed by the canonical `t{id}` name the production
    // API expects: true weights/inputs from t{id}.bin, and the attn mask as
    // all-zero (a `source` in the spec but not a file-backed weight — golden
    // uses a no-mask prefill mask).
    let mut owned: Vec<(String, Arg)> = Vec::new();
    for (&id, &(rows, cols, is_src)) in &b.shape {
        if is_src && Some(id) != b.mask_id {
            owned.push((
                format!("t{id}"),
                Arg::Tensor {
                    data: read_f32(&dir.join(format!("t{id}.bin"))),
                    shape: vec![rows, cols],
                    dtype: DType::F16,
                },
            ));
        }
    }
    if let Some(m) = b.mask_id {
        let (rows, cols, _) = b.shape[&m];
        owned.push((
            format!("t{m}"),
            Arg::Tensor {
                data: vec![0.0f32; rows * cols],
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

    let result_key = format!("t{}", b.result_id);
    let out =
        ktir_emulator::segmented::execute_segmented(&b.module, &b.spec, &args, &[&result_key])
            .expect("execute_segmented");
    let result = out.get(&result_key).expect("result produced").data.clone();
    (result, n_fused, n_native)
}

/// RESIDENT run via the PRODUCTION resident executor
/// (`ktir_emulator::resident::ResidentExecutor`): marshal every source weight into the
/// persistent HBM ONCE, then run one pass. The weights are NOT re-marshaled (the
/// whole point) — this is the apples-to-apples golden check for the resident path.
/// Returns the result tensor + (fused, native) segment counts.
#[cfg_attr(not(metal), allow(dead_code))]
fn run_resident_result(dir: &std::path::Path) -> (Vec<f32>, usize, usize) {
    let b = load_bundle(dir);
    let segments = plan_segments(&b.module, &b.spec).expect("plan segments");
    let n_fused = segments
        .iter()
        .filter(|s| matches!(s, Segment::Fused(_)))
        .count();
    let n_native = segments
        .iter()
        .filter(|s| matches!(s, Segment::Native(_)))
        .count();

    let mut owned: Vec<(String, Arg)> = Vec::new();
    for (&id, &(rows, cols, is_src)) in &b.shape {
        if is_src && Some(id) != b.mask_id {
            owned.push((
                format!("t{id}"),
                Arg::Tensor {
                    data: read_f32(&dir.join(format!("t{id}.bin"))),
                    shape: vec![rows, cols],
                    dtype: DType::F16,
                },
            ));
        }
    }
    if let Some(m) = b.mask_id {
        let (rows, cols, _) = b.shape[&m];
        owned.push((
            format!("t{m}"),
            Arg::Tensor {
                data: vec![0.0f32; rows * cols],
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

    let result_key = format!("t{}", b.result_id);
    let mut exec = ktir_emulator::resident::ResidentExecutor::new(b.module, &b.spec)
        .expect("build resident executor");
    exec.set_sources(&args).expect("marshal weights once");
    let out = exec.run(&[&result_key]).expect("resident run");
    let result = out.get(&result_key).expect("result produced").data.clone();
    (result, n_fused, n_native)
}

/// PERF: whole-model ms/pass through the PRODUCTION RESIDENT executor — weights
/// uploaded to the persistent HBM ONCE (across ALL passes, no per-pass / per-
/// segment re-marshal), only the pass-internal activations recomputed each pass.
/// This is the resident analogue of `segmented_mspass`; the gap between the two
/// is the per-pass weight-marshal cost the resident path eliminates.
///
/// BUNDLE / ITERS as in `segmented_mspass`. One warm-up pass excluded; median
/// over ITERS printed. Run with the GPU path ON and --test-threads=1.
#[cfg(metal)]
#[test]
#[ignore = "whole-model resident perf bench; needs the BUNDLE bundle. --ignored --nocapture"]
fn resident_mspass() {
    let bundle = std::env::var("BUNDLE").unwrap_or_else(|_| "smollm2-135m".to_string());
    let iters: u32 = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let Some(dir) = bundle_dir_named(&bundle) else {
        eprintln!("{bundle} bundle absent — skipping");
        return;
    };

    let b = load_bundle(&dir);
    let mut owned: Vec<(String, Arg)> = Vec::new();
    for (&id, &(rows, cols, is_src)) in &b.shape {
        if is_src && Some(id) != b.mask_id {
            owned.push((
                format!("t{id}"),
                Arg::Tensor {
                    data: read_f32(&dir.join(format!("t{id}.bin"))),
                    shape: vec![rows, cols],
                    dtype: DType::F16,
                },
            ));
        }
    }
    if let Some(m) = b.mask_id {
        let (rows, cols, _) = b.shape[&m];
        owned.push((
            format!("t{m}"),
            Arg::Tensor {
                data: vec![0.0f32; rows * cols],
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let result_key = format!("t{}", b.result_id);

    // Build the executor + upload weights ONCE — outside the timed loop. This is
    // the resident contract: the multi-pass loop re-uploads NOTHING.
    let mut exec = ktir_emulator::resident::ResidentExecutor::new(b.module, &b.spec)
        .expect("build resident executor");
    exec.set_sources(&args).expect("marshal weights once");

    // Warm up (pipeline compile, first-touch, weight-cache fill) — excluded.
    exec.run(&[&result_key]).expect("warmup");

    let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        exec.run(&[&result_key]).expect("timed resident run");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    eprintln!(
        "{bundle} e2e (Rust+Metal RESIDENT): {median:.1} ms/pass  ({} nodes, {iters} passes)",
        b.n_nodes
    );
}

/// PERF: whole-model ms/pass through the PRODUCTION segmented executor
/// (`ktir_emulator::segmented::execute_segmented`) — the apples-to-apples Rust+Metal
/// number for the Python per-node bench. Correct for BOTH decode and prefill
/// (head-parallel attention runs at its native grid; fused [1,1] segments carry
/// the K-loop GEMM + map-window + resident-weight-cache GPU offloads).
///
/// BUNDLE env selects the model (smollm2-135m / smollm2-135m-prefill /
/// llama-3.2-1b / llama-3.2-1b-prefill); ITERS env sets the timed pass count
/// (default 5). One warm-up pass is excluded; the median over ITERS is printed.
/// Run with the GPU path ON (do NOT set KTIR_NO_GPU_*) and --test-threads=1.
#[cfg(metal)]
#[test]
#[ignore = "whole-model perf bench; needs the BUNDLE bundle. --ignored --nocapture"]
fn segmented_mspass() {
    let bundle = std::env::var("BUNDLE").unwrap_or_else(|_| "smollm2-135m".to_string());
    let iters: u32 = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let Some(dir) = bundle_dir_named(&bundle) else {
        eprintln!("{bundle} bundle absent — skipping");
        return;
    };

    // Build the module + ProgramSpec once, and marshal the source args once
    // (weights from t{id}.bin, attn mask zeroed) — exactly the front-end of
    // `run_segmented_result`, but reused across all timed passes so we measure
    // execute_segmented itself, not the one-time parse/load.
    let b = load_bundle(&dir);
    let mut owned: Vec<(String, Arg)> = Vec::new();
    for (&id, &(rows, cols, is_src)) in &b.shape {
        if is_src && Some(id) != b.mask_id {
            owned.push((
                format!("t{id}"),
                Arg::Tensor {
                    data: read_f32(&dir.join(format!("t{id}.bin"))),
                    shape: vec![rows, cols],
                    dtype: DType::F16,
                },
            ));
        }
    }
    if let Some(m) = b.mask_id {
        let (rows, cols, _) = b.shape[&m];
        owned.push((
            format!("t{m}"),
            Arg::Tensor {
                data: vec![0.0f32; rows * cols],
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let result_key = format!("t{}", b.result_id);

    // Warm up (pipeline compile, first-touch, weight-cache fill) — excluded.
    ktir_emulator::segmented::execute_segmented(&b.module, &b.spec, &args, &[&result_key])
        .expect("warmup");

    let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        ktir_emulator::segmented::execute_segmented(&b.module, &b.spec, &args, &[&result_key])
            .expect("timed segmented run");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    eprintln!(
        "{bundle} e2e (Rust+Metal segmented): {median:.1} ms/pass  ({} nodes, {iters} passes)",
        b.n_nodes
    );
}

/// RESIDENT executor vs golden — the authoritative correctness gate for the
/// resident path. Runs all 4 configs (smollm2/llama × decode/prefill) that have a
/// bundle present, builds the `ResidentExecutor` (weights marshaled ONCE), runs
/// one pass, and compares to golden.bin. The resident path is byte-for-byte the
/// same segment plan + handlers + GPU offloads as `execute_segmented` — only the
/// HBM is persistent — so the diffs must match the known-current golden numbers
/// (smollm2 decode 0.0014 / prefill 0.0035; llama decode 0.0026 / prefill 0.0033;
/// all < 0.05).
#[cfg(metal)]
#[test]
#[ignore = "resident-executor golden gate; needs the bundles. --ignored --nocapture"]
fn resident_matches_golden() {
    let attn = [
        "KTIR_GPU_PLAIN_MATMUL",
        "KTIR_GPU_REDUCE",
        "KTIR_GPU_TRANSPOSE",
    ];
    let mut any = false;
    let mut failures: Vec<String> = Vec::new();
    for bundle in [
        "smollm2-135m",
        "smollm2-135m-prefill",
        "llama-3.2-1b",
        "llama-3.2-1b-prefill",
    ] {
        let Some(dir) = bundle_dir_named(bundle) else {
            eprintln!("{bundle} bundle absent — skipping");
            continue;
        };
        any = true;
        // Prefill bundles have head-parallel attention nodes; enable the opt-in
        // attention-island GPU offloads so the native segments exercise the GPU
        // path (matches the segmented golden tests' configuration).
        let is_prefill = bundle.ends_with("prefill");
        if is_prefill {
            for k in attn {
                unsafe { std::env::set_var(k, "1") };
            }
        }
        let (result, n_fused, n_native) = run_resident_result(&dir);
        if is_prefill {
            for k in attn {
                unsafe { std::env::remove_var(k) };
            }
        }
        let golden = read_f32(&dir.join("golden.bin"));
        assert_eq!(result.len(), golden.len(), "{bundle}: result length");
        let finite = result.iter().filter(|x| x.is_finite()).count();
        let max_abs = result
            .iter()
            .zip(&golden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "  RESIDENT {bundle} ({n_fused} fused + {n_native} native): \
             {finite}/{} finite, max abs diff {max_abs:.5}",
            result.len()
        );
        if finite != result.len() {
            failures.push(format!("{bundle}: non-finite result"));
        }
        if max_abs >= 0.05 {
            failures.push(format!("{bundle}: diverges from golden by {max_abs}"));
        }
    }
    if !any {
        eprintln!("no bundles present — skipping resident golden gate");
        return;
    }
    assert!(
        failures.is_empty(),
        "resident golden failures: {failures:?}"
    );
}

/// PREFILL multi-core SPMD vs golden — the AUTHORITATIVE gate for cross-core
/// grid execution. Runs every prefill node at its NATIVE grid ([1,1] / [8,1]
/// token-parallel matmuls / [9,1] attention heads), threading one shared HBM
/// buffer per tensor between nodes, and compares the final result to golden.bin.
///
/// Each [8,1] node has 8 cores compute one token row each (`%pid =
/// get_compute_tile_id`, store `view[%pid, ...]`); each [9,1] node has 9 cores
/// compute one attention head each (writing disjoint 64-column head slices of
/// the shared rows). All cores write to the SAME shared HBM, and the readback
/// must capture every core's slice. A broken multi-core path (e.g. cores
/// clobbering each other's rows/columns, or get_compute_tile_id mapping, or a
/// readback that only sees one core's HBM) produces head-0-only attention and
/// diverges from golden by ~0.18. A correct one matches to f16 tolerance.
///
/// 0.05 is well above the observed ~0.0034 f16 noise yet far below the ~0.18 a
/// broken multi-core attention would give, so it rigorously distinguishes
/// correct cross-core SPMD from broken — it is not a rubber-stamp tolerance.
#[test]
#[ignore = "real-model prefill per-node multi-core; needs smollm2-135m-prefill. --ignored --nocapture"]
fn smollm2_135m_prefill_per_node_multicore_matches_golden() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    let result = run_per_node_result(&dir);
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(result.len(), golden.len(), "result length");
    let finite = result.iter().filter(|x| x.is_finite()).count();
    let max_abs = result
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "PREFILL per-node MULTI-CORE result vs golden: {finite}/{} finite, max abs diff {max_abs:.5}",
        result.len()
    );
    assert_eq!(finite, result.len(), "all result elements finite");
    assert!(
        max_abs < 0.05,
        "prefill per-node multi-core diverges from golden by {max_abs} — cross-core SPMD is wrong"
    );
}

/// Marshal the fused-function args for a bundle (sources from t{id}.bin, mask +
/// results/intermediates zeroed), returning (fused module, arg list, result_id).
#[cfg_attr(not(metal), allow(dead_code))]
fn fused_run_inputs(dir: &std::path::Path) -> (IRModule, Vec<(String, Arg)>, u64) {
    let b = fuse_bundle(dir);
    let fused = b.func;
    let mut args: Vec<(String, Arg)> = Vec::new();
    for (name, _) in &fused.arguments {
        let id = tensor_id_of(name);
        let (rows, cols, is_src) = b.shape[&id];
        let data = if is_src && Some(id) != b.mask_id {
            read_f32(&dir.join(format!("t{id}.bin")))
        } else {
            vec![0.0f32; rows * cols]
        };
        args.push((
            name.trim_start_matches('%').to_string(),
            Arg::Tensor {
                data,
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let mut module = IRModule::default();
    module.add_function(fused);
    (module, args, b.result_id)
}

/// Median ms/pass of `execute_function` on a fused bundle over `iters` runs.
#[cfg(metal)]
fn time_fused(dir: &std::path::Path, iters: u32) -> f64 {
    let (module, args, result_id) = fused_run_inputs(dir);
    let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let result_ptr = format!("t{result_id}_ptr");
    // Warm up (pipeline compile, first-touch).
    execute_function_outputs(&module, "fused", &refs, &[&result_ptr]).expect("warmup");
    let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        execute_function_outputs(&module, "fused", &refs, &[&result_ptr]).expect("timed run");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// PERF: decode + prefill ms/pass with the ATTENTION-ISLAND offloads (plain
/// matmul + reduce + transpose) ON vs OFF. Both arms keep the K-loop GEMM and
/// map-window offloads ON — this isolates the attention contribution. Run ONE
/// bench at a time (no concurrency); the env toggles are process-global.
#[cfg(metal)]
#[test]
#[ignore = "perf bench; needs the smollm2-135m[-prefill] bundles. --ignored --nocapture"]
fn fused_attention_gpu_vs_cpu_mspass() {
    let iters: u32 = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    // Opt-in attention offloads (default OFF). Presence ENABLES.
    let all = [
        "KTIR_GPU_PLAIN_MATMUL",
        "KTIR_GPU_REDUCE",
        "KTIR_GPU_TRANSPOSE",
    ];
    let on = |k: &str| unsafe { std::env::set_var(k, "1") };
    let off = |k: &str| unsafe { std::env::remove_var(k) };
    for (model, dir_opt) in [
        ("decode", bundle_dir()),
        ("prefill", bundle_dir_named("smollm2-135m-prefill")),
    ] {
        let Some(dir) = dir_opt else {
            eprintln!("{model} bundle absent — skipping");
            continue;
        };
        // Baseline: every attention offload OFF (attention fully on CPU). The
        // K-loop GEMM + map offloads stay ON in all arms (KTIR_NO_GPU_GEMM unset).
        for k in all {
            off(k);
        }
        let cpu = time_fused(&dir, iters);
        // Sweep: enable each offload alone, then all three together.
        let mut report = vec![(format!("{model}: attention ALL-CPU"), cpu)];
        for combo in [
            vec!["KTIR_GPU_PLAIN_MATMUL"],
            vec!["KTIR_GPU_REDUCE"],
            vec!["KTIR_GPU_TRANSPOSE"],
            all.to_vec(),
        ] {
            for k in all {
                off(k);
            }
            for k in &combo {
                on(k);
            }
            let label = match combo.as_slice() {
                ["KTIR_GPU_PLAIN_MATMUL"] => "GPU plain-matmul only",
                ["KTIR_GPU_REDUCE"] => "GPU reduce only",
                ["KTIR_GPU_TRANSPOSE"] => "GPU transpose only",
                _ => "GPU all three",
            };
            let t = time_fused(&dir, iters);
            report.push((format!("{model}: {label}"), t));
        }
        for k in all {
            off(k);
        }
        for (label, t) in &report {
            eprintln!("  {label}: {t:.1} ms/pass  ({:.2}x vs ALL-CPU)", cpu / t);
        }
    }
}

/// PERF: decode + prefill ms/pass with the GPU GEMM offload ON vs OFF (the
/// interpreter's Accelerate K-loop). Run ONE bench at a time (no concurrency).
#[cfg(metal)]
#[test]
#[ignore = "perf bench; needs the smollm2-135m[-prefill] bundles. --ignored --nocapture"]
fn fused_gpu_vs_cpu_mspass() {
    let iters: u32 = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    for (model, dir_opt) in [
        ("decode", bundle_dir()),
        ("prefill", bundle_dir_named("smollm2-135m-prefill")),
    ] {
        let Some(dir) = dir_opt else {
            eprintln!("{model} bundle absent — skipping");
            continue;
        };
        // SAFETY: single-threaded test; toggling our own offload gate.
        unsafe { std::env::set_var("KTIR_NO_GPU_GEMM", "1") };
        let cpu = time_fused(&dir, iters);
        unsafe { std::env::remove_var("KTIR_NO_GPU_GEMM") };
        let gpu = time_fused(&dir, iters);
        eprintln!(
            "{model}: CPU K-loops {cpu:.1} ms/pass  |  GPU GEMMs {gpu:.1} ms/pass  |  speedup {:.2}x",
            cpu / gpu
        );
    }
}

/// ADVERSARIAL: the resident weight cache must NEVER serve a STALE weight. We run
/// the SAME fused function TWICE with DIFFERENT weight values bound to the SAME
/// argument names (the exact hazard a name-keyed cache would mishandle — each run
/// allocates HBM deterministically, so the SSA root names AND stick addresses
/// repeat across runs; only the weight *content* differs). The content
/// fingerprint in [`WeightKey`] must detect the changed bytes and force a refresh,
/// so the second result reflects the NEW weights.
///
/// If the cache keyed by name alone (the bug the user forbids), pass 2 would reuse
/// pass 1's resident buffers and the two results would be IDENTICAL. We assert
/// they DIFFER (the new weights took effect) and, as a positive control, that the
/// weight-cache MISS counter advanced on pass 2 (the fingerprint forced a
/// re-decode+re-upload), proving it was the fingerprint — not a coincidence — that
/// caught the change.
#[cfg(metal)]
#[test]
#[ignore = "weight-cache staleness guard; needs ~/.cache/cudaforge/ktir/smollm2-135m. \
            Run with --ignored --nocapture"]
fn weight_cache_refreshes_on_changed_weights() {
    use std::sync::atomic::Ordering;
    let Some(dir) = bundle_dir() else {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    };
    let (module, args, result_id) = fused_run_inputs(&dir);
    let result_ptr = format!("t{result_id}_ptr");

    // Start from a clean cache so this test's miss-counter assertion is isolated.
    ktir_emulator::metal::clear_weight_cache();

    // Pass 1: original weights. Capture the result.
    let refs1: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let out1 = execute_function_outputs(&module, "fused", &refs1, &[&result_ptr])
        .expect("pass 1")
        .get(&result_ptr)
        .expect("pass 1 result")
        .data
        .clone();

    // Build pass 2 args: SAME names, but every non-zero (source weight) tensor
    // scaled by 2.0 so its HBM bytes — and thus its fingerprint — change. The mask
    // / zeroed intermediates stay zero (scaling 0 is 0, harmless). Activations are
    // recomputed inside the run and are never cached, so this only exercises the
    // weight path.
    let scaled: Vec<(String, Arg)> = args
        .iter()
        .map(|(name, arg)| {
            let Arg::Tensor { data, shape, dtype } = arg else {
                return (name.clone(), arg.clone());
            };
            let data: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
            (
                name.clone(),
                Arg::Tensor {
                    data,
                    shape: shape.clone(),
                    dtype: *dtype,
                },
            )
        })
        .collect();

    let misses_before = ktir_emulator::metal::WEIGHT_CACHE_MISSES.load(Ordering::Relaxed);
    let refs2: Vec<(&str, Arg)> = scaled
        .iter()
        .map(|(n, a)| (n.as_str(), a.clone()))
        .collect();
    let out2 = execute_function_outputs(&module, "fused", &refs2, &[&result_ptr])
        .expect("pass 2")
        .get(&result_ptr)
        .expect("pass 2 result")
        .data
        .clone();
    let misses_after = ktir_emulator::metal::WEIGHT_CACHE_MISSES.load(Ordering::Relaxed);

    let max_abs = out1
        .iter()
        .zip(&out2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let refreshed = misses_after - misses_before;
    eprintln!(
        "staleness guard: pass1 vs pass2(scaled weights) max abs diff {max_abs:.5}; \
         weight-cache misses on pass 2 = {refreshed} (fingerprint-forced re-uploads)"
    );

    assert_eq!(out1.len(), out2.len(), "result length");
    // The new weights MUST have taken effect — a name-only cache would return
    // pass-1's stale buffers and give an identical result (max_abs == 0).
    assert!(
        max_abs > 1e-3,
        "scaled weights produced an IDENTICAL result ({max_abs}) — the cache served STALE weights"
    );
    // Positive control: the fingerprint detected the change and forced refreshes.
    assert!(
        refreshed > 0,
        "no weight-cache misses on pass 2 — the fingerprint did NOT detect the changed weights"
    );
}

/// Fuse a bundle, run the fused function through the interpreter (with the
/// matmul-loop GPU offload active under cfg(metal)), and compare the result to
/// golden.bin. Returns max-abs-diff vs golden.
fn run_fused_golden(dir: &std::path::Path, label: &str) -> (f32, Vec<f32>) {
    let b = fuse_bundle(dir);
    let (shape, result_id, mask_id, n_nodes) = (b.shape, b.result_id, b.mask_id, b.n_nodes);
    let fused = b.func;

    // Provide a buffer for EVERY pointer arg the fused function still declares:
    // sources from t{id}.bin (mask = zeros), results + non-forwarded
    // intermediates zero-initialized.
    let mut args: Vec<(String, Arg)> = Vec::new();
    for (name, _) in &fused.arguments {
        let id = tensor_id_of(name);
        let (rows, cols, is_src) = shape[&id];
        let data = if is_src && Some(id) != mask_id {
            read_f32(&dir.join(format!("t{id}.bin")))
        } else {
            vec![0.0f32; rows * cols]
        };
        args.push((
            name.trim_start_matches('%').to_string(),
            Arg::Tensor {
                data,
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let arg_refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

    let mut fused_module = IRModule::default();
    fused_module.add_function(fused);
    let result_ptr = format!("t{result_id}_ptr");
    let out = execute_function_outputs(&fused_module, "fused", &arg_refs, &[&result_ptr])
        .expect("run fused bundle");

    let got = &out
        .get(&format!("t{result_id}_ptr"))
        .expect("result tensor read back")
        .data;
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(got.len(), golden.len(), "result length");

    let mut max_abs = 0.0f32;
    let mut finite = 0usize;
    for (a, g) in got.iter().zip(&golden) {
        if a.is_finite() {
            finite += 1;
        }
        max_abs = max_abs.max((a - g).abs());
    }
    eprintln!(
        "{label} FUSED ({n_nodes} nodes -> 1 fn): result vs golden \
         {finite}/{} finite, max abs diff {max_abs:.4}",
        got.len()
    );
    assert_eq!(finite, got.len(), "all result elements finite");
    (max_abs, got.clone())
}

#[test]
#[ignore = "real-model fuse-then-run; needs the ~/.cache/cudaforge/ktir/smollm2-135m bundle. \
            Run with --ignored --nocapture"]
fn smollm2_135m_fused_matches_golden() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    };
    #[cfg(metal)]
    {
        ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        ktir_emulator::metal::MAP_REGION_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    let (max_abs, _) = run_fused_golden(&dir, "SmolLM2-135M decode");
    #[cfg(metal)]
    {
        let gpu =
            ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let maps =
            ktir_emulator::metal::MAP_REGION_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("  matmul K-loops offloaded to GPU GEMM: {gpu}");
        eprintln!("  map windows offloaded to fused GPU kernel: {maps}");
        // SIZE-GATED offload: decode is M=1, so its per-layer GEMMs/maps are tiny
        // (a net GPU loss) and route to the interpreter's Accelerate path; only
        // the big lm_head GEMM (k·n ≫ the work gate) goes to the GPU. So the proof
        // the Metal path is live is "at least one GEMM offloaded" (the lm_head),
        // not the old "all 200+" (which the gate now correctly keeps on AMX).
        assert!(
            gpu >= 1,
            "expected at least the lm_head K-loop on GPU, {gpu} did"
        );
        let _ = maps; // decode windows are below the map size gate (expected 0)
    }
    assert!(
        max_abs < 0.2,
        "decode fused diverges from golden by {max_abs}"
    );
}

/// PREFILL (M=8) end-to-end vs golden: the real throughput target. PARTIAL
/// FUSION — non-attention runs fuse into [1,1] segments (carrying the GPU GEMM /
/// map / attention offloads), and the head-parallel [9,1] attention nodes run at
/// their native grid (all 9 heads). Threaded through HBM in program order.
///
/// The earlier whole-program single-grid fuse collapsed attention to head 0 and
/// only passed (0.0271) because SmolLM2's gap is small; the segmented path runs
/// every head, so it should match golden more tightly (toward the per-node
/// oracle's ~0.003).
#[cfg(metal)]
#[test]
#[ignore = "real-model prefill fuse-then-run; needs smollm2-135m-prefill. --ignored --nocapture"]
fn smollm2_135m_prefill_fused_matches_golden() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.store(0, Relaxed);
    ktir_emulator::metal::MAP_REGION_GPU_COUNT.store(0, Relaxed);
    ktir_emulator::metal::PLAIN_MATMUL_GPU_COUNT.store(0, Relaxed);
    ktir_emulator::metal::REDUCE_GPU_COUNT.store(0, Relaxed);
    ktir_emulator::metal::TRANSPOSE_GPU_COUNT.store(0, Relaxed);
    let (fused, n_fused, n_native) = run_segmented_result(&dir);
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(fused.len(), golden.len(), "result length");
    let finite = fused.iter().filter(|x| x.is_finite()).count();
    let golden_diff = fused
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(finite, fused.len(), "all result elements finite");
    let gpu = ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.load(Relaxed);
    let amx = ktir_emulator::metal::MATMUL_LOOP_AMX_COUNT.load(Relaxed);
    let maps = ktir_emulator::metal::MAP_REGION_GPU_COUNT.load(Relaxed);
    eprintln!(
        "  SmolLM2-135M PREFILL SEGMENTED ({n_fused} fused segments + {n_native} native attn): \
         max abs diff {golden_diff:.5}"
    );
    eprintln!(
        "  prefill K-loops offloaded full-M: {gpu} NAX + {amx} AMX = {}",
        gpu + amx
    );
    eprintln!("  prefill map windows offloaded to fused GPU kernel: {maps}");
    // The fused [1,1] segments carry the GEMM offloads (NAX or AMX) + map windows;
    // the native attention nodes run at their [9,1] head-parallel grid via the
    // lockstep NAX executor (shared-weight matmul combine where it applies;
    // per-core for the head-distinct Q@K^T / softmax, tiny tensors where per-op GPU
    // dispatch is a net loss — see comm_sched).
    // SIZE-GATED backend: smollm2's layer GEMMs (k·n ≤ 0.9M) run full-M on AMX
    // (resident, no GPU dispatch — the win at M=8); only the lm_head (k·n=28M)
    // clears the NAX gate. Both are full-M resident offloads, so we assert the TOTAL
    // (NAX + AMX) is high — the path is live and every layer GEMM is offloaded, not
    // run on the interpreter. The M=8 map windows (≤4608 elems) are below the map
    // size gate, so they correctly stay on the interpreter (a net win), maps may be 0.
    assert!(
        gpu + amx >= 100,
        "expected most prefill K-loops offloaded full-M, only {gpu} NAX + {amx} AMX did"
    );
    let _ = maps;

    // AUTHORITATIVE GATE: the segmented + GPU-GEMM run must match golden.bin.
    // 0.05 is well above f16/GPU noise yet far below the ~0.18 a broken attention
    // (head-0-only) would produce — so this rigorously distinguishes correct from
    // broken, it is not a rubber-stamp tolerance.
    assert!(
        golden_diff < 0.05,
        "prefill segmented diverges from golden by {golden_diff} — fusion/attention is wrong"
    );

    // CROSS-CORE GATE: a from-scratch per-node oracle that runs each node at its
    // OWN grid ([8,1] token-parallel / [9,1] attention heads), the multi-core
    // SPMD path. It MATCHES golden to f16 tolerance (~0.0034) — i.e. the
    // emulator's MULTI-CORE SPMD execution of prefill nodes reproduces golden's
    // generation. The segmented path also matches golden, so the two agree.
    let oracle = run_per_node_result(&dir);
    let oracle_vs_golden = oracle
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let fused_vs_oracle = fused
        .iter()
        .zip(&oracle)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "  per-node multi-core oracle vs golden: {oracle_vs_golden:.5}; segmented vs oracle: {fused_vs_oracle:.5}"
    );
    assert!(
        oracle_vs_golden < 0.05,
        "prefill per-node multi-core SPMD diverges from golden by {oracle_vs_golden} — cross-core execution is wrong"
    );
}

/// PREFILL readiness: fuse the M=8 prefill bundle and confirm EVERY scf.for
/// K-loop is recognized as a single full-shape GEMM (the grid/token-parallel and
/// K-tiling decomposition collapses to one [8,k]@[k,n] matmul). This is the
/// proof that prefill is a first-class target, not a deferred one — the Metal
/// executor ignores the Spyre SPMD grid and reconstructs the whole GEMM.
#[cfg(metal)]
#[test]
#[ignore = "real-model prefill recognition; needs ~/.cache/cudaforge/ktir/smollm2-135m-prefill. \
            Run with --ignored --nocapture"]
fn prefill_matmul_loops_all_recognized() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    let b = fuse_bundle(&dir);
    let (total, recognized) = ktir_emulator::metal::count_matmul_loops(&b.func.operations);
    eprintln!(
        "prefill fused ({} nodes -> 1 fn): {} ops, {total} scf.for K-loops, \
         {recognized} recognized as GEMMs",
        b.n_nodes,
        b.func.operations.len()
    );
    assert!(
        total > 0,
        "expected matmul K-loops in the fused prefill function"
    );
    assert_eq!(
        total,
        recognized,
        "every prefill K-loop must collapse to one GEMM (M=8) — {} unrecognized",
        total - recognized
    );
}

/// MAP-WINDOW FUSION readiness: fuse the decode bundle and confirm `map_fusion_plan`
/// carves the elementwise op stream into fused GPU kernels — proving the runtime
/// map offload has work to do (the number of windows = the MAP_REGION_GPU_COUNT a
/// run produces). No GPU dispatch: just the plan, so it's fast and device-free.
#[cfg(metal)]
#[test]
#[ignore = "real-model map-fusion plan; needs ~/.cache/cudaforge/ktir/smollm2-135m. \
            Run with --ignored --nocapture"]
fn map_fusion_plan_carves_windows() {
    if bundle_dir().is_none() {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    }
    for which in ["smollm2-135m", "smollm2-135m-prefill"] {
        let Some(d) = bundle_dir_named(which) else {
            continue;
        };
        let b = fuse_bundle(&d);
        let ops = &b.func.operations;
        let (triggers, skip) = ktir_emulator::metal::map_fusion_plan(ops);
        eprintln!(
            "{which} fused ({} nodes -> 1 fn): {} map windows -> GPU kernels, {} op indices subsumed",
            b.n_nodes,
            triggers.len(),
            skip.len()
        );
        // element count per SSA result (product of its shape attr)
        let mut numel: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        fn rec(
            ops: &[ktir_emulator::ir::Operation],
            m: &mut std::collections::HashMap<String, i64>,
        ) {
            for op in ops {
                if let Some(r) = &op.result
                    && let Some(ktir_emulator::ir::Attr::IntList(s)) = op.attributes.get("shape")
                {
                    m.insert(r.trim_start_matches('%').to_string(), s.iter().product());
                }
                for region in &op.regions {
                    rec(region, m);
                }
            }
        }
        rec(ops, &mut numel);
        // For each kernel: a live-in read as `name[gid]` (not a broadcast index)
        // MUST have element count == out_len, else gid runs out of bounds.
        let mut mism = 0;
        for mrk in triggers.values() {
            let out_len: i64 = mrk.out_shape.iter().map(|&x| x as i64).product();
            for li in &mrk.live_ins {
                let key = li.trim_start_matches('%');
                let n = *numel.get(key).unwrap_or(&-1);
                let reads_gid = mrk.kernel.source.contains(&format!("{key}[gid]"));
                if reads_gid && n != out_len {
                    if mism < 15 {
                        eprintln!(
                            "  MISMATCH {which}: live_out {} reads {key}[gid] len={n} but out_len={out_len}",
                            mrk.live_out
                        );
                    }
                    mism += 1;
                }
            }
        }
        eprintln!("  {which}: {mism} live-ins read [gid] with len != out_len (would corrupt)");
        assert_eq!(mism, 0, "{which}: gid-indexed live-in length mismatch");
        assert!(!triggers.is_empty(), "{which}: expected fusable windows");
    }
}

// ===========================================================================
// Llama-3.2-1B — the BIG-model target (per the project goal: prefill/big-model,
// not the tiny 135M decode). ~8x SmolLM2; the GEMMs and attention are large
// enough that the GPU offloads should win decisively (the 135M numbers undersell
// them because tiny tensors are GPU-dispatch-overhead-bound).
// ===========================================================================

#[cfg(metal)]
#[test]
#[ignore = "big-model fuse-then-run; needs ~/.cache/cudaforge/ktir/llama-3.2-1b. --ignored --nocapture"]
fn llama_3_2_1b_fused_matches_golden() {
    let Some(dir) = bundle_dir_named("llama-3.2-1b") else {
        eprintln!("llama-3.2-1b bundle absent — skipping");
        return;
    };
    ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    ktir_emulator::metal::MAP_REGION_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let (max_abs, _) = run_fused_golden(&dir, "Llama-3.2-1B decode");
    let gemms =
        ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let maps =
        ktir_emulator::metal::MAP_REGION_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("  Llama-1B decode: {gemms} K-loop GEMMs + {maps} map windows on GPU");
    assert!(gemms > 0, "expected GPU GEMMs on the 1B model");
    assert!(
        max_abs < 0.05,
        "Llama-1B decode fused diverges from golden by {max_abs}"
    );
}

/// BIG-MODEL PREFILL correctness — the project's throughput target. The
/// whole-program single-grid fused run collapsed the head-parallel [32,1]
/// attention nodes to head 0 and diverged from golden by ~0.06 (FAIL). This runs
/// the PARTIAL-FUSION plan instead: the non-attention runs fuse into [1,1]
/// segments (carrying the GPU GEMM / map offloads), and every attention node runs
/// at its native [32,1] grid (all 32 heads), threaded through HBM in program
/// order. That restores every head and matches golden to f16 tolerance (~0.003).
///
/// The GPU offloads MUST still fire on the fused segments — asserted via the
/// global counters (they are process-global, so this test runs serially under
/// --test-threads=1).
#[cfg(metal)]
#[test]
#[ignore = "big-model prefill; needs ~/.cache/cudaforge/ktir/llama-3.2-1b-prefill. --ignored --nocapture"]
fn llama_3_2_1b_prefill_fused_matches_golden() {
    let Some(dir) = bundle_dir_named("llama-3.2-1b-prefill") else {
        eprintln!("llama-3.2-1b-prefill bundle absent — skipping");
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.store(0, Relaxed);
    ktir_emulator::metal::MAP_REGION_GPU_COUNT.store(0, Relaxed);
    // Attention-island offloads are opt-in; enable so the native attention nodes
    // exercise the GPU attention path. SAFETY: serial test (--test-threads=1).
    let attn = [
        "KTIR_GPU_PLAIN_MATMUL",
        "KTIR_GPU_REDUCE",
        "KTIR_GPU_TRANSPOSE",
    ];
    for k in attn {
        unsafe { std::env::set_var(k, "1") };
    }
    let (result, n_fused, n_native) = run_segmented_result(&dir);
    for k in attn {
        unsafe { std::env::remove_var(k) };
    }
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(result.len(), golden.len(), "result length");
    let finite = result.iter().filter(|x| x.is_finite()).count();
    let max_abs = result
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let gemms = ktir_emulator::metal::MATMUL_LOOP_GPU_COUNT.load(Relaxed);
    let maps = ktir_emulator::metal::MAP_REGION_GPU_COUNT.load(Relaxed);
    eprintln!(
        "Llama-3.2-1B PREFILL SEGMENTED ({n_fused} fused segments + {n_native} native attn): \
         {finite}/{} finite, max abs diff {max_abs:.5}; {gemms} K-loop GEMMs + {maps} map windows on GPU",
        result.len()
    );
    assert_eq!(finite, result.len(), "all result elements finite");
    // GPU offloads must still fire on the fused segments.
    assert!(
        gemms > 0,
        "expected GPU GEMMs on the fused prefill segments, none fired"
    );
    assert!(
        maps > 0,
        "expected GPU map windows on the fused prefill segments, none fired"
    );
    assert!(
        max_abs < 0.05,
        "Llama-1B prefill segmented diverges from golden by {max_abs} — attention/fusion is wrong"
    );
}

/// Big-model perf: GPU offloads ON vs OFF, on Llama-3.2-1B (where tensors are
/// large enough that the GPU wins). Run alone (no concurrent benches).
#[cfg(metal)]
#[test]
#[ignore = "big-model perf bench; needs the llama-3.2-1b[-prefill] bundles. --ignored --nocapture"]
fn llama_3_2_1b_gpu_vs_cpu_mspass() {
    let iters: u32 = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    for (model, dir) in [
        ("decode", bundle_dir_named("llama-3.2-1b")),
        ("prefill", bundle_dir_named("llama-3.2-1b-prefill")),
    ] {
        let Some(dir) = dir else {
            eprintln!("llama {model} absent — skipping");
            continue;
        };
        // SAFETY: single-threaded test toggling our own offload gates.
        unsafe {
            std::env::set_var("KTIR_NO_GPU_GEMM", "1");
            std::env::set_var("KTIR_NO_GPU_MAP", "1");
        }
        let cpu = time_fused(&dir, iters);
        unsafe {
            std::env::remove_var("KTIR_NO_GPU_GEMM");
            std::env::remove_var("KTIR_NO_GPU_MAP");
        }
        let gpu = time_fused(&dir, iters);
        eprintln!(
            "Llama-1B {model}: all-CPU {cpu:.0} ms/pass  |  GPU fusion {gpu:.0} ms/pass  |  speedup {:.2}x",
            cpu / gpu
        );
    }
}

/// LX-BUDGETED SEGMENTATION: a tiny `KTIR_LX_FUSION_BUDGET` must (a) split the
/// non-attention runs into MORE fused segments than the unbudgeted plan, and
/// (b) still match golden — proving the split (which routes the broken edges
/// through HBM) preserves correctness. This is the fix for the llama m=32 MLP
/// overflow: at the real budget an m=8 MLP run stays one segment (it fits), so a
/// tiny budget is how we exercise the splitter + its HBM boundary edges here.
/// (The fusion budget gates SEGMENTATION only; the runtime LX is still 2 MB, so
/// the more-split program executes fine and must reproduce golden.)
#[cfg(metal)]
#[test]
#[ignore = "LX-split golden; needs the smollm2 prefill bundle. --ignored --nocapture"]
fn lx_budget_split_preserves_golden() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    let b = load_bundle(&dir);
    let tensor_bytes: HashMap<u64, usize> = b
        .shape
        .iter()
        .map(|(&id, &(r, c, _))| (id, r * c * 2))
        .collect(); // f16
    let base = plan_segments(&b.module, &b.spec).expect("plan").len();
    let tiny = 40_000usize;
    let split = plan_segments_budgeted(&b.module, &b.spec, tiny, &tensor_bytes)
        .expect("plan budgeted")
        .len();
    eprintln!("  segments: {base} (no budget) -> {split} (budget {tiny}B)");
    assert!(
        split > base,
        "a tiny LX budget should split runs into MORE segments"
    );

    // Execute under the tiny budget (env-overridden) and confirm golden holds.
    unsafe { std::env::set_var("KTIR_LX_FUSION_BUDGET", tiny.to_string()) };
    let (result, _nf, _nn) = run_segmented_result(&dir);
    unsafe { std::env::remove_var("KTIR_LX_FUSION_BUDGET") };
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(result.len(), golden.len(), "result length");
    let diff = result
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("  split-budget execute vs golden: max abs diff {diff:.5}");
    assert!(
        diff < 0.05,
        "LX-split execution diverged from golden by {diff}"
    );
}

/// TURNKEY entrypoint smoke test: drive the whole program through
/// `ktir_emulator::program::execute` (per-node MLIR + ProgramSpec -> one optimized
/// run) and check it matches golden. Proves `module_from_nodes` merges the
/// per-node functions into one module that the optimized path runs correctly —
/// the single-call path scratchy would use instead of looping execute_function.
#[cfg(metal)]
#[test]
#[ignore = "turnkey program::execute smoke; needs the smollm2 bundle. --ignored --nocapture"]
fn program_execute_matches_golden() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    };
    let b = load_bundle(&dir);
    // The turnkey entrypoint takes the per-node MLIR as &[&str].
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let texts: Vec<String> = manifest["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| std::fs::read_to_string(dir.join(n["mlir"].as_str().unwrap())).unwrap())
        .collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Source args from t<id>.bin (+ zeroed attn mask) — same as run_segmented_result.
    let mut owned: Vec<(String, Arg)> = Vec::new();
    for (&id, &(rows, cols, is_src)) in &b.shape {
        if is_src && Some(id) != b.mask_id {
            owned.push((
                format!("t{id}"),
                Arg::Tensor {
                    data: read_f32(&dir.join(format!("t{id}.bin"))),
                    shape: vec![rows, cols],
                    dtype: DType::F16,
                },
            ));
        }
    }
    if let Some(m) = b.mask_id {
        let (rows, cols, _) = b.shape[&m];
        owned.push((
            format!("t{m}"),
            Arg::Tensor {
                data: vec![0.0f32; rows * cols],
                shape: vec![rows, cols],
                dtype: DType::F16,
            },
        ));
    }
    let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let result_key = format!("t{}", b.result_id);

    let out = ktir_emulator::program::execute(&refs, &b.spec, &args, &[&result_key])
        .expect("program::execute");
    let result = &out[&result_key].data;
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(result.len(), golden.len(), "result length");
    let diff = result
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "  program::execute ({} nodes) vs golden: max abs diff {diff:.5}",
        refs.len()
    );
    assert!(
        diff < 0.05,
        "turnkey program::execute diverges from golden by {diff}"
    );
}
