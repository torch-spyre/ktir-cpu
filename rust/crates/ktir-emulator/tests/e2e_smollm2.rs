// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! RUST-ONLY (not a port of a Python test): drive a REAL MODEL — the
//! SmolLM2-135M KTIR bundle scratchy emits under `-Fspyre` — end-to-end through
//! `parse_module` + `execute_function`, threading one host buffer per tensor
//! across all 452 nodes (the runner `scratchy-target-spyre` uses in production).
//!
//! Bundle layout (`~/.cache/cudaforge/ktir/smollm2-135m/`): `manifest.json`
//! (tensors, nodes, wiring), the `nodeN.mlir` kernels, `t{id}.bin` (f32 source
//! tensors), and `golden.bin` (f32 reference for the result tensor). The bundle
//! is machine-specific and NOT in the repo, so this test SKIPS when it is absent.
//!
//! This is the pressure test that surfaces real-model bugs: when a node fails,
//! the panic names the exact node + fn + error (e.g. the operand-dedup bug was
//! found this way — it broke node0, the first RMSNorm). `--ignored` because it
//! runs the whole model.

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::parser::parse_module;
use std::collections::HashMap;
use std::path::PathBuf;

fn bundle_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".cache/cudaforge/ktir/smollm2-135m");
    dir.join("manifest.json").is_file().then_some(dir)
}

/// Read a `.bin` file of little-endian f32.
fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "real-model e2e; needs the ~/.cache/cudaforge/ktir/smollm2-135m bundle. \
            Run with --ignored --nocapture"]
fn smollm2_135m_runs_end_to_end() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    };
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    // tensor id -> (rows, cols, is_source)
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

    // One host buffer (f32) per tensor; sources preloaded from t{id}.bin.
    let mut buf: HashMap<u64, Vec<f32>> = HashMap::new();
    for (&id, &(_, _, is_source)) in &shape {
        if is_source {
            buf.insert(id, read_f32(&dir.join(format!("t{id}.bin"))));
        }
    }

    // The attention mask (`attn_mask` in the manifest) is a runtime input — not
    // a source weight and not produced by any node, so it has no `t{id}.bin`.
    // For single-token decode at `decode_position`, the query attends to every
    // key 0..=decode_position (full causal visibility), so the additive mask is
    // all zeros. `scratchy-target-spyre` supplies this same buffer in production.
    if let Some(mask_id) = manifest["attn_mask"].as_u64() {
        let (r, c, _) = shape[&mask_id];
        buf.insert(mask_id, vec![0.0f32; r * c]);
    }

    let nodes = manifest["nodes"].as_array().unwrap();
    let n_nodes = nodes.len();
    let mut cache: HashMap<String, ktir_emulator::ir::IRModule> = HashMap::new();
    let mut ran = 0usize;

    // Profiling: re-run the whole node sweep `SMOLLM2_ITERS` times (sources are
    // reloaded each pass) so a sampling profiler has enough wall time.
    let iters: usize = std::env::var("SMOLLM2_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let sources: HashMap<u64, Vec<f32>> = buf.clone();
    for pass in 0..iters {
        if pass > 0 {
            buf.clone_from(&sources);
            ran = 0;
        }
        for (ni, node) in nodes.iter().enumerate() {
            let func = node["fn"].as_str().unwrap();
            let mlir_name = node["mlir"].as_str().unwrap();
            let module = cache.entry(mlir_name.to_string()).or_insert_with(|| {
                let src = std::fs::read_to_string(dir.join(mlir_name)).unwrap();
                parse_module(&src).unwrap_or_else(|e| panic!("node {ni} parse {mlir_name}: {e}"))
            });

            // Build args: every arg is a tensor ptr (f16 in HBM, f32 host buffer).
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
                    buf.get(&tid).cloned().unwrap_or_else(|| {
                        panic!("node {ni} ({func}): input tensor {tid} not yet produced")
                    })
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
            let arg_refs: Vec<(&str, Arg)> =
                args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

            let out = execute_function(module, func, &arg_refs).unwrap_or_else(|e| {
                panic!("NODE {ni}/{n_nodes} ({func}, {mlir_name}) FAILED: {e}")
            });
            // Thread outputs back into the tensor buffers.
            for (name, tid, is_out) in &arg_ids {
                if *is_out {
                    buf.insert(*tid, out.get(name).expect("output present").data.clone());
                }
            }
            ran += 1;
            if pass == 0 && ni % 50 == 0 {
                eprintln!("  node {ni}/{n_nodes} ({func}) ok");
            }
        }
    } // pass loop

    assert_eq!(ran, n_nodes, "all nodes ran");
    eprintln!("SmolLM2-135M: all {n_nodes} nodes executed end-to-end ✓");

    // Compare the result tensor to golden (f32, f16-compute tolerance).
    let result_id = manifest["result"].as_u64().unwrap();
    let got = &buf[&result_id];
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(got.len(), golden.len(), "result length");
    let mut max_abs = 0.0f32;
    let (mut g_finite, mut tot) = (0usize, 0usize);
    for (a, b) in got.iter().zip(&golden) {
        if a.is_finite() {
            g_finite += 1;
        }
        max_abs = max_abs.max((a - b).abs());
        tot += 1;
    }
    eprintln!("result vs golden: {g_finite}/{tot} finite, max abs diff {max_abs:.4} (f16 compute)");
}
