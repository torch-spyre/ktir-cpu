// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Profiling target for the real-model run: runs the SmolLM2-135M KTIR bundle
//! (`~/.cache/cudaforge/ktir/smollm2-135m/`) through `execute_function` in a
//! loop so a sampling profiler has wall time. Plain `fn main` (harness = false).
//!   cargo instruments -t time --release --bench smollm2_bench
//! Skips (exits 0) when the bundle is absent. SMOLLM2_ITERS controls the loop.

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::parser::parse_module;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    // MODEL selects which compiled-model dir under ~/.cache/cudaforge/ktir/ to
    // run (default the fp16 SmolLM2-135M decode model). The runner is
    // shape-agnostic, so a prefill model dir drops in the same way once one is
    // compiled.
    let model = std::env::var("MODEL").unwrap_or_else(|_| "smollm2-135m".to_string());
    let Some(dir) = std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".cache/cudaforge/ktir").join(&model))
        .filter(|d| d.join("manifest.json").is_file())
    else {
        eprintln!("model {model} absent under ~/.cache/cudaforge/ktir — skipping");
        return;
    };
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let read_f32 = |p: &std::path::Path| -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let mut shape: HashMap<u64, (usize, usize)> = HashMap::new();
    let mut sources: HashMap<u64, Vec<f32>> = HashMap::new();
    for t in manifest["tensors"].as_array().unwrap() {
        let id = t["id"].as_u64().unwrap();
        shape.insert(
            id,
            (
                t["rows"].as_u64().unwrap() as usize,
                t["cols"].as_u64().unwrap() as usize,
            ),
        );
        if t["is_source"].as_bool().unwrap_or(false) {
            sources.insert(id, read_f32(&dir.join(format!("t{id}.bin"))));
        }
    }

    // The attention mask (`attn_mask`) is a runtime input — not a source, not
    // produced by any node, no t{id}.bin — so seed it with the all-zeros causal
    // mask for single-token decode (matches tests/e2e_smollm2.rs and the
    // production runner). Without this, node 6 panics ("no entry found").
    if let Some(mid) = manifest["attn_mask"].as_u64() {
        let (r, c) = shape[&mid];
        sources.insert(mid, vec![0.0f32; r * c]);
    }

    let nodes = manifest["nodes"].as_array().unwrap();
    let mut cache: HashMap<String, ktir_emulator::ir::IRModule> = HashMap::new();
    for node in nodes {
        let name = node["mlir"].as_str().unwrap();
        cache.entry(name.to_string()).or_insert_with(|| {
            parse_module(&std::fs::read_to_string(dir.join(name)).unwrap()).unwrap()
        });
    }

    let iters: usize = std::env::var("SMOLLM2_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    // One full-model pass: every node, with per-node dispatch + arg marshaling +
    // inter-node tensor threading — i.e. the interpreter itself, end to end.
    // Mirrors bench_e2e_py_vs_rust.py.
    let one_pass = || {
        let mut buf = sources.clone();
        for node in nodes {
            let func = node["fn"].as_str().unwrap();
            let module = &cache[node["mlir"].as_str().unwrap()];
            let mut owned: Vec<(String, Arg)> = Vec::new();
            let mut outs: Vec<(String, u64)> = Vec::new();
            for a in node["args"].as_array().unwrap() {
                let nm = a["name"].as_str().unwrap().to_string();
                let tid = a["tensor"].as_u64().unwrap();
                let is_out = a["is_output"].as_bool().unwrap_or(false);
                let (r, c) = shape[&tid];
                let data = if is_out {
                    vec![0.0f32; r * c]
                } else {
                    buf[&tid].clone()
                };
                owned.push((
                    nm.clone(),
                    Arg::Tensor {
                        data,
                        shape: vec![r, c],
                        dtype: DType::F16,
                    },
                ));
                if is_out {
                    outs.push((nm, tid));
                }
            }
            let refs: Vec<(&str, Arg)> =
                owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
            let out = execute_function(module, func, &refs).unwrap();
            for (nm, tid) in outs {
                buf.insert(tid, out[&nm].data.clone());
            }
        }
    };

    // Profiling mode: when SMOLLM2_PROFILE is set, just run the passes (for a
    // sampling profiler). Otherwise report ms/pass (one warm-up excluded),
    // matching bench_e2e_py_vs_rust.py so the two are directly comparable.
    if std::env::var_os("SMOLLM2_PROFILE").is_some() {
        eprintln!("profiling {} nodes x {iters} passes...", nodes.len());
        for _ in 0..iters {
            one_pass();
        }
        eprintln!("done");
    } else {
        one_pass(); // warm-up
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            one_pass();
        }
        let ms = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
        println!(
            "{model} e2e (Rust): {ms:.1} ms/pass  ({} nodes, {iters} passes)",
            nodes.len()
        );
    }
}
