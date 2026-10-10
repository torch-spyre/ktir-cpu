// HONEST whole-program e2e measurement of the ktir-optimizer passes (TODO #1
// head-rewrite + TODO #2 flash-attention) on the REAL smollm2-135m-prefill bundle:
// the SAME segmented execution path, original module vs a pass-applied clone, both
// checked against golden.bin. This is end-to-end wall-clock — not a per-node
// microbenchmark, not a synthetic shape, not kernel-time-vs-wall-time. (A timing run
// legitimately uses the bundle's real weights; the CORRECTNESS of the passes is
// proven weight-free elsewhere in tests/head_rewrite_golden.rs.)
//
// NOTE: at this cached context length the scores tile fits LX, so flash-attention
// correctly NO-OPs (its win needs a long-context bundle that doesn't exist); the
// measured e2e win here is head-rewrite's.
#![cfg(metal)]

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::ir::IRModule;
use ktir_emulator::parser::parse_module;
use ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec, attention_needs_flash};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn bundle_dir(model: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home)
        .join(".cache/cudaforge/ktir")
        .join(model);
    dir.join("manifest.json").is_file().then_some(dir)
}
fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Bundle {
    module: IRModule,
    spec: ProgramSpec,
    shape: HashMap<u64, (usize, usize, bool)>,
    result_id: u64,
    mask_id: Option<u64>,
}

fn load_bundle(dir: &Path) -> Bundle {
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
        let src = std::fs::read_to_string(dir.join(node["mlir"].as_str().unwrap())).unwrap();
        for (_, f) in parse_module(&src).unwrap().functions {
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
    }
}

#[test]
#[ignore = "TIMING e2e on the real smollm2-135m-prefill bundle. --release --ignored --nocapture --test-threads=1"]
fn head_rewrite_e2e_smollm2_prefill() {
    let Some(dir) = bundle_dir("smollm2-135m-prefill") else {
        eprintln!("smollm2-135m-prefill bundle absent — skipping");
        return;
    };
    if ktir_emulator::metal::NaxGemm::new().is_err() {
        eprintln!("no NAX device — skipping");
        return;
    }
    let b = load_bundle(&dir);
    let golden = read_f32(&dir.join("golden.bin"));

    // Build the source args once (so disk reads don't contaminate timing).
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
    let result_key = format!("t{}", b.result_id);
    let run = |m: &IRModule| -> Vec<f32> {
        let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
        ktir_emulator::segmented::execute_segmented(m, &b.spec, &args, &[&result_key])
            .expect("execute_segmented")
            .get(&result_key)
            .expect("result")
            .data
            .clone()
    };
    let best = |m: &IRModule, iters: u32| -> (f64, Vec<f32>) {
        let out = run(m); // warm
        let mut bt = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            let o = run(m);
            bt = bt.min(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&o);
        }
        (bt, out)
    };
    let maxabs = |v: &[f32]| {
        v.iter()
            .zip(&golden)
            .map(|(a, g)| (a - g).abs())
            .fold(0.0f32, f32::max)
    };

    // WITHOUT the passes: the raw module (the current production segmented path).
    let (off_ms, off_out) = best(&b.module, 3);

    // WITH the passes: a clone with head-rewrite (+ flash-attention) applied, exactly
    // as program::module_from_nodes does (same LX-budget needs_flash predicate).
    let budget = ktir_emulator::memory::lx_fusion_budget();
    let mut m2 = b.module.clone();
    let n_head = ktir_optimizer::head_rewrite::apply_head_rewrite(&mut m2, |sb| {
        attention_needs_flash(sb, budget)
    });
    let n_flash = ktir_optimizer::flash_attn::apply_flash_attention(&mut m2, |sb| {
        attention_needs_flash(sb, budget)
    });
    let (on_ms, on_out) = best(&m2, 3);

    eprintln!(
        "\nsmollm2-135m-prefill e2e (best-of-3, release):\n  \
         WITHOUT passes: {off_ms:.1} ms/pass  (golden max-abs {:.5})\n  \
         WITH passes   : {on_ms:.1} ms/pass  (golden max-abs {:.5})  [head_rewrite fired on {n_head} nodes, flash_attn on {n_flash}]\n  \
         e2e speedup   : {:.2}x",
        maxabs(&off_out),
        maxabs(&on_out),
        off_ms / on_ms
    );
    // Both must match golden (the passes are correctness-preserving).
    assert!(
        maxabs(&off_out) < 0.05,
        "WITHOUT-passes diverged from golden"
    );
    assert!(
        maxabs(&on_out) < 0.05,
        "WITH-passes diverged from golden — the optimizer pass broke correctness e2e"
    );
    assert!(
        n_head > 0,
        "head_rewrite did not fire on the prefill bundle"
    );
}
