// The head-parallel attention RE-ROLL pass (`ktir_optimizer::head_rewrite`) must be
// SEMANTICS-PRESERVING: the re-rolled whole-`[m,*]` IR, run through the UNCHANGED
// per-core reference interpreter (`execute_function`), must equal the ORIGINAL
// unrolled node run the same way, for ARBITRARY inputs (shapes derived from the IR's
// own `construct_memory_view` sizes). This validates the COMPILER — it has nothing to
// do with model weights or golden outputs. We feed the REAL llama-3.2-1b-prefill and
// smollm2-135m-prefill attention node MLIR (present in the bundle dirs; no `.bin`/
// golden needed) and assert rewritten == original to f16 tolerance.
#![cfg(metal)]

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ir::{Attr, IRModule};
use ktir_emulator::parser::parse_module;
use ktir_optimizer::head_rewrite::apply_head_rewrite;
use std::path::PathBuf;

fn node_mlir(model: &str, node: &str) -> Option<String> {
    let p = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/cudaforge/ktir")
        .join(model)
        .join(node);
    std::fs::read_to_string(p).ok()
}

/// Build the arg list for `func`: each pointer-arg's [rows, cols] from its OWN
/// `ktdp.construct_memory_view` sizes (no manifest, no weights), filled with
/// DETERMINISTIC ARBITRARY f16 data (the exact formula from batched_equiv.rs so a
/// bug shows up). Returns the (name, Arg) pairs.
fn build_args(func: &ktir_emulator::ir::IRFunction) -> Vec<(String, Arg)> {
    let mut args: Vec<(String, Arg)> = Vec::new();
    for (arg_name, _) in &func.arguments {
        let shape = func
            .operations
            .iter()
            .find(|op| {
                op.op_type == "ktdp.construct_memory_view"
                    && op.operands.first().map(|s| s.as_str()) == Some(arg_name.as_str())
            })
            .and_then(|op| match op.attributes.get("shape") {
                Some(Attr::IntList(v)) if v.len() == 2 => Some(vec![v[0] as usize, v[1] as usize]),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no view shape for arg {arg_name}"));
        let n = shape[0] * shape[1];
        let seed = arg_name.bytes().map(|b| b as usize).sum::<usize>();
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 7 + seed) % 23) as f32 - 11.0) * 0.03)
            .collect();
        args.push((
            arg_name.trim_start_matches('%').to_string(),
            Arg::Tensor {
                data,
                shape,
                dtype: DType::F16,
            },
        ));
    }
    args
}

/// The pass uses the SAME LX-budget Contract-B predicate as flash_attn. For these
/// real below-cap nodes the re-rolled `[m, cap]` scores tile (≤ 32×64×2 = 4 KiB) is
/// far below LX, so the pass MUST fire. We force a generous budget so the gate is the
/// real one (and would correctly REFUSE on a long-context overflow).
fn never_flash(_scores_bytes: usize) -> bool {
    false
}

fn assert_rewrite_equals_original(model: &str) {
    if ktir_emulator::metal::NaxGemm::new().is_err() {
        eprintln!("no NAX device, skipping {model}");
        return;
    }
    let Some(src) = node_mlir(model, "node111.mlir") else {
        eprintln!("{model}/node111.mlir absent — skipping");
        return;
    };
    let module_orig = parse_module(&src).expect("parse real attention node");
    let (fname, func) = module_orig
        .functions
        .iter()
        .next()
        .map(|(n, f)| (n.clone(), f.clone()))
        .expect("one function");

    // Rewrite a CLONE; the pass MUST recognize and rewrite the real node (== 1).
    let mut module_rw: IRModule = module_orig.clone();
    let rewritten = apply_head_rewrite(&mut module_rw, never_flash);
    assert_eq!(
        rewritten, 1,
        "{model}: head_rewrite must recognize+rewrite the real node111 (got {rewritten})"
    );

    let args = build_args(&func);
    let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

    let orig = execute_function(&module_orig, &fname, &refs).expect("original per-core run");
    let rw = execute_function(&module_rw, &fname, &refs).expect("rewritten per-core run");

    // Compare every read-back tensor element-wise.
    let mut worst = 0.0f32;
    let mut compared = 0usize;
    for (name, o) in &orig {
        let r = rw
            .get(name)
            .unwrap_or_else(|| panic!("{model}: rewritten missing output {name}"));
        assert_eq!(o.data.len(), r.data.len(), "{model}: {name} length");
        let mx = o
            .data
            .iter()
            .zip(&r.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(mx);
        compared += 1;
    }
    eprintln!(
        "{model}/node111: head-rewritten vs original over {compared} tensors, worst max-abs {worst:.5}"
    );
    assert!(compared > 0, "{model}: nothing compared");
    assert!(
        worst < 0.05,
        "{model}: head re-roll diverged from the original unrolled node by {worst} — NOT semantics-preserving"
    );
}

#[test]
#[ignore = "real-IR semantics check; needs the smollm2-135m-prefill node MLIR (no weights). --ignored --nocapture --test-threads=1"]
fn head_rewrite_equals_original_smollm2_135m() {
    assert_rewrite_equals_original("smollm2-135m-prefill");
}

#[test]
#[ignore = "real-IR semantics check; needs the llama-3.2-1b-prefill node MLIR (no weights). --ignored --nocapture --test-threads=1"]
fn head_rewrite_equals_original_llama_3_2_1b() {
    assert_rewrite_equals_original("llama-3.2-1b-prefill");
}

/// HONEST end-to-end WALL-CLOCK: run the program WITH the pass (module_rw) vs WITHOUT
/// (module_orig) through the IDENTICAL `execute_function` path, on the real node,
/// weight-free, arbitrary inputs. best-of-N, release. Report the real ratio — if it
/// is not faster on the real tiny-m emit, the printed number SAYS SO.
#[test]
#[ignore = "TIMING on real IR, no weights. --release --ignored --nocapture --test-threads=1"]
fn time_head_rewrite_vs_original() {
    use std::time::Instant;
    if ktir_emulator::metal::NaxGemm::new().is_err() {
        eprintln!("no NAX device");
        return;
    }
    for model in ["smollm2-135m-prefill", "llama-3.2-1b-prefill"] {
        let Some(src) = node_mlir(model, "node111.mlir") else {
            eprintln!("{model} absent");
            continue;
        };
        let module_orig = parse_module(&src).expect("parse");
        let (fname, func) = module_orig
            .functions
            .iter()
            .next()
            .map(|(n, f)| (n.clone(), f.clone()))
            .unwrap();
        let mut module_rw: IRModule = module_orig.clone();
        let n = apply_head_rewrite(&mut module_rw, never_flash);
        assert_eq!(n, 1, "{model}: pass must fire for timing");

        let args = build_args(&func);
        let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

        let iters = 20u32;
        let best = |f: &mut dyn FnMut()| {
            f();
            let mut b = f64::INFINITY;
            for _ in 0..iters {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64() * 1e3);
            }
            b
        };
        let t_orig = best(&mut || {
            let _ = execute_function(&module_orig, &fname, &refs).unwrap();
        });
        let t_rw = best(&mut || {
            let _ = execute_function(&module_rw, &fname, &refs).unwrap();
        });
        eprintln!(
            "{model}/node111 ({} cores): original {t_orig:.3} ms | head-rewritten {t_rw:.3} ms | {:.2}x",
            func.grid.0 * func.grid.1 * func.grid.2,
            t_orig / t_rw
        );
    }
}
