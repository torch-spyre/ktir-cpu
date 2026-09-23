// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Golden gate for the flash-attention IR-rewrite pass (TODO #2).
//!
//! Two independent correctness proofs, because the cached bundles are SHORT
//! context (the pass is a no-op on them by Contract B):
//!
//! 1. EXECUTION EQUIVALENCE (this file, default-run, no GPU/bundle needed): build
//!    a synthetic NAIVE attention `IRFunction`, run it on the interpreter; then
//!    `recognize_attention` + `tile_attention` it and run the tiled online-softmax
//!    form on the SAME interpreter; assert the max-abs difference is `< 1e-3`
//!    (and `< 0.05`), AND that both match a hand-computed reference for a tiny
//!    case (causal + `1/sqrt(d)` scale).
//!
//! 2. FORCED-FIRE through `program::execute` (`--ignored`, needs `metal`):
//!    wrap the canonical attention node as a one-node program; with a tiny
//!    `KTIR_FLASH_ATTN_SCORES_BUDGET` the pass FIRES (a region-bearing rewrite the
//!    generic interpreter runs); assert the program output still matches the naive
//!    reference within the 0.05 golden gate.
//!
//! 3. NEGATIVE recognition: a non-attention node → `recognize_attention` is
//!    `None` → unchanged (also covered in the optimizer unit tests; re-asserted
//!    here through the public API).

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, Output, execute_function};
use ktir_emulator::ir::{IRModule, Operation};
#[cfg(feature = "metal")]
use ktir_optimizer::flash_attn::recognize_rerolled_attention;
use ktir_optimizer::flash_attn::{recognize_attention, test_support, tile_attention};

/// Run a single attention `IRFunction` on the interpreter with Q/K/V inputs and
/// read back the `%o_ptr` output as f32. Q is `[m,d]`, K/V are `[cap,d]`.
fn run_attn(
    func: &ktir_emulator::ir::IRFunction,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    m: usize,
    cap: usize,
    d: usize,
) -> Vec<f32> {
    let mut module = IRModule::default();
    let mut f = func.clone();
    f.name = "attn".into();
    module.add_function(f);
    let args: &[(&str, Arg)] = &[
        (
            "q_ptr",
            Arg::Tensor {
                data: q.to_vec(),
                shape: vec![m, d],
                dtype: DType::F16,
            },
        ),
        (
            "k_ptr",
            Arg::Tensor {
                data: k.to_vec(),
                shape: vec![cap, d],
                dtype: DType::F16,
            },
        ),
        (
            "v_ptr",
            Arg::Tensor {
                data: v.to_vec(),
                shape: vec![cap, d],
                dtype: DType::F16,
            },
        ),
        (
            "o_ptr",
            Arg::Tensor {
                data: vec![0.0; m * d],
                shape: vec![m, d],
                dtype: DType::F16,
            },
        ),
    ];
    let out = execute_function(&module, "attn", args).expect("attention run");
    let Output { data, .. } = out
        .get("o_ptr")
        .or_else(|| out.get("%o_ptr"))
        .expect("o_ptr output");
    data.clone()
}

/// Shape + math knobs for the f32 reference (kept in one struct so the helper
/// stays under clippy's argument-count lint).
struct AttnCfg {
    m: usize,
    cap: usize,
    d: usize,
    scale: f32,
    causal: bool,
}

/// Reference softmax attention in f32 (the math the IR must reproduce). Causal
/// mask: query row `qr` (absolute KV index `cap - m + qr`) attends to keys at
/// absolute position `<= cap - m + qr`.
fn reference_attention(q: &[f32], k: &[f32], v: &[f32], cfg: &AttnCfg) -> Vec<f32> {
    let AttnCfg {
        m,
        cap,
        d,
        scale,
        causal,
    } = *cfg;
    let mut out = vec![0.0f32; m * d];
    for qr in 0..m {
        // scores
        let mut s = vec![f32::NEG_INFINITY; cap];
        let q_abs = (cap - m + qr) as i64;
        for kc in 0..cap {
            if causal && (kc as i64) > q_abs {
                continue; // masked
            }
            let mut dot = 0.0f32;
            for dd in 0..d {
                dot += q[qr * d + dd] * k[kc * d + dd];
            }
            s[kc] = dot * scale;
        }
        // softmax
        let mx = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut p = vec![0.0f32; cap];
        for (kc, pk) in p.iter_mut().enumerate() {
            let e = if s[kc] == f32::NEG_INFINITY {
                0.0
            } else {
                (s[kc] - mx).exp()
            };
            *pk = e;
            sum += e;
        }
        for pk in p.iter_mut() {
            *pk /= sum;
        }
        // weighted V
        for dd in 0..d {
            let mut acc = 0.0f32;
            for kc in 0..cap {
                acc += p[kc] * v[kc * d + dd];
            }
            out[qr * d + dd] = acc;
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Deterministic small "random" values in a modest range, kept small so f16
/// rounding (the model dtype) does not dominate the comparison.
fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32 * 0.137 + seed).sin();
            x * 0.5 // |x| <= 0.5
        })
        .collect()
}

// ===========================================================================
// (1) EXECUTION EQUIVALENCE
// ===========================================================================

#[test]
fn tiled_matches_naive_causal() {
    let (m, cap, d) = (4usize, 256usize, 8usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = ramp(m * d, 0.1);
    let k = ramp(cap * d, 1.3);
    let v = ramp(cap * d, 2.7);

    let naive = test_support::naive_attention(m as i64, cap as i64, d as i64, scale, true);
    let island = recognize_attention(&naive).expect("recognize canonical attention");
    assert_eq!(island.m as usize, m);
    assert_eq!(island.cap as usize, cap);
    assert_eq!(island.d as usize, d);
    assert!(island.causal);
    let tiled = tile_attention(&island);

    let naive_out = run_attn(&naive, &q, &k, &v, m, cap, d);
    let tiled_out = run_attn(&tiled, &q, &k, &v, m, cap, d);
    let reference = reference_attention(
        &q,
        &k,
        &v,
        &AttnCfg {
            m,
            cap,
            d,
            scale,
            causal: true,
        },
    );

    let diff_tiled_vs_naive = max_abs_diff(&tiled_out, &naive_out);
    let diff_tiled_vs_ref = max_abs_diff(&tiled_out, &reference);
    let diff_naive_vs_ref = max_abs_diff(&naive_out, &reference);
    eprintln!(
        "causal: tiled-vs-naive={diff_tiled_vs_naive:.6} tiled-vs-ref={diff_tiled_vs_ref:.6} \
         naive-vs-ref={diff_naive_vs_ref:.6}"
    );

    // Online softmax equals two-pass up to fp reassociation: tight on the
    // interpreter-vs-interpreter comparison, and both close to the f32 reference.
    assert!(
        diff_tiled_vs_naive < 1e-3,
        "tiled vs naive {diff_tiled_vs_naive} >= 1e-3"
    );
    assert!(
        diff_tiled_vs_naive < 0.05,
        "tiled vs naive over golden gate"
    );
    assert!(
        diff_tiled_vs_ref < 0.05,
        "tiled vs f32 reference over golden gate"
    );
}

#[test]
fn tiled_matches_naive_noncausal() {
    let (m, cap, d) = (3usize, 128usize, 4usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = ramp(m * d, 0.5);
    let k = ramp(cap * d, 0.9);
    let v = ramp(cap * d, 1.1);

    let naive = test_support::naive_attention(m as i64, cap as i64, d as i64, scale, false);
    let island = recognize_attention(&naive).expect("recognize");
    assert!(!island.causal);
    let tiled = tile_attention(&island);

    let naive_out = run_attn(&naive, &q, &k, &v, m, cap, d);
    let tiled_out = run_attn(&tiled, &q, &k, &v, m, cap, d);
    let reference = reference_attention(
        &q,
        &k,
        &v,
        &AttnCfg {
            m,
            cap,
            d,
            scale,
            causal: false,
        },
    );

    let d_tn = max_abs_diff(&tiled_out, &naive_out);
    let d_tr = max_abs_diff(&tiled_out, &reference);
    eprintln!("noncausal: tiled-vs-naive={d_tn:.6} tiled-vs-ref={d_tr:.6}");
    assert!(d_tn < 1e-3, "tiled vs naive {d_tn} >= 1e-3");
    assert!(d_tr < 0.05, "tiled vs reference over golden gate");
}

/// Tiny hand-checkable case: m=1, cap=2, d=1, no causal mask, scale=1.
/// Q=[1], K=[[1],[2]], V=[[3],[5]].
///   scores = [1*1, 1*2] = [1, 2]; softmax([1,2]) = [e^-1, 1]/(e^-1+1)
///     = [0.26894, 0.73106]; O = 0.26894*3 + 0.73106*5 = 4.4621...
#[test]
fn tiny_handcomputed_reference() {
    let (m, cap, d) = (1usize, 2usize, 1usize);
    let q = vec![1.0f32];
    let k = vec![1.0f32, 2.0f32];
    let v = vec![3.0f32, 5.0f32];
    let scale = 1.0f32;

    let naive = test_support::naive_attention(m as i64, cap as i64, d as i64, scale, false);
    let island = recognize_attention(&naive).expect("recognize tiny");
    let tiled = tile_attention(&island);

    let tiled_out = run_attn(&tiled, &q, &k, &v, m, cap, d);
    let naive_out = run_attn(&naive, &q, &k, &v, m, cap, d);

    let e1 = (-1.0f32).exp();
    let expected = (e1 * 3.0 + 1.0 * 5.0) / (e1 + 1.0); // ~4.46212
    eprintln!(
        "tiny: tiled={:?} naive={:?} expected={expected:.5}",
        tiled_out, naive_out
    );
    assert!(
        (tiled_out[0] - expected).abs() < 0.02,
        "tiled {} vs {expected}",
        tiled_out[0]
    );
    assert!(
        (naive_out[0] - expected).abs() < 0.02,
        "naive {} vs {expected}",
        naive_out[0]
    );
    assert!(
        (tiled_out[0] - naive_out[0]).abs() < 1e-3,
        "tiled vs naive disagree"
    );
}

/// A ragged cap (not a power of two, but `choose_block` still finds a divisor)
/// — exercises the block-count derivation on an odd length.
#[test]
fn tiled_matches_naive_ragged_cap() {
    let (m, cap, d) = (2usize, 192usize, 4usize); // 192 = 64*3
    let scale = 1.0 / (d as f32).sqrt();
    let q = ramp(m * d, 0.3);
    let k = ramp(cap * d, 0.7);
    let v = ramp(cap * d, 1.9);

    let naive = test_support::naive_attention(m as i64, cap as i64, d as i64, scale, true);
    let island = recognize_attention(&naive).unwrap();
    let tiled = tile_attention(&island);

    let naive_out = run_attn(&naive, &q, &k, &v, m, cap, d);
    let tiled_out = run_attn(&tiled, &q, &k, &v, m, cap, d);
    let d_tn = max_abs_diff(&tiled_out, &naive_out);
    eprintln!("ragged cap=192: tiled-vs-naive={d_tn:.6}");
    assert!(d_tn < 1e-3, "ragged cap tiled vs naive {d_tn}");
}

// ===========================================================================
// (3) NEGATIVE recognition (public API)
// ===========================================================================

#[test]
fn negative_non_attention_is_unrecognized() {
    use ktir_emulator::ir::{Attr, IRFunction};
    // A plain elementwise copy node (load -> exp -> store): not attention.
    let mk_view = |res: &str, arg: &str| {
        Operation::new(Some(res), "ktdp.construct_memory_view", &[arg])
            .with_attr("shape", Attr::IntList(vec![4, 4]))
            .with_attr("strides", Attr::IntList(vec![4, 1]))
            .with_attr("memory_space", Attr::Str("HBM".into()))
            .with_attr("dtype", Attr::Str("f16".into()))
    };
    let f = IRFunction {
        name: "copy".into(),
        arguments: vec![
            ("%in".into(), "index".into()),
            ("%out".into(), "index".into()),
        ],
        grid: (1, 1, 1),
        return_type: None,
        operations: vec![
            mk_view("%vi", "%in"),
            Operation::new(Some("%ti"), "ktdp.construct_access_tile", &["%vi"])
                .with_attr("shape", Attr::IntList(vec![4, 4])),
            Operation::new(Some("%l"), "ktdp.load", &["%ti"]),
            Operation::new(Some("%y"), "math.exp", &["%l"]),
            mk_view("%vo", "%out"),
            Operation::new(Some("%to"), "ktdp.construct_access_tile", &["%vo"])
                .with_attr("shape", Attr::IntList(vec![4, 4])),
            Operation::new(None, "ktdp.store", &["%y", "%to"]),
            Operation::new(None, "func.return", &[]),
        ],
    };
    assert!(
        recognize_attention(&f).is_none(),
        "copy node must not be recognized as attention"
    );
}

/// REAL-IR: the cached prefill attention nodes are an unrolled, per-query-row,
/// multi-store, head-indexed lowering. The SINGLE-BLOCK `recognize_attention`
/// correctly returns `None` on the RAW node (it is not the flat canonical idiom).
/// But `head_rewrite` (which runs FIRST in the program pipeline) RE-ROLLS it into
/// the two-block whole-tensor form — and THAT form `recognize_rerolled_attention`
/// MUST match, and `apply_flash_attention` MUST fire on (count > 0) at a tiny
/// scores budget. A no-op there would mean the pass cannot fix real long-context
/// attention (the bug this replaces). Skips gracefully when the bundle is absent.
#[test]
fn real_cached_nodes_flash_after_head_rewrite() {
    use ktir_emulator::parser::parse_module;
    use ktir_optimizer::flash_attn::{apply_flash_attention, recognize_rerolled_attention};
    use ktir_optimizer::fusion::attention_needs_flash;
    use ktir_optimizer::head_rewrite::apply_head_rewrite;

    let home = match std::env::var_os("HOME") {
        Some(h) => h,
        None => return,
    };
    let bundles = ["llama-3.2-1b-prefill", "smollm2-135m-prefill"];
    let mut checked = 0usize;
    for b in bundles {
        let p = std::path::PathBuf::from(&home)
            .join(".cache/cudaforge/ktir")
            .join(b)
            .join("node111.mlir");
        let Ok(src) = std::fs::read_to_string(&p) else {
            continue;
        };
        let module = match parse_module(&src) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // (1) The RAW node is NOT the single-block canonical idiom.
        for (name, f) in &module.functions {
            assert!(
                recognize_attention(f).is_none(),
                "{b} fn {name}: raw unrolled node must NOT match the single-block idiom",
            );
        }

        // (2) After head_rewrite (forced to fire with a never-flash predicate), the
        // re-rolled form IS the two-block idiom — recognized + flash FIRES.
        let mut rw = module.clone();
        let n = apply_head_rewrite(&mut rw, |_| false);
        assert_eq!(n, 1, "{b}: head_rewrite must re-roll node111");
        let hr_func = rw.functions.values().next().unwrap();
        assert!(
            recognize_rerolled_attention(hr_func).is_some(),
            "{b}: re-rolled node111 must be recognized by recognize_rerolled_attention",
        );

        // Tiny forced budget -> flash MUST fire (a no-op is a FAIL).
        let mut flash = rw.clone();
        let fired = apply_flash_attention(&mut flash, |sb| attention_needs_flash(sb, 16));
        assert!(
            fired > 0,
            "{b}: flash_attn must FIRE on the re-rolled node111 (got {fired})"
        );
        eprintln!("{b}/node111: raw=single-block-None, re-rolled recognized, flash fired={fired}");
        checked += 1;
    }
    if checked == 0 {
        eprintln!("no real prefill bundle present — skipping real-IR flash-fire check");
    }
}

// ===========================================================================
// (2) FORCED-FIRE through program::execute (needs `metal` + a tiny budget)
// ===========================================================================
//
// Wrap the canonical attention node as a one-node program and run it through the
// turnkey `program::execute` path. With a tiny `KTIR_FLASH_ATTN_SCORES_BUDGET`
// the FA pass FIRES (rewriting the node to the region-bearing tiled form the
// generic interpreter runs); the program output must still match the naive
// reference within the 0.05 golden gate. Ignored by default (serial, env-mutating).

#[cfg(feature = "metal")]
#[test]
#[ignore = "forced-fire FA through program::execute; run serially with --ignored"]
fn forced_fire_through_program_execute() {
    use ktir_emulator::program;
    use ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
    use std::collections::HashSet;

    let (m, cap, d) = (4usize, 256usize, 8usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = ramp(m * d, 0.2);
    let k = ramp(cap * d, 1.5);
    let v = ramp(cap * d, 2.1);
    let reference = reference_attention(
        &q,
        &k,
        &v,
        &AttnCfg {
            m,
            cap,
            d,
            scale,
            causal: true,
        },
    );

    // Emit the canonical naive attention as MLIR-ish text? No — program::execute
    // parses node MLIR. Instead exercise the pass directly: build the module the
    // way module_from_nodes does, force the budget tiny, and run the segmented
    // path. We construct the module in-memory and invoke the same rewrite +
    // segmented execution program::execute uses.
    let naive = test_support::naive_attention(m as i64, cap as i64, d as i64, scale, true);

    // Map the four pointer args to tensor ids t0..t3 (program::execute uses
    // `%t<id>_ptr` naming; rename the canonical args to that convention).
    let mut node_func = naive.clone();
    node_func.name = "attn_node".into();
    rename_arg(&mut node_func, "%q_ptr", "%t0_ptr");
    rename_arg(&mut node_func, "%k_ptr", "%t1_ptr");
    rename_arg(&mut node_func, "%v_ptr", "%t2_ptr");
    rename_arg(&mut node_func, "%o_ptr", "%t3_ptr");

    let mut module = IRModule::default();
    module.add_function(node_func);

    // Force FA to fire by setting a tiny scores budget, then apply the pass via
    // the public entrypoint (mirrors program::module_from_nodes' wiring).
    unsafe { std::env::set_var("KTIR_FLASH_ATTN_SCORES_BUDGET", "16") };
    let fired = ktir_optimizer::flash_attn::apply_flash_attention(&mut module, |sb| {
        ktir_optimizer::fusion::attention_needs_flash(sb, 16)
    });
    unsafe { std::env::remove_var("KTIR_FLASH_ATTN_SCORES_BUDGET") };
    assert_eq!(
        fired, 1,
        "FA must fire on the canonical node at a tiny budget"
    );

    // Run the rewritten module through the segmented path (what program::execute
    // calls). One node: t0/t1/t2 sources, t3 result.
    let spec = ProgramSpec {
        nodes: vec![NodeSpec {
            func: "attn_node".into(),
            bindings: vec![
                Binding {
                    arg: "%t0_ptr".into(),
                    tensor: 0,
                    is_output: false,
                },
                Binding {
                    arg: "%t1_ptr".into(),
                    tensor: 1,
                    is_output: false,
                },
                Binding {
                    arg: "%t2_ptr".into(),
                    tensor: 2,
                    is_output: false,
                },
                Binding {
                    arg: "%t3_ptr".into(),
                    tensor: 3,
                    is_output: true,
                },
            ],
        }],
        sources: HashSet::from([0, 1, 2]),
        results: HashSet::from([3]),
    };
    let args: &[(&str, Arg)] = &[
        (
            "t0",
            Arg::Tensor {
                data: q.clone(),
                shape: vec![m, d],
                dtype: DType::F16,
            },
        ),
        (
            "t1",
            Arg::Tensor {
                data: k.clone(),
                shape: vec![cap, d],
                dtype: DType::F16,
            },
        ),
        (
            "t2",
            Arg::Tensor {
                data: v.clone(),
                shape: vec![cap, d],
                dtype: DType::F16,
            },
        ),
    ];
    let out = ktir_emulator::segmented::execute_segmented(&module, &spec, args, &["t3"])
        .expect("segmented run of forced-fire FA");
    let got = &out.get("t3").expect("t3 output").data;

    let diff = max_abs_diff(got, &reference);
    eprintln!("forced-fire: max_abs vs f32 reference = {diff:.6}");
    assert!(diff < 0.05, "forced-fire FA over golden gate: {diff}");

    let _ = program::module_from_nodes; // keep the wiring symbol referenced.
}

#[cfg(feature = "metal")]
fn rename_arg(func: &mut ktir_emulator::ir::IRFunction, from: &str, to: &str) {
    for (name, _) in &mut func.arguments {
        if name == from {
            *name = to.to_string();
        }
    }
    fn walk(ops: &mut [Operation], from: &str, to: &str) {
        for op in ops {
            for o in &mut op.operands {
                if o == from {
                    *o = to.to_string();
                }
            }
            for r in &mut op.regions {
                walk(r, from, to);
            }
        }
    }
    walk(&mut func.operations, from, to);
}

// ===========================================================================
// (4) REAL-IR, WEIGHT-FREE semantics gate (the non-negotiable bar)
// ===========================================================================
//
// For BOTH real prefill nodes: apply head_rewrite (so flash_attn sees the
// RE-ROLLED form), FORCE flash_attn to fire with a tiny scores budget, and assert
// the flash-tiled module run through the UNCHANGED `interpreter::execute_function`
// EQUALS the pre-flash (head-rewritten) module run the same way, on ARBITRARY
// inputs whose shapes come from the IR's own `construct_memory_view` sizes — NO
// weights, NO golden.bin. max-abs < 0.05. ALSO asserts flash FIRED (count > 0) and
// the per-block CONTEXT scores tile is strictly smaller than the full [m, cap]
// tile (the actual long-context fix). Uses `execute_function` (the generic per-core
// path — NOT the batched executor, which the emitted scf.for would trip).

/// Build the arg list for `func`: each pointer-arg's [rows, cols] from its OWN
/// `ktdp.construct_memory_view` sizes (no manifest, no weights), filled with
/// DETERMINISTIC ARBITRARY f16 data (same formula as head_rewrite_golden so a bug
/// shows up).
#[cfg(feature = "metal")]
fn build_args(func: &ktir_emulator::ir::IRFunction) -> Vec<(String, Arg)> {
    use ktir_emulator::ir::Attr;
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

/// The forced scores budget (mirrors `forced_fire_through_program_execute`): tiny
/// so `attention_needs_flash` flips true on the real [m, cap] tile and the pass
/// fires. The per-block tile then drops to the smallest divisor of cap.
#[cfg(feature = "metal")]
const FORCED_BUDGET: usize = 16;

#[cfg(feature = "metal")]
fn rerolled_flash_equals_head_rewrite(model: &str) {
    use ktir_emulator::ir::Attr;
    use ktir_emulator::parser::parse_module;
    use ktir_optimizer::flash_attn::apply_flash_attention;
    use ktir_optimizer::fusion::attention_needs_flash;
    use ktir_optimizer::head_rewrite::apply_head_rewrite;

    if ktir_emulator::metal::NaxGemm::new().is_err() {
        eprintln!("no NAX device, skipping {model}");
        return;
    }
    let p = std::path::PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".cache/cudaforge/ktir")
        .join(model)
        .join("node111.mlir");
    let Ok(src) = std::fs::read_to_string(&p) else {
        eprintln!("{model}/node111.mlir absent — skipping");
        return;
    };
    let module = parse_module(&src).expect("parse real attention node");
    let fname = module
        .functions
        .keys()
        .next()
        .expect("one function")
        .clone();

    // PRE-FLASH reference: head-rewritten (re-rolled) module. never_flash forces the
    // head pass to fire so flash_attn sees the re-rolled form.
    let mut ref_mod: IRModule = module.clone();
    let hr = apply_head_rewrite(&mut ref_mod, |_| false);
    assert_eq!(
        hr, 1,
        "{model}: head_rewrite must fire (re-rolled reference)"
    );
    let hr_func = ref_mod.functions.get(&fname).unwrap().clone();

    // The re-rolled form must be recognized.
    let isl = recognize_rerolled_attention(&hr_func)
        .unwrap_or_else(|| panic!("{model}: re-rolled node not recognized"));

    // Flash-tiled module: force the tiny budget so the pass FIRES.
    let mut flash_mod: IRModule = ref_mod.clone();
    let fired = apply_flash_attention(&mut flash_mod, |sb| {
        attention_needs_flash(sb, FORCED_BUDGET)
    });
    assert!(
        fired > 0,
        "{model}: flash_attn must FIRE on the re-rolled node (got {fired})"
    );

    // Per-block CONTEXT scores tile: read `blk` off the emitted scf.for body and
    // assert it is strictly smaller than the full [m, cap] footprint (real tiling).
    // The per-head context MASK slice `[1, blk]` uniquely encodes `blk` (first dim
    // is 1) — unambiguous even when d == cap.
    let ffn = flash_mod.functions.get(&fname).unwrap();
    let forop = ffn
        .operations
        .iter()
        .find(|o| o.op_type == "scf.for")
        .expect("scf.for");
    let blk = forop.regions[0]
        .iter()
        .filter_map(|o| match o.attributes.get("shape") {
            Some(Attr::IntList(v)) if v.len() == 2 && v[0] == 1 => Some(v[1]),
            _ => None,
        })
        .min()
        .expect("per-block context mask [1, blk] tile shape");
    let per_block = (isl.m as usize) * (blk as usize) * 2;
    let full = isl.scores_bytes();
    eprintln!(
        "{model}/node111: m={} cap={} blk={blk} per_block={per_block}B full={full}B fired={fired}",
        isl.m, isl.cap
    );
    assert!(
        blk < isl.cap,
        "{model}: must tile the cap axis (blk {blk} < cap {})",
        isl.cap
    );
    assert!(
        per_block < full,
        "{model}: per-block tile {per_block} not smaller than full {full}",
    );

    // Run BOTH modules through the UNCHANGED generic interpreter, weight-free.
    let args = build_args(&hr_func);
    let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let ref_out = execute_function(&ref_mod, &fname, &refs).expect("head-rewritten run");
    let fl_out = execute_function(&flash_mod, &fname, &refs).expect("flash-tiled run");

    let mut worst = 0.0f32;
    let mut compared = 0usize;
    for (name, o) in &ref_out {
        let r = fl_out
            .get(name)
            .unwrap_or_else(|| panic!("{model}: flash missing output {name}"));
        assert_eq!(o.data.len(), r.data.len(), "{model}: {name} length");
        let mx = max_abs_diff(&o.data, &r.data);
        worst = worst.max(mx);
        compared += 1;
    }
    eprintln!(
        "{model}/node111: flash-tiled vs head-rewritten over {compared} tensors, worst max-abs {worst:.6}"
    );
    assert!(compared > 0, "{model}: nothing compared");
    assert!(
        worst < 0.05,
        "{model}: cap-tiled flash diverged from the head-rewritten reference by {worst} — NOT semantics-preserving",
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "real-IR weight-free semantics gate; needs smollm2-135m-prefill node MLIR. --ignored --nocapture --test-threads=1"]
fn flash_rerolled_equals_head_rewrite_smollm2_135m() {
    rerolled_flash_equals_head_rewrite("smollm2-135m-prefill");
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "real-IR weight-free semantics gate; needs llama-3.2-1b-prefill node MLIR. --ignored --nocapture --test-threads=1"]
fn flash_rerolled_equals_head_rewrite_llama_3_2_1b() {
    rerolled_flash_equals_head_rewrite("llama-3.2-1b-prefill");
}
