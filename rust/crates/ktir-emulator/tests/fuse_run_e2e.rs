// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Fuse-then-run end-to-end (fusion increment 2): parse a small two-function
//! KTIR program that threads an intermediate through HBM, run it the unfused
//! (per-node) way to get an oracle, then run `ktir_optimizer::fusion::fuse_program`
//! output through the SAME execution layer and check the results agree.
//!
//! The edge here is a *tiled* one — the consumer reads a contiguous sub-tile of
//! the producer's output inside its access tile — so fusion forwards it as a
//! `tensor.extract_slice` of the producer's resident SSA value (not an HBM
//! round-trip). This exercises the whole increment-2 path: the optimizer emits
//! the slice, the emulator executes it.
//!
//! Note the results are *close*, not bit-identical: the unfused oracle narrows
//! the intermediate to f16 in HBM and back, while the fused path keeps it as an
//! f32 SSA value — so the fused result is the more precise of the two. We assert
//! agreement within an f16 tolerance.
#![cfg(feature = "optimizer")] // exercises ktir_optimizer::fusion::fuse_program

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, Output, execute_function};
use ktir_emulator::ir::IRModule;
use ktir_emulator::parser::parse_module;
use ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec, fuse_program};
use std::collections::HashSet;

const N: usize = 8; // producer tensor length
const TILE: usize = 4; // consumer reads t2[0:TILE]

/// `@a`: out = exp(in) over the whole length-N tensor (a whole-tensor edge on
/// the produce side). `@b`: out = exp(in[0:TILE]) — a contiguous sub-tile read,
/// the tiled edge increment 2 forwards via extract_slice. Both live in one
/// module (multi-function parsing works since the `last_top_level_block` fix).
fn program() -> &'static str {
    r#"
module {
  func.func @a(%in: index, %out: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %vin = ktdp.construct_memory_view %in, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %tin = ktdp.construct_access_tile %vin[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %loaded = ktdp.load %tin : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %y = math.exp %loaded : tensor<8xf16>
    %vout = ktdp.construct_memory_view %out, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %tout = ktdp.construct_access_tile %vout[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    ktdp.store %y, %tout : tensor<8xf16>, !ktdp.access_tile<8xindex>
    return
  }
  func.func @b(%in: index, %out: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %vin = ktdp.construct_memory_view %in, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %tin = ktdp.construct_access_tile %vin[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<4xindex>
    %loaded = ktdp.load %tin : !ktdp.access_tile<4xindex> -> tensor<4xf16>
    %y = math.exp %loaded : tensor<4xf16>
    %vout = ktdp.construct_memory_view %out, sizes: [4], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4xf16>
    %tout = ktdp.construct_access_tile %vout[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<4xf16> -> !ktdp.access_tile<4xindex>
    ktdp.store %y, %tout : tensor<4xf16>, !ktdp.access_tile<4xindex>
    return
  }
}
"#
}

/// a: t1(src) -> t2;  b: t2 -> t3(result).  t2 is the tiled forwarded edge.
fn spec() -> ProgramSpec {
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

fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Arg {
    Arg::Tensor {
        data,
        shape,
        dtype: DType::F16,
    }
}

fn out(map: &std::collections::HashMap<String, Output>, key: &str) -> Vec<f32> {
    map.get(key)
        .unwrap_or_else(|| panic!("missing output {key}"))
        .data
        .clone()
}

#[test]
fn fused_tiled_edge_matches_per_node_oracle() {
    let module = parse_module(program()).expect("parse two-node program");

    // Input t1: small values so exp(exp(.)) stays comfortably in f16 range.
    let t1: Vec<f32> = (0..N).map(|i| i as f32 * 0.1 - 0.3).collect();

    // --- Oracle: run the two nodes unfused, threading t2 through HBM. ---
    let a_out = execute_function(
        &module,
        "a",
        &[
            ("in", tensor(t1.clone(), vec![N])),
            ("out", tensor(vec![0.0; N], vec![N])),
        ],
    )
    .expect("run @a");
    let t2 = out(&a_out, "out");
    let b_out = execute_function(
        &module,
        "b",
        &[
            ("in", tensor(t2, vec![N])),
            ("out", tensor(vec![0.0; TILE], vec![TILE])),
        ],
    )
    .expect("run @b");
    let oracle = out(&b_out, "out");
    assert_eq!(oracle.len(), TILE);

    // --- Fuse, then run the single fused function. ---
    let fused = fuse_program(&module, &spec()).expect("fuse");

    // Structural: the tiled edge became an extract_slice; the intermediate's
    // store/load round-trip is gone (only the source load + result store remain).
    let count = |ty: &str| fused.operations.iter().filter(|o| o.op_type == ty).count();
    assert_eq!(
        count("tensor.extract_slice"),
        1,
        "tiled edge forwarded as a slice"
    );
    assert_eq!(count("ktdp.load"), 1, "only the source (t1) load remains");
    assert_eq!(count("ktdp.store"), 1, "only the result (t3) store remains");
    let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        arg_names,
        vec!["%t1_ptr", "%t3_ptr"],
        "no HBM pointer for t2"
    );

    let mut fused_module = IRModule::default();
    fused_module.add_function(fused);
    let f_out = execute_function(
        &fused_module,
        "fused",
        &[
            ("t1_ptr", tensor(t1, vec![N])),
            ("t3_ptr", tensor(vec![0.0; TILE], vec![TILE])),
        ],
    )
    .expect("run fused");
    let fused_res = out(&f_out, "t3_ptr");
    assert_eq!(fused_res.len(), TILE);

    // Agree within f16 tolerance (the fused path skips one f16 narrowing of the
    // intermediate, so it is the more precise of the two — not bit-identical).
    let max_diff = oracle
        .iter()
        .zip(&fused_res)
        .map(|(o, f)| (o - f).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-2,
        "fused vs per-node oracle disagree by {max_diff} (oracle={oracle:?}, fused={fused_res:?})"
    );
    eprintln!("fuse-then-run: tiled edge forwarded via extract_slice; max diff {max_diff:.5} ✓");
}
