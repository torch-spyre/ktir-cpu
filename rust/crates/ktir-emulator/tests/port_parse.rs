// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_dialects_parse.py` — op-parsing assertions, exercised
//! through the crate's module parser (`ktir_emulator::parser::parse_module`).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * The Python test harness parses a *single op* via `KTIRParser()
//!   ._parse_operations`. The Rust crate exposes only the module-level
//!   `parse_module`, so each single-op case is wrapped in a one-function module
//!   by [`parse_op`] / [`parse_ops`] and the op is pulled back out (skipping the
//!   trailing `return`).
//! * Rust `Operation.result` is a single `Option<String>` (the first result
//!   name), not Python's `str | list`. Multi-result `result` shape assertions
//!   therefore have no faithful analogue and are skipped (see integrator notes).
//! * The Rust parser is the *structural + ktdp* slice: it extracts
//!   `op_type` / `operands` / `result_type` generically for every op, but only
//!   fills the `attributes` map for `arith.constant`,
//!   `ktdp.construct_memory_view`, and `ktdp.construct_access_tile`. Dialect
//!   attribute assertions for cmpi/cmpf/linalg/tensor/scf/math (predicate, dim,
//!   reduce_fn, shape, iter_var, ...) are not produced by this slice and are
//!   skipped; their op_type/operand structure IS checked here (faithful to what
//!   the slice produces).

use ktir_emulator::ir::{Attr, Operation};
use ktir_emulator::parser::parse_module;

/// Wrap a single op in a one-function module, parse it, and return every
/// non-`return` operation. The Python harness parsed bare op text; the Rust
/// parser is module-scoped, so we synthesise the enclosing `func.func`.
fn parse_ops(op_text: &str) -> Vec<Operation> {
    let src = format!("module {{\n  func.func @f() {{\n    {op_text}\n    return\n  }}\n}}");
    let module = parse_module(&src).unwrap_or_else(|e| panic!("parse failed for {op_text:?}: {e}"));
    let f = module.get_function("f").expect("function f");
    f.operations
        .iter()
        .filter(|o| !o.op_type.ends_with("return"))
        .cloned()
        .collect()
}

/// Convenience: parse a single op and return exactly that op.
fn parse_op(op_text: &str) -> Operation {
    let ops = parse_ops(op_text);
    assert_eq!(
        ops.len(),
        1,
        "expected exactly one op from {op_text:?}, got {ops:?}"
    );
    ops.into_iter().next().unwrap()
}

/// Assert an op declares the given operand names, in any position (mirrors the
/// Python `assert_operand_names`, which is the regex-parser-specific check).
fn assert_operand_names(op: &Operation, names: &[&str]) {
    for n in names {
        assert!(
            op.operands.iter().any(|o| o == n),
            "operand {n:?} not found in {:?}",
            op.operands
        );
    }
}

// ===========================================================================
// module-level parser (TestModuleParser)
// ===========================================================================

#[test]
fn parser_basic() {
    // Minimal module with a grid attribute parses; grid and op count check out.
    let module = parse_module(
        r#"
        module {
            func.func @test_func() -> index attributes { grid = [32, 1, 1] } {
                %c0 = arith.constant 0 : index
                %grid0 = ktdp.get_compute_tile_id : index
                return %c0 : index
            }
        }
        "#,
    )
    .unwrap();
    assert!(module.functions.contains_key("test_func"));
    let f = module.get_function("test_func").unwrap();
    assert_eq!(f.grid, (32, 1, 1));
    // arith.constant, ktdp.get_compute_tile_id, return — >= 2 real ops.
    assert!(f.operations.len() >= 2);
}

#[test]
fn parser_attributes_body() {
    // Function with arguments, a 2-D grid, and ktdp ops all parse.
    let module = parse_module(
        r#"
        module {
          func.func @add(%a: index, %b: index, %c: index) -> index attributes { grid = [4, 4] } {
            %c0 = arith.constant 0 : index
            %grid0 = ktdp.get_compute_tile_id : index
            %acc = ktdp.construct_access_tile %ref[%c0, %c0] : memref<128x256xf16> -> !ktdp.access_tile<128x256xindex>
            %tile = ktdp.load %acc : !ktdp.access_tile<128x256xindex> -> tensor<128x256xf16>
            %out_acc = ktdp.construct_access_tile %out_ref[%c0, %c0] : memref<128x256xf16> -> !ktdp.access_tile<128x256xindex>
            ktdp.store %tile, %out_acc : tensor<128x256xf16>, !ktdp.access_tile<128x256xindex>
            return %c0 : index
          }
        }
        "#,
    )
    .unwrap();
    let f = module.get_function("add").unwrap();
    assert_eq!(f.grid, (4, 4, 1));
    assert_eq!(f.arguments.len(), 3);
    let op_types: Vec<&str> = f.operations.iter().map(|o| o.op_type.as_str()).collect();
    for expected in [
        "arith.constant",
        "ktdp.get_compute_tile_id",
        "ktdp.construct_access_tile",
        "ktdp.load",
        "ktdp.store",
    ] {
        assert!(op_types.contains(&expected), "missing op {expected}");
    }
}

#[test]
fn parser_no_attributes() {
    // Function without attributes defaults to grid (1,1,1).
    let module = parse_module(
        r#"
        module {
          func.func @simple() -> index {
            %c0 = arith.constant 0 : index
            return %c0 : index
          }
        }
        "#,
    )
    .unwrap();
    let f = module.get_function("simple").unwrap();
    assert_eq!(f.grid, (1, 1, 1));
    assert!(f.operations.len() >= 2);
}

#[test]
fn parser_1d_grid() {
    // grid = [X] — single element; Y and Z default to 1.
    let module = parse_module(
        r#"
        module {
          func.func @single() attributes { grid = [4] } {
            return
          }
        }
        "#,
    )
    .unwrap();
    assert_eq!(module.get_function("single").unwrap().grid, (4, 1, 1));
}

#[test]
fn parser_2d_grid() {
    // grid = [8, 4] — Z defaults to 1 (the second half of Python's
    // multiple-functions test, kept single-function since the Rust parser
    // extracts one function per module; see integrator notes).
    let module = parse_module(
        r#"
        module {
          func.func @first() attributes { grid = [8, 4] } {
            %c0 = arith.constant 0 : index
            return
          }
        }
        "#,
    )
    .unwrap();
    assert_eq!(module.get_function("first").unwrap().grid, (8, 4, 1));
}

// ===========================================================================
// arith dialect (TestArithParsers)
// ===========================================================================

#[test]
fn constant_scalar() {
    let op = parse_op("%c0 = arith.constant 42 : index");
    assert_eq!(op.op_type, "arith.constant");
    assert_eq!(op.attributes.get("value"), Some(&Attr::Int(42)));
}

#[test]
fn constant_hex_integer() {
    // 0xFF800000 — the regex/Rust parser returns the unsigned value 4286578688
    // (same bit pattern as signed i32 -8388608).
    let op = parse_op("%x = arith.constant 0xFF800000 : i32");
    assert_eq!(op.op_type, "arith.constant");
    match op.attributes.get("value") {
        Some(&Attr::Int(v)) => assert!(v == 0xFF80_0000 || v == -8_388_608),
        other => panic!("unexpected value attr: {other:?}"),
    }
}

#[test]
fn constant_float() {
    let op = parse_op("%x = arith.constant 0.0 : f32");
    assert_eq!(op.op_type, "arith.constant");
    assert_eq!(op.attributes.get("value"), Some(&Attr::Float(0.0)));
}

#[test]
fn constant_dense_tensor() {
    // dense<0.0> splat: the Rust slice records the splat scalar as `value`.
    let op = parse_op("%t = arith.constant dense<0.0> : tensor<4xf16>");
    assert_eq!(op.op_type, "arith.constant");
    assert_eq!(op.attributes.get("value"), Some(&Attr::Float(0.0)));
    assert_eq!(op.result_type.as_deref(), Some("tensor<4xf16>"));
}

#[test]
fn int_binops_structure() {
    // Every int binary op parses to op_type + two operands (no special attrs).
    for name in [
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.divui",
        "arith.remsi",
        "arith.remui",
        "arith.ceildivsi",
        "arith.floordivsi",
        "arith.minsi",
        "arith.maxsi",
        "arith.minui",
        "arith.maxui",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrsi",
        "arith.shrui",
        "arith.ceildivui",
    ] {
        let op = parse_op(&format!("%r = {name} %a, %b : i32"));
        assert_eq!(op.op_type, name);
        assert_eq!(op.operands.len(), 2, "op {name}");
        assert_operand_names(&op, &["%a", "%b"]);
    }
}

#[test]
fn int_casts_structure() {
    // One-operand casts; the operand is recorded (result_type carries " to ...").
    let cases = [
        ("arith.extsi", "i16", "i32"),
        ("arith.extui", "i16", "i32"),
        ("arith.trunci", "i32", "i16"),
        ("arith.fptosi", "f32", "i32"),
        ("arith.fptoui", "f32", "i32"),
        ("arith.uitofp", "i32", "f32"),
        ("arith.index_cast", "i32", "index"),
        ("arith.index_castui", "i32", "index"),
    ];
    for (name, src, dst) in cases {
        let op = parse_op(&format!("%r = {name} %a : {src} to {dst}"));
        assert_eq!(op.op_type, name);
        assert_eq!(op.operands.len(), 1, "op {name}");
        assert_operand_names(&op, &["%a"]);
    }
}

#[test]
fn select_structure() {
    let op = parse_op("%r = arith.select %cond, %a, %b : i32");
    assert_eq!(op.op_type, "arith.select");
    assert_eq!(op.operands.len(), 3);
    assert_operand_names(&op, &["%cond", "%a", "%b"]);
}

#[test]
fn cmpi_structure() {
    // The Rust slice does not extract the `predicate` attribute (Python did),
    // but op_type + both operands are faithful.
    let op = parse_op("%b = arith.cmpi slt, %a, %c0 : index");
    assert_eq!(op.op_type, "arith.cmpi");
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%a", "%c0"]);
}

#[test]
fn cmpf_structure() {
    let op = parse_op("%r = arith.cmpf olt, %a, %b : f16");
    assert_eq!(op.op_type, "arith.cmpf");
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%a", "%b"]);
}

#[test]
fn sitofp_structure() {
    let op = parse_op("%f = arith.sitofp %i : i32 to f16");
    assert_eq!(op.op_type, "arith.sitofp");
    assert_eq!(op.operands.len(), 1);
    assert_operand_names(&op, &["%i"]);
}

// ===========================================================================
// linalg dialect (TestLinalgParsers) — structural op_type + operands only.
// ===========================================================================

#[test]
fn reduce_structure() {
    // ins(%x) outs(%init): both operands captured. reduce_fn/dim attrs are not
    // produced by the Rust slice.
    let op = parse_op(
        "%r = linalg.reduce { arith.maxnumf } ins(%x : tensor<1x1024xf16>) \
         outs(%init : tensor<1xf16>) dimensions = [1]",
    );
    assert_eq!(op.op_type, "linalg.reduce");
    assert_operand_names(&op, &["%x", "%init"]);
}

#[test]
fn fill_structure() {
    let op =
        parse_op("%out = linalg.fill ins(%val : f16) outs(%buf : tensor<4xf16>) -> tensor<4xf16>");
    assert_eq!(op.op_type, "linalg.fill");
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%val", "%buf"]);
}

#[test]
fn broadcast_structure() {
    let op = parse_op(
        "%out = linalg.broadcast ins(%x : tensor<4xf16>) \
         outs(%buf : tensor<4x8xf16>) dimensions = [1]",
    );
    assert_eq!(op.op_type, "linalg.broadcast");
    assert_operand_names(&op, &["%x", "%buf"]);
}

#[test]
fn matmul_structure() {
    // operands are [A, B, C] (ins then outs).
    let op = parse_op(
        "%r = linalg.matmul ins(%a, %b : tensor<4x8xf16>, tensor<8x16xf16>) \
         outs(%c : tensor<4x16xf16>) -> tensor<4x16xf16>",
    );
    assert_eq!(op.op_type, "linalg.matmul");
    assert_eq!(op.operands.len(), 3);
    assert_operand_names(&op, &["%a", "%b", "%c"]);
}

#[test]
fn batch_matmul_structure() {
    let op = parse_op(
        "%r = linalg.batch_matmul ins(%a, %b : tensor<2x4x8xf16>, tensor<2x8x16xf16>) \
         outs(%c : tensor<2x4x16xf16>) -> tensor<2x4x16xf16>",
    );
    assert_eq!(op.op_type, "linalg.batch_matmul");
    assert_eq!(op.operands.len(), 3);
    assert_operand_names(&op, &["%a", "%b", "%c"]);
}

// ===========================================================================
// tensor dialect (TestTensorParsers) — structural only.
// ===========================================================================

#[test]
fn empty_structure() {
    let op = parse_op("%t = tensor.empty() : tensor<1x1024xf16>");
    assert_eq!(op.op_type, "tensor.empty");
    assert!(op.operands.is_empty());
    assert_eq!(op.result_type.as_deref(), Some("tensor<1x1024xf16>"));
}

#[test]
fn splat_structure() {
    let op = parse_op("%t = tensor.splat %val : tensor<4xf16>");
    assert_eq!(op.op_type, "tensor.splat");
    assert_eq!(op.operands.len(), 1);
    assert_operand_names(&op, &["%val"]);
}

#[test]
fn extract_structure() {
    let op = parse_op("%s = tensor.extract %t[%i, %j] : tensor<4x4xf16>");
    assert_eq!(op.op_type, "tensor.extract");
    assert_eq!(op.operands.len(), 3);
    assert_operand_names(&op, &["%t", "%i", "%j"]);
}

#[test]
fn expand_shape_structure() {
    let op = parse_op(
        "%out = tensor.expand_shape %in [[0, 1]] output_shape [1, 1024] \
         : tensor<1024xf16> into tensor<1x1024xf16>",
    );
    assert_eq!(op.op_type, "tensor.expand_shape");
    assert_eq!(op.operands.len(), 1);
    assert_operand_names(&op, &["%in"]);
}

#[test]
fn reshape_structure() {
    let op = parse_op(
        "%out = tensor.reshape %src(%shape) : (tensor<512xf32>, tensor<2xindex>) -> tensor<16x32xf32>",
    );
    assert_eq!(op.op_type, "tensor.reshape");
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%src", "%shape"]);
    assert_eq!(op.result_type.as_deref(), Some("tensor<16x32xf32>"));
}

#[test]
fn from_elements_structure() {
    let op = parse_op("%shape = tensor.from_elements %d0, %d1 : tensor<2xindex>");
    assert_eq!(op.op_type, "tensor.from_elements");
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%d0", "%d1"]);
}

#[test]
fn from_elements_n1_structure() {
    let op = parse_op("%shape = tensor.from_elements %a : tensor<1xindex>");
    assert_eq!(op.op_type, "tensor.from_elements");
    assert_eq!(op.operands.len(), 1);
    assert_operand_names(&op, &["%a"]);
}

#[test]
fn from_elements_three_structure() {
    let op = parse_op("%shape = tensor.from_elements %a, %b, %c : tensor<3xindex>");
    assert_eq!(op.operands.len(), 3);
    assert_operand_names(&op, &["%a", "%b", "%c"]);
}

// ===========================================================================
// ktdp dialect (TestKtdpParsers)
// ===========================================================================

#[test]
fn get_compute_tile_id_single() {
    let op = parse_op("%id = ktdp.get_compute_tile_id : index");
    assert_eq!(op.op_type, "ktdp.get_compute_tile_id");
    assert_eq!(op.result.as_deref(), Some("%id"));
}

#[test]
fn get_compute_tile_id_bundled_result_name() {
    // Bundled `%pid:2` form parses; the Rust slice keeps the single bundled
    // result name string (it does not split into N distinct names — Python's
    // multi-result list shape is not modelled by Operation.result).
    let op = parse_op("%pid:2 = ktdp.get_compute_tile_id : index, index");
    assert_eq!(op.op_type, "ktdp.get_compute_tile_id");
    let r = op.result.as_deref().expect("bundled result name");
    assert!(r.starts_with("%"));
}

#[test]
fn construct_memory_view_attributes() {
    // Records shape, strides, dtype, memory_space, and the pointer operand.
    let op = parse_op(
        "%view = ktdp.construct_memory_view %ptr, sizes: [1024], strides: [1] \
         { coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>, \
         memory_space = #ktdp.spyre_memory_space<HBM> } : memref<1024xf16>",
    );
    assert_eq!(op.op_type, "ktdp.construct_memory_view");
    assert_eq!(op.attributes.get("shape"), Some(&Attr::IntList(vec![1024])));
    assert_eq!(op.attributes.get("strides"), Some(&Attr::IntList(vec![1])));
    assert_eq!(op.attributes.get("dtype"), Some(&Attr::Str("f16".into())));
    assert_eq!(
        op.attributes.get("memory_space"),
        Some(&Attr::Str("HBM".into()))
    );
    assert_eq!(op.operands.len(), 1);
    assert_operand_names(&op, &["%ptr"]);
}

#[test]
fn construct_memory_view_lx_core_and_strides() {
    // Per-core LX memory space and a multi-dim strided view: lx_core_id recorded.
    let op = parse_op(
        "%view = ktdp.construct_memory_view %ptr, sizes: [16, 32], strides: [32, 1] \
         { memory_space = #ktdp.spyre_memory_space<LX, core=3> } : memref<16x32xf32>",
    );
    assert_eq!(
        op.attributes.get("shape"),
        Some(&Attr::IntList(vec![16, 32]))
    );
    assert_eq!(
        op.attributes.get("strides"),
        Some(&Attr::IntList(vec![32, 1]))
    );
    assert_eq!(op.attributes.get("dtype"), Some(&Attr::Str("f32".into())));
    assert_eq!(
        op.attributes.get("memory_space"),
        Some(&Attr::Str("LX".into()))
    );
    assert_eq!(op.attributes.get("lx_core_id"), Some(&Attr::Int(3)));
}

#[test]
fn construct_access_tile_attributes() {
    // Records the tile shape (from the access_tile result type) and the
    // identity base_map; both operands captured. The trivially-full
    // access_tile_set over shape 128 is normalised away (not in attributes).
    let op = parse_op(
        "%acc = ktdp.construct_access_tile %view[%c0] \
         { access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>, \
         access_tile_order = affine_map<(d0) -> (d0)> } \
         : memref<1024xf16> -> !ktdp.access_tile<128xindex>",
    );
    assert_eq!(op.op_type, "ktdp.construct_access_tile");
    assert_eq!(op.attributes.get("shape"), Some(&Attr::IntList(vec![128])));
    match op.attributes.get("base_map") {
        Some(Attr::AffineMap(m)) => assert!(m.is_identity()),
        other => panic!("expected base_map AffineMap, got {other:?}"),
    }
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%view", "%c0"]);
}

#[test]
fn construct_access_tile_non_index_elem_type_rejected() {
    // Per spec, AccessTileType element type must be `index`; `f16` is rejected.
    let src = "module {\n  func.func @f() {\n    \
        %acc = ktdp.construct_access_tile %view[%c0] \
        { access_tile_order = affine_map<(d0) -> (d0)> } \
        : memref<1024xf16> -> !ktdp.access_tile<128xf16>\n    return\n  }\n}";
    let err = parse_module(src).unwrap_err();
    assert!(
        err.contains("element type must be 'index'") && err.contains("f16"),
        "unexpected error: {err}"
    );
}

#[test]
fn construct_access_tile_malformed_type_rejected() {
    // `!ktdp.access_tile<128>` has no element type — rejected.
    let src = "module {\n  func.func @f() {\n    \
        %acc = ktdp.construct_access_tile %view[%c0] \
        { access_tile_order = affine_map<(d0) -> (d0)> } \
        : memref<1024xf16> -> !ktdp.access_tile<128>\n    return\n  }\n}";
    let err = parse_module(src).unwrap_err();
    assert!(
        err.contains("Malformed access_tile"),
        "unexpected error: {err}"
    );
}

#[test]
fn construct_memory_view_symbolic_dim() {
    // affine_set<(d0)[s0] : ...> with a symbolic dim; memref<?xf32> dynamic dim;
    // SSA size %n_idx registered as an operand alongside %ptr.
    let op = parse_op(
        "%view = ktdp.construct_memory_view %ptr, sizes: [%n_idx], strides: [1] \
         { coordinate_set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>, \
         memory_space = #ktdp.spyre_memory_space<HBM> } : memref<?xf32>",
    );
    assert_eq!(op.op_type, "ktdp.construct_memory_view");
    assert_eq!(op.attributes.get("dtype"), Some(&Attr::Str("f32".into())));
    assert_eq!(
        op.attributes.get("memory_space"),
        Some(&Attr::Str("HBM".into()))
    );
    // %ptr + %n_idx = 2 operands.
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%ptr", "%n_idx"]);
    // The coordinate_set carries the symbolic dim s0.
    match op.attributes.get("coordinate_set") {
        Some(Attr::AffineSet(s)) => assert_eq!(s.num_syms, 1),
        other => panic!("expected coordinate_set AffineSet, got {other:?}"),
    }
}

#[test]
fn construct_access_tile_dynamic_memref() {
    // shape comes from the access_tile result type, never from memref<?xf32>,
    // so the dynamic '?' passes through.
    let op = parse_op(
        "%x_tile = ktdp.construct_access_tile %x_mem[%off_idx] \
         { access_tile_order = affine_map<(d0) -> (d0)>, \
         access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)> } \
         : memref<?xf32> -> !ktdp.access_tile<1024xindex>",
    );
    assert_eq!(op.op_type, "ktdp.construct_access_tile");
    assert_eq!(op.attributes.get("shape"), Some(&Attr::IntList(vec![1024])));
    assert_operand_names(&op, &["%x_mem", "%off_idx"]);
}

// ===========================================================================
// scf dialect (TestScfParsers) — structural op_type + operands only.
// The Rust slice does not extract iter_var / iter_args attributes, nor does it
// parse the nested scf.for body region (regions are deferred), so only the
// top-level scf.for op_type and its lb/ub/step operands are checked.
// ===========================================================================

#[test]
fn scf_for_basic_structure() {
    let ops = parse_ops("scf.for %i = %lb to %ub step %step {\n      scf.yield\n    }");
    let for_op = ops
        .iter()
        .find(|o| o.op_type == "scf.for")
        .expect("scf.for op");
    assert_operand_names(for_op, &["%lb", "%ub", "%step"]);
}

// ===========================================================================
// math dialect (TestMathParsers) — structural op_type + operands.
// ===========================================================================

#[test]
fn math_unary_ops_structure() {
    for name in [
        "math.exp",
        "math.sqrt",
        "math.rsqrt",
        "math.log",
        "math.log2",
        "math.log1p",
        "math.tanh",
        "math.sin",
        "math.cos",
        "math.absf",
        "math.ceil",
        "math.floor",
        "math.erf",
    ] {
        let op = parse_op(&format!("%y = {name} %x : tensor<1024xf32>"));
        assert_eq!(op.op_type, name);
        assert_eq!(op.operands.len(), 1, "op {name}");
        assert_operand_names(&op, &["%x"]);
    }
}

#[test]
fn math_absi_structure() {
    let op = parse_op("%y = math.absi %x : tensor<1024xi32>");
    assert_eq!(op.op_type, "math.absi");
    assert_eq!(op.operands.len(), 1);
    assert_operand_names(&op, &["%x"]);
}

#[test]
fn math_powf_structure() {
    let op = parse_op("%y = math.powf %a, %b : tensor<1024xf32>");
    assert_eq!(op.op_type, "math.powf");
    assert_eq!(op.operands.len(), 2);
    assert_operand_names(&op, &["%a", "%b"]);
}

#[test]
fn math_fma_structure() {
    let op = parse_op("%y = math.fma %a, %b, %c : tensor<1024xf32>");
    assert_eq!(op.op_type, "math.fma");
    assert_eq!(op.operands.len(), 3);
    assert_operand_names(&op, &["%a", "%b", "%c"]);
}
