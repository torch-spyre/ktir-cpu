// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_parser_errors.py` — parser error / robustness handling,
//! exercised through the crate's module parser (`ktir_emulator::parser::parse_module`).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * The Python suite drives several private `KTIRParser` helpers that have NO
//!   public Rust analogue: `parse_file` (file IO + `FileNotFoundError`),
//!   `_parse_operation_text` (bare single-op parse), `_is_op_complete` (op
//!   boundary heuristic), `_preprocess_text` (comment stripping returning a
//!   string), and `_line_opens_region`. The Rust crate exposes only the
//!   module-scoped `parse_module(&str) -> Result<IRModule, String>`. Those cases
//!   are ported where the *observable behaviour* round-trips through
//!   `parse_module` (empty/whitespace/comment-only input, comment stripping,
//!   region detection, op-boundary flushing). The pure file-IO and
//!   bare-helper cases are Python-only infra and are recorded as `#[ignore]`
//!   stubs (see the `skipped` summary).
//!
//! * KEY DIVERGENCE: the Python parser is deliberately *lenient* — it never
//!   raises on truncated / unclosed / unicode-corrupted op text, instead
//!   skipping the bad op and returning a partial `IRModule` (the tests only
//!   assert `module is not None`). The Rust parser is *stricter* on those same
//!   inputs and returns `Err`. We port these faithfully to the ACTUAL Rust
//!   behaviour: where Python asserted "no raise, partial module" and Rust
//!   returns `Err`, we assert the `Err` (and the specific message where it is
//!   load-bearing). This honours the task focus ("malformed input -> Err")
//!   while documenting the strictness difference rather than weakening either
//!   side.
//!
//! * `Operation.result` is `Option<String>` carrying the leading `%` (e.g.
//!   `Some("%x")`), matching `port_parse.rs`.

use ktir_emulator::parser::parse_module;

/// Wrap a body in a minimal `func.func @test()` module. Mirrors the Python
/// `_minimal_func` helper.
fn minimal_func(body: &str) -> String {
    format!("module {{\n  func.func @test() {{\n{body}\n  }}\n}}\n")
}

// ===========================================================================
// parse_module: empty / whitespace / comment-only input
// (Python: test_parse_module_empty_string / _whitespace_only / _comments_only,
//  and the empty-file behaviour of test_parse_file_empty — empty content yields
//  a module with zero functions, not an error.)
// ===========================================================================

#[test]
fn parse_module_empty_string() {
    // Empty input is a valid (empty) module, not an error.
    let module = parse_module("").expect("empty string should parse to an empty module");
    assert_eq!(module.functions.len(), 0);
}

#[test]
fn parse_module_whitespace_only() {
    let module = parse_module("   \n\t\n   ").expect("whitespace-only should parse");
    assert_eq!(module.functions.len(), 0);
}

#[test]
fn parse_module_comments_only() {
    // Comment-only input (after `// ...` stripping) is empty -> zero functions.
    let module =
        parse_module("// nothing here\n// also nothing\n").expect("comment-only should parse");
    assert_eq!(module.functions.len(), 0);
}

// ===========================================================================
// Smoke: a minimal valid module parses (Python: test_parse_file_valid, but via
// parse_module since the Rust crate has no parse_file).
// ===========================================================================

#[test]
fn parse_module_minimal_valid() {
    let module = parse_module(&minimal_func("")).expect("minimal func should parse");
    assert_eq!(module.functions.len(), 1);
    assert!(module.functions.contains_key("test"));
}

// ===========================================================================
// Malformed MLIR
// ===========================================================================

#[test]
fn parse_module_unclosed_module_brace() {
    // Python (test_parse_module_unclosed_module_brace): missing outer `}` must
    // not crash; returns a partial module. The Rust parser likewise recovers:
    // the function body block is matched by the inner braces, so the function
    // is still extracted.
    let mlir = "module {\n  func.func @foo() {\n  }\n"; // missing closing }
    let module = parse_module(mlir).expect("unclosed outer brace should still recover");
    assert!(module.functions.contains_key("foo"));
}

#[test]
fn parse_module_func_unclosed_body() {
    // Python (test_parse_module_func_unclosed_body): the broken function is
    // skipped and `module is not None`.
    //
    // DIVERGENCE: the Rust parser cannot find a brace-matched body for the
    // unterminated function and returns Err("function missing body") rather
    // than silently dropping it. Assert the stricter Rust behaviour.
    let mlir = "module {\n  func.func @broken() {\n    %0 = arith.constant 1 : index\n";
    let err = parse_module(mlir).expect_err("unclosed function body should be an error in Rust");
    assert!(
        err.contains("function missing body"),
        "unexpected error: {err}"
    );
}

#[test]
fn parse_module_truncated_mid_op() {
    // Python (test_parse_module_truncated_mid_op): a constant truncated to
    // `arith.constant` (no value/type) is skipped; `module is not None`.
    //
    // DIVERGENCE: the Rust `arith.constant` parser requires a value literal and
    // returns Err on the empty value. Assert the Err faithfully.
    let mlir = minimal_func("  %x = arith.constant");
    let err = parse_module(&mlir).expect_err("truncated arith.constant should error in Rust");
    assert!(err.contains("arith.constant"), "unexpected error: {err}");
}

// ===========================================================================
// Unrecognized / garbage op text inside an otherwise-valid module
// ===========================================================================

#[test]
fn parse_module_garbage_op_line_skipped() {
    // Python (test_parse_operation_text_garbage / test_partial_parse_mixed_ops):
    // a garbage line that matches no op pattern is dropped without raising, and
    // the surrounding valid ops still parse. The Rust tokenizer/structural
    // parser likewise yields the two real `arith.constant` ops and silently
    // ignores the `@@@garbage@@@` line.
    let mlir = minimal_func(
        "    %c0 = arith.constant 0 : index\n\
         \x20   @@@garbage@@@\n\
         \x20   %c1 = arith.constant 1 : index\n",
    );
    let module = parse_module(&mlir).expect("module with one garbage line should still parse");
    let f = module.get_function("test").expect("function test");
    let op_types: Vec<&str> = f.operations.iter().map(|o| o.op_type.as_str()).collect();
    assert!(
        op_types.contains(&"arith.constant"),
        "valid arith.constant ops should survive: {op_types:?}"
    );
    // Both surrounding constants parse; the garbage line contributes no op.
    let n_const = f
        .operations
        .iter()
        .filter(|o| o.op_type == "arith.constant")
        .count();
    assert_eq!(
        n_const, 2,
        "both constants should parse around the garbage line"
    );
}

// ===========================================================================
// Unicode / binary-adjacent content
// ===========================================================================

#[test]
fn parse_module_unicode_in_comments() {
    // Python (test_parse_module_unicode_in_comments / test_parse_file_unicode_content):
    // unicode inside a `//` comment is stripped and must not crash; the real op
    // still parses.
    let mlir = minimal_func(
        "    // \u{3053}\u{3093}\u{306B}\u{3061}\u{306F} \u{1F389} unicode comment\n\
         \x20   %c0 = arith.constant 0 : index\n",
    );
    let module = parse_module(&mlir).expect("unicode comment must not crash the parser");
    assert_eq!(module.functions.len(), 1);
    let f = module.get_function("test").unwrap();
    assert!(f.operations.iter().any(|o| o.op_type == "arith.constant"));
}

#[test]
fn parse_module_unicode_in_op_body() {
    // Python (test_parse_module_unicode_in_op): unicode in the op *body* (not a
    // comment) may fail to parse the op, but `module is not None`.
    //
    // DIVERGENCE: the Rust `arith.constant` value parser rejects the non-ASCII
    // literal and returns Err rather than skipping the op. Assert the Err.
    let mlir = "module {\n  func.func @unicode_test() {\n    %x = arith.constant \u{3053} : index\n  }\n}\n";
    let err = parse_module(mlir).expect_err("unicode in op value should error in Rust");
    assert!(err.contains("arith.constant"), "unexpected error: {err}");
}

// ===========================================================================
// Grid attribute robustness
// ===========================================================================

#[test]
fn parse_grid_missing_defaults() {
    // Python (test_parse_grid_missing): no grid attribute -> default (1,1,1).
    let module = parse_module(&minimal_func("")).expect("minimal func parses");
    assert_eq!(module.get_function("test").unwrap().grid, (1, 1, 1));
}

#[test]
fn parse_grid_malformed_falls_back() {
    // Python (test_parse_grid_malformed): a non-numeric grid entry falls back to
    // the default (1,1,1) without crashing. The Rust `parse_grid` filters out
    // unparseable entries, so `[not_a_number]` yields the all-default grid.
    let mlir = "module {\n  func.func @g() attributes { grid = [not_a_number] } {\n  }\n}\n";
    let module = parse_module(mlir).expect("malformed grid must not crash");
    assert_eq!(module.get_function("g").unwrap().grid, (1, 1, 1));
}

// ===========================================================================
// Region detection robustness (Python: the `_line_opens_region` regression
// suite). Only the cases observable through `parse_module` are ported.
// ===========================================================================

#[test]
fn region_with_percent_in_comment_still_detected() {
    // Python (test_region_with_percent_in_comment_still_detected): a real
    // scf.for region containing a `// ... 100% ...` comment is still detected as
    // exactly one region; the bare `%` in the comment is not what triggers it.
    let mlir = "module {\n  func.func @test(%lb: index, %ub: index, %step: index) attributes { grid = [1] } {\n    scf.for %i = %lb to %ub step %step {\n        // iterate over 100% of elements\n        %c0 = arith.constant 0 : index\n        scf.yield\n    }\n    return\n  }\n}\n";
    let module = parse_module(mlir).expect("scf.for with %-comment must parse");
    let f = module.get_function("test").unwrap();
    let for_op = f
        .operations
        .iter()
        .find(|o| o.op_type == "scf.for")
        .expect("scf.for op");
    assert_eq!(
        for_op.regions.len(),
        1,
        "scf.for body should be detected as a single region"
    );
}

// ===========================================================================
// Operation boundary detection (Python: _is_op_complete suite). The Rust
// tokenizer is internal, but its boundary heuristics are observable through
// `parse_module`: blank-line flush and SSA-assignment flush.
// ===========================================================================

#[test]
fn blank_line_flushes_op() {
    // Python (test_blank_line_flushes_op): a linalg.reduce with no type terminal
    // is flushed by a following blank line, so both it and `return` are parsed.
    let mlir = "module {\n  func.func @test(%x: tensor<4xf16>, %init: tensor<1xf16>) attributes { grid = [1] } {\n    %r = linalg.reduce { arith.maxnumf } ins(%x : tensor<4xf16>) outs(%init : tensor<1xf16>) dimensions = [0]\n\n    return\n  }\n}\n";
    let module = parse_module(mlir).expect("reduce + blank line must parse");
    let f = module.get_function("test").unwrap();
    let op_types: Vec<&str> = f.operations.iter().map(|o| o.op_type.as_str()).collect();
    assert!(
        op_types.contains(&"linalg.reduce"),
        "reduce should be flushed by the blank line: {op_types:?}"
    );
    assert!(op_types.iter().any(|t| t.ends_with("return")));
}

#[test]
fn ssa_assignment_flushes_previous_op() {
    // Python (test_ssa_assignment_flushes_previous_op): two adjacent constants
    // with no blank line between them are parsed as two separate ops, because an
    // incoming `%name = ` line flushes the previous op.
    let mlir = "module {\n  func.func @test() attributes { grid = [1] } {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constant 1 : index\n    return\n  }\n}\n";
    let module = parse_module(mlir).expect("two adjacent constants must parse");
    let f = module.get_function("test").unwrap();
    let n_const = f
        .operations
        .iter()
        .filter(|o| o.op_type == "arith.constant")
        .count();
    assert_eq!(
        n_const, 2,
        "two adjacent constants should parse as separate ops"
    );
}

// ===========================================================================
// Python-only / GAP stubs
// ===========================================================================

#[test]
#[ignore = "GAP: no public parse_file in the Rust crate. Python's \
test_parse_file_nonexistent (FileNotFoundError) and test_parse_file_empty/\
_valid/_unicode_content exercise filesystem IO that parse_module does not \
cover; the content-level behaviour (empty -> 0 funcs, valid -> 1 func, unicode \
-> no crash) is already covered by parse_module_* tests above."]
fn parse_file_filesystem_cases() {}

#[test]
#[ignore = "GAP: no public _parse_operation_text. Python's \
test_parse_operation_text_empty/_whitespace/_unrecognized_op/_garbage parse a \
BARE single op and assert None for empty/whitespace/garbage and a passthrough \
op_type for an unknown dialect op. The Rust parser is module-scoped only; the \
garbage-passthrough behaviour is covered by parse_module_garbage_op_line_skipped."]
fn parse_operation_text_bare_helper() {}

#[test]
fn is_op_complete_predicate() {
    use ktir_emulator::parser::is_op_complete;
    // Type terminals: scalar / index / tensor types are complete.
    assert!(is_op_complete("%x = arith.addf %a, %b : f32"));
    assert!(is_op_complete("%x = arith.addi %a, %b : index"));
    assert!(is_op_complete("%x = arith.addi %a, %b : i32"));
    assert!(is_op_complete("%t = ktdp.load %a : tensor<128xf16>"));
    // Void terminators are complete.
    assert!(is_op_complete("return"));
    assert!(is_op_complete("scf.yield %a"));
    // No type terminal yet -> not complete (relies on a later flush).
    assert!(!is_op_complete("linalg.reduce ins(%x) dimensions = [1]"));
}

#[test]
fn preprocess_text_string_shape() {
    use ktir_emulator::parser::strip_comments;
    let text = "  %x = arith.addf %a, %b : f32  // add\n  return\n";
    let result = strip_comments(text);
    assert!(!result.contains("//"));
    assert!(result.contains("%x = arith.addf"));
    assert!(result.contains("return"));
    // Line structure preserved (newline count invariant).
    assert_eq!(result.matches('\n').count(), text.matches('\n').count());
    // A full-line comment becomes blank but the line remains.
    let r2 = strip_comments("    // this is a comment\n");
    assert!(!r2.contains("//"));
    assert_eq!(r2.matches('\n').count(), 1);
}

#[test]
fn arith_constant_attribute_block_region_detection() {
    // `arith.constant { value = 42 : i32 }` — the `{ }` is an attribute block,
    // NOT a region (no `%` SSA refs inside): the op parses with ZERO regions.
    let src = "module {\n  func.func @f() {\n    \
               %x = arith.constant { value = 42 : i32 } : index\n    return\n  }\n}";
    let module = parse_module(src).expect("attribute-block constant parses");
    let f = module.get_function("f").unwrap();
    let c = f
        .operations
        .iter()
        .find(|o| o.op_type == "arith.constant")
        .expect("constant op");
    assert!(c.regions.is_empty(), "attribute block must not be a region");
    assert_eq!(
        c.attributes.get("value"),
        Some(&ktir_emulator::ir::Attr::Int(42))
    );
}
