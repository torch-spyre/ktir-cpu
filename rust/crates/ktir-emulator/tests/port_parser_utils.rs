#![allow(
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::approx_constant
)]
// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_parser_utils.py` — `parse_tensor_type` helper behaviour.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python calls the public helper `ktir_emulator.parser_utils.parse_tensor_type`
//!   directly and asserts the returned `{"shape": (...), "dtype": ...}` dict
//!   (or `None`). The Rust crate's analogue, `parser::parse_tensor_type`, is a
//!   PRIVATE `fn` (not `pub`), so it cannot be called from an integration test.
//!   We therefore exercise it through its only observable effect: when the
//!   structural parser sees an op like `tensor.empty` whose result type is a
//!   ranked `tensor<...>`, it calls `parse_tensor_type(result_type)` and, on a
//!   `Some((shape, dtype))`, populates the op's `shape` (`Attr::IntList`) and
//!   `dtype` (`Attr::Str`) attributes (see `parser.rs` ~L463). A `None` leaves
//!   both attributes absent. So:
//!       Python `{"shape": s, "dtype": d}`  <->  Rust `shape`/`dtype` attrs set
//!       Python `None`                       <->  Rust `shape`/`dtype` absent
//!   This is a faithful 1:1 mapping for the values the helper produces; the
//!   only loss is that the wrapping op text must be a parseable module, which
//!   holds for every Python case below.
//!
//! * IMPORTANT DIVERGENCE — the Rust `parse_tensor_type` is a DIFFERENT, simpler
//!   implementation than the Python regex. It uses a naive `inner.split('x')`,
//!   i.e. exactly the pre-fix Python behaviour that several of these tests were
//!   written to PIN AS FIXED. Where the Rust crate still carries that bug (the
//!   `index` dtype, the encoding attribute, trailing context after `>`), the
//!   Python case is `#[ignore = "GAP: ..."]` with the divergence documented,
//!   rather than weakening the assertion to match the buggy Rust output.

/// Parse `tensor.empty() : <type_str>` inside a one-function module and return
/// the parsed `(shape, dtype)` the structural parser derived from the result
/// type — `None` when the helper rejected the type (no `shape`/`dtype` attrs).
///
/// Calls the public `parser::parse_tensor_type` helper directly, exactly as the
/// Python suite calls `parser_utils.parse_tensor_type` — not through a parsed
/// module, so trailing-context / encoding-attribute handling is the helper's,
/// matching Python.
fn parse_tensor_type(type_str: &str) -> Option<(Vec<i64>, String)> {
    ktir_emulator::parser::parse_tensor_type(type_str)
}

/// Assert `parse_tensor_type` produced the expected `(shape, dtype)`.
fn assert_parsed(type_str: &str, expected_shape: &[i64], expected_dtype: &str) {
    let got = parse_tensor_type(type_str);
    assert_eq!(
        got,
        Some((expected_shape.to_vec(), expected_dtype.to_string())),
        "parse_tensor_type({type_str:?})"
    );
}

/// Assert `parse_tensor_type` rejected the input (Python `None`).
fn assert_rejected(type_str: &str) {
    assert_eq!(
        parse_tensor_type(type_str),
        None,
        "parse_tensor_type({type_str:?}) expected None"
    );
}

// ---------------------------------------------------------------------------
// Basic shape/dtype combinations  (test_parse_tensor_type_basic)
//
// Plain numeric/float dtypes round-trip cleanly across rank 1-4. The Rust
// `split('x')` implementation agrees with the Python regex for all of these.
// ---------------------------------------------------------------------------

#[test]
fn parse_tensor_type_basic_floats() {
    assert_parsed("tensor<256xf16>", &[256], "f16");
    assert_parsed("tensor<1024xf32>", &[1024], "f32");
    assert_parsed("tensor<8xbf16>", &[8], "bf16");
}

#[test]
fn parse_tensor_type_basic_signless_ints() {
    assert_parsed("tensor<10xi32>", &[10], "i32");
    assert_parsed("tensor<3xi64>", &[3], "i64");
    assert_parsed("tensor<7xi1>", &[7], "i1");
}

#[test]
fn parse_tensor_type_basic_higher_rank() {
    assert_parsed("tensor<1x64xf32>", &[1, 64], "f32");
    assert_parsed("tensor<128x16xf16>", &[128, 16], "f16");
    assert_parsed("tensor<1x16x1x128xf16>", &[1, 16, 1, 128], "f16");
}

// ---------------------------------------------------------------------------
// Regression: dtypes whose name contains 'x'
// (test_parse_tensor_type_index_dtype)
//
// GAP — the Python test pins the FIX for `inner.split('x')` mis-tokenising
// `index` (e.g. `"2xindex"` -> `["2", "inde", ""]`, taking `""` as the dtype).
// The Rust crate's `parse_tensor_type` is precisely that pre-fix `split('x')`
// implementation, so it STILL mis-parses `index`: `tensor<2xindex>` yields no
// `shape`/`dtype` attrs (observed: `shape=None dtype=None`). Asserting `None`
// here would WEAKEN the test (Python demands the index dtype be preserved), so
// these are ignored and the divergence is recorded rather than masked.
// ---------------------------------------------------------------------------

#[test]
fn parse_tensor_type_index_dtype() {
    assert_parsed("tensor<2xindex>", &[2], "index");
    assert_parsed("tensor<3xindex>", &[3], "index");
    assert_parsed("tensor<2x3xindex>", &[2, 3], "index");
    assert_parsed("tensor<1x16x1xindex>", &[1, 16, 1], "index");
}

// ---------------------------------------------------------------------------
// Non-matching inputs return None  (test_parse_tensor_type_rejects_non_tensor)
//
// All of these parse as a module (the op text is well-formed) but the helper
// rejects the result type, so no shape/dtype is derived — faithful to the
// Python `is None`. Verified empirically against the Rust crate.
// ---------------------------------------------------------------------------

#[test]
fn parse_tensor_type_rejects_non_tensor() {
    assert_rejected("memref<10xf32>"); // Different aggregate type
    assert_rejected("f32"); // Bare element type
    assert_rejected("not a tensor"); // Random text
    assert_rejected(""); // Empty
    assert_rejected("tensor<>"); // Malformed: empty body
    assert_rejected("tensor<f32>"); // Rank-0 tensor — unsupported (no `x`)
    assert_rejected("tensor<?xf16>"); // All-dynamic dims — no static dims
}

// ---------------------------------------------------------------------------
// Real-MLIR forms  (test_parse_tensor_type_real_mlir_forms)
// ---------------------------------------------------------------------------

// Whitespace tolerance: the Rust helper `trim()`s each dim and the dtype, so
// these agree with the Python regex.
#[test]
fn parse_tensor_type_whitespace_tolerance() {
    assert_parsed("tensor< 2 x f32 >", &[2], "f32");
    assert_parsed("tensor<1 x 64 x i32>", &[1, 64], "i32");
}

// GAP — dynamic `?` dims. Python silently DROPS dynamic dims and keeps the
// static ones (`tensor<?x4xf32>` -> shape (4,)). The Rust helper instead fails
// the whole parse the moment any dim is non-numeric (`d.parse::<i64>().ok()?`
// returns `None`), so `tensor<?x4xf32>` yields no shape/dtype. Asserting `None`
// would contradict the Python expectation of `{shape:(4,),dtype:f32}`, so this
// is ignored and recorded.
#[test]
fn parse_tensor_type_dynamic_dims_dropped() {
    assert_parsed("tensor<?x4xf32>", &[4], "f32");
    assert_parsed("tensor<2x?x4xindex>", &[2, 4], "index");
}

// GAP — encoding attribute (RFC-allowed second positional). Python's regex
// stops the dtype before the `,` so `tensor<4x4xf32, #my_enc>` -> dtype "f32".
// The Rust helper has no `,` handling: it takes everything after the last `x`
// up to the final `>` as the dtype, yielding dtype "f32, #my_enc" (observed).
// The second form (`dense<0> : tensor<8xi1>`) additionally trips the naive
// `strip_suffix('>')` + `split('x')`, producing a garbage dtype `"i1>"`.
// Asserting the buggy strings would weaken the test, so this is ignored.
#[test]
fn parse_tensor_type_encoding_attribute() {
    assert_parsed("tensor<4x4xf32, #my_enc>", &[4, 4], "f32");
    assert_parsed("tensor<8xf16, dense<0> : tensor<8xi1>>", &[8], "f16");
}

// GAP — trailing context after the closing '>'. Python uses `re.match`, which
// anchors at the start and ignores trailing context, so `tensor<4xf32>
// loc(unknown)` -> {shape:(4,),dtype:f32}. The Rust helper requires the string
// to END at `>` (`strip_suffix('>')`), AND the structural parser's result-type
// extraction folds the trailing tokens into `result_type` (observed
// `"tensor<4xf32> loc(unknown) return"`), so no shape/dtype is derived. Both
// the helper and the surrounding extraction differ from Python here; ignored
// rather than weakened.
#[test]
fn parse_tensor_type_trailing_context() {
    assert_parsed("tensor<4xf32> loc(unknown)", &[4], "f32");
    assert_parsed("tensor<4xf32>, %arg0", &[4], "f32");
}

// ---------------------------------------------------------------------------
// Known limitation: nested-bracket element types
// (test_parse_tensor_type_nested_bracket_dtype)
//
// Python marks this `@pytest.mark.xfail(strict=True)`: dtypes whose own form
// contains `<...>` (complex<f32>, !tt.ptr<f32>, vector<4xf32>) cannot be parsed
// by a single non-counting regex. The Rust `split('x')`/`strip_suffix('>')`
// helper has the same limitation (and, like the regex, these forms do not reach
// this parser in lowered KTIR). Kept ignored to mirror the Python xfail; the
// asserts encode the *desired* (currently-failing) behaviour, so we do not run
// them — exactly like the Python `strict` xfail.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "Python xfail(strict=True): nested-bracket element types \
            (complex<f32>, !tt.ptr<f32>, vector<4xf32>) need a depth-counting \
            parser; the Rust split('x')/strip_suffix('>') helper has the same \
            limitation and these forms do not appear in lowered KTIR today"]
fn parse_tensor_type_nested_bracket_dtype() {
    assert_parsed("tensor<4xcomplex<f32>>", &[4], "complex<f32>");
    assert_parsed("tensor<4x!tt.ptr<f32>>", &[4], "!tt.ptr<f32>");
    assert_parsed("tensor<4xvector<4xf32>>", &[4], "vector<4xf32>");
}
