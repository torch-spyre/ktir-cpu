// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_ast.py` — affine-expression AST parsing + evaluation.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * The Python AST is tuple-shaped (`("dim", 0)`, `("add", ...)`); the Rust
//!   crate models it with the [`AffineExpr`] enum. AST-structure assertions are
//!   therefore translated to enum-pattern equality (e.g. `("dim", 0)` ->
//!   `AffineExpr::Dim(0)`).
//! * `parse_expr` returns `Result<AffineExpr, _>`; `eval_expr(node, env)` ->
//!   `AffineExpr::eval(dims, syms)`.
//! * `parse_affine_map` returns `Result<AffineMap, _>`; fields `n_dims`/`exprs`
//!   -> `num_dims`/`exprs`. `eval_affine_map(m, dims)` -> `AffineMap::eval`.
//! * `parse_affine_set` here is the *raw* form (Rust `parse_affine_set`, the
//!   port of Python `parse_affine_set_raw`): it always yields an [`AffineSet`].
//!   Constraints carry a [`ConstraintKind`] (`GreaterEq` / `Equal`) plus a
//!   normalised `expr` (the Python `(lhs - rhs >= 0)` / `("eq", ...)` form).
//!   `n_dims`/`n_syms`/`constraints` -> `num_dims`/`num_syms`/`constraints`.
//! * `affine_set_contains(s, pt, symbols=..)` -> `AffineSet::contains(pt, syms)`;
//!   `enumerate_affine_set(s, shape, symbols=..)` -> `AffineSet::enumerate`.
//! * The symbolic-bound helpers `sym_add` / `sym_neg` / `sym_max` / `sym_min` /
//!   `eval_bound` take/return [`Bound`] in Rust (concrete ints are
//!   `Bound::Concrete`, symbol refs are `Bound::Symbolic(AffineExpr::Sym(k))`).
//!
//! Intentionally ignored (see `skipped`):
//! * `TestTokenise::*` — the tokeniser (`_tokenise`) is a private impl detail in
//!   the Rust crate (not exported), so it has no public-API analogue.
//! * `m.source` round-trip — the Rust `AffineMap`/`AffineSet` carry no `source`
//!   field (no Rust analogue; Python-only).

use std::rc::Rc;

use ktir_emulator::affine::{AffineExpr, Bound, ConstraintKind};
use ktir_emulator::affine::{eval_bound, sym_add, sym_max, sym_min, sym_neg};
use ktir_emulator::parser_ast::{parse_affine_map, parse_affine_set, parse_expr};

// Convenience constructors mirroring the Python tuple AST.
fn dim(i: usize) -> AffineExpr {
    AffineExpr::Dim(i)
}
fn sym(i: usize) -> AffineExpr {
    AffineExpr::Sym(i)
}
fn cst(c: i64) -> AffineExpr {
    AffineExpr::Const(c)
}
fn add(a: AffineExpr, b: AffineExpr) -> AffineExpr {
    AffineExpr::Add(Rc::new(a), Rc::new(b))
}
fn mul(a: AffineExpr, b: AffineExpr) -> AffineExpr {
    AffineExpr::Mul(Rc::new(a), Rc::new(b))
}
// The Rust parser normalises unary minus to `-1 * x` and binary subtraction to
// `a + (-1 * b)` rather than building `Neg` / `Sub` nodes (the Python tuple AST
// used `("neg", ..)` / `("sub", ..)`). These are eval-equivalent; the AST-shape
// assertions below therefore target the Rust normal form.
fn neg(a: AffineExpr) -> AffineExpr {
    mul(cst(-1), a)
}
fn subx(a: AffineExpr, b: AffineExpr) -> AffineExpr {
    add(a, mul(cst(-1), b))
}

// ===========================================================================
// Tokeniser — TestTokenise
//
// The Rust `tokenise` function is private (not part of the public API), so
// these tests have no public analogue. Kept as ignored stubs.
// ===========================================================================

#[test]
fn tokenise_simple_map_inner() {
    assert_eq!(
        ktir_emulator::parser_ast::tokenise("(d0, d1) -> (d0, d1)"),
        ["(", "d0", ",", "d1", ")", "->", "(", "d0", ",", "d1", ")"]
    );
}

#[test]
fn tokenise_constraint_tokens() {
    let tokens = ktir_emulator::parser_ast::tokenise("(d0 >= 0, -d0 + 63 >= 0)");
    assert!(tokens.iter().any(|t| t == ">="));
    assert!(tokens.iter().any(|t| t == "0"));
}

#[test]
fn tokenise_arrow_token() {
    assert!(
        ktir_emulator::parser_ast::tokenise("(d0) -> (d0)")
            .iter()
            .any(|t| t == "->")
    );
}

#[test]
fn tokenise_whitespace_ignored() {
    assert_eq!(
        ktir_emulator::parser_ast::tokenise("(d0)->(d0)"),
        ktir_emulator::parser_ast::tokenise("( d0 ) -> ( d0 )")
    );
}

// ===========================================================================
// Expression parsing — TestParseExpr (parse_expr / eval_expr)
// ===========================================================================

#[test]
fn parse_constant() {
    let node = parse_expr("42").unwrap();
    assert_eq!(node, cst(42));
    assert_eq!(node.eval(&[], &[]), 42);
}

#[test]
fn parse_dim_variable() {
    let node = parse_expr("d0").unwrap();
    assert_eq!(node, dim(0));
    assert_eq!(node.eval(&[7], &[]), 7);
}

#[test]
fn parse_dim_variable_index() {
    let node = parse_expr("d2").unwrap();
    assert_eq!(node, dim(2));
    assert_eq!(node.eval(&[0, 0, 99], &[]), 99);
}

#[test]
fn parse_addition() {
    let node = parse_expr("d0 + d1").unwrap();
    assert_eq!(node, add(dim(0), dim(1)));
    assert_eq!(node.eval(&[3, 4], &[]), 7);
}

#[test]
fn parse_subtraction() {
    let node = parse_expr("d0 - d1").unwrap();
    assert_eq!(node, subx(dim(0), dim(1)));
    assert_eq!(node.eval(&[10, 3], &[]), 7);
}

#[test]
fn parse_unary_negation() {
    let node = parse_expr("-d0").unwrap();
    assert_eq!(node, neg(dim(0)));
    assert_eq!(node.eval(&[5], &[]), -5);
}

#[test]
fn parse_constant_coefficient() {
    let node = parse_expr("2 * d0").unwrap();
    assert_eq!(node, mul(cst(2), dim(0)));
    assert_eq!(node.eval(&[4], &[]), 8);
}

#[test]
fn parse_negative_coefficient_expr() {
    // -d0 + 63 (common RFC constraint pattern)
    let node = parse_expr("-d0 + 63").unwrap();
    assert_eq!(node, add(neg(dim(0)), cst(63)));
    assert_eq!(node.eval(&[0], &[]), 63);
    assert_eq!(node.eval(&[63], &[]), 0);
    assert_eq!(node.eval(&[64], &[]), -1);
}

#[test]
fn parse_compound_expr() {
    // d0 + 2 * d1 + 3
    let node = parse_expr("d0 + 2 * d1 + 3").unwrap();
    assert_eq!(node.eval(&[1, 2], &[]), 1 + 2 * 2 + 3); // == 8
}

#[test]
fn parse_left_associativity() {
    // a - b + c should be (a - b) + c, not a - (b + c)
    let node = parse_expr("d0 - d1 + d2").unwrap();
    assert_eq!(node.eval(&[10, 3, 1], &[]), 8); // (10-3)+1
}

#[test]
fn parse_parenthesised() {
    let node = parse_expr("2 * (d0 + 1)").unwrap();
    assert_eq!(node.eval(&[4], &[]), 10);
}

#[test]
fn parse_zero_constant() {
    let node = parse_expr("0").unwrap();
    assert_eq!(node, cst(0));
    assert_eq!(node.eval(&[], &[]), 0);
}

// ===========================================================================
// parse_affine_map — AST structure (TestParseAffineMap)
// ===========================================================================

#[test]
fn map_identity_1d() {
    let m = parse_affine_map("affine_map<(d0) -> (d0)>").unwrap();
    assert_eq!(m.num_dims, 1);
    assert_eq!(m.exprs.len(), 1);
    assert_eq!(m.exprs[0], dim(0));
}

#[test]
fn map_identity_2d() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>").unwrap();
    assert_eq!(m.num_dims, 2);
    assert_eq!(m.exprs, vec![dim(0), dim(1)]);
}

#[test]
fn map_non_identity_row_select() {
    // (i) -> (i, 0) — softmax_wide.mlir pattern
    let m = parse_affine_map("affine_map<(i) -> (i, 0)>").unwrap();
    assert_eq!(m.num_dims, 1);
    assert_eq!(m.exprs.len(), 2);
    assert_eq!(m.exprs[0], dim(0));
    assert_eq!(m.exprs[1], cst(0));
}

#[test]
fn map_transposed() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>").unwrap();
    assert_eq!(m.exprs, vec![dim(1), dim(0)]);
}

#[test]
fn map_constant_offset() {
    let m = parse_affine_map("affine_map<(d0) -> (d0 + 1)>").unwrap();
    assert_eq!(m.exprs[0], add(dim(0), cst(1)));
}

#[test]
fn map_scaled() {
    let m = parse_affine_map("affine_map<(d0) -> (2 * d0)>").unwrap();
    assert_eq!(m.exprs[0], mul(cst(2), dim(0)));
}

#[test]
fn map_complex_expr() {
    // (d0 + 2 * d1)
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0 + 2 * d1)>").unwrap();
    assert_eq!(m.exprs[0], add(dim(0), mul(cst(2), dim(1))));
}

#[test]
fn map_negative_expr() {
    // -d0 + 63
    let m = parse_affine_map("affine_map<(d0) -> (-d0 + 63)>").unwrap();
    assert_eq!(m.exprs[0], add(neg(dim(0)), cst(63)));
}

#[test]
fn map_inner_text_without_wrapper() {
    let m = parse_affine_map("(d0) -> (d0)").unwrap();
    assert_eq!(m.num_dims, 1);
}

#[test]
#[ignore = "GAP: AffineMap has no `source` field in the Rust crate; Python-only round-trip"]
fn map_source_preserved() {}

#[test]
fn map_zero_dims() {
    let m = parse_affine_map("affine_map<() -> (0)>").unwrap();
    assert_eq!(m.num_dims, 0);
    assert_eq!(m.exprs, vec![cst(0)]);
}

// ===========================================================================
// parse_affine_set — AST structure (TestParseAffineSet)
//
// Constraints are normalised: an inequality `lhs >= rhs` becomes
// `GreaterEq` with `expr = lhs - rhs`; `lhs <= rhs` becomes `rhs - lhs >= 0`.
// The Python AST used an explicit `("sub", lhs, ("const", 0))` wrapper; the
// Rust normalisation already subtracts the rhs into `expr`, so we assert on the
// resulting `expr` directly (equivalent to Python's `sub(lhs, 0)`-stripped form
// once `rhs == 0`).
// ===========================================================================

#[test]
fn set_1d_range() {
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>").unwrap();
    assert_eq!(s.num_dims, 1);
    assert_eq!(s.constraints.len(), 2);
    // d0 >= 0  ->  expr = (d0 - 0), GreaterEq. The Rust normaliser keeps the
    // `- rhs` term verbatim (here `+ (-1 * 0)`) rather than folding it away.
    assert_eq!(s.constraints[0].kind, ConstraintKind::GreaterEq);
    assert_eq!(s.constraints[0].expr, subx(dim(0), cst(0)));
    // -d0 + 31 >= 0  ->  expr = ((-d0 + 31) - 0), GreaterEq.
    assert_eq!(s.constraints[1].kind, ConstraintKind::GreaterEq);
    assert_eq!(
        s.constraints[1].expr,
        subx(add(neg(dim(0)), cst(31)), cst(0))
    );
}

#[test]
fn set_2d_rect() {
    let src = "affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>";
    let s = parse_affine_set(src).unwrap();
    assert_eq!(s.num_dims, 2);
    assert_eq!(s.constraints.len(), 4);
}

#[test]
#[ignore = "GAP: AffineSet has no `source` field in the Rust crate; Python-only round-trip"]
fn set_source_preserved() {}

#[test]
fn set_inner_text_without_wrapper() {
    let s = parse_affine_set("(d0) : (d0 >= 0, -d0 + 3 >= 0)").unwrap();
    assert_eq!(s.num_dims, 1);
    assert_eq!(s.constraints.len(), 2);
}

#[test]
fn set_leq_normalised() {
    // d0 <= 0 normalised to 0 - d0 >= 0
    let s = parse_affine_set("affine_set<(d0) : (d0 <= 0)>").unwrap();
    assert_eq!(s.constraints[0].kind, ConstraintKind::GreaterEq);
    assert_eq!(s.constraints[0].expr, subx(cst(0), dim(0)));
}

#[test]
fn set_general_rhs() {
    // d0 >= d1  ->  d0 - d1 >= 0
    // d0 <= 63  ->  63 - d0 >= 0
    let s = parse_affine_set("affine_set<(d0, d1) : (d0 >= d1, d0 <= 63)>").unwrap();
    assert_eq!(s.num_dims, 2);
    assert_eq!(s.constraints[0].expr, subx(dim(0), dim(1)));
    assert_eq!(s.constraints[1].expr, subx(cst(63), dim(0)));
}

#[test]
fn set_symbolic_dim_parsed() {
    // (d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0) — s0 is a runtime symbol
    let s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    assert_eq!(s.num_dims, 1);
    assert_eq!(s.num_syms, 1);
    assert_eq!(s.constraints.len(), 2);
    // The s0 token should appear as Sym(0) somewhere in the second constraint.
    assert_eq!(find_sym(&s.constraints[1].expr), Some(0));
}

fn find_sym(e: &AffineExpr) -> Option<usize> {
    match e {
        AffineExpr::Sym(i) => Some(*i),
        AffineExpr::Dim(_) | AffineExpr::Const(_) | AffineExpr::Ref(_) => None,
        AffineExpr::Neg(a) => find_sym(a),
        AffineExpr::Add(a, b)
        | AffineExpr::Sub(a, b)
        | AffineExpr::Mul(a, b)
        | AffineExpr::FloorDiv(a, b)
        | AffineExpr::Mod(a, b)
        | AffineExpr::Max(a, b)
        | AffineExpr::Min(a, b) => find_sym(a).or_else(|| find_sym(b)),
    }
}

#[test]
fn set_symbolic_dim_multiple_syms() {
    // Two symbols: (d0)[s0, s1]
    let s = parse_affine_set("affine_set<(d0)[s0, s1] : (d0 >= 0, -d0 + s0 - 1 >= 0, s1 >= 0)>")
        .unwrap();
    assert_eq!(s.num_syms, 2);
}

#[test]
fn set_no_symbol_list_n_syms_zero() {
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>").unwrap();
    assert_eq!(s.num_syms, 0);
}

// ===========================================================================
// eval_affine_map — TestEvalAffineMap
// ===========================================================================

#[test]
fn eval_map_identity_1d() {
    let m = parse_affine_map("affine_map<(d0) -> (d0)>").unwrap();
    assert_eq!(m.eval(&[5], &[]), vec![5]);
}

#[test]
fn eval_map_identity_2d() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>").unwrap();
    assert_eq!(m.eval(&[3, 7], &[]), vec![3, 7]);
}

#[test]
fn eval_map_row_select() {
    let m = parse_affine_map("affine_map<(i) -> (i, 0)>").unwrap();
    assert_eq!(m.eval(&[2], &[]), vec![2, 0]);
    assert_eq!(m.eval(&[0], &[]), vec![0, 0]);
}

#[test]
fn eval_map_transposed() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>").unwrap();
    assert_eq!(m.eval(&[3, 7], &[]), vec![7, 3]);
}

#[test]
fn eval_map_constant_offset() {
    let m = parse_affine_map("affine_map<(d0) -> (d0 + 1)>").unwrap();
    assert_eq!(m.eval(&[4], &[]), vec![5]);
}

#[test]
fn eval_map_scaled() {
    let m = parse_affine_map("affine_map<(d0) -> (2 * d0)>").unwrap();
    assert_eq!(m.eval(&[3], &[]), vec![6]);
}

#[test]
#[should_panic]
fn eval_map_wrong_dim_count_panics() {
    // Python raised ValueError("expects 2"); Rust enforces arity via
    // debug_assert in AffineMap::eval (tests run in debug -> panics).
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>").unwrap();
    let _ = m.eval(&[1], &[]);
}

#[test]
fn eval_map_zero_dims() {
    let m = parse_affine_map("affine_map<() -> (0)>").unwrap();
    assert_eq!(m.eval(&[], &[]), vec![0]);
}

// ===========================================================================
// affine_set_contains — TestAffineSetContains
// ===========================================================================

#[test]
fn contains_inside_1d() {
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>").unwrap();
    for i in 0..4 {
        assert!(s.contains(&[i], &[]));
    }
}

#[test]
fn contains_outside_1d() {
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>").unwrap();
    assert!(!s.contains(&[-1], &[]));
    assert!(!s.contains(&[4], &[]));
}

#[test]
fn contains_2d_boundary() {
    let s =
        parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0)>")
            .unwrap();
    assert!(s.contains(&[0, 0], &[]));
    assert!(s.contains(&[1, 1], &[]));
    assert!(!s.contains(&[2, 0], &[]));
}

#[test]
fn contains_general_rhs() {
    // d0 >= d1 and d0 <= 63
    let s = parse_affine_set("affine_set<(d0, d1) : (d0 >= d1, d0 <= 63)>").unwrap();
    assert!(s.contains(&[5, 3], &[])); // 5 >= 3, 5 <= 63
    assert!(s.contains(&[63, 63], &[])); // 63 >= 63, 63 <= 63
    assert!(!s.contains(&[2, 5], &[])); // 2 < 5
    assert!(!s.contains(&[64, 0], &[])); // 64 > 63
}

#[test]
fn contains_symbolic_with_symbol() {
    // (d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0) -> 0 <= d0 <= s0-1
    let s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    assert!(s.contains(&[0], &[8]));
    assert!(s.contains(&[7], &[8]));
    assert!(!s.contains(&[8], &[8]));
    assert!(!s.contains(&[-1], &[8]));
}

// ===========================================================================
// enumerate_affine_set — TestEnumerateAffineSet
// ===========================================================================

#[test]
fn enumerate_1d_range() {
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>").unwrap();
    let pts = s.enumerate(&[32], &[]);
    assert_eq!(pts.len(), 32);
    assert_eq!(pts[0], vec![0]);
    assert_eq!(pts[pts.len() - 1], vec![31]);
}

#[test]
fn enumerate_2d_rect_64x64() {
    let s =
        parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>")
            .unwrap();
    let pts = s.enumerate(&[64, 64], &[]);
    assert_eq!(pts.len(), 64 * 64);
    assert_eq!(pts[0], vec![0, 0]);
    assert_eq!(pts[pts.len() - 1], vec![63, 63]);
}

#[test]
fn enumerate_shape_larger_than_set() {
    // set says d0 in [0,3], shape is (8,) — only 4 points back
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>").unwrap();
    let pts = s.enumerate(&[8], &[]);
    assert_eq!(pts.len(), 4);
    assert!(pts.iter().all(|p| 0 <= p[0] && p[0] <= 3));
}

#[test]
fn enumerate_empty_set() {
    // infeasible: d0 >= 5 and d0 <= 3
    let s = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0, d0 + -5 >= 0)>").unwrap();
    let pts = s.enumerate(&[10], &[]);
    assert_eq!(pts, Vec::<Vec<i64>>::new());
}

#[test]
#[should_panic]
fn enumerate_shape_dim_mismatch_panics() {
    // Python raised ValueError("2 dim"); Rust enforces via assert/debug_assert.
    let s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0)>").unwrap();
    let _ = s.enumerate(&[4], &[]);
}

#[test]
fn enumerate_symbolic_with_symbol() {
    // (d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0) enumerates [0, s0)
    let s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    let pts = s.enumerate(&[16], &[5]);
    assert_eq!(pts, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);
}

#[test]
fn enumerate_symbolic_symbol_larger_than_shape() {
    // When s0 > shape bound, shape acts as the cap
    let s = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    let pts = s.enumerate(&[4], &[100]);
    assert_eq!(pts.len(), 4); // capped by shape
}

// ===========================================================================
// Equality constraints — TestEqualityConstraints (("eq", lhs, rhs) node)
//
// In Rust an equality constraint is `ConstraintKind::Equal` with the normalised
// `expr = lhs - rhs` (so `g == 0` -> expr `g`, Equal). The Python AST stored an
// explicit `("eq", lhs, rhs)`; the normalised Rust form is equivalent.
// ===========================================================================

#[test]
fn tokenise_eq_operator() {
    assert!(
        ktir_emulator::parser_ast::tokenise("(g == 0)")
            .iter()
            .any(|t| t == "==")
    );
}

#[test]
fn tokenise_eq_before_geq() {
    // `==` must be one token, not two `=`; and `>=` distinct.
    let tokens = ktir_emulator::parser_ast::tokenise("(d0 == 1, d1 >= 0)");
    assert!(tokens.iter().any(|t| t == "=="));
    assert!(tokens.iter().any(|t| t == ">="));
    assert!(!tokens.iter().any(|t| t == "="));
}

#[test]
fn parse_eq_simple() {
    let s = parse_affine_set("affine_set<(g) : (g == 0)>").unwrap();
    assert_eq!(s.constraints.len(), 1);
    let c = &s.constraints[0];
    assert_eq!(c.kind, ConstraintKind::Equal);
    // g == 0 normalises to expr `g - 0` (Rust keeps the `- rhs` term verbatim).
    assert_eq!(c.expr, subx(dim(0), cst(0)));
}

#[test]
fn parse_eq_one_node_not_two() {
    // A single == must produce one constraint node, not two.
    let s = parse_affine_set("affine_set<(g) : (g == 0)>").unwrap();
    assert_eq!(s.constraints.len(), 1);
    assert_eq!(s.constraints[0].kind, ConstraintKind::Equal);
}

#[test]
fn parse_eq_with_expression() {
    // p - c + 2 == 0
    let s = parse_affine_set("affine_set<(p)[c] : (p - c + 2 == 0)>").unwrap();
    assert_eq!(s.constraints.len(), 1);
    assert_eq!(s.constraints[0].kind, ConstraintKind::Equal);
}

#[test]
fn parse_eq_complex_lhs_rhs() {
    // p + c - 8*g - 3 == 0
    let s = parse_affine_set("affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>").unwrap();
    assert_eq!(s.constraints.len(), 1);
    assert_eq!(s.constraints[0].kind, ConstraintKind::Equal);
}

#[test]
fn parse_mixed_eq_and_ineq() {
    let s = parse_affine_set("affine_set<(d0, d1) : (d0 == 0, d1 >= 0)>").unwrap();
    assert_eq!(s.constraints.len(), 2);
    assert_eq!(s.constraints[0].kind, ConstraintKind::Equal);
    assert_eq!(s.constraints[1].kind, ConstraintKind::GreaterEq);
}

#[test]
fn eq_contains_matching_point() {
    let s = parse_affine_set("affine_set<(g) : (g == 0)>").unwrap();
    assert!(s.contains(&[0], &[]));
}

#[test]
fn eq_contains_nonmatching_point() {
    let s = parse_affine_set("affine_set<(g) : (g == 0)>").unwrap();
    assert!(!s.contains(&[1], &[]));
    assert!(!s.contains(&[-1], &[]));
}

#[test]
fn eq_enumerate_single_point() {
    let s = parse_affine_set("affine_set<(g) : (g == 0)>").unwrap();
    let pts = s.enumerate(&[4], &[]);
    assert_eq!(pts, vec![vec![0]]);
}

#[test]
fn eq_i_equals_zero() {
    // Spec example: affine_set<(i) : (i == 0)>
    let s = parse_affine_set("affine_set<(i) : (i == 0)>").unwrap();
    assert!(s.contains(&[0], &[]));
    assert!(!s.contains(&[1], &[]));
}

#[test]
fn eq_symbolic_constraint() {
    // p - c + 2 == 0 with symbol c=3 means p == 1.
    let s = parse_affine_set("affine_set<(p)[c] : (p - c + 2 == 0)>").unwrap();
    assert!(s.contains(&[1], &[3])); // 1 - 3 + 2 == 0
    assert!(!s.contains(&[2], &[3]));
}

#[test]
fn eq_complex_symbolic() {
    // p + c - 8*g - 3 == 0 with c=5, g=1 means p == 6.
    let s = parse_affine_set("affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>").unwrap();
    assert!(s.contains(&[6], &[5, 1]));
    assert!(!s.contains(&[5], &[5, 1]));
}

// ===========================================================================
// Edge cases — TestAffineEdgeCases
// ===========================================================================

#[test]
fn triangular_affine_set() {
    // Lower-triangular (d0 >= d1) over a 4x4 box -> 10 points.
    let s = parse_affine_set(
        "affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d0 - d1 >= 0)>",
    )
    .unwrap();
    let pts = s.enumerate(&[4, 4], &[]);
    assert_eq!(pts.len(), 10);
    for p in &pts {
        assert!(p[0] >= p[1], "({},{}) violates d0 >= d1", p[0], p[1]);
    }
}

#[test]
fn triangular_affine_set_sum_constraint() {
    // 4x4 box with d0 + d1 <= 3 -> 10 points.
    let s = parse_affine_set(
        "affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, -d0 + 3 >= 0, -d1 + 3 >= 0, -d0 - d1 + 3 >= 0)>",
    )
    .unwrap();
    let pts = s.enumerate(&[4, 4], &[]);
    assert_eq!(pts.len(), 10);
    for p in &pts {
        assert!(
            p[0] + p[1] <= 3,
            "({},{}) violates d0 + d1 <= 3",
            p[0],
            p[1]
        );
    }
}

#[test]
fn triangular_contains() {
    let s = parse_affine_set("affine_set<(d0, d1) : (d0 >= 0, d1 >= 0, d0 - d1 >= 0)>").unwrap();
    assert!(s.contains(&[3, 1], &[])); // 3 >= 1
    assert!(s.contains(&[2, 2], &[])); // 2 >= 2
    assert!(!s.contains(&[1, 3], &[])); // 1 < 3
}

#[test]
fn conflicting_constraints_empty_set() {
    // d0 >= 5 AND d0 <= 2 is infeasible.
    let s = parse_affine_set("affine_set<(d0) : (d0 - 5 >= 0, -d0 + 2 >= 0)>").unwrap();
    let pts = s.enumerate(&[10], &[]);
    assert_eq!(pts, Vec::<Vec<i64>>::new());
}

#[test]
fn conflicting_constraints_2d_empty() {
    // d0 > d1 AND d1 > d0 is unsatisfiable.
    let s =
        parse_affine_set("affine_set<(d0, d1) : (d0 - d1 - 1 >= 0, d1 - d0 - 1 >= 0)>").unwrap();
    let pts = s.enumerate(&[4, 4], &[]);
    assert_eq!(pts, Vec::<Vec<i64>>::new());
}

#[test]
fn zero_dim_affine_map_parse() {
    let m = parse_affine_map("affine_map<() -> (0)>").unwrap();
    assert_eq!(m.num_dims, 0);
    assert_eq!(m.exprs.len(), 1);
    assert_eq!(m.exprs[0], cst(0));
}

#[test]
fn zero_dim_affine_map_eval() {
    let m = parse_affine_map("affine_map<() -> (42)>").unwrap();
    assert_eq!(m.eval(&[], &[]), vec![42]);
}

#[test]
fn zero_dim_affine_map_multi_output() {
    let m = parse_affine_map("affine_map<() -> (1, 2, 3)>").unwrap();
    assert_eq!(m.num_dims, 0);
    assert_eq!(m.exprs.len(), 3);
    assert_eq!(m.eval(&[], &[]), vec![1, 2, 3]);
}

// ===========================================================================
// Symbolic bound helpers — TestSymBoundHelpers
//
// Bound = i64 (Concrete) | AffineExpr (Symbolic). The Python `("sym", 0)` ->
// Bound::Symbolic(Sym(0)). Idempotent / identity folds mirror parser_ast.
// ===========================================================================

fn bsym(i: usize) -> Bound {
    Bound::Symbolic(Rc::new(sym(i)))
}
fn bcst(c: i64) -> Bound {
    Bound::Concrete(c)
}

#[test]
fn concrete_operands_fold_to_int() {
    assert_eq!(sym_add(&bcst(2), &bcst(3)), bcst(5));
    assert_eq!(sym_neg(&bcst(5)), bcst(-5));
    assert_eq!(sym_max(&bcst(3), &bcst(7)), bcst(7));
    assert_eq!(sym_min(&bcst(3), &bcst(7)), bcst(3));
}

#[test]
fn mvp_simplifications() {
    let s0 = bsym(0);
    // additive identity: 0 + s0 -> s0, s0 + 0 -> s0
    assert_eq!(sym_add(&bcst(0), &s0), s0);
    assert_eq!(sym_add(&s0, &bcst(0)), s0);
    // double negation collapses: -(-s0) -> s0. The collapse in `sym_neg` fires
    // on an actual `Neg` node (Python's `("neg", s0)`), so build that directly
    // rather than via the `-1 * x` normalising helper.
    let neg_s0 = Bound::Symbolic(Rc::new(AffineExpr::Neg(Rc::new(sym(0)))));
    assert_eq!(sym_neg(&neg_s0), s0);
    // max/min idempotent on same SymRef (compare-by-value)
    assert_eq!(sym_max(&s0, &bsym(0)), s0);
    assert_eq!(sym_min(&s0, &bsym(0)), s0);
}

#[test]
fn int_operand_wrapped_as_const_node() {
    // When one side is symbolic, the int side gets wrapped in Const so the AST
    // is well-formed.
    let s0 = bsym(0);
    assert_eq!(
        sym_add(&bcst(5), &s0),
        Bound::Symbolic(Rc::new(add(cst(5), sym(0))))
    );
    assert_eq!(
        sym_max(&bcst(0), &s0),
        Bound::Symbolic(Rc::new(AffineExpr::Max(Rc::new(cst(0)), Rc::new(sym(0)))))
    );
    assert_eq!(
        sym_min(&s0, &bcst(10)),
        Bound::Symbolic(Rc::new(AffineExpr::Min(Rc::new(sym(0)), Rc::new(cst(10)))))
    );
}

#[test]
fn eval_bound_round_trip() {
    // eval_bound on a plain concrete short-circuits.
    assert_eq!(eval_bound(&bcst(7), &[]), 7);
    // AST nodes evaluate to the same value as plain arithmetic on resolved syms.
    assert_eq!(eval_bound(&sym_add(&bsym(0), &bcst(1)), &[128]), 129);
    assert_eq!(eval_bound(&sym_neg(&bsym(0)), &[3]), -3);
    for (a, b) in [(3i64, 7i64), (-2, 0), (10, 5)] {
        assert_eq!(eval_bound(&sym_max(&bsym(0), &bsym(1)), &[a, b]), a.max(b));
        assert_eq!(eval_bound(&sym_min(&bsym(0), &bsym(1)), &[a, b]), a.min(b));
    }
}
