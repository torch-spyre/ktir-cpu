// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Affine-text recursive-descent parser — Rust port of
//! `ktir_cpu/parser_ast.py` (the `_tokenise` / `_Parser` / `parse_affine_*`
//! half). Turns `affine_map<(d0,...) -> (e0,...)>` and
//! `affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>` text into the
//! [`AffineMap`](crate::affine::AffineMap) / [`AffineSet`](crate::affine::AffineSet)
//! value types declared in `affine.rs`.
//!
//! The Python AST is a tag-tuple soup (`("add", l, r)`, `("neg", x)`, ...).
//! Our target [`AffineExpr`](crate::affine::AffineExpr) has no `Sub`/`Neg`
//! constructor, so we normalise during construction:
//!
//!   * `a - b`  -> `Add(a, Mul(Const(-1), b))`
//!   * `-a`     -> `Mul(Const(-1), a)`
//!
//! Evaluation is therefore identical to MLIR's, and `affine.rs`'s existing
//! `eval` handles the result unchanged.
//!
//! Constraints follow the Python normalisation: `lhs >= rhs` and `lhs <= rhs`
//! both become a single `expr >= 0` ([`ConstraintKind::GreaterEq`]) over
//! `lhs - rhs` / `rhs - lhs`; `lhs == rhs` becomes `expr == 0`
//! ([`ConstraintKind::Equal`]) over `lhs - rhs`.

use crate::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Tokeniser — mirrors `_tokenise` / `_TOKEN_RE` in parser_ast.py.
//
// The Python regex matches, in order: `%name`, bare identifier, integer
// literal (possibly negative), and the operator set `== >= <= -> + - * ( ) ,
// : [ ]`. Whitespace is skipped. We reproduce the same token set with a manual
// scanner so the crate stays dependency-free (matching parser.rs's style).
// ---------------------------------------------------------------------------

/// Split affine-attribute text into the flat token stream the recursive-descent
/// parser consumes. Faithful to `_tokenise`: `%name` and bare identifiers, signed
/// integer literals, the multi-char operators `== >= <= ->`, and the single-char
/// punctuation `+ - * ( ) , : [ ]`. Unknown characters are skipped (matching the
/// Python `pos += 1` fall-through on a non-match).
pub fn tokenise(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;

        // Whitespace — skip.
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Two-char operators: `== >= <= ->`. `==` is matched before `>=` to
        // mirror the regex alternation order (group 4 in `_TOKEN_RE`).
        if i + 1 < bytes.len() {
            let pair = &text[i..i + 2];
            if matches!(pair, "==" | ">=" | "<=" | "->") {
                tokens.push(pair.to_string());
                i += 2;
                continue;
            }
        }

        // `%name` reference.
        if c == '%' {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i] as char) {
                i += 1;
            }
            tokens.push(text[start..i].to_string());
            continue;
        }

        // Bare identifier (letters / `_` then alphanumerics / `_`).
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i] as char) {
                i += 1;
            }
            tokens.push(text[start..i].to_string());
            continue;
        }

        // Integer literal. A leading `-` is consumed as part of the number only
        // when it is immediately followed by a digit — otherwise it is the
        // subtraction / unary-minus operator (handled below). This matches the
        // regex group 3 `(-?\d+)` taking precedence over the `-` operator.
        if c.is_ascii_digit()
            || (c == '-' && i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_digit())
        {
            let start = i;
            if c == '-' {
                i += 1;
            }
            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }
            tokens.push(text[start..i].to_string());
            continue;
        }

        // Single-char punctuation / operators.
        if matches!(c, '+' | '-' | '*' | '(' | ')' | ',' | ':' | '[' | ']') {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }

        // Unknown char — skip (Python falls through to `pos += 1`).
        i += 1;
    }
    tokens
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// ---------------------------------------------------------------------------
// Recursive-descent parser — mirrors the `_Parser` class.
//
// The parser is stateful (token cursor + name->index maps). It produces
// `AffineExpr` directly rather than the tag-tuple AST, normalising `sub`/`neg`
// into `Add`/`Mul(Const(-1), ...)` as it builds nodes.
// ---------------------------------------------------------------------------

struct Parser {
    tokens: Vec<String>,
    pos: usize,
    /// dim name -> positional index; populated by the caller after the dim list
    /// is parsed, exactly like `_Parser.dim_index`.
    dim_index: Vec<(String, usize)>,
    /// sym name -> positional index; mirrors `_Parser.sym_index`.
    sym_index: Vec<(String, usize)>,
}

impl Parser {
    fn new(tokens: Vec<String>) -> Self {
        Parser {
            tokens,
            pos: 0,
            dim_index: Vec::new(),
            sym_index: Vec::new(),
        }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    /// Advance one token, returning it. With `expected`, errors on mismatch —
    /// mirrors `_Parser.consume`.
    fn consume(&mut self, expected: Option<&str>) -> Result<String, String> {
        let tok = self
            .tokens
            .get(self.pos)
            .ok_or_else(|| "Unexpected end of expression".to_string())?
            .clone();
        if let Some(exp) = expected
            && tok != exp
        {
            return Err(format!("Expected {exp:?}, got {tok:?} (pos {})", self.pos));
        }
        self.pos += 1;
        Ok(tok)
    }

    // --- grammar ---

    /// Parse `(d0, d1, ...)` -> name list. Does NOT populate `dim_index`.
    fn parse_dim_list(&mut self) -> Result<Vec<String>, String> {
        self.consume(Some("("))?;
        let mut names = Vec::new();
        while self.peek() != Some(")") {
            names.push(self.consume(None)?);
            if self.peek() == Some(",") {
                self.consume(Some(","))?;
            }
        }
        self.consume(Some(")"))?;
        Ok(names)
    }

    /// Parse the optional `[s0, s1, ...]` symbol list. Empty when absent.
    fn parse_sym_list(&mut self) -> Result<Vec<String>, String> {
        if self.peek() != Some("[") {
            return Ok(Vec::new());
        }
        self.consume(Some("["))?;
        let mut names = Vec::new();
        while self.peek() != Some("]") {
            names.push(self.consume(None)?);
            if self.peek() == Some(",") {
                self.consume(Some(","))?;
            }
        }
        self.consume(Some("]"))?;
        Ok(names)
    }

    fn parse_expr(&mut self) -> Result<AffineExpr, String> {
        self.additive()
    }

    /// Left-associative `+` / `-`. `-` lowers to `Add(left, Mul(-1, right))`.
    fn additive(&mut self) -> Result<AffineExpr, String> {
        let mut left = self.term()?;
        while matches!(self.peek(), Some("+") | Some("-")) {
            let op = self.consume(None)?;
            let right = self.term()?;
            left = if op == "+" {
                AffineExpr::Add(Rc::new(left), Rc::new(right))
            } else {
                AffineExpr::Add(Rc::new(left), Rc::new(neg(right)))
            };
        }
        Ok(left)
    }

    /// Unary minus and constant-coefficient multiplication. Mirrors `_term`:
    /// `N * expr`, `expr * N`, bare `N`, and `-expr`.
    fn term(&mut self) -> Result<AffineExpr, String> {
        // Unary minus -> `Mul(Const(-1), atom)`.
        if self.peek() == Some("-") {
            self.consume(Some("-"))?;
            let operand = self.atom()?;
            return Ok(neg(operand));
        }

        // Integer that may be a coefficient: `N * expr`.
        if let Some(tok) = self.peek()
            && is_int_literal(tok)
        {
            let num: i64 = self
                .consume(None)?
                .parse()
                .map_err(|_| "bad integer literal".to_string())?;
            if self.peek() == Some("*") {
                self.consume(Some("*"))?;
                let operand = self.atom()?;
                return Ok(AffineExpr::Mul(
                    Rc::new(AffineExpr::Const(num)),
                    Rc::new(operand),
                ));
            }
            return Ok(AffineExpr::Const(num));
        }

        // Atom that may be followed by a coefficient: `expr * N`.
        let node = self.atom()?;
        if self.peek() == Some("*") {
            self.consume(Some("*"))?;
            let num_tok = self.peek().map(|s| s.to_string());
            match num_tok {
                Some(ref t) if is_int_literal(t) => {
                    let num: i64 = self
                        .consume(None)?
                        .parse()
                        .map_err(|_| "bad integer coefficient".to_string())?;
                    Ok(AffineExpr::Mul(
                        Rc::new(AffineExpr::Const(num)),
                        Rc::new(node),
                    ))
                }
                other => Err(format!(
                    "Expected integer coefficient after '*', got {other:?}"
                )),
            }
        } else {
            Ok(node)
        }
    }

    /// Base unit: parenthesised sub-expr, `%ref`, dim / sym variable, or const.
    /// Mirrors `_atom`, including the canonical `dN` / `sN` suffix fallback used
    /// when the dim/sym maps are empty (a bare `parse_expr` call).
    fn atom(&mut self) -> Result<AffineExpr, String> {
        let tok = self
            .peek()
            .ok_or_else(|| "Unexpected end of expression".to_string())?
            .to_string();

        // Parenthesised sub-expression.
        if tok == "(" {
            self.consume(Some("("))?;
            let node = self.parse_expr()?;
            self.consume(Some(")"))?;
            return Ok(node);
        }

        // `%name` reference. The existing `AffineExpr` has no `Ref` variant; in
        // the ktdp affine attributes a `%name` never appears inside an
        // `affine_map`/`affine_set` body (those reference only d/s variables),
        // so a ref here is an error rather than a silently-dropped node.
        if tok.starts_with('%') {
            return Err(format!(
                "affine expression cannot reference SSA value {tok:?}"
            ));
        }

        // Dimension variable named in the dim list.
        if let Some(idx) = lookup(&self.dim_index, &tok) {
            self.consume(None)?;
            return Ok(AffineExpr::Dim(idx));
        }
        // Fallback: canonical `dN` when no dim map is present.
        if self.dim_index.is_empty()
            && let Some(n) = canonical_index(&tok, 'd')
        {
            self.consume(None)?;
            return Ok(AffineExpr::Dim(n));
        }

        // Symbol variable named in the symbol list.
        if let Some(idx) = lookup(&self.sym_index, &tok) {
            self.consume(None)?;
            return Ok(AffineExpr::Sym(idx));
        }
        // Fallback: canonical `sN` when no sym map is present.
        if self.sym_index.is_empty()
            && let Some(n) = canonical_index(&tok, 's')
        {
            self.consume(None)?;
            return Ok(AffineExpr::Sym(n));
        }

        // Positive integer constant (negatives are consumed in `term`).
        if is_unsigned_int(&tok) {
            let num: i64 = self
                .consume(None)?
                .parse()
                .map_err(|_| "bad integer constant".to_string())?;
            return Ok(AffineExpr::Const(num));
        }

        Err(format!("Unexpected token: {tok:?}"))
    }

    /// Parse `(e0, e1, ...)` -> expression list. Mirrors `parse_expr_list`.
    fn parse_expr_list(&mut self) -> Result<Vec<AffineExpr>, String> {
        self.consume(Some("("))?;
        let mut exprs = Vec::new();
        while self.peek() != Some(")") {
            exprs.push(self.parse_expr()?);
            if self.peek() == Some(",") {
                self.consume(Some(","))?;
            }
        }
        self.consume(Some(")"))?;
        Ok(exprs)
    }

    /// Parse `(lhs OP rhs, ...)` constraint list, normalising each into a single
    /// `expr {>=,==} 0` [`Constraint`]. Mirrors `parse_constraint_list`.
    fn parse_constraint_list(&mut self) -> Result<Vec<Constraint>, String> {
        self.consume(Some("("))?;
        let mut constraints = Vec::new();
        while self.peek() != Some(")") {
            let lhs = self.parse_expr()?;
            let op = self.consume(None)?; // ">=", "<=", or "=="
            let rhs = self.parse_expr()?;
            let constraint = match op.as_str() {
                // `lhs >= rhs`  ->  `lhs - rhs >= 0`
                ">=" => Constraint {
                    expr: sub(lhs, rhs),
                    kind: ConstraintKind::GreaterEq,
                },
                // `lhs <= rhs`  ->  `rhs - lhs >= 0`
                "<=" => Constraint {
                    expr: sub(rhs, lhs),
                    kind: ConstraintKind::GreaterEq,
                },
                // `lhs == rhs`  ->  `lhs - rhs == 0`
                "==" => Constraint {
                    expr: sub(lhs, rhs),
                    kind: ConstraintKind::Equal,
                },
                other => return Err(format!("Unsupported constraint operator: {other:?}")),
            };
            constraints.push(constraint);
            if self.peek() == Some(",") {
                self.consume(Some(","))?;
            }
        }
        self.consume(Some(")"))?;
        Ok(constraints)
    }
}

// ---------------------------------------------------------------------------
// AST construction helpers — normalise `sub` / `neg` into the Add/Mul-only
// `AffineExpr`.
// ---------------------------------------------------------------------------

/// `-a` as `Mul(Const(-1), a)`.
fn neg(a: AffineExpr) -> AffineExpr {
    AffineExpr::Mul(Rc::new(AffineExpr::Const(-1)), Rc::new(a))
}

/// `a - b` as `Add(a, Mul(Const(-1), b))`.
fn sub(a: AffineExpr, b: AffineExpr) -> AffineExpr {
    AffineExpr::Add(Rc::new(a), Rc::new(neg(b)))
}

/// Linear lookup in a name->index assoc list (lists are tiny — at most a handful
/// of dims/syms — so a Vec keeps construction order without a HashMap).
fn lookup(map: &[(String, usize)], name: &str) -> Option<usize> {
    map.iter().find(|(n, _)| n == name).map(|(_, i)| *i)
}

fn build_index_map(names: &[String]) -> Vec<(String, usize)> {
    names
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, n)| (n, i))
        .collect()
}

/// `dN` / `sN` canonical suffix extraction (prefix char + decimal digits).
fn canonical_index(tok: &str, prefix: char) -> Option<usize> {
    let mut chars = tok.chars();
    if chars.next()? != prefix {
        return None;
    }
    let rest = &tok[1..];
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn is_int_literal(tok: &str) -> bool {
    let body = tok.strip_prefix('-').unwrap_or(tok);
    !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit())
}

fn is_unsigned_int(tok: &str) -> bool {
    !tok.is_empty() && tok.bytes().all(|b| b.is_ascii_digit())
}

// ---------------------------------------------------------------------------
// Outer-wrapper stripping — mirrors `_strip_outer`.
// ---------------------------------------------------------------------------

/// Strip a `keyword<...>` wrapper, returning the inner text. The wrapper is
/// optional: text that does not start with `keyword` is returned unchanged
/// (the caller may pass inner text directly). Text that starts with `keyword`
/// but is not a well-formed `keyword<...>` is an error — matching the Python
/// `fullmatch` + `startswith` logic.
fn strip_outer<'a>(s: &'a str, keyword: &str) -> Result<&'a str, String> {
    let s = s.trim();
    let open = format!("{keyword}<");
    if let Some(rest) = s.strip_prefix(&open) {
        if let Some(inner) = rest.strip_suffix('>') {
            return Ok(inner);
        }
        return Err(format!("Malformed {keyword} expression: {s:?}"));
    }
    if s.starts_with(keyword) {
        return Err(format!("Malformed {keyword} expression: {s:?}"));
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// Public parse functions — mirror `parse_affine_map` / `parse_affine_set_raw`.
// ---------------------------------------------------------------------------

/// Parse `affine_map<(d0,...) -> (e0,...)>` into an [`AffineMap`]. The wrapper is
/// optional. Mirrors `parse_affine_map`.
pub fn parse_affine_map(s: &str) -> Result<AffineMap, String> {
    let inner = strip_outer(s.trim(), "affine_map")?;
    let tokens = tokenise(inner);
    let mut p = Parser::new(tokens);
    let dim_names = p.parse_dim_list()?;
    p.dim_index = build_index_map(&dim_names);
    p.consume(Some("->"))?;
    let exprs = p.parse_expr_list()?;
    Ok(AffineMap {
        num_dims: dim_names.len(),
        num_syms: 0,
        exprs,
    })
}

/// Parse `affine_set<(d0,...)[s0,...] : (c0 >= 0, ...)>` into an [`AffineSet`]
/// (no `BoxSet` lowering). The wrapper and `[s0,...]` symbol list are optional;
/// the `:` separator is required. Mirrors `parse_affine_set_raw`.
pub fn parse_affine_set(s: &str) -> Result<AffineSet, String> {
    let inner = strip_outer(s.trim(), "affine_set")?;
    let colon = inner
        .find(':')
        .ok_or_else(|| format!("affine_set missing ':' separator: {inner:?}"))?;
    let dim_part = inner[..colon].trim();
    let con_part = inner[colon + 1..].trim();

    // Dim list and optional symbol list, e.g. `(d0)[s0]`.
    let mut p1 = Parser::new(tokenise(dim_part));
    let dim_names = p1.parse_dim_list()?;
    let sym_names = p1.parse_sym_list()?;

    // Constraints share the dim/sym index maps so they can reference both.
    let mut p2 = Parser::new(tokenise(con_part));
    p2.dim_index = build_index_map(&dim_names);
    p2.sym_index = build_index_map(&sym_names);
    let constraints = p2.parse_constraint_list()?;

    Ok(AffineSet {
        num_dims: dim_names.len(),
        num_syms: sym_names.len(),
        constraints,
    })
}

/// Parse a single bare affine expression (no wrapper). Mirrors the testing
/// helper `parse_expr`; the canonical `dN`/`sN` fallback is in effect because
/// no dim/sym maps are populated.
pub fn parse_expr(s: &str) -> Result<AffineExpr, String> {
    let mut p = Parser::new(tokenise(s.trim()));
    p.parse_expr()
}

// ---------------------------------------------------------------------------
// Normalisation predicates used by the construct_access_tile parser to drop
// trivial attributes (mirrors `AffineMap.is_identity` / `AffineSet.is_full`).
// ---------------------------------------------------------------------------

/// True when `map` is the identity map of its own rank: `(d0,...,dn) ->
/// (d0,...,dn)` with `num_syms == 0`. Mirrors `AffineMap.is_identity`; used to
/// normalise `access_tile_order` to absent.
pub fn is_identity_map(map: &AffineMap) -> bool {
    map.num_syms == 0
        && map.num_dims == map.exprs.len()
        && map
            .exprs
            .iter()
            .enumerate()
            .all(|(i, e)| matches!(e, AffineExpr::Dim(d) if *d == i))
}

/// True when `set` admits every point of the rectangular box `[0, shape)` —
/// i.e. it constrains nothing within the tile. Mirrors `AffineSet.is_full`;
/// used to normalise `access_tile_set` to absent so load/store can take the
/// contiguous fast path. Brute-force over `[0, shape)`, like
/// `enumerate_affine_set` + a count compare.
pub fn is_full_set(set: &AffineSet, shape: &[usize]) -> bool {
    if set.num_dims != shape.len() {
        return false;
    }
    // Symbolic sets are never treated as trivially full (we have no concrete
    // symbol values here), matching the Python guard.
    if set.num_syms != 0 {
        return false;
    }
    let syms: [i64; 0] = [];
    enumerate_box(shape).all(|pt| set.contains(&pt, &syms))
}

/// Iterate every integer point of the box `[0, shape)` in row-major order.
fn enumerate_box(shape: &[usize]) -> impl Iterator<Item = Vec<i64>> + '_ {
    let total: usize = shape.iter().product();
    (0..total).map(move |mut lin| {
        let mut pt = vec![0i64; shape.len()];
        for axis in (0..shape.len()).rev() {
            let s = shape[axis].max(1);
            pt[axis] = (lin % s) as i64;
            lin /= s;
        }
        pt
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::AffineExpr;

    // --- tokeniser ---

    #[test]
    fn tokenise_operators_and_idents() {
        assert_eq!(
            tokenise("(d0) -> (d0 + 1)"),
            vec!["(", "d0", ")", "->", "(", "d0", "+", "1", ")"]
        );
        // `==` matched before `>=`; `>=` and `->` are single tokens. The `-`
        // before `d0` is the unary-minus operator (only `-<digit>` fuses into a
        // negative literal), matching the Python `(-?\d+)` regex group.
        assert_eq!(
            tokenise("d0 >= 0, -d0 + 127 >= 0"),
            vec!["d0", ">=", "0", ",", "-", "d0", "+", "127", ">=", "0"]
        );
        assert_eq!(tokenise("a == b"), vec!["a", "==", "b"]);
    }

    #[test]
    fn tokenise_negative_literal_vs_minus_operator() {
        // `-127` is a single literal here (the `-` abuts a digit).
        assert_eq!(tokenise("-127"), vec!["-127"]);
        // `d0 - 1` keeps `-` as an operator when spaced away from the digit.
        assert_eq!(tokenise("d0 - 1"), vec!["d0", "-", "1"]);
    }

    // --- affine map ---

    #[test]
    fn parse_identity_map() {
        let m = parse_affine_map("affine_map<(d0) -> (d0)>").unwrap();
        assert_eq!(m.num_dims, 1);
        assert_eq!(m.num_syms, 0);
        assert_eq!(m.exprs, vec![AffineExpr::Dim(0)]);
        assert!(is_identity_map(&m));
        // Evaluates as identity.
        assert_eq!(m.eval(&[5], &[]), vec![5]);
    }

    #[test]
    fn parse_map_named_dims_and_arithmetic() {
        // Non-canonical dim names resolve positionally.
        let m = parse_affine_map("affine_map<(i, j) -> (i + 2 * j, j)>").unwrap();
        assert_eq!(m.num_dims, 2);
        // (i + 2*j, j) at (3, 4) -> (11, 4)
        assert_eq!(m.eval(&[3, 4], &[]), vec![11, 4]);
        assert!(!is_identity_map(&m));
    }

    #[test]
    fn parse_map_subtraction_normalises() {
        // d0 - d1 must evaluate as subtraction even though AffineExpr lacks Sub.
        let m = parse_affine_map("affine_map<(d0, d1) -> (d0 - d1)>").unwrap();
        assert_eq!(m.eval(&[10, 3], &[]), vec![7]);
    }

    #[test]
    fn map_wrapper_is_optional() {
        let m = parse_affine_map("(d0, d1) -> (d1, d0)").unwrap();
        assert_eq!(m.eval(&[1, 2], &[]), vec![2, 1]);
    }

    // --- affine set ---

    #[test]
    fn parse_set_bounds_membership() {
        // 0 <= d0 <= 127, exactly the vector_add access_tile_set.
        let set = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>").unwrap();
        assert_eq!(set.num_dims, 1);
        assert_eq!(set.num_syms, 0);
        assert_eq!(set.constraints.len(), 2);
        let syms: [i64; 0] = [];
        assert!(set.contains(&[0], &syms));
        assert!(set.contains(&[127], &syms));
        assert!(!set.contains(&[128], &syms));
        assert!(!set.contains(&[-1], &syms));
    }

    #[test]
    fn parse_set_le_normalises_to_geq() {
        // `d0 <= 5` should accept 0..=5 and reject 6.
        let set = parse_affine_set("affine_set<(d0) : (d0 >= 0, d0 <= 5)>").unwrap();
        let syms: [i64; 0] = [];
        assert!(set.contains(&[5], &syms));
        assert!(!set.contains(&[6], &syms));
    }

    #[test]
    fn parse_set_equality() {
        let set = parse_affine_set("affine_set<(d0, d1) : (d0 == 3, d1 >= 0)>").unwrap();
        assert_eq!(set.constraints[0].kind, ConstraintKind::Equal);
        let syms: [i64; 0] = [];
        assert!(set.contains(&[3, 0], &syms));
        assert!(!set.contains(&[4, 0], &syms));
    }

    #[test]
    fn parse_set_with_symbols() {
        // (d0)[s0] : 0 <= d0 <= s0 - 1
        let set = parse_affine_set("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
        assert_eq!(set.num_syms, 1);
        assert!(set.contains(&[0], &[4]));
        assert!(set.contains(&[3], &[4]));
        assert!(!set.contains(&[4], &[4]));
    }

    // --- normalisation predicates ---

    #[test]
    fn is_full_set_detects_unconstrained_box() {
        // 0 <= d0 <= 127 is full over a 128-extent tile.
        let full = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>").unwrap();
        assert!(is_full_set(&full, &[128]));
        // ... but not over a 256-extent tile (points 128..255 excluded).
        assert!(!is_full_set(&full, &[256]));
        // A genuine restriction (only d0 == 0) is not full.
        let partial = parse_affine_set("affine_set<(d0) : (d0 == 0)>").unwrap();
        assert!(!is_full_set(&partial, &[128]));
    }

    #[test]
    fn is_identity_map_distinguishes_permutation() {
        let id = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>").unwrap();
        assert!(is_identity_map(&id));
        let perm = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>").unwrap();
        assert!(!is_identity_map(&perm));
    }

    // --- bare expression helper ---

    #[test]
    fn parse_expr_canonical_fallback() {
        // No surrounding map/set: dN/sN resolve by suffix.
        let e = parse_expr("-d0 + 2 * d1 + 3").unwrap();
        // (-d0 + 2 d1 + 3) at d0=1, d1=4 -> -1 + 8 + 3 = 10
        assert_eq!(e.eval(&[1, 4], &[]), 10);
        let s = parse_expr("s0 + 1").unwrap();
        assert_eq!(s.eval(&[], &[7]), 8);
    }

    #[test]
    fn strip_outer_rejects_malformed() {
        assert!(parse_affine_map("affine_map<(d0) -> (d0)").is_err());
        assert!(parse_affine_set("affine_set<(d0) (d0 >= 0)>").is_err()); // missing ':'
    }
}
