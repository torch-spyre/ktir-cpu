// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Affine maps and sets — Rust port of `ktir_cpu/affine.py`.
//!
//! `AffineMap` is a pure function over dimension + symbol values (frozen
//! dataclass in Python -> immutable value type here). `BoxSet` is the
//! axis-aligned fast path with O(ndim) containment; `AffineSet` is the general
//! constraint-based set.
//!
//! Two box flavours coexist here:
//!
//! * [`BoxSet`] is the original concrete, **inclusive** `[lo, hi]` box used by
//!   the arith/ktdp slice (the partition origin is `min(coordinate_set)`).
//!   It is kept verbatim so its existing call sites and tests stay green.
//! * [`SymBoxSet`] is the faithful port of the Python `BoxSet`: a half-open
//!   `[lo, hi)` box whose per-axis bounds may be a concrete `i64` (fast path)
//!   or a symbolic [`Bound`] over symbol variables. It carries the parity
//!   surface — `enumerate` / `is_empty` / `is_full` / `lower_bounds` /
//!   `specialize` / `translate` / `intersect` / `try_from_affine_set`.
//!
//! The symbolic-bound helpers (`eval_bound`, `sym_add`, `sym_neg`, `sym_max`,
//! `sym_min`) mirror `parser_ast.py` 1:1, including its minimal constant
//! folding (concrete-on-concrete, additive identity, idempotent `sym`).

use std::rc::Rc;

/// Recursive affine-expression AST: `Dim`, `Sym`, `Const`, and the operators
/// MLIR affine exprs support plus the `Max`/`Min`/`Neg`/`Sub`/`Ref` shapes the
/// symbolic-bound layer constructs. `Rc`-recursive so clones are cheap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineExpr {
    Dim(usize),
    Sym(usize),
    Const(i64),
    /// A named, domain-specific reference atom (`"ref"` in the Python AST).
    /// Never linearizable — its presence forces the constraint slow path.
    Ref(String),
    // Children are `Rc`, not `Box`: the affine tree is immutable after parsing,
    // so cloning a whole expression (done per access-tile construction, per
    // K-tile per node in the kernels) is a refcount bump instead of a deep
    // Box-tree copy — killing the `AffineExpr` clone/drop the flamegraph flagged.
    Add(Rc<AffineExpr>, Rc<AffineExpr>),
    Sub(Rc<AffineExpr>, Rc<AffineExpr>),
    Neg(Rc<AffineExpr>),
    Mul(Rc<AffineExpr>, Rc<AffineExpr>),
    FloorDiv(Rc<AffineExpr>, Rc<AffineExpr>),
    Mod(Rc<AffineExpr>, Rc<AffineExpr>),
    Max(Rc<AffineExpr>, Rc<AffineExpr>),
    Min(Rc<AffineExpr>, Rc<AffineExpr>),
}

impl AffineExpr {
    /// Evaluate against concrete dimension and symbol values. Mirrors
    /// `parser_ast._eval_node`.
    pub fn eval(&self, dims: &[i64], syms: &[i64]) -> i64 {
        match self {
            AffineExpr::Dim(i) => dims[*i],
            AffineExpr::Sym(i) => syms[*i],
            AffineExpr::Const(c) => *c,
            AffineExpr::Ref(name) => panic!("cannot evaluate ref atom {name:?}"),
            AffineExpr::Add(a, b) => a.eval(dims, syms) + b.eval(dims, syms),
            AffineExpr::Sub(a, b) => a.eval(dims, syms) - b.eval(dims, syms),
            AffineExpr::Neg(a) => -a.eval(dims, syms),
            AffineExpr::Mul(a, b) => a.eval(dims, syms) * b.eval(dims, syms),
            // MLIR affine floordiv/mod are Euclidean (floor toward -inf).
            AffineExpr::FloorDiv(a, b) => a.eval(dims, syms).div_euclid(b.eval(dims, syms)),
            AffineExpr::Mod(a, b) => a.eval(dims, syms).rem_euclid(b.eval(dims, syms)),
            AffineExpr::Max(a, b) => a.eval(dims, syms).max(b.eval(dims, syms)),
            AffineExpr::Min(a, b) => a.eval(dims, syms).min(b.eval(dims, syms)),
        }
    }
}

/// A pure multi-result affine map: `(d0, d1)[s0] -> (expr, expr, ...)`.
/// Frozen/immutable, like the Python `@dataclass(frozen=True)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineMap {
    pub num_dims: usize,
    pub num_syms: usize,
    pub exprs: Vec<AffineExpr>,
}

impl AffineMap {
    /// Identity map of rank `n` — synthesized when MLIR omits `base_map`,
    /// matching `AccessTile.base_map` ("synthesized as identity if absent").
    pub fn identity(n: usize) -> Self {
        AffineMap {
            num_dims: n,
            num_syms: 0,
            exprs: (0..n).map(AffineExpr::Dim).collect(),
        }
    }

    /// Evaluate every result expression. Mirrors `AffineMap.eval`.
    pub fn eval(&self, dims: &[i64], syms: &[i64]) -> Vec<i64> {
        debug_assert_eq!(dims.len(), self.num_dims, "dim arity mismatch");
        debug_assert_eq!(syms.len(), self.num_syms, "sym arity mismatch");
        self.exprs.iter().map(|e| e.eval(dims, syms)).collect()
    }

    /// True iff this map is the identity: `output[i] == d_i` for every `i`.
    ///
    /// Implemented structurally via [`match_pure_dim_ref`]: each output
    /// expression must flatten to `1 * d_i + 0` with the output position `i`
    /// matching the dim index. A probe-based `eval(probe) == probe` check would
    /// wrongly accept maps like `(d0, d1) -> (d1 - 1, d0 + 1)`.
    pub fn is_identity(&self) -> bool {
        if self.exprs.len() != self.num_dims {
            return false;
        }
        for (i, expr) in self.exprs.iter().enumerate() {
            if match_pure_dim_ref(expr, self.num_dims) != Some(i) {
                return false;
            }
        }
        true
    }

    /// True iff this map permutes its input dimensions.
    ///
    /// Square (output count == input count), each output is a single dim
    /// variable, and every dim index appears exactly once. Accepts coordinate
    /// permutations like `(d0, d1, d2) -> (d2, d0, d1)`; rejects shears,
    /// scalings, constant offsets, and many-to-one collapses.
    pub fn is_permutation(&self) -> bool {
        if self.exprs.len() != self.num_dims {
            return false;
        }
        let mut seen = vec![false; self.num_dims];
        for expr in &self.exprs {
            match match_pure_dim_ref(expr, self.num_dims) {
                Some(idx) if !seen[idx] => seen[idx] = true,
                _ => return false,
            }
        }
        true
    }

    /// If every result expression is a plain dimension reference `d_i`, return
    /// the referenced dim index for each result — the map's projection /
    /// permutation pattern. `None` if any result is a non-trivial affine
    /// expression (shear, scaling, constant offset, sum of dims).
    ///
    /// Unlike [`is_permutation`], this does not require the map to be square:
    /// a matmul indexing map like `(d0, d1, d2) -> (d1, d2)` projects three
    /// iteration dims onto a 2-D operand and yields `Some(vec![1, 2])`.
    pub fn result_dims(&self) -> Option<Vec<usize>> {
        self.exprs
            .iter()
            .map(|e| match_pure_dim_ref(e, self.num_dims))
            .collect()
    }
}

/// Axis-aligned integer box `[lo, hi]` **inclusive** — the original concrete
/// fast path of `CoordinateSet`. O(ndim) containment and intersection.
///
/// This is the concrete, dim-only box used by the arith/ktdp slice; see
/// [`SymBoxSet`] for the half-open symbolic-bound port of the Python `BoxSet`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxSet {
    pub lo: Vec<i64>,
    pub hi: Vec<i64>,
}

impl BoxSet {
    pub fn new(lo: Vec<i64>, hi: Vec<i64>) -> Self {
        assert_eq!(lo.len(), hi.len(), "BoxSet lo/hi rank mismatch");
        BoxSet { lo, hi }
    }

    /// `min(coordinate_set)` = lower corner; used as the partition origin
    /// (`p_i`) in `distributed_tile_access`.
    pub fn origin(&self) -> &[i64] {
        &self.lo
    }

    pub fn contains(&self, point: &[i64]) -> bool {
        point.len() == self.lo.len()
            && point
                .iter()
                .zip(&self.lo)
                .zip(&self.hi)
                .all(|((&p, &lo), &hi)| lo <= p && p <= hi)
    }

    /// Per-axis intersection; `None` if the boxes are disjoint on any axis.
    /// Mirrors `BoxSet.intersect` (which returns an empty box on no overlap).
    pub fn intersect(&self, other: &BoxSet) -> Option<BoxSet> {
        assert_eq!(self.lo.len(), other.lo.len(), "BoxSet rank mismatch");
        let mut lo = Vec::with_capacity(self.lo.len());
        let mut hi = Vec::with_capacity(self.hi.len());
        for i in 0..self.lo.len() {
            let l = self.lo[i].max(other.lo[i]);
            let h = self.hi[i].min(other.hi[i]);
            if l > h {
                return None;
            }
            lo.push(l);
            hi.push(h);
        }
        Some(BoxSet { lo, hi })
    }
}

/// A per-axis bound on a [`SymBoxSet`]: either a concrete `i64` (fast path) or
/// a symbolic AST node over symbol variables only (no `Dim` nodes). Concrete
/// bounds stay unwrapped so the structural fast path can identify them without
/// walking the AST — the `Bound = Union[int, tuple]` of `parser_ast.py`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bound {
    Concrete(i64),
    Symbolic(Rc<AffineExpr>),
}

impl Bound {
    /// True iff this bound is a plain `i64` (the `isinstance(b, int)` check).
    pub fn is_concrete(&self) -> bool {
        matches!(self, Bound::Concrete(_))
    }
}

impl From<i64> for Bound {
    fn from(v: i64) -> Self {
        Bound::Concrete(v)
    }
}

impl From<AffineExpr> for Bound {
    fn from(e: AffineExpr) -> Self {
        // Fold a bare constant node into the concrete leaf so structural fast
        // paths keep working — mirrors `("const", k)` never appearing alone.
        match e {
            AffineExpr::Const(c) => Bound::Concrete(c),
            other => Bound::Symbolic(Rc::new(other)),
        }
    }
}

/// Evaluate a [`Bound`] against concrete `symbols`. Concrete ints short-circuit
/// without touching the AST. Mirrors `parser_ast.eval_bound`.
pub fn eval_bound(b: &Bound, symbols: &[i64]) -> i64 {
    match b {
        Bound::Concrete(c) => *c,
        // BoxSet bounds never reference dim variables by construction.
        Bound::Symbolic(node) => node.eval(&[], symbols),
    }
}

/// Build `a + b` over [`Bound`] operands with constant folding. Folds when both
/// are concrete; absorbs additive identity (`a + 0 -> a`). Mirrors
/// `parser_ast.sym_add`.
pub fn sym_add(a: &Bound, b: &Bound) -> Bound {
    match (a, b) {
        (Bound::Concrete(x), Bound::Concrete(y)) => Bound::Concrete(x + y),
        (Bound::Concrete(0), _) => b.clone(),
        (_, Bound::Concrete(0)) => a.clone(),
        _ => Bound::Symbolic(Rc::new(AffineExpr::Add(
            Rc::new(bound_to_node(a)),
            Rc::new(bound_to_node(b)),
        ))),
    }
}

/// Build `-a` over a [`Bound`] with constant folding and double-negation
/// collapse (`-(-x) -> x`). Mirrors `parser_ast.sym_neg`.
pub fn sym_neg(a: &Bound) -> Bound {
    match a {
        Bound::Concrete(c) => Bound::Concrete(-c),
        Bound::Symbolic(node) => match node.as_ref() {
            AffineExpr::Neg(inner) => Bound::from((**inner).clone()),
            other => Bound::Symbolic(Rc::new(AffineExpr::Neg(Rc::new(other.clone())))),
        },
    }
}

/// Build `max(a, b)` over [`Bound`] operands with MVP folding: concrete-on
/// -concrete folds; identical `Sym(k)` references are idempotent. No deeper
/// canonicalisation (per-axis candidate count is <= 2). Mirrors
/// `parser_ast.sym_max`.
pub fn sym_max(a: &Bound, b: &Bound) -> Bound {
    sym_minmax(a, b, true)
}

/// Build `min(a, b)`; mirror of [`sym_max`] (`parser_ast.sym_min`).
pub fn sym_min(a: &Bound, b: &Bound) -> Bound {
    sym_minmax(a, b, false)
}

fn sym_minmax(a: &Bound, b: &Bound, is_max: bool) -> Bound {
    if let (Bound::Concrete(x), Bound::Concrete(y)) = (a, b) {
        return Bound::Concrete(if is_max { *x.max(y) } else { *x.min(y) });
    }
    // Idempotent on identical symbol references: max(s_k, s_k) -> s_k.
    if let (Bound::Symbolic(na), Bound::Symbolic(nb)) = (a, b)
        && let (AffineExpr::Sym(i), AffineExpr::Sym(j)) = (na.as_ref(), nb.as_ref())
        && i == j
    {
        return a.clone();
    }
    let an = Rc::new(bound_to_node(a));
    let bn = Rc::new(bound_to_node(b));
    let node = if is_max {
        AffineExpr::Max(an, bn)
    } else {
        AffineExpr::Min(an, bn)
    };
    Bound::Symbolic(Rc::new(node))
}

/// Lift a [`Bound`] to an [`AffineExpr`] node (wraps concrete ints in `Const`),
/// matching `("const", a) if isinstance(a, int) else a`.
fn bound_to_node(b: &Bound) -> AffineExpr {
    match b {
        Bound::Concrete(c) => AffineExpr::Const(*c),
        Bound::Symbolic(node) => (**node).clone(),
    }
}

/// General affine set: a conjunction of affine constraints `expr >= 0` or
/// `expr == 0`. Containment substitutes the point and checks every constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineSet {
    pub num_dims: usize,
    pub num_syms: usize,
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub expr: AffineExpr,
    pub kind: ConstraintKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `expr >= 0`
    GreaterEq,
    /// `expr == 0`
    Equal,
}

impl AffineSet {
    /// Point membership: every constraint must hold. Mirrors `AffineSet.contains`.
    pub fn contains(&self, point: &[i64], syms: &[i64]) -> bool {
        self.constraints.iter().all(|c| {
            let v = c.expr.eval(point, syms);
            match c.kind {
                ConstraintKind::GreaterEq => v >= 0,
                ConstraintKind::Equal => v == 0,
            }
        })
    }

    /// All integer points in `[0, shape)` satisfying every constraint, in
    /// row-major (lexicographic, rightmost-innermost) order. Mirrors
    /// `parser_ast.enumerate_affine_set`.
    ///
    /// Panics on a `shape`/`num_dims` rank mismatch (the Python `ValueError`).
    pub fn enumerate(&self, shape: &[usize], syms: &[i64]) -> Vec<Vec<i64>> {
        assert_eq!(
            shape.len(),
            self.num_dims,
            "AffineSet has {} dim(s), got shape with {}",
            self.num_dims,
            shape.len()
        );
        let mut out = Vec::new();
        product(shape, &mut |pt| {
            if self.contains(pt, syms) {
                out.push(pt.to_vec());
            }
        });
        out
    }

    /// True iff this set covers every coordinate in `shape` (i.e. `[0, shape)`).
    ///
    /// Vertex check: an affine set is convex, so it covers `[0, shape)` iff it
    /// contains all `2^n_dims` corners of the box — `O(2^n_dims)` constraint
    /// evaluations instead of `O(∏ shape)`. Mirrors `AffineSet.is_full`.
    pub fn is_full(&self, shape: &[usize]) -> bool {
        if shape.len() != self.num_dims {
            return false;
        }
        // Empty extent on any axis means there are no corners to span.
        if shape.contains(&0) {
            return false;
        }
        let n = self.num_dims;
        // Enumerate the 2^n corners: each axis takes {0, shape[d]-1}.
        for mask in 0..(1u64 << n) {
            let corner: Vec<i64> = (0..n)
                .map(|d| {
                    if (mask >> d) & 1 == 0 {
                        0
                    } else {
                        shape[d] as i64 - 1
                    }
                })
                .collect();
            if !self.contains(&corner, &[]) {
                return false;
            }
        }
        true
    }

    /// Conjoin two affine sets: the intersection is every constraint of both.
    ///
    /// A point lies in the intersection iff it satisfies all constraints, so
    /// the conjunction of the constraint lists is exactly the intersection set.
    /// Both operands must agree on dim/symbol arity.
    pub fn intersect(&self, other: &AffineSet) -> AffineSet {
        assert_eq!(
            self.num_dims, other.num_dims,
            "AffineSet.intersect: n_dims mismatch {} vs {}",
            self.num_dims, other.num_dims
        );
        assert_eq!(
            self.num_syms, other.num_syms,
            "AffineSet.intersect: n_syms mismatch {} vs {}",
            self.num_syms, other.num_syms
        );
        let mut constraints = self.constraints.clone();
        constraints.extend(other.constraints.iter().cloned());
        AffineSet {
            num_dims: self.num_dims,
            num_syms: self.num_syms,
            constraints,
        }
    }
}

/// Half-open axis-aligned box `{p : lo[d] <= p[d] < hi[d]}` — the faithful port
/// of the Python `BoxSet`. Per-axis bounds may be concrete `i64` or symbolic
/// [`Bound`]s over symbol variables. `all_concrete` is cached at construction
/// so the hot path (`contains` / `is_empty`) skips per-element AST checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymBoxSet {
    pub lo: Vec<Bound>, // inclusive
    pub hi: Vec<Bound>, // exclusive
    all_concrete: bool,
}

impl SymBoxSet {
    /// Construct from per-axis bounds, caching the all-concrete flag. Panics on
    /// a `lo`/`hi` length mismatch (the Python `ValueError`).
    pub fn new(lo: Vec<Bound>, hi: Vec<Bound>) -> Self {
        assert_eq!(
            lo.len(),
            hi.len(),
            "SymBoxSet: lo/hi length mismatch: lo={lo:?} hi={hi:?}"
        );
        let all_concrete = lo.iter().all(Bound::is_concrete) && hi.iter().all(Bound::is_concrete);
        SymBoxSet {
            lo,
            hi,
            all_concrete,
        }
    }

    /// Convenience constructor from concrete `i64` bounds.
    pub fn from_concrete(lo: Vec<i64>, hi: Vec<i64>) -> Self {
        SymBoxSet::new(
            lo.into_iter().map(Bound::Concrete).collect(),
            hi.into_iter().map(Bound::Concrete).collect(),
        )
    }

    pub fn n_dims(&self) -> usize {
        self.lo.len()
    }

    /// True iff every `lo`/`hi` entry is a concrete `i64` (cached flag).
    pub fn is_concrete(&self) -> bool {
        self.all_concrete
    }

    /// True iff `lo[d] <= point[d] < hi[d]` for every dim. `symbols` resolves
    /// symbolic bounds; concrete boxes ignore it. Mirrors `BoxSet.contains`.
    pub fn contains(&self, point: &[i64], symbols: &[i64]) -> bool {
        if point.len() != self.n_dims() {
            return false;
        }
        (0..self.n_dims()).all(|d| {
            let lo = eval_bound(&self.lo[d], symbols);
            let hi = eval_bound(&self.hi[d], symbols);
            lo <= point[d] && point[d] < hi
        })
    }

    /// All integer points in the box in row-major (lexicographic) order.
    ///
    /// A `BoxSet` is self-bounded, so `shape` only serves as a sanity check:
    /// passed values must upper-bound `hi` componentwise. Symbolic boxes are
    /// specialised first. Mirrors `BoxSet.enumerate`. Panics on rank mismatch
    /// or when `hi[d] > shape[d]`.
    pub fn enumerate(&self, shape: Option<&[usize]>, symbols: &[i64]) -> Vec<Vec<i64>> {
        let boxed = if self.all_concrete {
            self.clone()
        } else {
            self.specialize(symbols)
        };
        // Concrete after specialise: read the bounds out as ints.
        let los: Vec<i64> = boxed.lo.iter().map(|b| eval_bound(b, &[])).collect();
        let his: Vec<i64> = boxed.hi.iter().map(|b| eval_bound(b, &[])).collect();
        if let Some(shape) = shape {
            assert_eq!(
                shape.len(),
                boxed.n_dims(),
                "SymBoxSet.enumerate: shape ndim {} does not match box ndim {}",
                shape.len(),
                boxed.n_dims()
            );
            for d in 0..boxed.n_dims() {
                assert!(
                    his[d] <= shape[d] as i64,
                    "SymBoxSet.enumerate: hi[{d}]={} exceeds shape[{d}]={} — box is not \
                     contained in the nominal bounding box.",
                    his[d],
                    shape[d]
                );
            }
        }
        // itertools.product over range(lo[d], hi[d]) per axis; rightmost dim
        // is innermost, giving lexicographic (row-major) order.
        let mut out = Vec::new();
        box_product(&los, &his, &mut |pt| out.push(pt.to_vec()));
        out
    }

    /// True iff any axis has an empty extent (`hi[d] <= lo[d]`). Symbolic boxes
    /// are resolved against `symbols` first. Mirrors `BoxSet.is_empty`.
    pub fn is_empty(&self, symbols: &[i64]) -> bool {
        (0..self.n_dims())
            .any(|d| eval_bound(&self.hi[d], symbols) <= eval_bound(&self.lo[d], symbols))
    }

    /// True iff this box equals `[0, shape)` exactly. A translated box
    /// `[x, x + shape)` returns `false` even when per-axis extent matches —
    /// the asymmetry is intentional (callers use `true` as licence to drop
    /// `coordinate_set` -> `None`). Mirrors `BoxSet.is_full`.
    pub fn is_full(&self, shape: &[usize], symbols: &[i64]) -> bool {
        if shape.len() != self.n_dims() {
            return false;
        }
        let spec = if self.all_concrete {
            self.clone()
        } else {
            self.specialize(symbols)
        };
        (0..self.n_dims()).all(|d| {
            eval_bound(&spec.lo[d], &[]) == 0 && eval_bound(&spec.hi[d], &[]) == shape[d] as i64
        })
    }

    /// Return `lo` — the per-axis minimum coordinate — resolved to `i64`.
    /// Used to get the partition origin in `distributed_tile_access`. Mirrors
    /// `BoxSet.lower_bounds`.
    pub fn lower_bounds(&self, symbols: &[i64]) -> Vec<i64> {
        self.lo.iter().map(|b| eval_bound(b, symbols)).collect()
    }

    /// Return a concrete `SymBoxSet` with all symbolic bounds resolved.
    /// Concrete boxes are returned unchanged (cached flag). Mirrors
    /// `BoxSet.specialize`.
    pub fn specialize(&self, symbols: &[i64]) -> SymBoxSet {
        if self.all_concrete {
            return self.clone();
        }
        SymBoxSet::new(
            self.lo
                .iter()
                .map(|b| Bound::Concrete(eval_bound(b, symbols)))
                .collect(),
            self.hi
                .iter()
                .map(|b| Bound::Concrete(eval_bound(b, symbols)))
                .collect(),
        )
    }

    /// Return a new box shifted by `offset` along each axis. `offset` may carry
    /// symbolic entries; `sym_add` folds concrete-on-concrete so a static box
    /// translated by a static offset stays concrete. Mirrors `BoxSet.translate`.
    /// Panics on an offset dim mismatch.
    pub fn translate(&self, offset: &[Bound]) -> SymBoxSet {
        assert_eq!(
            offset.len(),
            self.n_dims(),
            "SymBoxSet.translate: offset dim mismatch: offset={offset:?} n_dims={}",
            self.n_dims()
        );
        SymBoxSet::new(
            (0..self.n_dims())
                .map(|d| sym_add(&self.lo[d], &offset[d]))
                .collect(),
            (0..self.n_dims())
                .map(|d| sym_add(&self.hi[d], &offset[d]))
                .collect(),
        )
    }

    /// Axis-wise intersection; the result may be empty (check via `is_empty`).
    /// Uses `sym_max`/`sym_min` so concrete-on-concrete folds to ints. Mirrors
    /// `BoxSet.intersect`. Panics on a dim mismatch.
    pub fn intersect(&self, other: &SymBoxSet) -> SymBoxSet {
        assert_eq!(
            other.n_dims(),
            self.n_dims(),
            "SymBoxSet.intersect: n_dims mismatch {} vs {}",
            self.n_dims(),
            other.n_dims()
        );
        SymBoxSet::new(
            (0..self.n_dims())
                .map(|d| sym_max(&self.lo[d], &other.lo[d]))
                .collect(),
            (0..self.n_dims())
                .map(|d| sym_min(&self.hi[d], &other.hi[d]))
                .collect(),
        )
    }

    /// Lower an axis-aligned [`AffineSet`] to a `SymBoxSet`, or `None` when the
    /// set is not representable as an integer box.
    ///
    /// Lowering succeeds iff every constraint has the form `c * d_i + k(syms)
    /// >= 0` or `c * d_i + k(syms) == 0` with `c ∈ {+1, -1}` (single dim, unit
    /// coeff) and every axis is pinned on **both** sides. `k(syms)` may be an
    /// int constant or a linear combination of symbols (in which case the
    /// resulting bound carries an AST node). Equality constraints pin both
    /// `lo[i]` and `hi[i] = pin + 1` (exclusive). Inequality/equality bounds on
    /// the same axis combine with `sym_max` (lo) / `sym_min` (hi). Assumes all
    /// symbols `s_i >= 0`. Mirrors `BoxSet.try_from_affine_set`.
    pub fn try_from_affine_set(aset: &AffineSet) -> Option<SymBoxSet> {
        let n = aset.num_dims;
        let n_syms = aset.num_syms;
        let mut los: Vec<Option<Bound>> = vec![None; n];
        let mut his: Vec<Option<Bound>> = vec![None; n];

        for c in &aset.constraints {
            let is_eq = c.kind == ConstraintKind::Equal;
            // For an equality `lhs == 0` we already store the LHS in `expr`, so
            // the linearised form is the constraint expression directly.
            let (dim_coeffs, sym_coeffs, const_) = constraint_to_linear_syms(&c.expr, n, n_syms)?;
            let nz: Vec<usize> = dim_coeffs
                .iter()
                .enumerate()
                .filter(|&(_, &k)| k != 0)
                .map(|(i, _)| i)
                .collect();
            if nz.len() != 1 {
                return None;
            }
            let i = nz[0];
            let k = dim_coeffs[i];
            if k.abs() != 1 {
                return None;
            }
            // Build k(syms): int constant + sum(sym_coeffs[j] * s_j).
            let sym_term = build_sym_term(&sym_coeffs, const_);
            if is_eq {
                // k*d_i + k(syms) == 0  ->  d_i == pin
                let pin = if k == 1 { sym_neg(&sym_term) } else { sym_term };
                let pin_hi = sym_add(&pin, &Bound::Concrete(1));
                los[i] = Some(match &los[i] {
                    None => pin.clone(),
                    Some(cur) => sym_max(cur, &pin),
                });
                his[i] = Some(match &his[i] {
                    None => pin_hi.clone(),
                    Some(cur) => sym_min(cur, &pin_hi),
                });
            } else if k == 1 {
                // d_i + k(syms) >= 0  ->  d_i >= -k(syms)
                let candidate = sym_neg(&sym_term);
                los[i] = Some(match &los[i] {
                    None => candidate,
                    Some(cur) => sym_max(cur, &candidate),
                });
            } else {
                // -d_i + k(syms) >= 0  ->  d_i <= k(syms)  ->  hi excl = k+1
                let candidate = sym_add(&sym_term, &Bound::Concrete(1));
                his[i] = Some(match &his[i] {
                    None => candidate,
                    Some(cur) => sym_min(cur, &candidate),
                });
            }
        }

        if los.iter().any(Option::is_none) || his.iter().any(Option::is_none) {
            return None;
        }
        let los: Vec<Bound> = los.into_iter().map(Option::unwrap).collect();
        let his: Vec<Bound> = his.into_iter().map(Option::unwrap).collect();

        // Concrete boxes: detect contradictions early (e.g. d0 >= 5, d0 <= 3).
        // Symbolic boxes may resolve to contradictions at specialize time;
        // callers detect that via is_empty(symbols=...) after specialising.
        if los.iter().all(Bound::is_concrete) && his.iter().all(Bound::is_concrete) {
            for i in 0..n {
                if eval_bound(&los[i], &[]) >= eval_bound(&his[i], &[]) {
                    return None;
                }
            }
        }
        Some(SymBoxSet::new(los, his))
    }
}

/// Match `node` against `1 * d_i + 0` and return `i`, else `None`. A "pure dim
/// ref" flattens (via [`constraint_to_linear`]) to exactly one dim variable
/// with unit coefficient and zero constant. Cannot be fooled by linear
/// combinations whose evaluation on a probe coincides with the probe. Mirrors
/// `_match_pure_dim_ref`.
fn match_pure_dim_ref(node: &AffineExpr, n_dims: usize) -> Option<usize> {
    let (coeffs, const_) = constraint_to_linear(node, n_dims)?;
    if const_ != 0 {
        return None;
    }
    let nz: Vec<usize> = coeffs
        .iter()
        .enumerate()
        .filter(|&(_, &k)| k != 0)
        .map(|(i, _)| i)
        .collect();
    if nz.len() != 1 || coeffs[nz[0]] != 1 {
        return None;
    }
    Some(nz[0])
}

/// Flatten a dim-only constraint AST into `(coeffs, const)` representing
/// `sum(coeffs[i] * d_i) + const >= 0`. A thin wrapper over
/// [`constraint_to_linear_syms`] with `n_syms = 0` — any `Sym` atom trips the
/// guard and returns `None`. Mirrors `_constraint_to_linear`.
fn constraint_to_linear(node: &AffineExpr, n_dims: usize) -> Option<(Vec<i64>, i64)> {
    let (dim_coeffs, _sym_coeffs, const_) = constraint_to_linear_syms(node, n_dims, 0)?;
    Some((dim_coeffs, const_))
}

/// Reassemble a [`Bound`] from `sum(sym_coeffs[j] * s_j) + const`. Returns a
/// plain concrete `i64` when no symbol contributes — the structural fast path
/// on concrete bounds depends on that. Mirrors `_build_sym_term`.
fn build_sym_term(sym_coeffs: &[i64], const_: i64) -> Bound {
    let mut expr = Bound::Concrete(const_);
    for (j, &c) in sym_coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let sym = AffineExpr::Sym(j);
        let term: Bound = if c == -1 {
            sym_neg(&Bound::Symbolic(Rc::new(sym)))
        } else if c != 1 {
            Bound::Symbolic(Rc::new(AffineExpr::Mul(
                Rc::new(AffineExpr::Const(c)),
                Rc::new(sym),
            )))
        } else {
            Bound::Symbolic(Rc::new(sym))
        };
        expr = sym_add(&expr, &term);
    }
    expr
}

/// Flatten a parsed constraint AST into `(dim_coeffs, sym_coeffs, const)`
/// representing `sum(dim_coeffs[i] * d_i) + sum(sym_coeffs[j] * s_j) + const`.
/// Returns `None` if the expression is not separable into that form (a `Ref`
/// atom, a sym×dim product, or any non-linear structure). Mirrors
/// `_constraint_to_linear_syms`.
fn constraint_to_linear_syms(
    node: &AffineExpr,
    n_dims: usize,
    n_syms: usize,
) -> Option<(Vec<i64>, Vec<i64>, i64)> {
    let mut dim_coeffs = vec![0i64; n_dims];
    let mut sym_coeffs = vec![0i64; n_syms];
    let mut const_ = 0i64;

    fn walk(
        n: &AffineExpr,
        sign: i64,
        dim_coeffs: &mut [i64],
        sym_coeffs: &mut [i64],
        const_: &mut i64,
        n_syms: usize,
    ) -> bool {
        match n {
            AffineExpr::Const(c) => {
                *const_ += sign * c;
                true
            }
            AffineExpr::Dim(i) => {
                if *i >= dim_coeffs.len() {
                    return false;
                }
                dim_coeffs[*i] += sign;
                true
            }
            AffineExpr::Sym(j) => {
                if *j >= n_syms {
                    return false;
                }
                sym_coeffs[*j] += sign;
                true
            }
            AffineExpr::Add(a, b) => {
                walk(a, sign, dim_coeffs, sym_coeffs, const_, n_syms)
                    && walk(b, sign, dim_coeffs, sym_coeffs, const_, n_syms)
            }
            AffineExpr::Sub(a, b) => {
                walk(a, sign, dim_coeffs, sym_coeffs, const_, n_syms)
                    && walk(b, -sign, dim_coeffs, sym_coeffs, const_, n_syms)
            }
            AffineExpr::Neg(a) => walk(a, -sign, dim_coeffs, sym_coeffs, const_, n_syms),
            AffineExpr::Mul(lhs, inner) => {
                // The surface form is `coef * inner` with `coef` an int literal.
                let coef = match lhs.as_ref() {
                    AffineExpr::Const(c) => *c,
                    _ => return false,
                };
                match inner.as_ref() {
                    AffineExpr::Dim(i) => {
                        if *i >= dim_coeffs.len() {
                            return false;
                        }
                        dim_coeffs[*i] += sign * coef;
                        true
                    }
                    AffineExpr::Sym(j) => {
                        if *j >= n_syms {
                            return false;
                        }
                        sym_coeffs[*j] += sign * coef;
                        true
                    }
                    AffineExpr::Const(c) => {
                        *const_ += sign * coef * c;
                        true
                    }
                    _ => false,
                }
            }
            // 'ref' or anything else — not a separable linear combination.
            _ => false,
        }
    }

    if !walk(
        node,
        1,
        &mut dim_coeffs,
        &mut sym_coeffs,
        &mut const_,
        n_syms,
    ) {
        return None;
    }
    Some((dim_coeffs, sym_coeffs, const_))
}

/// Iterate `itertools.product(range(s) for s in shape)` in lexicographic order
/// (rightmost dim innermost), calling `f` with each point. Used by
/// [`AffineSet::enumerate`].
fn product(shape: &[usize], f: &mut impl FnMut(&[i64])) {
    let mut idx = vec![0i64; shape.len()];
    if shape.contains(&0) {
        return;
    }
    loop {
        f(&idx);
        // Increment rightmost-first (innermost dim moves fastest).
        let mut d = shape.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if (idx[d] as usize) < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// Iterate `itertools.product(range(lo[d], hi[d]))` in lexicographic order.
/// Used by [`SymBoxSet::enumerate`]; empty if any axis has `hi <= lo`.
fn box_product(lo: &[i64], hi: &[i64], f: &mut impl FnMut(&[i64])) {
    let n = lo.len();
    if (0..n).any(|d| hi[d] <= lo[d]) {
        return;
    }
    let mut idx: Vec<i64> = lo.to_vec();
    loop {
        f(&idx);
        let mut d = n;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < hi[d] {
                break;
            }
            idx[d] = lo[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers for building expressions concisely.
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
    fn sub(a: AffineExpr, b: AffineExpr) -> AffineExpr {
        AffineExpr::Sub(Rc::new(a), Rc::new(b))
    }
    fn neg(a: AffineExpr) -> AffineExpr {
        AffineExpr::Neg(Rc::new(a))
    }
    fn mul(c: i64, a: AffineExpr) -> AffineExpr {
        AffineExpr::Mul(Rc::new(cst(c)), Rc::new(a))
    }

    #[test]
    fn map_eval_and_identity() {
        // (d0, d1)[s0] -> (d0 + s0, d1 * 2)
        let m = AffineMap {
            num_dims: 2,
            num_syms: 1,
            exprs: vec![
                AffineExpr::Add(Rc::new(AffineExpr::Dim(0)), Rc::new(AffineExpr::Sym(0))),
                AffineExpr::Mul(Rc::new(AffineExpr::Dim(1)), Rc::new(AffineExpr::Const(2))),
            ],
        };
        assert_eq!(m.eval(&[5, 7], &[10]), vec![15, 14]);
        assert_eq!(AffineMap::identity(3).eval(&[1, 2, 3], &[]), vec![1, 2, 3]);
    }

    #[test]
    fn euclidean_floordiv_and_mod() {
        let fd = AffineExpr::FloorDiv(Rc::new(AffineExpr::Dim(0)), Rc::new(AffineExpr::Const(4)));
        let m = AffineExpr::Mod(Rc::new(AffineExpr::Dim(0)), Rc::new(AffineExpr::Const(4)));
        // -1 floordiv 4 == -1, -1 mod 4 == 3 (matches MLIR / Python semantics)
        assert_eq!(fd.eval(&[-1], &[]), -1);
        assert_eq!(m.eval(&[-1], &[]), 3);
    }

    #[test]
    fn expr_sub_neg_max_min_eval() {
        // 7 - d0
        assert_eq!(sub(cst(7), dim(0)).eval(&[3], &[]), 4);
        // -d0
        assert_eq!(neg(dim(0)).eval(&[5], &[]), -5);
        // max(d0, d1), min(d0, d1)
        let mx = AffineExpr::Max(Rc::new(dim(0)), Rc::new(dim(1)));
        let mn = AffineExpr::Min(Rc::new(dim(0)), Rc::new(dim(1)));
        assert_eq!(mx.eval(&[3, 8], &[]), 8);
        assert_eq!(mn.eval(&[3, 8], &[]), 3);
    }

    #[test]
    fn box_contains_and_intersect() {
        let a = BoxSet::new(vec![0, 0], vec![3, 3]);
        let b = BoxSet::new(vec![2, 2], vec![5, 5]);
        assert!(a.contains(&[1, 2]));
        assert!(!a.contains(&[4, 0]));
        assert_eq!(a.origin(), &[0, 0]);
        assert_eq!(a.intersect(&b), Some(BoxSet::new(vec![2, 2], vec![3, 3])));

        let disjoint = BoxSet::new(vec![10, 10], vec![11, 11]);
        assert_eq!(a.intersect(&disjoint), None);
    }

    #[test]
    fn affine_set_membership() {
        // { (d0) : d0 >= 0, 7 - d0 >= 0 }  ==  0 <= d0 <= 7
        let set = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::Dim(0),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: AffineExpr::Add(
                        Rc::new(AffineExpr::Const(7)),
                        Rc::new(AffineExpr::Mul(
                            Rc::new(AffineExpr::Const(-1)),
                            Rc::new(AffineExpr::Dim(0)),
                        )),
                    ),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        };
        assert!(set.contains(&[0], &[]));
        assert!(set.contains(&[7], &[]));
        assert!(!set.contains(&[8], &[]));
        assert!(!set.contains(&[-1], &[]));
    }

    // ---- new parity tests ----

    #[test]
    fn map_is_identity() {
        assert!(AffineMap::identity(3).is_identity());
        // (d0, d1) -> (d1, d0) is a permutation, not identity.
        let swap = AffineMap {
            num_dims: 2,
            num_syms: 0,
            exprs: vec![dim(1), dim(0)],
        };
        assert!(!swap.is_identity());
        // (d0) -> (d0 + 1) is not identity (nonzero const).
        let shifted = AffineMap {
            num_dims: 1,
            num_syms: 0,
            exprs: vec![add(dim(0), cst(1))],
        };
        assert!(!shifted.is_identity());
        // (d0, d1) -> (d0 + d1 - d1, d1) flattens to d0 — identity at pos 0.
        let folded = AffineMap {
            num_dims: 2,
            num_syms: 0,
            exprs: vec![sub(add(dim(0), dim(1)), dim(1)), dim(1)],
        };
        assert!(folded.is_identity());
    }

    #[test]
    fn map_is_permutation() {
        let perm = AffineMap {
            num_dims: 3,
            num_syms: 0,
            exprs: vec![dim(2), dim(0), dim(1)],
        };
        assert!(perm.is_permutation());
        assert!(!perm.is_identity());
        // Repeated dim index -> not a permutation.
        let dup = AffineMap {
            num_dims: 2,
            num_syms: 0,
            exprs: vec![dim(0), dim(0)],
        };
        assert!(!dup.is_permutation());
        // Scaling -> not a permutation.
        let scaled = AffineMap {
            num_dims: 1,
            num_syms: 0,
            exprs: vec![mul(2, dim(0))],
        };
        assert!(!scaled.is_permutation());
        // Identity is also a permutation.
        assert!(AffineMap::identity(2).is_permutation());
    }

    #[test]
    fn affine_set_enumerate_and_full() {
        // { (d0, d1) : d0 + d1 - 2 >= 0 }  over shape (3, 3): keep d0+d1 >= 2.
        let set = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: sub(add(dim(0), dim(1)), cst(2)),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        let pts = set.enumerate(&[3, 3], &[]);
        // Lexicographic order: (0,2),(1,1),(1,2),(2,0),(2,1),(2,2)
        assert_eq!(
            pts,
            vec![
                vec![0, 2],
                vec![1, 1],
                vec![1, 2],
                vec![2, 0],
                vec![2, 1],
                vec![2, 2],
            ]
        );
        assert!(!set.is_full(&[3, 3]));

        // A trivially-true set is full.
        let full = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        assert!(full.is_full(&[3, 3]));
        assert_eq!(full.enumerate(&[2, 2], &[]).len(), 4);
    }

    #[test]
    fn affine_set_intersect() {
        // A: d0 >= 0 ; B: 3 - d0 >= 0 ; A ∩ B == 0 <= d0 <= 3.
        let a = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        let b = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: sub(cst(3), dim(0)),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        let c = a.intersect(&b);
        assert_eq!(c.constraints.len(), 2);
        assert_eq!(
            c.enumerate(&[10], &[]),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
    }

    #[test]
    fn symbox_concrete_basics() {
        // [0,3) x [0,3) — a 3x3 box.
        let b = SymBoxSet::from_concrete(vec![0, 0], vec![3, 3]);
        assert!(b.is_concrete());
        assert_eq!(b.n_dims(), 2);
        assert!(b.contains(&[1, 2], &[]));
        assert!(!b.contains(&[3, 0], &[])); // hi is exclusive
        assert!(!b.is_empty(&[]));
        assert!(b.is_full(&[3, 3], &[]));
        // Translated box is not "full".
        let t = b.translate(&[Bound::Concrete(1), Bound::Concrete(0)]);
        assert_eq!(t.lo, vec![Bound::Concrete(1), Bound::Concrete(0)]);
        assert!(!t.is_full(&[3, 3], &[]));
        assert_eq!(b.lower_bounds(&[]), vec![0, 0]);
    }

    #[test]
    fn symbox_enumerate_lexicographic() {
        let b = SymBoxSet::from_concrete(vec![0, 0], vec![2, 3]);
        let pts = b.enumerate(None, &[]);
        assert_eq!(
            pts,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ]
        );
        // Sanity-check shape: hi must be <= shape componentwise.
        assert_eq!(b.enumerate(Some(&[2, 3]), &[]).len(), 6);
    }

    #[test]
    fn symbox_empty_and_intersect() {
        let a = SymBoxSet::from_concrete(vec![0, 0], vec![4, 4]);
        let b = SymBoxSet::from_concrete(vec![2, 2], vec![6, 6]);
        let c = a.intersect(&b);
        assert_eq!(c.lo, vec![Bound::Concrete(2), Bound::Concrete(2)]);
        assert_eq!(c.hi, vec![Bound::Concrete(4), Bound::Concrete(4)]);
        assert!(!c.is_empty(&[]));

        // Disjoint -> empty extent.
        let d = SymBoxSet::from_concrete(vec![10], vec![12]);
        let e = SymBoxSet::from_concrete(vec![0], vec![3]);
        assert!(d.intersect(&e).is_empty(&[]));
    }

    #[test]
    fn symbox_symbolic_specialize() {
        // lo = [s0], hi = [s0 + 2]  — a width-2 window starting at s0.
        let lo = Bound::Symbolic(Rc::new(sym(0)));
        let hi = sym_add(&Bound::Symbolic(Rc::new(sym(0))), &Bound::Concrete(2));
        let b = SymBoxSet::new(vec![lo], vec![hi]);
        assert!(!b.is_concrete());
        // contains uses symbols to resolve bounds.
        assert!(b.contains(&[5], &[5])); // s0=5: [5,7)
        assert!(b.contains(&[6], &[5]));
        assert!(!b.contains(&[7], &[5]));
        // lower_bounds resolves the symbolic origin.
        assert_eq!(b.lower_bounds(&[5]), vec![5]);
        // specialize produces a concrete box.
        let spec = b.specialize(&[5]);
        assert!(spec.is_concrete());
        assert_eq!(spec.lo, vec![Bound::Concrete(5)]);
        assert_eq!(spec.hi, vec![Bound::Concrete(7)]);
        assert_eq!(spec.enumerate(None, &[]), vec![vec![5], vec![6]]);
    }

    #[test]
    fn sym_helpers_fold() {
        // concrete + concrete folds
        assert_eq!(
            sym_add(&Bound::Concrete(2), &Bound::Concrete(3)),
            Bound::Concrete(5)
        );
        // additive identity
        assert_eq!(
            sym_add(&Bound::Concrete(0), &Bound::Symbolic(Rc::new(sym(0)))),
            Bound::Symbolic(Rc::new(sym(0)))
        );
        // double-negation collapse
        let s = Bound::Symbolic(Rc::new(sym(1)));
        assert_eq!(sym_neg(&sym_neg(&s)), s);
        // concrete min/max fold
        assert_eq!(
            sym_max(&Bound::Concrete(2), &Bound::Concrete(7)),
            Bound::Concrete(7)
        );
        assert_eq!(
            sym_min(&Bound::Concrete(2), &Bound::Concrete(7)),
            Bound::Concrete(2)
        );
        // idempotent on identical symbol refs
        let sk = Bound::Symbolic(Rc::new(sym(3)));
        assert_eq!(sym_max(&sk, &sk), sk);
    }

    #[test]
    fn lower_concrete_axis_aligned_set() {
        // { (d0, d1) : d0 >= 0, 3 - d0 >= 0, d1 - 1 >= 0, 4 - d1 >= 0 }
        // => d0 in [0,3], d1 in [1,4] => box lo=[0,1] hi=[4,5] (exclusive).
        let set = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![
                Constraint {
                    expr: dim(0),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: sub(cst(3), dim(0)),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: sub(dim(1), cst(1)),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: sub(cst(4), dim(1)),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        };
        let b = SymBoxSet::try_from_affine_set(&set).expect("should lower");
        assert_eq!(b.lo, vec![Bound::Concrete(0), Bound::Concrete(1)]);
        assert_eq!(b.hi, vec![Bound::Concrete(4), Bound::Concrete(5)]);
        assert!(b.is_concrete());
    }

    #[test]
    fn lower_rejects_non_axis_aligned() {
        // d0 + d1 >= 0 mixes two dims in one constraint -> not lowerable, and
        // axes are not pinned on both sides.
        let set = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: add(dim(0), dim(1)),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        assert!(SymBoxSet::try_from_affine_set(&set).is_none());

        // Non-unit coefficient -> reject.
        let set2 = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![
                Constraint {
                    expr: mul(2, dim(0)),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: sub(cst(3), dim(0)),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        };
        assert!(SymBoxSet::try_from_affine_set(&set2).is_none());
    }

    #[test]
    fn lower_contradiction_returns_none() {
        // d0 >= 5 and 3 - d0 >= 0 (d0 <= 3) -> empty -> None.
        let set = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![
                Constraint {
                    expr: sub(dim(0), cst(5)),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: sub(cst(3), dim(0)),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        };
        assert!(SymBoxSet::try_from_affine_set(&set).is_none());
    }

    #[test]
    fn lower_equality_pins_both_sides() {
        // d0 == 2 (stored as expr `d0 - 2 == 0`) => lo=[2], hi=[3].
        let set = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: sub(dim(0), cst(2)),
                kind: ConstraintKind::Equal,
            }],
        };
        let b = SymBoxSet::try_from_affine_set(&set).expect("equality lowers");
        assert_eq!(b.lo, vec![Bound::Concrete(2)]);
        assert_eq!(b.hi, vec![Bound::Concrete(3)]);
    }

    #[test]
    fn lower_symbolic_bounds() {
        // { (d0)[s0] : d0 >= 0, s0 - d0 >= 0 }  => d0 in [0, s0] => lo=0, hi=s0+1.
        let set = AffineSet {
            num_dims: 1,
            num_syms: 1,
            constraints: vec![
                Constraint {
                    expr: dim(0),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: sub(sym(0), dim(0)),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        };
        let b = SymBoxSet::try_from_affine_set(&set).expect("symbolic lowers");
        assert!(!b.is_concrete());
        assert_eq!(b.lo, vec![Bound::Concrete(0)]);
        // hi = s0 + 1; resolves to 4 when s0 = 3.
        let spec = b.specialize(&[3]);
        assert_eq!(spec.lo, vec![Bound::Concrete(0)]);
        assert_eq!(spec.hi, vec![Bound::Concrete(4)]);
        assert_eq!(spec.enumerate(None, &[]).len(), 4); // 0,1,2,3
    }
}
