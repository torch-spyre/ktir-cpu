// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_affine.py` — AffineMap, AffineSet, and BoxSet value
//! objects, exercised through the crate's affine API.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python `parse_affine_map` / `parse_affine_set` -> the free functions in
//!   `ktir_emulator::parser_ast`. The Rust `parse_affine_set` is the *raw* form: it
//!   always returns an [`AffineSet`] and does **not** lower axis-aligned sets to
//!   a box at parse time (Python's `parse_affine_set` did, returning `BoxSet`).
//!   Box lowering is therefore exercised explicitly via
//!   [`SymBoxSet::try_from_affine_set`], which is the Rust port of Python's
//!   `BoxSet.try_from_affine_set`.
//! * Python's `BoxSet` is a half-open `[lo, hi)` box whose bounds may be
//!   symbolic. The faithful Rust port of that type is
//!   [`ktir_emulator::affine::SymBoxSet`] (the crate also has an unrelated
//!   *inclusive* `[lo, hi]` `BoxSet` used by the ktdp slice — that is **not**
//!   the Python `BoxSet`, so this file never uses it).
//! * `AffineMap.eval` returns a tuple in Python; here it returns a `Vec<i64>`.
//! * Python's frozen-dataclass / `source`-field / wrong-arity-`ValueError`
//!   tests have no Rust analogue (immutability is a type-system property, there
//!   is no `source` field, and arity mismatch is a `debug_assert`). They are
//!   listed in the integrator notes as intentionally skipped.

use ktir_emulator::affine::{Bound, SymBoxSet};
use ktir_emulator::parser_ast::{parse_affine_map, parse_affine_set};

// ===========================================================================
// AffineMap — eval
// ===========================================================================

#[test]
fn map_eval_delegates() {
    // test_eval_delegates: (d0) -> (d0) at 7 is 7.
    let m = parse_affine_map("affine_map<(d0) -> (d0)>").unwrap();
    assert_eq!(m.eval(&[7], &[]), vec![7]);
}

#[test]
fn map_eval_non_identity() {
    // test_eval_non_identity: (i) -> (i, 0) at 3 is (3, 0).
    let m = parse_affine_map("affine_map<(i) -> (i, 0)>").unwrap();
    assert_eq!(m.eval(&[3], &[]), vec![3, 0]);
}

// ===========================================================================
// AffineMap — is_permutation
// ===========================================================================

#[test]
fn perm_1d_identity() {
    let m = parse_affine_map("affine_map<(d0) -> (d0)>").unwrap();
    assert!(m.is_permutation());
}

#[test]
fn perm_identity() {
    let m = parse_affine_map("affine_map<(d0, d1, d2) -> (d0, d1, d2)>").unwrap();
    assert!(m.is_permutation());
}

#[test]
fn perm_2d_swap() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>").unwrap();
    assert!(m.is_permutation());
}

#[test]
fn perm_3d_cycle() {
    let m = parse_affine_map("affine_map<(d0, d1, d2) -> (d2, d0, d1)>").unwrap();
    assert!(m.is_permutation());
}

#[test]
fn perm_shear_rejected() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0 + d1, d1)>").unwrap();
    assert!(!m.is_permutation());
}

#[test]
fn perm_many_to_one_rejected() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0, d0)>").unwrap();
    assert!(!m.is_permutation());
}

#[test]
fn perm_non_square_rejected() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0)>").unwrap();
    assert!(!m.is_permutation());
}

#[test]
fn perm_linear_combination_rejected() {
    // Regression: probe [1,2] -> (1,2) would fool a probe-based check; the
    // structural check rejects it. pt=(3,1) -> (2,3) confirms non-permutation.
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0 + d1 - 2, d0 + d1 - 1)>").unwrap();
    assert!(!m.is_permutation());
    assert_eq!(m.eval(&[3, 1], &[]), vec![2, 3]);
}

#[test]
fn perm_constant_offset_rejected() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d1 - 1, d0 + 1)>").unwrap();
    assert!(!m.is_permutation());
}

#[test]
fn perm_trivial_wrappers_accepted() {
    // `1 * d1 + 0` flattens to `d1`; still a permutation.
    let m = parse_affine_map("affine_map<(d0, d1) -> (1 * d1 + 0, d0)>").unwrap();
    assert!(m.is_permutation());
}

// ===========================================================================
// AffineMap — is_identity
// ===========================================================================

#[test]
fn id_identity_accepted() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0, d1)>").unwrap();
    assert!(m.is_identity());
}

#[test]
fn id_swap_rejected() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d1, d0)>").unwrap();
    assert!(!m.is_identity());
}

#[test]
fn id_constant_offset_rejected() {
    let m = parse_affine_map("affine_map<(d0, d1) -> (d1 - 1, d0 + 1)>").unwrap();
    assert!(!m.is_identity());
}

#[test]
fn id_trivial_wrappers_accepted() {
    // `d0 + 0` and `1 * d1` flatten to `d0` / `d1` and remain identity.
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0 + 0, 1 * d1)>").unwrap();
    assert!(m.is_identity());
}

#[test]
fn id_through_cancellation() {
    // `d0 + d1 - d1` flattens to `d0`; `d1 + d0 - d0` flattens to `d1`.
    let m = parse_affine_map("affine_map<(d0, d1) -> (d0 + d1 - d1, d1 + d0 - d0)>").unwrap();
    assert!(m.is_identity());
}

// ===========================================================================
// AffineSet — contains / enumerate / is_full / intersect
//
// The Rust `parse_affine_set` never lowers to a box, so these run on the
// AffineSet branch directly (Python used non-axis-aligned sets to force that
// branch; we keep the same inputs for fidelity).
// ===========================================================================

#[test]
fn set_contains_delegates() {
    // d1 >= d0 with the box bounds: (1,2) in, (2,1) out.
    let s = parse_affine_set(
        "affine_set<(d0, d1) : (d1 - d0 >= 0, d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>",
    )
    .unwrap();
    assert!(s.contains(&[1, 2], &[]));
    assert!(!s.contains(&[2, 1], &[]));
}

#[test]
fn set_enumerate_delegates() {
    // Upper-triangular 2x2: points satisfying d1 >= d0 in [0,2)^2.
    let s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>").unwrap();
    assert_eq!(
        s.enumerate(&[2, 2], &[]),
        vec![vec![0, 0], vec![0, 1], vec![1, 1]]
    );
}

#[test]
#[should_panic]
fn set_enumerate_wrong_shape_panics() {
    // Python raised ValueError on a shape/n_dims mismatch; Rust asserts.
    let s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>").unwrap();
    s.enumerate(&[4], &[]);
}

#[test]
fn set_is_full_false() {
    // Upper-triangular set (d1 >= d0) is not full — corner (3,0) is excluded.
    let s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>").unwrap();
    assert!(!s.is_full(&[4, 4]));
}

#[test]
fn set_is_full_wrong_ndim() {
    // Shape ndim != set n_dims always returns false.
    let s = parse_affine_set("affine_set<(d0, d1) : (d1 - d0 >= 0)>").unwrap();
    assert!(!s.is_full(&[2]));
}

#[test]
fn set_intersect_conjoins_constraints() {
    // A: d0 >= 0 ; B: 3 - d0 >= 0 ; A ∩ B == 0 <= d0 <= 3 over a width-10 box.
    let a = parse_affine_set("affine_set<(d0) : (d0 >= 0)>").unwrap();
    let b = parse_affine_set("affine_set<(d0) : (-d0 + 3 >= 0)>").unwrap();
    let c = a.intersect(&b);
    assert_eq!(c.constraints.len(), 2);
    assert_eq!(
        c.enumerate(&[10], &[]),
        vec![vec![0], vec![1], vec![2], vec![3]]
    );
}

// ===========================================================================
// SymBoxSet — the faithful port of Python's half-open `BoxSet`.
// ===========================================================================

fn box_of(lo: &[i64], hi: &[i64]) -> SymBoxSet {
    SymBoxSet::from_concrete(lo.to_vec(), hi.to_vec())
}

#[test]
fn box_contains() {
    // test_contains: hi is exclusive.
    let b = box_of(&[0, 0], &[2, 3]);
    assert!(b.contains(&[0, 0], &[]));
    assert!(b.contains(&[1, 2], &[]));
    assert!(!b.contains(&[2, 0], &[])); // hi exclusive
    assert!(!b.contains(&[0, 3], &[]));
    assert!(!b.contains(&[-1, 0], &[]));
}

#[test]
fn box_contains_wrong_ndim() {
    let b = box_of(&[0], &[3]);
    assert!(!b.contains(&[0, 0], &[]));
}

#[test]
fn box_enumerate_no_shape() {
    let b = box_of(&[1, 2], &[3, 4]);
    assert_eq!(
        b.enumerate(None, &[]),
        vec![vec![1, 2], vec![1, 3], vec![2, 2], vec![2, 3]]
    );
}

#[test]
fn box_enumerate_shape_matches_hi() {
    let b = box_of(&[0, 0], &[2, 2]);
    assert_eq!(
        b.enumerate(Some(&[2, 2]), &[]),
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );
}

#[test]
fn box_enumerate_shape_upper_bounds_hi() {
    // shape may be a strict upper bound — box stays self-bounded.
    let b = box_of(&[0, 0], &[2, 2]);
    assert_eq!(
        b.enumerate(Some(&[4, 4]), &[]),
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );
}

#[test]
#[should_panic]
fn box_enumerate_shape_below_hi_panics() {
    // If the box extends past shape, the call site has an invariant bug.
    let b = box_of(&[0, 0], &[3, 3]);
    b.enumerate(Some(&[2, 4]), &[]);
}

#[test]
#[should_panic]
fn box_enumerate_shape_ndim_mismatch_panics() {
    let b = box_of(&[0, 0], &[2, 2]);
    b.enumerate(Some(&[2]), &[]);
}

#[test]
fn box_is_empty() {
    assert!(!box_of(&[0, 0], &[2, 2]).is_empty(&[]));
    assert!(box_of(&[2, 0], &[2, 2]).is_empty(&[])); // zero-width axis
    assert!(box_of(&[3, 0], &[2, 2]).is_empty(&[])); // hi < lo
}

#[test]
fn box_is_full() {
    assert!(box_of(&[0, 0], &[2, 3]).is_full(&[2, 3], &[]));
    assert!(!box_of(&[0, 0], &[2, 3]).is_full(&[2, 4], &[]));
    assert!(!box_of(&[1, 0], &[2, 3]).is_full(&[2, 3], &[]));
}

#[test]
fn box_is_full_wrong_ndim() {
    assert!(!box_of(&[0], &[3]).is_full(&[3, 3], &[]));
}

#[test]
fn box_lower_bounds() {
    assert_eq!(box_of(&[2, 5], &[4, 7]).lower_bounds(&[]), vec![2, 5]);
}

#[test]
fn box_translate() {
    let b = box_of(&[0, 0], &[2, 2]);
    let t = b.translate(&[Bound::Concrete(10), Bound::Concrete(20)]);
    assert_eq!(t, box_of(&[10, 20], &[12, 22]));
}

#[test]
#[should_panic]
fn box_translate_wrong_ndim_panics() {
    box_of(&[0, 0], &[2, 2]).translate(&[Bound::Concrete(1)]);
}

#[test]
fn box_intersect_disjoint_is_empty() {
    let a = box_of(&[0, 0], &[2, 2]);
    let b = box_of(&[2, 0], &[4, 2]);
    assert!(a.intersect(&b).is_empty(&[]));
}

#[test]
fn box_intersect_overlap() {
    let a = box_of(&[0, 0], &[3, 3]);
    let b = box_of(&[1, 1], &[5, 5]);
    assert_eq!(a.intersect(&b), box_of(&[1, 1], &[3, 3]));
}

#[test]
#[should_panic]
fn box_intersect_ndim_mismatch_panics() {
    box_of(&[0], &[2]).intersect(&box_of(&[0, 0], &[2, 2]));
}

// ===========================================================================
// SymBoxSet::try_from_affine_set — parse-time lowering from AffineSet to box.
//
// Port of Python's TestTryFromAffineSet: build an AffineSet via the parser
// (`parse_affine_set` is the raw form, so it stays an AffineSet) and feed it
// to the lowering routine.
// ===========================================================================

fn try_lower(src: &str) -> Option<SymBoxSet> {
    let aset = parse_affine_set(src).unwrap();
    SymBoxSet::try_from_affine_set(&aset)
}

#[test]
fn lower_accept_1d_range() {
    let b = try_lower("affine_set<(d0) : (d0 >= 0, -d0 + 3 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[0], &[4]));
}

#[test]
fn lower_accept_2d_box() {
    let b =
        try_lower("affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[0, 0], &[2, 4]));
}

#[test]
fn lower_accept_nonzero_origin() {
    // d0 >= 2, d0 <= 5 -> lo=2, hi=6.
    let b = try_lower("affine_set<(d0) : (d0 - 2 >= 0, -d0 + 5 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[2], &[6]));
}

#[test]
fn lower_accept_tightest_bounds() {
    // lo = max(0, 2) = 2, hi = min(6, 4) = 4.
    let b =
        try_lower("affine_set<(d0) : (d0 >= 0, d0 - 2 >= 0, -d0 + 5 >= 0, -d0 + 3 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[2], &[4]));
}

#[test]
fn lower_reject_not_axis_aligned() {
    // Upper-triangular d1 >= d0: two dims in one constraint.
    assert!(try_lower("affine_set<(d0, d1) : (d1 - d0 >= 0)>").is_none());
}

#[test]
fn lower_reject_missing_upper_bound() {
    assert!(try_lower("affine_set<(d0) : (d0 >= 0)>").is_none());
}

#[test]
fn lower_reject_missing_lower_bound() {
    assert!(try_lower("affine_set<(d0) : (-d0 + 3 >= 0)>").is_none());
}

#[test]
fn lower_reject_nonunit_coefficient() {
    assert!(try_lower("affine_set<(d0) : (2 * d0 >= 0, -d0 + 3 >= 0)>").is_none());
}

#[test]
fn lower_accept_eq_and_range() {
    // d0 == 2, 1 <= d1 <= 3 -> BoxSet(lo=(2,1), hi=(3,4)).
    let b = try_lower("affine_set<(d0, d1) : (d0 - 2 == 0, d1 - 1 >= 0, -d1 + 3 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[2, 1], &[3, 4]));
}

#[test]
fn lower_reject_one_axis_unpinned() {
    assert!(try_lower("affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0)>").is_none());
}

#[test]
fn lower_reject_axis_with_no_constraints() {
    assert!(try_lower("affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0)>").is_none());
}

#[test]
fn lower_reject_eq_pins_one_axis_other_unconstrained() {
    assert!(try_lower("affine_set<(d0, d1) : (d0 == 0)>").is_none());
}

// ---- Equality-constraint lowering (TestEqualityBoxSetLowering) ----

#[test]
fn lower_eq_pins_single_dim() {
    let b = try_lower("affine_set<(g) : (g == 0)>").unwrap();
    assert_eq!(b, box_of(&[0], &[1]));
}

#[test]
fn lower_eq_i_equals_zero() {
    let b = try_lower("affine_set<(i) : (i == 0)>").unwrap();
    assert_eq!(b, box_of(&[0], &[1]));
}

#[test]
fn lower_eq_nonzero_pin() {
    let b = try_lower("affine_set<(g) : (g - 3 == 0)>").unwrap();
    assert_eq!(b, box_of(&[3], &[4]));
}

#[test]
fn lower_eq_pin_and_ineq_intersection() {
    // g == 0 combined with g <= 5 — pin wins: lo=0, hi=1.
    let b = try_lower("affine_set<(g) : (g == 0, -g + 5 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[0], &[1]));
}

#[test]
fn lower_eq_negative_coeff_pin() {
    // -g + 3 == 0 means g == 3 -> BoxSet(lo=(3,), hi=(4,)).
    let b = try_lower("affine_set<(g) : (-g + 3 == 0)>").unwrap();
    assert_eq!(b, box_of(&[3], &[4]));
}

#[test]
fn lower_eq_multi_dim_rejected() {
    // p - c == 0 involves two dims — cannot lower to a box.
    assert!(try_lower("affine_set<(p, c) : (p - c == 0)>").is_none());
}

#[test]
fn lower_reject_conflicting_eq_constraints() {
    assert!(try_lower("affine_set<(d0) : (d0 == 2, d0 == 3)>").is_none());
}

#[test]
fn lower_reject_eq_ineq_conflict() {
    assert!(try_lower("affine_set<(d0) : (d0 == 2, d0 >= 5)>").is_none());
}

#[test]
fn lower_reject_conflicting_inequalities() {
    assert!(try_lower("affine_set<(d0) : (d0 >= 5, -d0 + 3 >= 0)>").is_none());
}

// ===========================================================================
// Symbolic BoxSet — lowering, specialize, and the query surface.
//
// Port of TestSymbolicBoxSet and the symbolic-eq cases. Cross-checks the box
// answers against the AffineSet slow path on the same input, so the
// expectation comes from the IR rather than a hand-coded constant.
// ===========================================================================

#[test]
fn sym_lowering_preserves_bounds() {
    // -d0 + s0 - 1 >= 0 -> d0 < s0 -> hi[0] is symbolic in s0.
    let b = try_lower("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    assert!(!b.is_concrete());
    assert_eq!(b.lo, vec![Bound::Concrete(0)]);
}

#[test]
fn sym_specialize_resolves_to_concrete_box() {
    let b = try_lower("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    for n in [1i64, 16, 1024] {
        let spec = b.specialize(&[n]);
        assert!(spec.is_concrete());
        assert_eq!(spec, box_of(&[0], &[n]));
    }
}

#[test]
fn sym_query_methods_accept_symbols() {
    // Cross-check the symbolic box answers against the equivalent AffineSet
    // slow path so the expectation comes from the IR.
    let src = "affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>";
    let aset = parse_affine_set(src).unwrap();
    let box_ = SymBoxSet::try_from_affine_set(&aset).unwrap();
    for n in [1i64, 8] {
        for pt in [[0i64], [n - 1], [n], [-1]] {
            assert_eq!(
                box_.contains(&pt, &[n]),
                aset.contains(&pt, &[n]),
                "contains mismatch at pt={pt:?} n={n}"
            );
        }
        assert!(!box_.is_empty(&[n]));
        assert!(box_.is_full(&[n as usize], &[n]));
        let expected: Vec<Vec<i64>> = (0..n).map(|i| vec![i]).collect();
        assert_eq!(box_.enumerate(Some(&[n as usize]), &[n]), expected);
    }
    // Empty extent: s0 = 0 collapses hi to lo.
    assert!(box_.is_empty(&[0]));
}

#[test]
fn sym_intersect_specialized_then_concrete() {
    // Symbolic lo on d0, concrete elsewhere; after specialize the axis-wise
    // intersect falls to plain ints.
    let sym = try_lower("affine_set<(d0)[s0] : (d0 - s0 >= 0, -d0 + 1023 >= 0)>").unwrap();
    let concrete = box_of(&[0], &[8]);
    let spec = sym.specialize(&[3]); // lo=3, hi=1024
    let out = spec.intersect(&concrete);
    assert_eq!(out, box_of(&[3], &[8]));
    assert!(out.is_concrete());
}

#[test]
fn sym_translate_concrete_offset_preserves_symbols() {
    // Translating a symbolic box by a concrete offset retains the symbolic
    // side; concrete sides fold. hi specialises to 5 + n.
    let s = try_lower("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    let shifted = s.translate(&[Bound::Concrete(5)]);
    // lo folds to the concrete int 5.
    assert_eq!(shifted.lo, vec![Bound::Concrete(5)]);
    // hi[0] stays symbolic (not concrete) after translate.
    assert!(!shifted.hi[0].is_concrete());
    for n in [8i64, 64] {
        assert_eq!(shifted.specialize(&[n]), box_of(&[5], &[5 + n]));
    }
}

#[test]
fn sym_reject_multi_dim_with_symbol() {
    // Symbol mixed with two dims in one constraint — not separable.
    assert!(
        try_lower("affine_set<(d0, d1)[s0] : (d0 + d1 - s0 >= 0, -d0 + 3 >= 0, -d1 + 3 >= 0)>")
            .is_none()
    );
}

#[test]
fn sym_reject_nonunit_coefficient_with_symbol() {
    // 2 * d0 + s0 >= 0 — non-±1 dim coefficient on the symbolic path.
    assert!(try_lower("affine_set<(d0)[s0] : (2 * d0 + s0 >= 0, -d0 + 3 >= 0)>").is_none());
}

#[test]
fn sym_negative_symbol_coefficient_in_bound() {
    // d0 - s0 >= 0 -> d0 >= s0 -> lo[0] depends on +s0 via a negated sym term.
    let s = try_lower("affine_set<(d0)[s0] : (d0 - s0 >= 0, -d0 + 1023 >= 0)>").unwrap();
    assert!(!s.is_concrete());
    assert_eq!(s.hi, vec![Bound::Concrete(1024)]); // concrete fold
    assert!(!s.lo[0].is_concrete()); // symbolic AST retained

    let aset = parse_affine_set("affine_set<(d0)[s0] : (d0 - s0 >= 0, -d0 + 1023 >= 0)>").unwrap();
    for n in [0i64, 3, 1023] {
        assert_eq!(s.specialize(&[n]), box_of(&[n], &[1024]));
        for pt in [[n - 1], [n], [n + 1], [1022], [1023]] {
            assert_eq!(s.contains(&pt, &[n]), aset.contains(&pt, &[n]));
        }
    }
}

#[test]
fn lower_symbolic_eq_pin() {
    // d0 == s0 lowers to a symbolic box that specialises to a point.
    let aset = parse_affine_set("affine_set<(d0)[s0] : (d0 - s0 == 0)>").unwrap();
    let b = SymBoxSet::try_from_affine_set(&aset).unwrap();
    assert!(!b.is_concrete());
    assert_eq!(b.specialize(&[3]), box_of(&[3], &[4]));
    assert_eq!(b.specialize(&[7]), box_of(&[7], &[8]));
}

#[test]
fn lower_symbolic_eq_with_offset() {
    // p - c + 2 == 0 -> p == c - 2; specialise([5]) -> BoxSet(lo=(3,), hi=(4,)).
    let aset = parse_affine_set("affine_set<(p)[c] : (p - c + 2 == 0)>").unwrap();
    let b = SymBoxSet::try_from_affine_set(&aset).unwrap();
    assert!(!b.is_concrete());
    assert_eq!(b.specialize(&[5]), box_of(&[3], &[4]));
}

// ===========================================================================
// End-to-end: parse a raw AffineSet, lower it, and check the resulting box.
// (Python's TestParseAffineSetLowering checked that parse_affine_set returned a
// BoxSet directly; the Rust parser keeps it raw, so we lower explicitly and
// assert the same box / non-box outcome.)
// ===========================================================================

#[test]
fn end_to_end_axis_aligned_becomes_box() {
    let b =
        try_lower("affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>").unwrap();
    assert_eq!(b, box_of(&[0, 0], &[4, 4]));
}

#[test]
fn end_to_end_non_box_stays_affine_set() {
    // A non-axis-aligned set does not lower to a box.
    assert!(try_lower("affine_set<(d0, d1) : (d1 - d0 >= 0)>").is_none());
}

#[test]
fn end_to_end_symbolic_lowering_via_parse() {
    // -d0 + s0 - 1 >= 0 -> d0 < s0 -> symbolic hi; specialises per-symbol.
    let s = try_lower("affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>").unwrap();
    assert!(!s.is_concrete());
    assert_eq!(s.lo, vec![Bound::Concrete(0)]);
    for n in [1i64, 16, 1024] {
        assert_eq!(s.specialize(&[n]), box_of(&[0], &[n]));
    }
}
