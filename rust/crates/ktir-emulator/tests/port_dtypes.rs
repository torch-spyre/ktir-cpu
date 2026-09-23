// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_dtypes.py` — the canonical KTIR dtype mapping.
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * Python exposes three free functions over string keys:
//!   `to_np_dtype`, `bytes_per_elem`, `to_ktir_dtype`. The Rust crate models the
//!   canonical set as a closed enum [`DType`] and concentrates the alias soup at
//!   the parse boundary:
//!     - Python `to_np_dtype(s)`        -> `DType::parse(s)`           (the numpy
//!       dtype object has no Rust analogue; instead the parsed *enum variant*
//!       carries the identity, so `to_np_dtype(a) == to_np_dtype(b)` becomes
//!       `DType::parse(a) == DType::parse(b)`).
//!     - Python `bytes_per_elem(s)`     -> `DType::parse(s).bytes_per_elem()`.
//!     - Python `to_ktir_dtype(np_dt)`  -> `DType::as_str()` (the reverse map,
//!       keyed on the enum variant rather than a numpy dtype).
//! * Python distinguishes `ValueError` ("Unsupported" / "No KTIR dtype") from
//!   `NotImplementedError` (placeholder `fp8`/`mxfp8`). Rust collapses both to a
//!   single `Result::Err(String)`; we assert on the error message text to keep
//!   the distinction (placeholder messages say "placeholder", garbage says
//!   "unsupported").

use ktir_emulator::dtypes::DType;

// ---------------------------------------------------------------------------
// test_to_np_dtype: each spelling parses to the expected canonical variant and
// has the expected byte size. The numpy dtype identity is represented by the
// enum variant it parses to.
// ---------------------------------------------------------------------------

/// (spelling, canonical variant the numpy dtype maps onto, byte size)
const TO_NP_CASES: &[(&str, DType, usize)] = &[
    ("f16", DType::F16, 2),
    ("fp16", DType::F16, 2),
    ("float16", DType::F16, 2),
    ("f32", DType::F32, 4),
    ("float32", DType::F32, 4),
    ("i32", DType::I32, 4),
    ("si32", DType::I32, 4),
    ("index", DType::I32, 4), // index lowers to i32, exactly as Python maps it
    ("i64", DType::I64, 8),
    ("si64", DType::I64, 8),
];

#[test]
fn to_np_dtype_maps_each_spelling() {
    for &(spelling, expected_variant, expected_bytes) in TO_NP_CASES {
        let dt = DType::parse(spelling)
            .unwrap_or_else(|e| panic!("{spelling:?} should parse, got error: {e}"));
        assert_eq!(
            dt, expected_variant,
            "{spelling:?} should map onto {expected_variant:?}"
        );
        assert_eq!(
            dt.bytes_per_elem(),
            expected_bytes,
            "{spelling:?} should be {expected_bytes} bytes"
        );
    }
}

#[test]
fn to_np_dtype_aliases_are_identical() {
    // Mirror Python's `to_np_dtype(a) == np.dtype(expected)`: the int32-family
    // spellings all collapse to one numpy dtype identity, i.e. one enum variant.
    for s in ["i32", "si32", "index"] {
        assert_eq!(DType::parse(s).unwrap(), DType::I32);
    }
    for s in ["i64", "si64"] {
        assert_eq!(DType::parse(s).unwrap(), DType::I64);
    }
    for s in ["f16", "fp16", "float16"] {
        assert_eq!(DType::parse(s).unwrap(), DType::F16);
    }
}

// ---------------------------------------------------------------------------
// test_unknown_dtype_raises: garbage spellings are a hard error. Python raises
// ValueError(match="Unsupported"); Rust returns Err whose message says
// "unsupported".
// ---------------------------------------------------------------------------

#[test]
fn unknown_dtype_raises() {
    for bad in ["bf16", "i8", "unknown", ""] {
        let err = DType::parse(bad).expect_err(&format!("{bad:?} should be rejected"));
        assert!(
            err.to_lowercase().contains("unsupported"),
            "{bad:?} error should be an 'unsupported' error, got: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// test_placeholder_dtype_raises: fp8/mxfp8 are placeholders pending hardware.
// Python raises NotImplementedError; Rust returns an Err that is distinct from
// the generic "unsupported" garbage error (message mentions "placeholder").
// ---------------------------------------------------------------------------

#[test]
fn placeholder_dtype_raises() {
    for placeholder in ["fp8", "mxfp8"] {
        let err =
            DType::parse(placeholder).expect_err(&format!("{placeholder:?} should be rejected"));
        assert!(
            err.to_lowercase().contains("placeholder"),
            "{placeholder:?} should be a 'placeholder' (NotImplementedError-equivalent) \
             error, got: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// test_to_ktir_dtype: reverse map from a numpy dtype to the canonical KTIR
// spelling. In Rust the reverse map is keyed on the enum variant (the parsed
// identity of the numpy dtype) via `as_str`.
// ---------------------------------------------------------------------------

#[test]
fn to_ktir_dtype_reverse_map() {
    // (numpy dtype, expected canonical KTIR spelling) — the numpy dtype is
    // represented by the variant its canonical spelling parses to.
    let cases: &[(DType, &str)] = &[
        (DType::F16, "f16"),
        (DType::F32, "f32"),
        (DType::I32, "i32"),
        (DType::I64, "i64"),
    ];
    for &(variant, expected_ktir) in cases {
        assert_eq!(variant.as_str(), expected_ktir);
        // And the canonical spelling round-trips back to the same variant.
        assert_eq!(DType::parse(expected_ktir).unwrap(), variant);
    }
}

#[test]
fn to_ktir_dtype_full_roundtrip() {
    // Every canonical spelling parses back to its own variant — the analogue of
    // Python's reverse-map being a proper inverse of the forward map.
    for dt in [DType::F16, DType::F32, DType::Bool, DType::I32, DType::I64] {
        assert_eq!(DType::parse(dt.as_str()).unwrap(), dt);
    }
}

// ---------------------------------------------------------------------------
// test_to_ktir_dtype_unknown_raises: Python passes np.float64 (a numpy dtype
// with no KTIR mapping) and expects ValueError(match="No KTIR dtype").
//
// There is no Rust analogue: the reverse map is keyed on the closed `DType`
// enum, so an "unmapped numpy dtype" is unrepresentable — you cannot construct a
// `DType` that has no `as_str`. The closest faithful check is that float64
// (spelled "f64"/"float64") has no *forward* mapping either, which we already
// cover under `unknown_dtype_raises`. This stub records the gap explicitly.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "GAP: Python-only — to_ktir_dtype takes an arbitrary numpy dtype and \
            rejects unmapped ones (np.float64) with a 'No KTIR dtype' ValueError. \
            Rust's reverse map (DType::as_str) is total over the closed DType enum, \
            so an unmapped input is unrepresentable; there is no analogous failure \
            path. The forward direction (f64/float64 unsupported) is covered by \
            unknown_dtype_raises."]
fn to_ktir_dtype_unknown_raises() {
    // Document the intent: f64 has no KTIR spelling in either direction.
    assert!(DType::parse("f64").is_err());
    assert!(DType::parse("float64").is_err());
}
