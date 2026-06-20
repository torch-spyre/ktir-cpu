// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Canonical KTIR dtype mappings — Rust port of `ktir_cpu/dtypes.py`.
//!
//! The Python source is a string-keyed dict with several spelling aliases per
//! canonical type (`f16`/`fp16`/`float16`). Here the canonical form is a closed
//! enum and the alias soup lives only at the parse boundary (`DType::parse`).

use std::fmt;

/// A KTIR element type. Closed set mirroring `SUPPORTED_DTYPES`.
///
/// Note `index` lowers to `I32` and `i1` to `Bool`, exactly as the Python
/// `SUPPORTED_DTYPES` table maps them onto NumPy dtypes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    F16,
    F32,
    Bool, // i1
    I32,  // also: si32, index
    I64,  // also: si64
}

impl DType {
    /// Parse a KTIR dtype string, accepting all the aliases the Python table does.
    ///
    /// Mirrors `to_np_dtype`: placeholder dtypes (`fp8`, `mxfp8`) are a hard
    /// error so any example that uses them fails loudly until implemented.
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "f16" | "fp16" | "float16" => DType::F16,
            "f32" | "float32" => DType::F32,
            "i1" => DType::Bool,
            "i32" | "si32" | "index" => DType::I32,
            "i64" | "si64" => DType::I64,
            "fp8" | "mxfp8" => {
                return Err(format!(
                    "dtype {s:?} is a placeholder pending hardware confirmation; \
                     extend DType before adding examples that use it"
                ));
            }
            _ => return Err(format!("unsupported KTIR dtype: {s:?}")),
        })
    }

    /// Element size in bytes — mirrors `bytes_per_elem`.
    pub fn bytes_per_elem(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::Bool => 1,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }

    /// Canonical spelling — mirrors `to_ktir_dtype`'s reverse map.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::Bool => "i1",
            DType::I32 => "i32",
            DType::I64 => "i64",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_collapse_to_canonical() {
        for s in ["f16", "fp16", "float16"] {
            assert_eq!(DType::parse(s).unwrap(), DType::F16);
        }
        assert_eq!(DType::parse("index").unwrap(), DType::I32);
        assert_eq!(DType::parse("i1").unwrap(), DType::Bool);
    }

    #[test]
    fn roundtrip_through_canonical_string() {
        for dt in [DType::F16, DType::F32, DType::Bool, DType::I32, DType::I64] {
            assert_eq!(DType::parse(dt.as_str()).unwrap(), dt);
        }
    }

    #[test]
    fn placeholders_and_garbage_error() {
        assert!(DType::parse("fp8").is_err());
        assert!(DType::parse("mxfp8").is_err());
        assert!(DType::parse("bfloat16").is_err());
    }

    #[test]
    fn sizes_match_spec() {
        assert_eq!(DType::F16.bytes_per_elem(), 2);
        assert_eq!(DType::I64.bytes_per_elem(), 8);
    }
}
