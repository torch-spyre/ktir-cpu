// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Tile data value — minimal port of `Tile` from `ktir_cpu/ir_types.py`.
//!
//! STORAGE: a tile keeps its data in its NATIVE dtype representation
//! ([`TileStorage`]) — f16 as `Rc<[u16]>` bit patterns, f32 as `Rc<[f32]>`,
//! integers/bool as their own width — rather than always widening to f32. This
//! halves the memory of an f16 tile (the dominant real-model dtype) and means a
//! weight loaded once is never re-widened on each forward.
//!
//! TWO SEAMS, and only two:
//!   * ONE widening point — [`Tile::as_f32`] — decodes the native arm to f32 via
//!     the exact `codec` paths (f16 decode is lossless; integer/bool widen
//!     exactly). Every compute handler reads operands through it.
//!   * ONE narrowing point — [`Tile::compute`] — rounds to the dtype grid
//!     (`codec::round_to_dtype`, NumPy per-op semantics) and then ENCODES into
//!     the matching native arm.
//!
//! Fidelity is preserved by construction: rounding still happens exactly once in
//! `compute`, and the round-tripped value is bit-identical to the f32-storage
//! version — only the stored representation changes. Integer/bool index tensors
//! are kept losslessly by their own enum arm (never f16-quantized).

use crate::dtypes::DType;
use std::borrow::Cow;
use std::rc::Rc;

/// A tile's element data in its native dtype representation.
///
/// Each arm is reference-counted (`Rc<[_]>`): tiles are immutable once built
/// (every op produces a fresh result via [`Tile::compute`]), so cloning a Tile —
/// which the interpreter does constantly (binding op results, threading scf
/// iter_args, passing operands) — is a refcount bump, not a deep copy. The `F16`
/// arm stores IEEE half-precision *bit patterns* (`u16`), not raw bytes, so that
/// the slice length is always the element count (numel).
#[derive(Clone, Debug, PartialEq)]
pub enum TileStorage {
    /// IEEE-754 half-precision bit patterns (one `u16` per element).
    F16(Rc<[u16]>),
    F32(Rc<[f32]>),
    I32(Rc<[i32]>),
    I64(Rc<[i64]>),
    /// `i1`: one byte per element, 0 or 1.
    Bool(Rc<[u8]>),
}

impl TileStorage {
    /// Element count (numel) — the length of the underlying slice, independent of
    /// the storage width in bytes.
    fn len(&self) -> usize {
        match self {
            TileStorage::F16(s) => s.len(),
            TileStorage::F32(s) => s.len(),
            TileStorage::I32(s) => s.len(),
            TileStorage::I64(s) => s.len(),
            TileStorage::Bool(s) => s.len(),
        }
    }

    /// Widen the native representation to f32 — the single decode path.
    ///
    /// `F32` borrows (zero copy); every other arm owns a freshly widened buffer.
    /// f16 decode is exact (lossless); integer/bool widen exactly to the f32 grid
    /// they were rounded onto by [`Tile::compute`].
    fn as_f32(&self) -> Cow<'_, [f32]> {
        match self {
            TileStorage::F32(s) => Cow::Borrowed(s),
            TileStorage::F16(s) => Cow::Owned(crate::codec::f16_units_to_f32(s)),
            TileStorage::I32(s) => Cow::Owned(s.iter().map(|&x| x as f32).collect()),
            TileStorage::I64(s) => Cow::Owned(s.iter().map(|&x| x as f32).collect()),
            TileStorage::Bool(s) => Cow::Owned(s.iter().map(|&x| x as f32).collect()),
        }
    }

    /// Store f32 data that is ALREADY on `dtype`'s representable grid, keeping the
    /// f32 in place for float dtypes instead of narrowing to bytes.
    ///
    /// f16↔f32 is exact, so a value rounded to the f16 grid is represented
    /// bit-identically by either the f16 encoding or the rounded f32 — but keeping
    /// f32 lets every downstream [`Tile::as_f32`] borrow (zero copy) instead of
    /// re-decoding the f16 on each consumer. That per-op f16 round-trip
    /// (encode-then-decode) was pure overhead on the interpreter hot path. Integer
    /// and bool dtypes still narrow to native storage: f32 cannot hold i32/i64
    /// exactly, and `as_f32` for them is already a widening copy. `Tile::size_bytes`
    /// reports the *dtype* width regardless of storage, so LX accounting is
    /// unchanged whether an f16 tile is stored as f16 bytes or rounded f32.
    fn store_rounded(data: Vec<f32>, dtype: DType) -> TileStorage {
        match dtype {
            DType::F16 | DType::F32 => TileStorage::F32(data.into()),
            _ => TileStorage::encode(&data, dtype),
        }
    }

    /// Encode an f32 buffer (already rounded to `dtype`'s grid) into the native
    /// arm for `dtype` — the single narrowing path's storage step.
    fn encode(data: &[f32], dtype: DType) -> TileStorage {
        match dtype {
            DType::F32 => TileStorage::F32(data.into()),
            DType::F16 => TileStorage::F16(crate::codec::f32_to_f16_units(data).into()),
            DType::I32 => TileStorage::I32(data.iter().map(|&x| x as i32).collect()),
            DType::I64 => TileStorage::I64(data.iter().map(|&x| x as i64).collect()),
            DType::Bool => TileStorage::Bool(
                data.iter()
                    .map(|&x| if x != 0.0 { 1u8 } else { 0u8 })
                    .collect(),
            ),
        }
    }
}

/// A tensor of element data — `load` result / compute-op operand.
///
/// `data` is a [`TileStorage`] (the native-dtype representation); read it as f32
/// through [`Tile::as_f32`] and build new tiles through [`Tile::compute`].
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    data: TileStorage,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Distinct HBM sticks touched by the load that produced this tile.
    /// `None` for compute-produced tiles (mirrors the Python field).
    pub unique_sticks: Option<usize>,
    /// Distinct sticks touched by index-tensor reads during an indirect
    /// load/store. `None` for direct loads and compute-produced tiles.
    pub index_unique_sticks: Option<usize>,
}

impl Tile {
    /// Construct a compute-produced tile (no stick bookkeeping).
    ///
    /// The data is rounded to `dtype`'s representable set — exactly what a NumPy
    /// `np.float16`/`np.int32`/... array does on assignment — and then stored in
    /// the matching native arm. Because every op handler builds its result
    /// through this constructor, this gives per-op rounding for free: an f16
    /// compute *chain* rounds after each step the way NumPy does, rather than
    /// accumulating in f32 and rounding only at store.
    pub fn compute(mut data: Vec<f32>, dtype: DType, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "tile data length must equal product of shape"
        );
        crate::codec::round_to_dtype(&mut data, dtype);
        Tile {
            data: TileStorage::store_rounded(data, dtype),
            dtype,
            shape,
            unique_sticks: None,
            index_unique_sticks: None,
        }
    }

    /// Construct a tile from already-decoded `f32` data plus stick bookkeeping —
    /// the `ktdp.load` result path. The data was just `codec::decode`d from native
    /// HBM bytes, so it already sits on `dtype`'s representable set; encoding it
    /// back into the native arm is therefore lossless (the round-trip is exact for
    /// f16 and integers). This is the load-boundary analogue of [`Tile::compute`]
    /// that threads the `unique_sticks` sidebands through.
    pub fn from_decoded(
        data: Vec<f32>,
        dtype: DType,
        shape: Vec<usize>,
        unique_sticks: Option<usize>,
        index_unique_sticks: Option<usize>,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "tile data length must equal product of shape"
        );
        Tile {
            data: TileStorage::store_rounded(data, dtype),
            dtype,
            shape,
            unique_sticks,
            index_unique_sticks,
        }
    }

    /// The element data widened to `f32` — the single widening seam.
    ///
    /// This is the ONE place a tile's native storage is decoded to `f32`. Every
    /// compute handler reads operands through here; the returned [`Cow`] borrows
    /// when storage is already `f32` (zero copy) and owns when it had to widen a
    /// narrower native representation (e.g. f16). The widening is lossless: f16
    /// decode is exact and integer/bool widen exactly.
    pub fn as_f32(&self) -> Cow<'_, [f32]> {
        self.data.as_f32()
    }

    /// Element count (numel) — the product of `shape`. Independent of dtype/
    /// storage width; this is what `data.len()` meant before storage went native.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the tile holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len() * self.dtype.bytes_per_elem()
    }

    /// A stable identity for the tile's backing allocation — the address of the
    /// reference-counted element buffer. Two `Tile` values that alias the same
    /// underlying `Rc<[_]>` (the result of `clone`-ing one tile, e.g. an
    /// `scf.for` iter_arg rebind or a `linalg.reduce` result bound to both its
    /// SSA name and its `outs` buffer) return the SAME pointer here. This is the
    /// Rust analogue of Python's `id(Tile)` and lets LX accounting charge each
    /// physical allocation exactly once (alias dedup). Empty tiles can share a
    /// dangling/zero pointer; they cost no LX so that's harmless.
    pub fn data_ptr(&self) -> usize {
        match &self.data {
            TileStorage::F16(s) => Rc::as_ptr(s) as *const u8 as usize,
            TileStorage::F32(s) => Rc::as_ptr(s) as *const u8 as usize,
            TileStorage::I32(s) => Rc::as_ptr(s) as *const u8 as usize,
            TileStorage::I64(s) => Rc::as_ptr(s) as *const u8 as usize,
            TileStorage::Bool(s) => Rc::as_ptr(s) as *const u8 as usize,
        }
    }
}
