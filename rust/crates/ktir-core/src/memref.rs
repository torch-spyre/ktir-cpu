// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Memory-view IR types — Rust port of the memref/tileref half of
//! `ktir_cpu/ir_types.py`. These are the heart of KTIR: the separation of
//! memory interpretation (`MemRef`), address computation (`TileRef`), and the
//! coordinate descriptors (`AccessTile`) that load/store consume.

use std::collections::HashMap;

use crate::affine::{AffineExpr, AffineMap, AffineSet, BoxSet};
use crate::dtypes::DType;

/// HBM stick size in bytes (the Spyre layout granularity). Lives in core because
/// `memref` byte-addressing depends on it; the emulator's `memory` module
/// re-exports it so `crate::memory::STICK_BYTES` keeps resolving.
pub const STICK_BYTES: i64 = 128;

/// Memory space of a view. Replaces the Python `memory_space: str` +
/// `lx_core_id: Optional[int]` pair (with its `__post_init__` cross-check that
/// `lx_core_id` is only set for LX). As an enum that invariant is structural —
/// an invalid combination is unrepresentable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySpace {
    Hbm,
    /// `core_id = None` means "the executing core's own LX scratchpad"
    /// (default routing), per `#ktdp.spyre_memory_space<LX, core = N>`.
    Lx {
        core_id: Option<u32>,
    },
}

impl MemorySpace {
    pub fn parse(space: &str, core_id: Option<u32>) -> Result<Self, String> {
        match space {
            "HBM" => {
                if core_id.is_some() {
                    return Err("core id may only be set for LX, got HBM".into());
                }
                Ok(MemorySpace::Hbm)
            }
            "LX" => Ok(MemorySpace::Lx { core_id }),
            other => Err(format!("invalid memory_space {other:?}; must be HBM or LX")),
        }
    }
}

/// The set of global coords a view owns. Mirrors Python's
/// `CoordinateSet = Union[BoxSet, AffineSet, List[Tuple[int, ...]]]`.
#[derive(Clone, Debug, PartialEq)]
pub enum CoordinateSet {
    /// Axis-aligned fast path, O(ndim).
    Box(BoxSet),
    /// General non-box set.
    Affine(AffineSet),
    /// Pre-enumerated points (distributed slow path).
    Points(Vec<Vec<i64>>),
}

/// Hardware-aware memory view — result of `construct_memory_view`.
/// Constructs a logical view only; it does **not** allocate.
#[derive(Clone, Debug)]
pub struct MemRef {
    /// Element index — the number of elements from the start of the address
    /// space, matching what MLIR pointer operands carry (both HBM and LX).
    pub base_ptr: i64,
    pub shape: Vec<usize>,
    /// Element counts, not bytes.
    pub strides: Vec<i64>,
    pub space: MemorySpace,
    pub dtype: DType,
    /// Global coords this MemRef owns; the partition origin is its `min`.
    pub coordinate_set: Option<AffineSet>,
}

impl MemRef {
    /// Absolute byte address of this view's base, regardless of space.
    /// Mirrors the `byte_address` property: `base_ptr` is an element index, so
    /// the byte address is `base_ptr * bytes_per_elem(dtype)` for both spaces.
    pub fn byte_address(&self) -> i64 {
        self.base_ptr * self.dtype.bytes_per_elem() as i64
    }

    /// Split a byte address into `(main, intra)` per memory space.
    /// HBM: `(stick_index, intra_byte_offset)`; LX: `(byte_addr, 0)`.
    pub fn split_addr(&self, byte_addr: i64) -> (i64, i64) {
        match self.space {
            MemorySpace::Hbm => (byte_addr / STICK_BYTES, byte_addr % STICK_BYTES),
            MemorySpace::Lx { .. } => (byte_addr, 0),
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.bytes_per_elem()
    }

    /// Convert to a byte-addressed `TileRef` for load/store. Mirrors `to_tile_ref`.
    pub fn to_tile_ref(&self) -> TileRef {
        TileRef {
            base_ptr: self.byte_address(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            memref: Box::new(self.clone()),
            coordinate_set: None,
            partition_origin: None,
        }
    }
}

/// Composition of N per-partition `MemRef`s — result of
/// `construct_distributed_memory_view`. Bookkeeping only; no allocation.
#[derive(Clone, Debug)]
pub struct DistributedMemRef {
    pub partitions: Vec<MemRef>,
    /// Global logical shape (coordinate_sets use these coords).
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl DistributedMemRef {
    /// Validate-on-construct, mirroring Python's `__post_init__`.
    pub fn new(partitions: Vec<MemRef>, shape: Vec<usize>, dtype: DType) -> Result<Self, String> {
        if partitions.is_empty() {
            return Err("DistributedMemRef requires at least one partition".into());
        }
        for (i, p) in partitions.iter().enumerate() {
            if p.coordinate_set.is_none() {
                return Err(format!(
                    "DistributedMemRef partition {i} must have a coordinate_set"
                ));
            }
            if p.dtype != dtype {
                return Err(format!(
                    "DistributedMemRef partition {i} dtype {} != view dtype {}",
                    p.dtype, dtype
                ));
            }
        }
        Ok(DistributedMemRef {
            partitions,
            shape,
            dtype,
        })
    }

    /// First partition whose coordinate_set contains `coord`. Per RFC 0682 §3.3,
    /// overlapping sets are unspecified and "first match" is a legal resolution.
    /// Mirrors `find_partition`.
    pub fn find_partition(&self, coord: &[i64], syms: &[i64]) -> Result<(usize, &MemRef), String> {
        for (i, p) in self.partitions.iter().enumerate() {
            if p.coordinate_set.as_ref().unwrap().contains(coord, syms) {
                return Ok((i, p));
            }
        }
        Err(format!(
            "no partition of DistributedMemRef contains global coord {coord:?}"
        ))
    }
}

/// Byte-addressed sub-tile view — result of `construct_access_tile` on a
/// single allocation. `base_ptr` is always an absolute byte address.
#[derive(Clone, Debug)]
pub struct TileRef {
    pub base_ptr: i64,
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
    pub dtype: DType,
    /// Parent view — owns memory-space dispatch + hw address conversion.
    pub memref: Box<MemRef>,
    /// Per-survivor metadata from `distributed_tile_access` (None on ordinary refs).
    pub coordinate_set: Option<CoordinateSet>,
    /// `p_i = min(B_i)` in global coords.
    pub partition_origin: Option<Vec<i64>>,
}

impl TileRef {
    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.bytes_per_elem()
    }
}

/// Per-partition survivors of a distributed access — result of
/// `distributed_tile_access`.
#[derive(Clone, Debug)]
pub struct DistributedTileRef {
    pub partitions: Vec<TileRef>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// `x = base_map.eval(indices)` — origin of the access tile in global coords.
    pub global_base: Option<Vec<i64>>,
}

/// Parent of an `AccessTile`: single-allocation `TileRef` or, once partition
/// routing is resolved, a `DistributedTileRef`. Mirrors Python's
/// `Union[TileRef, DistributedTileRef]`.
#[derive(Clone, Debug)]
pub enum ParentRef {
    Tile(TileRef),
    Dist(DistributedTileRef),
}

/// Coordinate access tile referencing a sub-region of a memref — the affine
/// descriptor load/store consume.
#[derive(Clone, Debug)]
pub struct AccessTile {
    pub parent_ref: ParentRef,
    pub shape: Vec<usize>,
    /// Always present; synthesized as identity if absent in MLIR.
    pub base_map: AffineMap,
    /// Parsed `access_tile_set`; None if omitted.
    pub coordinate_set: Option<AffineSet>,
    /// Parsed `access_tile_order`; None if omitted.
    pub coordinate_order: Option<AffineMap>,
}

/// A per-dimension subscript expression: a quasi-affine [`AffineExpr`] over the
/// intermediate-variable point (`Dim(i)` = enumeration variable `%di`) plus
/// symbols (`Sym(j)`) for outer SSA scalars (`%grid0`, `%bt_idx`, `%c0`, ...).
///
/// `syms` holds those SSA scalars' concrete values, resolved against the value
/// table at `construct_indirect_access_tile` execution time — mirroring the
/// Python `_resolve_node` step that folds `("ssa", "%name")` to `("const", v)`
/// before load. Evaluating at an enumeration `pt` is `expr.eval(pt, &syms)`.
#[derive(Clone, Debug)]
pub struct SubExpr {
    pub expr: AffineExpr,
    /// Resolved outer-SSA symbol values, indexed by `Sym(j)`.
    pub syms: Vec<i64>,
}

impl SubExpr {
    /// Evaluate the subscript at the enumeration point `pt`.
    pub fn eval(&self, pt: &[i64]) -> i64 {
        self.expr.eval(pt, &self.syms)
    }
}

/// Per-dimension descriptor for an indirect access tile. Mirrors the `kind`
/// tagged dict entries in Python's `dim_subscripts`.
#[derive(Clone, Debug)]
pub enum DimSubscript {
    /// Dimension indexed directly by an intermediate variable.
    Direct { var_index: usize },
    /// Dimension indexed by an affine expression over the variable point.
    DirectExpr { map: AffineMap },
    /// Dimension indexed by a quasi-affine subscript expression over the
    /// variable point and outer SSA scalars (e.g. `%dim1_start + %d1`).
    DirectSub { sub: SubExpr },
    /// Dimension indexed indirectly via a lookup into `index_views[view]`.
    ///
    /// `idx_exprs` holds one subscript expression per dimension of the index
    /// view (e.g. `ind(%bt[%c0, %bt_idx + %d0])` → two exprs). Each is evaluated
    /// at every enumeration point to address the index view; the loaded value is
    /// the parent-tensor coordinate for this dim. An empty `idx_exprs` selects
    /// the legacy identity-subscript path (address the view by the point itself).
    Indirect {
        view: usize,
        idx_exprs: Vec<SubExpr>,
    },
}

/// Indirect access tile for gather/scatter — result of
/// `construct_indirect_access_tile`. Each output dimension is indexed directly
/// (via an intermediate variable) or indirectly (via an index memory view).
#[derive(Clone, Debug)]
pub struct IndirectAccessTile {
    /// Primary memory view being gathered/scattered (e.g. X).
    pub parent_ref: MemRef,
    /// Output access-tile shape.
    pub shape: Vec<usize>,
    /// Per-output-dim descriptor.
    pub dim_subscripts: Vec<DimSubscript>,
    /// Index memrefs for the indirect dims (used for byte addressing).
    pub index_views: Vec<MemRef>,
    /// Domain of the intermediate variables.
    pub variables_space_set: AffineSet,
    /// Iteration order over the variable space; `None` = default.
    pub variables_space_order: Option<AffineMap>,
    /// Extra per-dim metadata (parser-populated), kept open for the impl phase.
    pub extra: HashMap<String, Vec<i64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hbm_memref() -> MemRef {
        MemRef {
            // element index: stick 4 at f16 = 4*128/2 = 256 elements
            base_ptr: 256,
            shape: vec![64, 32],
            strides: vec![32, 1],
            space: MemorySpace::Hbm,
            dtype: DType::F16,
            coordinate_set: None,
        }
    }

    #[test]
    fn memory_space_invariant_is_structural() {
        assert!(MemorySpace::parse("HBM", Some(0)).is_err());
        assert_eq!(
            MemorySpace::parse("LX", Some(3)).unwrap(),
            MemorySpace::Lx { core_id: Some(3) }
        );
        assert_eq!(
            MemorySpace::parse("LX", None).unwrap(),
            MemorySpace::Lx { core_id: None }
        );
        assert!(MemorySpace::parse("DDR", None).is_err());
    }

    #[test]
    fn hbm_byte_address_and_split() {
        let m = hbm_memref();
        // element index 256 * 2 bytes (f16) = 512 bytes
        assert_eq!(m.byte_address(), 512);
        // a byte address splits into (stick, intra)
        assert_eq!(m.split_addr(512 + 5), (4, 5));
    }

    #[test]
    fn lx_is_byte_addressed_directly() {
        let m = MemRef {
            // element index 64 at f32 = 64*4 = 256 bytes
            base_ptr: 64,
            shape: vec![16],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        assert_eq!(m.byte_address(), 256);
        assert_eq!(m.split_addr(300), (300, 0));
    }

    #[test]
    fn to_tile_ref_carries_byte_address() {
        let m = hbm_memref();
        let tr = m.to_tile_ref();
        assert_eq!(tr.base_ptr, 512);
        assert_eq!(tr.dtype, DType::F16);
        assert_eq!(tr.shape, vec![64, 32]);
    }

    #[test]
    fn size_bytes_matches_shape_times_elem() {
        // 64*32 f16 = 2048 elems * 2 bytes
        assert_eq!(hbm_memref().size_bytes(), 64 * 32 * 2);
    }
}
