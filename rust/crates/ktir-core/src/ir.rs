// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Core IR data structures — Rust port of `ktir_cpu/ir_types.py` (the
//! `Operation` / `IRFunction` / `IRModule` half; the memref/tile types live in
//! `memref.rs` and `tile.rs`).
//!
//! The keystone type here is [`Value`]: it replaces Python's `Any` as the type
//! that flows through every SSA binding, operand lookup, and handler return.

use std::collections::HashMap;

use crate::affine::{AffineMap, AffineSet};
use crate::dtypes::DType;
use crate::memref::{
    AccessTile, DistributedMemRef, DistributedTileRef, IndirectAccessTile, MemRef, TileRef,
};
use crate::tile::Tile;

/// A scalar SSA value (e.g. `arith.constant`, a loop induction variable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    F32(f32),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl Scalar {
    pub fn as_f32(self) -> Option<f32> {
        match self {
            Scalar::F32(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Scalar::I32(v) => Some(v as i64),
            Scalar::I64(v) => Some(v),
            _ => None,
        }
    }
}

/// Anything an SSA value can hold — the single tagged union that replaces
/// Python's `Any` in the per-core scope map. Each dialect handler `match`es to
/// extract the variant it expects; an unexpected variant is a typed error
/// rather than a runtime `AttributeError`.
///
/// Memref/tileref variants are declared but unused in the arith slice; they
/// land as `memref.rs` / `tile.rs` grow. Kept here so the enum is the one
/// authoritative list of SSA value kinds from the start.
#[derive(Clone, Debug)]
pub enum Value {
    Scalar(Scalar),
    Tile(Tile),
    Index(i64),
    Tuple(Vec<Value>),
    MemRef(MemRef),
    DistMemRef(DistributedMemRef),
    TileRef(TileRef),
    DistTileRef(DistributedTileRef),
    AccessTile(AccessTile),
    IndirectAccessTile(IndirectAccessTile),
    /// Per-core handle produced by `ktdp.inter_tile_produce`, consumed by
    /// `ktdp.inter_tile_reduce`. Carries this core's partial plus the parsed
    /// producer/groups affine sets and the resolved group index. Mirrors the
    /// Python `TileFuture` dataclass.
    TileFuture(Box<TileFuture>),
}

/// Per-core handle produced by `ktdp.inter_tile_produce`. SPMD: each core holds
/// its own instance bound to its local `%fut` SSA value; cross-core data movement
/// happens via the scheduler's ring all-reduce when the matching
/// `ktdp.inter_tile_reduce` runs, not by reading other cores' futures. 1:1 with
/// `ktir_cpu.ir_types.TileFuture`.
#[derive(Clone, Debug)]
pub struct TileFuture {
    /// This core's yielded partial — the seed for the transport. `None` when the
    /// core is in `groups_set` but outside `producer_set` (the reduce backend
    /// substitutes the identity tensor for it). The examples yield a single tile.
    pub local_partial: Option<Tile>,
    /// Parsed `producer_tiles_per_group` set, kept on the future so the consumer
    /// can build the ring plan without re-parsing.
    pub producer_set: AffineSet,
    /// Parsed `groups` set.
    pub groups_set: AffineSet,
    /// The group this core belongs to, computed once at produce time.
    pub group_idx: i64,
}

/// A parsed operation attribute. Replaces the `Any` values in Python's
/// `Operation.attributes` dict; the parser picks the variant, handlers match.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Int(i64),
    IntList(Vec<i64>),
    Float(f64),
    FloatList(Vec<f64>),
    Str(String),
    StrList(Vec<String>),
    Bool(bool),
    Dtype(DType),
    AffineMap(AffineMap),
    AffineMapList(Vec<AffineMap>),
    AffineSet(AffineSet),
}

/// A single IR operation. 1:1 with the Python `Operation` dataclass.
#[derive(Clone, Debug)]
pub struct Operation {
    /// Result SSA name, e.g. `Some("%x")`. `None` for ops with no result.
    pub result: Option<String>,
    /// Op type, e.g. `"arith.addf"`, `"ktdp.construct_memory_view"`.
    pub op_type: String,
    pub operands: Vec<String>,
    pub attributes: HashMap<String, Attr>,
    pub result_type: Option<String>,
    /// Nested regions (scf bodies). Empty for straight-line ops.
    pub regions: Vec<Vec<Operation>>,
}

impl Operation {
    /// Terse constructor for tests / hand-built IR.
    pub fn new(result: Option<&str>, op_type: &str, operands: &[&str]) -> Self {
        Operation {
            result: result.map(String::from),
            op_type: op_type.to_string(),
            operands: operands.iter().map(|s| s.to_string()).collect(),
            attributes: HashMap::new(),
            result_type: None,
            regions: Vec::new(),
        }
    }

    pub fn with_attr(mut self, key: &str, val: Attr) -> Self {
        self.attributes.insert(key.to_string(), val);
        self
    }
}

/// An IR function: arguments, a flat op list, and the grid shape.
#[derive(Clone, Debug)]
pub struct IRFunction {
    pub name: String,
    pub arguments: Vec<(String, String)>, // (name, type)
    pub operations: Vec<Operation>,
    pub grid: (usize, usize, usize),
    pub return_type: Option<String>,
}

impl IRFunction {
    /// Argument names with the leading `%` stripped — mirrors `arg_names`.
    pub fn arg_names(&self) -> Vec<String> {
        self.arguments
            .iter()
            .map(|(n, _)| n.trim_start_matches('%').to_string())
            .collect()
    }
}

/// Top-level module: named functions plus module-scope attribute aliases.
#[derive(Clone, Debug, Default)]
pub struct IRModule {
    pub functions: HashMap<String, IRFunction>,
    /// `#name -> verbatim value string`, e.g. `"#X_coord_set" -> "affine_set<...>"`.
    pub aliases: HashMap<String, String>,
}

impl IRModule {
    pub fn get_function(&self, name: &str) -> Result<&IRFunction, String> {
        self.functions
            .get(name)
            .ok_or_else(|| format!("Function '{name}' not found in module"))
    }

    pub fn add_function(&mut self, func: IRFunction) {
        self.functions.insert(func.name.clone(), func);
    }
}
