// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Memory load/store data path — Rust port of the `MemoryOps.load` /
//! `MemoryOps.store` helpers from `ktir_emulator/ops/memory_ops.py` plus the
//! `ktdp.load` / `ktdp.store` handlers from `ktir_emulator/dialects/ktdp_ops.py`.
//!
//! `MemRef` / `TileRef` in this crate carry **byte-addressed** bases (see
//! `memref.rs::to_tile_ref` / `tile_access`), so the `_MemAccessor` stick/intra
//! split that Python performs is folded into the absolute-byte reads against
//! `HBMSimulator::read_bytes` / `LXScratchpad::read_bytes`.
//!
//! Tile storage is a flat `Vec<f32>` (see `tile.rs`). Bytes are decoded into
//! f32 at the load boundary per the source dtype (f16 half-precision decode,
//! f32 bit-cast, i32/i64 integer decode widened to f32) and re-encoded
//! symmetrically on store. The f16 round trip is implemented inline
//! (round-to-nearest-even) with no `half` crate dependency.
//!
//! Two paths, mirroring the Python source:
//!   * **fast path** — `coordinate_set` absent and the tile is contiguous
//!     (row-major). A single span read/write of the whole footprint.
//!   * **slow path** — a `coordinate_set` is present (or the tile is strided):
//!     enumerate the local coords via the affine set, optionally reorder them
//!     through `coordinate_order`, linearize to flat element offsets, read one
//!     contiguous span, and gather/scatter via fancy indexing.
//!
//! HBM loads/stores compute `unique_sticks` (the distinct 128-byte sticks the
//! transfer touches); LX has no stick concept and reports `None`/`0`.
//!
//! Beyond the single-allocation path this module also owns the distributed and
//! indirect data paths:
//!   * **distributed** (`distributed_tile_access` / `distributed_load` /
//!     `distributed_store`) — gather/scatter across the surviving partitions of
//!     a `DistributedMemRef`, mirroring `MemoryOps.distributed_*`.
//!   * **indirect** (`indirect_load` / `indirect_store`) — gather/scatter via
//!     index views, mirroring `MemoryOps.indirect_*`.

use crate::affine::{AffineMap, AffineSet, BoxSet, SymBoxSet, eval_bound};
use crate::context::CoreContext;
use crate::dialects::{Dispatch, LatencyCategory};
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};
use crate::memory::STICK_BYTES;
use crate::memref::{
    AccessTile, CoordinateSet, DimSubscript, DistributedMemRef, DistributedTileRef,
    IndirectAccessTile, MemorySpace, ParentRef, TileRef,
};
use crate::tile::Tile;

pub fn register(d: &mut Dispatch) {
    d.register("ktdp.load", LatencyCategory::Memory, load);
    d.register("ktdp.store", LatencyCategory::Memory, store);
}

// ===========================================================================
// ktdp.load / ktdp.store handlers
// ===========================================================================

/// `%t = ktdp.load %access_tile` — gather the access tile's footprint into LX.
///
/// Mirrors `ktdp__load`. Three shapes of operand are accepted:
///   * a single-allocation `AccessTile` (`ParentRef::Tile`) — the original
///     fast/slow gather path; when the access tile carries a `coordinate_set`,
///     enumerate its coords (reordered through `coordinate_order`) before the
///     slow gather, otherwise load the whole contiguous/strided tile;
///   * a distributed `AccessTile` (`ParentRef::Dist`) — gather across the
///     surviving partitions (`distributed_load`);
///   * an `IndirectAccessTile` — a gather through index views
///     (`indirect_load`).
fn load(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("ktdp.load: missing access-tile operand".into());
    }
    // Extract ONLY what the gather needs (the parent ref + resolved coords + shape)
    // while borrowing the value, then drop the borrow before the &mut-ctx gather.
    // This avoids cloning the whole `AccessTile` per load — notably its affine
    // `base_map`/`coordinate_set`/`coordinate_order` (consumed by `enumerated_coords`
    // here and never needed again) — which dominated the load hot path's self time.
    enum Plan {
        Indirect(IndirectAccessTile),
        Tile(TileRef, Option<Vec<Vec<i64>>>, Option<Vec<usize>>),
        Dist(DistributedTileRef, Vec<usize>),
    }
    let plan = match ctx.get_value(&op.operands[0])? {
        Value::IndirectAccessTile(iat) => Plan::Indirect(iat.clone()),
        Value::AccessTile(access) => {
            let coords = enumerated_coords(access);
            let result_shape = coords.as_ref().map(|_| access.shape.clone());
            match &access.parent_ref {
                ParentRef::Tile(tr) => Plan::Tile(tr.clone(), coords, result_shape),
                ParentRef::Dist(dist) => Plan::Dist(dist.clone(), access.shape.clone()),
            }
        }
        other => {
            return Err(format!(
                "ktdp.load: expected an AccessTile or IndirectAccessTile, got {other:?}"
            ));
        }
    };
    let tile = match plan {
        Plan::Indirect(iat) => indirect_load(ctx, &iat, None)?,
        Plan::Tile(tile_ref, coords, result_shape) => {
            load_data(ctx, &tile_ref, coords.as_deref(), result_shape)?
        }
        Plan::Dist(dist, shape) => distributed_load(ctx, &dist, Some(shape))?,
    };
    Ok(Some(Value::Tile(tile)))
}

/// `ktdp.store %tile, %access_tile` — scatter a tile back to its footprint.
///
/// Stores have no IR result; the handler computes `unique_sticks` (the latency
/// sideband Python returns) but binds nothing. Mirrors `ktdp__store`. The
/// second operand may be a single-allocation `AccessTile`, a distributed
/// `AccessTile` (`ParentRef::Dist`), or an `IndirectAccessTile`.
fn store(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.len() < 2 {
        return Err(format!(
            "ktdp.store expects 2 operands (tile, access_tile), got {}",
            op.operands.len()
        ));
    }
    let tile = match ctx.get_value(&op.operands[0])? {
        Value::Tile(t) => t.clone(),
        other => return Err(format!("ktdp.store expects a Tile, got {other:?}")),
    };

    // Extract the destination plan while borrowing the access-tile value, then drop
    // the borrow before the &mut-ctx scatter — avoids cloning the whole `AccessTile`
    // (incl. its affine maps) per store, mirroring `load`. The unique-stick sideband
    // is computed inside the scatter and bound to no SSA result (the op has none).
    enum Plan {
        Indirect(IndirectAccessTile),
        Tile(TileRef, Option<Vec<Vec<i64>>>),
        Dist(DistributedTileRef),
    }
    let plan = match ctx.get_value(&op.operands[1])? {
        Value::IndirectAccessTile(iat) => Plan::Indirect(iat.clone()),
        Value::AccessTile(access) => {
            let coords = enumerated_coords(access);
            match &access.parent_ref {
                ParentRef::Tile(tr) => Plan::Tile(tr.clone(), coords),
                ParentRef::Dist(dist) => Plan::Dist(dist.clone()),
            }
        }
        other => {
            return Err(format!(
                "ktdp.store: expected an AccessTile or IndirectAccessTile, got {other:?}"
            ));
        }
    };
    match plan {
        Plan::Indirect(iat) => {
            let _unique_sticks = indirect_store(ctx, &tile, &iat)?;
        }
        Plan::Tile(tile_ref, coords) => {
            let _unique_sticks = store_data(ctx, &tile, &tile_ref, coords.as_deref())?;
        }
        Plan::Dist(dist) => {
            let _unique_sticks = distributed_store(ctx, &tile, &dist)?;
        }
    }
    Ok(None)
}

/// Resolve the access tile's coordinate list, if it carries a coordinate_set.
/// Enumerate over `access.shape`, then reorder each point through
/// `coordinate_order` when present (mirrors `css.enumerate` + `cso.eval`).
fn enumerated_coords(access: &AccessTile) -> Option<Vec<Vec<i64>>> {
    let css = access.coordinate_set.as_ref()?;
    // Fast-path bypass: a coordinate set covering the full `[0, shape)` box with
    // an identity iteration order selects exactly the whole tile in row-major
    // order — identical to a plain contiguous load/store. Returning `None` takes
    // that fast path instead of enumerating every point (the O(2^n) `is_full`
    // vertex check vs O(∏ shape) enumeration + element-wise gather/scatter).
    let order_is_identity = access
        .coordinate_order
        .as_ref()
        .is_none_or(|m| m.is_identity());
    if order_is_identity && css.is_full(&access.shape) {
        return None;
    }
    let mut coords = css.enumerate(&access.shape, &[]);
    if let Some(order) = &access.coordinate_order {
        coords = coords.iter().map(|pt| order.eval(pt, &[])).collect();
    }
    Some(coords)
}

// ===========================================================================
// Distributed memory views — port of MemoryOps.distributed_* (RFC 0682 §3.3)
//
// Naming used throughout:
//   x   = global_base = base_map.eval(indices) — global origin of the access
//   A   = access_tile_set, in local coords 0..access_shape-1; None means the
//         full box [0, access_shape)
//   x+A = global footprint of the access tile
//   B_i = partition i's coordinate_set, in global coords
//   C_i = (x + A) ∩ B_i — global coords covered by both; per-survivor set
//   p_i = min(B_i) — partition i's origin in global coords
//
// distributed_load consumes C_i and p_i directly:
//   load coords (partition-local) = C_i - p_i
//   output coords (access-local)  = C_i - x
// ===========================================================================

/// Port of `MemoryOps.distributed_tile_access`. Resolve partition routing once
/// and return a [`DistributedTileRef`] whose survivors each carry a
/// per-survivor `coordinate_set` (`C_i`) and `partition_origin` (`p_i`).
///
/// Fast path: when partition `B_i` lowers to a [`BoxSet`] and `x + A` is a box,
/// compute `C_i = B_i ∩ (x + A)` in O(ndim). Slow path: enumerate `B_i` over
/// the global shape and filter by membership in `x + A`. Empty intersections
/// are skipped. Raises if no partition covers the access region.
pub fn distributed_tile_access(
    dist_ref: &DistributedMemRef,
    access_shape: &[usize],
    base_map: &AffineMap,
    indices: &[i64],
    access_tile_set: Option<&AffineSet>,
) -> Result<DistributedTileRef, String> {
    let x = base_map.eval(indices, &[]);
    let ndim = dist_ref.shape.len();
    if x.len() != ndim {
        return Err(format!(
            "distributed_tile_access: base_map produced {} coords but view has {} dims",
            x.len(),
            ndim
        ));
    }

    // Pre-compute (x + A) as an (inclusive) BoxSet when possible. None ⇒ A is
    // the implicit full box [0, access_shape). The inclusive box spans
    // [x, x + access_shape - 1] per axis.
    let xa_box: Option<BoxSet> = match access_tile_set {
        None => Some(BoxSet::new(
            x.clone(),
            (0..ndim)
                .map(|d| x[d] + access_shape[d] as i64 - 1)
                .collect(),
        )),
        // Lower A to an inclusive box (if axis-aligned) then translate by x.
        Some(aset) => lower_to_box(aset).map(|b| {
            BoxSet::new(
                (0..ndim).map(|d| b.lo[d] + x[d]).collect(),
                (0..ndim).map(|d| b.hi[d] + x[d]).collect(),
            )
        }),
    };

    // Slow-path membership: point ∈ x + A.
    let in_xa = |p: &[i64]| -> bool {
        match access_tile_set {
            None => (0..ndim).all(|d| {
                let local = p[d] - x[d];
                0 <= local && local < access_shape[d] as i64
            }),
            Some(aset) => {
                let local: Vec<i64> = (0..ndim).map(|d| p[d] - x[d]).collect();
                aset.contains(&local, &[])
            }
        }
    };

    let mut survivors: Vec<TileRef> = Vec::new();
    for part in &dist_ref.partitions {
        // Every distributed partition carries a coordinate_set (enforced at
        // construction). It is stored as an AffineSet (B_i in global coords).
        let b_set = part.coordinate_set.as_ref().ok_or_else(|| {
            "distributed_tile_access: partition missing coordinate_set".to_string()
        })?;

        // Try the box fast path: B_i lowers to a box and x+A is a box.
        let b_box = lower_to_box(b_set);
        let (coordinate_set_out, p_i): (CoordinateSet, Vec<i64>) =
            match (b_box.as_ref(), xa_box.as_ref()) {
                (Some(bbox), Some(xa)) => match bbox.intersect(xa) {
                    None => continue, // empty intersection
                    Some(ci) => (CoordinateSet::Box(ci), bbox.origin().to_vec()),
                },
                _ => {
                    // Slow path: enumerate B_i and filter by membership in x+A.
                    let b_pts = b_set.enumerate(&dist_ref.shape, &[]);
                    if b_pts.is_empty() {
                        continue;
                    }
                    let p_i: Vec<i64> = (0..ndim)
                        .map(|d| b_pts.iter().map(|pt| pt[d]).min().unwrap())
                        .collect();
                    let ci_pts: Vec<Vec<i64>> = b_pts.into_iter().filter(|pt| in_xa(pt)).collect();
                    if ci_pts.is_empty() {
                        continue;
                    }
                    (CoordinateSet::Points(ci_pts), p_i)
                }
            };

        survivors.push(TileRef {
            base_ptr: part.byte_address(),
            shape: part.shape.clone(),
            strides: part.strides.clone(),
            dtype: part.dtype,
            memref: Box::new(part.clone()),
            coordinate_set: Some(coordinate_set_out),
            partition_origin: Some(p_i),
        });
    }

    if survivors.is_empty() {
        return Err(format!(
            "distributed_tile_access: no partition covers access region \
             global_base={x:?} shape={access_shape:?}"
        ));
    }
    Ok(DistributedTileRef {
        partitions: survivors,
        shape: dist_ref.shape.clone(),
        dtype: dist_ref.dtype,
        global_base: Some(x),
    })
}

/// Lower an [`AffineSet`] to an **inclusive** [`BoxSet`] (`[lo, hi]`), or
/// `None` when the set is not axis-aligned / not representable as a box.
///
/// `SymBoxSet::try_from_affine_set` yields a half-open `[lo, hi)` box; for
/// distributed routing the partition / access sets are concrete, so we resolve
/// with no symbols and shrink the exclusive upper bound to inclusive (`hi - 1`).
fn lower_to_box(aset: &AffineSet) -> Option<BoxSet> {
    let sym = SymBoxSet::try_from_affine_set(aset)?;
    if !sym.is_concrete() {
        return None;
    }
    let lo: Vec<i64> = sym.lo.iter().map(|b| eval_bound(b, &[])).collect();
    let hi: Vec<i64> = sym.hi.iter().map(|b| eval_bound(b, &[]) - 1).collect();
    Some(BoxSet::new(lo, hi))
}

/// Port of `MemoryOps._subtile_ref`. Build a `TileRef` covering exactly the
/// global-coordinate `box` within `survivor`. Inherits the survivor's strides
/// verbatim; `shape` shrinks to the box extent and `base_ptr` shifts to the
/// box's partition-local origin (`box.lo - p_i`, scaled by bpe).
fn subtile_ref(survivor: &TileRef, b: &BoxSet) -> TileRef {
    let ndim = survivor.shape.len();
    let zero = vec![0i64; ndim];
    let p_i = survivor.partition_origin.as_deref().unwrap_or(&zero);
    let local_lo: Vec<i64> = (0..ndim).map(|d| b.lo[d] - p_i[d]).collect();
    // Inclusive box -> extent is hi - lo + 1.
    let sub_shape: Vec<usize> = (0..ndim)
        .map(|d| (b.hi[d] - b.lo[d] + 1) as usize)
        .collect();
    let bpe = survivor.dtype.bytes_per_elem() as i64;
    let byte_offset: i64 = (0..ndim)
        .map(|d| local_lo[d] * survivor.strides[d])
        .sum::<i64>()
        * bpe;
    TileRef {
        base_ptr: survivor.base_ptr + byte_offset,
        shape: sub_shape,
        strides: survivor.strides.clone(),
        dtype: survivor.dtype,
        memref: survivor.memref.clone(),
        coordinate_set: None,
        partition_origin: None,
    }
}

/// Port of `MemoryOps.distributed_load`. Gather across surviving partitions
/// into a single LX-resident [`Tile`].
///
/// Fast path (BoxSet `C_i`): build a sub-`TileRef` of the partition covering
/// exactly `C_i`, delegate the read to [`load_data`], and slot its data into a
/// rectangular slice of the output buffer. Slow path (`Points` `C_i`):
/// per-coord scatter — translate `C_i` to partition-local coords, read one
/// span, and scatter each element into the access-local position.
pub fn distributed_load(
    ctx: &mut CoreContext,
    dist_tile_ref: &DistributedTileRef,
    result_shape: Option<Vec<usize>>,
) -> Result<Tile, String> {
    let ndim = dist_tile_ref.shape.len();
    let zero_x = vec![0i64; ndim];
    let x = dist_tile_ref.global_base.as_deref().unwrap_or(&zero_x);
    let out_shape = result_shape.unwrap_or_else(|| dist_tile_ref.shape.clone());
    let out_len: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_len];
    let out_strides = row_major_strides(&out_shape);

    let mut total_unique_sticks = 0usize;
    let mut any_hbm = false;

    for survivor in &dist_tile_ref.partitions {
        let cs = survivor
            .coordinate_set
            .as_ref()
            .ok_or_else(|| "distributed_load: survivor missing coordinate_set".to_string())?;
        match cs {
            CoordinateSet::Box(b) => {
                // Fast path: rectangular sub-tile, then copy into out[C_i - x].
                let sub = subtile_ref(survivor, b);
                let tile = load_data(ctx, &sub, None, None)?;
                // access-local rectangle = C_i - x; copy row-major from tile.
                let access_lo: Vec<i64> = (0..ndim).map(|d| b.lo[d] - x[d]).collect();
                let sub_shape = &sub.shape;
                copy_rect_into(
                    &mut out,
                    &out_strides,
                    &access_lo,
                    sub_shape,
                    &tile.as_f32(),
                );
                if let Some(s) = tile.unique_sticks {
                    total_unique_sticks += s;
                    any_hbm = true;
                }
            }
            CoordinateSet::Points(ci) => {
                let zero_p = vec![0i64; ndim];
                let p_i = survivor.partition_origin.as_deref().unwrap_or(&zero_p);
                let local_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - p_i[d]).collect())
                    .collect();
                let access_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - x[d]).collect())
                    .collect();
                let space = survivor.memref.space;
                let stick_bytes = if ctx.track_sticks() {
                    stick_bytes_for(space)
                } else {
                    None
                };
                let (offsets, unique_sticks) = flat_memory_offsets(
                    survivor.base_ptr,
                    &survivor.shape,
                    &survivor.strides,
                    survivor.dtype,
                    Some(&local_coords),
                    stick_bytes,
                );
                let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
                let raw = read_raw(
                    ctx,
                    space,
                    survivor.base_ptr,
                    span * survivor.dtype.bytes_per_elem(),
                );
                let flat = decode(&raw, survivor.dtype, span);
                for (ac, &off) in access_coords.iter().zip(&offsets) {
                    let lin = lin_index(ac, &out_strides);
                    out[lin] = flat[off as usize];
                }
                if let Some(s) = unique_sticks {
                    total_unique_sticks += s;
                    any_hbm = true;
                }
            }
            CoordinateSet::Affine(_) => {
                return Err(
                    "distributed_load: survivor carries an un-lowered AffineSet \
                     coordinate_set (distributed_tile_access emits Box/Points only)"
                        .into(),
                );
            }
        }
    }

    write_to_lx(ctx, &out, dist_tile_ref.dtype);
    Ok(Tile::from_decoded(
        out,
        dist_tile_ref.dtype,
        out_shape,
        if any_hbm {
            Some(total_unique_sticks)
        } else {
            None
        },
        None,
    ))
}

/// Port of `MemoryOps.distributed_store`. Scatter a tile to surviving
/// partitions, symmetric to [`distributed_load`]. Returns the aggregate
/// `unique_sticks` (HBM stick cost; `0` for all-LX).
pub fn distributed_store(
    ctx: &mut CoreContext,
    tile: &Tile,
    dist_tile_ref: &DistributedTileRef,
) -> Result<usize, String> {
    let ndim = dist_tile_ref.shape.len();
    let zero_x = vec![0i64; ndim];
    let x = dist_tile_ref.global_base.as_deref().unwrap_or(&zero_x);
    let src_strides = row_major_strides(&tile.shape);
    let tile_data = tile.as_f32();

    let mut total_unique_sticks = 0usize;
    for survivor in &dist_tile_ref.partitions {
        let cs = survivor
            .coordinate_set
            .as_ref()
            .ok_or_else(|| "distributed_store: survivor missing coordinate_set".to_string())?;
        match cs {
            CoordinateSet::Box(b) => {
                let sub = subtile_ref(survivor, b);
                // Slice the source tile rectangularly at C_i - x (row-major copy).
                let access_lo: Vec<i64> = (0..ndim).map(|d| b.lo[d] - x[d]).collect();
                let src = gather_rect(&tile_data, &src_strides, &access_lo, &sub.shape);
                let sub_tile = Tile::compute(src, survivor.dtype, sub.shape.clone());
                total_unique_sticks += store_data(ctx, &sub_tile, &sub, None)?;
            }
            CoordinateSet::Points(ci) => {
                let zero_p = vec![0i64; ndim];
                let p_i = survivor.partition_origin.as_deref().unwrap_or(&zero_p);
                let local_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - p_i[d]).collect())
                    .collect();
                let access_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - x[d]).collect())
                    .collect();
                let space = survivor.memref.space;
                let stick_bytes = if ctx.track_sticks() {
                    stick_bytes_for(space)
                } else {
                    None
                };
                let (offsets, unique_sticks) = flat_memory_offsets(
                    survivor.base_ptr,
                    &survivor.shape,
                    &survivor.strides,
                    survivor.dtype,
                    Some(&local_coords),
                    stick_bytes,
                );
                let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
                let raw = read_raw(
                    ctx,
                    space,
                    survivor.base_ptr,
                    span * survivor.dtype.bytes_per_elem(),
                );
                let mut flat = decode(&raw, survivor.dtype, span);
                for (ac, &off) in access_coords.iter().zip(&offsets) {
                    let lin = lin_index(ac, &src_strides);
                    flat[off as usize] = tile_data[lin];
                }
                let new_raw = encode(&flat, survivor.dtype);
                write_raw(ctx, space, survivor.base_ptr, &new_raw);
                if let Some(s) = unique_sticks {
                    total_unique_sticks += s;
                }
            }
            CoordinateSet::Affine(_) => {
                return Err(
                    "distributed_store: survivor carries an un-lowered AffineSet \
                     coordinate_set (distributed_tile_access emits Box/Points only)"
                        .into(),
                );
            }
        }
    }
    Ok(total_unique_sticks)
}

// ===========================================================================
// Indirect access tiles — port of MemoryOps.indirect_load / indirect_store
// ===========================================================================

/// Port of `MemoryOps.indirect_load`. Enumerate the variable space (in
/// `variables_space_order` order), resolve each coordinate tuple (direct dims
/// from the variable point, indirect dims via index-view lookups), and delegate
/// the gather to [`load_data`]. Stamps `index_unique_sticks` on the result.
pub fn indirect_load(
    ctx: &mut CoreContext,
    iat: &IndirectAccessTile,
    result_shape: Option<Vec<usize>>,
) -> Result<Tile, String> {
    if let Some(vso) = &iat.variables_space_order
        && !vso.is_permutation()
    {
        return Err(format!(
            "indirect_load: variables_space_order must permute its input \
                 dimensions; got non-permutation map: {vso:?}"
        ));
    }

    let (idx_values, idx_unique_sticks) = resolve_idx_reads(ctx, iat)?;
    let coords = build_indirect_coords(iat, &idx_values)?;

    let out_shape = result_shape.unwrap_or_else(|| iat.shape.clone());
    let tile_ref = iat.parent_ref.to_tile_ref();
    let mut tile = load_data(ctx, &tile_ref, Some(&coords), Some(out_shape))?;
    tile.index_unique_sticks = Some(idx_unique_sticks);
    Ok(tile)
}

/// Port of `MemoryOps.indirect_store`. Mirror of [`indirect_load`]: enumerate,
/// resolve, build coords, then delegate the scatter to [`store_data`]. Returns
/// the aggregate stick cost (`data_sticks + idx_unique_sticks`).
pub fn indirect_store(
    ctx: &mut CoreContext,
    tile: &Tile,
    iat: &IndirectAccessTile,
) -> Result<usize, String> {
    if tile.shape != iat.shape {
        return Err(format!(
            "indirect_store: source tile shape {:?} does not match IAT shape {:?}",
            tile.shape, iat.shape
        ));
    }
    if let Some(vso) = &iat.variables_space_order
        && !vso.is_permutation()
    {
        return Err(format!(
            "indirect_store: variables_space_order must permute its input \
                 dimensions; got non-permutation map: {vso:?}"
        ));
    }

    let (idx_values, idx_unique_sticks) = resolve_idx_reads(ctx, iat)?;
    let coords = build_indirect_coords(iat, &idx_values)?;
    let tile_ref = iat.parent_ref.to_tile_ref();
    let data_sticks = store_data(ctx, tile, &tile_ref, Some(&coords))?;
    Ok(data_sticks + idx_unique_sticks)
}

/// Port of `_enumerate_in_vso_order`. Enumerate variable-space points; if a
/// non-identity `variables_space_order` is set, sort the points by the map's
/// image (lexicographic on the result vector) so idx reads and coord build stay
/// in lockstep (RFC 0682 §473). Callers must already have rejected
/// non-permutation maps.
fn enumerate_in_vso_order(iat: &IndirectAccessTile) -> Vec<Vec<i64>> {
    let mut points = iat.variables_space_set.enumerate(&iat.shape, &[]);
    if let Some(vso) = &iat.variables_space_order
        && !vso.is_identity()
    {
        points.sort_by_key(|a| vso.eval(a, &[]));
    }
    points
}

/// Port of `_resolve_idx_reads`. For every indirect dimension, read the index
/// value its index view holds at each enumerated point, returning a map from
/// `view -> values` (one entry per enumerated point, in pt order) plus the
/// total distinct HBM sticks touched by those reads.
///
/// Each indirect dim's [`SubExpr`] list (`idx_exprs`) gives the per-view-dim
/// subscript expression, evaluated at the enumeration point with `%di` bound to
/// the point and outer SSA scalars (`%grid0`, `%bt_idx`, ...) pre-resolved into
/// the `SubExpr`'s `syms`. The view element offset is
/// `Σ idx_exprs[d].eval(pt) * stride[d]` over the view's rank — mirroring the
/// Python `sum(eval_subscript_expr(e, pt) * s for e, s in zip(idx_exprs, strides))`.
///
/// Legacy / structural callers may leave `idx_exprs` empty (the identity case
/// the `port_indirect_access` tests exercise): then the point itself addresses
/// the view via `Σ pt[d] * stride[d]`, the prior behaviour.
fn resolve_idx_reads(
    ctx: &CoreContext,
    iat: &IndirectAccessTile,
) -> Result<(std::collections::HashMap<usize, Vec<i64>>, usize), String> {
    let points = enumerate_in_vso_order(iat);

    // The distinct index views used by indirect dims, in first-seen order.
    let mut view_idxs: Vec<usize> = Vec::new();
    for sub in &iat.dim_subscripts {
        if let DimSubscript::Indirect { view, .. } = sub
            && !view_idxs.contains(view)
        {
            view_idxs.push(*view);
        }
    }

    let mut per_view_values: std::collections::HashMap<usize, Vec<i64>> =
        std::collections::HashMap::new();
    let mut total_sticks = 0usize;

    for &iv_idx in &view_idxs {
        let iv = iat
            .index_views
            .get(iv_idx)
            .ok_or_else(|| format!("indirect: index_view {iv_idx} out of range"))?;
        let rank = iv.strides.len();
        let bpe = iv.dtype.bytes_per_elem();
        let base = iv.byte_address();
        let space = iv.space;
        let stick_bytes = if ctx.track_sticks() {
            stick_bytes_for(space)
        } else {
            None
        };
        let mut sticks: std::collections::HashSet<i64> = std::collections::HashSet::new();

        // For every enumerated point (and every indirect dim that uses this
        // view), read one index value. Indirect dims sharing a view append in
        // pt-major, dim-minor order — matching build_indirect_coords.
        let mut values: Vec<i64> = Vec::new();
        for pt in &points {
            for sub in &iat.dim_subscripts {
                if let DimSubscript::Indirect { view, idx_exprs } = sub {
                    if *view != iv_idx {
                        continue;
                    }
                    let offset: i64 = if idx_exprs.is_empty() {
                        // Legacy identity subscript: address the view by the
                        // point itself, projected onto the view's rank.
                        (0..rank)
                            .map(|d| pt.get(d).copied().unwrap_or(0) * iv.strides[d])
                            .sum()
                    } else {
                        // Evaluate the per-view-dim subscript expressions and dot
                        // with the view's strides (Python's zip(idx_exprs, strides)).
                        idx_exprs
                            .iter()
                            .zip(&iv.strides)
                            .map(|(e, &s)| e.eval(pt) * s)
                            .sum()
                    };
                    let byte_addr = base + offset * bpe as i64;
                    if let Some(sb) = stick_bytes {
                        sticks.insert(byte_addr / sb);
                    }
                    let raw = read_raw(ctx, space, byte_addr, bpe);
                    let v = decode(&raw, iv.dtype, 1)[0];
                    values.push(v as i64);
                }
            }
        }
        per_view_values.insert(iv_idx, values);
        if stick_bytes.is_some() {
            total_sticks += sticks.len();
        }
    }

    Ok((per_view_values, total_sticks))
}

/// Port of `_build_indirect_coords`. For each enumerated point, build the
/// parent-tensor coordinate tuple: `Direct` dims read the variable point,
/// `DirectExpr` dims evaluate their affine map over the point, and `Indirect`
/// dims consume the next pre-resolved index value (pt-major, dim-minor order).
/// Rejects negative indirect indices (NumPy would silently wrap).
fn build_indirect_coords(
    iat: &IndirectAccessTile,
    idx_values: &std::collections::HashMap<usize, Vec<i64>>,
) -> Result<Vec<Vec<i64>>, String> {
    let points = enumerate_in_vso_order(iat);
    // Per-view consumption cursors (positional, in lockstep with resolve_idx_reads).
    let mut cursors: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    let mut coords: Vec<Vec<i64>> = Vec::with_capacity(points.len());
    for pt in &points {
        let mut coord: Vec<i64> = Vec::with_capacity(iat.dim_subscripts.len());
        for sub in &iat.dim_subscripts {
            match sub {
                DimSubscript::Direct { var_index } => {
                    let v = *pt.get(*var_index).ok_or_else(|| {
                        format!("indirect: direct var_index {var_index} out of range")
                    })?;
                    coord.push(v);
                }
                DimSubscript::DirectExpr { map } => {
                    let r = map.eval(pt, &[]);
                    coord.push(r[0]);
                }
                DimSubscript::DirectSub { sub } => {
                    coord.push(sub.eval(pt));
                }
                DimSubscript::Indirect { view, .. } => {
                    let cur = cursors.entry(*view).or_insert(0);
                    let vals = idx_values
                        .get(view)
                        .ok_or_else(|| format!("indirect: no resolved values for view {view}"))?;
                    let raw_idx = *vals.get(*cur).ok_or_else(|| {
                        format!("indirect: ran out of resolved values for view {view}")
                    })?;
                    *cur += 1;
                    if raw_idx < 0 {
                        return Err(format!(
                            "indirect index {raw_idx} from index_view {view} is negative"
                        ));
                    }
                    coord.push(raw_idx);
                }
            }
        }
        coords.push(coord);
    }
    Ok(coords)
}

// ===========================================================================
// Row-major helpers for distributed rectangular slice copies
// ===========================================================================

/// Row-major (C-order) element strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as i64;
    }
    strides
}

/// Linear flat index of `coord` under `strides`.
fn lin_index(coord: &[i64], strides: &[i64]) -> usize {
    coord.iter().zip(strides).map(|(&c, &s)| c * s).sum::<i64>() as usize
}

/// Copy the row-major `src` (extent `sub_shape`) into `out` at the rectangle
/// whose origin is `lo` (access-local coords), under `out_strides`.
fn copy_rect_into(
    out: &mut [f32],
    out_strides: &[i64],
    lo: &[i64],
    sub_shape: &[usize],
    src: &[f32],
) {
    let mut i = 0usize;
    rect_iter(sub_shape, &mut |rel| {
        let abs: Vec<i64> = (0..rel.len()).map(|d| lo[d] + rel[d]).collect();
        out[lin_index(&abs, out_strides)] = src[i];
        i += 1;
    });
}

/// Gather the rectangle of `src` (origin `lo`, extent `sub_shape`, strides
/// `src_strides`) into a fresh row-major buffer.
fn gather_rect(src: &[f32], src_strides: &[i64], lo: &[i64], sub_shape: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(sub_shape.iter().product());
    rect_iter(sub_shape, &mut |rel| {
        let abs: Vec<i64> = (0..rel.len()).map(|d| lo[d] + rel[d]).collect();
        out.push(src[lin_index(&abs, src_strides)]);
    });
    out
}

/// Iterate the cartesian rectangle `[0, shape)` in row-major order, calling `f`
/// with each relative coordinate.
fn rect_iter(shape: &[usize], f: &mut impl FnMut(&[i64])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.contains(&0) {
        return;
    }
    let mut idx = vec![0i64; shape.len()];
    loop {
        f(&idx);
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

// ===========================================================================
// Core data path — port of MemoryOps.load / MemoryOps.store
// ===========================================================================

/// Port of `MemoryOps.load`. Reads the tile footprint from HBM/LX, decodes per
/// dtype into an f32 `Tile`, and writes the decoded tile into the executing
/// core's LX scratchpad. All loaded tiles land in LX regardless of source.
pub fn load_data(
    ctx: &mut CoreContext,
    tile_ref: &TileRef,
    coords: Option<&[Vec<i64>]>,
    result_shape: Option<Vec<usize>>,
) -> Result<Tile, String> {
    let dtype = tile_ref.dtype;
    let bpe = dtype.bytes_per_elem();
    let space = tile_ref.memref.space;
    // `unique_sticks` is only consumed by the latency tracker; skip computing it
    // (notably the per-element stick `HashSet` in the slow path) when untracked.
    let stick_bytes = if ctx.track_sticks() {
        stick_bytes_for(space)
    } else {
        None
    };

    // Fast path: contiguous tile, no coord filtering.
    if coords.is_none() && is_contiguous(&tile_ref.shape, &tile_ref.strides) {
        let n: usize = tile_ref.shape.iter().product();
        // Decode straight from the backing buffer — no intermediate byte Vec.
        let data = read_decoded(ctx, space, tile_ref.base_ptr, n, dtype);
        write_to_lx(ctx, &data, dtype);
        let unique_sticks = stick_bytes.map(|sb| {
            let end = tile_ref.base_ptr + (n * bpe) as i64;
            ((end + sb - 1) / sb - tile_ref.base_ptr / sb) as usize
        });
        return Ok(Tile::from_decoded(
            data,
            dtype,
            tile_ref.shape.clone(),
            unique_sticks,
            None,
        ));
    }

    // Row-contiguous fast path: the innermost axis is contiguous (stride 1) but an
    // outer axis is strided — a sub-tile of a wider tensor (e.g. a [w_q, w_k]
    // attention block carved from a [cap, w] tensor, stride [w, 1]). Each innermost
    // run of `inner` elements IS contiguous, so read+decode each run directly
    // (SIMD), skipping the offset Vec, the full-span copy (which spans the WHOLE
    // strided extent — ~16× the data actually read), and the per-element gather.
    // Only when not metering (the sticks sideband uses the slow path's set).
    if coords.is_none()
        && stick_bytes.is_none()
        && tile_ref.strides.last() == Some(&1)
        && !tile_ref.shape.is_empty()
    {
        let nd = tile_ref.shape.len();
        let inner = tile_ref.shape[nd - 1];
        let n: usize = tile_ref.shape.iter().product();
        let mut data = vec![0.0f32; n];
        if n > 0 {
            let outer_shape = &tile_ref.shape[..nd - 1];
            let outer_strides = &tile_ref.strides[..nd - 1];
            let outer_n = n / inner.max(1);
            let mut idx = vec![0usize; outer_shape.len()];
            for run in 0..outer_n {
                let elem_off: i64 = idx
                    .iter()
                    .zip(outer_strides)
                    .map(|(&c, &s)| c as i64 * s)
                    .sum();
                let addr = tile_ref.base_ptr + elem_off * bpe as i64;
                read_decoded_into(
                    ctx,
                    space,
                    addr,
                    &mut data[run * inner..run * inner + inner],
                    dtype,
                );
                // Advance the outer multi-index (rightmost = innermost outer axis).
                for d in (0..outer_shape.len()).rev() {
                    idx[d] += 1;
                    if idx[d] < outer_shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }
        write_to_lx(ctx, &data, dtype);
        let out_shape = result_shape.unwrap_or_else(|| tile_ref.shape.clone());
        return Ok(Tile::from_decoded(data, dtype, out_shape, None, None));
    }

    // Slow path: linearize coords/shape -> flat element offsets, single span
    // read, fancy-index gather.
    let (offsets, unique_sticks) = flat_memory_offsets(
        tile_ref.base_ptr,
        &tile_ref.shape,
        &tile_ref.strides,
        dtype,
        coords,
        stick_bytes,
    );
    let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
    let raw = read_raw(ctx, space, tile_ref.base_ptr, span * bpe);
    // Decode only the selected offsets — not the whole span (which can be ~18×
    // larger for strided accesses).
    let gathered = decode_gather(&raw, &offsets, dtype);

    let out_shape = result_shape.unwrap_or_else(|| tile_ref.shape.clone());
    write_to_lx(ctx, &gathered, dtype);
    Ok(Tile::from_decoded(
        gathered,
        dtype,
        out_shape,
        unique_sticks,
        None,
    ))
}

/// Port of `MemoryOps.store`. Encodes the tile's f32 data per dtype and writes
/// it to HBM/LX. Returns `unique_sticks` (distinct HBM sticks touched; `0` for
/// LX) — the latency sideband.
pub fn store_data(
    ctx: &mut CoreContext,
    tile: &Tile,
    tile_ref: &TileRef,
    coords: Option<&[Vec<i64>]>,
) -> Result<usize, String> {
    let dtype = tile_ref.dtype;
    let bpe = dtype.bytes_per_elem();
    let space = tile_ref.memref.space;
    // Latency-only sideband — skip the per-element stick set on untracked runs.
    let stick_bytes = if ctx.track_sticks() {
        stick_bytes_for(space)
    } else {
        None
    };
    let tile_data = tile.as_f32();

    // GRID-M-TILED STORE (fused single-core offload). When a K-loop GEMM is
    // offloaded as ONE full-M GEMM (the matmul-loop offload reconstructs M from
    // the activation view, e.g. prefill's [8,k]@[k,n]), the stored tile carries
    // ALL M rows, but the access-tile footprint is a SINGLE row `[1, w]` at
    // `[pid, off]` (the per-core SPMD store: each of M cores writes its own row).
    // In the fused [1,1] run pid=0, so the per-row store would write only row 0
    // and leave rows 1..M stale. A CONTIGUOUS footprint already writes the whole
    // tile via the fast path below (the full-width lm_head, all projections — they
    // work); only a STRIDED footprint (an N-tiled column window, the wide lm_head
    // split into column tiles) takes the slow path and would drop rows 1..M.
    //
    // Detect that case here: the stored tile has more elements than the footprint
    // and is a 2-D `[M, w]` whose width matches the footprint's last dim. Scatter
    // ALL M rows, mapping tile element `(r, c)` to `r * row_stride + c * col_stride`
    // off `base_ptr` (the view's own strides), which lands each row in its place.
    // This only fires when the stored tile is bigger than the footprint, which can
    // only happen via the single-core offload (multi-core keeps per-row [1,w]
    // tiles), so it never changes the multi-core SPMD scatter.
    let foot_numel: usize = tile_ref.shape.iter().product();
    if tile.len() > foot_numel
        && coords.is_none()
        // Whether the FULL M-row tile, laid out at the footprint's strides, is
        // non-contiguous — i.e. the rows are spread (row stride > width) so a plain
        // contiguous write would pack them wrong and drop rows 1..M. Checked on the
        // tile's `[M, w]` shape, NOT the `[1, w]` footprint: a `[1, w]` row is
        // itself contiguous (its extent-1 axis is never stepped), so testing the
        // footprint would wrongly skip the scatter for a strided multi-row store.
        && tile.shape.len() == 2
        && !is_contiguous(&tile.shape, &tile_ref.strides)
        && tile_ref.shape.len() == 2
        && tile_ref.strides.len() == 2
        && tile.shape[1] == tile_ref.shape[1]
        && tile.shape[0] * tile.shape[1] == tile.len()
    {
        let (rows, cols) = (tile.shape[0] as i64, tile.shape[1] as i64);
        let (rs, cs) = (tile_ref.strides[0], tile_ref.strides[1]);
        let mut offsets = Vec::with_capacity(tile.len());
        for r in 0..rows {
            for c in 0..cols {
                offsets.push(r * rs + c * cs);
            }
        }
        let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
        let raw = read_raw(ctx, space, tile_ref.base_ptr, span * bpe);
        let mut flat = decode(&raw, dtype, span);
        for (i, &o) in offsets.iter().enumerate() {
            flat[o as usize] = tile_data[i];
        }
        let new_raw = encode(&flat, dtype);
        write_raw(ctx, space, tile_ref.base_ptr, &new_raw);
        // Unique-stick sideband isn't needed on this single-core offload path.
        return Ok(0);
    }

    // Fast path: contiguous tile, no coord filtering.
    if coords.is_none() && is_contiguous(&tile_ref.shape, &tile_ref.strides) {
        let raw = encode(&tile_data, dtype);
        write_raw(ctx, space, tile_ref.base_ptr, &raw);
        return Ok(match stick_bytes {
            None => 0,
            Some(sb) => {
                let n: usize = tile_ref.shape.iter().product();
                let end = tile_ref.base_ptr + (n * bpe) as i64;
                ((end + sb - 1) / sb - tile_ref.base_ptr / sb) as usize
            }
        });
    }

    // Row-contiguous fast path (mirror of the load path): innermost axis is
    // contiguous (stride 1) but an outer axis is strided. Each innermost run is a
    // contiguous block, and the strided gaps between runs are NOT touched by this
    // store, so we can write each run directly — no read-modify-write of the whole
    // strided span (which read + decoded + re-encoded + wrote ~16× the data the
    // store actually changes). Untracked only (the sticks sideband uses the slow
    // path), and only when the tile exactly fills the footprint.
    let n: usize = tile_ref.shape.iter().product();
    if coords.is_none()
        && stick_bytes.is_none()
        && tile.len() == n
        && tile_ref.strides.last() == Some(&1)
        && !tile_ref.shape.is_empty()
        && n > 0
    {
        let nd = tile_ref.shape.len();
        let inner = tile_ref.shape[nd - 1];
        let outer_shape = &tile_ref.shape[..nd - 1];
        let outer_strides = &tile_ref.strides[..nd - 1];
        let outer_n = n / inner.max(1);
        let mut idx = vec![0usize; outer_shape.len()];
        for run in 0..outer_n {
            let elem_off: i64 = idx
                .iter()
                .zip(outer_strides)
                .map(|(&c, &s)| c as i64 * s)
                .sum();
            let addr = tile_ref.base_ptr + elem_off * bpe as i64;
            let run_raw = encode(&tile_data[run * inner..run * inner + inner], dtype);
            write_raw(ctx, space, addr, &run_raw);
            for d in (0..outer_shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < outer_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        return Ok(0);
    }

    // Slow path: read-modify-write via scatter offsets.
    let (offsets, unique_sticks) = flat_memory_offsets(
        tile_ref.base_ptr,
        &tile_ref.shape,
        &tile_ref.strides,
        dtype,
        coords,
        stick_bytes,
    );
    // Spec: a ktdp.store's data-tile shape is 1:1 with the access tile's logical
    // iteration shape, so the enumerated coord count must equal the data length.
    // A mismatch means the access tile escaped its nominal box (the production-
    // sized paged-tensor-write fixture: the indirect access enumerates far more
    // points than the loaded data tile holds). Return a clean Err instead of
    // panicking on the out-of-bounds index, so the differential harness can match
    // it against Python's ValueError rather than aborting the batch.
    if offsets.len() != tile_data.len() {
        return Err(format!(
            "ktdp.store: data tile has {} elements but the access tile enumerates \
             {} coordinates — store shape mismatch (access tile not contained in \
             its nominal box)",
            tile_data.len(),
            offsets.len()
        ));
    }
    let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
    let raw = read_raw(ctx, space, tile_ref.base_ptr, span * bpe);
    let mut flat = decode(&raw, dtype, span);
    // Scatter the C-order tile data into the span; last-writer-wins on coord
    // collisions (matches NumPy assignment).
    for (i, &o) in offsets.iter().enumerate() {
        flat[o as usize] = tile_data[i];
    }
    let new_raw = encode(&flat, dtype);
    write_raw(ctx, space, tile_ref.base_ptr, &new_raw);
    Ok(unique_sticks.unwrap_or(0))
}

// ===========================================================================
// Memory-space dispatch (folds the _MemAccessor stick/intra split)
// ===========================================================================

fn stick_bytes_for(space: MemorySpace) -> Option<i64> {
    match space {
        MemorySpace::Hbm => Some(STICK_BYTES),
        MemorySpace::Lx { .. } => None,
    }
}

/// Read `len` raw bytes at absolute `byte_addr` from the right backing store.
/// HBM is shared; LX routes via `lx_core_id` (None => executing core's own LX).
/// Gather-decode: one f32 per `offsets` entry, decoded directly at that element
/// offset in `raw` — WITHOUT first decoding the whole `[0, max_offset]` span.
/// Strided slow-path loads select far fewer elements than their span covers
/// (measured ~18×), so decoding only the selected offsets avoids that waste and
/// the span-sized intermediate `Vec`. Contiguous offsets (`0..n`) fall through
/// to the SIMD batch [`decode`]. Out-of-range / short bytes decode as `0.0`,
/// matching [`decode`]'s zero-pad.
fn decode_gather(raw: &[u8], offsets: &[i64], dtype: DType) -> Vec<f32> {
    // Contiguous prefix → batch decode hits the f16 SIMD fast path, no gather.
    if offsets.iter().enumerate().all(|(i, &o)| o == i as i64) {
        return decode(raw, dtype, offsets.len());
    }
    // Hoist the dtype dispatch OUT of the per-element loop: `dtype` is constant
    // across the gather, so each tight loop below decodes one fixed element type
    // (the f16 KV-gather is a u16->f32 loop with no per-element match). The
    // `raw.get(..)` OOB-as-0.0 / short-pad semantics are preserved exactly.
    let mut out = Vec::with_capacity(offsets.len());
    match dtype {
        DType::F16 => {
            for &o in offsets {
                let off = o as usize * 2;
                let v = match raw.get(off..off + 2) {
                    Some(c) => crate::codec::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])),
                    None => 0.0,
                };
                out.push(v);
            }
        }
        DType::F32 => {
            for &o in offsets {
                let off = o as usize * 4;
                let v = match raw.get(off..off + 4) {
                    Some(c) => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    None => 0.0,
                };
                out.push(v);
            }
        }
        DType::I32 => {
            for &o in offsets {
                let off = o as usize * 4;
                let v = match raw.get(off..off + 4) {
                    Some(c) => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32,
                    None => 0.0,
                };
                out.push(v);
            }
        }
        DType::I64 => {
            for &o in offsets {
                let off = o as usize * 8;
                let v = match raw.get(off..off + 8) {
                    Some(c) => {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    }
                    None => 0.0,
                };
                out.push(v);
            }
        }
        DType::Bool => {
            for &o in offsets {
                let off = o;
                let v = match raw.get(off as usize..off as usize + 1) {
                    Some(c) => (c[0] != 0) as i32 as f32,
                    None => 0.0,
                };
                out.push(v);
            }
        }
    }
    out
}

fn read_raw(ctx: &CoreContext, space: MemorySpace, byte_addr: i64, len: usize) -> Vec<u8> {
    match space {
        MemorySpace::Hbm => ctx.hbm.borrow().read_bytes(byte_addr, len),
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));

            lx.borrow().read_bytes(byte_addr, len)
        }
    }
}

/// Read `n` elements of `dtype` and decode to f32 directly from the backing
/// store — no intermediate byte `Vec`. Used by the contiguous load fast path.
fn read_decoded(
    ctx: &CoreContext,
    space: MemorySpace,
    byte_addr: i64,
    n: usize,
    dtype: DType,
) -> Vec<f32> {
    match space {
        MemorySpace::Hbm => ctx.hbm.borrow().read_decoded(byte_addr, n, dtype),
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));
            lx.borrow().read_decoded(byte_addr, n, dtype)
        }
    }
}

/// Decode `out.len()` elements at `byte_addr` directly INTO `out` (no Vec) — the
/// row-contiguous load decodes each strided run into its slice of the result.
fn read_decoded_into(
    ctx: &CoreContext,
    space: MemorySpace,
    byte_addr: i64,
    out: &mut [f32],
    dtype: DType,
) {
    match space {
        MemorySpace::Hbm => ctx.hbm.borrow().read_decoded_into(byte_addr, out, dtype),
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));
            lx.borrow().read_decoded_into(byte_addr, out, dtype);
        }
    }
}

// NB: the `let bytes = ...; bytes` form keeps the LX `RefCell` borrow scoped to
// the read, dropping it before the value is returned.

/// Write raw bytes at absolute `byte_addr` to the right backing store.
fn write_raw(ctx: &mut CoreContext, space: MemorySpace, byte_addr: i64, data: &[u8]) {
    match space {
        MemorySpace::Hbm => ctx.hbm.borrow_mut().write_bytes(byte_addr, data),
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));
            lx.borrow_mut().write_bytes(byte_addr, data);
        }
    }
}

/// Port of `MemoryOps._write_to_lx`: reserve a stick-aligned span in the
/// executing core's LX and write the decoded tile there. All loaded tiles land
/// in LX regardless of source memory space. The bytes written are the dtype's
/// native encoding so a subsequent LX-sourced load round-trips exactly.
fn write_to_lx(ctx: &mut CoreContext, data: &[f32], dtype: DType) {
    // A loaded tile's data reaches its consumers through the returned SSA `Tile`
    // (compute ops and the GPU/AMX GEMM operand resolver all read `ctx.get_value`
    // -> `Tile::as_f32`, never the physical LX bytes), and the bump address written
    // here is discarded — no SSA value can name it. So these bytes are never read
    // back: VERIFIED by making the whole write a no-op and finding the full suite +
    // e2e golden bit-identical. Drop the dead per-load f32->dtype encode + byte
    // write (which dominated the load hot path), keeping only the stick-aligned LX
    // residence watermark advance (cheap; preserves the simulated LX occupancy in
    // case the latency model ever reads it).
    let size = (data.len() * dtype.bytes_per_elem()) as i64;
    let lx = ctx.get_lx(None);
    let lxm = lx.borrow_mut();
    lxm.next_ptr = (lxm.next_ptr + size + STICK_BYTES - 1) & !(STICK_BYTES - 1);
}

// ===========================================================================
// Offset linearization + contiguity (port of _flat_memory_offsets / _is_contiguous)
// ===========================================================================

/// Port of `MemoryOps._is_contiguous`: row-major C-order check.
pub fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    let mut expected: i64 = 1;
    for (&dim, &stride) in shape.iter().rev().zip(strides.iter().rev()) {
        // An axis of extent 0 or 1 is never stepped (its coordinate is always 0),
        // so it contributes nothing to any element offset and its stride is
        // irrelevant to contiguity. Skipping the check here routes the common
        // `[1, w]` row access (e.g. a decode row carved from a `[cap, w]` tensor,
        // stride `[cap_w, 1]`) to the contiguous fast path instead of the slow
        // gather — it IS a contiguous `w`-element read.
        if dim <= 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        expected *= dim as i64;
    }
    true
}

/// Port of `MemoryOps._flat_memory_offsets`. Linearizes N-d coords (or the full
/// shape when `coords` is None) into flat element offsets, and counts distinct
/// HBM sticks when `stick_bytes` is set.
fn flat_memory_offsets(
    base_ptr: i64,
    shape: &[usize],
    strides: &[i64],
    dtype: DType,
    coords: Option<&[Vec<i64>]>,
    stick_bytes: Option<i64>,
) -> (Vec<i64>, Option<usize>) {
    let bpe = dtype.bytes_per_elem() as i64;
    let mut sticks: Option<std::collections::HashSet<i64>> =
        stick_bytes.map(|_| std::collections::HashSet::new());

    match coords {
        Some(cs) => {
            // Pre-size to the coord count; specialize the small-rank `coord·strides`
            // dot (the KV reads are 2-D) to direct multiply-adds, skipping the
            // iterator-zip-sum.
            let mut offsets = Vec::with_capacity(cs.len());
            match strides.len() {
                2 => {
                    let (s0, s1) = (strides[0], strides[1]);
                    for c in cs {
                        let o = c[0] * s0 + c[1] * s1;
                        offsets.push(o);
                        if let (Some(set), Some(sb)) = (sticks.as_mut(), stick_bytes) {
                            set.insert((base_ptr + o * bpe) / sb);
                        }
                    }
                }
                3 => {
                    let (s0, s1, s2) = (strides[0], strides[1], strides[2]);
                    for c in cs {
                        let o = c[0] * s0 + c[1] * s1 + c[2] * s2;
                        offsets.push(o);
                        if let (Some(set), Some(sb)) = (sticks.as_mut(), stick_bytes) {
                            set.insert((base_ptr + o * bpe) / sb);
                        }
                    }
                }
                _ => {
                    for c in cs {
                        let o: i64 = c.iter().zip(strides).map(|(&c, &s)| c * s).sum();
                        offsets.push(o);
                        if let (Some(set), Some(sb)) = (sticks.as_mut(), stick_bytes) {
                            set.insert((base_ptr + o * bpe) / sb);
                        }
                    }
                }
            }
            (offsets, sticks.map(|s| s.len()))
        }
        None => {
            // np.ndindex(*shape): row-major, rightmost dim innermost. Pre-size to the
            // exact element count and walk an INCREMENTAL ODOMETER — maintain the
            // running offset by +stride[d] per innermost step and the carry fixups on
            // wrap — instead of a per-element `coord·strides` dot. Empty/zero-extent
            // shapes match `ndindex` (which emits nothing if any dim is 0, and a single
            // scalar `0` for the rank-0 case).
            let n: usize = shape.iter().product();
            let mut offsets = Vec::with_capacity(n);
            if shape.is_empty() {
                // Rank-0: a single element at offset 0 (matches `ndindex(&[])`).
                offsets.push(0);
                if let (Some(set), Some(sb)) = (sticks.as_mut(), stick_bytes) {
                    set.insert(base_ptr / sb);
                }
                return (offsets, sticks.map(|s| s.len()));
            }
            if n == 0 {
                return (offsets, sticks.map(|s| s.len()));
            }
            let nd = shape.len();
            let mut idx = vec![0i64; nd];
            let mut off: i64 = 0;
            loop {
                offsets.push(off);
                if let (Some(set), Some(sb)) = (sticks.as_mut(), stick_bytes) {
                    set.insert((base_ptr + off * bpe) / sb);
                }
                // Advance the rightmost (innermost) axis; carry left, subtracting the
                // wrapped axis's full span and adding the next axis's stride.
                let mut d = nd;
                loop {
                    if d == 0 {
                        return (offsets, sticks.map(|s| s.len()));
                    }
                    d -= 1;
                    idx[d] += 1;
                    off += strides[d];
                    if (idx[d] as usize) < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                    off -= strides[d] * shape[d] as i64;
                }
            }
        }
    }
}

/// Iterate the cartesian index space of `shape` in row-major order. Retained as
/// the reference odometer the `flat_memory_offsets` incremental walk is checked
/// against (its only callers are the unit tests below).
#[cfg(test)]
fn ndindex(shape: &[usize], f: &mut impl FnMut(&[i64])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.contains(&0) {
        return;
    }
    let mut idx = vec![0i64; shape.len()];
    loop {
        f(&idx);
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

// ===========================================================================
// dtype <-> bytes (decode to f32, encode from f32)
// ===========================================================================

/// Decode `n` elements of `dtype` from the front of `raw` into f32 values.
/// Missing/short bytes decode as zero (matches the simulator's zero-padding).
///
/// Delegates to the single `codec` implementation (which carries the SIMD/
/// table-backed f16 fast path) — only the operand order differs locally.
#[inline]
fn decode(raw: &[u8], dtype: DType, n: usize) -> Vec<f32> {
    crate::codec::decode(raw, n, dtype)
}

/// Encode f32 element values into `dtype`'s native little-endian byte layout.
/// Delegates to the single `codec` implementation.
#[inline]
fn encode(data: &[f32], dtype: DType) -> Vec<u8> {
    crate::codec::encode(data, dtype)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::ir::Attr;
    use crate::memref::{MemRef, MemorySpace};
    use std::rc::Rc;

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    // ---- pure unit tests for the byte codecs ----

    #[test]
    fn f16_roundtrips_exact_representables() {
        for &v in &[0.0f32, 1.0, -2.0, 0.5, 1024.0, -0.25, 3.5] {
            let h = crate::codec::f32_to_f16_bits(v);
            assert_eq!(
                crate::codec::f16_bits_to_f32(h),
                v,
                "f16 round trip for {v}"
            );
        }
    }

    #[test]
    fn encode_decode_roundtrip_per_dtype() {
        for dt in [DType::F32, DType::I32, DType::I64, DType::F16] {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let raw = encode(&data, dt);
            assert_eq!(raw.len(), 4 * dt.bytes_per_elem());
            let back = decode(&raw, dt, 4);
            assert_eq!(back, data, "round trip dtype {dt}");
        }
    }

    #[test]
    fn decode_zero_pads_short_input() {
        // Only 4 bytes available but 2 f32 elements requested -> second is 0.
        let raw = 7.0f32.to_le_bytes().to_vec();
        assert_eq!(decode(&raw, DType::F32, 2), vec![7.0, 0.0]);
    }

    #[test]
    fn contiguity_check() {
        assert!(is_contiguous(&[4, 4], &[4, 1]));
        assert!(is_contiguous(&[4], &[1]));
        assert!(!is_contiguous(&[4], &[4])); // strided column
        assert!(!is_contiguous(&[2, 3], &[1, 2]));
        // Extent-1 axes are never stepped, so their stride is irrelevant: a `[1, w]`
        // row carved from a wider `[cap, w]` tensor (stride `[cap*w, 1]` or any
        // value) is a contiguous `w`-element read.
        assert!(is_contiguous(&[1, 64], &[512, 1]));
        assert!(is_contiguous(&[1, 64], &[999, 1]));
        assert!(is_contiguous(&[64, 1], &[1, 7])); // extent-1 trailing axis
        assert!(is_contiguous(&[2, 1, 3], &[3, 99, 1]));
        // ...but a genuine multi-row stride is still non-contiguous.
        assert!(!is_contiguous(&[64, 64], &[512, 1]));
        assert!(!is_contiguous(&[2, 64], &[512, 1]));
    }

    #[test]
    fn flat_offsets_full_shape_rowmajor() {
        let (offsets, sticks) = flat_memory_offsets(0, &[2, 2], &[2, 1], DType::F32, None, None);
        assert_eq!(offsets, vec![0, 1, 2, 3]);
        assert_eq!(sticks, None);
    }

    #[test]
    fn flat_offsets_strided_column_counts_sticks() {
        // f16 column of a 4x4 matrix: base byte 4, strides [4], shape [4].
        // offsets 0,4,8,12 -> byte addrs 4,12,20,28 all in stick 0.
        let (offsets, sticks) =
            flat_memory_offsets(4, &[4], &[4], DType::F16, None, Some(STICK_BYTES));
        assert_eq!(offsets, vec![0, 4, 8, 12]);
        assert_eq!(sticks, Some(1));
    }

    // ---- helpers to build views ----

    /// Build an HBM MemRef whose data lives at HBM stick `stick` (byte address
    /// `stick * STICK_BYTES`). Since `base_ptr` is now an ELEMENT index (RFC
    /// #110), convert: `base_ptr = stick * STICK_BYTES / bytes_per_elem` so
    /// `byte_address() == stick * STICK_BYTES`.
    fn hbm_memref(stick: i64, shape: Vec<usize>, strides: Vec<i64>, dtype: DType) -> MemRef {
        MemRef {
            base_ptr: stick * STICK_BYTES / dtype.bytes_per_elem() as i64,
            shape,
            strides,
            space: MemorySpace::Hbm,
            dtype,
            coordinate_set: None,
        }
    }

    // ---- load fast path ----

    #[test]
    fn load_contiguous_hbm_decodes_f32() {
        let mut ctx = single_core_context();
        // Allocate a 4-element f32 region in HBM.
        let stick = ctx.hbm.borrow_mut().allocate(4 * 4);
        let byte_addr = stick * STICK_BYTES;
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm.borrow_mut().write_bytes(byte_addr, &payload);

        let m = hbm_memref(stick, vec![4], vec![1], DType::F32);
        let tr = m.to_tile_ref();
        let tile = load_data(&mut ctx, &tr, None, None).unwrap();
        assert_eq!(tile.as_f32().to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tile.shape, vec![4]);
        // 16 bytes from a stick boundary -> exactly 1 stick.
        assert_eq!(tile.unique_sticks, Some(1));
    }

    #[test]
    fn load_lx_decodes_f16() {
        let mut ctx = single_core_context();
        let raw: Vec<u8> = [1.0f32, 2.0, 4.0]
            .iter()
            .flat_map(|x| crate::codec::f32_to_f16_bits(*x).to_le_bytes())
            .collect();
        ctx.lx.borrow_mut().write_bytes(0, &raw);

        let m = MemRef {
            base_ptr: 0,
            shape: vec![3],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F16,
            coordinate_set: None,
        };
        let tile = load_data(&mut ctx, &m.to_tile_ref(), None, None).unwrap();
        assert_eq!(tile.as_f32().to_vec(), vec![1.0, 2.0, 4.0]);
        assert_eq!(tile.unique_sticks, None); // LX: no sticks
    }

    // ---- load slow path (coords) ----

    #[test]
    fn load_with_coords_gathers() {
        let mut ctx = single_core_context();
        // 4x4 f32 matrix values 0..15 in HBM.
        let stick = ctx.hbm.borrow_mut().allocate(16 * 4);
        let byte_addr = stick * STICK_BYTES;
        let payload: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
        ctx.hbm.borrow_mut().write_bytes(byte_addr, &payload);

        let m = hbm_memref(stick, vec![4, 4], vec![4, 1], DType::F32);
        let tr = m.to_tile_ref();
        // Gather the diagonal: (0,0),(1,1),(2,2),(3,3) -> 0,5,10,15.
        let coords = vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]];
        let tile = load_data(&mut ctx, &tr, Some(&coords), Some(vec![4])).unwrap();
        assert_eq!(tile.as_f32().to_vec(), vec![0.0, 5.0, 10.0, 15.0]);
    }

    // ---- store round trips ----

    #[test]
    fn store_contiguous_hbm_roundtrips() {
        let mut ctx = single_core_context();
        let stick = ctx.hbm.borrow_mut().allocate(4 * 4);
        let m = hbm_memref(stick, vec![4], vec![1], DType::F32);
        let tr = m.to_tile_ref();

        let tile = Tile::compute(vec![10.0, 20.0, 30.0, 40.0], DType::F32, vec![4]);
        let sticks = store_data(&mut ctx, &tile, &tr, None).unwrap();
        assert_eq!(sticks, 1);

        let back = load_data(&mut ctx, &tr, None, None).unwrap();
        assert_eq!(back.as_f32().to_vec(), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn store_lx_returns_zero_sticks() {
        let mut ctx = single_core_context();
        let m = MemRef {
            base_ptr: 256,
            shape: vec![3],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        let tr = m.to_tile_ref();
        let tile = Tile::compute(vec![1.0, 2.0, 3.0], DType::F32, vec![3]);
        let sticks = store_data(&mut ctx, &tile, &tr, None).unwrap();
        assert_eq!(sticks, 0);
        let back = load_data(&mut ctx, &tr, None, None).unwrap();
        assert_eq!(back.as_f32().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn store_with_coords_scatters_rmw() {
        let mut ctx = single_core_context();
        // Pre-fill a 4-element f32 LX region with zeros, then scatter into
        // offsets 0 and 2 via coords on a [2] logical tile with stride [2].
        // base_ptr is an element index (RFC #110): byte_address = 128*4 = 512.
        let m = MemRef {
            base_ptr: 128,
            shape: vec![2],
            strides: vec![2],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        ctx.lx.borrow_mut().write_bytes(512, &[0u8; 16]); // 4 f32 zeros
        let tr = m.to_tile_ref();
        let tile = Tile::compute(vec![7.0, 9.0], DType::F32, vec![2]);
        // coords (0) and (1) over strides [2] -> flat offsets 0 and 2.
        let coords = vec![vec![0], vec![1]];
        store_data(&mut ctx, &tile, &tr, Some(&coords)).unwrap();

        let raw = ctx.lx.borrow().read_bytes(512, 16);
        let vals = decode(&raw, DType::F32, 4);
        assert_eq!(vals, vec![7.0, 0.0, 9.0, 0.0]);
    }

    // ---- end-to-end: load -> addf -> store through the dispatch table ----

    fn ident1() -> Attr {
        Attr::AffineMap(AffineMap::identity(1))
    }

    #[test]
    fn vector_add_load_addf_store_end_to_end() {
        let mut ctx = single_core_context();

        // Two input vectors of length 4 in HBM, plus an output region.
        let n = 4usize;
        let a_stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let b_stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let out_stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let a_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = [10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm
            .borrow_mut()
            .write_bytes(a_stick * STICK_BYTES, &a_bytes);
        ctx.hbm
            .borrow_mut()
            .write_bytes(b_stick * STICK_BYTES, &b_bytes);

        // Bind the three base pointers as ELEMENT indices (RFC #110): the byte
        // address is base_ptr*4 (f32), so elem = stick*STICK_BYTES/4 lands the
        // view at the seeded stick.
        let elem = |stick: i64| Value::Index(stick * STICK_BYTES / 4);
        ctx.set_value("%pa", elem(a_stick));
        ctx.set_value("%pb", elem(b_stick));
        ctx.set_value("%pout", elem(out_stick));
        ctx.set_value("%i", Value::Index(0));

        let view = |res: &str, ptr: &str| {
            Operation::new(Some(res), "ktdp.construct_memory_view", &[ptr])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr("memory_space", Attr::Str("HBM".into()))
                .with_attr("dtype", Attr::Str("f32".into()))
        };
        let access = |res: &str, view: &str| {
            Operation::new(Some(res), "ktdp.construct_access_tile", &[view, "%i"])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("base_map", ident1())
        };

        let ops = vec![
            view("%va", "%pa"),
            view("%vb", "%pb"),
            view("%vout", "%pout"),
            access("%aa", "%va"),
            access("%ab", "%vb"),
            access("%aout", "%vout"),
            Operation::new(Some("%ta"), "ktdp.load", &["%aa"]),
            Operation::new(Some("%tb"), "ktdp.load", &["%ab"]),
            Operation::new(Some("%tc"), "arith.addf", &["%ta", "%tb"]),
            Operation::new(None, "ktdp.store", &["%tc", "%aout"]),
        ];
        run(&ops, &mut ctx).unwrap();

        // The stored output region should hold the element-wise sum.
        let raw = ctx.hbm.borrow().read_bytes(out_stick * STICK_BYTES, n * 4);
        let vals = decode(&raw, DType::F32, n);
        assert_eq!(vals, vec![11.0, 22.0, 33.0, 44.0]);
    }

    // ---- end-to-end with a coordinate_set on the access tile ----

    #[test]
    fn load_via_coordinate_set_through_handler() {
        let mut ctx = single_core_context();
        let n = 4usize;
        let stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let payload: Vec<u8> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm
            .borrow_mut()
            .write_bytes(stick * STICK_BYTES, &payload);

        // base_ptr is an element index (RFC #110): elem = stick*STICK_BYTES/4 (f32).
        ctx.set_value("%p", Value::Index(stick * STICK_BYTES / 4));
        ctx.set_value("%i", Value::Index(0));

        // coordinate_set { d0 : d0 >= 0 } over shape [4] selects all coords in
        // order -> behaves like a full contiguous gather.
        let css = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        };

        let ops = vec![
            Operation::new(Some("%v"), "ktdp.construct_memory_view", &["%p"])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr("memory_space", Attr::Str("HBM".into()))
                .with_attr("dtype", Attr::Str("f32".into())),
            Operation::new(Some("%a"), "ktdp.construct_access_tile", &["%v", "%i"])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("base_map", ident1())
                .with_attr("coordinate_set", Attr::AffineSet(css)),
            Operation::new(Some("%t"), "ktdp.load", &["%a"]),
        ];
        run(&ops, &mut ctx).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::Tile(t) => assert_eq!(t.as_f32().to_vec(), vec![5.0, 6.0, 7.0, 8.0]),
            other => panic!("expected Tile, got {other:?}"),
        }
    }

    #[test]
    fn store_rejects_non_tile_first_operand() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Index(3));
        ctx.set_value("%y", Value::Index(4));
        let op = Operation::new(None, "ktdp.store", &["%x", "%y"]);
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("Tile"), "unexpected error: {err}");
    }

    #[test]
    fn ndindex_scalar_shape_emits_one_point() {
        let mut count = 0;
        ndindex(&[], &mut |_| count += 1);
        assert_eq!(count, 1);
        // Empty axis -> no points.
        let mut count2 = 0;
        ndindex(&[0, 3], &mut |_| count2 += 1);
        assert_eq!(count2, 0);
    }

    // =======================================================================
    // Distributed + indirect path tests
    // =======================================================================

    use crate::memref::{
        CoordinateSet, DimSubscript, DistributedMemRef, IndirectAccessTile, ParentRef,
    };

    /// Inclusive box `[lo, hi]` as an `AffineSet`: for each axis i,
    /// `d_i - lo_i >= 0` and `hi_i - d_i >= 0`.
    fn box_affine(lo: &[i64], hi: &[i64]) -> AffineSet {
        let mut constraints = Vec::new();
        for i in 0..lo.len() {
            constraints.push(Constraint {
                expr: AffineExpr::Sub(
                    Rc::new(AffineExpr::Dim(i)),
                    Rc::new(AffineExpr::Const(lo[i])),
                ),
                kind: ConstraintKind::GreaterEq,
            });
            constraints.push(Constraint {
                expr: AffineExpr::Sub(
                    Rc::new(AffineExpr::Const(hi[i])),
                    Rc::new(AffineExpr::Dim(i)),
                ),
                kind: ConstraintKind::GreaterEq,
            });
        }
        AffineSet {
            num_dims: lo.len(),
            num_syms: 0,
            constraints,
        }
    }

    // ---- lower_to_box ----

    #[test]
    fn lower_to_box_is_inclusive() {
        // affine [2,5] on one axis -> inclusive BoxSet lo=2 hi=5.
        let b = lower_to_box(&box_affine(&[2], &[5])).expect("lowerable");
        assert_eq!(b.lo, vec![2]);
        assert_eq!(b.hi, vec![5]);
        // non-axis-aligned -> None.
        let diag = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Add(Rc::new(AffineExpr::Dim(0)), Rc::new(AffineExpr::Dim(1))),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        assert!(lower_to_box(&diag).is_none());
    }

    // ---- distributed_tile_access: 2-partition routing ----

    /// Two HBM partitions of a 1-D length-8 f32 tensor: B_0 owns coords [0,3],
    /// B_1 owns [4,7]. Each partition's data lives at its own stick.
    fn two_partition_dist(ctx: &mut CoreContext) -> (DistributedMemRef, i64, i64) {
        let s0 = ctx.hbm.borrow_mut().allocate(4 * 4);
        let s1 = ctx.hbm.borrow_mut().allocate(4 * 4);
        // Partition 0 holds global coords 0..3 -> values 0,1,2,3.
        let p0: Vec<u8> = [0.0f32, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        // Partition 1 holds global coords 4..7 -> values 40,50,60,70.
        let p1: Vec<u8> = [40.0f32, 50.0, 60.0, 70.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm.borrow_mut().write_bytes(s0 * STICK_BYTES, &p0);
        ctx.hbm.borrow_mut().write_bytes(s1 * STICK_BYTES, &p1);

        // base_ptr is an element index (RFC #110): elem = stick*STICK_BYTES/4 (f32)
        // lands byte_address back on the seeded stick.
        let mk = |stick: i64, lo: i64, hi: i64| MemRef {
            base_ptr: stick * STICK_BYTES / 4,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Hbm,
            dtype: DType::F32,
            coordinate_set: Some(box_affine(&[lo], &[hi])),
        };
        let dist =
            DistributedMemRef::new(vec![mk(s0, 0, 3), mk(s1, 4, 7)], vec![8], DType::F32).unwrap();
        (dist, s0, s1)
    }

    #[test]
    fn distributed_tile_access_survivors_box_fastpath() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        // Access the full [0,8) window: x=0, access_shape=8, both partitions survive.
        let dtr =
            distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        assert_eq!(dtr.partitions.len(), 2);
        assert_eq!(dtr.global_base, Some(vec![0]));
        // Each survivor carries a Box coordinate_set and partition origin.
        match &dtr.partitions[0].coordinate_set {
            Some(CoordinateSet::Box(b)) => {
                assert_eq!(b.lo, vec![0]);
                assert_eq!(b.hi, vec![3]);
            }
            other => panic!("expected Box C_0, got {other:?}"),
        }
        assert_eq!(dtr.partitions[0].partition_origin, Some(vec![0]));
        assert_eq!(dtr.partitions[1].partition_origin, Some(vec![4]));
    }

    #[test]
    fn distributed_tile_access_partial_window_drops_partition() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        // Access window [0,3) only -> only partition 0 survives.
        let dtr =
            distributed_tile_access(&dist, &[3], &AffineMap::identity(1), &[0], None).unwrap();
        assert_eq!(dtr.partitions.len(), 1);
        match &dtr.partitions[0].coordinate_set {
            Some(CoordinateSet::Box(b)) => {
                assert_eq!(b.lo, vec![0]);
                assert_eq!(b.hi, vec![2]); // C_0 = [0,3] ∩ [0,2] = [0,2]
            }
            other => panic!("expected Box, got {other:?}"),
        }
    }

    #[test]
    fn distributed_tile_access_no_coverage_errors() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        // Window starting at global 100 covers no partition.
        let err = distributed_tile_access(&dist, &[2], &AffineMap::identity(1), &[100], None)
            .unwrap_err();
        assert!(err.contains("no partition"), "unexpected: {err}");
    }

    // ---- distributed_load: 2-partition gather ----

    #[test]
    fn distributed_load_gathers_across_two_partitions() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        let dtr =
            distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        let tile = distributed_load(&mut ctx, &dtr, Some(vec![8])).unwrap();
        // Concatenation of both partitions in global-coord order.
        assert_eq!(
            tile.as_f32().to_vec(),
            vec![0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0]
        );
        assert_eq!(tile.shape, vec![8]);
        // Both partitions are HBM -> unique_sticks aggregated (1 each).
        assert_eq!(tile.unique_sticks, Some(2));
    }

    #[test]
    fn distributed_store_then_load_roundtrips_two_partitions() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        let dtr =
            distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();

        // Scatter a fresh 8-vector across both partitions.
        let tile = Tile::compute(
            vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            DType::F32,
            vec![8],
        );
        let sticks = distributed_store(&mut ctx, &tile, &dtr).unwrap();
        assert_eq!(sticks, 2); // one HBM stick per partition

        // Re-resolve (survivor TileRefs are consumed) and read back.
        let dtr2 =
            distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        let back = distributed_load(&mut ctx, &dtr2, Some(vec![8])).unwrap();
        assert_eq!(
            back.as_f32().to_vec(),
            vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        );
    }

    // ---- distributed end-to-end through the ktdp.load dispatch handler ----

    #[test]
    fn distributed_load_through_access_tile_parent() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        let dtr =
            distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        // Wrap the DistributedTileRef in an AccessTile and load via the handler.
        let access = AccessTile {
            parent_ref: ParentRef::Dist(dtr),
            shape: vec![8],
            base_map: AffineMap::identity(1),
            coordinate_set: None,
            coordinate_order: None,
        };
        ctx.set_value("%a", Value::AccessTile(access));
        let op = Operation::new(Some("%t"), "ktdp.load", &["%a"]);
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::Tile(t) => assert_eq!(
                t.as_f32().to_vec(),
                vec![0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0]
            ),
            other => panic!("expected Tile, got {other:?}"),
        }
    }

    // ---- indirect gather ----

    /// 1-D vss over a single intermediate var (length 4), trivially satisfiable.
    fn vss_1d() -> AffineSet {
        AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        }
    }

    #[test]
    fn indirect_gather_reads_through_index_view() {
        let mut ctx = single_core_context();
        // Parent X: 8 f32 values in LX at byte 0 -> 10,11,...,17.
        let x_data: Vec<u8> = (0..8)
            .flat_map(|i| (10.0f32 + i as f32).to_le_bytes())
            .collect();
        ctx.lx.borrow_mut().write_bytes(0, &x_data);
        // Index view IDX: i32 values [3, 0, 5, 1] at byte 256.
        let idx_data: Vec<u8> = [3i32, 0, 5, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ctx.lx.borrow_mut().write_bytes(256, &idx_data);

        let x_view = MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        let idx_view = MemRef {
            // base_ptr is an element index (RFC #110): byte 64*4 = 256 (i32).
            base_ptr: 64,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::I32,
            coordinate_set: None,
        };

        // X[ ind(IDX[m]) ] over intermediate var m in [0,4): gather X at the
        // indices held in IDX -> X[3], X[0], X[5], X[1] = 13, 10, 15, 11.
        let iat = IndirectAccessTile {
            parent_ref: x_view,
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Indirect {
                view: 0,
                idx_exprs: vec![],
            }],
            index_views: vec![idx_view],
            variables_space_set: vss_1d(),
            variables_space_order: None,
            extra: std::collections::HashMap::new(),
        };

        let tile = indirect_load(&mut ctx, &iat, None).unwrap();
        assert_eq!(tile.as_f32().to_vec(), vec![13.0, 10.0, 15.0, 11.0]);
        assert_eq!(tile.shape, vec![4]);
        // LX index view -> no index sticks.
        assert_eq!(tile.index_unique_sticks, Some(0));
    }

    #[test]
    fn indirect_gather_negative_index_rejected() {
        let mut ctx = single_core_context();
        let x_data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        ctx.lx.borrow_mut().write_bytes(0, &x_data);
        // IDX holds a negative index -> must be rejected (no NumPy wrap).
        let idx_data: Vec<u8> = [-1i32, 0, 1, 2]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ctx.lx.borrow_mut().write_bytes(256, &idx_data);

        let x_view = MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        let idx_view = MemRef {
            // base_ptr is an element index (RFC #110): byte 64*4 = 256 (i32).
            base_ptr: 64,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::I32,
            coordinate_set: None,
        };
        let iat = IndirectAccessTile {
            parent_ref: x_view,
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Indirect {
                view: 0,
                idx_exprs: vec![],
            }],
            index_views: vec![idx_view],
            variables_space_set: vss_1d(),
            variables_space_order: None,
            extra: std::collections::HashMap::new(),
        };
        let err = indirect_load(&mut ctx, &iat, None).unwrap_err();
        assert!(err.contains("negative"), "unexpected: {err}");
    }

    #[test]
    fn indirect_scatter_then_direct_load_roundtrips() {
        let mut ctx = single_core_context();
        // Destination X: 8 f32 zeros in HBM.
        let xs = ctx.hbm.borrow_mut().allocate(8 * 4);
        ctx.hbm
            .borrow_mut()
            .write_bytes(xs * STICK_BYTES, &[0u8; 32]);
        // IDX in LX: scatter positions [2, 5, 0, 7].
        let idx_data: Vec<u8> = [2i32, 5, 0, 7]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        ctx.lx.borrow_mut().write_bytes(512, &idx_data);

        let x_view = MemRef {
            // base_ptr is an element index (RFC #110): elem = xs*STICK_BYTES/4 (f32)
            // lands byte_address back on the seeded stick xs.
            base_ptr: xs * STICK_BYTES / 4,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Hbm,
            dtype: DType::F32,
            coordinate_set: None,
        };
        let idx_view = MemRef {
            // base_ptr is an element index (RFC #110): byte 128*4 = 512 (i32).
            base_ptr: 128,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::I32,
            coordinate_set: None,
        };
        let iat = IndirectAccessTile {
            parent_ref: x_view.clone(),
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Indirect {
                view: 0,
                idx_exprs: vec![],
            }],
            index_views: vec![idx_view],
            variables_space_set: vss_1d(),
            variables_space_order: None,
            extra: std::collections::HashMap::new(),
        };

        // Scatter [100,200,300,400] to X[2],X[5],X[0],X[7].
        let src = Tile::compute(vec![100.0, 200.0, 300.0, 400.0], DType::F32, vec![4]);
        let sticks = indirect_store(&mut ctx, &src, &iat).unwrap();
        // Parent is HBM (one stick), idx view is LX (0) -> at least 1.
        assert!(sticks >= 1);

        // Direct full load of X confirms the scatter.
        let back = load_data(&mut ctx, &x_view.to_tile_ref(), None, None).unwrap();
        assert_eq!(
            back.as_f32().to_vec(),
            vec![300.0, 0.0, 100.0, 0.0, 0.0, 200.0, 0.0, 400.0]
        );
    }

    #[test]
    fn indirect_load_rejects_non_permutation_vso() {
        let mut ctx = single_core_context();
        let x_view = MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        // vso (d0) -> (2*d0) is a scaling, not a permutation.
        let bad = AffineMap {
            num_dims: 1,
            num_syms: 0,
            exprs: vec![AffineExpr::Mul(
                Rc::new(AffineExpr::Const(2)),
                Rc::new(AffineExpr::Dim(0)),
            )],
        };
        let iat = IndirectAccessTile {
            parent_ref: x_view,
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Direct { var_index: 0 }],
            index_views: vec![],
            variables_space_set: vss_1d(),
            variables_space_order: Some(bad),
            extra: std::collections::HashMap::new(),
        };
        let err = indirect_load(&mut ctx, &iat, None).unwrap_err();
        assert!(err.contains("permut"), "unexpected: {err}");
    }
}
