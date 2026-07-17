// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Memory hierarchy — port of `ktir_emulator/memory.py`.
//!
//! Storage model: Python keys a dict by byte address -> ndarray. Here each
//! allocation is a contiguous byte buffer keyed by its base byte address; reads
//! interpret bytes per the requested dtype at the load/store boundary (the
//! load/store data path itself is an implement-phase fill in the ktdp dialect).
//! Cross-core sharing uses `Rc<RefCell<>>` to mirror Python reference semantics
//! — the scheduler is single-threaded/cooperative, so no `Arc`/`Mutex` needed.

use crate::dtypes::DType;
use crate::fxhash::FxHashMap;
use std::rc::Rc;

/// An interior-mutability cell with the same `borrow`/`borrow_mut` surface as
/// `RefCell` but no runtime borrow flag, so the cores of a comm-free multi-core
/// grid can access DISJOINT regions of one shared `HBMSimulator` /
/// `InternTable` / `LXScratchpad` concurrently — each core writes disjoint output
/// columns and only READS the shared weights/inputs — without the flag racing.
///
/// Soundness rests on the access pattern, not the type: the single-core path is
/// double-borrow-free (it ran correctly under `RefCell`), and the multi-core path
/// (see `execute_with_communication`) touches disjoint memory with a pre-warmed
/// intern table (no concurrent insert). It asserts `Sync` unconditionally, so it
/// is NOT a general-purpose cell — only use it where that contract holds.
#[derive(Debug)]
pub struct UnsafeShared<T>(std::cell::UnsafeCell<T>);
// SAFETY: callers guarantee disjoint/read-only concurrent access; see the type doc.
unsafe impl<T> Sync for UnsafeShared<T> {}
impl<T> UnsafeShared<T> {
    pub fn new(v: T) -> Self {
        Self(std::cell::UnsafeCell::new(v))
    }
    // `borrow`/`borrow_mut` intentionally mirror `RefCell`'s names (drop-in surface).
    #[allow(clippy::should_implement_trait, clippy::mut_from_ref)]
    pub fn borrow(&self) -> &T {
        // SAFETY: see type doc — disjoint/read-only concurrent access by contract.
        unsafe { &*self.0.get() }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn borrow_mut(&self) -> &mut T {
        // SAFETY: see type doc.
        unsafe { &mut *self.0.get() }
    }
}

/// HBM "stick" (cache block) size in bytes. Defined in `ktir-core` (memref byte
/// addressing needs it); re-exported here so `crate::memory::STICK_BYTES`
/// resolves unchanged.
pub use crate::memref::STICK_BYTES;

/// Shared, byte-addressed HBM with stick-granular addressing.
#[derive(Debug)]
pub struct HBMSimulator {
    pub size_bytes: i64,
    /// base byte address -> raw allocation bytes
    allocations: FxHashMap<i64, Vec<u8>>,
    /// next unallocated byte address (stick-aligned); starts at 0x10000
    pub next_ptr: i64,
}

impl Default for HBMSimulator {
    fn default() -> Self {
        HBMSimulator::new(128)
    }
}

impl HBMSimulator {
    pub fn new(size_gb: i64) -> Self {
        HBMSimulator {
            size_bytes: size_gb * 1024 * 1024 * 1024,
            allocations: FxHashMap::default(),
            next_ptr: 0x10000,
        }
    }

    /// Allocate `size` bytes, advance `next_ptr` to the next stick boundary,
    /// return the stick address (`ptr / STICK_BYTES`). Mirrors `allocate`.
    pub fn allocate(&mut self, size: i64) -> i64 {
        debug_assert_eq!(
            self.next_ptr % STICK_BYTES,
            0,
            "next_ptr must be stick-aligned"
        );
        let ptr = self.next_ptr;
        self.allocations
            .entry(ptr)
            .or_insert_with(|| vec![0u8; size.max(0) as usize]);
        let advanced = ptr + size;
        self.next_ptr = (advanced + STICK_BYTES - 1) & !(STICK_BYTES - 1);
        ptr / STICK_BYTES
    }

    /// Read `len` raw bytes starting at absolute `byte_addr`, zero-padding past
    /// the end of the containing allocation. Byte-level analogue of `_read_flat`.
    pub fn read_bytes(&self, byte_addr: i64, len: usize) -> Vec<u8> {
        read_bytes(&self.allocations, byte_addr, len)
    }

    /// Read `buf.len()` bytes at `byte_addr` into `buf` (zero-padding past the
    /// allocation end) WITHOUT allocating — the hot-path analogue of
    /// [`read_bytes`](Self::read_bytes). Used by the weight-cache fingerprint,
    /// which samples many small elements per pass and must not heap-allocate each.
    pub fn read_bytes_into(&self, byte_addr: i64, buf: &mut [u8]) {
        read_bytes_into(&self.allocations, byte_addr, buf);
    }

    /// Borrow the backing allocation containing `byte_addr` as `(buffer, offset)`,
    /// so a caller sampling MANY bytes from one region (the weight-cache
    /// fingerprint) resolves the allocation ONCE and then indexes — instead of a
    /// per-byte [`find_allocation`], which falls to an O(num-allocations) linear
    /// scan whenever the address isn't an exact allocation base (always true for an
    /// offset INTO a weight). `None` if `byte_addr` is in no allocation.
    pub fn allocation_at(&self, byte_addr: i64) -> Option<(&[u8], usize)> {
        let (base, _) = find_allocation(&self.allocations, byte_addr)?;
        let buf = &self.allocations[&base];
        Some((buf, (byte_addr - base) as usize))
    }

    /// Mutable analogue of [`allocation_at`](Self::allocation_at): the backing
    /// allocation containing `byte_addr` as `(buffer, offset)`, for callers that
    /// write a region IN PLACE (e.g. zeroing a stick) instead of allocating a
    /// fresh byte `Vec` and `write_bytes`-copying it. `None` if no allocation
    /// contains `byte_addr` (the caller falls back to `write_bytes`).
    pub fn allocation_at_mut(&mut self, byte_addr: i64) -> Option<(&mut [u8], usize)> {
        let (base, _) = find_allocation(&self.allocations, byte_addr)?;
        let off = (byte_addr - base) as usize;
        let buf = self.allocations.get_mut(&base)?;
        Some((buf, off))
    }

    /// Read `n` elements of `dtype` and decode to f32 in one pass, straight from
    /// the backing buffer — no intermediate byte `Vec`/memmove. `codec::decode`
    /// zero-pads when the read runs past the allocation. The contiguous-load fast
    /// path (the common case) uses this.
    pub fn read_decoded(&self, byte_addr: i64, n: usize, dtype: DType) -> Vec<f32> {
        read_decoded(&self.allocations, byte_addr, n, dtype)
    }

    /// Decode directly INTO `out` (no allocation) — the no-Vec analogue of
    /// [`read_decoded`](Self::read_decoded) for the row-contiguous load.
    pub fn read_decoded_into(&self, byte_addr: i64, out: &mut [f32], dtype: DType) {
        read_decoded_into(&self.allocations, byte_addr, out, dtype);
    }

    /// Write raw bytes at absolute `byte_addr`, growing/creating the allocation.
    pub fn write_bytes(&mut self, byte_addr: i64, data: &[u8]) {
        write_bytes(&mut self.allocations, byte_addr, data);
    }
}

/// LX live-set budget for fused-segment planning: 7/8 of the 2 MB per-core LX,
/// leaving headroom for a node's transient temporaries. Passed to
/// `ktir_optimizer::fusion::plan_segments_budgeted` so a fused segment's
/// co-resident `[m, *]` intermediates never overflow LX (without it, a whole
/// transformer MLP fuses into one segment and overflows at larger token counts —
/// llama m=32). 2 MB matches `LXScratchpad::new(.., 2)` below.
pub const LX_FUSION_BUDGET_BYTES: usize = (2 * 1024 * 1024) * 7 / 8;

/// The effective LX fusion budget — [`LX_FUSION_BUDGET_BYTES`] unless overridden
/// by `KTIR_LX_FUSION_BUDGET` (bytes). The override lets a host with a different
/// LX size tune fusion, and lets tests force splitting on small models.
pub fn lx_fusion_budget() -> usize {
    std::env::var("KTIR_LX_FUSION_BUDGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(LX_FUSION_BUDGET_BYTES)
}

/// Per-core local scratchpad. Plain byte addressing, no stick concept.
#[derive(Debug)]
pub struct LXScratchpad {
    pub capacity: i64,
    pub used: i64,
    pub core_id: usize,
    allocations: FxHashMap<i64, Vec<u8>>,
    pub next_ptr: i64,
}

impl LXScratchpad {
    pub fn new(core_id: usize, size_mb: i64) -> Self {
        LXScratchpad {
            capacity: size_mb * 1024 * 1024,
            used: 0,
            core_id,
            allocations: FxHashMap::default(),
            next_ptr: 0,
        }
    }

    pub fn read_bytes(&self, ptr: i64, len: usize) -> Vec<u8> {
        read_bytes(&self.allocations, ptr, len)
    }

    /// Decode `n` elements of `dtype` directly from the backing buffer (no
    /// intermediate byte `Vec`). See [`HBMSimulator::read_decoded`].
    pub fn read_decoded(&self, ptr: i64, n: usize, dtype: DType) -> Vec<f32> {
        read_decoded(&self.allocations, ptr, n, dtype)
    }

    /// Decode directly INTO `out` (no allocation). See [`HBMSimulator::read_decoded_into`].
    pub fn read_decoded_into(&self, ptr: i64, out: &mut [f32], dtype: DType) {
        read_decoded_into(&self.allocations, ptr, out, dtype);
    }

    pub fn write_bytes(&mut self, ptr: i64, data: &[u8]) {
        write_bytes(&mut self.allocations, ptr, data);
    }

    /// Reset for the next execution round. Mirrors `clear`.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.next_ptr = 0;
        self.used = 0;
    }
}

/// Shared HBM + one LX per core. Mirrors `SpyreMemoryHierarchy`.
pub struct SpyreMemoryHierarchy {
    pub num_cores: usize,
    pub hbm: Rc<UnsafeShared<HBMSimulator>>,
    pub lx_scratchpads: Vec<Rc<UnsafeShared<LXScratchpad>>>,
}

impl SpyreMemoryHierarchy {
    pub fn new(num_cores: usize) -> Self {
        let lx = (0..num_cores)
            .map(|c| Rc::new(UnsafeShared::new(LXScratchpad::new(c, 2))))
            .collect();
        SpyreMemoryHierarchy {
            num_cores,
            hbm: Rc::new(UnsafeShared::new(HBMSimulator::default())),
            lx_scratchpads: lx,
        }
    }

    /// Route to a core's LX. Mirrors `get_lx`.
    pub fn get_lx(&self, core_id: usize) -> Rc<UnsafeShared<LXScratchpad>> {
        Rc::clone(&self.lx_scratchpads[core_id])
    }
}

// --- shared byte-buffer helpers (port of _find_allocation/_read_flat/_write_flat) ---

/// Find the allocation containing `ptr`, returning `(base, len)`.
fn find_allocation(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64) -> Option<(i64, usize)> {
    if let Some(buf) = allocs.get(&ptr) {
        return Some((ptr, buf.len()));
    }
    allocs
        .iter()
        .find(|(base, buf)| ptr > **base && ptr < **base + buf.len() as i64)
        .map(|(&base, buf)| (base, buf.len()))
}

fn read_bytes(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    read_bytes_into(allocs, ptr, &mut out);
    out // zero-padded past allocation end (matches Python)
}

fn read_bytes_into(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64, buf: &mut [u8]) {
    buf.fill(0);
    if let Some((base, _)) = find_allocation(allocs, ptr) {
        let src = &allocs[&base];
        let off = (ptr - base) as usize;
        let avail = src.len().saturating_sub(off);
        let n = avail.min(buf.len());
        buf[..n].copy_from_slice(&src[off..off + n]);
    }
}

/// Read + decode in one pass: `codec::decode` reads the backing bytes directly
/// (zero-padding a short tail itself), so the contiguous load fast path skips
/// the intermediate byte `Vec` and its memmove.
fn read_decoded(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64, n: usize, dtype: DType) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    read_decoded_into(allocs, ptr, &mut out, dtype);
    out
}

/// Read + decode directly INTO `out` (no allocation) — the no-Vec analogue of
/// [`read_decoded`], for the row-contiguous load decoding each strided run into
/// its slice of the result buffer.
fn read_decoded_into(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64, out: &mut [f32], dtype: DType) {
    match find_allocation(allocs, ptr) {
        Some((base, _)) => {
            let buf = &allocs[&base];
            let off = (ptr - base) as usize;
            let end = (off + out.len() * dtype.bytes_per_elem()).min(buf.len());
            crate::codec::decode_into(&buf[off..end], out, dtype);
        }
        None => crate::codec::decode_into(&[], out, dtype),
    }
}

fn write_bytes(allocs: &mut FxHashMap<i64, Vec<u8>>, ptr: i64, data: &[u8]) {
    if let Some((base, buflen)) = find_allocation(allocs, ptr) {
        let off = (ptr - base) as usize;
        let needed = off + data.len();
        let buf = allocs.get_mut(&base).unwrap();
        if needed > buflen {
            buf.resize(needed, 0);
        }
        buf[off..off + data.len()].copy_from_slice(data);
    } else {
        allocs.insert(ptr, data.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_is_stick_aligned() {
        let mut hbm = HBMSimulator::default();
        let s0 = hbm.allocate(100); // 100 bytes -> rounds to 128
        let s1 = hbm.allocate(10);
        assert_eq!(s0, 0x10000 / STICK_BYTES);
        assert_eq!(hbm.next_ptr % STICK_BYTES, 0);
        assert_eq!(s1 - s0, 1); // next stick
    }

    #[test]
    fn write_then_read_roundtrips_with_zero_pad() {
        let mut lx = LXScratchpad::new(0, 2);
        lx.write_bytes(64, &[1, 2, 3, 4]);
        assert_eq!(lx.read_bytes(64, 4), vec![1, 2, 3, 4]);
        // reading past the end zero-pads
        assert_eq!(lx.read_bytes(64, 6), vec![1, 2, 3, 4, 0, 0]);
        // unmapped reads are all zero
        assert_eq!(lx.read_bytes(4096, 3), vec![0, 0, 0]);
    }

    #[test]
    fn hierarchy_shares_hbm_routes_lx() {
        let mem = SpyreMemoryHierarchy::new(4);
        assert_eq!(mem.num_cores, 4);
        mem.get_lx(2).borrow_mut().write_bytes(0, &[9]);
        assert_eq!(mem.get_lx(2).borrow().read_bytes(0, 1), vec![9]);
        assert_eq!(mem.get_lx(0).borrow().read_bytes(0, 1), vec![0]); // independent
    }
}
