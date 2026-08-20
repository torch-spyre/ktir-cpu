// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Per-core execution state — port of `CoreContext` from `ktir_emulator/grid.py`.
//!
//! Replaces the slice-1 `Scope`. Holds the region-scoped SSA value stack, the
//! LX bump-allocator with watermark rewinding, and the grid position. Comm
//! wiring (`send_to` / remote `get_lx`) is present as the locked seam; the
//! scheduler that fills it is implement-phase.

use crate::fxhash::FxHashMap;
use std::rc::Rc;

use super::memory::{HBMSimulator, LXScratchpad, UnsafeShared};
use crate::ir::Value;

/// Maps SSA names (the leading `%` stripped) to dense `u32` ids, so the per-core
/// value table can be a flat `Vec` indexed by id instead of a `HashMap<String,_>`
/// allocating a key string on every result binding. Interning is DYNAMIC (an id
/// is assigned on first sight of a name) and the table is SHARED across passes
/// (one per function, cached by `plan_key` in the scheduler), so each distinct
/// name allocates exactly once for the whole session — not once per forward pass.
#[derive(Default)]
pub struct InternTable {
    ids: FxHashMap<String, u32>,
}

impl InternTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Id for `name`, assigning a fresh one (and the single key allocation) the
    /// first time this name is ever seen.
    pub fn intern(&mut self, name: &str) -> u32 {
        let key = name.trim_start_matches('%');
        if let Some(&id) = self.ids.get(key) {
            return id;
        }
        let id = self.ids.len() as u32;
        self.ids.insert(key.to_string(), id);
        id
    }

    /// Id for `name` if already interned — used by reads/liveness, which never
    /// need to create an id (an unseen name is simply undefined).
    pub fn get(&self, name: &str) -> Option<u32> {
        self.ids.get(name.trim_start_matches('%')).copied()
    }
}

/// Feature flags for [`CoreContext`] LX tracking — Rust port of Python's
/// `LXOptions` (#134). Both default to `true` (full tracking, production
/// behavior).
///
/// * `alias_dedup` — charge each physical tile allocation once via
///   [`CoreContext::tile_refcount`], so multiple SSA names bound to the same
///   backing buffer (iter_arg aliases, `reduce` result == `outs`) don't
///   double-count (#118).
/// * `consume_last_use` — free a tile at its single (last) use rather than
///   waiting for scope exit, keeping peak LX bounded in reduction loops (#134).
///   Requires `alias_dedup` to be correct.
#[derive(Clone, Copy, Debug)]
pub struct LxOptions {
    pub alias_dedup: bool,
    pub consume_last_use: bool,
}

impl Default for LxOptions {
    fn default() -> Self {
        LxOptions {
            alias_dedup: true,
            consume_last_use: true,
        }
    }
}

/// Per-core execution context. One per core; handlers receive `&mut CoreContext`.
pub struct CoreContext {
    pub core_id: usize,
    /// (x, y, z) position; derived from `core_id`, immutable.
    pub grid_pos: (usize, usize, usize),
    pub hbm: Rc<UnsafeShared<HBMSimulator>>,
    pub lx: Rc<UnsafeShared<LXScratchpad>>,
    /// All cores' LX, for remote `get_lx` during comm.
    all_lx: Vec<Rc<UnsafeShared<LXScratchpad>>>,
    /// SSA-name -> dense id, shared across passes (one per function). The value
    /// table below is indexed by these ids.
    intern: Rc<UnsafeShared<InternTable>>,
    /// Flat SSA value table: `slots[id]` is the value currently bound to that id
    /// (`None` = unbound). SSA names are unique within a function, so a flat table
    /// replaces the old scope-stack of `HashMap`s; region scoping is handled by the
    /// undo `trail` below rather than per-scope maps.
    slots: Vec<Option<Value>>,
    /// `lx_bytes[id]` = LX bytes charged to that id (0 = none); single source of
    /// truth for `lx.used`. Parallel to `slots`.
    ///
    /// With alias-dedup (the default), bytes are charged to the *physical tile
    /// allocation* (`tile_refcount`), not the SSA name: when an id binds a tile
    /// whose backing buffer is already charged under another live id (an
    /// `scf.for` iter_arg alias, or a `linalg.reduce` result also bound to its
    /// `outs` name), this id charges 0 bytes here but bumps the shared refcount.
    /// `lx_bytes[id]` therefore records only what *this* id is responsible for
    /// freeing on untrack (0 for an alias). This mirrors Python's `_tile_refcount`
    /// keyed by `id(Tile)` (#118).
    lx_bytes: Vec<i64>,
    /// `tile_ptr[id]` = the backing-allocation pointer ([`Tile::data_ptr`]) of the
    /// tile this id last charged LX for, or 0 if none. Parallel to `slots`. Used to
    /// find the shared refcount entry when this id is untracked. (#118 alias dedup)
    tile_ptr: Vec<usize>,
    /// `tile_refcount[ptr] = (refcount, bytes)`: how many live ids alias the tile
    /// allocation at `ptr`, and the LX bytes charged once for it. The bytes are
    /// freed (and the entry removed) when the refcount drops to 0. This is the
    /// single physical-allocation charge ledger — the Rust analogue of Python's
    /// `_tile_refcount` (#118).
    tile_refcount: FxHashMap<usize, (i64, i64)>,
    /// LX tracking feature flags (`alias_dedup`, `consume_last_use`). Default: both
    /// on (production behavior). Mirrors Python's `LXOptions` (#134).
    lx_options: LxOptions,
    /// SSA-id -> total operand-use count across the whole function (incl. nested
    /// regions). Drives consume-on-last-use in [`Self::consume_if_last_use`]: a tile
    /// with use_count 1 is freed at its single use so the consuming op's result can
    /// reuse the slot at no net LX increase. Mirrors Python's `_use_counts` (#134).
    use_counts: Vec<u32>,
    /// Undo log for region-scoped bindings: the FIRST `set_value` to an id inside a
    /// region saves `(id, pre-region slot)` here; `pop_scope` restores them in
    /// reverse so region-local values vanish (and any shadowed outer value
    /// reappears) on exit. Only the first write is saved — later overwrites (e.g. an
    /// `scf.for` accumulator re-bound every iteration within one scope) just replace
    /// the slot, exactly like the old HashMap, so the trail can't grow per iteration
    /// and dead tiles are dropped on overwrite, not pinned until the loop exits.
    trail: Vec<(u32, Option<Value>)>,
    /// `trail` length captured at each `push_scope`, so `pop_scope` knows how far
    /// to unwind.
    scope_marks: Vec<usize>,
    /// Current scope's generation (0 = function body). Each `push_scope` takes a
    /// fresh monotonic generation; `saved_gen[id] == cur_gen` means id's pre-region
    /// value is already on the trail for THIS scope, so further writes skip the save.
    cur_gen: u32,
    /// Generation stack restored by `pop_scope` (the parent scope's `cur_gen`).
    gen_stack: Vec<u32>,
    /// Monotonic source of fresh generations.
    next_gen: u32,
    /// `saved_gen[id]` = the generation in which id was last saved to the trail.
    /// Parallel to `slots`.
    saved_gen: Vec<u32>,
    /// Bump-allocator watermarks; one per live region.
    lx_next_ptr_stack: Vec<i64>,
    /// Pending cross-core sends `(dst_core, tile)`, drained by the comm
    /// scheduler after each step. The Rust analogue of Python's scheduler-wired
    /// `send_fn` (set by `attach_scheduler`).
    outbox: Vec<(usize, crate::tile::Tile)>,
    /// Whether load/store should compute the `unique_sticks` latency sideband — a
    /// per-element HBM-stick `HashSet` ONLY consumed by the latency tracker.
    /// `true` by default (faithful when metering); the comm scheduler flips it off
    /// for untracked runs (resident decode/prefill), where building the set is
    /// pure overhead on the gather hot path. See [`Self::set_track_sticks`].
    track_sticks: bool,
}

impl CoreContext {
    /// New context with its OWN fresh intern table — for one-shot executions and
    /// tests. The resident/scheduler hot path uses [`with_intern`](Self::with_intern)
    /// to SHARE one table across passes so names intern once for the session.
    pub fn new(
        core_id: usize,
        grid_pos: (usize, usize, usize),
        hbm: Rc<UnsafeShared<HBMSimulator>>,
        lx: Rc<UnsafeShared<LXScratchpad>>,
        all_lx: Vec<Rc<UnsafeShared<LXScratchpad>>>,
    ) -> Self {
        Self::with_intern(
            core_id,
            grid_pos,
            hbm,
            lx,
            all_lx,
            Rc::new(UnsafeShared::new(InternTable::new())),
        )
    }

    /// New context sharing a caller-owned intern table (so SSA names are interned
    /// once across all forward passes of one function, not once per pass).
    pub fn with_intern(
        core_id: usize,
        grid_pos: (usize, usize, usize),
        hbm: Rc<UnsafeShared<HBMSimulator>>,
        lx: Rc<UnsafeShared<LXScratchpad>>,
        all_lx: Vec<Rc<UnsafeShared<LXScratchpad>>>,
        intern: Rc<UnsafeShared<InternTable>>,
    ) -> Self {
        CoreContext {
            core_id,
            grid_pos,
            hbm,
            lx,
            all_lx,
            intern,
            slots: Vec::new(),
            lx_bytes: Vec::new(),
            tile_ptr: Vec::new(),
            tile_refcount: FxHashMap::default(),
            lx_options: LxOptions::default(),
            use_counts: Vec::new(),
            trail: Vec::new(),
            scope_marks: Vec::new(),
            cur_gen: 0,
            gen_stack: Vec::new(),
            next_gen: 1,
            saved_gen: Vec::new(),
            lx_next_ptr_stack: Vec::new(),
            outbox: Vec::new(),
            track_sticks: true,
        }
    }

    /// Grow `slots`/`lx_bytes`/`saved_gen` to cover `id`.
    #[inline]
    fn ensure_slot(&mut self, id: u32) {
        let need = id as usize + 1;
        if self.slots.len() < need {
            self.slots.resize(need, None);
            self.lx_bytes.resize(need, 0);
            self.tile_ptr.resize(need, 0);
            self.saved_gen.resize(need, 0);
        }
    }

    /// Enable/disable the `unique_sticks` latency sideband on load/store.
    /// The scheduler sets this from whether a latency tracker is attached.
    #[inline]
    pub fn set_track_sticks(&mut self, on: bool) {
        self.track_sticks = on;
    }

    /// Whether the `unique_sticks` latency sideband should be computed.
    #[inline]
    pub fn track_sticks(&self) -> bool {
        self.track_sticks
    }

    /// Queue `tile` for delivery to `dst_core`. Mirrors `send_to`; the comm
    /// scheduler drains the outbox after each step and routes the message.
    pub fn send_to(&mut self, dst_core: usize, tile: crate::tile::Tile) {
        self.outbox.push((dst_core, tile));
    }

    /// Take and clear all pending sends (called by the comm scheduler).
    pub fn drain_outbox(&mut self) -> Vec<(usize, crate::tile::Tile)> {
        std::mem::take(&mut self.outbox)
    }

    /// Grid coordinate for a dimension (0=x, 1=y, 2=z). Mirrors `get_grid_id`.
    pub fn get_grid_id(&self, dim: usize) -> usize {
        match dim {
            0 => self.grid_pos.0,
            1 => self.grid_pos.1,
            2 => self.grid_pos.2,
            _ => 0,
        }
    }

    /// Bind an SSA value. Mirrors `set_value`. SSA names are unique within a
    /// function, so this writes a flat slot; a write made inside a region records
    /// the previous slot on the undo `trail` for `pop_scope`.
    pub fn set_value(&mut self, name: &str, value: Value) {
        let id = self.intern.borrow_mut().intern(name);
        self.ensure_slot(id);
        let i = id as usize;
        // Inside a region, save the pre-region value ONCE (first write this scope)
        // so `pop_scope` can restore it; later overwrites just replace the slot.
        if self.cur_gen != 0 && self.saved_gen[i] != self.cur_gen {
            self.saved_gen[i] = self.cur_gen;
            self.trail.push((id, self.slots[i].take()));
        }
        self.slots[i] = Some(value);
    }

    /// Look up an SSA value. Mirrors `get_value`. O(1) slot index after one id
    /// lookup; an unseen name is undefined.
    pub fn get_value(&self, name: &str) -> Result<&Value, String> {
        let id = self.intern.borrow().get(name);
        match id
            .and_then(|i| self.slots.get(i as usize))
            .and_then(Option::as_ref)
        {
            Some(v) => Ok(v),
            None => Err(format!("undefined SSA value: {name}")),
        }
    }

    pub fn has_value(&self, name: &str) -> bool {
        self.intern
            .borrow()
            .get(name)
            .and_then(|i| self.slots.get(i as usize))
            .is_some_and(Option::is_some)
    }

    /// Enter a region: snapshot the LX watermark, mark the undo trail, take a fresh
    /// generation.
    pub fn push_scope(&mut self) {
        self.lx_next_ptr_stack.push(self.lx.borrow().next_ptr);
        self.scope_marks.push(self.trail.len());
        self.gen_stack.push(self.cur_gen);
        self.cur_gen = self.next_gen;
        self.next_gen += 1;
    }

    /// Exit the current region: restore region-scoped bindings (untracking their
    /// LX), rewind LX to the watermark. Panics on the function-body scope.
    pub fn pop_scope(&mut self) {
        let mark = self
            .scope_marks
            .pop()
            .expect("cannot pop function-body scope");
        // Unwind region writes in reverse: free each id's LX and restore the slot
        // to its pre-region value (`None` for a region-local binding, or the
        // shadowed outer value).
        while self.trail.len() > mark {
            let (id, old) = self.trail.pop().unwrap();
            self.untrack_lx_id(id);
            self.slots[id as usize] = old;
        }
        self.cur_gen = self
            .gen_stack
            .pop()
            .expect("cannot pop function-body scope");
        let watermark = self.lx_next_ptr_stack.pop().unwrap();
        self.lx.borrow_mut().next_ptr = watermark;
    }

    /// Record an SSA value occupying `size_bytes` in LX. Mirrors `track_lx`;
    /// returns an error instead of raising `MemoryError` on overflow.
    ///
    /// Pointer-blind: charges `size_bytes` against `name` with no alias dedup
    /// (every call charges). Used by tests and the resident/Metal offload sites
    /// that bind a freshly-computed (non-aliased) tile under a single name.
    /// Tile-producing interpreter bindings go through [`Self::track_lx_tile`],
    /// which dedups aliases of the same allocation (#118).
    pub fn track_lx(&mut self, name: &str, size_bytes: i64) -> Result<(), String> {
        let (used, capacity) = {
            let lx = self.lx.borrow();
            (lx.used, lx.capacity)
        };
        if used + size_bytes > capacity {
            return Err(format!(
                "LX capacity exceeded on core {}: {} + {} > {}",
                self.core_id, used, size_bytes, capacity
            ));
        }
        self.lx.borrow_mut().used += size_bytes;
        let id = self.intern.borrow_mut().intern(name);
        self.ensure_slot(id);
        // Release whatever this id was previously charged with (alias-aware) and
        // record this as a fresh, ptr-less charge so untrack frees it directly.
        self.untrack_lx_id(id);
        self.lx_bytes[id as usize] = size_bytes;
        self.tile_ptr[id as usize] = 0;
        Ok(())
    }

    /// Charge LX for binding `tile` to `name`, deduping aliases of the same
    /// physical allocation. Port of Python's refcount-by-`id(Tile)` (#118): the
    /// FIRST live id to charge a given backing buffer pays `tile.size_bytes()`;
    /// later ids that bind the *same* `Tile::data_ptr` (iter_arg rebinds, the
    /// `reduce` result/`outs` alias) only bump the shared refcount, charging 0
    /// bytes themselves. Overflow is checked only on the first (charging) bind.
    pub fn track_lx_tile(&mut self, name: &str, tile: &crate::tile::Tile) -> Result<(), String> {
        if !self.lx_options.alias_dedup {
            return self.track_lx(name, tile.size_bytes() as i64);
        }
        let id = self.intern.borrow_mut().intern(name);
        self.ensure_slot(id);
        // Re-binding the same name: release its previous charge first.
        self.untrack_lx_id(id);
        let ptr = tile.data_ptr();
        let bytes = tile.size_bytes() as i64;
        let entry = self.tile_refcount.entry(ptr).or_insert((0, bytes));
        if entry.0 == 0 {
            // First live alias of this allocation — charge it once.
            let (used, capacity) = {
                let lx = self.lx.borrow();
                (lx.used, lx.capacity)
            };
            if used + bytes > capacity {
                // Leave the (count 0) entry; it carries no charge.
                return Err(format!(
                    "LX capacity exceeded on core {}: {} + {} > {}",
                    self.core_id, used, bytes, capacity
                ));
            }
            self.lx.borrow_mut().used += bytes;
            entry.1 = bytes;
        }
        entry.0 += 1;
        // This id holds a reference to `ptr`; it charges no standalone bytes — the
        // bytes live on the shared refcount entry, freed when it hits 0.
        self.lx_bytes[id as usize] = 0;
        self.tile_ptr[id as usize] = ptr;
        Ok(())
    }

    /// Free LX for `name`. No-op if untracked. Mirrors `untrack_lx`.
    pub fn untrack_lx(&mut self, name: &str) {
        let id = self.intern.borrow().get(name);
        if let Some(id) = id {
            self.untrack_lx_id(id);
        }
    }

    #[inline]
    fn untrack_lx_id(&mut self, id: u32) {
        let i = id as usize;
        // Alias-deduped charge: drop this id's reference to the shared allocation;
        // free the bytes only when the last alias releases (refcount -> 0). (#118)
        if let Some(ptr_slot) = self.tile_ptr.get_mut(i)
            && *ptr_slot != 0
        {
            let ptr = *ptr_slot;
            *ptr_slot = 0;
            if let Some(entry) = self.tile_refcount.get_mut(&ptr) {
                entry.0 -= 1;
                if entry.0 <= 0 {
                    let bytes = entry.1;
                    self.tile_refcount.remove(&ptr);
                    self.lx.borrow_mut().used -= bytes;
                }
            }
        }
        // Pointer-blind charge (`track_lx`): free its standalone bytes directly.
        if let Some(sz) = self.lx_bytes.get_mut(i)
            && *sz != 0
        {
            self.lx.borrow_mut().used -= *sz;
            *sz = 0;
        }
    }

    /// Drop a dead SSA value entirely: free its LX accounting AND remove its host
    /// backing (the tile `Rc<[f32]>`) so it can be freed. Used by the liveness
    /// reclaim — without it a whole-program-fused function would hold every
    /// intermediate tile resident at once. Only TILES are evicted; pointers/
    /// scalars are tiny and may still be read after their last operand use
    /// (e.g. the GPU matmul-loop offload resolves a weight's base pointer at the
    /// loop op, after the memory-view that "used" it), so they're kept.
    pub fn forget(&mut self, name: &str) {
        let id = self.intern.borrow().get(name);
        if let Some(id) = id {
            self.forget_id(id);
        }
    }

    /// [`forget`](Self::forget) by pre-resolved id — the liveness-reclaim hot path,
    /// where `dies_at` is resolved to ids once per function so per-op reclaim skips
    /// the name lookup entirely.
    pub fn forget_id(&mut self, id: u32) {
        self.untrack_lx_id(id);
        if let Some(slot @ Some(Value::Tile(_))) = self.slots.get_mut(id as usize) {
            *slot = None;
        }
    }

    /// Intern `name` to its id (assigning one on first sight). For callers that
    /// pre-resolve liveness/result names to ids.
    pub fn intern_id(&self, name: &str) -> u32 {
        self.intern.borrow_mut().intern(name)
    }

    /// Override the LX tracking feature flags. Tests use this to isolate
    /// `alias_dedup` / `consume_last_use` and measure each mechanism's effect on
    /// `lx.used`; production keeps the default (both on). Mirrors passing a custom
    /// `LXOptions` to Python's `CoreContext`.
    pub fn set_lx_options(&mut self, opts: LxOptions) {
        self.lx_options = opts;
    }

    /// Install the per-function operand use-count map (SSA name -> total operand
    /// occurrences, counting uses inside nested regions), interning each name to
    /// its id. Enables consume-on-last-use (#134). Call once per function before
    /// execution; safe to call again to refresh.
    pub fn set_use_counts(&mut self, counts: &std::collections::HashMap<String, usize>) {
        for (name, &n) in counts {
            let id = self.intern.borrow_mut().intern(name) as usize;
            if self.use_counts.len() <= id {
                self.use_counts.resize(id + 1, 0);
            }
            self.use_counts[id] = n as u32;
        }
    }

    /// Whether `consume_last_use` is enabled (interpreter operand-resolution hook).
    #[inline]
    pub fn consume_last_use_enabled(&self) -> bool {
        self.lx_options.consume_last_use
    }

    /// Consume `name` if this fetch is its last use, freeing its LX before the
    /// consuming op's result is charged — Rust port of Python's consume-on-last-use
    /// in `get_value` (#134). A tile is consumed when:
    ///   * `consume_last_use` is enabled, and
    ///   * its global `use_count == 1` (single use — this one), and
    ///   * it is bound in the *currently active* (topmost) region scope.
    ///
    /// The topmost-scope guard (mirroring Python's `scope is _scope_stack[-1]`)
    /// keeps a value loaded before a loop and read once *per iteration* from being
    /// freed on the first iteration: such a name lives in an outer scope, so its
    /// most recent write was not made in the current generation.
    ///
    /// Removes the slot binding (so a later stray read errors, as in Python) and
    /// releases the LX charge (alias-aware). No-op for non-tiles, multi-use names,
    /// outer-scope names, or unbound names.
    pub fn consume_if_last_use(&mut self, name: &str) {
        if !self.lx_options.consume_last_use {
            return;
        }
        // Only consume INSIDE a region (`cur_gen != 0` — an scf.for/scf.if body).
        // At function top level the resident scheduler owns liveness via its
        // `dies_at`/`forget` reclaim AND the Metal map-window / matmul-loop offloads,
        // which read tiles at a DEFERRED point (e.g. a window trigger or the K-loop
        // op resolving a weight pointer) AFTER an operand's nominal last use. Eagerly
        // consuming top-level tiles here would race those reads (the e2e golden
        // diverged by 30 logits). The conformance-critical peak-LX reduction (#134)
        // is the per-iteration intermediates of reduction / softmax loop BODIES,
        // which are region-scoped — exactly the `cur_gen != 0` case. The function
        // body is never offloaded as a whole, so dies_at covers its top level.
        if self.cur_gen == 0 {
            return;
        }
        let Some(id) = self.intern.borrow().get(name) else {
            return;
        };
        let i = id as usize;
        // Single-use only.
        if self.use_counts.get(i).copied().unwrap_or(0) != 1 {
            return;
        }
        // Must be a live Tile.
        if !matches!(self.slots.get(i), Some(Some(Value::Tile(_)))) {
            return;
        }
        // Topmost-scope guard (Python's `scope is _scope_stack[-1]`): the value must
        // have been written in THIS region generation (its pre-region value was
        // saved on the trail), i.e. `saved_gen[id] == cur_gen`. This keeps a value
        // loaded before the loop and read once per iteration (outer scope) from
        // being freed on the first iteration.
        if self.saved_gen.get(i).copied() != Some(self.cur_gen) {
            return;
        }
        // Free LX and drop the binding.
        self.untrack_lx_id(id);
        self.slots[i] = None;
    }

    /// Return the LX for a core: local fast path, else a remote handle.
    /// Mirrors `get_lx`.
    pub fn get_lx(&self, core_id: Option<usize>) -> Rc<UnsafeShared<LXScratchpad>> {
        match core_id {
            None => Rc::clone(&self.lx),
            Some(id) if id == self.core_id => Rc::clone(&self.lx),
            Some(id) => Rc::clone(&self.all_lx[id]),
        }
    }

    /// Reset for the next execution round. Mirrors `clear_values`. Keeps the
    /// `slots`/`lx_bytes` capacity (refilled lazily) and the shared intern table.
    pub fn clear_values(&mut self) {
        self.slots.iter_mut().for_each(|s| *s = None);
        self.lx_bytes.iter_mut().for_each(|b| *b = 0);
        self.tile_ptr.iter_mut().for_each(|p| *p = 0);
        self.tile_refcount.clear();
        self.saved_gen.iter_mut().for_each(|g| *g = 0);
        self.trail.clear();
        self.scope_marks.clear();
        self.gen_stack.clear();
        self.cur_gen = 0;
        self.lx_next_ptr_stack.clear();
        self.lx.borrow_mut().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::memory::SpyreMemoryHierarchy;
    use super::*;
    use crate::dtypes::DType;
    use crate::ir::Scalar;

    fn ctx() -> CoreContext {
        let mem = SpyreMemoryHierarchy::new(2);
        CoreContext::new(
            0,
            (0, 0, 0),
            Rc::clone(&mem.hbm),
            mem.get_lx(0),
            mem.lx_scratchpads.clone(),
        )
    }

    #[test]
    fn scopes_shadow_and_resolve_outward() {
        let mut c = ctx();
        c.set_value("%a", Value::Index(1));
        c.push_scope();
        c.set_value("%b", Value::Index(2));
        assert!(matches!(c.get_value("%a").unwrap(), Value::Index(1))); // outer visible
        assert!(matches!(c.get_value("%b").unwrap(), Value::Index(2)));
        c.pop_scope();
        assert!(c.get_value("%b").is_err()); // inner gone
        assert!(c.has_value("%a"));
    }

    #[test]
    fn lx_tracking_and_watermark_rewind() {
        let mut c = ctx();
        c.track_lx("%t", 256).unwrap();
        assert_eq!(c.lx.borrow().used, 256);
        c.push_scope();
        c.set_value("%inner", Value::Scalar(Scalar::I64(0)));
        c.track_lx("%inner", 128).unwrap();
        assert_eq!(c.lx.borrow().used, 384);
        c.pop_scope(); // frees %inner
        assert_eq!(c.lx.borrow().used, 256);
    }

    #[test]
    fn lx_overflow_is_an_error() {
        let mut c = ctx();
        let cap = c.lx.borrow().capacity;
        assert!(c.track_lx("%big", cap + 1).is_err());
    }

    // Re-binding a name many times WITHIN one scope (an scf.for accumulator) must
    // not grow the undo trail per write, and must drop the overwritten values —
    // pop restores the single pre-region value. Regression for the trail bug.
    #[test]
    fn loop_rebind_within_one_scope_is_bounded_and_pops_clean() {
        let mut c = ctx();
        c.set_value("%acc", Value::Index(0)); // outer (function-scope) binding
        c.push_scope();
        for i in 1..=100 {
            c.set_value("%acc", Value::Index(i)); // re-bind every "iteration"
        }
        assert_eq!(c.trail.len(), 1, "only the first write per id is saved");
        assert!(matches!(c.get_value("%acc").unwrap(), Value::Index(100)));
        c.pop_scope();
        // The outer binding (pre-region value) is restored.
        assert!(matches!(c.get_value("%acc").unwrap(), Value::Index(0)));
    }

    // A region-LOCAL name (no outer binding) vanishes on pop; nested scopes unwind.
    #[test]
    fn region_local_vanishes_and_nesting_unwinds() {
        let mut c = ctx();
        c.push_scope();
        c.set_value("%x", Value::Index(1));
        c.push_scope();
        c.set_value("%y", Value::Index(2));
        assert!(c.has_value("%x") && c.has_value("%y"));
        c.pop_scope();
        assert!(c.has_value("%x") && !c.has_value("%y")); // inner gone
        c.pop_scope();
        assert!(!c.has_value("%x")); // outer region gone too
    }

    // forget evicts TILES but keeps pointers/scalars (the GPU offload reads a
    // weight pointer after its liveness "death").
    #[test]
    fn forget_evicts_tiles_keeps_pointers() {
        use crate::tile::Tile;
        let mut c = ctx();
        c.set_value(
            "%t",
            Value::Tile(Tile::compute(vec![1.0], DType::F32, vec![1])),
        );
        c.set_value("%p", Value::Index(42));
        c.forget("%t");
        c.forget("%p");
        assert!(!c.has_value("%t")); // tile evicted
        assert!(matches!(c.get_value("%p").unwrap(), Value::Index(42))); // pointer kept
    }
}
