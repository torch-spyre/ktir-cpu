// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Cross-core communication seam — the locked contract for the one genuine
//! redesign in the port. Python comm ops are generators that `yield
//! RecvRequest` and resume via `gen.send(tile)`; the `GridExecutor` scheduler
//! parks and wakes cores. Rust has no generators, so a comm op is an explicit
//! state machine: [`CommOp::step`] is called repeatedly, returning [`CommStep`]
//! to either request a receive (park) or finish.
//!
//! Crucially (per the map): comm only happens at the **top-level** function
//! body, never inside nested regions. So `execute_region` stays synchronous and
//! only the top-level driver in `interpreter.rs` runs this protocol. The
//! scheduler + concrete comm ops (ring reduce, send/recv) are an implement-phase
//! fill against these types.

use crate::ir::Value;
use crate::tile::Tile;

/// Yielded by a parked comm op: "resume me when a tile arrives from `src`".
/// Mirrors the frozen `RecvRequest` dataclass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecvRequest {
    pub src: usize,
}

/// One step of a comm op's state machine. Replaces the Python generator's
/// `yield RecvRequest` / `return value` duality.
pub enum CommStep {
    /// Park this core until a tile arrives from `req.src`. The scheduler resumes
    /// by calling `step` again with that tile.
    Recv(RecvRequest),
    /// The op finished; bind this (optional) value to the op's result. Boxed
    /// because `Value` is large and the `Recv` variant is tiny.
    Done(Box<Option<Value>>),
}

/// A comm op as an explicit, resumable state machine. The driver calls `step`
/// with `None` first, then with each delivered `Tile` until it returns `Done`.
///
/// `env` is threaded in so a comm op whose fold is an IR region (the inter-tile
/// reduce combiner) can drive that region synchronously via the dispatch table —
/// the same `execute_region` the synchronous handlers use. Ring algorithms with a
/// fixed combiner (the legacy `ktdp.reduce`) simply ignore it.
pub trait CommOp {
    fn step(
        &mut self,
        ctx: &mut crate::context::CoreContext,
        env: &crate::env::ExecutionEnv,
        incoming: Option<Tile>,
    ) -> Result<CommStep, String>;
}
