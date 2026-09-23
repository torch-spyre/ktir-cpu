// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Dialect dispatch — Rust port of the handler registry in
//! `ktir_emulator/dialects/registry.py`.
//!
//! Python registers handlers at import time via a `@register` decorator that
//! mutates a global dict. Rust uses the explicit-table approach (Option A from
//! the design sketch): each dialect module exposes `register(&mut Dispatch)`,
//! and [`Dispatch::new`] assembles them. Greppable, no macro magic, and the
//! registration is visible rather than hidden in attribute macros.

pub mod arith;
pub mod func;
pub mod ktdp;
pub mod ktdp_comm;
pub mod ktdp_extra;
pub mod linalg;
pub mod math;
pub mod scf;
pub mod tensor;

use crate::fxhash::FxHashMap;

use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};

// The single source of truth for latency categories is `crate::latency`.
// Re-exported here so dialect modules can write `super::LatencyCategory` (or
// `crate::dialects::LatencyCategory`) and get the full 7-variant enum.
pub use crate::latency::LatencyCategory;

/// Handler signature. Mirrors Python's `HandlerFn = (op, context, env) -> Any`:
/// reads operands via `ctx.get_value`, runs nested regions via the dispatch
/// table in `env`, and returns the value to bind to `op.result` (or `None`).
pub type HandlerFn =
    fn(&Operation, &mut CoreContext, &ExecutionEnv) -> Result<Option<Value>, String>;

/// Op-name -> handler table, plus the parallel latency-category table that the
/// Python registry keeps in lockstep.
pub struct Dispatch {
    handlers: FxHashMap<&'static str, HandlerFn>,
    latency: FxHashMap<&'static str, LatencyCategory>,
}

impl Dispatch {
    /// Build the table by letting each dialect register its ops.
    pub fn new() -> Self {
        let mut d = Dispatch {
            handlers: FxHashMap::default(),
            latency: FxHashMap::default(),
        };
        arith::register(&mut d);
        func::register(&mut d);
        ktdp::register(&mut d);
        ktdp_comm::register(&mut d);
        ktdp_extra::register(&mut d);
        math::register(&mut d);
        linalg::register(&mut d);
        tensor::register(&mut d);
        scf::register(&mut d);
        crate::ops_memory::register(&mut d);
        d
    }

    /// Process-wide shared dispatch table, built once. The registry is immutable
    /// after construction (op-name -> fn pointer, plus the parallel latency map),
    /// so there is no reason to rebuild it per call — `execute_function` does so
    /// once per node, ~271K times in a real-model run, which the profile flagged.
    /// Function pointers are `Send + Sync`, so the `&'static` table is safe to
    /// share across the grid's SPMD threads.
    pub fn shared() -> &'static Dispatch {
        use std::sync::OnceLock;
        static SHARED: OnceLock<Dispatch> = OnceLock::new();
        SHARED.get_or_init(Dispatch::new)
    }

    /// Called by dialect modules. Mirrors the `@register(name, latency_category)` decorator.
    pub fn register(&mut self, op_name: &'static str, cat: LatencyCategory, f: HandlerFn) {
        self.handlers.insert(op_name, f);
        self.latency.insert(op_name, cat);
    }

    /// Look up a handler — mirrors `dispatch(op_name)`.
    pub fn handler(&self, op_name: &str) -> Option<HandlerFn> {
        self.handlers.get(op_name).copied()
    }

    /// Latency category, defaulting to `Zero` — mirrors `get_latency_category`.
    pub fn latency_category(&self, op_name: &str) -> LatencyCategory {
        self.latency
            .get(op_name)
            .copied()
            .unwrap_or(LatencyCategory::Zero)
    }
}

impl Default for Dispatch {
    fn default() -> Self {
        Self::new()
    }
}
