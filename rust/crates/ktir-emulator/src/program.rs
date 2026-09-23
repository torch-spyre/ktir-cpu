// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Turnkey entrypoints to run a whole KTIR program through the optimized
//! (fused / resident) execution path.
//!
//! Running a whole program *optimized* — whole-program fusion + GPU offloads +
//! head-parallel attention — otherwise takes a few steps: parse every node's MLIR
//! into one [`IRModule`], build a [`ProgramSpec`], then call
//! [`crate::segmented::execute_segmented`] or [`crate::resident::ResidentExecutor`].
//! (The per-node `interpreter::execute_function` path does NONE of this — it runs
//! each node in isolation at its native grid, where the GPU offloads, gated on a
//! single-core grid, never fire. That is the slow parity-oracle path.)
//!
//! These helpers collapse that to one call. They are **manifest-agnostic**: the
//! caller supplies the per-node MLIR and a [`ProgramSpec`] built from its own
//! manifest (`ProgramSpec` / `NodeSpec` / `Binding` are plain public structs in
//! `ktir_optimizer::fusion`, re-exported as `ktir_emulator::ktir_optimizer`).
//!
//! - [`execute`] — turnkey single-shot (e.g. one prefill pass).
//! - [`Session`] — resident multi-pass serving (decode): weights uploaded ONCE,
//!   kernels chained on-device per pass with no weight re-marshal.
//!
//! Both require the `optimizer` feature (on by default).

use std::collections::HashMap;

use crate::interpreter::{Arg, Output};
use crate::ir::IRModule;
use crate::parser::parse_module;
use ktir_optimizer::fusion::ProgramSpec;

/// Parse a program's per-node MLIR into ONE module (every `func.func` merged) so
/// the optimizer sees the whole program. `node_mlir[i]` is the MLIR text for one
/// node; each node may declare one or more functions (all are added). Function
/// names must be unique across nodes (they are looked up by name during
/// execution) — the per-node bundles scratchy emits already satisfy this.
pub fn module_from_nodes(node_mlir: &[&str]) -> Result<IRModule, String> {
    let mut module = IRModule::default();
    for (i, src) in node_mlir.iter().enumerate() {
        let parsed = parse_module(src).map_err(|e| format!("program: parse node {i}: {e}"))?;
        for (_, f) in parsed.functions {
            module.add_function(f);
        }
    }
    // NOTE: the attention IR rewrites (head re-roll, TODO #1; flash cap-tiling,
    // TODO #2) are NOT applied here. They run at the EXECUTION ENTRY POINT
    // (`segmented::apply_attention_rewrites`, called by `execute_segmented` and
    // `ResidentExecutor`), so they fire for EVERY path that executes a module —
    // including a caller that builds the module itself and runs `execute_segmented`
    // directly (e.g. the real-model e2e harness), not just this turnkey builder.
    Ok(module)
}

/// Turnkey single-shot: parse all node MLIR into one module and run the whole
/// program through the optimized segmented path (whole-program fusion + K-loop
/// GEMM / map-window GPU offloads + native head-parallel attention). Equivalent
/// to [`module_from_nodes`] followed by [`crate::segmented::execute_segmented`].
///
/// `spec` describes the program (node order, arg↔tensor bindings, which tensors
/// are sources vs results) — build it from your manifest. `args` are the source
/// tensors (weights + inputs); `outputs` are the result keys (`t<id>`) to read
/// back. For repeated passes over the same weights (decode), use [`Session`].
pub fn execute(
    node_mlir: &[&str],
    spec: &ProgramSpec,
    args: &[(&str, Arg)],
    outputs: &[&str],
) -> Result<HashMap<String, Output>, String> {
    let module = module_from_nodes(node_mlir)?;
    crate::segmented::execute_segmented(&module, spec, args, outputs)
}

/// A resident serving session for MULTI-pass execution (e.g. autoregressive
/// decode). Weights are marshaled into a persistent GPU HBM **once** at
/// construction; each [`run`](Session::run) chains the program's kernels
/// on-device with no per-pass weight re-marshal. Between passes, overwrite only
/// the changing source tensors (the next token's input activation, the updated
/// attention mask) with [`set_sources`](Session::set_sources).
///
/// The session OWNS its module (moved in), so it owns its entire object graph and
/// is therefore `Send` — a serving worker can store it and move it between threads
/// (it is single-threaded internally, so NOT `Sync`: don't share one by `&` across
/// threads; run it serially). Build the module with [`module_from_nodes`] and move
/// it in:
/// ```ignore
/// let module = program::module_from_nodes(&node_mlir)?;
/// let mut sess = program::Session::new(module, &spec, &weights)?;  // module moved in
/// loop {
///     sess.set_sources(&[("t0", next_input), ("t_mask", mask)])?;
///     let out = sess.run(&["t_result"])?;
/// }
/// ```
pub struct Session {
    exec: crate::resident::ResidentExecutor,
}

impl Session {
    /// Build the session, taking OWNERSHIP of `module` (from [`module_from_nodes`]).
    /// `weights` is the full initial source set (weights + mask + first input);
    /// it is uploaded to resident HBM once here.
    pub fn new(
        module: IRModule,
        spec: &ProgramSpec,
        weights: &[(&str, Arg)],
    ) -> Result<Self, String> {
        let mut exec = crate::resident::ResidentExecutor::new(module, spec)?;
        exec.set_sources(weights)?;
        Ok(Self { exec })
    }

    /// Build a session whose resident weights are SHARED across multiple programs
    /// — the prefill and decode bundles, which use the SAME weights but different
    /// shapes (M). The weight set is uploaded ONCE here; both programs run against
    /// it with no second load. `programs` are in declaration order: `run_program(0,
    /// ..)` runs the first, `run_program(1, ..)` the second, etc.
    pub fn new_multi(
        programs: Vec<(IRModule, &ProgramSpec)>,
        weights: &[(&str, Arg)],
    ) -> Result<Self, String> {
        let mut exec = crate::resident::ResidentExecutor::new_multi(programs)?;
        exec.set_sources(weights)?;
        Ok(Self { exec })
    }

    /// Overwrite source tensors in resident HBM (the per-pass input / mask).
    /// Sources you don't pass keep their resident bytes — so weights stay put.
    pub fn set_sources(&mut self, args: &[(&str, Arg)]) -> Result<(), String> {
        self.exec.set_sources(args)
    }

    /// Run one forward pass of program 0 and read back `outputs` (empty = results).
    pub fn run(&mut self, outputs: &[&str]) -> Result<HashMap<String, Output>, String> {
        self.exec.run(outputs)
    }

    /// Run one forward pass of program `idx` (e.g. 0 = prefill, 1 = decode) against
    /// the shared resident weights, reading back `outputs` (empty = that program's
    /// results). Switching programs re-uploads no weights.
    pub fn run_program(
        &mut self,
        idx: usize,
        outputs: &[&str],
    ) -> Result<HashMap<String, Output>, String> {
        self.exec.run_program(idx, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Locks in the `unsafe impl Send for ResidentExecutor` contract: a serving
    // worker needs `Session: Send`. If a future change reintroduces a borrow or a
    // non-Send field, this stops compiling instead of silently regressing.
    #[test]
    fn session_and_executor_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Session>();
        assert_send::<crate::resident::ResidentExecutor>();
    }
}
