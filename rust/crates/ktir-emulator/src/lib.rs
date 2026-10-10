// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! KTIR CPU validation interpreter — the execution layer (RFC 0682). The IR
//! types, parser, affine, dtypes, tile/memref, and f16 codec live in the
//! dependency-free `ktir-core` crate and are re-exported here, so emulator
//! modules keep using `crate::ir` / `crate::tile` / … and downstream code keeps
//! using `ktir_emulator::ir` / `ktir_emulator::parser` / … unchanged.
//!
//! Execution contract: handlers have signature
//! `(op, &mut CoreContext, &ExecutionEnv) -> Result<Option<Value>, String>`,
//! run nested regions via `interpreter::execute_region`, and the cross-core
//! comm seam lives in `comm` (only the top-level driver suspends).

// Links the BLAS backend — Accelerate on macOS (default), or the feature-chosen
// provider elsewhere. `blas-src` must be referenced once at the crate root for
// its linker directives to take effect. See blas.rs.
#[cfg(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
))]
extern crate blas_src;

// Re-export the core IR/parse/codec layer at this crate's root.
pub use ktir_core::{affine, codec, dtypes, fxhash, ir, memref, parser, parser_ast, tile};

// Re-export the optimizer (whole-program fusion / ProgramSpec / plan_segments)
// when the `optimizer` feature is on, so consumers reach it through ktir-emulator
// without a separate ktir-optimizer dependency.
#[cfg(feature = "optimizer")]
pub use ktir_optimizer;

pub mod blas;
pub mod comm;
pub mod comm_sched;
pub mod dialects;
pub mod env;
pub mod interpreter;
pub mod latency;
// The emulated Spyre machine state (per-core `context` + `memory` hierarchy).
// Re-exported at the root so `crate::context::…` / `crate::memory::…` still resolve.
pub mod machine_state;
pub use machine_state::{context, memory};
#[cfg(metal)]
pub mod metal;
pub mod ops_memory;
// The fused/serving execution drivers depend on the optimizer's ProgramSpec.
#[cfg(feature = "optimizer")]
pub mod resident;
#[cfg(feature = "optimizer")]
pub mod segmented;
// Drive a single-function example program through the resident/segmented path.
#[cfg(feature = "optimizer")]
pub mod resident_runner;
// Turnkey entrypoints (program::execute / Session) over the fused + resident path.
#[cfg(feature = "optimizer")]
pub mod program;
