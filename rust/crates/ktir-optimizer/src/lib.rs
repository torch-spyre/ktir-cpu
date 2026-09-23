// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! KTIR IR→IR optimization passes. Operates purely on `ktir-core` IR types — no
//! dependency on the execution layer, so passes cannot accidentally reach into
//! the interpreter (the compiler enforces it).
//!
//! First pass (in progress): manifest-guided **function fusion** — collapse a
//! multi-function KTIR program whose nodes thread intermediates through HBM into
//! a single function where those intermediates are SSA values, eliminating the
//! per-edge `store → HBM → load` round-trip.

pub mod flash_attn;
pub mod fusion;
pub mod head_rewrite;
pub mod tile_coalesce;
