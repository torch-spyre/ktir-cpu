//! The emulated Spyre machine state: the per-core [`context`] (SSA values, scope
//! stack, LX accounting, grid id / comm) and the [`memory`] hierarchy (HBM +
//! per-core LX scratchpad).

pub mod context;
pub mod memory;
