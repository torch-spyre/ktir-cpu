// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `func` dialect — the function terminator. KTIR functions write outputs to
//! HBM, so `return` is usually void; but it may carry operands. The handler
//! surfaces them (single -> that value, multiple -> `Value::Tuple`, none ->
//! `None`). The op has no SSA result name, so the value isn't bound into scope —
//! it's observable only by a direct handler/`execute_op` call (matching how the
//! Python `func.return` returns its operand values).

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};

pub fn register(d: &mut Dispatch) {
    d.register("return", LatencyCategory::Zero, ret);
    d.register("func.return", LatencyCategory::Zero, ret);
}

fn ret(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let mut vals: Vec<Value> = op
        .operands
        .iter()
        .map(|name| ctx.get_value(name).cloned())
        .collect::<Result<_, _>>()?;
    Ok(match vals.len() {
        0 => None,
        1 => Some(vals.pop().unwrap()),
        _ => Some(Value::Tuple(vals)),
    })
}
