// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Parse-then-execute test relocated from `ktir-core`'s `parser.rs` when the
//! workspace was split: it parses an MLIR function and runs it, so it needs the
//! execution layer (`ktir-emulator`), which `ktir-core` must not depend on.

use ktir_emulator::dialects::Dispatch;
use ktir_emulator::env::{ExecutionEnv, GridExecutor};
use ktir_emulator::interpreter::{execute_ops, single_core_context};
use ktir_emulator::ir::{Scalar, Value};
use ktir_emulator::parser::parse_module;

#[test]
fn parse_then_execute_arith_function() {
    let src = r#"
        module {
          func.func @f() attributes {grid = [1]} {
            %a = arith.constant 2.0 : f32
            %b = arith.constant 3.0 : f32
            %c = arith.addf %a, %b : f32
            %d = arith.mulf %c, %a : f32
            return
          }
        }
    "#;
    let module = parse_module(src).unwrap();
    let f = module.get_function("f").unwrap();

    let dispatch = Dispatch::new();
    let grid = GridExecutor::new(f.grid);
    let env = ExecutionEnv::new(&dispatch, &grid);
    let mut ctx = single_core_context();
    execute_ops(&f.operations, &mut ctx, &env).unwrap();
    match ctx.get_value("%d").unwrap() {
        Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0), // (2+3)*2
        other => panic!("expected F32(10.0), got {other:?}"),
    }
}
