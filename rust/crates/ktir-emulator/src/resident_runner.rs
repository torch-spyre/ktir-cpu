// Copyright 2025 The Torch-Spyre Authors. Apache-2.0.
//
//! Drive a SINGLE-FUNCTION example program through the PRODUCTION resident /
//! segmented Metal executor ([`crate::segmented::execute_segmented`]).
//!
//! The fused / segmented path threads logical tensors by id (`t<id>`) and runs
//! each non-attention node as a `[1,1]` [`Segment::Fused`] whose body carries the
//! GPU offloads (NAX K-loop GEMM reconstruction, fused map windows, fused decode
//! attention). It was built for the OPTIMIZER's emitted IR, where every launch
//! scalar (tile-loop bound, block size, `K`) is already a literal `arith.constant`
//! and the only function arguments are tensor pointers.
//!
//! The hand-written `examples/*.mlir` programs instead carry their launch scalars
//! as `index`/`i32`/`f16` FUNCTION ARGUMENTS (`%K`, `%BLOCK_SIZE_M`, `%n_rows`,
//! `%scale`, ...). Fusion only keeps pointer args, so those scalar SSA values go
//! dangling ("undefined SSA value %n0_BLOCK_SIZE_M"). [`ResidentRunner`] closes
//! that gap with ONE principled, decomposition-agnostic rewrite: SPECIALIZE the
//! kernel to its concrete launch scalars — replace each scalar function argument
//! with an `arith.constant` of the bound value, prepended to the body, and drop it
//! from the signature. That produces exactly the literal-bound form the optimizer
//! already emits, so the unmodified `execute_segmented` then runs it end-to-end on
//! the Metal path. (It is NOT a model-specific hack: it specializes ANY function's
//! scalar args, recognizing no op pattern.)
//!
//! A program with NO scalar args needs no rewrite; one whose tensors live at
//! hardcoded HBM addresses (the RFC `hbm_seed` fixtures) cannot be expressed as a
//! marshalled-arg `ProgramSpec` and is reported as not-drivable (see the diff
//! CLI), not faked.

use crate::dtypes::DType;
use crate::interpreter::{Arg, Output};
use crate::ir::{Attr, IRFunction, IRModule, Operation, Scalar};
use ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec};
use std::collections::{HashMap, HashSet};

/// One tensor argument of an example kernel: the function arg name (no `%`), the
/// raw little-endian bytes already in `dtype` layout, its shape, and whether the
/// kernel WRITES it (an output to read back) or only reads it (an input source).
pub struct TensorArg {
    pub name: String,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub is_output: bool,
}

/// One scalar argument: the function arg name (no `%`) and its value. These are
/// specialized into `arith.constant` ops, NOT threaded as tensors.
pub struct ScalarArg {
    pub name: String,
    pub value: Scalar,
}

/// The plan that drives a single example function through `execute_segmented`:
/// the scalar-specialized module + the one-node `ProgramSpec` + the synthetic
/// tensor-id assignment, so the runner can marshal inputs / read back outputs by
/// the ORIGINAL arg name.
pub struct ResidentRunner {
    module: IRModule,
    func: String,
    spec: ProgramSpec,
    /// original arg name -> synthetic tensor id.
    arg_tid: HashMap<String, u64>,
    /// arg name -> (bytes, shape, dtype, is_output).
    tensors: Vec<TensorArg>,
}

/// A scalar value -> the `value` attribute an `arith.constant` carries. Integer /
/// index scalars become `Attr::Int` (the constant handler binds `Scalar::I64`,
/// which every index consumer — `scf.for` bounds, `arith.muli`, access-tile index
/// operands — coerces exactly like a literal `arith.constant : index`); floats
/// become `Attr::Float`; bool becomes `Attr::Bool`.
fn scalar_value_attr(s: &Scalar) -> Attr {
    match s {
        Scalar::I32(v) => Attr::Int(*v as i64),
        Scalar::I64(v) => Attr::Int(*v),
        Scalar::F32(v) => Attr::Float(*v as f64),
        Scalar::Bool(b) => Attr::Bool(*b),
    }
}

/// The MLIR result-type spelling for a scalar (so the prepended `arith.constant`
/// reads `: index` / `: i32` / `: f16` like a hand-written literal). Index-typed
/// scalars (the common loop-bound / block-size case) keep `index`.
fn scalar_result_type(s: &Scalar) -> &'static str {
    match s {
        Scalar::I32(_) => "i32",
        Scalar::I64(_) => "index",
        Scalar::F32(_) => "f16",
        Scalar::Bool(_) => "i1",
    }
}

impl ResidentRunner {
    /// Build a runner for one example function. `module` is the parsed example
    /// program; `func` its function name. `tensors` are the pointer args (in order)
    /// and `scalars` the launch scalars to specialize. Tensor ids are assigned in
    /// the given tensor order (`t0, t1, ...`).
    pub fn new(
        module: &IRModule,
        func: &str,
        tensors: Vec<TensorArg>,
        scalars: Vec<ScalarArg>,
    ) -> Result<Self, String> {
        let original = module.get_function(func)?.clone();

        // The resident executor is an ALL-F16 path: it sizes every stick and binds
        // every pointer arg with the single model dtype (F16), and `set_sources`
        // re-encodes any non-F16 host bytes THROUGH f16. So a program with a
        // non-F16 tensor arg — an i64/i32 GATHER-INDEX tensor (indexed_add's
        // `index`, paged_attention's `block_tables`) or an f32 data tensor
        // (vector_add_dynamic) — would have its indices/data silently rounded to
        // f16 and read the wrong rows. Rather than DIVERGE silently, report it as
        // not-drivable through the resident path (it stays correct on the default
        // mixed-dtype `execute_function` CPU path). Honest, not faked.
        if let Some(t) = tensors.iter().find(|t| t.dtype != DType::F16) {
            return Err(format!(
                "resident path is all-F16; tensor arg {:?} is {:?} (non-F16 index/data \
                 tensors are not drivable here — the stick/base binding and set_sources \
                 both assume the model dtype). Runs correctly on the default CPU path.",
                t.name, t.dtype
            ));
        }

        // SPECIALIZE: drop every scalar arg from the signature and prepend an
        // `arith.constant` binding `%<arg>` to its value, so the body's uses
        // resolve to a literal (the form fusion expects).
        let scalar_names: HashSet<&str> = scalars.iter().map(|s| s.name.as_str()).collect();
        let mut new_args: Vec<(String, String)> = Vec::new();
        for (an, ty) in &original.arguments {
            let bare = an.trim_start_matches('%');
            if !scalar_names.contains(bare) {
                new_args.push((an.clone(), ty.clone()));
            }
        }
        let mut const_ops: Vec<Operation> = Vec::new();
        for s in &scalars {
            let res = format!("%{}", s.name);
            let op = Operation::new(Some(&res), "arith.constant", &[])
                .with_attr("value", scalar_value_attr(&s.value));
            let mut op = op;
            op.result_type = Some(scalar_result_type(&s.value).to_string());
            const_ops.push(op);
        }
        let mut operations = const_ops;
        operations.extend(original.operations.iter().cloned());

        let mut specialized = IRModule::default();
        specialized.add_function(IRFunction {
            name: original.name.clone(),
            arguments: new_args,
            operations,
            grid: original.grid,
            return_type: original.return_type.clone(),
        });

        // Assign a synthetic tensor id per pointer arg, in order, and build the
        // one-node ProgramSpec. Bindings use the REAL arg name; sources = inputs,
        // results = outputs.
        let mut arg_tid: HashMap<String, u64> = HashMap::new();
        let mut bindings: Vec<Binding> = Vec::new();
        let mut sources: HashSet<u64> = HashSet::new();
        let mut results: HashSet<u64> = HashSet::new();
        for (i, t) in tensors.iter().enumerate() {
            let tid = i as u64;
            arg_tid.insert(t.name.clone(), tid);
            bindings.push(Binding {
                arg: format!("%{}", t.name.trim_start_matches('%')),
                tensor: tid,
                is_output: t.is_output,
            });
            if t.is_output {
                results.insert(tid);
            } else {
                sources.insert(tid);
            }
        }
        // A tensor that is BOTH read and written (e.g. reduce_generic's arg0,
        // sdpa-style in-place) is bound once with is_output per the caller; mark it
        // a source too so the buffer is seeded with the caller bytes (not zeroed).
        for t in &tensors {
            // Always seed the output buffer with the caller's bytes (zeros for a
            // pure output, real data for an in-place arg): make it a source so the
            // resident executor marshals it from `args` rather than zeroing it.
            if t.is_output
                && let Some(&tid) = arg_tid.get(&t.name)
            {
                sources.insert(tid);
            }
        }

        let spec = ProgramSpec {
            nodes: vec![NodeSpec {
                func: func.to_string(),
                bindings,
            }],
            sources,
            results,
        };

        Ok(ResidentRunner {
            module: specialized,
            func: func.to_string(),
            spec,
            arg_tid,
            tensors,
        })
    }

    /// Run the program end-to-end through the PRODUCTION resident executor
    /// ([`crate::resident::ResidentExecutor::new_native`]) at the kernel's native
    /// grid, reading back every output tensor keyed by its ORIGINAL arg name.
    ///
    /// Native-grid (not `[1,1]`-fused) because the hand-written example kernels are
    /// SPMD-tiled: each compute-tile writes a DISJOINT output slice keyed off
    /// `ktdp.get_compute_tile_id`, so only the native grid computes the WHOLE
    /// output. The resident HBM, weight cache, per-segment seg-plan (K-loop GEMM
    /// reconstruction where recognizable) and per-op Metal offloads
    /// (`metal_gemm_or_blas`, fused map windows) all ride along — this is the real
    /// resident Metal path, not the gate-forced `execute_function` shortcut. Each
    /// tensor arg (incl. zero-seeded outputs and in-place args) is marshaled by
    /// `t<id>` as a SOURCE so its stick holds the caller bytes before the run.
    pub fn run(&self) -> Result<HashMap<String, Output>, String> {
        let mut owned: Vec<(String, Arg)> = Vec::new();
        for t in &self.tensors {
            let tid = self.arg_tid[&t.name];
            owned.push((
                format!("t{tid}"),
                Arg::TensorBytes {
                    data: t.data.clone(),
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                },
            ));
        }
        let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

        let out_keys: Vec<String> = self
            .tensors
            .iter()
            .filter(|t| t.is_output)
            .map(|t| format!("t{}", self.arg_tid[&t.name]))
            .collect();
        let out_refs: Vec<&str> = out_keys.iter().map(|s| s.as_str()).collect();

        let mut exec =
            crate::resident::ResidentExecutor::new_native(self.module.clone(), &self.spec)?;
        exec.set_sources(&args)?;
        let raw = exec.run(&out_refs)?;

        // Re-key from `t<id>` back to the original output arg name.
        let mut out: HashMap<String, Output> = HashMap::new();
        for t in &self.tensors {
            if !t.is_output {
                continue;
            }
            let key = format!("t{}", self.arg_tid[&t.name]);
            if let Some(o) = raw.get(&key) {
                out.insert(t.name.clone(), o.clone());
            }
        }
        Ok(out)
    }

    /// The specialized module + spec (for callers that want the resident executor
    /// directly).
    pub fn module(&self) -> &IRModule {
        &self.module
    }
    pub fn spec(&self) -> &ProgramSpec {
        &self.spec
    }
    pub fn func(&self) -> &str {
        &self.func
    }
}
