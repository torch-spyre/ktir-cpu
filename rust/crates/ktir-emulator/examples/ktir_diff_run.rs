// Copyright 2025 The Torch-Spyre Authors. Apache-2.0.
//
//! DIFFERENTIAL conformance CLI: read a batch of (program, function, inputs)
//! cases from a JSON request file (whose tensor args reference raw little-endian
//! bytes files), run `parse_module` + `execute_function` for each, and write each
//! result tensor's raw `dtype`-encoded bytes back out plus a JSON manifest.
//!
//! This is the Rust half of `tests/equiv/diff_py_vs_rust.py` — a head-to-head
//! Python-KTIRInterpreter ⟷ Rust-execute_function check. The driver writes the
//! request + input byte files, invokes this binary ONCE for the whole batch, and
//! diffs the output bytes against the Python interpreter's outputs.
//!
//! Run (built as an example so it can use the `serde_json` dev-dependency):
//!   cargo run --release --example ktir_diff_run -- <request.json>
//!
//! I/O FORMAT
//! ----------
//! Request JSON (one object):
//!   {
//!     "out_dir": "/abs/dir/for/outputs",
//!     "cases": [
//!       {
//!         "id": "vector_add/seed0",
//!         "program": "/abs/path/to/program.mlir",
//!         "function": "add_kernel",
//!         "args": [
//!           {"name":"x_ptr","kind":"tensor","dtype":"f16","shape":[4096],
//!            "bytes":"/abs/path/x_ptr.bin"},
//!           {"name":"BLOCK_SIZE","kind":"scalar","scalar_dtype":"i64","value":128},
//!           ...
//!         ],
//!         "outputs": ["output_ptr"]   // tensor arg names to read back
//!       }, ...
//!     ]
//!   }
//!
//! Tensor bytes files hold raw little-endian elements already encoded in the
//! arg's `dtype` (e.g. f16 = 2 bytes/elem). Scalars are inline.
//!
//! Response: for each (case, output) the raw `dtype`-encoded bytes are written to
//! `<out_dir>/<sanitized id>__<output>.bin`, and `<out_dir>/manifest.json` lists
//! every output's {case_id, name, dtype, shape, bytes_file}.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{
    Arg, HbmRead, HbmSeed, execute_function_outputs, execute_function_seeded,
};
use ktir_emulator::ir::Scalar;
use ktir_emulator::parser::parse_module;

use serde_json::{Value, json};

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("ktir_diff_run: {}", msg.as_ref());
    std::process::exit(1);
}

/// GPU-conformance engine selector. `KTIR_DIFF_ENGINE=gpu` makes this CLI prove
/// the Metal fast path actually ran: before each case it resets the per-op GPU
/// GEMM proof counter ([`metal::reset_gemm_or_blas_gpu_count`]), and after each
/// case it records `gpu_gemm_count` in the manifest. Combined with the gate
/// override (`KTIR_FORCE_GPU_GEMM=1`) the Python driver sets, this lets the
/// differential harness assert the tiled example matmuls dispatched to
/// NAX/simdgroup rather than the AMX fallback (a count of 0 on a compute-heavy
/// program is a FALSE pass and the driver flags it). Default (`cpu`) is the
/// existing bit-exact AMX/f32 path — unchanged.
fn gpu_engine_mode() -> bool {
    std::env::var("KTIR_DIFF_ENGINE").map(|s| s == "gpu") == Ok(true)
}

/// RESIDENT engine selector. `KTIR_DIFF_ENGINE=resident` runs every marshalled
/// case through the PRODUCTION resident/segmented Metal executor
/// ([`ktir_emulator::resident_runner::ResidentRunner`]) — the real serving path
/// (resident HBM + weight cache + per-segment seg-plan GEMM reconstruction +
/// per-op Metal offloads + fused map windows / decode attention), at the kernel's
/// native grid — instead of the per-op `execute_function` path the `gpu` engine
/// uses. Per case it resets the FULL offload proof ([`metal::reset_offload_proof`])
/// and records the per-offload breakdown (gemm-loop / gemm-or-blas / map-window) in
/// the manifest, so the driver can assert WHICH offload fired (a GEMM/attention
/// program with a zero offload total secretly ran all-CPU — a FALSE pass).
fn resident_engine_mode() -> bool {
    std::env::var("KTIR_DIFF_ENGINE").map(|s| s == "resident") == Ok(true)
}

/// Reset the Metal GPU GEMM proof counter (no-op without the `metal` feature).
fn reset_gpu_proof() {
    #[cfg(metal)]
    ktir_emulator::metal::reset_gemm_or_blas_gpu_count();
}

/// Read the Metal GPU GEMM proof counter (always 0 without the `metal` feature).
fn read_gpu_proof() -> usize {
    #[cfg(metal)]
    {
        ktir_emulator::metal::gemm_or_blas_gpu_count()
    }
    #[cfg(not(metal))]
    {
        0
    }
}

/// Reset EVERY Metal offload proof counter (resident engine; no-op off-metal).
fn reset_offload_proof() {
    #[cfg(metal)]
    ktir_emulator::metal::reset_offload_proof();
}

/// Snapshot the per-offload proof breakdown as a manifest JSON object. Off-metal
/// every count is 0 (the resident engine then legitimately reports an all-CPU run).
fn offload_proof_json() -> Value {
    #[cfg(metal)]
    {
        let p = ktir_emulator::metal::offload_proof();
        json!({
            "matmul_loop_gpu": p.matmul_loop_gpu,
            "matmul_loop_amx": p.matmul_loop_amx,
            "gemm_or_blas_gpu": p.gemm_or_blas_gpu,
            "map_region_gpu": p.map_region_gpu,
        })
    }
    #[cfg(not(metal))]
    {
        json!({
            "matmul_loop_gpu": 0,
            "matmul_loop_amx": 0,
            "gemm_or_blas_gpu": 0,
            "map_region_gpu": 0,
        })
    }
}

/// Run ONE marshalled case through the resident/segmented Metal executor. Splits
/// the case args into tensor args (with `is_output` from the case `outputs` list)
/// and scalar args (specialized to `arith.constant` inside the runner), runs at the
/// native grid, and records each requested output's raw bytes + the per-offload
/// proof. Errors are recorded per-case (never abort the batch) so the driver can
/// assert a matched failure / conformance gap against Python.
fn handle_resident_case(
    id: &str,
    function: &str,
    module: &ktir_emulator::ir::IRModule,
    case: &Value,
    out_dir: &str,
    manifest_outputs: &mut Vec<Value>,
    manifest_errors: &mut Vec<Value>,
) {
    use ktir_emulator::resident_runner::{ResidentRunner, ScalarArg, TensorArg};

    let outputs: Vec<String> = case["outputs"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|o| o.as_str().map(|s| s.trim_start_matches('%').to_string()))
        .collect();
    let is_out = |name: &str| outputs.iter().any(|o| o == name);

    let mut tensors: Vec<TensorArg> = Vec::new();
    let mut scalars: Vec<ScalarArg> = Vec::new();
    for a in case["args"].as_array().unwrap_or(&Vec::new()) {
        let name = a["name"]
            .as_str()
            .unwrap_or_else(|| die("arg missing name"))
            .trim_start_matches('%')
            .to_string();
        match a["kind"].as_str().unwrap_or("tensor") {
            "scalar" => scalars.push(ScalarArg {
                name,
                value: parse_scalar(a),
            }),
            "tensor" => {
                let dtype = dtype_of(
                    a["dtype"]
                        .as_str()
                        .unwrap_or_else(|| die("tensor arg missing dtype")),
                );
                let shape: Vec<usize> = a["shape"]
                    .as_array()
                    .unwrap_or_else(|| die("tensor arg missing shape"))
                    .iter()
                    .map(|d| d.as_u64().unwrap_or_else(|| die("shape dim not uint")) as usize)
                    .collect();
                let bytes_path = a["bytes"]
                    .as_str()
                    .unwrap_or_else(|| die("tensor arg missing bytes path"));
                let data = fs::read(bytes_path)
                    .unwrap_or_else(|e| die(format!("read bytes {bytes_path}: {e}")));
                let is_output = is_out(&name);
                tensors.push(TensorArg {
                    name,
                    data,
                    shape,
                    dtype,
                    is_output,
                });
            }
            other => die(format!("unsupported arg kind {other:?}")),
        }
    }

    let runner = match ResidentRunner::new(module, function, tensors, scalars) {
        Ok(r) => r,
        Err(e) => {
            manifest_errors
                .push(json!({ "case_id": id, "error": format!("resident build {function}: {e}") }));
            return;
        }
    };
    let result = match runner.run() {
        Ok(r) => r,
        Err(e) => {
            manifest_errors
                .push(json!({ "case_id": id, "error": format!("resident run {function}: {e}") }));
            return;
        }
    };

    for name in &outputs {
        let out = match result.get(name) {
            Some(o) => o,
            None => {
                manifest_errors.push(json!({
                    "case_id": id,
                    "error": format!("resident output {name:?} not produced"),
                }));
                continue;
            }
        };
        let fname = format!("{}__{}.bin", sanitize(id), sanitize(name));
        let fpath = Path::new(out_dir).join(&fname);
        fs::write(&fpath, &out.raw)
            .unwrap_or_else(|e| die(format!("write output {}: {e}", fpath.display())));
        manifest_outputs.push(json!({
            "case_id": id,
            "name": name,
            "dtype": out.dtype.as_str(),
            "shape": out.shape,
            "bytes_file": fname,
        }));
    }
}

fn dtype_of(s: &str) -> DType {
    DType::parse(s).unwrap_or_else(|e| die(format!("bad dtype {s:?}: {e}")))
}

fn parse_scalar(arg: &Value) -> Scalar {
    let sd = arg["scalar_dtype"].as_str().unwrap_or("i64");
    let v = &arg["value"];
    match sd {
        "i64" | "si64" | "index" => Scalar::I64(
            v.as_i64()
                .unwrap_or_else(|| die("scalar i64 value not an integer")),
        ),
        "i32" | "si32" => Scalar::I32(
            v.as_i64()
                .unwrap_or_else(|| die("scalar i32 value not an integer")) as i32,
        ),
        "f32" | "f16" | "float32" | "float16" => Scalar::F32(
            v.as_f64()
                .unwrap_or_else(|| die("scalar f32 value not a float")) as f32,
        ),
        "i1" | "bool" => Scalar::Bool(v.as_bool().unwrap_or(false)),
        other => die(format!("unsupported scalar_dtype {other:?}")),
    }
}

fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// Run one HBM-seeded case (no marshalled tensor args). Seeds every `hbm_seed`
/// region at its stick, binds the scalar args, runs `execute_function_seeded`,
/// and reads back every `hbm_read` region. On error, records a per-case error so
/// the driver asserts a matched failure against Python; never aborts the batch.
#[allow(clippy::too_many_arguments)]
fn handle_seeded_case(
    id: &str,
    function: &str,
    module: &ktir_emulator::ir::IRModule,
    case: &Value,
    out_dir: &str,
    manifest_outputs: &mut Vec<Value>,
    manifest_errors: &mut Vec<Value>,
) {
    // Scalars: bind by name exactly as the marshalled path.
    let mut scalars: Vec<(String, Scalar)> = Vec::new();
    if let Some(arr) = case["args"].as_array() {
        for a in arr {
            if a["kind"].as_str() == Some("scalar") {
                let name = a["name"]
                    .as_str()
                    .unwrap_or_else(|| die("scalar arg missing name"))
                    .to_string();
                scalars.push((name, parse_scalar(a)));
            }
        }
    }

    // Seed regions: each has an ELEMENT-index base (`elem`), a dtype, and a raw
    // bytes file (already dtype-encoded). The harness converts elem→byte address.
    let mut seeds: Vec<HbmSeed> = Vec::new();
    for s in case["hbm_seed"].as_array().unwrap_or(&Vec::new()) {
        let elem = s["elem"]
            .as_i64()
            .unwrap_or_else(|| die("hbm_seed missing elem"));
        let dtype = dtype_of(
            s["dtype"]
                .as_str()
                .unwrap_or_else(|| die("hbm_seed missing dtype")),
        );
        let bytes_path = s["bytes"]
            .as_str()
            .unwrap_or_else(|| die("hbm_seed missing bytes path"));
        let bytes =
            fs::read(bytes_path).unwrap_or_else(|e| die(format!("read seed {bytes_path}: {e}")));
        let lx_core = s["lx_core"].as_u64().map(|c| c as usize);
        let next_ptr = s["next_ptr"].as_i64();
        seeds.push(HbmSeed {
            elem,
            dtype,
            bytes,
            lx_core,
            next_ptr,
        });
    }

    // Read-back regions: name, ELEMENT-index base (`elem`), dtype, shape.
    let mut reads: Vec<HbmRead> = Vec::new();
    for r in case["hbm_read"].as_array().unwrap_or(&Vec::new()) {
        let name = r["name"]
            .as_str()
            .unwrap_or_else(|| die("hbm_read missing name"))
            .to_string();
        let elem = r["elem"]
            .as_i64()
            .unwrap_or_else(|| die("hbm_read missing elem"));
        let dtype = dtype_of(
            r["dtype"]
                .as_str()
                .unwrap_or_else(|| die("hbm_read missing dtype")),
        );
        let shape: Vec<usize> = r["shape"]
            .as_array()
            .unwrap_or_else(|| die("hbm_read missing shape"))
            .iter()
            .map(|d| d.as_u64().unwrap_or_else(|| die("shape dim not uint")) as usize)
            .collect();
        let n_elements: usize = shape.iter().product();
        reads.push(HbmRead {
            name,
            elem,
            n_elements,
            shape,
            dtype,
        });
    }

    let result = match execute_function_seeded(module, function, &scalars, &seeds, &reads) {
        Ok(r) => r,
        Err(e) => {
            manifest_errors.push(json!({
                "case_id": id,
                "error": format!("execute {function}: {e}"),
            }));
            return;
        }
    };

    for r in &reads {
        let out = match result.get(&r.name) {
            Some(o) => o,
            None => {
                manifest_errors.push(json!({
                    "case_id": id,
                    "error": format!("hbm_read {:?} not produced", r.name),
                }));
                continue;
            }
        };
        let fname = format!("{}__{}.bin", sanitize(id), sanitize(&r.name));
        let fpath = Path::new(out_dir).join(&fname);
        fs::write(&fpath, &out.raw)
            .unwrap_or_else(|e| die(format!("write output {}: {e}", fpath.display())));
        manifest_outputs.push(json!({
            "case_id": id,
            "name": r.name,
            "dtype": out.dtype.as_str(),
            "shape": out.shape,
            "bytes_file": fname,
        }));
    }
}

fn main() {
    let req_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| die("usage: ktir_diff_run <request.json>"));
    let req_text = fs::read_to_string(&req_path)
        .unwrap_or_else(|e| die(format!("read request {req_path}: {e}")));
    let req: Value =
        serde_json::from_str(&req_text).unwrap_or_else(|e| die(format!("parse request json: {e}")));

    let out_dir = req["out_dir"]
        .as_str()
        .unwrap_or_else(|| die("request missing out_dir"));
    fs::create_dir_all(out_dir).unwrap_or_else(|e| die(format!("mkdir {out_dir}: {e}")));

    let cases = req["cases"]
        .as_array()
        .unwrap_or_else(|| die("request missing cases array"));

    // Cache parsed modules by program path — the batch reruns the same few
    // programs across many seeds. A program that fails to PARSE is cached as the
    // parse error so every seed of it records a per-case error (not a hard abort).
    let mut module_cache: HashMap<String, Result<ktir_emulator::ir::IRModule, String>> =
        HashMap::new();
    let mut manifest_outputs: Vec<Value> = Vec::new();
    // Per-case execution/parse errors: a program Python runs but Rust cannot.
    // Recorded (NOT swallowed) so the driver reports it as a real conformance
    // FAIL — and one crashing program does not blank the whole batch.
    let mut manifest_errors: Vec<Value> = Vec::new();
    // GPU engine mode: per-case proof the Metal fast path ran (gpu_gemm_count).
    let gpu_mode = gpu_engine_mode();
    // RESIDENT engine mode: run through the resident/segmented Metal executor and
    // record the per-offload proof breakdown per case.
    let resident_mode = resident_engine_mode();
    let mut manifest_gpu: Vec<Value> = Vec::new();

    for case in cases {
        let id = case["id"]
            .as_str()
            .unwrap_or_else(|| die("case missing id"));

        // GPU/resident mode: zero the proof counter(s) before this case so the
        // post-run read (recorded below) reflects only THIS case's offloads.
        if resident_mode {
            reset_offload_proof();
        } else if gpu_mode {
            // Reset the GEMM proof AND the full per-offload proof breakdown so a
            // GPU-mode case can prove a non-GEMM Metal offload too — specifically
            // the fused MAP-window kernel (`map_region_gpu`) that the elementwise
            // non-F16 programs (vector_add_dynamic f32, indexed_add i64-gather)
            // dispatch when the GPU-mode env forces the map offload. The per-op
            // `execute_function` path the GPU engine uses bumps these same atomics.
            reset_gpu_proof();
            reset_offload_proof();
        }
        let program = case["program"]
            .as_str()
            .unwrap_or_else(|| die("case missing program"));
        let function = case["function"]
            .as_str()
            .unwrap_or_else(|| die("case missing function"));

        let module = module_cache.entry(program.to_string()).or_insert_with(|| {
            let src = fs::read_to_string(program)
                .unwrap_or_else(|e| die(format!("read program {program}: {e}")));
            parse_module(&src).map_err(|e| format!("parse {program}: {e}"))
        });
        let module = match module {
            Ok(m) => m,
            Err(e) => {
                manifest_errors.push(json!({ "case_id": id, "error": e }));
                continue;
            }
        };

        // HBM-SEEDED path: programs whose tensors live at hardcoded HBM stick
        // addresses (RFC fixtures, ring-reduce) carry "hbm_seed"/"hbm_read"
        // instead of marshalled ndarray tensor args. Seed both sides identically,
        // run, read back the named stick regions. Errors record a per-case error
        // so the driver can assert a MATCHED FAILURE against Python.
        if case.get("hbm_seed").is_some() {
            if resident_mode {
                // HBM-seeded fixtures (RFC indirect/distributed/ring-reduce) place
                // their tensors at hardcoded HBM stick addresses, NOT as marshalled
                // pointer args bound to a ProgramSpec — they cannot be expressed as a
                // marshalled-arg resident run. Report honestly (not faked).
                manifest_errors.push(json!({
                    "case_id": id,
                    "error": "resident engine: HBM-seeded fixture is not drivable through \
                              the marshalled-arg ProgramSpec path (tensors live at hardcoded \
                              HBM addresses, not function-arg pointers)",
                }));
                manifest_gpu.push(json!({ "case_id": id, "offload_proof": offload_proof_json() }));
                continue;
            }
            handle_seeded_case(
                id,
                function,
                module,
                case,
                out_dir,
                &mut manifest_outputs,
                &mut manifest_errors,
            );
            if gpu_mode {
                manifest_gpu.push(json!({ "case_id": id, "gpu_gemm_count": read_gpu_proof() }));
            }
            continue;
        }

        // RESIDENT engine: run this marshalled case through the production
        // resident/segmented Metal executor (native grid + offloads), and record the
        // per-offload proof breakdown.
        if resident_mode {
            handle_resident_case(
                id,
                function,
                module,
                case,
                out_dir,
                &mut manifest_outputs,
                &mut manifest_errors,
            );
            manifest_gpu.push(json!({ "case_id": id, "offload_proof": offload_proof_json() }));
            continue;
        }

        // Build args. Keep owned Strings/Args alive in a Vec, then borrow for the call.
        let arg_specs = case["args"]
            .as_array()
            .unwrap_or_else(|| die("case missing args array"));
        let mut owned: Vec<(String, Arg)> = Vec::with_capacity(arg_specs.len());
        for a in arg_specs {
            let name = a["name"]
                .as_str()
                .unwrap_or_else(|| die("arg missing name"))
                .to_string();
            let kind = a["kind"].as_str().unwrap_or("tensor");
            let arg = match kind {
                "scalar" => Arg::Scalar(parse_scalar(a)),
                "tensor" => {
                    let dtype = dtype_of(
                        a["dtype"]
                            .as_str()
                            .unwrap_or_else(|| die("tensor arg missing dtype")),
                    );
                    let shape: Vec<usize> = a["shape"]
                        .as_array()
                        .unwrap_or_else(|| die("tensor arg missing shape"))
                        .iter()
                        .map(|d| d.as_u64().unwrap_or_else(|| die("shape dim not uint")) as usize)
                        .collect();
                    let bytes_path = a["bytes"]
                        .as_str()
                        .unwrap_or_else(|| die("tensor arg missing bytes path"));
                    let data = fs::read(bytes_path)
                        .unwrap_or_else(|e| die(format!("read bytes {bytes_path}: {e}")));
                    Arg::TensorBytes { data, shape, dtype }
                }
                other => die(format!("unsupported arg kind {other:?}")),
            };
            owned.push((name, arg));
        }
        let args: Vec<(&str, Arg)> = owned.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

        let outputs: Vec<String> = case["outputs"]
            .as_array()
            .unwrap_or_else(|| die("case missing outputs array"))
            .iter()
            .map(|o| {
                o.as_str()
                    .unwrap_or_else(|| die("output not a string"))
                    .to_string()
            })
            .collect();
        let out_refs: Vec<&str> = outputs.iter().map(|s| s.as_str()).collect();

        let result = match execute_function_outputs(module, function, &args, &out_refs) {
            Ok(r) => r,
            Err(e) => {
                manifest_errors.push(json!({
                    "case_id": id,
                    "error": format!("execute {function}: {e}"),
                }));
                continue;
            }
        };
        // GPU proof: read the per-op Metal GEMM counter accumulated by THIS case's
        // `execute_function_outputs` (the per-tile `linalg.matmul` path). Recorded
        // so the driver can assert a compute-heavy program actually hit the GPU.
        if gpu_mode {
            manifest_gpu.push(json!({
                "case_id": id,
                "gpu_gemm_count": read_gpu_proof(),
                "offload_proof": offload_proof_json(),
            }));
        }

        for name in &outputs {
            let key = name.trim_start_matches('%');
            let out = match result.get(key) {
                Some(o) => o,
                None => {
                    manifest_errors.push(json!({
                        "case_id": id,
                        "error": format!(
                            "output {name:?} not returned (keys: {:?})",
                            result.keys().collect::<Vec<_>>()
                        ),
                    }));
                    continue;
                }
            };
            let fname = format!("{}__{}.bin", sanitize(id), sanitize(name));
            let fpath = Path::new(out_dir).join(&fname);
            fs::write(&fpath, &out.raw)
                .unwrap_or_else(|e| die(format!("write output {}: {e}", fpath.display())));
            manifest_outputs.push(json!({
                "case_id": id,
                "name": name,
                "dtype": out.dtype.as_str(),
                "shape": out.shape,
                "bytes_file": fname,
            }));
        }
    }

    let manifest = json!({
        "outputs": manifest_outputs,
        "errors": manifest_errors,
        "gpu": manifest_gpu,
    });
    let mpath = Path::new(out_dir).join("manifest.json");
    fs::write(&mpath, serde_json::to_string_pretty(&manifest).unwrap())
        .unwrap_or_else(|e| die(format!("write manifest {}: {e}", mpath.display())));
    eprintln!(
        "ktir_diff_run: wrote {} output(s) for {} case(s) to {out_dir}",
        manifest["outputs"].as_array().unwrap().len(),
        cases.len(),
    );
}
