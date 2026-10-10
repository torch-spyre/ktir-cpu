// End-to-end execution: parse a real KTIR kernel and run it through the full
// driver (HBM marshalling -> multi-core execution -> read-back), checking the
// computed output. This is the parity checkpoint — the interpreter actually
// runs a kernel and produces correct tensor results.

use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, Output, execute_function, execute_function_with_latency};
use ktir_emulator::latency::HardwareConfig;
use ktir_emulator::parser::parse_module;

#[test]
fn vector_add_executes_end_to_end() {
    let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");

    // 32 cores x BLOCK_SIZE=128 = 4096 elements, matching the kernel's views.
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let out = vec![0.0f32; n];

    let args = [
        (
            "x_ptr",
            Arg::Tensor {
                data: x.clone(),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "y_ptr",
            Arg::Tensor {
                data: y.clone(),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::Tensor {
                data: out,
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];

    let outputs = execute_function(&module, "add_kernel", &args).expect("run add_kernel");
    let Output { data, .. } = outputs.get("output_ptr").expect("output_ptr present");

    let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_eq!(data.len(), n);
    assert_eq!(*data, expected, "elementwise x + y mismatch");
}

// RUST-ONLY (not a port of a Python test): the typed-bytes input path
// (`Arg::TensorBytes`) feeds pre-encoded f16 straight to HBM and must produce
// the identical result to the f32 `Arg::Tensor` path (which narrows on the way
// in) — proving the f32 round-trip is avoidable with no behavior change.
#[test]
fn tensor_bytes_input_matches_f32_path() {
    let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();

    // Pre-encode the inputs to f16 bytes (what a real f16 host runner holds).
    let enc = |v: &[f32]| ktir_emulator::codec::encode(v, DType::F16);
    let args = [
        (
            "x_ptr",
            Arg::TensorBytes {
                data: enc(&x),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "y_ptr",
            Arg::TensorBytes {
                data: enc(&y),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::TensorBytes {
                data: vec![0u8; n * 2],
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];
    let outputs = execute_function(&module, "add_kernel", &args).expect("run add_kernel");
    let Output { data, .. } = outputs.get("output_ptr").expect("output_ptr present");

    let expected: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_eq!(
        *data, expected,
        "TensorBytes f16 path must match the f32 path"
    );
}

// RUST-ONLY: the bf16 ingest path (`Arg::TensorBf16`) narrows bf16 host bytes to
// the f16 HBM stick in one fused pass. Spyre is f16-only, so the result must
// equal feeding the same values pre-narrowed to f16 via `Arg::TensorBytes` — i.e.
// ktir-emulator owning the bf16->f16 narrow changes nothing vs the caller doing it.
#[test]
fn tensor_bf16_input_matches_f16_path() {
    let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");
    let n = 4096usize;
    // Values chosen to be bf16-exact (small ints / halves) so the comparison is
    // about the path, not bf16 rounding noise.
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.5).collect();

    // bf16 bytes = the high 16 bits of each f32.
    let to_bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let bf16_arg = |v: &[f32]| Arg::TensorBf16 {
        data: to_bf16(v),
        shape: vec![n],
    };
    let bf16_args = [
        ("x_ptr", bf16_arg(&x)),
        ("y_ptr", bf16_arg(&y)),
        (
            "output_ptr",
            Arg::TensorBf16 {
                data: vec![0u8; n * 2],
                shape: vec![n],
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];

    // Reference: the SAME values narrowed bf16->f16 on the host, fed as f16 bytes.
    let to_f16_via_bf16 = |v: &[f32]| ktir_emulator::codec::bf16_to_f16(&to_bf16(v), v.len());
    let f16_args = [
        (
            "x_ptr",
            Arg::TensorBytes {
                data: to_f16_via_bf16(&x),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "y_ptr",
            Arg::TensorBytes {
                data: to_f16_via_bf16(&y),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::TensorBytes {
                data: vec![0u8; n * 2],
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];

    let got = execute_function(&module, "add_kernel", &bf16_args).expect("run bf16");
    let want = execute_function(&module, "add_kernel", &f16_args).expect("run f16");
    assert_eq!(
        got["output_ptr"].raw, want["output_ptr"].raw,
        "bf16 ingest must equal host-narrowed f16 ingest (byte-identical HBM)"
    );
}

// RUST-ONLY (not a port of a Python test): the output side of the typed-bytes
// feature — `Output.raw` is the undecoded f16 HBM bytes, and `data == decode(raw)`.
// A typed host runner can thread `output.raw` straight into the next node's
// `Arg::TensorBytes` with no f16→f32→f16 round-trip. Demonstrated by feeding one
// kernel's raw output back as another's input.
#[test]
fn output_raw_bytes_thread_without_roundtrip() {
    let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();

    let enc = |v: &[f32]| ktir_emulator::codec::encode(v, DType::F16);
    let args = [
        (
            "x_ptr",
            Arg::TensorBytes {
                data: enc(&x),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "y_ptr",
            Arg::TensorBytes {
                data: enc(&y),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::TensorBytes {
                data: vec![0u8; n * 2],
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];
    let out = execute_function(&module, "add_kernel", &args).expect("run add_kernel");
    let o = out.get("output_ptr").expect("output_ptr present");

    // raw is the f16-encoded HBM bytes; data is its decode; sizes line up.
    assert_eq!(o.raw.len(), n * 2, "f16 raw bytes are 2 per element");
    assert_eq!(o.raw, enc(&o.data), "raw must equal encode(data)");

    // Thread the raw output back as a TensorBytes input — no widen/narrow.
    let args2 = [
        (
            "x_ptr",
            Arg::TensorBytes {
                data: o.raw.clone(),
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "y_ptr",
            Arg::TensorBytes {
                data: vec![0u8; n * 2],
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::TensorBytes {
                data: vec![0u8; n * 2],
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];
    let out2 = execute_function(&module, "add_kernel", &args2).expect("run add_kernel again");
    // x + 0 == x == previous (x + y).
    assert_eq!(out2.get("output_ptr").unwrap().data, o.data);
}

#[test]
fn vector_add_latency_report_is_populated() {
    let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).expect("parse vector_add");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let args = [
        (
            "x_ptr",
            Arg::Tensor {
                data: x,
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "y_ptr",
            Arg::Tensor {
                data: y,
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "output_ptr",
            Arg::Tensor {
                data: vec![0.0; n],
                shape: vec![n],
                dtype: DType::F16,
            },
        ),
        (
            "BLOCK_SIZE",
            Arg::Scalar(ktir_emulator::ir::Scalar::I64(128)),
        ),
    ];

    let (outputs, report) =
        execute_function_with_latency(&module, "add_kernel", &args, HardwareConfig::default())
            .expect("run with latency");

    // Correctness is unaffected by tracking.
    assert_eq!(outputs.get("output_ptr").unwrap().data.len(), n);

    // The kernel does 2 HBM loads + 1 HBM store per core across 32 cores, plus
    // an addf — so the report must show real memory and compute cost.
    assert!(
        report.kernel_cycles() > 0.0,
        "expected non-zero kernel cycles"
    );
    let summary = report.per_core_summary();
    assert_eq!(summary.len(), 32, "one row per core");
    let mem: f64 = summary.iter().map(|c| c.memory_cycles).sum();
    let compute: f64 = summary.iter().map(|c| c.compute_cycles).sum();
    assert!(mem > 0.0, "expected non-zero memory cycles, got {mem}");
    assert!(
        compute > 0.0,
        "expected non-zero compute cycles, got {compute}"
    );
}
