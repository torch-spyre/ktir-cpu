// Phase-breakdown benchmark for the interpreter hot path.
// Run: cargo run --release --example bench
use ktir_emulator::codec;
use ktir_emulator::dtypes::DType;
use ktir_emulator::interpreter::{Arg, execute_function};
use ktir_emulator::ir::Scalar;
use ktir_emulator::memory::{STICK_BYTES, SpyreMemoryHierarchy};
use ktir_emulator::parser::parse_module;
use std::time::Instant;

fn us(t: Instant, iters: u32) -> f64 {
    t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0
}

fn main() {
    let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
    let module = parse_module(src).unwrap();
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let iters = 5000u32;

    let mk_args = || {
        [
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
                    data: vec![0.0; n],
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
        ]
    };

    // Whole pipeline (module pre-parsed — the realistic per-invocation cost).
    let t = Instant::now();
    for _ in 0..iters {
        let out = execute_function(&module, "add_kernel", &mk_args()).unwrap();
        std::hint::black_box(&out);
    }
    println!(
        "execute_function (pre-parsed)   : {:.1} us/iter",
        us(t, iters)
    );

    // Phase 1: memory hierarchy construction (32 cores: HBM + 32 LX).
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(SpyreMemoryHierarchy::new(32));
    }
    println!(
        "  SpyreMemoryHierarchy::new(32)  : {:.1} us/iter",
        us(t, iters)
    );

    // Phase 2: input marshalling (codec::encode + HBM write) for 3 tensors.
    let t = Instant::now();
    for _ in 0..iters {
        let mem = SpyreMemoryHierarchy::new(32);
        for data in [&x, &y] {
            let bytes = codec::encode(data, DType::F16);
            let hbm = mem.hbm.borrow_mut();
            let stick = hbm.allocate(bytes.len() as i64);
            hbm.write_bytes(stick * STICK_BYTES, &bytes);
        }
        std::hint::black_box(&mem);
    }
    println!(
        "  marshal 2x4096 f16 (+mem)      : {:.1} us/iter",
        us(t, iters)
    );

    // Phase 3: round_to_dtype micro-bench (f16), a 128-elem tile.
    let tile: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
    let t = Instant::now();
    for _ in 0..iters {
        let mut d = tile.clone();
        codec::round_to_dtype(&mut d, DType::F16);
        std::hint::black_box(&d);
    }
    println!(
        "  round_to_dtype 128 f16 (+clone): {:.3} us/iter",
        us(t, iters)
    );

    // Phase 4: codec round-trip for a 4096 f16 buffer (encode then decode).
    let t = Instant::now();
    for _ in 0..iters {
        let b = codec::encode(&x, DType::F16);
        std::hint::black_box(codec::decode(&b, n, DType::F16));
    }
    println!(
        "  codec enc+dec 4096 f16         : {:.1} us/iter",
        us(t, iters)
    );

    // Phase 5: the load slow-path trigger — affine enumerate of a contiguous
    // 0..127 box (what every vector_add load/store currently does) vs the O(2^n)
    // is_full vertex check that could bypass it.
    use ktir_emulator::parser_ast::parse_affine_set;
    let set = parse_affine_set("affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>").unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(set.enumerate(&[128], &[]));
    }
    println!(
        "  AffineSet::enumerate [128] box : {:.2} us/iter",
        us(t, iters)
    );
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(set.is_full(&[128]));
    }
    println!(
        "  AffineSet::is_full [128] box   : {:.3} us/iter",
        us(t, iters)
    );

    // Phase 6: pure scheduler/context overhead — drive 32 cores over an EMPTY
    // op list (builds 32 CoreContexts incl. the all_lx Vec clone, runs the
    // scheduler loop, no real ops). Isolates fixed multi-core setup cost.
    use ktir_emulator::comm_sched::execute_with_communication;
    use ktir_emulator::dialects::Dispatch;
    use ktir_emulator::env::GridExecutor;
    let grid = GridExecutor::new((32, 1, 1));
    let dispatch = Dispatch::new();
    let t = Instant::now();
    for _ in 0..iters {
        let mem = SpyreMemoryHierarchy::new(32);
        execute_with_communication(&grid, &mem, &[], &[], &dispatch, None, None).unwrap();
        std::hint::black_box(&mem);
    }
    println!(
        "  32-core scheduler, empty ops   : {:.1} us/iter",
        us(t, iters)
    );
}
