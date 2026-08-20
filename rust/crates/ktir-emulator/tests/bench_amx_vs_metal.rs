// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! PER-KERNEL AMX-vs-Metal microbench for PERFORMANCE.md. Times the two matmul
//! primitives the interpreter dispatches between — Apple Accelerate
//! (`blas::sgemm_rowmajor`, the AMX coprocessor, f32) vs the M5 NAX tensor engine
//! (`metal::NaxGemm::run`, f16) — at the per-kernel bench's logical GEMM
//! shape (64×2048×8192) and at a larger prefill-scale shape where the GPU wins.
//!
//! WHY a direct-primitive bench (not `execute_function`): the matmul KTIR kernel
//! is a grid[2,16] SPMD body whose inner `linalg.matmul` tiles are 32×128@128×512
//! — far below the NAX gate (`NAX_MIN_BLOCKS`), so the interpreter routes every
//! tile to Accelerate. There is therefore no Metal path for that kernel *through
//! the interpreter*; to report a real AMX→Metal ratio we time the whole-GEMM
//! primitives at the kernel's logical shape directly. Mirrors the f16-quantized
//! inputs and warm-up-excluded methodology of `bench_py_vs_rust.rs`.
//!
//!   cargo test --release -p ktir-emulator --features metal --test bench_amx_vs_metal \
//!       -- --ignored --nocapture --test-threads=1

#[cfg(metal)]
use std::time::Instant;

#[cfg(metal)]
fn f16(x: f32) -> f32 {
    ktir_emulator::codec::f16_bits_to_f32(ktir_emulator::codec::f32_to_f16_bits(x))
}

/// Median of `iters` timings of `f`, one warm-up excluded. Seconds.
#[cfg(metal)]
fn median_secs<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f(); // warm-up (excluded)
    let mut times: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// AMX (Accelerate sgemm) vs Metal (NAX GEMM) at the per-kernel matmul shape and
/// a larger prefill-scale shape. Reports ms each, GFLOP/s, and the AMX→Metal
/// speedup. Skips cleanly if the NAX device is unavailable.
#[cfg(metal)]
#[test]
#[ignore = "per-kernel AMX-vs-Metal bench; --release --features metal --ignored --nocapture"]
fn matmul_amx_vs_metal() {
    use ktir_emulator::metal::NaxGemm;
    let gemm = match NaxGemm::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("NAX device unavailable ({e}) — skipping Metal matmul bench");
            return;
        }
    };

    // (m, k, n, iters): the per-kernel bench shape, then a prefill-scale shape.
    for (m, k, n, iters) in [
        (64usize, 2048usize, 8192usize, 20usize),
        (512, 4096, 4096, 20),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| f16((i % 13) as f32 * 0.01)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| f16((i % 11) as f32 * 0.01)).collect();
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        let amx = median_secs(iters, || {
            std::hint::black_box(ktir_emulator::blas::sgemm_rowmajor(m, k, n, &a, &b));
        });
        let metal = median_secs(iters, || {
            std::hint::black_box(gemm.run(m, k, n, &a, &b).expect("nax gemm"));
        });
        eprintln!(
            "matmul {m}x{k}x{n}: AMX(Accelerate) {:.2} ms ({:.0} GFLOP/s)  |  \
             Metal(NAX) {:.2} ms ({:.0} GFLOP/s)  |  AMX->Metal {:.2}x",
            amx * 1e3,
            flops / amx / 1e9,
            metal * 1e3,
            flops / metal / 1e9,
            amx / metal,
        );
    }
}
