//! Microbench: the per-dispatch round-trip floor of one synchronous GPU GEMM.
//!
//! The resident prefill pass spends ~225 ms in 178 serial `commit + waitUntilCompleted`
//! GEMM dispatches (~1.26 ms each) while the actual matmul compute for a 1B model at
//! m=32 is only ~10-15 ms. This isolates how much of that ~1.26 ms is FIXED per-dispatch
//! cost (command-buffer create + encode + host↔GPU round-trip) vs GEMM compute, by
//! timing `NaxGemm::run` across sizes from trivial to llama-sized at the same m=32.
//!
//! If a trivial GEMM and a down-proj-sized GEMM both cost ~the same, the pass is
//! dispatch-bound and batching dispatches (one command buffer for N independent GEMMs)
//! is the lever. Run: `cargo test --release --test gpu_dispatch_floor -- --ignored --nocapture`.

#![cfg(metal)]

use ktir_emulator::metal::{Epilogue, NaxGemm};

/// Median ms of one resident `matmul_unified` over a fixed A and a B buffer that is
/// f16 (`b_f16=true`, the `KTIR_F16_WEIGHTS` weight buffer) or f32. Isolates the
/// weight-streaming delta: B is the big streamed operand, A/C stay f32.
fn time_unified(g: &NaxGemm, m: usize, k: usize, n: usize, b_f16: bool, iters: usize) -> f64 {
    let a = g.unified_from(&vec![0.5f32; m * k]).expect("a buf");
    let bvec = vec![0.25f32; k * n];
    let b = if b_f16 {
        g.unified_f16_from_f32(&bvec).expect("b f16")
    } else {
        g.unified_from(&bvec).expect("b f32")
    };
    let mut c = g.unified(m * n).expect("c buf");
    g.matmul_unified(m, k, n, &a, &b, &mut c, None, Epilogue::NONE, false)
        .expect("warm gemm");
    let mut ts: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        g.matmul_unified(m, k, n, &a, &b, &mut c, None, Epilogue::NONE, false)
            .expect("timed gemm");
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ts[ts.len() / 2]
}

fn time_gemm(g: &NaxGemm, m: usize, k: usize, n: usize, iters: usize) -> f64 {
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    // Warm: compile/alloc/first-dispatch excluded.
    let _ = g.run(m, k, n, &a, &b).expect("warm gemm");
    let mut ts: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        let _ = g.run(m, k, n, &a, &b).expect("timed gemm");
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    ts[ts.len() / 2]
}

/// Decides the AOT design: (1) how long the runtime JIT compile of all kernel
/// variants takes (what AOT would eliminate from cold start), and (2) how the
/// pre-M5 simdgroup kernel compares to NAX on the same M5 (whether a pre-M5 GPU
/// path is worth tuning vs falling back to AMX).
#[test]
#[ignore = "JIT-compile cost + NAX-vs-simdgroup throughput; run --release --ignored --nocapture"]
fn nax_compile_and_tier_probe() {
    let iters = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    // (1) JIT compile cost: full NaxGemm::new() (all ~10 pipeline variants).
    let t = std::time::Instant::now();
    let Ok(nax) = NaxGemm::new() else {
        eprintln!("no NaxGemm (no Metal device?) — skipping");
        return;
    };
    let nax_compile_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let simd = NaxGemm::new_simdgroup();
    let simd_compile_ms = t.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "\n[compile] NaxGemm::new() (NAX) JIT = {nax_compile_ms:.1} ms; new_simdgroup() = {simd_compile_ms:.1} ms"
    );

    // (2) Throughput: NAX vs simdgroup (forced on this M5) on llama MLP shapes, m=32.
    let cases = [(2048usize, 2048usize), (2048, 8192), (8192, 2048)];
    eprintln!("[tier] m=32, median of {iters}  (NAX vs forced-simdgroup):");
    for (k, n) in cases {
        let gflop = 2.0 * (32 * k * n) as f64 / 1e9;
        let nax_ms = time_gemm(&nax, 32, k, n, iters);
        let line = match &simd {
            Ok(s) => {
                let s_ms = time_gemm(s, 32, k, n, iters);
                format!(
                    "{k}x{n}: NAX {nax_ms:6.3}ms ({:7.0} GFLOP/s) | simd {s_ms:6.3}ms ({:7.0} GFLOP/s) | {:.2}x",
                    gflop / (nax_ms / 1e3),
                    gflop / (s_ms / 1e3),
                    s_ms / nax_ms
                )
            }
            Err(e) => format!(
                "{k}x{n}: NAX {nax_ms:6.3}ms ({:7.0} GFLOP/s) | simd unavailable: {e}",
                gflop / (nax_ms / 1e3)
            ),
        };
        eprintln!("  {line}");
    }
    eprintln!();
}

#[test]
#[ignore = "GPU per-dispatch floor microbench; run --release --ignored --nocapture"]
fn gpu_dispatch_floor() {
    let Ok(g) = NaxGemm::new() else {
        eprintln!("no NaxGemm (no Metal device?) — skipping");
        return;
    };
    let iters = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    // m=32 throughout (llama prefill token batch). (label, k, n, ~GFLOP for 1 GEMM).
    let cases = [
        ("trivial   1x1", 1usize, 1usize),
        ("tiny    64x64", 64, 64),
        ("qkv  2048x2048", 2048, 2048),
        ("gate 2048x8192", 2048, 8192),
        ("down 8192x2048", 8192, 2048),
    ];
    // Probe several M to expose the 128-tall-block padding waste: if m=32 and
    // m=128 cost about the SAME for a fixed (k,n), m=32 is computing 4× padded
    // rows — the underutilization the small-m kernel removes.
    for m in [32usize, 64, 128] {
        eprintln!("\n[gpu-dispatch-floor] m={m}, median of {iters} iters:");
        for (label, k, n) in cases {
            let ms = time_gemm(&g, m, k, n, iters);
            let gflop = 2.0 * (m * k * n) as f64 / 1e9;
            let achieved = gflop / (ms / 1e3);
            eprintln!("  {label:>16}  {ms:6.3} ms   ({gflop:6.3} GFLOP, {achieved:7.1} GFLOP/s)");
        }
    }
    eprintln!(
        "\n  Interpretation: if 'trivial' ≈ 'down', the cost is FIXED per-dispatch \
         overhead, not compute → dispatch-bound; batch independent GEMMs per command buffer.\n"
    );

    // f32-B vs f16-B WEIGHT streaming, in isolation (resident matmul_unified, A/C
    // f32, only B's element width changes). The MLP weights are streaming-bound, so
    // halving B's bytes should roughly halve these wide-N GEMMs.
    eprintln!("[f16-weight] m=32, median of {iters} iters  (B f32 vs B f16):");
    for (label, k, n) in cases {
        let f32b = time_unified(&g, 32, k, n, false, iters);
        let f16b = time_unified(&g, 32, k, n, true, iters);
        let speedup = f32b / f16b;
        eprintln!("  {label:>16}  f32 {f32b:6.3} ms  f16 {f16b:6.3} ms  ({speedup:.2}x)");
    }
}
