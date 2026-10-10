// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Row-major single-precision GEMM, with an optional BLAS backend.
//!
//! `sgemm_rowmajor(m, k, n, a, b)` computes `C = A·B` for row-major `A` (m×k)
//! and `B` (k×n), returning a flat row-major `C` (m×n).
//!
//! - **macOS:** dispatches to Apple's `cblas_sgemm` (Accelerate, AMX-backed) by
//!   default — Accelerate ships with the OS, so it's on with no feature flag.
//! - **Linux / other:** a portable naive triple loop by default; enable a
//!   provider feature (`openblas-system` / `mkl` / `blis` / `openblas`) to route
//!   through that library's `cblas_sgemm` instead.
//!
//! BLAS is deterministic and — since NumPy's matmul is itself BLAS-backed —
//! tends to *tighten* parity with the reference. Both paths take the same
//! flat-`Vec<f32>` tile storage, so `linalg` matmul just calls `sgemm_rowmajor`.

/// `C(m×n) = A(m×k) · B(k×n)`, all row-major and contiguous. Naive loop —
/// the cross-platform default (and the parity oracle for the BLAS path).
#[cfg(not(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
)))]
pub fn sgemm_rowmajor(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    naive_sgemm(m, k, n, a, b)
}

/// `C(m×n) = A(m×k) · B(k×n)` via the linked BLAS `cblas_sgemm` (Accelerate on
/// macOS, or the selected provider — the cblas ABI is identical across them).
#[cfg(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
))]
pub fn sgemm_rowmajor(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
    let mut c = vec![0.0f32; m * n];
    // SAFETY: a has m*k elements, b has k*n, c has m*n; leading dimensions match
    // the row-major contiguous layout (lda=k, ldb=n, ldc=n). All non-negative.
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

/// Portable reference GEMM — always available (also the oracle for the
/// accelerate path's parity test).
pub fn naive_sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// `C(m×n) = A(m×k) · B(n×k)ᵀ` — the **transpose-B** GEMM, all row-major. `B` is
/// stored `[n, k]` (the on-disk PyTorch `Linear` `[out, in]` layout); the
/// contraction is over `k`, the LAST axis of BOTH operands, so each output is a
/// dot product of an `A` row with a `B` row — both contiguous. This is `xWᵀ` read
/// directly, with **no transpose of the weight data**. Routes to `cblas_sgemm`
/// with `transB` (native, free) where available, else a naive loop.
#[cfg(not(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
)))]
pub fn sgemm_rowmajor_bt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    naive_sgemm_bt(m, k, n, a, b)
}

/// `C(m×n) = A(m×k) · B(n×k)ᵀ` via `cblas_sgemm` with `transB = CblasTrans`.
#[cfg(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
))]
pub fn sgemm_rowmajor_bt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
    let mut c = vec![0.0f32; m * n];
    // SAFETY: a has m*k, b has n*k, c has m*n. B is transposed (op = CblasTrans),
    // stored row-major [n,k] so its leading dimension is k. lda=k, ldb=k, ldc=n —
    // all non-negative and matching the buffers.
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

/// Portable reference for the transpose-B GEMM (oracle for the cblas path).
pub fn naive_sgemm_bt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// ===========================================================================
// GEMV — the matrix-VECTOR specialization for the M=1 decode matmul.
//
// `sgemm_rowmajor` with `m == 1` is a single row-vector times a matrix, i.e. a
// GEMV. The tiled GEMM (and the NAX `matmul2d`) is built for M ≥ 8 and at M=1
// wastes ~94% of its 16-row tiles; the BLAS `cblas_sgemv` (Accelerate=AMX on
// macOS, OpenBLAS on Linux) is the purpose-built routine. These two functions
// are the `m == 1` slices of `sgemm_rowmajor` / `sgemm_rowmajor_bt`, so they
// compute bit-identical math — just through the level-2 routine. Same cfg-gating
// + naive fallback as the GEMM pair above.
// ===========================================================================

/// `y(n) = a(k)ᵀ · B(k×n)` — the `m == 1` case of [`sgemm_rowmajor`]: one row
/// vector `a` of length `k` times row-major `B` (k×n), returning `y` of length
/// `n`. Naive loop — the cross-platform default (and the parity oracle for the
/// BLAS path).
#[cfg(not(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
)))]
pub fn sgemv_rowmajor(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    naive_sgemv(k, n, a, b)
}

/// `y(n) = a(k)ᵀ · B(k×n)` via `cblas_sgemv`. `B` is row-major `[k, n]`, so to
/// form `Bᵀ·a` (length `n`) we ask cblas to transpose the m=k × n=n matrix:
/// `cblas_sgemv(RowMajor, CblasTrans, k, n, 1, B, lda=n, a, 1, 0, y, 1)`.
#[cfg(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
))]
pub fn sgemv_rowmajor(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemv};
    let mut y = vec![0.0f32; n];
    // SAFETY: B is row-major [k, n] (lda = n), a has k elems, y has n. With
    // CblasTrans the routine computes y = Bᵀ·a (the [k,n]→[n] contraction over
    // the leading axis), matching sgemm_rowmajor's m=1 row. All dims non-negative.
    unsafe {
        cblas_sgemv(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasTrans,
            k as i32,
            n as i32,
            1.0,
            b.as_ptr(),
            n as i32,
            a.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
    y
}

/// Portable reference GEMV (oracle for the cblas path). `y[j] = Σ_k a[k]·B[k,n]`.
pub fn naive_sgemv(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for (kk, &av) in a.iter().enumerate().take(k) {
        let row = &b[kk * n..kk * n + n];
        for (yj, &bv) in y.iter_mut().zip(row) {
            *yj += av * bv;
        }
    }
    y
}

/// `y(n) = a(k) · B(n×k)ᵀ` — the **transpose-B** GEMV, the `m == 1` case of
/// [`sgemm_rowmajor_bt`]. `B` is stored `[n, k]` (the on-disk PyTorch `Linear`
/// `[out, in]` layout); the contraction is over `k`, the last axis of both, so
/// each output `y[j]` is the dot of `a` with `B`'s row `j` (both contiguous) —
/// `aWᵀ` read directly, no weight transpose. Naive loop — portable default.
#[cfg(not(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
)))]
pub fn sgemv_rowmajor_bt(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    naive_sgemv_bt(k, n, a, b)
}

/// `y(n) = a(k) · B(n×k)ᵀ` via `cblas_sgemv`. `B` is row-major `[n, k]`, so
/// `B·a` (length `n`) is the NON-transposed product of the m=n × n=k matrix:
/// `cblas_sgemv(RowMajor, CblasNoTrans, n, k, 1, B, lda=k, a, 1, 0, y, 1)`.
#[cfg(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
))]
pub fn sgemv_rowmajor_bt(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemv};
    let mut y = vec![0.0f32; n];
    // SAFETY: B is row-major [n, k] (lda = k), a has k elems, y has n. CblasNoTrans
    // computes y = B·a — each y[j] is the dot of a with B's row j — exactly the
    // m=1 row of sgemm_rowmajor_bt. All dims non-negative.
    unsafe {
        cblas_sgemv(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            n as i32,
            k as i32,
            1.0,
            b.as_ptr(),
            k as i32,
            a.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
    y
}

/// Portable reference for the transpose-B GEMV (oracle for the cblas path).
/// `y[j] = Σ_k a[k]·B[j,k]`.
pub fn naive_sgemv_bt(k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for (j, yj) in y.iter_mut().enumerate() {
        let row = &b[j * k..j * k + k];
        let mut acc = 0.0f32;
        for (&av, &bv) in a.iter().take(k).zip(row) {
            acc += av * bv;
        }
        *yj = acc;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_matches_known_product() {
        // [[1,2,3],[4,5,6]] · [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(
            sgemm_rowmajor(2, 3, 2, &a, &b),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    #[test]
    fn sgemm_bt_equals_sgemm_on_transposed_b() {
        // A·Bᵀ where B is stored [n,k]. Build a contiguous [k,n] = Bᵀ and check
        // sgemm_rowmajor_bt(A, B[n,k]) == sgemm_rowmajor(A, Bᵀ[k,n]).
        let (m, k, n) = (3usize, 4usize, 5usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b_nk: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 - 2.0).collect(); // [n,k]
        // Bᵀ as contiguous [k,n]: bt[kk*n + j] = b_nk[j*k + kk].
        let mut bt = vec![0.0f32; k * n];
        for j in 0..n {
            for kk in 0..k {
                bt[kk * n + j] = b_nk[j * k + kk];
            }
        }
        assert_eq!(
            sgemm_rowmajor_bt(m, k, n, &a, &b_nk),
            sgemm_rowmajor(m, k, n, &a, &bt),
            "A·Bᵀ (transB over [n,k]) must equal A·(Bᵀ materialized [k,n])"
        );
    }

    #[test]
    fn naive_sgemm_bt_known_product() {
        // A=[[1,2,3],[4,5,6]] (2×3), B stored [n,k]=[[1,0,0],[0,1,0]] (2×3).
        // A·Bᵀ = [[1,2],[4,5]].
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(naive_sgemm_bt(2, 3, 2, &a, &b), vec![1.0, 2.0, 4.0, 5.0]);
    }

    /// With a BLAS backend active, `cblas_sgemm` must agree with the naive oracle.
    #[cfg(any(
        target_os = "macos",
        feature = "openblas",
        feature = "mkl",
        feature = "blis"
    ))]
    #[test]
    fn blas_matches_naive() {
        let m = 7;
        let k = 5;
        let n = 3;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 9) as f32 - 4.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_eq!(
            sgemm_rowmajor(m, k, n, &a, &b),
            naive_sgemm(m, k, n, &a, &b)
        );
    }

    // --- gemv (the m == 1 fast path) -------------------------------------

    #[test]
    fn sgemv_matches_known_product() {
        // a = [1,2,3] · B[3×2] = [[7,8],[9,10],[11,12]] = [58, 64].
        let a = [1.0, 2.0, 3.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(sgemv_rowmajor(3, 2, &a, &b), vec![58.0, 64.0]);
    }

    /// The whole point of the fast path: `sgemv_rowmajor` must equal the m=1 row
    /// of `sgemm_rowmajor` exactly (same math, level-2 routine). Mirrors
    /// `sgemm_matches_known_product`'s intent for the GEMV.
    #[test]
    fn sgemv_equals_sgemm_at_m1() {
        let (k, n) = (5usize, 3usize);
        let a: Vec<f32> = (0..k).map(|i| (i % 9) as f32 - 4.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_eq!(
            sgemv_rowmajor(k, n, &a, &b),
            sgemm_rowmajor(1, k, n, &a, &b),
            "sgemv must equal sgemm at m=1"
        );
    }

    #[test]
    fn naive_sgemv_bt_known_product() {
        // a = [1,2,3], B stored [n,k] = [[1,0,0],[0,1,0]] (2×3). a·Bᵀ = [1, 2].
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(naive_sgemv_bt(3, 2, &a, &b), vec![1.0, 2.0]);
    }

    /// `sgemv_rowmajor_bt` must equal the m=1 row of `sgemm_rowmajor_bt` exactly.
    #[test]
    fn sgemv_bt_equals_sgemm_bt_at_m1() {
        let (k, n) = (4usize, 5usize);
        let a: Vec<f32> = (0..k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b_nk: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 - 2.0).collect(); // [n,k]
        assert_eq!(
            sgemv_rowmajor_bt(k, n, &a, &b_nk),
            sgemm_rowmajor_bt(1, k, n, &a, &b_nk),
            "sgemv_bt must equal sgemm_bt at m=1"
        );
    }

    /// With a BLAS backend active, `cblas_sgemv` (both forms) must agree with the
    /// naive GEMV oracle.
    #[cfg(any(
        target_os = "macos",
        feature = "openblas",
        feature = "mkl",
        feature = "blis"
    ))]
    #[test]
    fn blas_gemv_matches_naive() {
        let (k, n) = (5usize, 3usize);
        let a: Vec<f32> = (0..k).map(|i| (i % 9) as f32 - 4.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_eq!(sgemv_rowmajor(k, n, &a, &b), naive_sgemv(k, n, &a, &b));
        // transpose-B: B stored [n,k].
        let b_nk: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_eq!(
            sgemv_rowmajor_bt(k, n, &a, &b_nk),
            naive_sgemv_bt(k, n, &a, &b_nk)
        );
    }
}
