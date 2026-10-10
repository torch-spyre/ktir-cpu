#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr uint BK     = 16;
constant constexpr uint SG_M   = 32;    // simdgroup sub-block rows (2 tiles of 16)
constant constexpr uint SG_N   = 64;    // simdgroup sub-block cols (2 tiles of 32)
// SGS_M (simdgroup rows per threadgroup) is overridable via #define so the host
// can compile a SMALL-M variant (SGS_M=1 → TG_M=32) that computes only 32 output
// rows per block. The default (SGS_M=4 → TG_M=128) is the full-M kernel. At m≤32
// the full kernel pads 3/4 of every block with zero rows and runs their matmuls
// anyway (~4× wasted compute); the small-M variant skips that waste. Both are the
// SAME source — identical fragment math, edge guards, and epilogue — so the result
// is bit-identical; only the block height (#threadgroups dispatched) differs.
#ifndef KTIR_SGS_M
#define KTIR_SGS_M 4
#endif
// B (weight) operand element type. Default f32; KTIR_B_F16=1 reads B as `half`
// directly (the KTIR_F16_WEIGHTS path — half the bytes streamed). The NAX engine
// stages B into `half` either way, so the result is bit-identical; only the device
// load width changes. `KTIR_BT` is the host-side buffer's element type.
#ifndef KTIR_B_F16
#define KTIR_B_F16 0
#endif
#if KTIR_B_F16
typedef half  ktir_bt;
typedef half4 ktir_bt4;   // vector chunk type for the vectorized B loader
#else
typedef float  ktir_bt;
typedef float4 ktir_bt4;
#endif
constant constexpr uint SGS_M  = KTIR_SGS_M;  // simdgroup rows per threadgroup
constant constexpr uint SGS_N  = 4;     // simdgroup cols per threadgroup
constant constexpr uint TG_M   = SG_M * SGS_M;   // threadgroup block rows  = 128
constant constexpr uint TG_N   = SG_N * SGS_N;   // threadgroup block cols  = 256
constant constexpr uint TG_THREADS = SGS_M * SGS_N * 32;   // = 512

// Fused elementwise epilogue applied in the GEMM store: out = act(c BINOP e).
// binop: 0 none, 1 add, 2 mul, 3 sub, 4 max, 5 min.  act: 0 none, 1 relu,
// 2 tanh, 3 exp, 4 sigmoid. This is the matmul->elementwise fusion — the
// activation/bias runs in the same kernel as the matmul, with no readback.
inline float nax_epilogue(float v, float ev, uint binop, uint act) {
    switch (binop) {
        case 1: v = v + ev; break;
        case 2: v = v * ev; break;
        case 3: v = v - ev; break;
        case 4: v = max(v, ev); break;
        case 5: v = min(v, ev); break;
        default: break;
    }
    switch (act) {
        case 1: v = max(v, 0.0f); break;
        case 2: v = tanh(v); break;
        case 3: v = exp(v); break;
        case 4: v = 1.0f / (1.0f + exp(-v)); break;
        default: break;
    }
    return v;
}

[[kernel]] void nax_matmul(
    device const float* a_in [[buffer(0)]],   // M x K row-major
    device const ktir_bt* b_in [[buffer(1)]], // K x N row-major (f32 or half)
    device float* c_out      [[buffer(2)]],   // M x N row-major
    constant uint3& dims     [[buffer(3)]],   // (M, N, K)
    device const float* e_in [[buffer(4)]],   // M x N epilogue operand (or dummy)
    constant uint2& epi      [[buffer(5)]],   // (binop, act) codes
    uint3 tg  [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    const uint M = dims.x, N = dims.y, K = dims.z;
    // Batch index (grid z): each slice is an independent same-shape GEMM, so the
    // whole batch runs concurrently in one dispatch. tg.z = 0 for a single GEMM.
    a_in  += tg.z * M * K;
    b_in  += tg.z * K * N;
    c_out += tg.z * M * N;
    e_in  += tg.z * M * N;   // only dereferenced when binop != 0 (guarded below)
    const uint tm0 = tg.y * TG_M;          // threadgroup block base row
    const uint tn0 = tg.x * TG_N;          // threadgroup block base column
    const uint sm  = sgid / SGS_N;         // simdgroup's row slot
    const uint sn  = sgid % SGS_N;         // simdgroup's col slot
    const uint m0  = tm0 + sm * SG_M;      // this simdgroup's base row
    const uint n0  = tn0 + sn * SG_N;      // this simdgroup's base column
    const uint tid = sgid * 32u + lid;     // flat thread id in threadgroup

    // Double-buffered staging: while one panel feeds the matmuls, the next is
    // prefetched into the other half, so device-load latency overlaps compute.
    threadgroup half a_tg[2 * TG_M * BK];   // [2][TG_M, K-step]
    threadgroup half b_tg[2 * TG_N * BK];   // [2][TG_N, K-step] = transpose(B)

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto a0 = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto a1 = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto b0 = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto b1 = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto c00 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto c01 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto c10 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto c11 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();

    const short qid = (short)lid >> 2;
    const short fm = (qid & 4) | (((short)lid >> 1) & 3);
    const short fn = ((qid & 2) | ((short)lid & 1)) * 4;

    for (short e = 0; e < 8; ++e) {
        c00[e] = 0.0f; c00[8 + e] = 0.0f;  c01[e] = 0.0f; c01[8 + e] = 0.0f;
        c10[e] = 0.0f; c10[8 + e] = 0.0f;  c11[e] = 0.0f; c11[8 + e] = 0.0f;
    }

    const uint ar = sm * SG_M;   // this simdgroup's row offset into a_tg panel
    const uint bn = sn * SG_N;    // this simdgroup's col offset into b_tg panel
    const uint nk = (K + BK - 1u) / BK;  // number of K-steps

    // Stage one K-panel (rows of A, transposed cols of B) at K-offset `kc` into
    // buffer half `buf`. Zero-pads ragged M/N/K. (Macro so it inlines cleanly.)
    //
    // VARIANT A — vectorized loader. The threadgroup layout is unchanged
    // (ap[r*BK+c], bp[n*BK+c]) so the fragment math stays bit-identical; only the
    // FILL is rewritten. Both A and the transpose-B operand are contiguous along
    // the staged K axis (BK consecutive elements of a row are BK consecutive bytes
    // in device memory AND in the threadgroup tile), so each thread copies a
    // contiguous 4-wide run (BK=16 → 4 chunks/row). A FAST PATH (`gk0+4<=K` and the
    // row in-bounds) skips all per-element bounds and does one vector load + one
    // vector store; the ragged K-tail / M/N-edge falls to a scalar guard. The
    // non-transpose-B operand is strided by N along K (not contiguous), so it keeps
    // a scalar per-element fill — but with the div/mod replaced by precomputed
    // row/col arithmetic. Chunk width 4 maps 512 threads to exactly one A-chunk
    // each (TG_M*BK/4 = 512) and two B-chunks each (TG_N*BK/4 = 1024).
#define VW 4u                  /* vector chunk width along K (BK divisible by VW) */
#define CPR (BK / VW)          /* chunks per row = 4 */
#define STAGE_PANEL(buf, kc)                                                    \
    do {                                                                       \
        threadgroup half* ap = a_tg + (buf) * (TG_M * BK);                     \
        threadgroup half* bp = b_tg + (buf) * (TG_N * BK);                     \
        const uint _kc = (kc);                                                 \
        /* ---- A: M x K, contiguous along K ---- */                          \
        for (uint ch = tid; ch < TG_M * CPR; ch += TG_THREADS) {               \
            uint r  = ch / CPR;                                                 \
            uint cb = (ch - r * CPR) * VW;     /* col base within BK */         \
            uint gm = tm0 + r, gk0 = _kc + cb;                                  \
            uint aidx = gm * K + gk0;                                           \
            threadgroup half* dst = ap + r * BK + cb;                          \
            if (gm < M && gk0 + VW <= K && (aidx & (VW - 1u)) == 0u) {          \
                float4 v = *(device const float4*)(a_in + aidx);               \
                *(threadgroup half4*)dst = half4(v);                          \
            } else if (gm < M) {                                                \
                for (uint c = 0; c < VW; ++c)                                   \
                    dst[c] = (gk0 + c < K) ? half(a_in[gm * K + gk0 + c]) : half(0);\
            } else {                                                           \
                *(threadgroup half4*)dst = half4(0);                          \
            }                                                                  \
        }                                                                      \
        /* ---- B: transpose -> K-contiguous (fast); else N-strided (scalar) ---- */ \
        for (uint ch = tid; ch < TG_N * CPR; ch += TG_THREADS) {               \
            uint n  = ch / CPR;                                                 \
            uint cb = (ch - n * CPR) * VW;                                      \
            uint gn = tn0 + n, gk0 = _kc + cb;                                  \
            threadgroup half* dst = bp + n * BK + cb;                          \
            if (KTIR_TRANSPOSE_B) {                                             \
                uint bidx = gn * K + gk0;                                       \
                if (gn < N && gk0 + VW <= K && (bidx & (VW - 1u)) == 0u) {      \
                    ktir_bt4 v = *(device const ktir_bt4*)(b_in + bidx);       \
                    *(threadgroup half4*)dst = half4(v);                      \
                } else if (gn < N) {                                            \
                    for (uint c = 0; c < VW; ++c)                               \
                        dst[c] = (gk0 + c < K) ? half(b_in[gn * K + gk0 + c]) : half(0);\
                } else {                                                       \
                    *(threadgroup half4*)dst = half4(0);                      \
                }                                                              \
            } else {                                                           \
                for (uint c = 0; c < VW; ++c) {                                 \
                    uint gk = gk0 + c;                                          \
                    dst[c] = (gn < N && gk < K) ? half(b_in[gk * N + gn]) : half(0);\
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

    STAGE_PANEL(0u, 0u);                       // prime buffer 0 with K-step 0
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ki = 0; ki < nk; ++ki) {
        uint cur = ki & 1u;
        // Prefetch the next panel into the other buffer; its device loads are
        // in flight while this step's matmuls run.
        if (ki + 1u < nk) {
            STAGE_PANEL(cur ^ 1u, (ki + 1u) * BK);
        }
        // Load this simdgroup's fragments from the current buffer and accumulate.
        threadgroup half* ap = a_tg + cur * (TG_M * BK);
        threadgroup half* bp = b_tg + cur * (TG_N * BK);
        for (short e = 0; e < 8; ++e) {
            short r = fm + (e >> 2) * 8;
            short c = fn + (e % 4);
            a0[e] = ap[(ar + r) * BK + c];
            a1[e] = ap[(ar + r + 16) * BK + c];
            b0[e]     = bp[(bn + r) * BK + c];
            b0[8 + e] = bp[(bn + r + 16) * BK + c];
            b1[e]     = bp[(bn + r + 32) * BK + c];
            b1[8 + e] = bp[(bn + r + 48) * BK + c];
        }
        gemm_op.run(a0, b0, c00);
        gemm_op.run(a0, b1, c01);
        gemm_op.run(a1, b0, c10);
        gemm_op.run(a1, b1, c11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef STAGE_PANEL
#undef VW
#undef CPR

    // Store this simdgroup's 2x2 tile block (rows m0+{0,16}, cols n0+{0,16,32,48}),
    // applying the fused elementwise epilogue out = act(c BINOP e) per element.
    const uint binop = epi.x, act = epi.y;
#define EPI_STORE(rr, cc, cval)                                                \
    do {                                                                       \
        if ((rr) < M && (cc) < N) {                                            \
            float ev = (binop != 0u) ? e_in[(rr) * N + (cc)] : 0.0f;           \
            c_out[(rr) * N + (cc)] = nax_epilogue((cval), ev, binop, act);     \
        }                                                                      \
    } while (0)
    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        uint r0 = m0 + (uint)r;
        uint r1 = r0 + 16u;
        uint c0a = n0 + (uint)c;          uint c0b = c0a + 16u;   // tj=0 -> cols 0..31
        uint c1a = n0 + 32u + (uint)c;    uint c1b = c1a + 16u;   // tj=1 -> cols 32..63
        EPI_STORE(r0, c0a, c00[e]);
        EPI_STORE(r0, c0b, c00[8 + e]);
        EPI_STORE(r0, c1a, c01[e]);
        EPI_STORE(r0, c1b, c01[8 + e]);
        EPI_STORE(r1, c0a, c10[e]);
        EPI_STORE(r1, c0b, c10[8 + e]);
        EPI_STORE(r1, c1a, c11[e]);
        EPI_STORE(r1, c1b, c11[8 + e]);
    }
#undef EPI_STORE
}
