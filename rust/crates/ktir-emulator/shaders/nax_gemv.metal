#include <metal_stdlib>
using namespace metal;

inline float gemv_epilogue(float v, float ev, uint binop, uint act) {
    switch (binop) {
        case 1: v = v + ev; break;  case 2: v = v * ev; break;
        case 3: v = v - ev; break;  case 4: v = max(v, ev); break;
        case 5: v = min(v, ev); break;  default: break;
    }
    switch (act) {
        case 1: v = max(v, 0.0f); break;  case 2: v = tanh(v); break;
        case 3: v = exp(v); break;  case 4: v = 1.0f/(1.0f+exp(-v)); break;
        default: break;
    }
    return v;
}

// y[N] = x[K] . B  (B is [K,N] row-major, or [N,K] under KTIR_TRANSPOSE_B).
[[kernel]] void nax_gemv(
    device const float* a_in [[buffer(0)]],   // x: length K (the m=1 A row)
    device const float* b_in [[buffer(1)]],   // B: K x N, or N x K (transpose_b)
    device float* c_out      [[buffer(2)]],   // y: length N
    constant uint3& dims     [[buffer(3)]],   // (M=1, N, K)
    device const float* e_in [[buffer(4)]],   // length N epilogue operand (or dummy)
    constant uint2& epi      [[buffer(5)]],   // (binop, act) codes
    uint gid [[thread_position_in_grid]])
{
    const uint N = dims.y, K = dims.z;
    const uint j = gid;                         // output column this thread owns
    if (j >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        // f16 operands (NAX/simdgroup input precision), f32 accumulate.
        half xk = half(a_in[k]);
        half bkj = half(b_in[KTIR_TRANSPOSE_B ? (j * K + k) : (k * N + j)]);
        acc += float(xk) * float(bkj);
    }
    const uint binop = epi.x, act = epi.y;
    float ev = (binop != 0u) ? e_in[j] : 0.0f;
    c_out[j] = gemv_epilogue(acc, ev, binop, act);
}
