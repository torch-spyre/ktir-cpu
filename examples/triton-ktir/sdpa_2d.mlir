// SDPA (Scaled Dot-Product Attention) - 2D Tensors
// =================================================
// Translated from triton-examples/sdpa_2d.py
//
// Shapes: Q, K, V, Output = [seq_len=32, head_dim=64], dtype=f16
// Block sizes: BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, HEAD_DIM=64
// Grid: (num_query_blocks=1,)  [one program processes all 32 query rows]
//
// Algorithm (Flash Attention style):
//   1. Load Q [32, 64] from HBM
//   2. Load K [32, 64] from HBM (BLOCK_SIZE_N=64 >= seq_len=32; rows [32..63]
//      are out-of-bounds in the Triton kernel but are already absent in the
//      physical tensor, so no masking is needed at the load level)
//   3. Compute QK = Q @ K^T => [32, 32]
//   4. Scale:  QK = QK * (1/sqrt(64)) = QK * 0.125
//   5. Mask:   (all 32 key columns are valid; Triton's n_mask is trivially true)
//   6. Softmax (row-wise):
//        m_i  = max(QK, axis=1)          [32]
//        P    = exp(QK - m_i[:, None])   [32, 32]
//        l_i  = sum(P, axis=1)           [32]
//        P    = P / l_i[:, None]         [32, 32]
//   7. Load V [32, 64] from HBM
//   8. Compute acc = P @ V => [32, 64]
//   9. Store acc [32, 64] to HBM

module {
  func.func @sdpa_kernel_2d(
    %q_ptr      : index,   // base address of Q   [32, 64] f16
    %k_ptr      : index,   // base address of K   [32, 64] f16
    %v_ptr      : index,   // base address of V   [32, 64] f16
    %output_ptr : index    // base address of Out [32, 64] f16
  ) attributes {grid = [1]} {

    // -----------------------------------------------------------------------
    // Program id (axis=0): which query block.  With seq_len=32, BLOCK_SIZE_M=32
    // there is exactly one block, but the translation is general.
    // -----------------------------------------------------------------------
    %pid_m = ktdp.get_compute_tile_id : index

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    %c0       = arith.constant 0          : index
    %zero_f16 = arith.constant 0.0        : f16
    %scale    = arith.constant 1.25e-01   : f16   // 1/sqrt(64) = 0.125
    %neg_inf  = arith.constant 0xFF80 : f16   // -infinity (f16)

    // -----------------------------------------------------------------------
    // 1. Load Q block: [BLOCK_SIZE_M=32, HEAD_DIM=64]
    //    Q layout in HBM: [32, 64], strides [64, 1]
    // -----------------------------------------------------------------------
    %q_view = ktdp.construct_memory_view %q_ptr, sizes: [32, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %q_tile = ktdp.construct_access_tile %q_view[%pid_m, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>

    %q = ktdp.load %q_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>

    // -----------------------------------------------------------------------
    // 2. Load K block: [BLOCK_SIZE_N=32, HEAD_DIM=64]
    //    K physical shape in HBM: [32, 64] (seq_len=32 valid rows).
    //    The Triton kernel requests a [64, 64] block, but rows [32..63] are
    //    masked to -inf in the QK result.  Since the physical tensor only has
    //    32 rows, we load the full [32, 64] physical tensor; the resulting QK
    //    is [32, 32] and every column is already valid — no extra masking needed.
    // -----------------------------------------------------------------------
    %k_view = ktdp.construct_memory_view %k_ptr, sizes: [32, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %k_tile = ktdp.construct_access_tile %k_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>

    %k = ktdp.load %k_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>

    // -----------------------------------------------------------------------
    // 3. Compute QK = Q @ K^T
    //    q: [32, 64]   k: [32, 64]  => k_t: [64, 32] => qk: [32, 32]
    // -----------------------------------------------------------------------
    %k_t_init = tensor.empty() : tensor<64x32xf16>
    %k_t = linalg.transpose ins(%k : tensor<32x64xf16>)
                            outs(%k_t_init : tensor<64x32xf16>)
                            permutation = [1, 0]

    %qk_init = tensor.empty() : tensor<32x32xf16>
    %qk = linalg.matmul ins(%q, %k_t : tensor<32x64xf16>, tensor<64x32xf16>)
                        outs(%qk_init : tensor<32x32xf16>) -> tensor<32x32xf16>

    // -----------------------------------------------------------------------
    // 4. Scale: qk = qk * scale  (scale = 1/sqrt(64) = 0.125)
    //    Broadcast scalar scale over [32, 32] using arith.mulf on tensors.
    // -----------------------------------------------------------------------
    %scale_splat = tensor.splat %scale : tensor<32x32xf16>
    %qk_scaled   = arith.mulf %qk, %scale_splat : tensor<32x32xf16>

    // -----------------------------------------------------------------------
    // 5. Mask: all 32 key positions are valid (n < seq_len=32 always holds
    //    for the loaded block), so no masking operation is required.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // 6. Softmax (row-wise, numerically stable)
    //    a) m_i = row-wise max over [32, 32]  => [32]
    //    b) P   = exp(qk_scaled - m_i[:, None])  => [32, 32]
    //    c) l_i = row-wise sum of P  => [32]
    //    d) P   = P / l_i[:, None]   => [32, 32]
    // -----------------------------------------------------------------------

    // 6a. Row-wise max: m_i [32]
    %m_i_init   = tensor.empty() : tensor<32xf16>
    %m_i_neginf = linalg.fill ins(%neg_inf : f16) outs(%m_i_init : tensor<32xf16>) -> tensor<32xf16>

    %m_i = linalg.reduce { arith.maximumf }
             ins(%qk_scaled : tensor<32x32xf16>)
             outs(%m_i_neginf : tensor<32xf16>)
             dimensions = [1]

    // 6b. P = exp(qk_scaled - m_i[:, None])  [32, 32]
    //    Broadcast m_i [32] -> [32, 32], then subtract and exp.
    %m_i_bcast_init = tensor.empty() : tensor<32x32xf16>
    %m_i_bcast = linalg.broadcast ins(%m_i : tensor<32xf16>)
                                   outs(%m_i_bcast_init : tensor<32x32xf16>)
                                   dimensions = [1]
    %qk_shifted = arith.subf %qk_scaled, %m_i_bcast : tensor<32x32xf16>
    %p          = math.exp %qk_shifted : tensor<32x32xf16>

    // 6c. l_i = row-wise sum of P  [32]
    %l_i_init  = tensor.empty() : tensor<32xf16>
    %l_i_zeros = linalg.fill ins(%zero_f16 : f16) outs(%l_i_init : tensor<32xf16>) -> tensor<32xf16>

    %l_i = linalg.reduce { arith.addf }
             ins(%p : tensor<32x32xf16>)
             outs(%l_i_zeros : tensor<32xf16>)
             dimensions = [1]

    // 6d. P_norm = P / l_i[:, None]  [32, 32]
    %l_i_bcast_init = tensor.empty() : tensor<32x32xf16>
    %l_i_bcast = linalg.broadcast ins(%l_i : tensor<32xf16>)
                                   outs(%l_i_bcast_init : tensor<32x32xf16>)
                                   dimensions = [1]
    %p_norm    = arith.divf %p, %l_i_bcast : tensor<32x32xf16>

    // -----------------------------------------------------------------------
    // 7. Load V block: [BLOCK_SIZE_N=32, HEAD_DIM=64]
    //    Same as K: physical shape [32, 64], all rows valid.
    // -----------------------------------------------------------------------
    %v_view = ktdp.construct_memory_view %v_ptr, sizes: [32, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %v_tile = ktdp.construct_access_tile %v_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>

    %v = ktdp.load %v_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>

    // -----------------------------------------------------------------------
    // 8. Compute acc = P_norm @ V
    //    p_norm: [32, 32]   v: [32, 64]  => acc: [32, 64]
    // -----------------------------------------------------------------------
    %acc_init  = tensor.empty() : tensor<32x64xf16>
    %acc_zeros = linalg.fill ins(%zero_f16 : f16) outs(%acc_init : tensor<32x64xf16>) -> tensor<32x64xf16>

    %acc = linalg.matmul
             ins(%p_norm, %v : tensor<32x32xf16>, tensor<32x64xf16>)
             outs(%acc_zeros : tensor<32x64xf16>) -> tensor<32x64xf16>

    // -----------------------------------------------------------------------
    // 9. Store output [32, 64] back to HBM
    // -----------------------------------------------------------------------
    %output_view = ktdp.construct_memory_view %output_ptr, sizes: [32, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %output_tile = ktdp.construct_access_tile %output_view[%pid_m, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>

    ktdp.store %acc, %output_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>

    return
  }
}
