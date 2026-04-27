// Kernel: kernel_unified_attention_spyre_2d
// Translated from: triton-examples/triton_unified_attention_spyre.py
//
// Concrete parameters used in this translation
// (derived from triton-examples/test_kernel_unified_attention_2d.py):
//   num_tokens = 8
//   seq_len_total = 128              (total sequence length including context)
//   num_query_heads = 32
//   num_kv_heads = 8
//   num_queries_per_kv = 32 / 8 = 4 (GQA factor)
//   head_size = HEAD_SIZE = HEAD_SIZE_PADDED = 128
//   BLOCK_SIZE = TILE_SIZE = 16
//   BLOCK_Q = 2                      (tokens per pid0 step; targeting 32 cores)
//   BLOCK_M = num_queries_per_kv * BLOCK_Q = 4 * 2 = 8
//   num_blks = 64                    (total KV cache blocks)
//   max_num_blocks_per_seq = 16
//   blk_size = 16                    (tokens per KV-cache page)
//   num_seqs = 1
//   context_len = seq_len_total - num_tokens = 128 - 8 = 120
//
// Tensor shapes and strides:
//   query:        [8, 32, 128], strides [4096, 128, 1],    f16
//                   (num_tokens, num_query_heads, head_size)
//   key_cache:    [64, 16, 8, 128], strides [16384, 1024, 128, 1],  f16
//                   (num_blks, blk_size, num_kv_heads, head_size)
//   value_cache:  [64, 16, 8, 128], strides [16384, 1024, 128, 1],  f16
//                   (same layout as key_cache)
//   block_tables: [1, 16], strides [16, 1],  i32
//                   (num_seqs, max_blocks_per_seq)
//   output:       [8, 32, 128], strides [4096, 128, 1],    f16
//                   (same layout as query)
//
// Grid: [ceil(num_tokens/BLOCK_Q), num_kv_heads] = [4, 8]
//   dim 0 (pid0): query-block index  (BLOCK_Q=2 consecutive tokens per tile)
//   dim 1 (pid1): KV-head index
//
// Per-tile computation (BLOCK_M=8 query rows × 1 KV head):
//   1. Load Q tile:  query[cur_batch_start+pid0*2 : ..+2, pid1*4 : pid1*4+4, :]
//                    → (2, 4, 128) collapsed to (8, 128)  f16
//   2. Loop j = 0 .. num_tiles:
//        a. Indirect-load K tile via block_tables:
//               key_cache[block_tables[0, block_table_offset+j], :, pid1, :]
//               → (128, 16)  f16  (transposed for Q@K)
//        b. Indirect-load V tile via block_tables:
//               value_cache[block_tables[0, block_table_offset+j], :, pid1, :]
//               → (16, 128)  f16
//        c. S  = scale * (Q @ K)       (8,128)@(128,16) → (8,16) f32
//        d. Causal masking on S:
//               query_pos[row] = pid0*2 + row//4
//               query_abs_pos[row] = context_len + query_pos[row]
//               seq_offset[col]  = j*16 + col
//               mask: S[row,col] = -inf  when seq_offset[col] > query_abs_pos[row]
//        e. Running online-softmax update:
//               m_j   = max(M, rowmax(S))
//               P     = exp(S - m_j[:,None])   (8,16)
//               l_j   = rowsum(P)
//               alpha = exp(M - m_j)
//               acc   = acc * alpha[:,None] + P @ V
//               L     = L * alpha + l_j
//               M     = m_j
//   3. Epilogue: acc = acc / L[:,None]
//   4. Truncate acc (f32) → f16 and store to output

module {
  func.func @kernel_unified_attention_spyre_2d(
    %output_ptr            : index,
    %query_ptr             : index,
    %key_cache_ptr         : index,
    %value_cache_ptr       : index,
    %block_tables_ptr      : index,
    %cur_batch_start_index : index,   // first token index in global query array
    %block_table_offset    : index,   // column offset into block_tables row
    %num_tiles             : index,   // number of KV tiles to process
    %context_len           : index,   // context length = seq_len - cur_batch_query_len
    %scale                 : f32      // attention scale = 1/sqrt(head_size)
  ) attributes {grid = [4, 8]} {

    %pid0, %pid1 = ktdp.get_compute_tile_id : index, index   // dim 0: query-block, dim 1: KV-head
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %c2   = arith.constant 2 : index   // BLOCK_Q
    %c4   = arith.constant 4 : index   // num_queries_per_kv
    %c16  = arith.constant 16 : index  // TILE_SIZE

    // q_row_start = cur_batch_start_index + pid0 * BLOCK_Q  (BLOCK_Q=2)
    %pid0_x2     = arith.muli %pid0, %c2 : index
    %q_row_start = arith.addi %cur_batch_start_index, %pid0_x2 : index

    // pid1_x4 = pid1 * num_queries_per_kv  (=4)
    %pid1_x4 = arith.muli %pid1, %c4 : index

    // -----------------------------------------------------------------------
    // Memory views
    // -----------------------------------------------------------------------

    // query: [8, 32, 128]  strides [4096, 128, 1]
    %query_view = ktdp.construct_memory_view %query_ptr,
        sizes: [8, 32, 128], strides: [4096, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2) : (
          d0 >= 0, -d0 + 7   >= 0,
          d1 >= 0, -d1 + 31  >= 0,
          d2 >= 0, -d2 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x32x128xf16>

    // key_cache: [64, 16, 8, 128]  strides [16384, 1024, 128, 1]
    //   dims: (num_blks, blk_size, num_kv_heads, head_size)
    %key_cache_view = ktdp.construct_memory_view %key_cache_ptr,
        sizes: [64, 16, 8, 128], strides: [16384, 1024, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2, d3) : (
          d0 >= 0, -d0 + 63  >= 0,
          d1 >= 0, -d1 + 15  >= 0,
          d2 >= 0, -d2 + 7   >= 0,
          d3 >= 0, -d3 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x16x8x128xf16>

    // value_cache: [64, 16, 8, 128]  strides [16384, 1024, 128, 1]
    %value_cache_view = ktdp.construct_memory_view %value_cache_ptr,
        sizes: [64, 16, 8, 128], strides: [16384, 1024, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2, d3) : (
          d0 >= 0, -d0 + 63  >= 0,
          d1 >= 0, -d1 + 15  >= 0,
          d2 >= 0, -d2 + 7   >= 0,
          d3 >= 0, -d3 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x16x8x128xf16>

    // block_tables: [1, 16]  strides [16, 1]
    %block_tables_view = ktdp.construct_memory_view %block_tables_ptr,
        sizes: [1, 16], strides: [16, 1] {
      coordinate_set = affine_set<(d0, d1) : (
          d0 >= 0, -d0 + 0  >= 0,
          d1 >= 0, -d1 + 15 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x16xi32>

    // output: [8, 32, 128]  strides [4096, 128, 1]
    %output_view = ktdp.construct_memory_view %output_ptr,
        sizes: [8, 32, 128], strides: [4096, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2) : (
          d0 >= 0, -d0 + 7   >= 0,
          d1 >= 0, -d1 + 31  >= 0,
          d2 >= 0, -d2 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8x32x128xf16>

    // -----------------------------------------------------------------------
    // Load Q tile: query[q_row_start : q_row_start+2, pid1*4 : pid1*4+4, 0:128]
    //   Access tile shape: [2, 4, 128]
    //     dim0: 2 tokens  (BLOCK_Q=2)
    //     dim1: 4 query heads  (num_queries_per_kv=4)
    //     dim2: 128 features   (HEAD_SIZE=128)
    // -----------------------------------------------------------------------
    %q_access_tile = ktdp.construct_access_tile %query_view[%q_row_start, %pid1_x4, %c0] {
      access_tile_set = affine_set<(d0, d1, d2) : (
          d0 >= 0, -d0 + 1   >= 0,
          d1 >= 0, -d1 + 3   >= 0,
          d2 >= 0, -d2 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    } : memref<8x32x128xf16> -> !ktdp.access_tile<2x4x128xindex>

    %Q_3d = ktdp.load %q_access_tile : !ktdp.access_tile<2x4x128xindex> -> tensor<2x4x128xf16>

    // Collapse [2, 4, 128] → [8, 128] for matrix multiply  (2*4=BLOCK_M=8)
    %Q_f16 = tensor.collapse_shape %Q_3d [[0, 1], [2]]
        : tensor<2x4x128xf16> into tensor<8x128xf16>

    // Upcast Q to f32 for numerical stability
    %Q = arith.extf %Q_f16 : tensor<8x128xf16> to tensor<8x128xf32>

    // -----------------------------------------------------------------------
    // Initialize running softmax state (f32)
    //   M   : tensor<8xf32>     row-wise running maximum,  init = -inf
    //   L   : tensor<8xf32>     row-wise running sum,      init = 1.0
    //   acc : tensor<8x128xf32> output accumulator,        init = 0.0
    // -----------------------------------------------------------------------
    %neg_inf_i32 = arith.constant 0xFF800000 : i32
    %neg_inf_f32 = arith.bitcast %neg_inf_i32 : i32 to f32
    %zero_f32    = arith.constant 0.0 : f32
    %one_f32     = arith.constant 1.0 : f32

    %M_init   = tensor.splat %neg_inf_f32 : tensor<8xf32>
    %L_init   = tensor.splat %one_f32     : tensor<8xf32>
    %acc_init = tensor.splat %zero_f32    : tensor<8x128xf32>

    // -----------------------------------------------------------------------
    // KV tile loop: for j in range(0, num_tiles)
    //   Loop-carried state: (M, L, acc)
    // -----------------------------------------------------------------------
    %M_final, %L_final, %acc_final =
      scf.for %j = %c0 to %num_tiles step %c1
        iter_args(%M = %M_init, %L = %L_init, %acc = %acc_init)
        -> (tensor<8xf32>, tensor<8xf32>, tensor<8x128xf32>) {

      // block_table column index for this iteration
      %bt_idx = arith.addi %block_table_offset, %j : index

      // ---- Load K tile (transposed) via block_tables ----------------------
      //   key_cache[block_tables[0, bt_idx], t, pid1, d]
      //   intermediate_variables: one per dim of key_cache (4 dims):
      //     d0=%bt_idx (block dim, scalar), d1=%m (t start=0, range [0,15]),
      //     d2=%pid1 (kv-head, scalar),     d3=%k (d start=0, range [0,127])
      //   variables_space_order:  (d0,d1,d2,d3) → (d0,d1,d2,d3) — identity,
      //     preserving all 4 dims as [1,16,1,128]; reshape+transpose done after load
      %K_tile = ktdp.construct_indirect_access_tile
          intermediate_variables(%d0, %d1, %d2, %d3)
          %key_cache_view[ind(%block_tables_view[%c0, %bt_idx + %d0]),
                          (%d1), (%pid1 + %d2), (%d3)] {
        variables_space_set   = affine_set<(d0, d1, d2, d3) : (
            d0 >= 0, -d0 + 0   >= 0,
            d1 >= 0, -d1 + 15  >= 0,
            d2 >= 0, -d2 + 0   >= 0,
            d3 >= 0, -d3 + 127 >= 0)>,
        variables_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      } : memref<64x16x8x128xf16>, memref<1x16xi32>
          -> !ktdp.access_tile<1x16x1x128xindex>

      // ---- Load V tile via block_tables -----------------------------------
      //   value_cache[block_tables[0, bt_idx], t, pid1, d]
      //   intermediate_variables: same structure as K tile
      %V_tile = ktdp.construct_indirect_access_tile
          intermediate_variables(%d0, %d1, %d2, %d3)
          %value_cache_view[ind(%block_tables_view[%c0, %bt_idx + %d0]),
                            (%d1), (%pid1 + %d2), (%d3)] {
        variables_space_set   = affine_set<(d0, d1, d2, d3) : (
            d0 >= 0, -d0 + 0   >= 0,
            d1 >= 0, -d1 + 15  >= 0,
            d2 >= 0, -d2 + 0   >= 0,
            d3 >= 0, -d3 + 127 >= 0)>,
        variables_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      } : memref<64x16x8x128xf16>, memref<1x16xi32>
          -> !ktdp.access_tile<1x16x1x128xindex>

      %K_f16_4d = ktdp.load %K_tile : !ktdp.access_tile<1x16x1x128xindex> -> tensor<1x16x1x128xf16>
      %V_f16_4d = ktdp.load %V_tile : !ktdp.access_tile<1x16x1x128xindex> -> tensor<1x16x1x128xf16>

      // Reshape K: [1, 16, 1, 128] → [16, 128]
      %K_f16_2d = tensor.collapse_shape %K_f16_4d [[0, 1], [2, 3]]
          : tensor<1x16x1x128xf16> into tensor<16x128xf16>

      // Transpose K: [16, 128] → [128, 16] for Q@K matmul
      %K_f16_2d_empty = tensor.empty() : tensor<128x16xf16>
      %K_f16 = linalg.transpose
          ins(%K_f16_2d : tensor<16x128xf16>)
          outs(%K_f16_2d_empty : tensor<128x16xf16>)
          permutation = [1, 0]

      // Reshape V: [1, 16, 1, 128] → [16, 128]
      %V_f16 = tensor.collapse_shape %V_f16_4d [[0, 1], [2, 3]]
          : tensor<1x16x1x128xf16> into tensor<16x128xf16>

      %K = arith.extf %K_f16 : tensor<128x16xf16> to tensor<128x16xf32>
      %V = arith.extf %V_f16 : tensor<16x128xf16>  to tensor<16x128xf32>

      // ---- S = scale * (Q @ K)  :  (8,128) @ (128,16) = (8,16) -----------
      %S_zeros = tensor.splat %zero_f32 : tensor<8x16xf32>
      %QK = linalg.matmul
            ins(%Q, %K  : tensor<8x128xf32>, tensor<128x16xf32>)
            outs(%S_zeros : tensor<8x16xf32>) -> tensor<8x16xf32>

      %scale_tile = tensor.splat %scale : tensor<8x16xf32>
      %S = arith.mulf %scale_tile, %QK : tensor<8x16xf32>

      // ---- Causal masking -------------------------------------------------
      // query_pos[row]      = pid0 * BLOCK_Q + row // num_queries_per_kv
      //                     = pid0 * 2 + row // 4
      //   row=0,1,2,3 → query_pos = pid0*2 + 0
      //   row=4,5,6,7 → query_pos = pid0*2 + 1
      // query_abs_pos[row]  = context_len + query_pos[row]  (varies per row)
      // seq_offset[col]     = j * TILE_SIZE + col
      // Set S[row, col] = -inf  when  seq_offset[col] > query_abs_pos[row]
      %S_masked_empty = tensor.empty() : tensor<8x16xf32>
      %S_masked = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%S : tensor<8x16xf32>)
          outs(%S_masked_empty : tensor<8x16xf32>) {
        ^bb0(%s_val : f32, %out : f32):
          %row       = linalg.index 0 : index
          %col       = linalg.index 1 : index
          %row_div4  = arith.divui %row, %c4 : index
          %qpos_row  = arith.addi %pid0_x2, %row_div4 : index
          %qabs_row  = arith.addi %context_len, %qpos_row : index
          %j_x16     = arith.muli %j, %c16 : index
          %seq_off   = arith.addi %j_x16, %col : index
          %in_causal = arith.cmpi ule, %seq_off, %qabs_row : index
          %result    = arith.select %in_causal, %s_val, %neg_inf_f32 : f32
          linalg.yield %result : f32
      } -> tensor<8x16xf32>

      // ---- m_j = max(M, rowmax(S_masked)) ---------------------------------
      %row_max_init = tensor.splat %neg_inf_f32 : tensor<8xf32>
      %S_row_max = linalg.reduce
          ins(%S_masked : tensor<8x16xf32>)
          outs(%row_max_init : tensor<8xf32>)
          dimensions = [1]
          (%a : f32, %b : f32) {
        %max_val = arith.maximumf %a, %b : f32
        linalg.yield %max_val : f32
      }

      %m_j_empty = tensor.empty() : tensor<8xf32>
      %m_j = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%M, %S_row_max : tensor<8xf32>, tensor<8xf32>)
          outs(%m_j_empty : tensor<8xf32>) {
        ^bb0(%m_old : f32, %m_new : f32, %out : f32):
          %max_val = arith.maximumf %m_old, %m_new : f32
          linalg.yield %max_val : f32
      } -> tensor<8xf32>

      // ---- P = exp(S_masked - m_j[:, None]) : (8, 16) f32 ----------------
      %P_empty = tensor.empty() : tensor<8x16xf32>
      %P = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%S_masked, %m_j : tensor<8x16xf32>, tensor<8xf32>)
          outs(%P_empty : tensor<8x16xf32>) {
        ^bb0(%s : f32, %m : f32, %out : f32):
          %sub = arith.subf %s, %m : f32
          %exp = math.exp %sub : f32
          linalg.yield %exp : f32
      } -> tensor<8x16xf32>

      // ---- l_j = rowsum(P) : (8,) f32 ------------------------------------
      %zero_vec = tensor.splat %zero_f32 : tensor<8xf32>
      %l_j = linalg.reduce
          ins(%P : tensor<8x16xf32>)
          outs(%zero_vec : tensor<8xf32>)
          dimensions = [1]
          (%a : f32, %b : f32) {
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
      }

      // ---- alpha = exp(M - m_j) : (8,) f32 --------------------------------
      %alpha_empty = tensor.empty() : tensor<8xf32>
      %alpha = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%M, %m_j : tensor<8xf32>, tensor<8xf32>)
          outs(%alpha_empty : tensor<8xf32>) {
        ^bb0(%m_old : f32, %m_new : f32, %out : f32):
          %diff = arith.subf %m_old, %m_new : f32
          %exp  = math.exp %diff : f32
          linalg.yield %exp : f32
      } -> tensor<8xf32>

      // ---- acc = acc * alpha[:, None] + P @ V -----------------------------
      // Step 1: acc_scaled = acc * alpha (broadcast over head dim)
      %acc_scaled_empty = tensor.empty() : tensor<8x128xf32>
      %acc_scaled = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%acc, %alpha : tensor<8x128xf32>, tensor<8xf32>)
          outs(%acc_scaled_empty : tensor<8x128xf32>) {
        ^bb0(%a : f32, %al : f32, %out : f32):
          %scaled = arith.mulf %a, %al : f32
          linalg.yield %scaled : f32
      } -> tensor<8x128xf32>

      // Step 2: acc_new = acc_scaled + P @ V  :  (8,16) @ (16,128) = (8,128)
      %acc_new = linalg.matmul
          ins(%P, %V   : tensor<8x16xf32>, tensor<16x128xf32>)
          outs(%acc_scaled : tensor<8x128xf32>) -> tensor<8x128xf32>

      // ---- L = L * alpha + l_j : (8,) f32 --------------------------------
      %L_new_empty = tensor.empty() : tensor<8xf32>
      %L_new = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>,
                           affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}
          ins(%L, %alpha, %l_j : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
          outs(%L_new_empty : tensor<8xf32>) {
        ^bb0(%l : f32, %al : f32, %lj : f32, %out : f32):
          %la  = arith.mulf %l,  %al : f32
          %new = arith.addf %la, %lj : f32
          linalg.yield %new : f32
      } -> tensor<8xf32>

      scf.yield %m_j, %L_new, %acc_new
          : tensor<8xf32>, tensor<8xf32>, tensor<8x128xf32>
    }  // end scf.for

    // -----------------------------------------------------------------------
    // Epilogue: acc = acc / L[:, None]
    // -----------------------------------------------------------------------
    %out_f32_empty = tensor.empty() : tensor<8x128xf32>
    %out_f32 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%acc_final, %L_final : tensor<8x128xf32>, tensor<8xf32>)
        outs(%out_f32_empty : tensor<8x128xf32>) {
      ^bb0(%a : f32, %l : f32, %out : f32):
        %div = arith.divf %a, %l : f32
        linalg.yield %div : f32
    } -> tensor<8x128xf32>

    // Truncate f32 → f16 for storage
    %out_f16 = arith.truncf %out_f32 : tensor<8x128xf32> to tensor<8x128xf16>

    // Expand [8, 128] → [2, 4, 128] to match the 3-D output memory view  (2*4=8)
    %out_3d = tensor.expand_shape %out_f16 [[0, 1], [2]] output_shape [2, 4, 128]
        : tensor<8x128xf16> into tensor<2x4x128xf16>

    // -----------------------------------------------------------------------
    // Store output tile: output[q_row_start : q_row_start+2, pid1*4 : pid1*4+4, 0:128]
    // -----------------------------------------------------------------------
    %output_access_tile = ktdp.construct_access_tile %output_view[%q_row_start, %pid1_x4, %c0] {
      access_tile_set = affine_set<(d0, d1, d2) : (
          d0 >= 0, -d0 + 1   >= 0,
          d1 >= 0, -d1 + 3   >= 0,
          d2 >= 0, -d2 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    } : memref<8x32x128xf16> -> !ktdp.access_tile<2x4x128xindex>

    ktdp.store %out_3d, %output_access_tile
        : tensor<2x4x128xf16>, !ktdp.access_tile<2x4x128xindex>

    return
  }
}
