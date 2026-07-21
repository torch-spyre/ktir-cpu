// RMSNorm: 4-core embarrassingly parallel (no allreduce).
//
// === Algorithm per core pid in grid [4, 1] ===
//
//   Each core processes a contiguous block of 64 rows: [pid*64, (pid+1)*64).
//   For each row:
//     partial_sq = sum(x[row, :] ** 2)       // full 4096 hidden dim
//     rstd = rsqrt(partial_sq / 4096 + eps)
//     y[row, :] = x[row, :] * rstd * w[:]
//
// No cross-core communication — each core independently normalizes its rows.
//
// === Dimensions ===
//
//   seq = 256, hidden_dim = 4096, grid = [4, 1]
//   Rows/core = 64 (blocked), BLOCK_SIZE = 1024, inner iters = 4
//   Per-core HBM footprint: ~1.5 MB (512 KB X + 512 KB W + 512 KB Y)
//

#tensor_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 4095 >= 0)>
#blk_set    = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>
#identity   = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @rmsnorm_4x1(
      %X: index,          // input  [256, 4096] f16
      %Y: index,          // output [256, 4096] f16
      %W: index,          // weight [256, 4096] f16, 1D tiled across rows
      %N: index,          // hidden_dim = 4096
      %eps: f16,          // epsilon, e.g. 1e-5
      %BLOCK_SIZE: index  // 1024
  ) attributes {grid = [4, 1]} {

    %pid = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index

    %row_start = arith.muli %pid, %c64 : index
    %row_end = arith.addi %row_start, %c64 : index

    %X_view = ktdp.construct_memory_view %X, sizes: [256, 4096], strides: [4096, 1] {
        coordinate_set = #tensor_set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x4096xf16>

    %Y_view = ktdp.construct_memory_view %Y, sizes: [256, 4096], strides: [4096, 1] {
        coordinate_set = #tensor_set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x4096xf16>

    %W_view = ktdp.construct_memory_view %W, sizes: [256, 4096], strides: [4096, 1] {
        coordinate_set = #tensor_set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x4096xf16>

    // Row loop: each core processes a contiguous block of 64 rows
    scf.for %row = %row_start to %row_end step %c1 : index {

        // === Pass 1: sum of squares over full hidden dim ===
        %zero_block = arith.constant dense<0.0> : tensor<1x1024xf16>

        %sq_acc = scf.for %col = %c0 to %c4096 step %BLOCK_SIZE
            iter_args(%acc = %zero_block) -> tensor<1x1024xf16> {

            %X_acc = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = #blk_set,
                access_tile_order = #identity
            } : memref<256x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %x = ktdp.load %X_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            %x_sq = arith.mulf %x, %x : tensor<1x1024xf16>
            %acc_next = arith.addf %acc, %x_sq : tensor<1x1024xf16>

            scf.yield %acc_next : tensor<1x1024xf16>
        }

        // Reduce 1x1024 accumulator to scalar sum
        %zero_scalar = arith.constant 0.0 : f16
        %reduce_init = tensor.splat %zero_scalar : tensor<1xf16>
        %sum_sq = linalg.reduce { arith.addf }
            ins(%sq_acc : tensor<1x1024xf16>)
            outs(%reduce_init : tensor<1xf16>)
            dimensions = [1]

        // === Compute rstd = rsqrt(sum_sq / N + eps) ===
        %c0_idx = arith.constant 0 : index
        %sum_scalar = tensor.extract %sum_sq[%c0_idx] : tensor<1xf16>

        %N_i32 = arith.index_cast %N : index to i32
        %N_f16 = arith.sitofp %N_i32 : i32 to f16
        %mean_sq = arith.divf %sum_scalar, %N_f16 : f16
        %mean_sq_plus_eps = arith.addf %mean_sq, %eps : f16
        %rstd_scalar = math.rsqrt %mean_sq_plus_eps : f16
        %rstd_block = tensor.splat %rstd_scalar : tensor<1x1024xf16>

        // === Pass 2: normalize and scale ===
        scf.for %col = %c0 to %c4096 step %BLOCK_SIZE {

            %X_acc2 = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = #blk_set,
                access_tile_order = #identity
            } : memref<256x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %W_acc = ktdp.construct_access_tile %W_view[%row, %col] {
                access_tile_set = #blk_set,
                access_tile_order = #identity
            } : memref<256x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %x2 = ktdp.load %X_acc2 : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
            %w  = ktdp.load %W_acc  : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            %x_norm = arith.mulf %x2, %rstd_block : tensor<1x1024xf16>
            %y = arith.mulf %x_norm, %w : tensor<1x1024xf16>

            %Y_acc = ktdp.construct_access_tile %Y_view[%row, %col] {
                access_tile_set = #blk_set,
                access_tile_order = #identity
            } : memref<256x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            ktdp.store %y, %Y_acc : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

            scf.yield
        }
        scf.yield
    }
    return
  }
}
