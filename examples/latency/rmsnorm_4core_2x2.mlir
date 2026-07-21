// RMSNorm: 4-core distributed hidden-dimension sharding with allreduce.
//
// === Distributed memory view pattern ===
//
// Each tensor (X, W, Y) is column-sharded into 2 partitions of width 2048.
// The sharding is made explicit via construct_distributed_memory_view:
//
//   1. Declare 2 partition views per tensor, each with a coordinate_set
//      covering its half of the global [256, 4096] coordinate space.
//   2. Compose into a single logical distributed view (no data movement).
//   3. Each core accesses at [row, col_id*2048 + col_off] — the runtime
//      routes to the correct partition via coordinate_set intersection.
//
// === Algorithm per core (col_id, row_id) in grid [2, 2] ===
//
//   Linearised pid = row_id * 2 + col_id
//   Row-group g = row_id (cores with same row_id share a row-group)
//   Col-lane  l = col_id (which half of hidden_dim this core owns)
//
//   col_start = col_id * 2048
//   row_start = row_id * 128
//   for row in range(row_start, row_start + 128):
//     // Pass 1: partial sum of squares over local hidden-dim slice
//     partial_sq = 0
//     for col_blk in range(0, 2048, 1024):
//       x = load X[row, col_start + col_blk]   // 1x1024
//       partial_sq += sum(x * x)
//
//     // Allreduce partial sums across M=2 col-partners
//     full_sq = allreduce_sum(partial_sq)
//
//     // Compute rstd
//     rstd = rsqrt(full_sq / 4096 + eps)
//
//     // Pass 2: normalize and scale local slice
//     for col_blk in range(0, 2048, 1024):
//       x = load X[row, col_start + col_blk]
//       w = load W[col_start + col_blk]
//       y = x * rstd * w
//       store y -> Y[row, col_start + col_blk]
//
// === Dimensions ===
//
//   seq = 256, hidden_dim = 4096, grid = [2, 2]
//   K = 2 (row sharding), M = 2 (column/hidden-dim sharding)
//   Rows/core = 128 (blocked), Cols/core = 2048, BLOCK_SIZE = 1024
//   Per-core HBM footprint: ~1.5 MB (512 KB X-shard + 512 KB W-shard + 512 KB Y-shard)
//

// Partition coordinate sets for column sharding (2 partitions of width 2048)
#part_col_0 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 2047 >= 0)>
#part_col_1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 - 2048 >= 0, -d1 + 4095 >= 0)>

// Access tile shape: 1 row x 1024 cols
#blk_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>
#identity  = affine_map<(d0, d1) -> (d0, d1)>

// Allreduce grouping: 2 groups of 2 cores.
// Linearised pids: {0,1} = row-group 0, {2,3} = row-group 1.
// i in [2*g, 2*g+1] for group g in {0, 1}.
#col_partners = affine_set<(i)[g] : (i - 2*g >= 0, -i + 2*g + 1 >= 0)>
#row_groups   = affine_set<(g) : (g >= 0, -g + 1 >= 0)>

module {
  func.func @rmsnorm_2x2(
      %X: index,          // input  256x4096 f16
      %Y: index,          // output 256x4096 f16
      %W: index,          // weight 256x4096 f16, 1D tiled across rows
      %N: index,          // hidden_dim = 4096
      %eps: f16,          // epsilon, e.g. 1e-5
      %BLOCK_SIZE: index  // 1024
  ) attributes {grid = [2, 2]} {

    // grid = [2, 2]: dim 0 = x (col axis), dim 1 = y (row axis)
    %col_id, %row_id = ktdp.get_compute_tile_id : index, index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index

    %row_start = arith.muli %row_id, %c128 : index
    %row_end = arith.addi %row_start, %c128 : index
    %col_start = arith.muli %col_id, %c2048 : index

    %X_base_1 = arith.addi %X, %c2048 : index
    %W_base_1 = arith.addi %W, %c2048 : index
    %Y_base_1 = arith.addi %Y, %c2048 : index

    // === Construct distributed views for X ===
    %X_p0 = ktdp.construct_memory_view %X, sizes: [256, 2048], strides: [4096, 1] {
        coordinate_set = #part_col_0, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x2048xf16>

    %X_p1 = ktdp.construct_memory_view %X_base_1, sizes: [256, 2048], strides: [4096, 1] {
        coordinate_set = #part_col_1, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x2048xf16>

    %X_view = ktdp.construct_distributed_memory_view
        (%X_p0, %X_p1 : memref<256x2048xf16>, memref<256x2048xf16>)
        : memref<256x4096xf16>

    // === Construct distributed views for W ===
    %W_p0 = ktdp.construct_memory_view %W, sizes: [256, 2048], strides: [4096, 1] {
        coordinate_set = #part_col_0, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x2048xf16>

    %W_p1 = ktdp.construct_memory_view %W_base_1, sizes: [256, 2048], strides: [4096, 1] {
        coordinate_set = #part_col_1, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x2048xf16>

    %W_view = ktdp.construct_distributed_memory_view
        (%W_p0, %W_p1 : memref<256x2048xf16>, memref<256x2048xf16>)
        : memref<256x4096xf16>

    // === Construct distributed views for Y ===
    %Y_p0 = ktdp.construct_memory_view %Y, sizes: [256, 2048], strides: [4096, 1] {
        coordinate_set = #part_col_0, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x2048xf16>

    %Y_p1 = ktdp.construct_memory_view %Y_base_1, sizes: [256, 2048], strides: [4096, 1] {
        coordinate_set = #part_col_1, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x2048xf16>

    %Y_view = ktdp.construct_distributed_memory_view
        (%Y_p0, %Y_p1 : memref<256x2048xf16>, memref<256x2048xf16>)
        : memref<256x4096xf16>

    // Row loop: each core processes a contiguous block of 128 rows
    scf.for %row = %row_start to %row_end step %c1 : index {

        // === Pass 1: partial sum of squares ===
        %zero_block = arith.constant dense<0.0> : tensor<1x1024xf16>

        %sq_acc = scf.for %col_off = %c0 to %c2048 step %BLOCK_SIZE
            iter_args(%acc = %zero_block) -> tensor<1x1024xf16> {

            %col = arith.addi %col_start, %col_off : index

            %X_acc = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = #blk_set,
                access_tile_order = #identity
            } : memref<256x4096xf16> -> !ktdp.access_tile<1x1024xindex>

            %x = ktdp.load %X_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            %x_sq = arith.mulf %x, %x : tensor<1x1024xf16>
            %acc_next = arith.addf %acc, %x_sq : tensor<1x1024xf16>

            scf.yield %acc_next : tensor<1x1024xf16>
        }

        // Reduce 1x1024 accumulator to scalar partial sum
        %zero_scalar = arith.constant 0.0 : f16
        %reduce_init = tensor.splat %zero_scalar : tensor<1xf16>
        %partial_sum = linalg.reduce { arith.addf }
            ins(%sq_acc : tensor<1x1024xf16>)
            outs(%reduce_init : tensor<1xf16>)
            dimensions = [1]

        // Reshape to 1x1 for inter_tile ops
        %partial_2d = tensor.expand_shape %partial_sum [[0, 1]] output_shape [1, 1]
                        : tensor<1xf16> into tensor<1x1xf16>

        // === Allreduce: sum partial sums across M=2 col-partner cores ===
        %fut = ktdp.inter_tile_produce
            producer_tiles_per_group = #col_partners,
            groups                   = #row_groups
            : tensor<1x1xf16> -> !ktdp.tile_future<tensor<1x1xf16>>
        {
          ^bb0(%gid: index):
            ktdp.yield_partial %partial_2d : tensor<1x1xf16>
        }

        %id_init = tensor.empty() : tensor<1x1xf16>
        %add_id  = linalg.fill ins(%zero_scalar : f16) outs(%id_init : tensor<1x1xf16>)
                     -> tensor<1x1xf16>

        %full_sum_1d = ktdp.inter_tile_reduce(%fut)
            consumer_tiles_per_group = #col_partners,
            groups                   = #row_groups,
            identity(%add_id : tensor<1x1xf16>)
            : !ktdp.tile_future<tensor<1x1xf16>> -> tensor<1xf16>
        {
          ^bb0(%lhs: tensor<1x1xf16>, %rhs: tensor<1x1xf16>):
            %init = tensor.empty() : tensor<1x1xf16>
            %sum  = linalg.add ins(%lhs, %rhs : tensor<1x1xf16>, tensor<1x1xf16>)
                               outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
            ktdp.yield_reduced %sum : tensor<1x1xf16>
        }

        // === Compute rstd = rsqrt(sum_sq / N + eps) ===
        %c0_idx = arith.constant 0 : index
        %full_sum_scalar = tensor.extract %full_sum_1d[%c0_idx] : tensor<1xf16>

        %N_i32 = arith.index_cast %N : index to i32
        %N_f16 = arith.sitofp %N_i32 : i32 to f16
        %mean_sq = arith.divf %full_sum_scalar, %N_f16 : f16
        %mean_sq_plus_eps = arith.addf %mean_sq, %eps : f16
        %rstd_scalar = math.rsqrt %mean_sq_plus_eps : f16
        %rstd_block = tensor.splat %rstd_scalar : tensor<1x1024xf16>

        // === Pass 2: normalize and scale ===
        scf.for %col_off = %c0 to %c2048 step %BLOCK_SIZE {

            %col = arith.addi %col_start, %col_off : index

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
