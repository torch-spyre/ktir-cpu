// Decode SDPA P@V on 32 cores: C[1,128] = A[1,8192] @ B[8192,128], f16.
//
// Single-stream decode: Sq = 1 query token, Sk = 8192 cached tokens, D = 128
// head dim, H = 1 head.  Shapes, block sizes and grid are baked in; the
// runtime args are just the a/b/c HBM base pointers.
//
// Grid [2, 16] = 32 cores, dimension 0 varying fastest, so the flat core id is
// ``pid_in * 2 + pid_out``: dim 0 splits the output (N = 128) into one 64-wide
// f16 stick per core, dim 1 splits the KV contraction (K = 8192) and is the
// reduce dimension.  The H = 1 row of the table below gives both split factors
// and the slice sizes they imply.
//
// The fold runs ktdp.inter_tile_produce / ktdp.inter_tile_reduce over two
// *strided* groups — group g = cores {g, g+2, ..., g+30}, spelled as
// #red_tiles below — distinct from the contiguous grouping in
// examples/latency/ring_reduce_multi_group.mlir.  Partial and combiner region
// are both f16, so the running sum rounds back to f16 at every step; the
// identity is required by the op but never read, because the producer set
// covers every core of the group and so no core seeds from it.  The
// ``pid_in == 0`` core of each group stores that group's 1x64 output slice
// under an scf.if guard, so each column block has exactly one writer.
//
// Source: a torch-spyre SuperDSC capture of
// F.scaled_dot_product_attention(Q[1,1,1,128], K/V[1,1,8192,128]) at 32
// cores — op 14 of the 22-op chain, and the only op in it whose work
// division splits a contraction dimension.  The core-to-slice map is the
// capture's own: core c takes output slice ``c % 2`` and KV slice ``c / 2``.
// Decode is the only SDPA shape that crosses cores at all; a prefill shape's
// parallel dimensions fill the grid on their own.
//
// Sweep roadmap, not part of this kernel's contract — running it needs nothing
// below this line.  The head count H is the one dial that changes the comm
// structure.  Everything else is pinned — B = Sq = 1, Sk = 8192, D = 128, 32
// cores — and the work division obeys one identity:
//
//   rows(H) x out(<= D / 64) x num_splits = 32 cores,   fan-in = num_splits
//
// Giving the output dimensions more 64-wide sticks to spend takes cores away
// from the contraction, so sweeping H walks the cross-core fold from its
// maximum down to nothing:
//
//    H   hidden = H*D   rows x out x num_splits   fan-in   tokens/core
//    1       128          1  x  2  x  16            16        512   <- here
//    2       256          2  x  2  x   8             8       1024
//    4       512          4  x  2  x   4             4       2048
//    8      1024          8  x  2  x   2             2       4096
//   16      2048         16  x  2  x   1             1       8192  (no fold)
//   32      4096         32  x  1  x   1             1       8192  (no fold)
//
// H = 1 is the far corner, not a workload: starving the output dimensions
// (hidden width 128) drives the fan-in to the largest the chip allows, which is
// what makes this the strongest cross-core stress case a decode shape offers.
// Serving widths are 4096 and up, where single-stream decode does not cross
// cores at all — the criterion at B = Sq = 1 is just hidden / 64 >= 32.  So the
// traffic is a property of how the model is served, not of attention.  The other
// two knobs: Sk rescales tokens/core without touching fan-in, and B multiplies
// output sticks exactly as H does.
//
// Only H = 1 and H = 2 are captured; the rows past those are the identity
// above run forwards.  Its divisibility premise holds at every stop on the
// dial because each H is a power of two and Sk = 8192 = 2^13.
#red_tiles = affine_set<(i)[g] : ((i - g) mod 2 == 0, i - g >= 0, -i + g + 30 >= 0)>
module {
  func.func @sdpa_pv_ksplit(%a_ptr: index, %b_ptr: index, %c_ptr: index) attributes {grid = [2, 16]} {
    %pid_out_1, %pid_in_2 = ktdp.get_compute_tile_id : index, index
    %c0_3 = arith.constant 0 : index
    %a_view_4 = ktdp.construct_memory_view %a_ptr, sizes: [1, 8192], strides: [8192, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.memory_space<global>
    } : memref<1x8192xf16>
    %b_view_5 = ktdp.construct_memory_view %b_ptr, sizes: [8192, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 8191 >= 0, d1 >= 0, -d1 + 127 >= 0)>, memory_space = #ktdp.memory_space<global>
    } : memref<8192x128xf16>
    %c_view_6 = ktdp.construct_memory_view %c_ptr, sizes: [1, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 127 >= 0)>, memory_space = #ktdp.memory_space<global>
    } : memref<1x128xf16>
    %blk_in_7 = arith.constant 512 : index
    %off_in_8 = arith.muli %pid_in_2, %blk_in_7 : index
    %blk_out_9 = arith.constant 64 : index
    %off_out_10 = arith.muli %pid_out_1, %blk_out_9 : index
    %a_acc_11 = ktdp.construct_access_tile %a_view_4[%c0_3, %off_in_8] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x8192xf16> -> !ktdp.access_tile<1x512xindex>
    %b_acc_12 = ktdp.construct_access_tile %b_view_5[%off_in_8, %off_out_10] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<8192x128xf16> -> !ktdp.access_tile<512x64xindex>
    %a_13 = ktdp.load %a_acc_11 : !ktdp.access_tile<1x512xindex> -> tensor<1x512xf16>
    %b_14 = ktdp.load %b_acc_12 : !ktdp.access_tile<512x64xindex> -> tensor<512x64xf16>
    %init_15 = arith.constant dense<0.0> : tensor<1x64xf16>
    %prod_16 = linalg.matmul ins(%a_13, %b_14 : tensor<1x512xf16>, tensor<512x64xf16>) outs(%init_15 : tensor<1x64xf16>) -> tensor<1x64xf16>
    %fut_17 = ktdp.inter_tile_produce
        producer_tiles_per_group = #red_tiles
        : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<(g) : (g >= 0, -g + 1 >= 0)>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %prod_16 : tensor<1x64xf16>
    }
    %zero_18 = arith.constant 0.0 : f16
    %id_init_19 = tensor.empty() : tensor<1x64xf16>
    %add_id_20 = linalg.fill ins(%zero_18 : f16) outs(%id_init_19 : tensor<1x64xf16>) -> tensor<1x64xf16>
    %reduced_21 = ktdp.inter_tile_reduce(%fut_17)
        consumer_tiles_per_group = #red_tiles,
        identity(%add_id_20 : tensor<1x64xf16>)
        : !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<(g) : (g >= 0, -g + 1 >= 0)>> -> tensor<1x64xf16>
    {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %r_init_22 = tensor.empty() : tensor<1x64xf16>
        %r_sum_23 = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>) outs(%r_init_22 : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %r_sum_23 : tensor<1x64xf16>
    }
    %blk_out_24 = arith.constant 64 : index
    %off_out_25 = arith.muli %pid_out_1, %blk_out_24 : index
    %c_acc_26 = ktdp.construct_access_tile %c_view_6[%c0_3, %off_out_25] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x128xf16> -> !ktdp.access_tile<1x64xindex>
    %wcmp_27 = arith.cmpi eq, %pid_in_2, %c0_3 : index
    scf.if %wcmp_27 {
      ktdp.store %reduced_21, %c_acc_26 : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }
}
