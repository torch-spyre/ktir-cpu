// Decode SDPA P@V on 32 cores: C[1,128] = A[1,8192] @ B[8192,128], f16.
//
// One query token (Sq = 1, decode), KV length 8192, head dim 128.  Shapes,
// block sizes and grid are baked in; the runtime args are just the a/b/c
// HBM base pointers.
//
// Grid [2, 16] = 32 cores.  Dimension 0 varies fastest, so the flat core id
// is ``pid_in * 2 + pid_out``:
//   dim 0 — output (N = 128) split x2: 64 columns = one f16 stick per core
//   dim 1 — KV contraction (K = 8192) split x16: 512 rows per core (reduce)
// Each output element is therefore a sum over 16 cores, folded with
// ktdp.inter_tile_produce / ktdp.inter_tile_reduce in two *strided* reduce
// groups: group g = cores {g, g+2, ..., g+30}, the ``(i - g) mod 2 == 0``
// producer set below — distinct from the contiguous grouping in
// examples/latency/ring_reduce_multi_group.mlir.  The partial, the combiner
// region and the identity are all f16, so the fold rounds back to f16 at
// every step.  The ``pid_in == 0`` core of each group (flat core id g) stores
// that group's 1x64 output slice under an scf.if guard, so each column block
// has exactly one writer.
//
// Source: a torch-spyre SuperDSC capture of
// F.scaled_dot_product_attention(Q[1,1,1,128], K/V[1,1,8192,128]) at 32
// cores — op 14 of the 22-op chain, and the only op in it whose work
// division splits a contraction dimension.  The core-to-slice map is the
// capture's own: core c takes output slice ``c % 2`` and KV slice ``c / 2``.
// Decode is the only SDPA shape that crosses cores at all — a prefill
// shape's parallel dimensions fill the grid on their own, so work division
// never has to reach for the reduction dimension.
#red_tiles = affine_set<(i)[g] : ((i - g) mod 2 == 0, i - g >= 0, -i + g + 30 >= 0)>
module {
  func.func @sdpa_pv_ksplit(%a_ptr: index, %b_ptr: index, %c_ptr: index) attributes {grid = [2, 16]} {
    %pid_out_1, %pid_in_2 = ktdp.get_compute_tile_id : index, index
    %c0_3 = arith.constant 0 : index
    %a_view_4 = ktdp.construct_memory_view %a_ptr, sizes: [1, 8192], strides: [8192, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1x8192xf16>
    %b_view_5 = ktdp.construct_memory_view %b_ptr, sizes: [8192, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 8191 >= 0, d1 >= 0, -d1 + 127 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8192x128xf16>
    %c_view_6 = ktdp.construct_memory_view %c_ptr, sizes: [1, 128], strides: [128, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 127 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
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
