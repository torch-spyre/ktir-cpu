// FFN-SwiGLU: 4-core distributed hidden-dimension sharding with ring all-reduce.
//
// Algorithm per core pid in [0, 3]:
//   ffn_start = pid * 256
//   gate_local = x @ W_gate[:, ffn_start:ffn_start+256]
//   up_local = x @ W_up[:, ffn_start:ffn_start+256]
//   silu_local = gate_local * sigmoid(gate_local)
//   fused_local = silu_local * up_local
//   out_partial_local = fused_local @ W_down[ffn_start:ffn_start+256, :]
//   out = all_reduce_sum(out_partial_local)
//   result = x + out
//   core 0 stores result
//
// Dimensions:
//   seq = 4, d_model = 256, d_ffn = 1024, grid = [4]
//   Shard width = 256 hidden units/core
//
// Layout and stick granularity:
//   x        : [4, 256]
//   W_gate   : [256, 1024], row-major strides [1024, 1]
//   W_up     : [256, 1024], row-major strides [1024, 1]
//   W_down   : [1024, 256], row-major strides [256, 1]
//   out      : [4, 256]
//
// With f16, one stick = 64 elements.  The 256-wide FFN shard width preserves
// stick granularity for all three weight matrices.

#x_set         = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#gate_full_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 1023 >= 0)>
#gate_shard_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#down_full_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#partial_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#flat_1024_set = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>
#identity_2d   = affine_map<(d0, d1) -> (d0, d1)>
#identity_1d   = affine_map<(d0) -> (d0)>
#all_tiles     = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group     = affine_set<(g) : (g == 0)>

module {
  func.func @ffn_swiglu_4core(
      %x_ptr: index,
      %w_gate_ptr: index,
      %w_up_ptr: index,
      %w_down_ptr: index,
      %out_ptr: index
  ) attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index

    %pid = ktdp.get_compute_tile_id : index
    %ffn_start = arith.muli %pid, %c256 : index

    // Base element offsets for this core's weight shards.
    %gate_base = arith.addi %w_gate_ptr, %ffn_start : index
    %up_base = arith.addi %w_up_ptr, %ffn_start : index
    %down_row_offset = arith.muli %ffn_start, %c256 : index
    %down_base = arith.addi %w_down_ptr, %down_row_offset : index

    // Replicated input and final output views.
    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [4, 256], strides: [256, 1] {
      coordinate_set = #x_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x256xf16>

    %out_view = ktdp.construct_memory_view %out_ptr, sizes: [4, 256], strides: [256, 1] {
      coordinate_set = #x_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x256xf16>

    // Local FFN-dimension shards for this core.
    %w_gate_view = ktdp.construct_memory_view %gate_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_shard_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    %w_up_view = ktdp.construct_memory_view %up_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_shard_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    %w_down_view = ktdp.construct_memory_view %down_base, sizes: [256, 256], strides: [256, 1] {
      coordinate_set = #gate_shard_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    // Step 1: load full replicated x.
    %x_acc = ktdp.construct_access_tile %x_view[%c0, %c0] {
      access_tile_set = #x_set,
      access_tile_order = #identity_2d
    } : memref<4x256xf16> -> !ktdp.access_tile<4x256xindex>
    %x = ktdp.load %x_acc : !ktdp.access_tile<4x256xindex> -> tensor<4x256xf16>

    // Steps 2-3: load local W_gate and W_up shards.
    %w_gate_acc = ktdp.construct_access_tile %w_gate_view[%c0, %c0] {
      access_tile_set = #gate_shard_set,
      access_tile_order = #identity_2d
    } : memref<256x256xf16> -> !ktdp.access_tile<256x256xindex>
    %w_gate = ktdp.load %w_gate_acc : !ktdp.access_tile<256x256xindex> -> tensor<256x256xf16>

    %w_up_acc = ktdp.construct_access_tile %w_up_view[%c0, %c0] {
      access_tile_set = #gate_shard_set,
      access_tile_order = #identity_2d
    } : memref<256x256xf16> -> !ktdp.access_tile<256x256xindex>
    %w_up = ktdp.load %w_up_acc : !ktdp.access_tile<256x256xindex> -> tensor<256x256xf16>

    // Steps 4-5: local gate and up projections.
    %gate_init = tensor.empty() : tensor<4x256xf16>
    %gate = linalg.matmul ins(%x, %w_gate : tensor<4x256xf16>, tensor<256x256xf16>)
                          outs(%gate_init : tensor<4x256xf16>) -> tensor<4x256xf16>

    %up_init = tensor.empty() : tensor<4x256xf16>
    %up = linalg.matmul ins(%x, %w_up : tensor<4x256xf16>, tensor<256x256xf16>)
                        outs(%up_init : tensor<4x256xf16>) -> tensor<4x256xf16>

    // Step 6: local SwiGLU activation and fusion.
    %neg_gate = arith.negf %gate : tensor<4x256xf16>
    %exp_neg_gate = math.exp %neg_gate : tensor<4x256xf16>
    %one = arith.constant dense<1.0> : tensor<4x256xf16>
    %one_plus_exp = arith.addf %one, %exp_neg_gate : tensor<4x256xf16>
    %sigmoid = arith.divf %one, %one_plus_exp : tensor<4x256xf16>
    %silu = arith.mulf %gate, %sigmoid : tensor<4x256xf16>
    %fused = arith.mulf %silu, %up : tensor<4x256xf16>

    // Step 7: load local W_down shard and compute local output partial.
    %w_down_acc = ktdp.construct_access_tile %w_down_view[%c0, %c0] {
      access_tile_set = #gate_shard_set,
      access_tile_order = #identity_2d
    } : memref<256x256xf16> -> !ktdp.access_tile<256x256xindex>
    %w_down = ktdp.load %w_down_acc : !ktdp.access_tile<256x256xindex> -> tensor<256x256xf16>

    %out_partial_init = tensor.empty() : tensor<4x256xf16>
    %out_partial = linalg.matmul ins(%fused, %w_down : tensor<4x256xf16>, tensor<256x256xf16>)
                                 outs(%out_partial_init : tensor<4x256xf16>) -> tensor<4x256xf16>

    %out_partial_flat = tensor.collapse_shape %out_partial [[0, 1]] : tensor<4x256xf16> into tensor<1024xf16>

    // Step 8: all-reduce the local output partial across all 4 cores.
    %fut = ktdp.inter_tile_produce
        producer_tiles_per_group = #all_tiles,
        groups                   = #one_group
        : tensor<1024xf16> -> !ktdp.tile_future<tensor<1024xf16>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %out_partial_flat : tensor<1024xf16>
    }

    %c_zero = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<1024xf16>
    %add_id = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<1024xf16>) -> tensor<1024xf16>

    %out_reduced_flat = ktdp.inter_tile_reduce(%fut)
        consumer_tiles_per_group = #all_tiles,
        groups                   = #one_group,
        identity(%add_id : tensor<1024xf16>)
        : !ktdp.tile_future<tensor<1024xf16>> -> tensor<1024xf16>
    {
      ^bb0(%lhs: tensor<1024xf16>, %rhs: tensor<1024xf16>):
        %init = tensor.empty() : tensor<1024xf16>
        %sum = linalg.add ins(%lhs, %rhs : tensor<1024xf16>, tensor<1024xf16>)
                          outs(%init : tensor<1024xf16>) -> tensor<1024xf16>
        ktdp.yield_reduced %sum : tensor<1024xf16>
    }

    %out_reduced = tensor.expand_shape %out_reduced_flat [[0, 1]] output_shape [4, 256]
                     : tensor<1024xf16> into tensor<4x256xf16>

    // Step 9: residual add; only core 0 stores the final result.
    %result = arith.addf %x, %out_reduced : tensor<4x256xf16>
    %is_writer = arith.cmpi eq, %pid, %c0 : index
    scf.if %is_writer {
      %out_acc = ktdp.construct_access_tile %out_view[%c0, %c0] {
        access_tile_set = #x_set,
        access_tile_order = #identity_2d
      } : memref<4x256xf16> -> !ktdp.access_tile<4x256xindex>
      ktdp.store %result, %out_acc : tensor<4x256xf16>, !ktdp.access_tile<4x256xindex>
    }

    return
  }
}
