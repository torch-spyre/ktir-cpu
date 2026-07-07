// FFN-SwiGLU: 4-core distributed hidden-dimension sharding with all-reduce.
//
// === Algorithm per core pid in [0, 3] ===
//
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
// === Distributed view pattern ===
//
// Weight tensors are too large for one core.  We shard them across 4 cores
// using ktdp.construct_distributed_memory_view, which works in 3 steps:
//
// 1. Declare partition views — one construct_memory_view per shard, each
//    carrying a coordinate_set that maps it to a region of the global tensor.
//    For W_gate [256, 1024] column-sharded into 4:
//
//      partition 0: coordinate_set covers cols [0, 255]     → base = w_gate_ptr
//      partition 1: coordinate_set covers cols [256, 511]   → base = w_gate_ptr + 256
//      partition 2: coordinate_set covers cols [512, 767]   → base = w_gate_ptr + 512
//      partition 3: coordinate_set covers cols [768, 1023]  → base = w_gate_ptr + 768
//
//    Each partition is [256, 256] with strides [1024, 1] (a strided subview
//    of the contiguous row-major [256, 1024] matrix in HBM).
//
// 2. Compose into a global view — construct_distributed_memory_view wraps
//    the 4 partition MemRefs into a single logical memref<256x1024xf16>.
//    No data is moved or allocated; the op is purely declarative.
//
// 3. Access via pid-positioned tile — each core creates an access tile at
//    [0, pid*256] on the global view.  The runtime intersects the access
//    region with each partition's coordinate_set, finds exactly one match,
//    and loads from that partition's physical memory.
//
// This separates layout declaration (where data lives) from access logic
// (which coordinates each core needs), making the sharding strategy explicit
// and verifiable from the coordinate_set attributes alone.
//
// W_down [1024, 256] is row-sharded instead: partition k covers rows
// [k*256, (k+1)*256-1], and each core accesses at [pid*256, 0].
//
// === Dimensions ===
//
//   seq = 4, d_model = 256, d_ffn = 1024, grid = [4]
//   Shard width = 256 hidden units/core
//
// === Layout and stick granularity ===
//
//   x        : [4, 256]   — replicated, all cores read the same input
//   W_gate   : [256, 1024], strides [1024, 1], column-sharded into 4×[256,256]
//   W_up     : [256, 1024], strides [1024, 1], column-sharded into 4×[256,256]
//   W_down   : [1024, 256], strides [256, 1],  row-sharded into 4×[256,256]
//   out      : [4, 256]   — only core 0 writes the final result
//
// With f16, one stick = 64 elements.  The 256-wide shard width preserves
// stick granularity for all three weight matrices.
//
// === Note on repetition ===
//
// The per-partition construct_memory_view declarations below repeat 4× for
// each weight matrix.  This is the expected lowered form: MLIR's static SSA
// design requires each partition to be a distinct named value, and the
// variadic construct_distributed_memory_view op takes them all as explicit
// operands — there is no loop construct that can build a variadic operand
// list dynamically.
//
// In practice, this MLIR is not hand-authored at scale.  A frontend
// (e.g. ktir-mlir-frontend or a kernel-gen Python script) emits it from a
// compact high-level description like:
//   distribute(W_gate, axis=1, num_shards=4, shard_width=256)
// The repetitive partition declarations are generated, not written.
// See: https://gist.github.com/kiszk/f575757845d8a637c81ec26735b39643
// for a kernel-gen script that parameterizes tiling, core count, and fusion.

// x and output coordinate set: [4, 256]
#x_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 255 >= 0)>

// Access tile shape for each core's 256x256 shard (relative to access origin)
#shard_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 255 >= 0)>

// W_gate / W_up column-shard coordinate sets (global coords)
#gate_col_0 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#gate_col_1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 - 256 >= 0, -d1 + 511 >= 0)>
#gate_col_2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 - 512 >= 0, -d1 + 767 >= 0)>
#gate_col_3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 - 768 >= 0, -d1 + 1023 >= 0)>

// W_down row-shard coordinate sets (global coords)
#down_row_0 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#down_row_1 = affine_set<(d0, d1) : (d0 - 256 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#down_row_2 = affine_set<(d0, d1) : (d0 - 512 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 + 255 >= 0)>
#down_row_3 = affine_set<(d0, d1) : (d0 - 768 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 + 255 >= 0)>

#identity_2d = affine_map<(d0, d1) -> (d0, d1)>
#all_tiles   = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group   = affine_set<(g) : (g == 0)>

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
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    %c65536 = arith.constant 65536 : index
    %c131072 = arith.constant 131072 : index
    %c196608 = arith.constant 196608 : index

    %pid = ktdp.get_compute_tile_id : index
    %ffn_start = arith.muli %pid, %c256 : index

    // ---- Replicated input and output views ----

    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [4, 256], strides: [256, 1] {
      coordinate_set = #x_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x256xf16>

    %out_view = ktdp.construct_memory_view %out_ptr, sizes: [4, 256], strides: [256, 1] {
      coordinate_set = #x_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x256xf16>

    // ---- W_gate [256, 1024] distributed view: 4 column shards ----

    %gate_p1_base = arith.addi %w_gate_ptr, %c256 : index
    %gate_p2_base = arith.addi %w_gate_ptr, %c512 : index
    %gate_p3_base = arith.addi %w_gate_ptr, %c768 : index

    %gate_p0 = ktdp.construct_memory_view %w_gate_ptr, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_0,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %gate_p1 = ktdp.construct_memory_view %gate_p1_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_1,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %gate_p2 = ktdp.construct_memory_view %gate_p2_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_2,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %gate_p3 = ktdp.construct_memory_view %gate_p3_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_3,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    %w_gate_dist = ktdp.construct_distributed_memory_view
        (%gate_p0, %gate_p1, %gate_p2, %gate_p3 : memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>)
        : memref<256x1024xf16>

    // ---- W_up [256, 1024] distributed view: 4 column shards ----

    %up_p1_base = arith.addi %w_up_ptr, %c256 : index
    %up_p2_base = arith.addi %w_up_ptr, %c512 : index
    %up_p3_base = arith.addi %w_up_ptr, %c768 : index

    %up_p0 = ktdp.construct_memory_view %w_up_ptr, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_0,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %up_p1 = ktdp.construct_memory_view %up_p1_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_1,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %up_p2 = ktdp.construct_memory_view %up_p2_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_2,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %up_p3 = ktdp.construct_memory_view %up_p3_base, sizes: [256, 256], strides: [1024, 1] {
      coordinate_set = #gate_col_3,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    %w_up_dist = ktdp.construct_distributed_memory_view
        (%up_p0, %up_p1, %up_p2, %up_p3 : memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>)
        : memref<256x1024xf16>

    // ---- W_down [1024, 256] distributed view: 4 row shards ----

    %down_p1_base = arith.addi %w_down_ptr, %c65536 : index
    %down_p2_base = arith.addi %w_down_ptr, %c131072 : index
    %down_p3_base = arith.addi %w_down_ptr, %c196608 : index

    %down_p0 = ktdp.construct_memory_view %w_down_ptr, sizes: [256, 256], strides: [256, 1] {
      coordinate_set = #down_row_0,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %down_p1 = ktdp.construct_memory_view %down_p1_base, sizes: [256, 256], strides: [256, 1] {
      coordinate_set = #down_row_1,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %down_p2 = ktdp.construct_memory_view %down_p2_base, sizes: [256, 256], strides: [256, 1] {
      coordinate_set = #down_row_2,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>
    %down_p3 = ktdp.construct_memory_view %down_p3_base, sizes: [256, 256], strides: [256, 1] {
      coordinate_set = #down_row_3,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<256x256xf16>

    %w_down_dist = ktdp.construct_distributed_memory_view
        (%down_p0, %down_p1, %down_p2, %down_p3 : memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>, memref<256x256xf16>)
        : memref<1024x256xf16>

    // ---- Step 1: load replicated x ----

    %x_acc = ktdp.construct_access_tile %x_view[%c0, %c0] {
      access_tile_set = #x_set,
      access_tile_order = #identity_2d
    } : memref<4x256xf16> -> !ktdp.access_tile<4x256xindex>
    %x = ktdp.load %x_acc : !ktdp.access_tile<4x256xindex> -> tensor<4x256xf16>

    // ---- Steps 2-3: load local W_gate and W_up shards via distributed view ----

    %w_gate_acc = ktdp.construct_access_tile %w_gate_dist[%c0, %ffn_start] {
      access_tile_set = #shard_set,
      access_tile_order = #identity_2d
    } : memref<256x1024xf16> -> !ktdp.access_tile<256x256xindex>
    %w_gate = ktdp.load %w_gate_acc : !ktdp.access_tile<256x256xindex> -> tensor<256x256xf16>

    %w_up_acc = ktdp.construct_access_tile %w_up_dist[%c0, %ffn_start] {
      access_tile_set = #shard_set,
      access_tile_order = #identity_2d
    } : memref<256x1024xf16> -> !ktdp.access_tile<256x256xindex>
    %w_up = ktdp.load %w_up_acc : !ktdp.access_tile<256x256xindex> -> tensor<256x256xf16>

    // ---- Steps 4-5: local gate and up projections ----

    %gate_init = tensor.empty() : tensor<4x256xf16>
    %gate = linalg.matmul ins(%x, %w_gate : tensor<4x256xf16>, tensor<256x256xf16>)
                          outs(%gate_init : tensor<4x256xf16>) -> tensor<4x256xf16>

    %up_init = tensor.empty() : tensor<4x256xf16>
    %up = linalg.matmul ins(%x, %w_up : tensor<4x256xf16>, tensor<256x256xf16>)
                        outs(%up_init : tensor<4x256xf16>) -> tensor<4x256xf16>

    // ---- Step 6: SwiGLU activation and fusion ----

    %neg_gate = arith.negf %gate : tensor<4x256xf16>
    %exp_neg_gate = math.exp %neg_gate : tensor<4x256xf16>
    %one = arith.constant dense<1.0> : tensor<4x256xf16>
    %one_plus_exp = arith.addf %one, %exp_neg_gate : tensor<4x256xf16>
    %sigmoid = arith.divf %one, %one_plus_exp : tensor<4x256xf16>
    %silu = arith.mulf %gate, %sigmoid : tensor<4x256xf16>
    %fused = arith.mulf %silu, %up : tensor<4x256xf16>

    // ---- Step 7: load local W_down shard and compute partial output ----

    %w_down_acc = ktdp.construct_access_tile %w_down_dist[%ffn_start, %c0] {
      access_tile_set = #shard_set,
      access_tile_order = #identity_2d
    } : memref<1024x256xf16> -> !ktdp.access_tile<256x256xindex>
    %w_down = ktdp.load %w_down_acc : !ktdp.access_tile<256x256xindex> -> tensor<256x256xf16>

    %out_partial_init = tensor.empty() : tensor<4x256xf16>
    %out_partial = linalg.matmul ins(%fused, %w_down : tensor<4x256xf16>, tensor<256x256xf16>)
                                 outs(%out_partial_init : tensor<4x256xf16>) -> tensor<4x256xf16>

    %out_partial_flat = tensor.collapse_shape %out_partial [[0, 1]] : tensor<4x256xf16> into tensor<1024xf16>

    // ---- Step 8: all-reduce partial outputs across 4 cores ----

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

    // ---- Step 9: residual add and store ----
    // After all-reduce, every core holds the full output.  Two strategies:
    //   (a) All-reduce → 1 core stores (used here): simple, sufficient for
    //       small outputs.  Cores 1-3 discard their identical result.
    //   (b) Reduce-scatter → each core stores its slice: more bandwidth-
    //       efficient for large outputs (e.g. seq=2048, d_model=4096) since
    //       each core writes only 1/N of the result with no redundant traffic.
    // We use (a) because the output is small ([4, 256]) and because
    // inter_tile_reduce_scatter is not yet implemented — it depends on
    // upstream ktir-mlir-frontend PR #23 (unmerged).

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
