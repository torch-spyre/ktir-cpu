// Rank-0 (scalar) tensor regression anchor.
//
// The Triton -> KTIR lowering normalises a 1x1 broadcast source through a
// rank-0 intermediate: it collapses tensor<1x1xf16> down to a scalar
// tensor<f16>, then broadcasts that scalar across the output. batch_norm's
// per-channel multiply is the first torch-spyre driver to emit this shape.
//
// This exercises the full pipeline on the rank-0 path:
//   load 1x1 -> tensor.collapse_shape into tensor<f16> -> linalg.broadcast -> store.
module {
  func.func @scalar_broadcast(
      %in_ptr: index,
      %out_ptr: index
  ) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index

    // Load the 1x1 scalar source from HBM.
    %in_view = ktdp.construct_memory_view %in_ptr, sizes: [1, 1], strides: [1, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<1x1xf16>

    %in_tile = ktdp.construct_access_tile %in_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1xf16> -> !ktdp.access_tile<1x1xindex>

    %in = ktdp.load %in_tile : !ktdp.access_tile<1x1xindex> -> tensor<1x1xf16>

    // Collapse (1, 1) -> rank-0 scalar () -- the op the rank-0 support targets.
    %s = tensor.collapse_shape %in [] : tensor<1x1xf16> into tensor<f16>

    // Broadcast the scalar across the whole 4x64 output.
    %out_init = tensor.empty() : tensor<4x64xf16>
    %b = linalg.broadcast ins(%s : tensor<f16>)
                          outs(%out_init : tensor<4x64xf16>)
                          dimensions = [0, 1]

    // Store to HBM.
    %out_view = ktdp.construct_memory_view %out_ptr, sizes: [4, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<4x64xf16>

    %out_tile = ktdp.construct_access_tile %out_view[%c0, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>

    ktdp.store %b, %out_tile : tensor<4x64xf16>, !ktdp.access_tile<4x64xindex>

    return
  }
}
