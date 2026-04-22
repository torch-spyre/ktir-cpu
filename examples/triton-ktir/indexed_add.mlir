module {
  func.func @indexed_add_kernel(
    %x_ptr : index,
    %y_ptr : index,
    %index_ptr : index,
    %output_ptr : index,
    %dim1_start : index
  ) attributes {grid = [2, 8]} {
    %grid0 = ktdp.get_compute_tile_id { dim = 0 } : index
    %grid1 = ktdp.get_compute_tile_id { dim = 1 } : index
    %c0 = arith.constant 0 : index

    // Memory view for index tensor: shape [2] (num_indices), stride [1]
    %index_view = ktdp.construct_memory_view %index_ptr, sizes: [2], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2xi64>

    // Memory view for x: shape [128, 64, 8, 128]
    // strides: [64*8*128, 8*128, 128, 1] = [65536, 1024, 128, 1]
    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [128, 64, 8, 128], strides: [65536, 1024, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x64x8x128xf16>

    %x_tile = ktdp.construct_indirect_access_tile
        intermediate_variables(%d0, %d1, %d2, %d3)
        %x_view[ind(%index_view[%grid0 + %d0]), (%dim1_start + %d1), (%grid1 + %d2), (%d3)] {
          variables_space_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 0 >= 0, d3 >= 0, -d3 + 127 >= 0)>,
          variables_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        } : memref<128x64x8x128xf16>, memref<2xi64> -> !ktdp.access_tile<1x32x1x128xindex>

    // Memory view for y: shape [2, 32, 8, 128]
    // strides: [32*8*128, 8*128, 128, 1] = [32768, 1024, 128, 1]
    %y_view = ktdp.construct_memory_view %y_ptr, sizes: [2, 32, 8, 128], strides: [32768, 1024, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x32x8x128xf16>

    // Direct access tile for y: y[grid0, dim1_start, grid1, 0] covering [1, 32, 1, 128]
    %y_tile = ktdp.construct_access_tile %y_view[%grid0, %dim1_start, %grid1, %c0] {
      access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 0 >= 0, d3 >= 0, -d3 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    } : memref<2x32x8x128xf16> -> !ktdp.access_tile<1x32x1x128xindex>

    %x = ktdp.load %x_tile : !ktdp.access_tile<1x32x1x128xindex> -> tensor<1x32x1x128xf16>
    %y = ktdp.load %y_tile : !ktdp.access_tile<1x32x1x128xindex> -> tensor<1x32x1x128xf16>

    %output = arith.addf %x, %y : tensor<1x32x1x128xf16>

    // Memory view for output (same layout as y)
    %output_view = ktdp.construct_memory_view %output_ptr, sizes: [2, 32, 8, 128], strides: [32768, 1024, 128, 1] {
      coordinate_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 + 127 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x32x8x128xf16>

    %output_tile = ktdp.construct_access_tile %output_view[%grid0, %dim1_start, %grid1, %c0] {
      access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 0 >= 0, d3 >= 0, -d3 + 127 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    } : memref<2x32x8x128xf16> -> !ktdp.access_tile<1x32x1x128xindex>

    ktdp.store %output, %output_tile : tensor<1x32x1x128xf16>, !ktdp.access_tile<1x32x1x128xindex>

    return
  }
}