module {
  func.func @add_kernel(
      %x_ptr: index,
      %y_ptr: index,
      %output_ptr: index
  ) attributes {grid = [32, 1]} {
    %c128 = arith.constant 128 : index
    %core_id = ktdp.get_compute_tile_id : index
    %offset = arith.muli %core_id, %c128 : index

    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [4096], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 4095 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096xf16>

    %x_tile = ktdp.construct_access_tile %x_view[%offset] {
          access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,
          access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<4096xf16> -> !ktdp.access_tile<128xindex>

    %y_view = ktdp.construct_memory_view %y_ptr, sizes: [4096], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 4095 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096xf16>

    %y_tile = ktdp.construct_access_tile %y_view[%offset] {
          access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,
          access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<4096xf16> -> !ktdp.access_tile<128xindex>
    
    %x = ktdp.load %x_tile : !ktdp.access_tile<128xindex> -> tensor<128xf16>
    %y = ktdp.load %y_tile : !ktdp.access_tile<128xindex> -> tensor<128xf16>

    %output = arith.addf %x, %y : tensor<128xf16>

    %output_view = ktdp.construct_memory_view %output_ptr, sizes: [4096], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 4095 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4096xf16>
  
    %output_tile = ktdp.construct_access_tile %output_view[%offset] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 127 >= 0)>,
      access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<4096xf16> -> !ktdp.access_tile<128xindex>

    ktdp.store %output, %output_tile : tensor<128xf16>, !ktdp.access_tile<128xindex>

    return
  }
}