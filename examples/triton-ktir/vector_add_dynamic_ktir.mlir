// #set bounds the memory view to n_elements (symbolic s0); elements outside
// this range are masked on load and ignored on store — matching tensor_descriptor
// semantics. The access tile (#set1) is fixed at 1024; when n_elements < 1024
// the out-of-bounds positions are masked rather than triggering an error.
#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 1023 >= 0)>
module {
  func.func @add_kernel_dynamic(
      %x_ptr: index,
      %y_ptr: index,
      %output_ptr: index,
      %n_elements: i32
  ) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %n = arith.index_cast %n_elements : i32 to index

    %x_view = ktdp.construct_memory_view %x_ptr, sizes: [%n], strides: [1] {
      coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?xf32>
    %x_tile = ktdp.construct_access_tile %x_view[%c0] {
      access_tile_order = #map, access_tile_set = #set1
    } : memref<?xf32> -> !ktdp.access_tile<1024xindex>
    %x = ktdp.load %x_tile : <1024xindex> -> tensor<1024xf32>

    %y_view = ktdp.construct_memory_view %y_ptr, sizes: [%n], strides: [1] {
      coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?xf32>
    %y_tile = ktdp.construct_access_tile %y_view[%c0] {
      access_tile_order = #map, access_tile_set = #set1
    } : memref<?xf32> -> !ktdp.access_tile<1024xindex>
    %y = ktdp.load %y_tile : <1024xindex> -> tensor<1024xf32>

    %output = arith.addf %x, %y : tensor<1024xf32>

    %output_view = ktdp.construct_memory_view %output_ptr, sizes: [%n], strides: [1] {
      coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<?xf32>
    %output_tile = ktdp.construct_access_tile %output_view[%c0] {
      access_tile_order = #map, access_tile_set = #set1
    } : memref<?xf32> -> !ktdp.access_tile<1024xindex>
    ktdp.store %output, %output_tile : tensor<1024xf32>, <1024xindex>

    return
  }
}
