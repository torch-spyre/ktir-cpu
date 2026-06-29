// Dynamic-shape exercise for ktdp.construct_distributed_memory_view.
//
// Sibling of distributed-view-copy-dynamic-s16.mlir; same kernel shape
// with s0 = 32 so global = 64.  Two iterations of a 32-wide access
// tile each land inside one partition, exercising the routing branch
// where one partition must be correctly excluded.
//
// Layout (logical 64x64 row-major, s0 = 32):
//   A0 = cols [0, s0),     stored on HBM (memref<64x?xf16>)
//   A1 = cols [s0, 2*s0),  stored on HBM (memref<64x?xf16>)
//   B  = full 64x64,       stored on HBM (output)

#A0_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>
#A1_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 - s0 >= 0, -d1 + 2*s0 - 1 >= 0)>
#full   = affine_set<(d0, d1)     : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#tile   = affine_set<(d0, d1)     : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#order  = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @distributed_view_copy_dynamic(
      %a0_ptr: index,
      %a1_ptr: index,
      %b_ptr: index,
      %s0_in: i32
  ) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %s0 = arith.index_cast %s0_in : i32 to index
    %ub = arith.muli %s0, %c2 : index

    %A0_view = ktdp.construct_memory_view %a0_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A0_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    %A1_view = ktdp.construct_memory_view %a1_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A1_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    %A_view = ktdp.construct_distributed_memory_view
        (%A0_view, %A1_view : memref<64x?xf16>, memref<64x?xf16>)
        : memref<64x64xf16>

    %B_view = ktdp.construct_memory_view %b_ptr, sizes: [64, 64], strides: [64, 1] {
      coordinate_set = #full,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x64xf16>

    // Trip count = 2*s0 / 32 = 2 for s0=32: one tile lands inside A0
    // (offset 0), the next inside A1 (offset 32).
    scf.for %off = %c0 to %ub step %c32 {
      %A_tile = ktdp.construct_access_tile %A_view[%c0, %off] {
        access_tile_set = #tile, access_tile_order = #order
      } : memref<64x64xf16> -> !ktdp.access_tile<64x32xindex>

      %B_tile = ktdp.construct_access_tile %B_view[%c0, %off] {
        access_tile_set = #tile, access_tile_order = #order
      } : memref<64x64xf16> -> !ktdp.access_tile<64x32xindex>

      %data = ktdp.load %A_tile : !ktdp.access_tile<64x32xindex> -> tensor<64x32xf16>
      ktdp.store %data, %B_tile : tensor<64x32xf16>, !ktdp.access_tile<64x32xindex>
    }

    return
  }
}
