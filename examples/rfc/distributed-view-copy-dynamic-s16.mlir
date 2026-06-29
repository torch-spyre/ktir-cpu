// Dynamic-shape exercise for ktdp.construct_distributed_memory_view.
//
// Drives a symbolic-BoxSet distributed view through the full path:
//   regex parser -> construct_memory_view (symbol binding)
//                -> construct_distributed_memory_view
//                -> construct_access_tile
//                -> distributed_tile_access (per-partition routing)
//                -> ktdp.load / ktdp.store
//
// Layout (logical 64x32 row-major, s0 = 16):
//   A0 = cols [0, s0),     stored on HBM (memref<64x?xf16>)
//   A1 = cols [s0, 2*s0),  stored on HBM (memref<64x?xf16>)
//   B  = full 64x32,       stored on HBM (output)
//
// A 32-wide access tile straddles both 16-wide partitions, so the
// per-partition routing C_i = B_i ∩ (x+A) splits non-trivially on each
// side.  A sibling fixture (-s32.mlir) covers the dual case where each
// tile lands inside one partition and the other is correctly excluded.
//
// The global distributed view, B, and the access tile are concrete
// because the regex parser does not accept '?' in distributed-view
// result types or in access_tile<NxMxindex> shapes.  Symbolic binding
// is exercised on every construct_memory_view call for the partitions.

#A0_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>
#A1_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 - s0 >= 0, -d1 + 2*s0 - 1 >= 0)>
#full   = affine_set<(d0, d1)     : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 31 >= 0)>
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

    // (1) Per-partition memory views with symbolic trailing dim.
    %A0_view = ktdp.construct_memory_view %a0_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A0_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    %A1_view = ktdp.construct_memory_view %a1_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A1_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    // (1) Compose into a single logical 64x32 distributed view.
    %A_view = ktdp.construct_distributed_memory_view
        (%A0_view, %A1_view : memref<64x?xf16>, memref<64x?xf16>)
        : memref<64x32xf16>

    // (1) Output view B (concrete).
    %B_view = ktdp.construct_memory_view %b_ptr, sizes: [64, 32], strides: [32, 1] {
      coordinate_set = #full,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x32xf16>

    // (2)+(3) Iterate access tiles across the global domain.  Trip count
    // = 2*s0 / 32; for s0=16 this is exactly 1 (tile spans both
    // partitions).  Constructing a fresh access tile per iteration with
    // %off as the column index keeps access_tile_set concrete.
    scf.for %off = %c0 to %ub step %c32 {
      %A_tile = ktdp.construct_access_tile %A_view[%c0, %off] {
        access_tile_set = #tile, access_tile_order = #order
      } : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>

      %B_tile = ktdp.construct_access_tile %B_view[%c0, %off] {
        access_tile_set = #tile, access_tile_order = #order
      } : memref<64x32xf16> -> !ktdp.access_tile<64x32xindex>

      %data = ktdp.load %A_tile : !ktdp.access_tile<64x32xindex> -> tensor<64x32xf16>
      ktdp.store %data, %B_tile : tensor<64x32xf16>, !ktdp.access_tile<64x32xindex>
    }

    return
  }
}
