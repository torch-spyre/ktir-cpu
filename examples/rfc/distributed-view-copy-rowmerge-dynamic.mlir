// Dynamic-shape exercise for ktdp.construct_distributed_memory_view --
// SPIKE: merge on the NON-symbolic (row) axis, symbolic dim passed through.
//
// Sibling of distributed-view-copy-dynamic.mlir, transposed: there the
// partition boundary sits on the symbolic axis (cols) and the symbolic
// extent is what gets summed across partitions.  Here the boundary sits
// on the CONCRETE axis (rows 0..63 | 64..127) and the symbolic dim s0
// (cols) is shared by both partitions and passed through unchanged.
//
// Layout (logical 128 x s0 row-major):
//   A0 = rows [0, 64),    cols [0, s0),  HBM (memref<64x?xf16>)
//   A1 = rows [64, 128),  cols [0, s0),  HBM (memref<64x?xf16>)
//   B  = full 128 x s0,   HBM (output, memref<128x?xf16>)
//
// Composition sums the concrete row axis (64 + 64 = 128); the symbolic
// col axis is identical in both partitions and transparently carried
// into the composed view (memref<128x?xf16>).
//
// Routing coverage (per col offset, over the symbolic extent):
//   %roff = 0  -> 64x16 tile lands in A0        -> A1 excluded
//   %roff = 64 -> 64x16 tile lands in A1        -> A0 excluded
//   %roff = 32 -> 64x16 tile rows[32,96) straddle the row-64 boundary
//                 -> both partitions participate, routing splits C_i on each side
// so per-partition routing exercises both modes on the concrete axis --
// exclude-a-partition and split-across-two -- while binding the symbolic
// col dim from each partition's `sizes:`.  The straddle tile re-copies an
// overlapping (identical-valued) region, so bit-exact B == A_full holds.

#A0_set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>
#A1_set = affine_set<(d0, d1)[s0] : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>
#B_set  = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + s0 - 1 >= 0)>
#tile   = affine_set<(d0, d1)     : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 15 >= 0)>
#order  = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @distributed_view_copy_rowmerge_dynamic(
      %a0_ptr: index,
      %a1_ptr: index,
      %b_ptr: index,
      %s0_in: i32
  ) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %s0 = arith.index_cast %s0_in : i32 to index

    // (1) Per-partition memory views with symbolic trailing dim (shared s0).
    %A0_view = ktdp.construct_memory_view %a0_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A0_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    %A1_view = ktdp.construct_memory_view %a1_ptr, sizes: [64, %s0], strides: [%s0, 1] {
      coordinate_set = #A1_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x?xf16>

    // (1) Compose along the concrete row axis: 64 + 64 = 128 rows, s0 cols.
    %A_view = ktdp.construct_distributed_memory_view
        (%A0_view, %A1_view : memref<64x?xf16>, memref<64x?xf16>)
        : memref<128x?xf16>

    // (1) Output view B, 128 x s0.
    %B_view = ktdp.construct_memory_view %b_ptr, sizes: [128, %s0], strides: [%s0, 1] {
      coordinate_set = #B_set,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<128x?xf16>

    // (2)+(3) Iterate: cols over the symbolic extent, rows over the two
    // partitions.  Concrete 64x16 tile appears only at the access level.
    scf.for %coff = %c0 to %s0 step %c16 {
      // Aligned tiles: each lands wholly inside one partition (exclude mode).
      scf.for %roff = %c0 to %c128 step %c64 {
        %A_tile = ktdp.construct_access_tile %A_view[%roff, %coff] {
          access_tile_set = #tile, access_tile_order = #order
        } : memref<128x?xf16> -> !ktdp.access_tile<64x16xindex>

        %B_tile = ktdp.construct_access_tile %B_view[%roff, %coff] {
          access_tile_set = #tile, access_tile_order = #order
        } : memref<128x?xf16> -> !ktdp.access_tile<64x16xindex>

        %data = ktdp.load %A_tile : !ktdp.access_tile<64x16xindex> -> tensor<64x16xf16>
        ktdp.store %data, %B_tile : tensor<64x16xf16>, !ktdp.access_tile<64x16xindex>
      }

      // Straddle tile: rows[32,96) cross the row-64 boundary, so both
      // partitions participate and routing splits the tile across them.
      %As_tile = ktdp.construct_access_tile %A_view[%c32, %coff] {
        access_tile_set = #tile, access_tile_order = #order
      } : memref<128x?xf16> -> !ktdp.access_tile<64x16xindex>

      %Bs_tile = ktdp.construct_access_tile %B_view[%c32, %coff] {
        access_tile_set = #tile, access_tile_order = #order
      } : memref<128x?xf16> -> !ktdp.access_tile<64x16xindex>

      %sdata = ktdp.load %As_tile : !ktdp.access_tile<64x16xindex> -> tensor<64x16xf16>
      ktdp.store %sdata, %Bs_tile : tensor<64x16xf16>, !ktdp.access_tile<64x16xindex>
    }

    return
  }
}
