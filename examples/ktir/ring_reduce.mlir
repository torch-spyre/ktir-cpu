// Ring reduce: 4 cores, each holds a 1×128 row in HBM, reduce-sum into core 0.
//
// Grid: [4, 1, 1] — axis 0 distributes work.
//
// Each core c:
//   1. Constructs a memory view for its row at hbm_ptr + c*128 elements.
//   2. Loads it into a tensor<1x128xf16> partial.
//   3. Calls ktdp.reduce (reduce_to_core<0>, sum across axis 0).
//   4. Core 0 (pid == 0) stores the reduced tile back to HBM output.
//
// After execution, output[0..127] = sum of all 4 input rows.

#row_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @ring_reduce(%in_ptr: index, %out_ptr: index)
      attributes {grid = array<i64: 4>} {

    %c0   = arith.constant 0   : index
    %c128 = arith.constant 128 : index

    %pid = ktdp.get_compute_tile_id : index

    // Base pointer for this core's input row: in_ptr + pid * 128
    %offs = arith.muli %pid, %c128 : index
    %row_ptr = arith.addi %in_ptr, %offs : index

    // (1) Memory view over this core's 1×128 row
    %row_view = ktdp.construct_memory_view %row_ptr,
                  sizes: [1, 128], strides: [128, 1] {
      coordinate_set = #row_set,
      memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<1x128xf16>

    // (2) Access tile and load
    %row_acc = ktdp.construct_access_tile %row_view[%c0, %c0] {
      access_tile_set   = #row_set,
      access_tile_order = #identity
    } : memref<1x128xf16> -> !ktdp.access_tile<1x128xindex>

    %partial = ktdp.load %row_acc
                : !ktdp.access_tile<1x128xindex> -> tensor<1x128xf16>

    // (3) Cross-core reduce-sum: result lands on group-rank 0 (pid_k == 0).
    //     On other cores %reduced is unspecified / poison.
    %reduced = ktdp.reduce %partial {
      kind   = #ktdp.reduce_kind<sum>,
      mode   = #ktdp.reduce_mode<reduce_to_core<0>>,
      across = #ktdp.grid_axis<0>
    } : tensor<1x128xf16> -> tensor<1x128xf16>

    // (4) Only core 0 stores the result to HBM output
    %is_writer = arith.cmpi eq, %pid, %c0 : index
    scf.if %is_writer {
      %out_view = ktdp.construct_memory_view %out_ptr,
                    sizes: [1, 128], strides: [128, 1] {
        coordinate_set = #row_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
      } : memref<1x128xf16>

      %out_acc = ktdp.construct_access_tile %out_view[%c0, %c0] {
        access_tile_set   = #row_set,
        access_tile_order = #identity
      } : memref<1x128xf16> -> !ktdp.access_tile<1x128xindex>

      ktdp.store %reduced, %out_acc
        : tensor<1x128xf16>, !ktdp.access_tile<1x128xindex>
    }

    return
  }
}
