// Ring reduce: 4 cores, each holds a 1×128 row in HBM, reduce-sum across all 4.
//
// Grid: [4, 1, 1] — axis 0 distributes work.
//
// HBM addressing.  ``%in_ptr`` / ``%out_ptr`` are *element* indices for
// f16 (base_ptr convention after RFC fix).  Each 1×128 f16 row is 128
// elements.  Per-core stride = 128 elements.
//
// Each core c:
//   1. Constructs a memory view for its row at in_ptr + c*128 elements.
//   2. Loads it into a tensor<1x128xf16> partial.
//   3. Calls ktdp.inter_tile_produce + ktdp.inter_tile_reduce — every
//      core ends up holding the full sum (all-reduce).
//   4. Core 0 (pid == 0) stores the reduced tile back to HBM output.
//
// After execution, output[0..127] = sum of all 4 input rows.

#row_set    = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#identity   = affine_map<(d0, d1) -> (d0, d1)>

// Single group containing all 4 cores: i in [4*g, 4*g+3], with g == 0.
#all_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group  = affine_set<(g) : (g == 0)>

module {
  func.func @ring_reduce(%in_ptr: index, %out_ptr: index)
      attributes {grid = [4]} {

    %c0   = arith.constant 0 : index
    // 1×128 f16 = 128 elements per row.
    %row_elems = arith.constant 128 : index

    %pid = ktdp.get_compute_tile_id : index

    // Element-index of this core's input row: in_ptr + pid * 128.
    %offs = arith.muli %pid, %row_elems : index
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

    // (3a) Produce: every core contributes its 1×128 partial.
    %fut = ktdp.inter_tile_produce
        producer_tiles_per_group = #all_tiles,
        groups                   = #one_group
        : tensor<1x128xf16> -> !ktdp.tile_future<tensor<1x128xf16>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial : tensor<1x128xf16>
    }

    // (3b) Reduce: every core (consumer set == producer set) ends up holding
    //      the same 128-element group sum.
    %c_zero    = arith.constant 0.0 : f16
    %id_init   = tensor.empty() : tensor<1x128xf16>
    %add_id    = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<1x128xf16>)
                   -> tensor<1x128xf16>

    %reduced = ktdp.inter_tile_reduce(%fut)
        consumer_tiles_per_group = #all_tiles,
        groups                   = #one_group,
        identity(%add_id : tensor<1x128xf16>)
        : !ktdp.tile_future<tensor<1x128xf16>> -> tensor<128xf16>
    {
      ^bb0(%lhs: tensor<1x128xf16>, %rhs: tensor<1x128xf16>):
        %init = tensor.empty() : tensor<1x128xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x128xf16>, tensor<1x128xf16>)
                           outs(%init : tensor<1x128xf16>) -> tensor<1x128xf16>
        ktdp.yield_reduced %sum : tensor<1x128xf16>
    }

    // (4) Only core 0 stores the result back as a 1×128 row
    %is_writer = arith.cmpi eq, %pid, %c0 : index
    scf.if %is_writer {
      %reduced_2d = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 128]
                      : tensor<128xf16> into tensor<1x128xf16>

      %out_view = ktdp.construct_memory_view %out_ptr,
                    sizes: [1, 128], strides: [128, 1] {
        coordinate_set = #row_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
      } : memref<1x128xf16>

      %out_acc = ktdp.construct_access_tile %out_view[%c0, %c0] {
        access_tile_set   = #row_set,
        access_tile_order = #identity
      } : memref<1x128xf16> -> !ktdp.access_tile<1x128xindex>

      ktdp.store %reduced_2d, %out_acc
        : tensor<1x128xf16>, !ktdp.access_tile<1x128xindex>
    }

    return
  }
}
