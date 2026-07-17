// Multi-group ring reduce: 16 cores split into 4 groups of 4.
//
// Grid: [16, 1, 1] — axis 0 distributes work across 16 cores.
//
// HBM addressing.  ``%in_ptr`` / ``%out_ptr`` are *element* indices for
// f16 (base_ptr convention after RFC fix).  Each 1×128 f16 row is 128
// elements.  Per-core stride = 128 elements.
//
// Each core c:
//   1. Constructs a memory view for its row at in_ptr + c*128 elements.
//   2. Loads it into a tensor<1x128xf16> partial.
//   3. Calls ktdp.inter_tile_produce + ktdp.inter_tile_reduce — every
//      core in group g (= c / 4) ends up holding the sum of that
//      group's 4 partials (all-reduce within the group).
//   4. The first core of each group (c % 4 == 0) stores the reduced
//      tile to its group's output slot at out_ptr + g*2 sticks.
//
// After execution, output[g, 0..127] = sum of group g's 4 input rows
// for g in [0, 4).
//
// This kernel is the multi-group analogue of
// ``examples/ktir/ring_reduce.mlir``; the latency-test harness uses
// it to validate that ``RingReduceBackend.run`` produces correct
// per-group results when the workgroup hosts multiple concurrent
// reductions, and that the ring still spans the whole 16-core
// workgroup.

#row_set    = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#identity   = affine_map<(d0, d1) -> (d0, d1)>

// 4 groups of 4 cores: i in [4*g, 4*g+3], with g in [0, 3].
#all_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups = affine_set<(g) : (g >= 0, -g + 3 >= 0)>

module {
  func.func @ring_reduce_multi_group(%in_ptr: index, %out_ptr: index)
      attributes {grid = [16, 1, 1]} {

    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // 1×128 f16 = 128 elements per row.
    %row_elems = arith.constant 128 : index

    %pid = ktdp.get_compute_tile_id : index

    // Element-index of this core's input row: in_ptr + pid * 128.
    %in_offs = arith.muli %pid, %row_elems : index
    %row_ptr = arith.addi %in_ptr, %in_offs : index

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
    //      ``groups`` defines 4 groups; each core's group is derived
    //      from its position in ``producer_tiles_per_group``.
    %fut = ktdp.inter_tile_produce
        producer_tiles_per_group = #all_tiles,
        groups                   = #all_groups
        : tensor<1x128xf16> -> !ktdp.tile_future<tensor<1x128xf16>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial : tensor<1x128xf16>
    }

    // (3b) Reduce: within each group of 4, every core holds the same
    //      group sum.  consumer_tiles_per_group = producer_tiles_per_group
    //      → in-group all-reduce.
    %c_zero    = arith.constant 0.0 : f16
    %id_init   = tensor.empty() : tensor<1x128xf16>
    %add_id    = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<1x128xf16>)
                   -> tensor<1x128xf16>

    %reduced = ktdp.inter_tile_reduce(%fut)
        consumer_tiles_per_group = #all_tiles,
        groups                   = #all_groups,
        identity(%add_id : tensor<1x128xf16>)
        : !ktdp.tile_future<tensor<1x128xf16>> -> tensor<128xf16>
    {
      ^bb0(%lhs: tensor<1x128xf16>, %rhs: tensor<1x128xf16>):
        %init = tensor.empty() : tensor<1x128xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x128xf16>, tensor<1x128xf16>)
                           outs(%init : tensor<1x128xf16>) -> tensor<1x128xf16>
        ktdp.yield_reduced %sum : tensor<1x128xf16>
    }

    // (4) The first core of each group writes its group's result.
    //     Writer condition: pid % 4 == 0.  Output offset: g * 128 elements
    //     where g = pid / 4.  So core 0 writes to out_ptr+0, core 4
    //     writes to out_ptr+128, core 8 writes to out_ptr+256, core 12
    //     writes to out_ptr+384.
    %lane      = arith.remui %pid, %c4 : index
    %is_writer = arith.cmpi eq, %lane, %c0 : index
    scf.if %is_writer {
      %reduced_2d = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 128]
                      : tensor<128xf16> into tensor<1x128xf16>

      %group_idx = arith.divui %pid, %c4 : index
      %out_offs  = arith.muli %group_idx, %row_elems : index
      %group_out = arith.addi %out_ptr, %out_offs : index

      %out_view = ktdp.construct_memory_view %group_out,
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
