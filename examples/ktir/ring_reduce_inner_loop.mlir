// Ring reduce inside scf.for: 4 cores, each holds a 1×128 row in HBM.
// The loop runs K iterations; each iteration does an all-reduce sum across
// all 4 cores and accumulates into a running total.  After K iterations the
// accumulated result is written back by core 0.
//
// This exercises ktdp.inter_tile_produce + ktdp.inter_tile_reduce placed
// inside an scf.for body — the main structural difference from ring_reduce.mlir
// where the reduce appears at the top level of the function.
//
// Grid: [4, 1, 1] — axis 0 distributes work.
//
// HBM addressing.  One HBM stick = 128 bytes = 64 f16 elements; each 1×128
// f16 row spans 2 sticks.  Per-core stride = 2 sticks.
//
// Each core c:
//   1. Constructs a memory view for its row at in_ptr + c*2 sticks.
//   2. Loads it into a tensor<1x128xf16> partial.
//   3. Loops K times, each iteration doing a full ring all-reduce of the
//      partial, accumulating the result into %acc.
//   4. Core 0 (pid == 0) stores the final %acc back to HBM output.
//
// After execution, output[0..127] = K * sum(all 4 input rows).

#row_set    = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0)>
#identity   = affine_map<(d0, d1) -> (d0, d1)>

#all_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group  = affine_set<(g) : (g == 0)>

module {
  func.func @ring_reduce_inner_loop(%in_ptr: index, %out_ptr: index, %n_iters: index)
      attributes {grid = [4]} {

    %c0         = arith.constant 0 : index
    %c1         = arith.constant 1 : index
    %row_sticks = arith.constant 2 : index

    %pid = ktdp.get_compute_tile_id : index

    // Stick-index of this core's input row: in_ptr + pid * 2.
    %offs    = arith.muli %pid, %row_sticks : index
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

    // Zero-initialised accumulator — iter_arg carried across loop iterations.
    %c_zero  = arith.constant 0.0 : f16
    %acc_init = tensor.empty() : tensor<1x128xf16>
    %zero_acc = linalg.fill ins(%c_zero : f16) outs(%acc_init : tensor<1x128xf16>)
                  -> tensor<1x128xf16>

    // (3) Loop K times; each iteration all-reduces %partial and adds to %acc.
    %acc_final = scf.for %k = %c0 to %n_iters step %c1
        iter_args(%acc = %zero_acc) -> (tensor<1x128xf16>) {

      // (3a) Produce: every core contributes its partial for this round.
      %fut = ktdp.inter_tile_produce
          producer_tiles_per_group = #all_tiles,
          groups                   = #one_group
          : tensor<1x128xf16> -> !ktdp.tile_future<tensor<1x128xf16>>
      {
        ^bb0(%gid: index):
          ktdp.yield_partial %partial : tensor<1x128xf16>
      }

      // (3b) Reduce across all 4 cores.
      %add_id = linalg.fill ins(%c_zero : f16) outs(%acc_init : tensor<1x128xf16>)
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

      // Accumulate: acc += reduced (broadcast reduced back to 1×128).
      %reduced_2d = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 128]
                      : tensor<128xf16> into tensor<1x128xf16>
      %new_acc = tensor.empty() : tensor<1x128xf16>
      %acc_sum = linalg.add ins(%acc, %reduced_2d : tensor<1x128xf16>, tensor<1x128xf16>)
                            outs(%new_acc : tensor<1x128xf16>) -> tensor<1x128xf16>

      scf.yield %acc_sum : tensor<1x128xf16>
    }

    // (4) Only core 0 writes the accumulated result back to HBM.
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

      ktdp.store %acc_final, %out_acc
        : tensor<1x128xf16>, !ktdp.access_tile<1x128xindex>
    }

    return
  }
}
