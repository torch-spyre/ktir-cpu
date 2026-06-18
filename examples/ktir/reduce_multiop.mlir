// linalg.reduce with a MULTI-OP combiner region: max expressed as
// cmpf(ogt) + select rather than a single arith.maximumf.  This exercises the
// general tree fold — every op in the region is executed and charged, and the
// result does not depend on recognising a single combiner op by name.
module {
  func.func @reduce_multiop(%arg0: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFC00 : f16  // -inf (identity for max)
    %view = ktdp.construct_memory_view %arg0, sizes : [1, 8], strides : [8, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x8xf16>
    %acc = ktdp.construct_access_tile %view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x8xf16> -> !ktdp.access_tile<1x8xindex>
    %data = ktdp.load %acc : !ktdp.access_tile<1x8xindex> -> tensor<1x8xf16>

    %init = tensor.empty() : tensor<1xf16>
    %init_filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1xf16>) -> tensor<1xf16>

    %reduced = linalg.reduce ins(%data : tensor<1x8xf16>) outs(%init_filled : tensor<1xf16>) dimensions = [1]
      (%in: f16, %out: f16) {
        %cmp = arith.cmpf ogt, %in, %out : f16
        %max = arith.select %cmp, %in, %out : f16
        linalg.yield %max : f16
      }

    %scalar = tensor.extract %reduced[%c0] : tensor<1xf16>
    %splat = tensor.splat %scalar : tensor<1x8xf16>
    %out_view = ktdp.construct_memory_view %arg0, sizes : [1, 8], strides : [8, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x8xf16>
    %out_acc = ktdp.construct_access_tile %out_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 7 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x8xf16> -> !ktdp.access_tile<1x8xindex>
    ktdp.store %splat, %out_acc : tensor<1x8xf16>, !ktdp.access_tile<1x8xindex>
    return
  }
}
