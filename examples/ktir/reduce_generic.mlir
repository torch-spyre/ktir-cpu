module {
  func.func @reduce_explicit_region(%arg0: index) attributes {grid = [1, 1]} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %view = ktdp.construct_memory_view %arg0, sizes : [1, 4], strides : [4, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x4xf16>
    %acc = ktdp.construct_access_tile %view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x4xf16> -> !ktdp.access_tile<1x4xindex>
    %data = ktdp.load %acc : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf16>

    %init = tensor.empty() : tensor<1xf16>
    %init_filled = linalg.fill ins(%cst : f16) outs(%init : tensor<1xf16>) -> tensor<1xf16>

    %reduced = linalg.reduce ins(%data : tensor<1x4xf16>) outs(%init_filled : tensor<1xf16>) dimensions = [1]
      (%in: f16, %out: f16) {
        %sum = arith.addf %in, %out : f16
        linalg.yield %sum : f16
      }

    // Store result (broadcast to 1x4 for simplicity)
    // linalg.reduce generic format returns tensor<1xf16>; extract scalar before splat
    %scalar = tensor.extract %reduced[%c0] : tensor<1xf16>
    %splat = tensor.splat %scalar : tensor<1x4xf16>
    %out_view = ktdp.construct_memory_view %arg0, sizes : [1, 4], strides : [4, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x4xf16>
    %out_acc = ktdp.construct_access_tile %out_view[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x4xf16> -> !ktdp.access_tile<1x4xindex>
    ktdp.store %splat, %out_acc : tensor<1x4xf16>, !ktdp.access_tile<1x4xindex>
    return
  }
}
