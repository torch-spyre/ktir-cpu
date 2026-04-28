module {
  func.func @softmax_kernel_small(
      %output_ptr: index,
      %input_ptr: index
  ) attributes {grid = [32, 1]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index

    %start_row = arith.muli %core_id, %c2 : index
    %end_row = arith.addi %start_row, %c2 : index

    %input_view = ktdp.construct_memory_view %input_ptr, sizes: [64, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x64xf16>

    %output_view = ktdp.construct_memory_view %output_ptr, sizes: [64, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x64xf16>

    scf.for %row = %start_row to %end_row step %c1 : index {

      %input_acc = ktdp.construct_access_tile %input_view[%row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>

      %input_row = ktdp.load %input_acc : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

      %neg_inf = arith.constant 0xFF80 : f16
      %max_init = tensor.splat %neg_inf : tensor<1xf16>
      %reduce_max = linalg.reduce { arith.maximumf }
        ins(%input_row : tensor<1x64xf16>)
        outs(%max_init : tensor<1xf16>)
        dimensions = [1]

      %max_scalar = tensor.extract %reduce_max[%c0] : tensor<1xf16>
      %max_row = tensor.splat %max_scalar : tensor<1x64xf16>

      %input_minus_max = arith.subf %input_row, %max_row : tensor<1x64xf16>

      %numerator = math.exp %input_minus_max : tensor<1x64xf16>

      %zero = arith.constant 0.0 : f16
      %sum_init = tensor.splat %zero : tensor<1xf16>
      %reduce_add = linalg.reduce { arith.addf }
        ins(%numerator : tensor<1x64xf16>)
        outs(%sum_init : tensor<1xf16>)
        dimensions = [1]

      %denom_scalar = tensor.extract %reduce_add[%c0] : tensor<1xf16>
      %denominator_row = tensor.splat %denom_scalar : tensor<1x64xf16>

      %softmax_output = arith.divf %numerator, %denominator_row : tensor<1x64xf16>

      %output_acc = ktdp.construct_access_tile %output_view[%row, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>

      ktdp.store %softmax_output, %output_acc : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

      scf.yield
    }
    return
  }
}
