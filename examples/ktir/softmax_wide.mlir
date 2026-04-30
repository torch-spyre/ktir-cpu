module {
  func.func @softmax_kernel(
      %output_ptr: index,
      %input_ptr: index
  ) attributes {grid = [1]} {
    %core_id = ktdp.get_compute_tile_id : index
    %c_chunk = arith.constant 2 : index  // chunk = ceil(R/K) = ceil(2/1)
    %c_R = arith.constant 2 : index  // R = total rows
    %c1 = arith.constant 1 : index
    %start = arith.muli %core_id, %c_chunk : index  // start = core_id * chunk
    %end_raw = arith.addi %start, %c_chunk : index  // end_raw = start + chunk
    %cmp = arith.cmpi slt, %end_raw, %c_R : index  // end_raw < R?
    %end = arith.select %cmp, %end_raw, %c_R : index  // end = min(end_raw, R)
    %input_view_1 = ktdp.construct_memory_view %input_ptr, sizes: [2, 262144], strides: [262144, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 262143 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x262144xf16>

    %output_view_2 = ktdp.construct_memory_view %output_ptr, sizes: [2, 262144], strides: [262144, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 262143 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x262144xf16>

    scf.for %row = %start to %end step %c1  : index {
      %c0 = arith.constant 0 : index
      %input_acc_1 = ktdp.construct_access_tile %input_view_1[%row, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 262143 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<2x262144xf16> -> !ktdp.access_tile<1x262144xindex>
      %row_13 = ktdp.load %input_acc_1 : !ktdp.access_tile<1x262144xindex> -> tensor<1x262144xf16>
      
      %neg_inf = arith.constant 0xFF80 : f16
      %max_init = tensor.splat %neg_inf : tensor<1xf16>
      %row_max = linalg.reduce { arith.maxnumf }
          ins(%row_13 : tensor<1x262144xf16>)
          outs(%max_init : tensor<1xf16>)
          dimensions = [1]
      
      %max_scalar = tensor.extract %row_max[%c0] : tensor<1xf16>
      %row_minus_max_15 = tensor.splat %max_scalar : tensor<1x262144xf16>
      %row_minus_max_16 = arith.subf %row_13, %row_minus_max_15 : tensor<1x262144xf16>
      %numerator = math.exp %row_minus_max_16 : tensor<1x262144xf16>
      
      %zero = arith.constant 0.0 : f16
      %sum_init = tensor.splat %zero : tensor<1xf16>
      %denominator = linalg.reduce { arith.addf }
          ins(%numerator : tensor<1x262144xf16>)
          outs(%sum_init : tensor<1xf16>)
          dimensions = [1]
      
      %denom_scalar = tensor.extract %denominator[%c0] : tensor<1xf16>
      %softmax_output = tensor.splat %denom_scalar : tensor<1x262144xf16>
      %softmax_output_17 = arith.divf %numerator, %softmax_output : tensor<1x262144xf16>
      %output_acc_2 = ktdp.construct_access_tile %output_view_2[%row, %c0] {
          access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 262143 >= 0)>,
          access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<2x262144xf16> -> !ktdp.access_tile<1x262144xindex>
      ktdp.store %softmax_output_17, %output_acc_2 : tensor<1x262144xf16>, !ktdp.access_tile<1x262144xindex>
      scf.yield
    }
    return
  }
}
