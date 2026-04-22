module {
  func.func @matmul_kernel(
      %a_ptr: index, // [M, K]
      %b_ptr: index, // [K, N]
      %c_ptr: index, // [M, N]
      %K: index, // 2048
      %BLOCK_SIZE_M: index, // 32
      %BLOCK_SIZE_N: index, // 512
      %BLOCK_SIZE_K: index  // 128
  ) attributes {grid = [2, 16]} {
    %pid_m, %pid_n = ktdp.get_compute_tile_id : index, index
  
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index

    %a_view = ktdp.construct_memory_view %a_ptr, sizes: [64, 2048], strides: [2048, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 2047 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x2048xf16>

    %b_view = ktdp.construct_memory_view %b_ptr, sizes: [2048, 8192], strides: [8192, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2047 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2048x8192xf16>

    %c_view = ktdp.construct_memory_view %c_ptr, sizes: [64, 8192], strides: [8192, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<64x8192xf16>

    %offs_am = arith.muli %pid_m, %BLOCK_SIZE_M : index
    %offs_bn = arith.muli %pid_n, %BLOCK_SIZE_N : index
   
    %accum_zero = arith.constant dense<0.0> : tensor<32x512xf16>

    %c = scf.for %off_k = %c0 to %K step %BLOCK_SIZE_K iter_args(%accum_itr = %accum_zero) -> (tensor<32x512xf16>) {

      // A tile: 32 x 128 = BLOCK_SIZE_M x BLOCK_SIZE_K
      %a_acc = ktdp.construct_access_tile %a_view[%offs_am, %off_k] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 127 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<64x2048xf16> -> !ktdp.access_tile<32x128xindex>

      // B tile: 128 x 512 = BLOCK_SIZE_K x BLOCK_SIZE_N
      %b_acc = ktdp.construct_access_tile %b_view[%off_k, %offs_bn] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<2048x8192xf16> -> !ktdp.access_tile<128x512xindex>

      %a = ktdp.load %a_acc : !ktdp.access_tile<32x128xindex> -> tensor<32x128xf16>
      %b = ktdp.load %b_acc : !ktdp.access_tile<128x512xindex> -> tensor<128x512xf16>

      %c_init = tensor.empty() : tensor<32x512xf16>
      %a_dot_b = linalg.matmul ins(%a, %b : tensor<32x128xf16>, tensor<128x512xf16>)
                               outs(%c_init : tensor<32x512xf16>) -> tensor<32x512xf16>

      %accum_next = arith.addf %accum_itr, %a_dot_b : tensor<32x512xf16>

      scf.yield %accum_next : tensor<32x512xf16>
    }

    %c_acc = ktdp.construct_access_tile %c_view[%offs_am, %offs_bn] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 511 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<64x8192xf16> -> !ktdp.access_tile<32x512xindex>
  
    ktdp.store %c, %c_acc : tensor<32x512xf16>, !ktdp.access_tile<32x512xindex>

    return
  }
}
