module {
  func.func @_layer_norm_fwd_fused(
      %X: index, // input 1152x8192xf16
      %Y: index, // output 1152x8192xf16
      %W: index, // weight 1152x8192xf16
      %B: index, // bias 1152x8192xf16
      %Mean: index, // mean 1152xf16
      %Rstd: index, // rstd 1152xf16
      %eps: f16
  ) attributes {grid = [32]} {
    %core_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c36 = arith.constant 36 : index
    %c1152 = arith.constant 1152 : index
    %c1024 = arith.constant 1024 : index
    %c8192 = arith.constant 8192 : index
    %f1_f16 = arith.constant 1.0 : f16

    %start_row = arith.muli %core_id, %c36 : index
    %end_row = arith.addi %start_row, %c36 : index

    %X_view = ktdp.construct_memory_view %X, sizes: [1152, 8192], strides: [8192, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1151 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1152x8192xf16>

    %Y_view = ktdp.construct_memory_view %Y, sizes: [1152, 8192], strides: [8192, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1151 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1152x8192xf16>

    %W_view = ktdp.construct_memory_view %W, sizes: [1152, 8192], strides: [8192, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1151 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1152x8192xf16>

    %B_view = ktdp.construct_memory_view %B, sizes: [1152, 8192], strides: [8192, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1151 >= 0, d1 >= 0, -d1 + 8191 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1152x8192xf16>

    %Mean_view = ktdp.construct_memory_view %Mean, sizes: [1152], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 1151 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1152xf16>

    %Rstd_view = ktdp.construct_memory_view %Rstd, sizes: [1152], strides: [1] {
        coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 1151 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<1152xf16>

    scf.for %row = %start_row to %end_row step %c1 : index {

        %zero_block = arith.constant dense<0.0> : tensor<1x1024xf16>

        %sum_block = scf.for %col = %c0 to %c8192 step %c1024
            iter_args(%sum_iter = %zero_block) -> tensor<1x1024xf16> {

            %X_acc = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<1152x8192xf16> -> !ktdp.access_tile<1x1024xindex>

            %a = ktdp.load %X_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            %sum_next = arith.addf %sum_iter, %a : tensor<1x1024xf16>

            scf.yield %sum_next : tensor<1x1024xf16>
        }

        %zero_scalar = arith.constant 0.0 : f16
        %sum_init = tensor.splat %zero_scalar : tensor<1xf16>
        %reduce_sum = linalg.reduce { arith.addf }
            ins(%sum_block : tensor<1x1024xf16>)
            outs(%sum_init : tensor<1xf16>)
            dimensions = [1]

        %N_i32 = arith.index_cast %c8192 : index to i32
        %N_f16 = arith.sitofp %N_i32 : i32 to f16
        %N_tensor = tensor.splat %N_f16 : tensor<1xf16>
        %mean = arith.divf %reduce_sum, %N_tensor : tensor<1xf16>

        %mean_scalar = tensor.extract %mean[%c0] : tensor<1xf16>
        %mean_block = tensor.splat %mean_scalar : tensor<1x1024xf16>

        %var_block = scf.for %col = %c0 to %c8192 step %c1024
            iter_args(%var_iter = %zero_block) -> tensor<1x1024xf16> {

            %X_acc = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<1152x8192xf16> -> !ktdp.access_tile<1x1024xindex>

            %a = ktdp.load %X_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            %x = arith.subf %a, %mean_block : tensor<1x1024xf16>

            %x2 = arith.mulf %x, %x : tensor<1x1024xf16>

            %var_next = arith.addf %var_iter, %x2 : tensor<1x1024xf16>

            scf.yield %var_next : tensor<1x1024xf16>
        }

        %var_init = tensor.splat %zero_scalar : tensor<1xf16>
        %reduce_var = linalg.reduce { arith.addf }
            ins(%var_block : tensor<1x1024xf16>)
            outs(%var_init : tensor<1xf16>)
            dimensions = [1]

        %var = arith.divf %reduce_var, %N_tensor : tensor<1xf16>

        %eps_tensor = tensor.splat %eps : tensor<1xf16>
        %var_plus_eps = arith.addf %var, %eps_tensor : tensor<1xf16>

        %var_sqrt = math.sqrt %var_plus_eps : tensor<1xf16>

        %f1_tensor = tensor.splat %f1_f16 : tensor<1xf16>
        %rstd = arith.divf %f1_tensor, %var_sqrt : tensor<1xf16>

        %rstd_scalar = tensor.extract %rstd[%c0] : tensor<1xf16>
        %rstd_block = tensor.splat %rstd_scalar : tensor<1x1024xf16>

        %Mean_acc = ktdp.construct_access_tile %Mean_view[%row] {
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 0 >= 0)>,
            access_tile_order = affine_map<(d0) -> (d0)>
        } : memref<1152xf16> -> !ktdp.access_tile<1xindex>

        %Rstd_acc = ktdp.construct_access_tile %Rstd_view[%row] {
            access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 0 >= 0)>,
            access_tile_order = affine_map<(d0) -> (d0)>
        } : memref<1152xf16> -> !ktdp.access_tile<1xindex>

        ktdp.store %mean, %Mean_acc : tensor<1xf16>, !ktdp.access_tile<1xindex>
        ktdp.store %rstd, %Rstd_acc : tensor<1xf16>, !ktdp.access_tile<1xindex>

        scf.for %col = %c0 to %c8192 step %c1024 {

            %W_acc = ktdp.construct_access_tile %W_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<1152x8192xf16> -> !ktdp.access_tile<1x1024xindex>

            %B_acc = ktdp.construct_access_tile %B_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<1152x8192xf16> -> !ktdp.access_tile<1x1024xindex>

            %X_acc = ktdp.construct_access_tile %X_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<1152x8192xf16> -> !ktdp.access_tile<1x1024xindex>

            %w = ktdp.load %W_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
            %b = ktdp.load %B_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
            %x = ktdp.load %X_acc : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>

            %x_minus_mean = arith.subf %x, %mean_block : tensor<1x1024xf16>
            %x_hat = arith.mulf %x_minus_mean, %rstd_block : tensor<1x1024xf16>

            %x_hat_mul_w = arith.mulf %x_hat, %w : tensor<1x1024xf16>

            %y = arith.addf %x_hat_mul_w, %b : tensor<1x1024xf16>

            %Y_acc = ktdp.construct_access_tile %Y_view[%row, %col] {
                access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 1023 >= 0)>,
                access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
            } : memref<1152x8192xf16> -> !ktdp.access_tile<1x1024xindex>

            ktdp.store %y, %Y_acc : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

            scf.yield
        }
        scf.yield
    }
    return
  }
}
