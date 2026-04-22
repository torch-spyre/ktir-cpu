// RFC 0682 Section C.5, Example 1 — construct_indirect_access_tile
// This is an example program to illustrate the use of ktdp.construct_indirect_access_tile
// The program makes a tensor copy from an input X into an output Y using two index tensors IDX1 and IDX2.
// Y[m, k] = X[ IDX1[m, k], IDX2[m, k] ]

// X input  = 2D tensor, shape {64, 64}
// IDX1     = 2D tensor, shape {64, 64}, provides row indices into X
// IDX2     = 2D tensor, shape {64, 64}, provides column indices into X
// Y output = 2D tensor, shape {64, 64}

#X_coord_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#IDX_coord_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#Y_coord_set   = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#var_space_order = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @indirect_access_copy() {
        // In this example, all tensors X, IDX1, IDX2, Y are in a single memory space (HBM)
        %X_addr    = arith.constant 0 : index
        %IDX1_addr = arith.constant 8192 : index
        %IDX2_addr = arith.constant 16384 : index
        %Y_addr    = arith.constant 24576 : index

        // Accessing a tensor in KTIR follows a 3 step process:
        // Note1: Accesses are single-ended i.e.,
        //        load = read from memory to produce a data-tile
        //        store = write a data-tile to memory
        // Note2: To accomplish a data-transfer (or copy) from source to destination,
        //        we need a separate load from source and a store to destination
        // (1) Create memory view: Informs how the tensor is present in memory
        // (2) Create access tile: Informs the logical coordinates of the tensor that must be accessed
        //    Note: For indirect access, an auxiliary index tensor supplies the coordinates per dimension
        // (3) Create data tile: Extract a sub-portion of the tensor corresponding to the coordinates present in the access tile

        // (1) Construct memory view for X
        // Note: number of entries in sizes, strides, dims in coordinate_set, shape of memref must be identical
        %X = ktdp.construct_memory_view %X_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #X_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xf16>

        // (1) Construct memory view for IDX1 (row indices into X)
        %IDX1 = ktdp.construct_memory_view %IDX1_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #IDX_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xi32>

        // (1) Construct memory view for IDX2 (column indices into X)
        %IDX2 = ktdp.construct_memory_view %IDX2_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #IDX_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xi32>

        // (1) Construct memory view for output Y
        %Y_view = ktdp.construct_memory_view %Y_addr, sizes: [64, 64], strides: [64, 1] {
            coordinate_set = #Y_coord_set,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<64x64xf16>

        // (2) Construct indirect access tile for X[IDX1[m, k], IDX2[m, k]]
        // Note: Number of entries in intermediate_variables and var_space_set must be equal
        // Note: Number of entries in subscript of mem_view, shape of memref, shape of ktdp.access_tile must be equal
        %X_access_tile = ktdp.construct_indirect_access_tile
            intermediate_variables(%m, %k)
            %X[ind(%IDX1[%m, %k]), ind(%IDX2[%m, %k])] {
                variables_space_set = #var_space_set,
                variables_space_order = #var_space_order
            } : memref<64x64xf16>, memref<64x64xi32>, memref<64x64xi32> -> !ktdp.access_tile<64x64xindex>

        // (2) Construct direct access tile for Y[m, k]
        // Note: No need for intermediate_variables in directly accessed tiles
        %c0 = arith.constant 0 : index
        %Y_access_tile = ktdp.construct_access_tile %Y_view[%c0, %c0] {
            access_tile_set = #Y_coord_set,
            access_tile_order = #var_space_order
        } : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>

        // (3) Load data tile from X using the indirect access tile
        %X_data_tile = ktdp.load %X_access_tile : !ktdp.access_tile<64x64xindex> -> tensor<64x64xf16>

        // (4) Store Y[m, k] = X_data_tile
        ktdp.store %X_data_tile, %Y_access_tile : tensor<64x64xf16>, !ktdp.access_tile<64x64xindex>

        return
  }
}
