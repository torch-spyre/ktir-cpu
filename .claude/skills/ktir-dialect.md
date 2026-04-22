# KTIR Dialect Reference

> **RFC last updated: 2026-03-16.** Check [RFC 0682](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md) occasionally for updates and re-sync this file if the spec has changed.

The Kernel Tile Intermediate Representation (KTIR) is a tile-based, block-structured IR for multi-core accelerator architectures. It uses the `ktdp` dialect for memory/access abstractions and reuses standard MLIR dialects (Arith, Math, LinAlg, SCF, Tensor, MemRef) for compute and control flow.

## Hardware Abstraction

KTIR targets accelerators with:
- Multiple cores, each with a compute engine and on-chip scratchpad memory (LX)
- Cores connected via on-chip interconnect fabric
- One or more off-chip memory banks (HBM)

Tensors can be distributed across memory elements. Compute operations are split into tiles assigned to cores. Each tile has global view of all memory via the interconnect.

## Operations in KTIR

### 1. Memory and Layout Operations (`ktdp` dialect)
- Memory views: logical views over allocated memory regions
- Access tiles: structured coordinate collections for memory access
- Load/store: explicit data movement primitives

### 2. Compute Operations
Reuses MLIR dialects:
- **Arith**: arithmetic operations (https://mlir.llvm.org/docs/Dialects/ArithOps/)
- **Math**: mathematical functions (https://mlir.llvm.org/docs/Dialects/MathOps/)
- **LinAlg**: linear algebra (https://mlir.llvm.org/docs/Dialects/Linalg/)

### 3. Control Flow Operations (SCF dialect)
`scf.for`, `scf.if`, `scf.yield`, `scf.reduce`, `scf.reduce.return`, `scf.parallel`, `scf.forall`

### 4. Miscellaneous
- **Tensor**: `tensor.extract_slice` etc. (https://mlir.llvm.org/docs/Dialects/TensorOps/)
- **MemRef**: `memref.subview` etc. (https://mlir.llvm.org/docs/Dialects/MemRef/)

---

## `ktdp` Dialect Operations

### 1. `ktdp.get_compute_tile_id`

Returns the multidimensional identifier of the currently executing compute tile.

**Syntax:**
```mlir
operation ::= `ktdp.get_compute_tile_id` attr-dict `:` type(results)
```

**Examples:**
```mlir
// 1D grid
%tile_id = ktdp.get_compute_tile_id : index

// 2D grid
%tile_id:2 = ktdp.get_compute_tile_id : index, index
```

**Results:** `result` — index

---

### 2. `ktdp.construct_memory_view`

Creates a memref view over allocated memory at a given base address. Does not allocate memory — only constructs a logical view.

**Syntax:**
```mlir
operation ::= `ktdp.construct_memory_view` $offset `,` `sizes` `:`
              custom<DynamicIndexList>($sizes, $static_sizes)
              `,` `strides` `:`
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($result)
```

**Attributes:**
| Attribute | MLIR Type | Description |
|-----------|-----------|-------------|
| `static_sizes` | `::mlir::DenseI64ArrayAttr` | i64 dense array |
| `static_strides` | `::mlir::DenseI64ArrayAttr` | i64 dense array |
| `memory_space` | `::mlir::Attribute` | Physical memory space (e.g., HBM, LX) |
| `coordinate_set` | `::mlir::IntegerSetAttr` | Coordinate subset for this view |

**Operands:** `offset` (index), `sizes` (variadic index), `strides` (variadic index)

**Results:** `result` — memref of any type values

**Example:**
```mlir
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 64 >= 0)>
%A_view = ktdp.construct_memory_view %A_start_address, sizes: [32, 64], strides: [64, 1] {
    coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>
} : memref<32x64xf16>
```

**Traits:** `AttrSizedOperandSegments`, `MemRefsNormalizable`

---

### 3. `ktdp.construct_distributed_memory_view`

Constructs a single logical memref view by composing multiple per-partition memory views. Does not allocate or move data.

**Syntax:**
```mlir
operation ::= `ktdp.construct_distributed_memory_view` `(` ($memrefs^ `:` type($memrefs))? `)` attr-dict `:` type(results)
```

**Operands:** `memrefs` — variadic of memref of any type values
**Results:** memref of any type values

**Example:**
```mlir
%A_dview = ktdp.construct_distributed_memory_view (%A0_view, %A1_view : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>
```

The global coordinate domain is the union of the `coordinate_set` attributes from input views. If coordinate sets overlap, behavior is unspecified unless constrained by dialect rules.

---

### 4. `ktdp.construct_access_tile`

Constructs a structured subset of indices into a memref/tensor. Separates address computation from data access.

**Attributes:**
| Attribute | MLIR Type | Description |
|-----------|-----------|-------------|
| `base_map` | `::mlir::AffineMapAttr` | AffineMap attribute |
| `access_tile_set` | `::mlir::IntegerSetAttr` | Coordinate domain (IntegerSet) |
| `access_tile_order` | `::mlir::AffineMapAttr` | Traversal order (rightmost = innermost) |

**Operands:** `base` (memref or tensor), `indices` (variadic index)
**Results:** `result` — `!ktdp.access_tile<NxMxindex>`

**Example:**
```mlir
%A_access_tile = ktdp.construct_access_tile %A_view[%c0, %c0] {
    access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
    access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
} : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>
```

The `access_tile_set` can represent general polyhedral regions (rectangular, skewed, strided, triangular). The `access_tile_order` follows lexicographic semantics where the rightmost output dimension is the innermost iteration dimension.

---

### 5. `ktdp.construct_indirect_access_tile`

Constructs an access tile for indirect (gather/scatter) indexing. Uses memory views as index sources per dimension.

**Attributes:**
| Attribute | MLIR Type | Description |
|-----------|-----------|-------------|
| `memory_view_subscripts` | `::mlir::ArrayAttr` | Affine subscript per dimension |
| `variables_space_set` | `::mlir::IntegerSetAttr` | Domain of intermediate variables |
| `variables_space_order` | `::mlir::AffineMapAttr` | Iteration order over variable space |

**Operands:** `base` (memref or tensor), `memory_view_names` (variadic index), `common_variables` (variadic index)
**Results:** `result` — access_tile

**Traits:** `AttrSizedOperandSegments`

**Example (simple indirect):**
Pattern: `Y[m, k] = X[ IDX1[m, k], IDX2[m, k] ]`
```mlir
%Y = ktdp.construct_indirect_access_tile intermediate_variables (%m, %k) %X[%IDX1[%m, %k], %IDX2[%m, %k]] {
    variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
    variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
} : memref<64x64xfp16>, memref<64x64xfp16>, memref<64x64xfp16>, !ktdp.access_tile<64x64xindex>
```

**Example (paged attention):**
Pattern: `X[Idx[b][tkv/64], hkv, tkv % 64, dkv]`
```mlir
#X_var_space_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 3 >= 0,
                                          d1 >= 0, -d1 + 7 >= 0,
                                          d2 >= 0, -d2 + 2047>= 0,
                                          d3 >= 0, -d3 + 127 >= 0)>
#X_var_space_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
%X_access_tile = ktdp.construct_indirect_access_tile
                    intermediate_variables(%b, %h, %tkv, %dkv)
                    %X_mem_view[Idx_mem_view[%b, %tkv / 64] , %h, %tkv % 64, %dkv] {
    variables_space_set = #X_var_space_set,
    variables_space_order = #X_var_space_order
} : memref<10000x8x64x128xf16>,memref<4x32xi32> -> !ktdp.access_tile<4x8x2048x128xindex>
```

**Key constraint:** Indirect subscripts must appear as standalone index values — they cannot be combined multiplicatively with other variables. Dimension fusion is not supported when one fused dimension is accessed indirectly.

---

### 6. `ktdp.load`

Reads data using an access tile's coordinates. Produces a tensor.

**Syntax:**
```mlir
operation ::= `ktdp.load` $access_tile attr-dict `:` type($access_tile) `->` type(results)
```

**Operands:** `access_tile` — `!ktdp.access_tile<NxMxindex>`
**Results:** `result` — tensor

**Example:**
```mlir
%A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>
```

---

### 7. `ktdp.store`

Writes data to coordinates specified by an access tile.

**Syntax:**
```mlir
operation ::= `ktdp.store` $data_tile `,` $access_tile attr-dict `:` type($data_tile) `,` type($access_tile)
```

**Operands:** `data_tile` (tensor), `access_tile` (access_tile type)

**Example:**
```mlir
ktdp.store %A_data_tile, %A_access_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>
```

Data tile shape must match access tile's logical iteration shape (1:1 correspondence).

---

## `ktdp` Types

### AccessTileType

N-dimensional tile with fixed rank and index element type only.

```
tile-type ::= `access_tile` `<` dimension-list `index` `>`
dimension-list ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
```

**Examples:**
```mlir
access_tile<? x ? x index>      // Unknown dimensions
access_tile<? x 64 x index>     // Partially known
access_tile<1 x 64 x index>     // Fully static
```

---

## `ktdp` Attributes

### MemorySpaceAttr

Generic abstraction for device-specific memory space attributes within `ktdp`. Provides a uniform mechanism to associate IR values with a target-specific memory hierarchy while remaining extensible across backends.

Concrete implementations (e.g., `SpyreMemorySpaceAttr`) describe specific memory kinds (on-chip scratchpad, HBM) and optional core affinity. This is preferred over integer-based memory space annotations in memref types, which lack readability and semantic richness for distributed scratchpad architectures.

```mlir
memory_space = #ktdp.spyre_memory_space<HBM>
```

Future extensions will include richer metadata for compute-memory affinity.

---

## Complete Example: Tile-Parallel Matrix Add

```mlir
module {
  func.func @add() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    scf.for %i = %c0 to %tile_size step %c1 {
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        linalg.add ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
                    outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row + %i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        ktdp.store %C_data_tile, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }
}
```

This kernel distributes a 96x64 matrix add across 32 cores, each processing a 3x64 slice. Each core identifies its region via `get_compute_tile_id`, constructs memory views, builds access tiles per row, loads data, computes the addition, and stores the result.

---

## RFC Source

Full RFC: https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md
