# MLIRFrontendParser — known divergences from regex parser

## `intermediate_vars` is empty for `construct_indirect_access_tile`

**Status**: fixed for indexed_add; paged_attention still diverges

**Observed**: all `construct_indirect_access_tile` ops (indexed_add.mlir,
paged_attention.mlir)

```
regex: intermediate_vars = ['d0', 'd1', 'd2', 'd3']
mlir:  intermediate_vars = []    # was hardcoded to [] in adapter
```

**Fix applied**: derive `intermediate_vars` from the region block
argument count in `ktir_cpu/mlir_frontend/parser.py` line 333.

**Result after fix**:

- **indexed_add.mlir**: now **bit-identical** (max_diff=0, 65536/65536
  exact matches).
- **paged_attention.mlir**: still diverges (max_diff=0.73, mean=0.10,
  12/32768 exact matches). Root cause identified: this is a **regex
  parser bug**, not an MLIR frontend issue. The regex parser parses
  `arith.constant 0xFF800000 : i32` as `0` instead of `-8388608`
  (the signed i32 value of hex `0xFF800000`). This constant is then
  `arith.bitcast` to `f32`:
    - regex: `0` → bitcast → `0.0`
    - mlir:  `-8388608` → bitcast → `-inf`
  The correct value is `-inf` (IEEE 754 negative infinity). This is
  used as `M_init` (initial max accumulator for online softmax), so
  the regex parser starts the accumulator at `0.0` instead of `-inf`,
  which changes the entire softmax normalization. Both outputs happen
  to pass the NumPy reference within float16 tolerance, but the MLIR
  frontend parser is the one producing the correct result.
  See `examples/triton-ktir/paged_attention.mlir:179`.

**Reproduce** (indexed_add — crashes):
```python
import numpy as np
from ktir_cpu import KTIRInterpreter
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

interp = KTIRInterpreter(parser=MLIRFrontendParser())
interp.load('examples/triton-ktir/indexed_add.mlir')
# crashes with: ValueError: Read from unmapped address
```

**Reproduce** (compare parsed attributes):
```python
from ktir_cpu.parser import KTIRParser
from ktir_cpu.mlir_frontend.parser import MLIRFrontendParser

for path, func_name in [
    ('examples/triton-ktir/indexed_add.mlir', 'indexed_add_kernel'),
    ('examples/triton-ktir/paged_attention.mlir', 'kernel_unified_attention_spyre_2d'),
]:
    regex_mod = KTIRParser().parse_file(path)
    mlir_mod = MLIRFrontendParser().parse_file(path)
    rf = regex_mod.get_function(func_name)
    mf = mlir_mod.get_function(func_name)
    for i, (ro, mo) in enumerate(zip(rf.operations, mf.operations)):
        if 'indirect' in ro.op_type:
            print(f'{path} op {i}: {ro.op_type}')
            print(f'  regex intermediate_vars: {ro.attributes["intermediate_vars"]}')
            print(f'  mlir  intermediate_vars: {mo.attributes["intermediate_vars"]}')
    # For paged_attention, also check inside scf.for region (op 26)
    if 'paged' in path:
        for i, (ro, mo) in enumerate(zip(
            rf.operations[26].regions[0], mf.operations[26].regions[0]
        )):
            if 'indirect' in ro.op_type:
                print(f'{path} scf.for region op {i}: {ro.op_type}')
                print(f'  regex intermediate_vars: {ro.attributes["intermediate_vars"]}')
                print(f'  mlir  intermediate_vars: {mo.attributes["intermediate_vars"]}')
```

## Other (benign) divergences

These are cosmetic or semantically equivalent:

- **Operand names**: regex uses source names (`%query_ptr`), MLIR uses
  positional (`%arg1`). Expected — test infrastructure uses
  `interp.arg_names()` to handle this.

- **`coordinate_set` source strings**: whitespace differences only. AST
  constraints are identical.

- **`-d0 + 0` vs `-d0`**: MLIR canonicalizes away the `+ 0` in affine
  set constraints. Semantically identical.

- **`_result_dtype` / `_result_shape`**: present in regex parser output,
  missing from MLIR frontend. These are hints inferred from result type
  strings; execution handlers derive them independently when absent.

- **`reduce_fn`**: regex gives `None` (resolves combiner from
  `op.regions`), MLIR gives `'arith.maximumf'` / `'arith.addf'`
  (extracted from the explicit-region body). Both paths reach the same
  executor logic.

- **`return` vs `func.return`**: cosmetic op_type difference at function
  end.
