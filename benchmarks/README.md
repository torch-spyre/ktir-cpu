# Benchmarks: Indirect Access Emulation Timing

## Usage

```bash
cd ktir-cpu

# Fast-path vs general-path comparison + per-step breakdown
uv run python benchmarks/bench_indirect_emul_time.py block

# Gather memcpy optimization (3-copy vs 1-copy)
uv run python benchmarks/bench_indirect_emul_time.py gather

# Custom config
uv run python benchmarks/bench_indirect_emul_time.py block --config configs/custom.toml
```

## Output

### `block` — fast-path vs general-path + per-step breakdown

```
Workload |       Shape |    Points | General (ms) | Fast (ms) | Speedup
---------+-------------+-----------+--------------+-----------+--------
 xl-262K | 128x256x128 |   262,144 |       293.89 |      2.74 |  107.3x
    16MB | 128x64x1000 |   512,000 |       616.80 |      5.99 |  102.9x
    64MB | 256x64x2000 | 1,024,000 |      1154.30 |     10.70 |  107.9x

Step breakdown (64MB workload):
------------------------------------------------------------
  Old path (7 steps):
    1. Enumerate iteration space:         24.668 ms  (  2.1%)
    2. Read index tensors:               779.872 ms  ( 66.8%)
    3. Build coordinate list:            232.776 ms  ( 19.9%)
    4. Linearize flat offsets:           128.898 ms  ( 11.0%)
    5. Gather from HBM:                    1.187 ms  (  0.1%)
    6. Reshape:                            0.003 ms  (  0.0%)
    7. Write to LX:                        0.005 ms  (  0.0%)
    TOTAL                              1167.409 ms

  New path (3 steps):
    1. Read K index values:                0.032 ms  (  0.8%)
    2. Numpy broadcast offsets:            2.578 ms  ( 67.6%)
    3. Gather + reshape + LX:              1.202 ms  ( 31.5%)
    TOTAL                                 3.812 ms

  Speedup: 306x
```

### `gather` — 3-memcpy vs 1-memcpy data movement

```
Workload | Gather (elems) | Span (elems) | Old (ms) | New (ms) | Speedup
---------+----------------+--------------+----------+----------+--------
 xl-262K |        262,144 |    3,211,264 |   0.5312 |   0.2526 |    2.1x
    16MB |        512,000 |    7,744,000 |   1.3399 |   0.5175 |    2.6x
    64MB |      1,024,000 |   28,544,000 |   1.5259 |   1.1510 |    1.3x

  Old: _read_flat(span) + flat[offsets] + write_flat(lx) — 3 memcpys
  New: mgr.gather(offsets) + place_in_lx — 1 memcpy
```

## Config

`configs/indirect_emul.toml` — workloads specified as `(num_experts, M, N, n_selected)`. Size strings like `"2K"` are auto-parsed. List fields require explicit `mode = "product"` or `"zip"`.

## Code Structure

```
bench_indirect_emul_time.py   — CLI entry point (block/gather subcommands)
bench_utils.py                — parse_size, BenchTimer, BenchTable, load_config,
                                make_bench_context, build_moe_iat
configs/indirect_emul.toml    — workload parameter matrix
```
