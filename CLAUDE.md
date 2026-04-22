# Project Rules

## Golden Specification: KTIR RFC 0682

All code in this project must conform to the KTIR specification defined in:
https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md

The full spec is also mirrored in `.claude/skills/ktir-dialect.md`.

### When reviewing PRs

- Check every changed file against the KTIR spec. Flag any deviation from the spec's defined ops, types, attributes, semantics, or constraints.
- Specifically verify:
  - `ktdp` dialect ops (`get_compute_tile_id`, `construct_memory_view`, `construct_distributed_memory_view`, `construct_access_tile`, `construct_indirect_access_tile`, `load`, `store`) match the spec's syntax, operands, attributes, and semantics.
  - `AccessTileType` element type is always `index`.
  - `SpyreMemorySpaceAttr` uses `#ktdp.spyre_memory_space<...>` syntax.
  - `coordinate_set` uses `IntegerSetAttr` (affine integer sets).
  - `base_map` and `access_tile_order` use `AffineMapAttr`.
  - `access_tile_order` follows lexicographic semantics (rightmost = innermost).
  - Indirect access tiles: indirectly-loaded indices are not combined multiplicatively.
  - `ktdp.load` / `ktdp.store` data tile shape matches access tile logical iteration shape (1:1 correspondence).
  - `construct_memory_view` does not allocate — only constructs a logical view.
  - Compute ops use standard MLIR dialects (Arith, Math, LinAlg); control flow uses SCF.
  - Overlapping coordinate sets in distributed views produce unspecified behavior unless constrained.

### When generating code

- All MLIR generation, parsing, and interpretation code must respect the KTIR spec.
- Do not invent new `ktdp` ops or attributes not in the spec.
- Do not change the semantics of existing `ktdp` ops (e.g., making `construct_memory_view` allocate memory).
- If the spec does not cover a needed feature, flag it explicitly rather than silently extending the dialect.
- Preserve the separation of concerns: memory interpretation (`construct_memory_view`), address computation (`construct_access_tile`), data movement (`load`/`store`), and compute (Arith/Math/LinAlg).

## Gap Analysis

`docs/gap_analysis.md` tracks conformance between the RFC spec and the implementation. Keep it up to date when:
- Making changes that implement, partially implement, or close a tracked gap — update the relevant row's status.
- Creating a GitHub issue for a tracked gap — note the issue in the relevant row.
- Closing a GitHub issue — reflect the resolved status in the relevant row.

## Pull Request Guidelines

When creating a pull request:
- Always CC the maintainers (see CODEOWNERS) by adding a comment on the PR.
- If the PR is linked to an issue, also CC the issue author in the same comment.

## Environment

- Python project using `uv` for dependency management
- Tests: `uv run pytest tests/ -v`
- MLIR examples live in `examples/`
