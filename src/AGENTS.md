# Rules for `src/`

Also follow the root `AGENTS.md`. Use the `change-halo-exchange` skill for edits to
packing, tags, or ordering.

## Exchange invariants

- Full exchange: dimension `d+1` packs after dimension `d` unpacks. This carries
  corner data. Do not reorder or parallelize across dimensions.
- `halo_exchange_start!` completes every dimension but the last. Only the last
  dimension stays in the `ExchangeHandle`.
- Face exchange uses tags from `face_tag(dim, side)` (base 100). Full exchange picks tags
  by direction, not one shared tag (a shared tag mismatches on 2-rank periodic
  runs). Never reuse a tag range between the two.
- Periodic dimensions with 2 ranks and with 1 rank (self-neighbor) are special
  cases. Test both after any tag or neighbor logic change.
- `cache.pending` must be set by `start!` and cleared by `finish!`. A second `start!`
  must error, not corrupt buffers.
- `Field.overlaps[d]` moves the sent strip inward. Receive ranges stay at the
  array edge. `face_range`, `coordinates.jl`, and the I/O code all read `overlaps`.
  Update all of them together.
- I/O extensions reject non-default overlaps. Keep that check when touching I/O.

## Performance

- Hot paths are tuple-specialized recursion (`pack_faces!(cache, fields::Tuple, ...)`
  with `Base.tail`). Do not replace with `Vector{Field}` loops.
- The default (non-threaded, CPU) exchange target is zero steady-state allocation
  beyond the documented small constant. Check with `test/test_allocations.jl`.
- Do not add type-unstable fields to `ExchangeCache`. `gpu_cache` is the one
  deliberate `Any` slot.

## Backends

- Core never imports CUDA or AMDGPU. Add behavior by a new method on
  `pack_target`/`unpack_source`/`stage_send!`/`stage_recv!` in the extension.
- Edit `src/CUDAExt/` and `src/AMDGPUExt/` together. AMDGPU cannot be tested here:
  say so in the report.
- Files in `src/*Ext/` are `include`d by `ext/*.jl`. They are not part of the main
  module and do not appear in `src/HaloGrids.jl`.

## Public API

- Add new exports to the `export` line at the top of the defining file and to the
  module docstring in `src/HaloGrids.jl`.
- Every exported name needs a docstring and must appear in a `@docs` block in
  `docs/src/api.md`.
