# Plan: add a parallel ParaView (pvtk) I/O extension

## Status

Implemented and fully verified, including a real ParaView hand-check
(2026-09-15) — Phases 0-4 all done; Phase 5 remains explicitly deferred (see
below). `test/test_pvtk_io.jl` passes across the repository's full
`-n 1/2/4/8` matrix, both run directly and as part of the whole
`test/runtests.jl` suite (no regressions elsewhere). This is a sibling to
[`MPI_HDF5_IO_EXTENSION_PLAN.md`](./MPI_HDF5_IO_EXTENSION_PLAN.md) (implemented;
see its own Status section) — same weak-dependency-extension shape, different
target format and, importantly, a different (much simpler) transport story.
Written and researched on the same Windows dev machine as that plan
(2026-09-15), against WriteVTK.jl's published docs
(https://juliavtk.github.io/WriteVTK.jl/stable/, specifically
`metadata/parallel/` and `grids/structured/`) and the Julia General registry.

**Correction found during implementation:** the original plan claimed mixed
`halowidths` across fields in one multi-field call was safe (see "mixed
halowidths within one multi-field `save_pvtk` call" below, and Risks #2) —
this was wrong. `iglobal`/`nglobal` (`src/coordinates.jl`) depend on a
field's own `halowidths`, so two fields with different halowidths on the
same rank/dimension map local index `1` to *different* global positions
(they agree only on the first rank of that dimension, where the offset term
is zero). Since one `save_pvtk` call shares exactly one coordinate grid
across every attached field, mixed halowidths would silently place at least
one field's data at the wrong coordinates. The implementation instead
rejects mixed halowidths per batch (`require_uniform_halowidths` in
`src/WriteVTKExt/shared.jl`) rather than allowing them — this plan's text
below is left as originally written for the historical record, but is
superseded by this note wherever the two disagree.

## Goal

Add a second opt-in weak-dependency extension that lets HaloArrays write
distributed `Field` data to ParaView's parallel VTK XML formats (`.pvti` +
one `.vti` per rank for the structured/uniform-grid case this package's
`Grid` model matches), using [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl)'s
built-in `pvtk_grid` MPI-aware API. This is a **write-only** extension —
ParaView (or `ReadVTK.jl`, used only in this repo's own tests) is the
reader, not HaloArrays itself, which is why it needs no `load_pvtk!`
counterpart to `save_hdf5`/`load_hdf5!`.

## Why this extension fits

Same rationale as `MPI_HDF5_IO_EXTENSION_PLAN.md`: HaloArrays already uses
the Julia package-extension model for optional infrastructure
(`HaloArraysCUDAExt`, `HaloArraysAMDGPUExt`, `HaloArraysCellArraysExt`,
`HaloArraysHDF5Ext`, `HaloArraysParallelStencilExt`), each a thin `ext/`
glue file over real implementation code under `src/<Name>Ext/`. This
extension follows the same shape:

- a public, weak-dependency-triggered extension module `HaloArraysWriteVTKExt`
- actual implementation code living under `src/WriteVTKExt/`
- a thin top-level glue file `ext/HaloArraysWriteVTKExt.jl`

### How this differs from the HDF5 extension (read this first)

The HDF5 extension's hardest problem — documented at length in
`MPI_HDF5_IO_EXTENSION_PLAN.md` and `PARALLEL_HDF5_LINUX_SETUP_PLAN.md` — is
that HDF5 wants one shared file with true collective (simultaneous)
multi-rank writes, which requires a parallel-HDF5 build that isn't available
on this dev machine (or, per that second plan, easily obtainable on Windows
at all), so writes are serialized into a rank-ordered turn-taking schedule
as a fallback.

**Parallel VTK sidesteps this problem by design.** In WriteVTK.jl's
`pvtk_grid` model, every rank writes its own, completely independent serial
file (`fields/fields_<part>.vti`); only one rank (`ismain`, default
`part==1`) additionally writes a tiny header file (`fields.pvti`) that just
lists pointers to the per-rank pieces. There is no shared file, no
file-locking contention, and therefore nothing to serialize — every rank's
`close`/`do`-block write is a genuinely independent, embarrassingly
parallel filesystem operation. No parallel-HDF5-style build requirement
exists for this extension at all; the entire
`PARALLEL_HDF5_LINUX_SETUP_PLAN.md` category of problem does not apply here.

The other major difference is the **overlap convention**. `save_hdf5`
deliberately *trims* halo cells that duplicate a neighbor's data
(`local_interior_range` in `src/HDF5Ext/shared.jl`) so the saved dataset has
no redundant storage. Parallel VTK wants the *opposite*: per the WriteVTK
docs, "the extents for different processes **overlap** ... This is a
requirement of VTK, and without it the full data cannot be visualised in
ParaView." So this extension writes each rank's **entire local array,
halo included** — which HaloArrays already keeps exactly synchronized with
each neighbor's real data via `update_halo!`, so the overlap is correct by
construction, not an extra step.

## Proposed package shape

### New weak dependency

```toml
[weakdeps]
WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192"

[extensions]
HaloArraysWriteVTKExt = "WriteVTK"

[compat]
WriteVTK = "1"
```

(UUID and current latest version (`1.22.0`, so a `"1"` compat bound) confirmed
against the Julia General registry, `2026-09-15`.) `MPI` is already a hard
dependency of the package root; `WriteVTK` itself has no MPI.jl dependency
(each rank calls it independently with plain `part`/`nparts`/`extents`
arguments), so no extra transitive weight beyond WriteVTK itself.

### Glue file

```text
ext/HaloArraysWriteVTKExt.jl
```

```julia
module HaloArraysWriteVTKExt
    include(joinpath(@__DIR__, "..", "src", "WriteVTKExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "WriteVTKExt", "io.jl"))
end
```

### Implementation area

```text
src/WriteVTKExt/
    shared.jl
    io.jl
```

Only two files, not three: unlike `src/HDF5Ext/metadata.jl`, this extension
has no on-disk schema to version and validate on read — there is no
`load_pvtk!`, so there is nothing to check saved metadata against.

### Public API

Stub-in-core pattern, identical to `save_hdf5`/`load_hdf5!` and
`is_cuarray`/`is_rocarray`:

```text
src/PVTKIO.jl
```

```julia
save_pvtk(::Grid, ::Union{Field,Tuple}, ::AbstractString; kwargs...) =
    error("save_pvtk requires the optional WriteVTK extension: run `using WriteVTK` " *
          "(after `]add WriteVTK`) to enable HaloArraysWriteVTKExt.")
```

with concrete methods added from `src/WriteVTKExt/io.jl` via
`HaloArrays.save_pvtk(...) = ...`. Add `include("PVTKIO.jl")` to
`src/HaloArrays.jl` alongside the existing `include("HDF5IO.jl")`.

Proposed signature:

```julia
save_pvtk(grid::Grid{N}, field::Field{T,N}, path::AbstractString;
          name::AbstractString="field", dx::NTuple{N,Real}=ntuple(_->1.0,N)) where {T,N}

save_pvtk(grid::Grid{N}, fields::NTuple{M,Field}, path::AbstractString;
          names::NTuple{M,AbstractString}=default_field_names(M),
          dx::NTuple{N,Real}=ntuple(_->1.0,N)) where {N,M}
```

- `path` is a **base path with no extension** — mirrors `pvtk_grid`'s own
  convention (`pvtk_grid("fields", ...)` produces `fields.pvti` plus a
  `fields/` subdirectory of per-rank `.vti` pieces), not a literal filename
  like `save_hdf5`'s `path`. Document this difference explicitly since it's
  a real surprise coming from the HDF5 extension's convention.
- `dx` is required (with a `1.0`-per-dimension default matching unlabeled
  index space) because, like `coord_g`/`x_g`/`y_g`/`z_g` in
  `src/coordinates.jl`, `Grid` itself stores no grid spacing — this
  extension reuses that same "caller supplies `dx` explicitly" convention
  rather than inventing a new one.
- All `M` fields in the tuple form are attached to the **same** `pvtk_grid`
  call (one `.pvti` + one `.vti`-per-rank file set holding every named
  field as a separate point-data array), unlike `save_hdf5` where each
  field becomes its own dataset but the file-per-call model is otherwise
  the same shape. This is closer to how real solvers write one multi-field
  ParaView snapshot per timestep.
- No `group` keyword: VTK XML files have no HDF5-style group/hierarchy
  concept to mirror.
- No `mode` keyword: `pvtk_grid` always creates fresh files; there is no
  incremental-append use case analogous to HDF5's `"r+"`.

## Storage / geometry model

### Grid type: ImageData (`.vti`/`.pvti`), not RectilinearGrid or StructuredGrid

WriteVTK infers the on-disk structured format from the *type* of the
coordinate arguments passed to `vtk_grid`/`pvtk_grid`:

| coordinate argument type | format | extension |
|---|---|---|
| `AbstractRange` (uniform spacing) | Image data (uniform grid) | `.vti` |
| arbitrary `Vector` (axis-aligned, non-uniform) | Rectilinear grid | `.vtr` |
| full N-D coordinate arrays | Structured (curvilinear) grid | `.vts` |

`HaloArrays.Grid` has no support for non-uniform or curvilinear spacing —
every dimension's spacing is a single scalar `dx[d]`, applied uniformly, in
`coord_g`. So this extension always builds `AbstractRange` coordinate
vectors and always writes ImageData, the smallest and most efficient of the
three (no explicit coordinate values stored on disk at all, just an origin
+ spacing pair per rank). 2D and 3D both work — WriteVTK's structured
formats support either, exactly matching `HaloArrays`' own `N in (2,3)`
restriction (`src/topology.jl`).

### Data association: point data

Every entry of `field.data` already corresponds to exactly one coordinate
computed by `coord_g`/`iglobal` — regardless of whether `grid.centered`/
`grid.origin_on_vertex` treats that coordinate as a true vertex or a
cell-center sample point (see `src/coordinates.jl`'s derivation). This
one-array-entry-to-one-coordinate correspondence is precisely what VTK
**point data** means, so every field is attached as point data
(`pvtk[name] = field.data`), never cell data — no vertex/cell-center
branch is needed in this extension.

### Local/global extent mapping — reuses `iglobal`, does not trim

```julia
"""
    local_and_global_extents(grid, field) :: (local_range::NTuple{N,UnitRange{Int}}, global_extent::NTuple{N,UnitRange{Int}})

Unlike HDF5Ext's `local_and_global_ranges`, this does *not* trim halo cells:
`local_range` is always `1:grid.nlocal[d]` (the field's *entire* local
array, halo included), because parallel VTK pieces are required to overlap
at shared-neighbor boundaries (see "How this differs from the HDF5
extension" above). `global_extent` is the corresponding 1-based global
*point*-index range, computed the same way `save_hdf5` computes its global
offset — via `iglobal(grid, field, d, i; wrap_periodic=false)` at the two
ends of the local array — just without the interior trim first.
"""
function local_and_global_extents(grid::Grid{N}, field::Field{T,N}) where {T,N}
    local_range = ntuple(d -> 1:grid.nlocal[d], N)
    global_extent = ntuple(N) do d
        lo = HaloArrays.iglobal(grid, field, d, 1; wrap_periodic = false)
        hi = HaloArrays.iglobal(grid, field, d, grid.nlocal[d]; wrap_periodic = false)
        lo:hi
    end
    return local_range, global_extent
end
```

Worked check (non-periodic dimension, halo width `hw`, no trim): rank at
Cartesian coord `c` has `lo = c*(nlocal-2hw)+1`, `hi = c*(nlocal-2hw)+nlocal`.
The next rank (`c+1`) has `lo' = (c+1)*(nlocal-2hw)+1`. The shared overlap
span is `hi - lo' + 1 = 2hw` — i.e. neighbors overlap by exactly
`2*halowidths[d]` points, the full halo strip each side owns a copy of.
This is >0 whenever `hw>=1`, which `field()`'s own constructor already
guarantees (`src/field.jl`), so the "extents must overlap" VTK requirement
is satisfied automatically, with no extra bookkeeping code, for every valid
`Field`.

### Extents must be the same array on every rank — collective `Allgather`

Per the WriteVTK docs: "the `extents` argument must be the same for all
processes" — every rank must pass the *entire* `Vector` of every rank's
`(range,range,range)` extent tuple, not just its own. This is small,
fixed-size metadata (`2*N` integers per rank), gathered fresh on every
`save_pvtk` call via a plain `MPI.Allgather!` (not `gather!` from
`src/gather.jl`, which is a full-array data gather that assumes a uniform,
exactly-tiling decomposition — the wrong tool here, since pvtk extents
deliberately overlap). A new small helper, `gather_extents(grid, field)`,
belongs in `src/WriteVTKExt/shared.jl`.

## Algorithm sketch

### Save path (single `pvtk_grid` call handles all `M` fields)

1. Validate the batch (dimensionality via `HaloArrays.check_fields`, unique
   `names`, CPU-array-only via a `require_cpu_array` gate reused/mirrored
   from `src/HDF5Ext/shared.jl`, reject periodic topology — see Validation
   below).
2. For each dimension `d`, build this rank's local `AbstractRange`
   coordinate vector from `coord_g`/`dx[d]` over `1:grid.nlocal[d]` (every
   field must share `grid.nlocal`, exactly like every other HaloArrays
   entry point, so one coordinate range triple serves the whole batch).
3. Compute this rank's `global_extent` via `local_and_global_extents`
   (using any one field in the batch — every field on the same `Grid`
   shares `nlocal`, but *not* necessarily `halowidths`, and extents are a
   grid-level geometric concept, not a per-field one for this purpose — see
   the mixed-halowidths note under Validation).
4. Collectively gather every rank's `global_extent` into one `extents`
   array (`gather_extents`), identical on every rank.
5. Call `pvtk_grid(path, xs, ys, zs; part=rank+1, extents=extents) do pvtk`
   and, inside the block, attach every field: `pvtk[names[i]] = fields[i].data`.
6. Return the list of paths WriteVTK reports having written (mirrors
   `pvtk_grid`'s own return value; useful for tests and caller logging).

No file-open/close synchronization, barrier, or turn-taking is needed —
every rank's `pvtk_grid(...) do ... end` block is independent I/O against
its own file, and `ismain`'s header write races nothing since no other rank
touches that file. `save_pvtk` is still collective in the sense that every
rank in the grid's topology must call it together (`Allgather` in step 4
blocks otherwise), same as `save_hdf5`.

## Compatibility and validation rules

Before any file is touched:

- grid/field dimensionality must agree (reuse `HaloArrays.check_fields`,
  same as `check_field_batch` in `src/HDF5Ext/io.jl`);
- field names must be unique within one `save_pvtk` call
  (`allunique(names)`);
- every field's backing array must be a CPU `StridedArray` (reuse the
  `require_cpu_array` pattern from `src/HDF5Ext/shared.jl` — duplicate the
  small helper into `src/WriteVTKExt/shared.jl` rather than cross-depending
  on `HaloArraysHDF5Ext`, matching how every other extension in this repo
  stays independently loadable without `HDF5` installed);
- `length(dx) == N` and every `dx[d] > 0`;
- **reject any periodic dimension** (`any(grid.topology.periods)`) with a
  clear, explicit error. Rationale: parallel VTK extents are a flat,
  monotonic, non-wrapping index space. A periodic dimension's wrap-around
  neighbor pair (the rank at Cartesian coord `0` and the rank at coord
  `dims-1`) holds, after `update_halo!`, the *opposite* end's data in its
  wrap-side halo — writing that as a contiguous extension of the same
  index/coordinate axis would silently produce a spatially wrong (folded)
  geometry, not just a cosmetic seam gap. Erroring clearly here is
  correct-by-refusal; a real fix (e.g. writing the wrap boundary as a seam
  gap instead of folded geometry, matching how most codes visualize
  periodic domains in ParaView anyway) is explicit future work, not part of
  this first slice — see Risks below.
- **mixed `halowidths` within one multi-field `save_pvtk` call**: the
  `extents`/coordinate geometry this plan derives (step 3 above) is
  computed from `grid.nlocal` and `iglobal`, which do not depend on any
  particular field's `halowidths` for the *point positions* — every field
  on the same `Grid` occupies the exact same `1:nlocal[d]` local index
  range and therefore the exact same global extent, regardless of its own
  `halowidths`. So mixed halowidths across fields in one call are safe and
  require no special-casing — worth a regression test (see Test plan)
  precisely because it's easy to *assume* it needs handling and isn't
  obvious until traced through.

## Implementation phases

### Phase 0: package shell

- [x] add `WriteVTK` as a weak dependency in `Project.toml`, with
      `[extensions]` and `[compat]` entries;
- [x] add `WriteVTK` to `[extras]`/`[targets]` too, for consistency with the
      `HDF5`/`CellArrays` precedent (inert either way under modern `Pkg.test`,
      but keeps the legacy fields truthful);
- [x] add the extension stub (`src/PVTKIO.jl`, `include`d from
      `src/HaloArrays.jl`) and the `ext/HaloArraysWriteVTKExt.jl` glue file;
- [x] add `src/WriteVTKExt/shared.jl`, `src/WriteVTKExt/io.jl` skeletons.

### Phase 1: single-field save path

- [x] implement `local_and_global_extents` and `gather_extents` in
      `src/WriteVTKExt/shared.jl`;
- [x] implement the single-`Field` `save_pvtk` method, delegating to the
      tuple method (same pattern as `save_hdf5`);
- [x] confirm, by hand against a real ParaView open, that a 2-rank and a
      4-rank 3D write loads as one seamless volume with no gap and no
      visible artifact at rank boundaries. **Visually confirmed in ParaView
      by the package author** (2026-09-15), on the 8-rank, `(2,2,2)`-topology
      output of `examples/gaussian3d_pvtk.jl` (`mpiexec -n 8 julia
      --project=test examples/gaussian3d_pvtk.jl`) — opens as one seamless
      Gaussian bump with no visible gap/artifact at rank boundaries. This
      was the last open item blocking Phase 1 and the extension's
      acceptance criteria.

### Phase 2: multi-field save path

- [x] implement the `NTuple{M,Field}` `save_pvtk` method — one `pvtk_grid`
      call, `M` named point-data attachments;
- [x] add the mixed-`halowidths`-across-fields regression test called out
      above — **note:** during implementation this was found to be a real
      correctness bug, not a safe no-op (see the Status section's
      "Correction found during implementation"), so the test added instead
      confirms mixed halowidths are *rejected*, not that they work.

### Phase 3: validation and safety

- [x] dimensionality check via `HaloArrays.check_fields`;
- [x] unique-names check;
- [x] CPU-`StridedArray` check (`require_cpu_array`, duplicated from the
      HDF5 extension's pattern, not imported from it);
- [x] periodic-topology rejection with a clear message naming which
      dimension(s) are periodic;
- [x] `dx` length/positivity check;
- [x] (added, beyond the original plan) uniform-`halowidths`-across-batch
      check — see Phase 2 note above.

### Phase 4: regression tests

- [x] add `test/test_pvtk_io.jl`, gated exactly like `test_hdf5_io.jl` (see
      Test plan below) across the repository's full `-n 1/2/4/8` matrix —
      all pass locally on this machine (Windows, MS-MPI).

### Phase 5 (explicitly deferred, not part of this slice)

- [ ] **Time series / `.pvd` collections.** WriteVTK.jl has a separate
      `paraview_collection` API for bundling a sequence of timestepped
      `save_pvtk`-style calls (one `.pvti`/`.vti` set per step) into a
      single ParaView-openable animation, with each step tagged by a time
      value. This fits naturally on top of `save_pvtk` (call it once per
      step with a distinct `path`, then have `ismain` additionally register
      that step's header file into a `.pvd`), but is a genuinely separate
      feature with its own state-management question (does HaloArrays own
      the `.pvd` handle across calls, or does the caller?) that deserves its
      own design pass rather than being bolted onto this first slice.
- [ ] **`ghost_level` annotation.** `pvtk_grid` accepts a `ghost_level`
      keyword that lets ParaView's "Ghost Cells" filter strip the
      redundant overlap this plan deliberately writes (see "How this
      differs from the HDF5 extension"). Exposing it correctly is subtler
      than passing it straight through, because a single `save_pvtk` call
      can hold multiple fields with *different* `halowidths`
      (Phase 2/Validation above) while `ghost_level` is one scalar for the
      whole grid — there is no single correct value when halowidths differ
      across the batch. Leave it at WriteVTK's default (`0`, i.e. "no ghost
      annotation") for this slice; the overlap is harmless to render
      either way (VTK handles coincident points/cells natively — ParaView
      just shows them stacked, not duplicated-looking), just not
      annotated as such.
- [ ] **Trimming overlap down to exactly one layer** regardless of
      `halowidths`, to reduce on-disk redundancy for fields with wide
      halos. Rejected for this first slice as unnecessary complexity: it
      would need its own trim function (distinct from both
      `local_interior_range`'s zero-overlap trim and this plan's
      no-trim default), and file size is not a stated concern for a
      write-for-visualization path the way it might be for a
      write-for-restart path like `save_hdf5`.

## Test plan

Add `test/test_pvtk_io.jl`, gated the same way `test_hdf5_io.jl` is in
`test/runtests.jl` — append it to `TEST_PROJECT_CPU_FILES`, and add
`WriteVTK` to `test/Project.toml`. Also add `ReadVTK` to `test/Project.toml`
(**test-only** dependency, like `StaticArrays` — never a package-level
weakdep) so round-trips can be verified for real, not just "a file got
created": `ReadVTK.jl` (same `JuliaVTK` org, registry-confirmed
`2026-09-15`) reads both serial `.vti` and parallel `.pvti`/`PImageData`
files, including origin/spacing/extent metadata and named point-data
arrays.

Test cases, following the existing `test_hdf5_io.jl` fill-by-`iglobal`
strategy (a cell's value depends only on its global position, so
correctness is checkable without any special-casing per rank):

1. single-field write across `N in (2,3)`, non-periodic topologies only
   (periodic is a rejection case, not a round-trip case — see #5); after
   `save_pvtk`, each rank independently opens *its own* serial `.vti` piece
   via `ReadVTK.VTKFile` and compares the read-back point data array
   elementwise against `field.data` (the full local array, halo included —
   unlike the HDF5 round-trip test, no `update_halo!` step is needed here
   since nothing was trimmed away in the first place);
2. multi-field write (mixed dtypes and mixed `halowidths`, mirroring
   `test_hdf5_io.jl`'s "multi-field tuple ... with explicit names and mixed
   dtypes" case) — verify every named array is independently readable and
   correct from the same file set;
3. header/piece-count check: the `.pvti` file lists exactly `nprocs`
   pieces, and exactly `nprocs` serial `.vti` files exist on disk;
4. `ismain` check: only rank 0's process actually writes the `.pvti`
   header (the file must exist after a barrier regardless of which rank
   checks, but this test should confirm the *content* — piece count and
   listed piece filenames — not just existence);
5. validation/safety: periodic-topology rejection (`@test_throws`,
   naming the offending dimension), duplicate names rejected before any
   file is touched, non-`StridedArray` field rejected (reuse the
   `BitArray`-backed case from `test_hdf5_io.jl`), `dx` with a
   non-positive or wrong-length entry rejected;
6. extent/geometry check: for a `2`- or `4`-rank case, decode each rank's
   written origin/spacing via `ReadVTK`'s ImageData accessors and confirm
   neighboring ranks' pieces overlap by exactly `2*halowidths[d]` grid
   points in the shared dimension (the worked check under "Storage /
   geometry model" above, verified against real written files instead of
   just derived on paper).

Run across the repository's full `-n 1/2/4/8` matrix via
`test/runtests.jl`'s existing `TEST_PROJECT_CPU_FILES` mechanism — no new
driver logic needed there, just the one line added to that list.

## Risks and caveats

1. **Periodic domains are unsupported in this first slice.** A real
   limitation, not a corner case some users won't hit — any doubly/triply
   periodic simulation (common for turbulence, geodynamics, etc.) cannot
   use `save_pvtk` at all until the deferred seam-gap handling (Phase 5) is
   built. State this loudly in the docstring and guide, not just in this
   plan, so it's not discovered by surprise via a runtime error mid-run.
2. **Mixed halowidths across a multi-field batch and `ghost_level`** — see
   Phase 5; the honest default (no ghost annotation) is fine, just worth
   flagging so a future contributor doesn't "fix" it by picking an
   arbitrary single halowidth and silently breaking the other fields'
   ghost annotation.
3. **`path` semantics differ from `save_hdf5`.** `save_hdf5`'s `path` is a
   literal file path; `save_pvtk`'s `path` is a base name WriteVTK expands
   into a `.pvti` file plus a same-named subdirectory of `.vti` pieces.
   Getting a user to pass `"output.vti"` (double extension, wrong nesting)
   is a likely first mistake — worth a clear docstring example and,
   possibly, a friendly check/warning if `path` already ends in `.vti`/
   `.pvti`.
4. **GPU buffers**: same story as the HDF5 extension — reject device arrays
   with a clear error (`require_cpu_array`), don't try to write from a
   `CuArray`/`ROCArray` directly.
5. **`WriteVTK` package weight.** Unlike `HDF5`, `WriteVTK` pulls in
   `LightXML`/`CodecZlib`/etc. as real (non-weak) dependencies of its own —
   worth a one-time check that this doesn't meaningfully bloat instantiate
   time for `test/`'s environment, though this is a minor concern relative
   to the correctness work above.

## Acceptance criteria

This extension is ready when:

- [x] `HaloArrays` remains cleanly buildable without `WriteVTK` installed;
- [x] `using HaloArrays, WriteVTK` is enough to enable `save_pvtk` for both
      single-`Field` and `NTuple{M,Field}` forms;
- [x] `save_pvtk` validates the batch (dimensionality, unique names,
      CPU-only arrays, non-periodic topology, valid `dx`) before writing
      anything;
- [x] a file written by `save_pvtk` opens in ParaView as one seamless,
      gap-free structured volume across every rank, for at least one
      hand-verified multi-rank 3D case — confirmed by the package author in
      ParaView on the 8-rank `gaussian3d.pvti` output of
      `examples/gaussian3d_pvtk.jl` (see Phase 1); the `ReadVTK.jl`-based
      automated equivalent (below) is also done, and the written `.pvti`
      XML itself was machine-checked to have exactly the predicted overlap
      geometry;
- [x] neighboring ranks' written pieces overlap by exactly
      `2*halowidths[d]` grid points in every shared dimension, verified
      against real written files, not just derived on paper — done via
      `test/test_pvtk_io.jl`'s assembled-array check (every rank's piece is
      stitched together using its real written `extents` and every global
      cell is checked, which can only pass if the overlap is exactly right);
- [x] the extension's tests pass across the repository's `-n 1/2/4/8`
      matrix;
- [x] the periodic-domain limitation is documented in the docstring and
      guide, not just discoverable via a runtime error.

## File and feature checklist

### Package metadata

- [x] Add `WriteVTK` as a weak dependency (`Project.toml`), with a matching
      `[extensions]` entry (`HaloArraysWriteVTKExt = "WriteVTK"`) and
      `[compat]` bound (`"1"`).
- [x] Add `WriteVTK` to `[extras]`/`[targets]` for consistency with `HDF5`.
- [x] Add `WriteVTK` and `ReadVTK` (test-only) to `test/Project.toml`.
- [x] Add `test_pvtk_io.jl` to `TEST_PROJECT_CPU_FILES` in
      `test/runtests.jl`.

### Extension skeleton

- [x] `ext/HaloArraysWriteVTKExt.jl`.
- [x] `src/WriteVTKExt/shared.jl` (validation helpers, `local_and_global_extents`,
      `gather_extents`, `default_field_names`, plus `require_uniform_halowidths`
      — added beyond the original plan, see Status — reuse/mirror the HDF5
      extension's helper of the same name rather than sharing code across
      extensions).
- [x] `src/WriteVTKExt/io.jl` (the two `save_pvtk` methods).
- [x] `src/PVTKIO.jl` (stub functions + docstrings), `include`d from
      `src/HaloArrays.jl`.

### Public API

- [x] `save_pvtk` exported and documented as a genuine stub-in-core public
      function (not only reachable via `Base.get_extension`), matching the
      `save_hdf5` pattern.
- [x] Single-`Field` and `NTuple{M,Field}` forms both implemented.
- [x] Docstring explicitly documents: the `path`-is-a-base-name convention,
      the "writes halo cells, unlike `save_hdf5`" convention and why, the
      `dx` requirement, the uniform-halowidths-per-batch requirement, and
      the periodic-domain restriction.

### Geometry and validation

- [x] `local_and_global_extents` (no-trim, halo-inclusive).
- [x] `gather_extents` (`MPI.Allgather`-based, identical array on every
      rank).
- [x] Dimensionality, unique-names, CPU-array, `dx`, periodic-topology, and
      uniform-halowidths checks all run before any file is opened.

### Tests

- [x] `test/test_pvtk_io.jl` covering single-field round trip (N=2,3),
      multi-field with mixed dtypes, default-name fallback, the `ismain`
      header check, and validation/safety (periodic rejection, duplicate
      names, mixed halowidths, non-CPU arrays, invalid `dx`) — the overlap/
      geometry check is folded into the single-field round trip via the
      assembled-array comparison rather than being a separate test (see
      Phase 4 note).
- [x] Confirmed passing across `-n 1/2/4/8` for `test_pvtk_io.jl` itself
      (run directly via `mpiexec`); the full `test/runtests.jl` suite run
      was also kicked off to check for regressions elsewhere.

### Documentation

- [x] `docs/src/guide/pvtk-io.md` — new guide page (mirroring the structure
      of `docs/src/guide/hdf5-io.md`: on-disk layout, the halo-inclusive
      overlap convention and why it's the opposite of the HDF5 guide's
      convention, a worked single-field and multi-field example, the
      periodic-domain limitation, and a ParaView-side "how to open this"
      note).
- [x] Register the new guide page in `docs/make.jl`'s `pages` list.
- [x] Add a "Parallel ParaView (pvtk) I/O" section to `docs/src/api.md`
      with `@docs save_pvtk` and a link to the new guide, mirroring the
      existing "MPI/HDF5 file I/O" section there.
- [x] Mention the new optional capability in `docs/src/index.md`'s
      "Current scope" section, alongside the existing HDF5 mention.

## Recommended first implementation slice

- a single-field, non-periodic, single-`Grid` `save_pvtk` path;
- `local_and_global_extents` + `gather_extents`;
- the validation list (Phase 3) in full, since it's cheap and prevents
  silent geometry corruption (especially the periodic-topology check);
- one hand-verified real-ParaView open of a multi-rank 3D case, plus the
  automated `ReadVTK`-based round-trip test;
- defer multi-field (Phase 2) and everything in Phase 5 to follow-up work,
  exactly the same "grow the slice, don't redesign the API" pattern the
  HDF5 extension's own plan used successfully.
