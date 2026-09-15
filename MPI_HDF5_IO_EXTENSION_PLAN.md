# Plan: add an MPI/HDF5 I/O extension

## Status (2026-09-15)

The extension is implemented and passing the full test matrix
(`test/test_hdf5_io.jl`, `-n 1/2/4/8`, plus the rest of `test/runtests.jl`).
What's below is kept as the historical design record; see inline notes for
where the implementation landed relative to the original sketch. In short:

- Public API is `save_hdf5`/`load_hdf5!` (not `save!`/`load!`), genuinely
  exported from `HaloArrays` itself via a stub-function pattern
  (`src/HDF5IO.jl`, mirroring `is_cuarray`/`is_rocarray` in `backend.jl`) —
  calling either without `using HDF5` raises a clear error instead of
  requiring `Base.get_extension` at the call site.
- Both single-`Field` and `NTuple{M,Field}` forms are implemented.
- Halo cells that duplicate a neighbor's data are never written; a
  non-periodic domain edge (no neighbor, so not halo) is written normally.
  Loading leaves real halo cells untouched — call `update_halo!` after.
- One shared file gets one true `nglobal`-shaped dataset per field. No
  parallel-HDF5 build is available in this environment
  (`HDF5.has_parallel() == false`), so writes are rank-ordered
  (`MPI.Barrier`/`MPI.Allreduce`-synchronized turns, at most one rank in the
  file at a time) rather than truly simultaneous; reads are independent
  per-rank (safe without parallel HDF5). See `docs/src/guide/hdf5-io.md`.
- Schema bumped to `0.2.0` (per-dataset attributes, interior-only payload);
  the earlier `0.1.0` scaffold's full-halo/group-level-attributes files are
  not compatible and were never a released format.
- True simultaneous collective I/O (a real parallel-HDF5 `dxpl_mpio`/
  `fapl_mpio` path) and allocation/overhead benchmarking remain explicitly
  deferred: the former can't be validated without a parallel-HDF5 build,
  which isn't available here; the latter wasn't in scope for this pass.

## Goal

Add an opt-in weak-dependency extension that lets HaloArrays write and read distributed grid/field data to and from HDF5 files using MPI-aware collective semantics while staying consistent with the package's explicit `Topology`/`Grid`/`Field` ownership model.

## Why this extension fits

HaloArrays already uses the Julia package extension model for optional infrastructure:

- `HaloArraysCUDAExt` is a thin wrapper around `src/CUDAExt/`
- `HaloArraysAMDGPUExt` is the same pattern for ROCm
- `HaloArraysCellArraysExt` is a thin wrapper around `src/CellArraysExt/`

The extension should follow that same model:

- a public, weak-dependency-triggered extension module such as `HaloArraysHDF5Ext`
- actual implementation code living under `src/HDF5Ext/`
- a thin top-level extension glue file under `ext/`

This keeps HDF5/MPI code out of the normal CPU core path and makes the dependency optional.

## Proposed package shape

### New weak dependency

Add a weak dependency on `HDF5` to the package project:

```toml
[weakdeps]
HDF5 = "..."

[extensions]
HaloArraysHDF5Ext = "HDF5"
```

The current package already includes `MPI` as a hard dependency. The new extension can import `MPI`, `HDF5`, and `HaloArrays`-owned references without changing the base package’s load path.

### Glue file

Create:

```text
ext/HaloArraysHDF5Ext.jl
```

with a structure mirroring the existing extension conventions:

```julia
module HaloArraysHDF5Ext
    include(joinpath(@__DIR__, "..", "src", "HDF5Ext", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "HDF5Ext", "io.jl"))
end
```

### Implementation area

Create a new source subtree:

```text
src/HDF5Ext/
    shared.jl
    io.jl
    metadata.jl
```

The extension should stay in ordinary package source layout and avoid putting all work directly inside `ext/`.

## Public API sketch

Keep the API minimal and explicit. Proposed namespace pattern:

```julia
save_hdf5(grid, field, path; group="/", mode="w", collective=true)
save_hdf5(grid, fields, path; group="/", mode="w", collective=true)
load_hdf5(grid, field, path; group="/", collective=true)
load_hdf5(grid, fields, path; group="/", collective=true)
```

**As implemented:** `save_hdf5(grid, field_or_fields, path; name/names, group="/", mode="w")`
and `load_hdf5!(grid, field_or_fields, path; name/names, group="/")` — the
load form got a `!` (it mutates the field(s) in place, matching `gather!`/
`update_halo!`'s naming convention elsewhere in the package). `collective`
was dropped rather than kept as a no-op: there's no parallel-HDF5 build here
to make a `false` setting meaningfully different from `true`, and every call
is collective in the sense that matters regardless (every rank in the
grid's topology must call it together, whether or not the underlying I/O is
simultaneous) — see Phase 3 above.

or a more object-oriented but still explicit alternative:

```julia
HDF5IO.save!(grid, fields, path)
HDF5IO.load!(grid, fields, path)
```

Design principle:

- Do not introduce a global active grid or a global file handle.
- Make the caller pass `Topology`/`Grid` and the `Field` object explicitly.
- Mirror the existing halo-exchange style of explicit ownership and staged state.

## Storage model

The extension should store data in a rank-local shape convention:

```text
/group
    /metadata
    /topology/dims
    /grid/nlocal
    /grid/nglobal
    /field_name/data
```

This layout should keep:

- MPI communicator rank metadata
- grid shape metadata
- per-field halo widths
- array data written as a rank-local block

To stay consistent with the package’s `Field` object model, the HDF5 metadata should record:

- `nlocal`
- `nglobal`
- `halowidths`
- `ndims`
- `dtype`
- `name`

For distributed writes, every rank writes only its own local block into the same file and group path.

## MPI behavior

This extension should not try to hide MPI. It should be a careful, explicit match to the package’s communication model.

### Recommended transport strategy

Use `HDF5` for the file representation, but make the file access MPI-aware by:

1. letting each rank read/write its own local `Field` data block only;
2. using an MPI collective file-open pattern for the shared HDF5 file; and
3. using a consistent rank-local coordinate/order model in the file schema.

### Preferred file access shape

Keep the parallel I/O semantics as close to HaloArrays’ existing model as possible:

- each rank owns one contiguous local array slice;
- file metadata is written once by the root rank or collectively by all ranks;
- rank 0 participates in the metadata schema only if a file header is required;
- all data writes are shape- and offset-defined from the grid metadata.

The extension should never allow an uncoordinated per-rank HDF5 write that starts a separate file descriptor pathway per array object. The file I/O should remain rank-explicit and deterministic.

## Algorithm sketch

### Save path

1. Validate that the `Grid` and the `Field` objects agree on coordinate and halo metadata.
2. Open/create the HDF5 file using a rank-aware, collective strategy.
3. Write file metadata:
   - topology dimensions
   - grid local sizes
   - halo widths
   - field names
4. For each field:
   - compute the local field extent from the grid and the field halo width;
   - write the local data slice into the field dataset;
   - optionally write a ghost/hilo shape or halo metadata object separately.
5. Close the file and leave the `Grid`/`Topology` untouched.

### Load path

1. Open the HDF5 file in read mode.
2. Validate metadata against the requested `Grid` and `Field` objects.
3. Read the per-rank domain metadata and field descriptors.
4. Read each `Field` array into the local storage owned by that rank.
5. Optionally validate that the halo widths and the field name map correctly.
6. Close the file.

## Compatibility and validation rules

The extension should implement all of the following validation checks before it writes or reads:

- grid topology dimensions must match the file’s saved dimensions;
- `nlocal` must match the requested `Field` shape;
- halo widths must match the file’s stored metadata;
- field names must remain unique inside a file;
- the extension must reject mixed CPU/GPU arrays unless a distinct staging path exists;
- no file may silently overwrite an existing layout with mismatched metadata.

## Implementation phases

### Phase 0: add the package shell

- [x] add `HDF5` as a weak dependency in the main project;
- [x] add extension registration in `Project.toml`;
- [x] add the extension stub and implementation skeleton;
- [x] add documentation comments explaining that this extension is optional and MPI-aware.

### Phase 1: metadata and file schema

- [x] define the on-disk metadata schema;
- [x] implement a metadata-only write/read path in the extension scaffold;
- [x] validate the first draft schema using a 1-rank and 2-rank test path in the repository plan as a future integration goal, while the shipped repository smoke test remains the first rank-local round trip and is deliberately bound to the package extension registry handle rather than a package-level import alias.

The current metadata object is intentionally versioned as `HDF5Metadata`, with a constructor using `Grid` and `Field` types and validation staying conservative around `schema_version`, `topology_dims`, `nlocal`, and `halowidths` before an HDF5 path reads data.

### Phase 2: per-field data I/O

- [x] implement a first-slice save/load path for a single `Field` object;
- [x] implement a save/load path for a tuple of `Field` objects;
- [x] ensure halos are not written accidentally as if they were part of the canonical local array storage.

`save_hdf5`/`load_hdf5!` each have a single-`Field` method that delegates to
an `NTuple{M,Field}` method (`src/HDF5Ext/io.jl`); the tuple method is where
the real per-field loop lives. Halo cells are excluded via
`local_interior_range`/`local_and_global_ranges` (`src/HDF5Ext/shared.jl`),
which trims `field.halowidths[d]` on a side only when
`HaloArrays.has_neighbor` is true there — so a non-periodic domain edge
(no neighbor) is written as ordinary data, not skipped.

### Phase 3: stencil-safe collective I/O

- [x] make rank-local writes happen under a common offset model;
- [x] ensure that the write strategy handles the halo and local block decomposition correctly;
- [x] add tests for exact shape and order preservation.

Implemented as a single shared, `nglobal`-shaped dataset per field, with each
rank's block placed via [`iglobal`](@ref) (reused rather than re-derived) at
its correct offset. This is *not* true simultaneous collective I/O — no
parallel-HDF5 build is available to implement or validate a real
`dxpl_mpio`/`fapl_mpio` path against (`HDF5.has_parallel() == false` for this
package's ordinary HDF5_jll-backed weakdep) — so `save_hdf5` instead has
ranks take turns, in order, each opening the file, writing only its own
hyperslab, and closing it before the next rank's turn
(`MPI.Barrier`/`MPI.Allreduce`-synchronized). Every rank in the grid's
topology must still call `save_hdf5` together (it is collective in that
sense), just not simultaneously. `load_hdf5!` needs none of this — reads are
independent per rank, since concurrent read-only file access is ordinary,
safe behavior even without parallel HDF5. Exact shape/order preservation is
covered by `test/test_hdf5_io.jl`'s round-trip tests (global-index-consistent
fill, `save_hdf5` → `load_hdf5!` → `update_halo!` → exact full-array equality,
including at non-periodic domain edges), run across `-n 1/2/4/8`. A real
collective path remains explicit future work, gated on a parallel-HDF5 build
becoming available to validate against.

### Phase 4: regression and benchmark guardrails

- [x] keep a single-thread and multi-rank test matrix;
- [ ] measure allocations and overhead for the metadata-only and field-save paths;
- [x] make sure the implementation remains weak-dependency-triggered and does not drag HDF5 into the core import path.

`test_hdf5_io.jl` runs across the repository's full `NPROCS_MATRIX` (1, 2, 4,
8) via `test/runtests.jl`'s `TEST_PROJECT_CPU_FILES` gate, same as
`test_cellarrays.jl`. Allocation/overhead measurement (in the style of
`test/test_allocations.jl`) was out of scope for this pass and remains
future work — file I/O dominates any per-call allocation here regardless, so
it's a lower-priority guardrail than it is for the in-memory exchange path.

## Test plan

Add tests in the style already used by the repository:

```text
test/test_hdf5_io.jl
```

Test cases:

1. write one field from a 1-rank topology and read it back;
2. write two fields and recover exact data;
3. compare rank-local metadata against `Grid` and `Field` values;
4. run a 2- or 4-rank MPI case and verify each rank sees the exact local block it owns;
5. verify that mismatched halo widths or `nlocal` shapes fail cleanly;
6. verify that GPU fields require an explicit CPU staging path or are rejected with a clear error message.

The active test file path should converge into a single checked-in test file under the repository’s `test/` tree, and it should keep the failure mode explicit by naming the HDF5 extension file, the saved group path, and the expected metadata string values in the same file’s comments.

## Risks and caveats

1. HDF5 package size and weak dependency shape
   - HDF5 should not be a hard dependency.
   - The extension should be available only when `HDF5` is installed.

2. GPU buffers
   - Do not put device-resident buffers directly into the HDF5 extension.
   - Use a CPU stage path, mirroring the ext conventions already used elsewhere in the repo.

3. Rank-local file offsets
   - Keep the rank file offsets explicit because file layout must be per-grid/domain consistent.

4. Ownership of metadata
   - The file metadata should not drift from the `Grid`/`Field` data model.

5. Overlap with existing gather routines
   - The existing `gather!` path already builds a globally gathered array from MPI rank data.
   - This extension should not reinvent that path unless the file format explicitly needs a global multi-rank schema.

Tuple-of-fields, rank-aware file offset semantics, and GPU staging safety
(via the `StridedArray` gate) are all implemented and tested, not deferred.
What remains genuinely deferred is a true simultaneous parallel-HDF5
collective path (item 3's "rank-local file offsets" are handled, but the
*transport* is still rank-ordered/serialized, not simultaneous — see Phase
3) and allocation/overhead benchmarking (Phase 4) — both gated on
infrastructure (a parallel-HDF5 build; a deliberate benchmarking pass) not
available or in scope for this round.

## Acceptance criteria

This extension is ready when:

- [x] `HaloArrays` remains cleanly buildable without `HDF5` installed;
- [x] `using HaloArrays, HDF5` or the extension path is enough to enable the entire optional I/O interface;
- [x] `save_hdf5`/`load_hdf5!` verify metadata before touching field data;
- [x] distributed shape and halo metadata are preserved exactly across save/load;
- [x] the extension’s tests pass in the repository’s MPI test matrix.

All five criteria are met: `test/test_hdf5_io.jl` passes across `-n 1/2/4/8`
(confirmed alongside the full `test/runtests.jl` suite, exit code 0), and
`load_hdf5!` validates schema version, topology dims, `nlocal`, `nglobal`,
halo widths, and element type against the requesting `Grid`/`Field` before
reading any payload (`validate_metadata!` in `src/HDF5Ext/metadata.jl`).
Remaining work (a true parallel-HDF5 collective path, allocation
benchmarking) is optional hardening beyond this acceptance gate, not a
blocker for it — see Phase 3/4 above and "Risks and caveats" below.

## File and feature checklist

### Package metadata

- [x] Add `HDF5` as a weak dependency in the package project file.
- [x] Register a new extension entry `HaloArraysHDF5Ext = "HDF5"` in the package project file.
- [x] Add the HDF5 dependency explicitly in the optional `test/Project.toml` environment used for the new test file.
- [x] Add the extension to the tests/`extras`/`targets` environment, mirroring `CellArrays`'s precedent.

The package root remains weak-dependency driven; `[extras]`/`[targets]` in
the root `Project.toml` are legacy fields superseded by `test/Project.toml`
for `Pkg.test` in modern Julia (inert either way), but `HDF5` was added
there too for consistency with how `CellArrays` (also a weakdep with its own
`test/`-project test file) is listed.

### Extension skeleton

- [x] Add the extension glue file `ext/HaloArraysHDF5Ext.jl`.
- [x] Add the implementation tree `src/HDF5Ext/shared.jl`.
- [x] Add the implementation tree `src/HDF5Ext/io.jl`.
- [x] Add the implementation tree `src/HDF5Ext/metadata.jl`.

### Public API

- [x] Add an exported or documented `save_hdf5`/`load_hdf5!` API shape.
- [x] Keep the API explicit about `Grid` and `Field` ownership.
- [x] Provide a single-field save/load first, then a tuple-of-fields path.
- [x] Keep HDF5 filename/group selection explicit and make the local rank layout deterministic.

`save_hdf5`/`load_hdf5!` are declared as stub functions and exported from
`src/HDF5IO.jl` (core package), with concrete methods added from
`src/HDF5Ext/io.jl` via `HaloArrays.save_hdf5(...) = ...` — the same
"stub in core, methods from the extension" pattern as
`is_cuarray`/`is_rocarray` in `backend.jl`. This is a real fix over the
original scaffold, which only defined these functions inside the extension
module itself (reachable solely via `Base.get_extension(HaloArrays,
:HaloArraysHDF5Ext)`, not as genuine public API); now `using HaloArrays,
HDF5; save_hdf5(...)` works directly, and calling either without `HDF5`
loaded raises a clear message instead of a bare `MethodError`. `path` and
`group` are always explicit arguments/keywords (no global default file or
group beyond the `"/"` root default); the per-rank layout is deterministic,
derived entirely from the `Grid`'s `Topology` via `iglobal`.

### Schema and metadata

- [x] Define a file layout with a metadata group and one rank-local data dataset per field.
- [x] Store `nlocal`, `nglobal`, `halowidths`, `ndims`, `dtype`, and `name` in metadata.
- [x] Store enough topology metadata to validate and reconstruct the distributed view.
- [x] Decide whether halo data is stored, dropped, or represented separately by metadata.

Decided: halo data is dropped, not stored. A cell is written only when no
neighbor also owns a copy of it (`local_interior_range` in
`src/HDF5Ext/shared.jl` trims `halowidths[d]` on a side only when a real
neighbor exists there), so the saved dataset is always exactly
`nglobal`-shaped with no duplicated cells — see the "Halo cells are not
stored" section of `docs/src/guide/hdf5-io.md`.

Schema is `HDF5Metadata` (`src/HDF5Ext/metadata.jl`), versioned
`schema_version` (currently `"0.2.0"`; bumped from the original scaffold's
`"0.1.0"` because both the storage location and the payload convention
changed — see below). Attributes are written **per-dataset** (via
`write_metadata_attrs!`/`metadata_from_dataset`), not on the parent group as
the original scaffold did — a group-level bundle can't distinguish two
fields with different `halowidths`/dtype in the same group, which the
tuple-of-fields path needs. `topology_dims`/`nlocal`/`nglobal`/`halowidths`
are stored as native HDF5 integer-array attributes (`Vector{Int}`) rather
than string-encoded tuples parsed back with a hand-rolled parser — HDF5.jl
supports this natively, so the earlier scaffold's `parse_int_tuple_string`
was unnecessary and has been removed.

### I/O implementation

- [x] Implement a first-slice `save_hdf5` for one `Field` and one `Grid`.
- [x] Implement a first-slice `load_hdf5!` for one `Field` and one `Grid`.
- [x] Implement a tuple-of-fields save/load loop.
- [x] Ensure ranks only write/read their own local slices.
- [x] Make file creation/opening and metadata writes collective and rank-aware.
- [x] Avoid touching halo cells as normal stored data.

`test_hdf5_io.jl` lives under `test/` and runs via `test/Project.toml` (the
environment that carries the real `HDF5` dependency), gated through
`TEST_PROJECT_CPU_FILES` in `test/runtests.jl` exactly like
`test_cellarrays.jl` — same weakdep/real-dep split the plan called for. It
covers: a metadata-contract check, single-field and tuple-of-fields
save/load round trips (2D and 3D, periodic and non-periodic, with
`update_halo!` closing the loop), the positional-default-names path, and the
full validation/safety list below. All of it passes across the repository's
full `-n 1/2/4/8` matrix.

### Validation and safety

- [x] Validate `Field` shape versus saved `nlocal` metadata.
- [x] Validate halo width metadata against the field object.
- [x] Reject a mismatched topology dimensionality.
- [x] Reject overlapping or duplicate field names.
- [x] Reject GPU arrays without a CPU staging path.
- [x] Reject partial file overwrite or schema drift with a clear error.

`validate_metadata!` (`src/HDF5Ext/metadata.jl`) checks, in order,
`schema_version`, `topology_dims`, `nlocal`, `halowidths`, `nglobal`, and
`dtype_name` against the requesting `Grid`/`Field`, before any payload
read — each check names both the saved and requested values on failure.
Dimensionality mismatches are additionally caught structurally (both the
`Grid{N}`/`Field{T,N}` type parameter and an explicit `ndims(f) == N` check
in `check_field_batch`). Duplicate field names are rejected via
`allunique(names)` before any file is touched. GPU rejection uses a
`field.data isa StridedArray` gate (`require_cpu_array`) — deliberately not
an `isa Array` check, since that would also reject legitimate CPU
`SubArray`/`ReshapedArray` views; `CuArray`/`ROCArray` don't subtype
`StridedArray` (it implies a valid CPU `pointer(A)`), so this stays correct
without importing CUDA/AMDGPU into this extension. This is validated in
`test_hdf5_io.jl` with a `BitArray`-backed field (a real, dependency-free
non-`StridedArray` type) rather than real GPU hardware, which isn't
available in this environment — same caveat as the AMDGPU extension itself.
Reusing an existing dataset name in an existing file is rejected by
`write_field_dataset!`'s `haskey` guard (only reachable via
`mode="r+"`/`"cw"`, since `mode="w"`, the default, always starts from a
truncated, empty file).

### Tests

- [x] Add `test/test_hdf5_io.jl`.
- [x] Add a 1-rank write/read smoke test.
- [x] Add a 2-rank MPI write/read test.
- [x] Add a field-shape mismatch test.
- [x] Add a halo-width mismatch test.
- [x] Add a GPU fallback or staged CPU-path test.

`test_hdf5_io.jl` is a real regression suite, not a placeholder: it runs
across `-n 1/2/4/8` (`TEST_PROJECT_CPU_FILES` in `test/runtests.jl`, same
gate as `test_cellarrays.jl`), and every case passed there and in a full
`test/runtests.jl` run (exit code 0). "Field-shape mismatch" and
"halo-width mismatch" are covered by the `"validation and safety"` testset
(`nlocal mismatch on load`, `halowidths mismatch on load`, plus unknown
field name, duplicate names, dataset-already-exists, and non-CPU-array
cases); "GPU fallback" is covered by the `BitArray`-based
non-`StridedArray` rejection case described above.

### Documentation

- [x] Document the extension’s optional `HDF5` dependency in the package docs.
- [x] Add an example showing `save_hdf5` and `load_hdf5!` with one rank-local field.
- [x] Document the new metadata contract clearly.

`docs/src/guide/hdf5-io.md` is the full guide (single-field and multi-field
examples, the halo/interior storage convention, validation behavior, the
rank-ordered-not-simultaneous write rationale, and the CPU-only
requirement), linked from `docs/src/index.md`'s "Current scope" section and
`docs/src/api.md` (which also carries the `save_hdf5`/`load_hdf5!`
docstrings via `@docs` blocks). `docs/make.jl` registers the new guide page.

## Recommended first implementation slice

The first useful, minimum-risk slice was:

- a single-field, single-grid `save_hdf5`/`load_hdf5!` path;
- an explicit metadata schema;
- a single data layout; and
- a 1-rank and 2-rank correctness smoke test.

That slice is done, and the implementation grew beyond it in this pass (see
"Status" at the top): tuple-of-fields, interior-only/halo-excluded storage
with a real global offset model, and a `-n 1/2/4/8` test matrix, without a
full package-level API redesign — `save_hdf5`/`load_hdf5!` are additive,
stub-based exports that don't touch any existing `HaloArrays` function.
