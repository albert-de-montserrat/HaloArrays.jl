# MPI/HDF5 file I/O

`save_hdf5`/`load_hdf5!` write and read `Field` data to/from a single shared
HDF5 file, using the `Grid`'s Cartesian topology to place each rank's block
at its correct offset. This is an optional extension: it activates only once
`HDF5` is loaded alongside `HaloArrays`.

```julia
using HaloArrays
using HDF5   # loads HaloArraysHDF5Ext

t = topology((0, 0))
g = grid(t, (66, 66))
f = field(rand(66, 66); halowidths=(1, 1))

save_hdf5(g, f, "state.h5")

loaded = field(zeros(66, 66); halowidths=(1, 1))
load_hdf5!(g, loaded, "state.h5")
update_halo!(g, loaded)   # repopulate the halo cells load_hdf5! left untouched
```

Calling `save_hdf5`/`load_hdf5!` without `using HDF5` raises a clear error
naming the missing dependency, rather than a bare `MethodError` — see
[`save_hdf5`](@ref)/[`load_hdf5!`](@ref) in the API reference.

## Halo cells are not stored

Just like [`gather!`](./gather.md#halos-are-not-removed-automatically), this
extension is explicit about halos, but in the opposite direction: rather than
requiring the caller to strip them, `save_hdf5` strips them *for* you. A
cell is written only if no neighbor also owns a copy of it — the outer
`halowidths[d]` cells on a side with a real neighbor are skipped, while the
same-width strip on a non-periodic domain boundary (which has no neighbor,
so is genuine, uniquely-owned data, not halo) is written normally. This
keeps the file's per-field dataset exactly `nglobal`-shaped, with no
duplicated cells anywhere.

The consequence: after `load_hdf5!`, halo cells that *do* have a real
neighbor are left exactly as they were in the caller's array (typically
stale or zero) — they were never saved, so there is nothing to load. Call
[`update_halo!`](./halo-exchange.md) afterward to repopulate them from the
now-loaded neighbor data, as in the example above.

## Multiple fields in one file

```julia
u = field(zeros(66, 66); halowidths=(1, 1))
v = field(zeros(66, 66); halowidths=(1, 1))
save_hdf5(g, (u, v), "state.h5"; names=("u", "v"))

u2 = field(zeros(66, 66); halowidths=(1, 1))
v2 = field(zeros(66, 66); halowidths=(1, 1))
load_hdf5!(g, (u2, v2), "state.h5"; names=("u", "v"))
```

`names` defaults to `("field1", "field2", ...)` by position when omitted, so
a save/load pair that both omit it agree on the dataset labels without
having to repeat them. Field names must be unique within a file/group.

## Validation

`load_hdf5!` checks a dataset's saved metadata (schema version, topology
dimensions, `nlocal`, `nglobal`, halo widths, element type) against the
`Grid`/`Field` requesting the load *before* touching any payload, and raises
a plain `error(...)` naming the saved and requested values on the first
mismatch — most commonly a different process-grid shape or `nlocal` than
the file was written with. `save_hdf5` similarly refuses to reuse an
existing dataset name in an existing file (append a distinct `name`/`names`,
use a distinct `group`, or start from a fresh file with `mode="w"`, the
default).

## Why every rank still calls these together

There is no parallel-HDF5 build to make a genuinely simultaneous,
collective multi-rank write meaningful here (`HDF5.has_parallel()` is
`false` for the ordinary HDF5_jll-backed install this package's weak
dependency resolves to). `save_hdf5` still requires every rank in the
grid's topology to call it together — internally, ranks take turns, in
order, opening the shared file, writing only their own hyperslab, and
closing it before the next rank's turn, synchronized by `MPI.Barrier`/
`MPI.Allreduce` — so it remains collective in the sense that matters
(every rank must call it, and the on-disk result depends on all of them),
just not simultaneous. Reads are not serialized this way: concurrent
read-only access to the same file from multiple processes is ordinary, safe
behavior even without a parallel-HDF5 build, so every rank in `load_hdf5!`
opens and reads the file independently.

This also means a call that raises an error must be treated as having
failed on every rank together, even though (depending on which rank's turn
was in progress) not every rank observes the identical exception object —
don't rely on `save_hdf5`/`load_hdf5!` errors being non-fatal on a subset of
ranks.

## CPU arrays only

Both functions require every field's array to be a `StridedArray` (`Array`,
or a contiguous `SubArray`/`ReshapedArray`/`PermutedDimsArray` of one) —
i.e. ordinary, host-addressable memory. A `CuArray`/`ROCArray` field must be
staged to the host first:

```julia
save_hdf5(g, field(Array(gpu_field.data); halowidths=gpu_field.halowidths), path)
```
