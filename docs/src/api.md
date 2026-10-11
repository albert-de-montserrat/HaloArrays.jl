# Public API

```@docs
HaloArrays
```

This reference is generated from the docstrings in `src/`, so it always
matches the current source. Examples use CPU arrays. Constructor functions,
`update_halo!`/`halo_exchange_start!`/`halo_exchange_finish!`, `gather!`, the
global-reduction functions (`global_sum`/`global_max`/`global_min`/
`global_norm`/`global_reduce`/`global_allreduce`), and the type names
(`Topology`, `Grid`, `Field`, `ExchangeCache`, `ExchangeHandle`) are all
exported — no qualification is needed for any of them.

## Topology

```@docs
Topology
topology
subtopology
has_neighbor
```

## Grid

```@docs
Grid
grid
nlocal_from_global
```

## Field

```@docs
Field
field
halowidths
overlaps
```

## Halo exchange

```@docs
update_halo!
halo_exchange_start!
halo_exchange_finish!
ExchangeHandle
@hide_halo_communication
```

See [halo exchange](./guide/halo-exchange.md) for ordering and overlap
constraints, and [hiding communication and boundary widths](./guide/hide-communication.md)
for the ParallelStencil schedule, sizing examples, sketches, and safety rules.

## Communication buffers

```@docs
ExchangeCache
```

## Gathering arrays

```@docs
gather!
```

The [gather guide](./guide/gather.md) covers the standard full-dimensional case.

## MPI/HDF5 file I/O

```@docs
save_hdf5
load_hdf5!
```

Optional: these are stub functions until `using HDF5` loads
`HaloArraysHDF5Ext`. See the [MPI/HDF5 file I/O guide](./guide/hdf5-io.md) for
the on-disk layout, the interior/halo storage convention, and multi-field
usage.

## Parallel ParaView (pvtk) I/O

```@docs
save_pvtk
```

Optional: this is a stub function until `using WriteVTK` loads
`HaloArraysWriteVTKExt`. See the
[Parallel ParaView (pvtk) I/O guide](./guide/pvtk-io.md) for the on-disk
layout, the halo-inclusive overlap convention (the opposite of the HDF5
extension's), and worked single-/multi-field examples.

## Global reductions

```@docs
global_allreduce
global_reduce
global_sum
global_max
global_min
global_norm
```

See [global reductions](./guide/reductions.md) for the full rationale and
worked examples.

## Coordinate helpers

Per-field, migrated from the old package's `nx_g`/`x_g`/etc. — see
[coordinate helpers](./guide/coordinates.md) for the full derivation and a
worked example. All are exported.

```@docs
nglobal
nx_g
ny_g
nz_g
iglobal
ix_g
iy_g
iz_g
coord_g
x_g
y_g
z_g
```
