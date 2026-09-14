# Public API

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
has_neighbor
```

## Grid

```@docs
Grid
grid
```

## Field

```@docs
Field
field
halowidths
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
constraints, and [hiding communication](./guide/halo-exchange.md#parallelstencil-convenience-macro)
for the full ParallelStencil schedule and safety rules.

## Communication buffers

```@docs
ExchangeCache
```

## Gathering arrays

```@docs
gather!
```

The [gather guide](./guide/gather.md) covers the standard full-dimensional case.

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
