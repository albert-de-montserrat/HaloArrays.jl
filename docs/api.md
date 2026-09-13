# Public API

This reference follows the current source. Examples use CPU arrays. Constructor
functions, `update_halo!`/`halo_exchange_start!`/`halo_exchange_finish!`,
`gather!`, and the type names (`Topology`, `Grid`, `Field`, `ExchangeCache`,
`ExchangeHandle`) are all exported — no qualification is needed for any of them.

## `topology`

```julia
topology(dims; comm=MPI.COMM_WORLD, periods=ntuple(_ -> false, N),
         disp=1, reorder=true)
```

Returns `Topology{N}` for a 2D or 3D tuple of nonnegative process
counts. Zero entries are resolved by MPI. Initializes MPI if necessary and creates
a Cartesian communicator. `disp` is the process displacement for neighbor lookup;
the default `1` selects adjacent processes.

Properties: `comm`, `dims`, `periods`, `rank`, `coords`, and `neighbors`.
`ndims(t)` returns `N`.

### `has_neighbor`

```julia
has_neighbor(rank::Integer) -> Bool
```

Returns whether the rank differs from `MPI.PROC_NULL`.

## `grid`

```julia
grid(t, nlocal; origin=ntuple(_ -> 0.0, N), origin_on_vertex=false,
     centered=ntuple(_ -> false, N), threaded_pack=false)
```

Returns `Grid{N}` with a new exchange cache. `nlocal` is a tuple of
positive local array sizes including halos, with the same dimensionality as `t`.
For each centered dimension, `origin_on_vertex=true` requires an even local size;
`origin_on_vertex=false` requires an odd local size.

Properties: `topology`, `nlocal`, `origin`, `origin_on_vertex`, and `centered`.
The `excache` property is internal communication state. `ndims(g)` returns `N`.

## `field`

```julia
field(A::AbstractArray; halowidths::NTuple{N,Integer} = ntuple(_ -> 1, N))
```

Returns `Field` wrapping `A`. Requires a bits element type and, in
each dimension, `halowidths[d] >= 1` and `2 * halowidths[d] <= size(A,d)`.
`halowidths` defaults to `1` in every dimension; this is a constructor
default only (nothing is stored on `Grid` or anywhere module-level), so two
`Field`s wrapping different arrays — or the same array in different calls —
remain free to use different widths. Construction does not attach the field
to a grid.

`f.data` is the original array. `size(f)`, `size(f,d)`, `ndims(f)`, and
`eltype(f)` expose its shape and element type.

### `halowidths`

```julia
halowidths(f) -> NTuple{N,Int}
```

Returns the field's halo widths.

## `update_halo!`

```julia
update_halo!(g, fields::Field...; dims=ntuple(identity, ndims(g))) -> Nothing
update_halo!(g, args::Union{Field,AbstractArray}...; dims=ntuple(identity, ndims(g))) -> Nothing
```

Starts and immediately finishes an exchange. Every field must match the grid's
local shape and dimensionality. Repeated references to the same array are rejected.
`dims` must be a nonempty tuple of indices within `1:ndims(g)`.

The second form accepts bare arrays mixed freely with `Field`s in the same
call: a bare array is wrapped with `field`'s default halowidths, while a
`Field` keeps whatever explicit width it was built with —
`update_halo!(g, A, B, field(C; halowidths=(2, 2)))` gives `A` and `B` the
default width and `C` a custom one, in one call. A call whose arguments are
all already `Field`s resolves to the first form (it is the more specific
signature) — the one measured allocation-free on the exchange hot path — so
this convenience costs nothing extra as long as you don't use it.

## `halo_exchange_start!`

```julia
halo_exchange_start!(g, fields::Field...; dims=ntuple(identity, ndims(g)))
halo_exchange_start!(g, args::Union{Field,AbstractArray}...; dims=ntuple(identity, ndims(g)))
```

Returns an opaque `ExchangeHandle`. Completes all requested dimensions
except the last synchronously. Only one exchange may be pending on a grid.
The bare-array form mixes with `Field`s the same way as `update_halo!`'s,
above. See [halo exchange](./guide/halo-exchange) for ordering and overlap
constraints.

## `halo_exchange_finish!`

```julia
halo_exchange_finish!(handle) -> Nothing
```

Waits for outstanding communication, unpacks received halos, and clears the grid's
pending state. Pair each start with one finish.

## `gather!`

```julia
gather!(A, A_global, topology_or_grid; root=0) -> Nothing
```

Collectively assembles arrays on `root` in the Cartesian communicator. Pass
`nothing` as `A_global` on other ranks. For equal dimensionality,
`size(A_global) == size(A) .* topology.dims` and element types must match.
The implementation also permits lower-dimensional local arrays when
`ndims(A) <= ndims(A_global) <= ndims(topology)`, padding the local shape with
singleton dimensions and using the corresponding leading topology dimensions.
The [gather guide](./guide/gather) covers the standard full-dimensional case.

## Coordinate helpers

Per-field, migrated from the old package's `nx_g`/`x_g`/etc. — see
[coordinate helpers](./guide/coordinates) for the full derivation and a
worked example. All are exported.

```julia
nglobal(g::Grid{N}, f::Field{T,N}, dim::Integer) -> Int
nx_g(g, f) -> Int   # = nglobal(g, f, 1)
ny_g(g, f) -> Int   # = nglobal(g, f, 2)
nz_g(g, f) -> Int   # = nglobal(g, f, 3), requires N == 3
```

Global domain size in dimension `dim`, as seen by `f` — depends on `f`'s own
`halowidths`, since that determines how many cells are duplicated across a
rank boundary. Every `Field` exchanged on `g` must already match
`g.nlocal` exactly, so (unlike the old package's `nx_g(A)`) there is no
separate "array bigger/smaller than the base grid" case to account for.

```julia
iglobal(g::Grid{N}, f::Field{T,N}, dim::Integer, i::Integer; wrap_periodic=true) -> Int
ix_g(i, g, f; wrap_periodic=true) -> Int   # = iglobal(g, f, 1, i; wrap_periodic)
iy_g(i, g, f; wrap_periodic=true) -> Int   # = iglobal(g, f, 2, i; wrap_periodic)
iz_g(i, g, f; wrap_periodic=true) -> Int   # = iglobal(g, f, 3, i; wrap_periodic), requires N == 3
```

Global index, in dimension `dim`, of local index `i` (`1` is the first
element of the local array, including its halo). On a periodic dimension,
wraps into `1:nglobal` by default; `wrap_periodic=false` returns the raw,
possibly out-of-range index instead.

```julia
coord_g(g::Grid{N}, f::Field{T,N}, dim::Integer, i::Integer, dx::Real) -> Float64
x_g(i, dx, g, f) -> Float64   # = coord_g(g, f, 1, i, dx)
y_g(i, dx, g, f) -> Float64   # = coord_g(g, f, 2, i, dx)
z_g(i, dx, g, f) -> Float64   # = coord_g(g, f, 3, i, dx), requires N == 3
```

Global physical coordinate of local index `i`, honoring `g.origin`,
`g.origin_on_vertex`, and `g.centered`. Argument order (`i, dx, ...`)
matches the old package's `x_g(ix, dx, A)` for migration ease.
