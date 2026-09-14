# Topologies, grids, and fields

The package separates process layout, local sizing, and array storage:

| Value | Responsibility | Constructor |
| --- | --- | --- |
| `Topology` | Cartesian communicator, process dimensions, neighbors | `topology((0, 0))` |
| `Grid` | Local array size, coordinate metadata, exchange cache | `grid(t, (34, 34))` |
| `Field` | Existing array and halo widths | `field(A; halowidths=(1, 1))` |

Both the constructor functions and the type names are exported.

## Process topology

```julia
t = topology((0, 0); periods=(true, false))
```

Topology dimensions count MPI processes. A zero lets MPI choose that dimension;
fixed dimensions must be compatible with the communicator size. Only two and
three dimensions are supported. Every constructor call creates a fresh Cartesian
communicator. `t.rank` and `t.coords` are zero-based; use `t.rank` when selecting
a root inside `t.comm`, since rank reordering is enabled by default.

`t.neighbors[d]` contains the lower and upper neighbors along dimension `d`.
Use `has_neighbor(rank)` to distinguish a real rank from `MPI.PROC_NULL`.
At a nonperiodic domain edge, the missing neighbor does not supply boundary data;
the application must impose its physical boundary conditions.

## Local sizes include halos

```julia
g = grid(t, (34, 34))
A = zeros(g.nlocal)
f = field(A; halowidths=(1, 1))
```

The field wraps `A` without copying it. Access values through `f.data` or `A`.
`Field` is a wrapper, not an `AbstractArray` with its own indexing interface.
Its interior is `A[2:33, 2:33]`. In general, dimension `d` has interior indices
`h+1:size(A,d)-h`, where `h = halowidths(f)[d]`.

Field construction requires a bits element type, positive halo widths, and
`2h <= size(A,d)`. Choose local sizes with enough interior cells for the intended
stencil; constructor validation alone does not establish that a numerical stencil
is meaningful.

All fields in an exchange must match `g.nlocal` and the grid dimensionality.
Fields may have different element types or halo widths. Use distinct backing
arrays: duplicate array objects are rejected, but overlapping views are not
comprehensively checked for aliasing.

## Multiple grids

A grid owns its cache, so sequential use of multiple grids needs no global reset.
For independently overlapping exchanges, construct a separate topology for each
grid. Grids built on the same topology share its communicator, and exchange
messages use the same tag; separate caches alone do not isolate those messages.

`origin`, `origin_on_vertex`, and `centered` are coordinate metadata on `Grid`.
They do not generate coordinates or change halo widths. See [the API](../api.md#grid)
for the parity rules used when constructing centered grids.
