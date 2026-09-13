export nglobal, nx_g, ny_g, nz_g, iglobal, ix_g, iy_g, iz_g, coord_g, x_g, y_g, z_g

"""
    nglobal(grid::Grid{N}, field::Field{T,N}, dim::Integer) :: Int

Size of the global domain in dimension `dim`, as seen by `field` — i.e.
accounting for `field`'s own halo width in that dimension. This is a
per-field quantity, not a pure `Grid` property: two fields with different
`halowidths` on the same `Grid` see different global sizes, because the
number of cells duplicated across a rank boundary (`2*halowidths[dim]`) is
part of the accounting.

Unlike the old package's `nx_g`/`ny_g`/`nz_g`, there is no separate
"local array bigger/smaller than the base grid" adjustment: every `Field`
exchanged on a `Grid` must have `size(field.data) == grid.nlocal` exactly
(see [`check_fields`](@ref)), so that source of complexity in the old
`_adjust_nx_g` simply does not exist here — this is the direct benefit of
"one halo-width concept" for the coordinate layer, not just the exchange
one.

Derivation: each rank owns `nlocal[dim] - 2*halowidths[dim]` cells that are
not duplicated with a neighbor, plus, on a non-periodic domain, the outer
`halowidths[dim]` cells at each of the two global edges are genuine domain
cells (not halo from a neighbor that doesn't exist) and must be counted
once, not zero times:

`nglobal[dim] = dims[dim] * (nlocal[dim] - 2*halowidths[dim]) + 2*halowidths[dim] * !periods[dim]`
"""
function nglobal(grid::Grid{N}, field::Field{T,N}, dim::Integer) where {T,N}
    1 <= dim <= N || error("dim=$dim out of range for a $N-dimensional grid.")
    hw = field.halowidths[dim]
    dims = grid.topology.dims[dim]
    nlocal = grid.nlocal[dim]
    return dims * (nlocal - 2 * hw) + 2 * hw * !grid.topology.periods[dim]
end

"""
    nx_g(grid, field) = nglobal(grid, field, 1)
    ny_g(grid, field) = nglobal(grid, field, 2)
    nz_g(grid, field) = nglobal(grid, field, 3)

Convenience wrappers around [`nglobal`](@ref) for the first three dimensions,
named to match the old package's `nx_g()`/`ny_g()`/`nz_g()` for migration
ease. `nz_g` requires a 3D grid.
"""
nx_g(grid::Grid, field::Field) = nglobal(grid, field, 1)
ny_g(grid::Grid, field::Field) = nglobal(grid, field, 2)
nz_g(grid::Grid{3}, field::Field{T,3}) where {T} = nglobal(grid, field, 3)

"""
    iglobal(grid::Grid{N}, field::Field{T,N}, dim::Integer, i::Integer;
            wrap_periodic::Bool=true) :: Int

Global index, in dimension `dim`, corresponding to local index `i` of
`field`'s array on the calling rank. Local index `1` is the first element
of the local array *including* its halo — this matches the old package's
`ix_g`/`iy_g`/`iz_g` convention.

On a periodic dimension, the global index wraps into `1:nglobal` by default;
pass `wrap_periodic=false` to get the raw (possibly `<1` or `>nglobal`)
index instead, e.g. to detect that a cell is a genuine wrap-around halo
rather than a real neighboring cell.

# Examples
```julia
grid = HaloArrays.grid(topology((2,1)), (10, 8))
f = field(zeros(10, 8))
iglobal(grid, f, 1, 1)   # first local index (including halo) -> its global index
```
"""
function iglobal(grid::Grid{N}, field::Field{T,N}, dim::Integer, i::Integer;
                  wrap_periodic::Bool = true) where {T,N}
    1 <= dim <= N || error("dim=$dim out of range for a $N-dimensional grid.")
    hw = field.halowidths[dim]
    coord = grid.topology.coords[dim]
    nlocal = grid.nlocal[dim]
    periodic = grid.topology.periods[dim]
    ig0 = periodic ? -hw : 0
    ig = coord * (nlocal - 2 * hw) + i + ig0
    if periodic && wrap_periodic
        ng = nglobal(grid, field, dim)
        ig > ng && (ig -= ng)
        ig < 1 && (ig += ng)
    end
    return ig
end

"""
    ix_g(i, grid, field; wrap_periodic=true) = iglobal(grid, field, 1, i; wrap_periodic)
    iy_g(i, grid, field; wrap_periodic=true) = iglobal(grid, field, 2, i; wrap_periodic)
    iz_g(i, grid, field; wrap_periodic=true) = iglobal(grid, field, 3, i; wrap_periodic)

Convenience wrappers around [`iglobal`](@ref), named to match the old
package's `ix_g`/`iy_g`/`iz_g`. `iz_g` requires a 3D grid.
"""
ix_g(i::Integer, grid::Grid, field::Field; wrap_periodic::Bool = true) =
    iglobal(grid, field, 1, i; wrap_periodic)
iy_g(i::Integer, grid::Grid, field::Field; wrap_periodic::Bool = true) =
    iglobal(grid, field, 2, i; wrap_periodic)
iz_g(i::Integer, grid::Grid{3}, field::Field{T,3}; wrap_periodic::Bool = true) where {T} =
    iglobal(grid, field, 3, i; wrap_periodic)

"""
    coord_g(grid::Grid{N}, field::Field{T,N}, dim::Integer, i::Integer, dx::Real) :: Float64

Global physical coordinate, in dimension `dim`, of local index `i` of
`field`'s array on the calling rank, with grid spacing `dx`. Honors
`grid.origin[dim]`, `grid.origin_on_vertex`, and `grid.centered[dim]`.

The domain's physical length is `dx*(nglobal-1)` when `origin_on_vertex`
(the `nglobal` points are grid vertices spanning `nglobal-1` cells) or
`dx*nglobal` otherwise (the `nglobal` points are cell centers, each of
width `dx`, so there is an extra half-cell margin at either edge). When
`centered[dim]`, this length is shifted so it is centered on
`grid.origin[dim]`; otherwise `grid.origin[dim]` is the coordinate of
global index `1` (vertex case) or the left edge of the domain (cell-center
case).

Does not attempt to replicate the old package's staggered-array coordinate
adjustment (`nx_g(A)`'s "array bigger/smaller than the base grid" case):
since every `Field` on a `Grid` must share `grid.nlocal` exactly, that case
cannot arise here — see [`nglobal`](@ref).
"""
function coord_g(grid::Grid{N}, field::Field{T,N}, dim::Integer, i::Integer, dx::Real) where {T,N}
    ig = iglobal(grid, field, dim, i; wrap_periodic = false)
    ng = nglobal(grid, field, dim)
    L = grid.origin_on_vertex ? dx * (ng - 1) : dx * ng
    x0 = grid.origin[dim] - (grid.centered[dim] ? L / 2 : 0.0) + (grid.origin_on_vertex ? 0.0 : dx / 2)
    return x0 + (ig - 1) * dx
end

"""
    x_g(i, dx, grid, field) = coord_g(grid, field, 1, i, dx)
    y_g(i, dx, grid, field) = coord_g(grid, field, 2, i, dx)
    z_g(i, dx, grid, field) = coord_g(grid, field, 3, i, dx)

Convenience wrappers around [`coord_g`](@ref), named and ordered
(`i, dx, ...`) to match the old package's `x_g(ix, dx, A)`/`y_g`/`z_g` for
migration ease. `z_g` requires a 3D grid.
"""
x_g(i::Integer, dx::Real, grid::Grid, field::Field) = coord_g(grid, field, 1, i, dx)
y_g(i::Integer, dx::Real, grid::Grid, field::Field) = coord_g(grid, field, 2, i, dx)
z_g(i::Integer, dx::Real, grid::Grid{3}, field::Field{T,3}) where {T} = coord_g(grid, field, 3, i, dx)
