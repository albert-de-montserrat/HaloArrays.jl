"""
    check_fields(grid::Grid{N}, fields::NTuple{M,Field}) where {N,M}

Validate a batch of fields against the `Grid` they are about to be exchanged
on. Raises a plain `error(...)` (no custom exception hierarchy, matching the
old package's actual hot-path convention) describing the first problem found:

- a field's dimensionality is smaller than the grid's, or its size disagrees
  with `grid.nlocal` in the grid's own leading dimensions,
- two fields in the same call alias the same underlying array.

A field may have *more* dimensions than the grid (e.g. the CellArrays
extension's batched `B=0` path, `src/CellArraysExt/halo_exchange.jl`, wraps a
CellArray's cell-component axis as one extra trailing field dimension): any
dimension beyond the grid's own `N` is never checked against `grid.nlocal`
and is only ever touched by the exchange machinery
(`sendrange`/`recvrange`/`halo_length` in halo_exchange.jl, which already
loop over a field's *own* `ndims`, not the grid's) if it is explicitly named
in a call's `dims` argument — which nothing in this package does for a
dimension beyond `N`, since `dims` is itself validated against the grid's
`N` (`check_dims`, below). A field with *fewer* dimensions than the grid is
still rejected, same as before.
"""
function check_fields(grid::Grid{N}, fields::Tuple) where {N}
    check_field_shapes(grid, fields, 1)
    check_no_aliases(fields, 1)
    return nothing
end

# Recursing on the `Tuple` type (rather than a `for i in 1:n; fields[i]; end`
# indexed loop) is deliberate: `fields` is heterogeneous (different `Field{T,...}`
# per position), and indexing it with a runtime `Int` is not type-stable — the
# compiler can't know which concrete `Field` type comes back, so it boxes.
# Recursing on `Base.tail` instead lets each call specialize on the
# ever-shrinking `Tuple` type and fully unroll/inline away, at zero allocation
# (measured: indexed ~416B/call for 2 fields vs 0B for this pattern — see
# "Performance, memory footprint, and GC-pressure goals" in REDESIGN_PLAN.md).
@inline check_field_shapes(::Grid, ::Tuple{}, i) = nothing
@inline function check_field_shapes(grid::Grid{N}, fields::Tuple, i) where {N}
    f = fields[1]
    f isa Field || error(
        "field at position $i is not a Field (got $(typeof(f))).")
    ndims(f) >= N || error(
        "field at position $i has $(ndims(f)) dimensions, but grid has $N; a field may " *
        "have extra trailing dimensions beyond the grid's own, but not fewer.")
    size(f.data)[1:N] == grid.nlocal || error(
        "field at position $i has size $(size(f.data)); its first $N dimension(s) must " *
        "equal grid.nlocal=$(grid.nlocal) (any dimensions beyond the grid's own $N are " *
        "unconstrained).")
    check_field_shapes(grid, Base.tail(fields), i + 1)
    return nothing
end

@inline check_no_aliases(::Tuple{}, i) = nothing
@inline function check_no_aliases(fields::Tuple, i)
    check_no_alias_with(fields[1].data, Base.tail(fields), i, i + 1)
    check_no_aliases(Base.tail(fields), i + 1)
    return nothing
end
@inline check_no_alias_with(data, ::Tuple{}, i, j) = nothing
@inline function check_no_alias_with(data, rest::Tuple, i, j)
    rest[1].data === data && error(
        "field at position $j is a duplicate of the one at position $i; remove the duplicate from the call.")
    check_no_alias_with(data, Base.tail(rest), i, j + 1)
    return nothing
end
