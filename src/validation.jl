"""
    check_fields(grid::Grid{N}, fields::NTuple{M,Field}) where {N,M}

Validate a batch of fields against the `Grid` they are about to be exchanged
on. Raises a plain `error(...)` (no custom exception hierarchy, matching the
old package's actual hot-path convention) describing the first problem found:

- a field's dimensionality doesn't match the grid's,
- a field's local array size doesn't match `grid.nlocal`,
- two fields in the same call alias the same underlying array.
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
    f isa Field{T,N} where {T} || error(
        "field at position $i has $(ndims(f)) dimensions, but grid has $N.")
    size(f.data) == grid.nlocal || error(
        "field at position $i has size $(size(f.data)), but grid.nlocal=$(grid.nlocal).")
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
