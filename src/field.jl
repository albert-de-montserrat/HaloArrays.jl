export field, halowidths, Field

"""
    Field{T,N,A}

A local array paired with its own per-dimension halo width. Not tied to any
`Grid` at construction time — the `Grid` is supplied later, at the exchange
call site — so a `Field` never needs to know about multi-grid concerns, and
the same field could (in principle) be exchanged against different grids in
different calls.

`halowidths[d]` is the number of boundary cells, in dimension `d`, that are
exchanged with neighbors: the outer `halowidths[d]` cells on each side of the
local array are treated as halo (overwritten on receive), and the adjacent
interior cells of the same width are what gets sent to the neighbor. This
replaces the old package's separate `overlaps`/`halowidths` pair with a single
concept: there is no grid-wide default and no notion of a halo wider than what
each field itself declares.
"""
struct Field{T,N,A<:AbstractArray{T,N}}
    data::A
    halowidths::NTuple{N,Int}
end

"""
    field(A::AbstractArray; halowidths::NTuple{N,Integer} = ntuple(_->1,N)) :: Field

Wrap array `A` as a `Field` with the given per-dimension `halowidths`, defaulting
to a width of `1` in every dimension (the common case for a single-cell-wide
stencil). Requires, for every dimension `d`: `halowidths[d] >= 1` and
`2*halowidths[d] <= size(A,d)` (the local array must be at least wide enough to
hold two non-overlapping halo copies — one sent, one received — in every
dimension). `eltype(A)` must be a bits type.

This default lives here, on the constructor, and nowhere else — there is still
no grid-wide or package-wide halowidths default stored anywhere, so different
`Field`s (even wrapping the same array in different calls) remain free to use
different widths. This is what lets [`update_halo!`](@ref)'s bare-`AbstractArray`
convenience methods mix a default-width array and an explicitly-`field`-wrapped,
custom-width array in the very same call.
"""
function field(A::AbstractArray{T,N}; halowidths::NTuple{N,Integer} = ntuple(_ -> 1, N)) where {T,N}
    isbitstype(T) || error("Field element type $T is not a bits type.")
    hw = Int.(halowidths)
    for d in 1:N
        hw[d] >= 1 || error("halowidths[$d]=$(hw[d]) must be >= 1.")
        2 * hw[d] <= size(A, d) || error(
            "halowidths[$d]=$(hw[d]) is too large for size(A,$d)=$(size(A,d)): " *
            "need 2*halowidths[$d] <= size(A,$d) so the sent and received halo " *
            "strips don't overlap.")
    end
    return Field{T,N,typeof(A)}(A, Tuple(hw))
end

"""
    halowidths(f::Field) :: NTuple{N,Int}

The per-dimension halo widths `f` was constructed with.
"""
halowidths(f::Field) = f.halowidths

Base.size(f::Field, args...) = size(f.data, args...)
Base.ndims(::Field{T,N}) where {T,N} = N
Base.eltype(::Field{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", f::Field{T,N}) where {T,N}
    print(io, "Field{$T,$N}(size=$(size(f.data)), halowidths=$(f.halowidths))")
end
