export grid, Grid

"""
    Grid{N}

Sizing information layered onto a [`Topology`](@ref), plus the per-grid
[`ExchangeCache`](@ref) it owns. No halo-width or field information lives
here — that is the responsibility of [`Field`](@ref) — which is what fixes the
old package's `overlaps` confusion (one halo-width concept, declared per
field, instead of two conflated grid-wide/per-field ones).

There is no module-global "current grid": every `Grid` is an explicit value
returned by [`grid`](@ref) and passed explicitly to every function that needs
it (`update_halo!`, `gather!`, ...). This is what makes multiple, independently
sized, concurrently live grids work by construction.

Fields:
- `topology::Topology{N}`
- `nlocal::NTuple{N,Int}`: the local array size every `Field` exchanged on
  this grid must have, in every dimension (uniform per rank).
- `origin::NTuple{N,Float64}`, `origin_on_vertex::Bool`, `centered::NTuple{N,Bool}`:
  physical-coordinate metadata, kept for migration ease from the old package.
- `excache::ExchangeCache`: this grid's own communication scratch state.
"""
struct Grid{N}
    topology::Topology{N}
    nlocal::NTuple{N,Int}
    origin::NTuple{N,Float64}
    origin_on_vertex::Bool
    centered::NTuple{N,Bool}
    excache::ExchangeCache
end

Base.ndims(::Grid{N}) where {N} = N

"""
    grid(topology::Topology{N}, nlocal::NTuple{N,Integer};
         origin=ntuple(_->0.0,N), origin_on_vertex=false,
         centered=ntuple(_->false,N), threaded_pack=false) :: Grid{N}

Construct a `Grid` from an explicit `Topology` and the local array size every
field on this grid will share. `centered[d]=true` requires `nlocal[d]` to be
even when `origin_on_vertex=true`, or odd when `origin_on_vertex=false` — the
same coherence rule the old package enforced at grid-creation time, since it
determines whether the origin can fall exactly on a vertex/cell-center.

`threaded_pack=true` opts this grid's exchanges into splitting large
pack/unpack copies across `Threads.@spawn` tasks (only when
`Threads.nthreads() > 1` and a given field's halo slab is large enough to be
worth it — see `THREADED_PACK_MIN_LEN` in `halo_exchange.jl`). Off by
default: it trades a real, unavoidable per-call allocation (each spawned task
allocates, regardless of how large the copy is) for wall-clock speed on large
transfers, which is the right trade only when you've asked for it — see
"Performance, memory footprint, and GC-pressure goals" in REDESIGN_PLAN.md
for the measurements behind this default.
"""
function grid(topology::Topology{N}, nlocal::NTuple{N,Integer};
              origin::NTuple{N,Real} = ntuple(_ -> 0.0, N),
              origin_on_vertex::Bool = false,
              centered::NTuple{N,Bool} = ntuple(_ -> false, N),
              threaded_pack::Bool = false) where {N}
    any(<(1), nlocal) && error("nlocal entries must be >= 1; got nlocal=$nlocal.")
    for d in 1:N
        if centered[d] && origin_on_vertex && isodd(nlocal[d])
            error("dimension $d: cannot center on the origin with origin_on_vertex=true " *
                  "and an odd nlocal[$d]=$(nlocal[d]); set origin_on_vertex=false or use an even nlocal[$d].")
        end
        if centered[d] && !origin_on_vertex && iseven(nlocal[d])
            error("dimension $d: cannot center on the origin with origin_on_vertex=false " *
                  "and an even nlocal[$d]=$(nlocal[d]); set origin_on_vertex=true or use an odd nlocal[$d].")
        end
    end
    return Grid{N}(topology, Tuple(Int.(nlocal)), Float64.(origin), origin_on_vertex, centered, ExchangeCache(threaded_pack))
end

function Base.show(io::IO, ::MIME"text/plain", g::Grid{N}) where {N}
    print(io, "Grid{$N}(nlocal=$(g.nlocal), dims=$(g.topology.dims), rank=$(g.topology.rank))")
end
