# Benchmark-only candidate. No methods in HaloArrays are replaced.
module ExchangeCompletionCandidate

using MPI, HaloArrays
const HA = HaloArrays

struct CompletionCache{G<:Grid}
    grid::G
    recvs::MPI.RequestSet
    sends::MPI.RequestSet
end

CompletionCache(g::Grid, nfields::Integer) = CompletionCache(g,
    MPI.RequestSet(fill(MPI.REQUEST_NULL, 2nfields)),
    MPI.RequestSet(fill(MPI.REQUEST_NULL, 2nfields)))

@inline collect_requests!(state, ::Tuple{}, lo, hi, i) = nothing
@inline function collect_requests!(state, fields::Tuple, lo, hi, i)
    cache = state.grid.excache
    for (side, neighbor) in ((1, lo), (2, hi))
        slot = 2(i-1) + side
        state.recvs[slot] = has_neighbor(neighbor) ? cache.reqs[i, side, 1] : MPI.REQUEST_NULL
        state.sends[slot] = has_neighbor(neighbor) ? cache.reqs[i, side, 2] : MPI.REQUEST_NULL
    end
    collect_requests!(state, Base.tail(fields), lo, hi, i + 1)
    return nothing
end

@inline stage_and_unpack!(cache, ::Tuple{}, dim, lo, hi, i) = nothing
@inline function stage_and_unpack!(cache, fields::Tuple, dim, lo, hi, i)
    f = fields[1]
    len = HA.halo_length(f, dim)
    if has_neighbor(lo)
        HA.stage_recv!(cache, i, 1, eltype(f), len, f.data)
        HA.unpack!(f.data, HA.unpack_source(cache, i, 1, eltype(f), len, f.data),
            HA.recvrange(f, dim, 1), cache.threaded_pack)
    end
    if has_neighbor(hi)
        HA.stage_recv!(cache, i, 2, eltype(f), len, f.data)
        HA.unpack!(f.data, HA.unpack_source(cache, i, 2, eltype(f), len, f.data),
            HA.recvrange(f, dim, 2), cache.threaded_pack)
    end
    stage_and_unpack!(cache, Base.tail(fields), dim, lo, hi, i + 1)
    return nothing
end

function complete_dim!(state, handle::ExchangeHandle)
    handle.dim == 0 && return nothing
    g = state.grid
    if handle.self_periodic
        HA.unpack_dim!(g, handle.fields, handle.dim)
    else
        lo, hi = g.topology.neighbors[handle.dim]
        collect_requests!(state, handle.fields, lo, hi, 1)
        MPI.Waitall(state.recvs)
        stage_and_unpack!(g.excache, handle.fields, handle.dim, lo, hi, 1)
        MPI.Waitall(state.sends)
    end
    return nothing
end

@inline exchange_dims!(state, fields, dims::Tuple{Any}) =
    HA.exchange_dim!(state.grid, fields, dims[1]; overlap=true)
@inline function exchange_dims!(state, fields, dims::Tuple)
    handle = HA.exchange_dim!(state.grid, fields, dims[1]; overlap=true)
    complete_dim!(state, handle)
    return exchange_dims!(state, fields, Base.tail(dims))
end

function start!(state::CompletionCache, fields::Tuple, dims::Tuple)
    g = state.grid
    g.excache.pending && error("an exchange is already pending on this grid")
    length(state.recvs) == 2length(fields) || error("candidate cache field count mismatch")
    HA.check_fields(g, fields)
    HA.check_dims(Val(ndims(g)), dims)
    g.excache.pending = true
    return exchange_dims!(state, fields, dims)
end

function finish!(state::CompletionCache, handle::ExchangeHandle)
    handle.grid === state.grid || error("handle belongs to a different grid")
    complete_dim!(state, handle)
    state.grid.excache.pending = false
    return nothing
end

function update!(state::CompletionCache, fields::Tuple, dims::Tuple)
    finish!(state, start!(state, fields, dims))
    return nothing
end

end
