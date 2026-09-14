import HaloArrays
import HaloArrays: ExchangeCache, typed_buf!
using CUDA

HaloArrays.is_cuarray(::AnyCuArray) = true

"""
    CUDACache

Per-`Grid`-owned device-buffer pool for GPU halo exchange, lazily constructed
into `ExchangeCache.gpu_cache` (see `src/backend.jl` and `src/exchange_cache.jl`
in HaloArrays.jl) on this grid's first `CuArray` field — never module-global,
same rule as `ExchangeCache` itself. Structurally mirrors `ExchangeCache`'s own
`sendbufs`/`recvbufs`: one raw byte buffer per `(field position, neighbor)`,
grown on demand by the same `typed_buf!` HaloArrays already uses for its host
buffers (it is generic over `AbstractVector{UInt8}` for exactly this reason).

Never holds an MPI request or touches MPI directly — `ExchangeCache.reqs`
remains the only request array regardless of a field's array type, since MPI
only ever sends/receives the host `sendbuf`/`recvbuf` (see `stage_send!`/
`stage_recv!` in halo_exchange.jl).
"""
mutable struct CUDACache
    sendbufs::Vector{NTuple{2,CuVector{UInt8}}}
    recvbufs::Vector{NTuple{2,CuVector{UInt8}}}
end

CUDACache() = CUDACache(NTuple{2,CuVector{UInt8}}[], NTuple{2,CuVector{UInt8}}[])

function ensure_capacity!(gc::CUDACache, nfields::Integer)
    while length(gc.sendbufs) < nfields
        push!(gc.sendbufs, (CuVector{UInt8}(undef, 0), CuVector{UInt8}(undef, 0)))
        push!(gc.recvbufs, (CuVector{UInt8}(undef, 0), CuVector{UInt8}(undef, 0)))
    end
    return nothing
end

# Lazily builds (once per grid) and returns this ExchangeCache's CUDACache. The
# core package never constructs or names this type — cache.gpu_cache is a bare
# Ref{Any} it never inspects (see exchange_cache.jl) — so a CPU-only grid never
# allocates one.
function cuda_cache!(cache::ExchangeCache)
    gc = cache.gpu_cache[]
    if gc === nothing
        gc = CUDACache()
        cache.gpu_cache[] = gc
    end
    return gc::CUDACache
end

function gpu_sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T}
    gc = cuda_cache!(cache)
    ensure_capacity!(gc, i)
    return typed_buf!(gc.sendbufs[i][n], T, len)
end
function gpu_recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T}
    gc = cuda_cache!(cache)
    ensure_capacity!(gc, i)
    return typed_buf!(gc.recvbufs[i][n], T, len)
end
