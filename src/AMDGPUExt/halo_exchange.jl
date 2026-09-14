import HaloArrays
import HaloArrays: ExchangeCache, sendbuf, recvbuf
using AMDGPU

# Mirrors src/CUDAExt/halo_exchange.jl exactly (see that file for the full
# rationale on why no custom kernel is needed and why staging must move raw,
# unwrapped bytes rather than typed/viewed buffers) — ROCArray gets the same
# generic strided copyto! from GPUArrays.jl that CuArray does. Untested on
# this machine; see src/AMDGPUExt/shared.jl.
HaloArrays.pack_target(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T} =
    gpu_sendbuf(cache, i, n, T, len)
HaloArrays.unpack_source(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T} =
    gpu_recvbuf(cache, i, n, T, len)

function HaloArrays.stage_send!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T}
    nbytes = len * sizeof(T)
    sendbuf(cache, i, n, T, len)
    gpu_sendbuf(cache, i, n, T, len)
    copyto!(cache.sendbufs[i][n], 1, rocm_cache!(cache).sendbufs[i][n], 1, nbytes)
    return nothing
end
function HaloArrays.stage_recv!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T}
    nbytes = len * sizeof(T)
    recvbuf(cache, i, n, T, len)
    gpu_recvbuf(cache, i, n, T, len)
    copyto!(rocm_cache!(cache).recvbufs[i][n], 1, cache.recvbufs[i][n], 1, nbytes)
    return nothing
end
