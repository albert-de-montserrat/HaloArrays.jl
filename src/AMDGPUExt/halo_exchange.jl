import HaloGrids
import HaloGrids: ExchangeCache, sendbuf, recvbuf, MPI
using AMDGPU

# Mirrors src/CUDAExt/halo_exchange.jl exactly (see that file for the full
# rationale on why no custom kernel is needed and why staging must move raw,
# unwrapped bytes rather than typed/viewed buffers) — ROCArray gets the same
# generic strided copyto! from GPUArrays.jl that CuArray does. Untested on
# this machine; see src/AMDGPUExt/shared.jl.
HaloGrids.pack_target(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T} =
    gpu_sendbuf(cache, i, n, T, len)
HaloGrids.unpack_source(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T} =
    gpu_recvbuf(cache, i, n, T, len)

function HaloGrids.stage_send!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T}
    cache.gpu_aware_mpi && return nothing
    nbytes = len * sizeof(T)
    sendbuf(cache, i, n, T, len)
    gpu_sendbuf(cache, i, n, T, len)
    copyto!(cache.sendbufs[i][n], 1, rocm_cache!(cache).sendbufs[i][n], 1, nbytes)
    return nothing
end
function HaloGrids.stage_recv!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T}
    cache.gpu_aware_mpi && return nothing
    nbytes = len * sizeof(T)
    recvbuf(cache, i, n, T, len)
    gpu_recvbuf(cache, i, n, T, len)
    copyto!(rocm_cache!(cache).recvbufs[i][n], 1, cache.recvbufs[i][n], 1, nbytes)
    return nothing
end

# GPU-aware MPI: MPI reads and writes the raw device byte buffers directly. Both
# sides of a message use the same element type and length, so sending `nbytes`
# of `UInt8` is equivalent to sending `len` elements of `T`, and the plain
# `ROCVector{UInt8}` avoids relying on MPI.jl's handling of reinterpreted views.
function HaloGrids.mpi_sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T}
    cache.gpu_aware_mpi || return MPI.Buffer(sendbuf(cache, i, n, T, len))
    gpu_sendbuf(cache, i, n, T, len)
    return MPI.Buffer(rocm_cache!(cache).sendbufs[i][n], len * sizeof(T), MPI.Datatype(UInt8))
end
function HaloGrids.mpi_recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyROCArray) where {T}
    cache.gpu_aware_mpi || return MPI.Buffer(recvbuf(cache, i, n, T, len))
    gpu_recvbuf(cache, i, n, T, len)
    return MPI.Buffer(rocm_cache!(cache).recvbufs[i][n], len * sizeof(T), MPI.Datatype(UInt8))
end

# Pack kernels, and the previous dimension's unpack kernels, run asynchronously
# on the task's stream and use the same device buffers MPI is about to access.
function HaloGrids.prepare_mpi!(cache::ExchangeCache, ::AnyROCArray)
    cache.gpu_aware_mpi || return nothing
    AMDGPU.synchronize()
    return nothing
end
