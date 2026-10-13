import HaloArrays
import HaloArrays: ExchangeCache, sendbuf, recvbuf, mpi_sendbuf, mpi_recvbuf
import MPI
using CUDA

# pack!/unpack! (HaloArrays.jl's halo_exchange.jl) are already just
# `copyto!(buf, view(A, ranges...))`/`copyto!(view(A, ranges...), buf)` —
# generic over `A`'s type. GPUArrays.jl already gives `CuArray` a strided,
# scalar-indexing-free `copyto!` for these device<->device shapes (both a
# contiguous linear buffer and a genuinely strided halo-slab view work,
# verified directly against this package's own sendrange/recvrange shapes in
# 2D and 3D), so no custom kernel is needed here: pack_target/unpack_source
# just need to hand pack!/unpack! a device-resident buffer instead of the
# host one, and pack!/unpack! themselves need no GPU awareness at all.
HaloArrays.pack_target(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    gpu_sendbuf(cache, i, n, T, len)
HaloArrays.unpack_source(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    gpu_recvbuf(cache, i, n, T, len)

# MPI.jl's CUDA-aware capability check is automatic for supported Open MPI
# builds. Other MPI implementations can report support through
# JULIA_MPI_HAS_CUDA=true (for example, a configured Cray MPICH runtime).
HaloArrays.mpi_sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    MPI.has_cuda() ? gpu_mpi_buffer(cache, i, n, T, len, true) : sendbuf(cache, i, n, T, len)
HaloArrays.mpi_recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    MPI.has_cuda() ? gpu_mpi_buffer(cache, i, n, T, len, false) : recvbuf(cache, i, n, T, len)

# MPI.jl's CUDA buffer conversion accepts CuArray/SubArray-of-CuArray. Our
# typed pack view is a SubArray over ReinterpretArray{T,CuVector{UInt8}}, so
# pass the owning contiguous CuVector with an explicit byte count instead.
# The send/receive pack views address the same raw storage.
function gpu_mpi_buffer(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, sending::Bool) where {T}
    gc = cuda_cache!(cache)
    ensure_capacity!(gc, i)
    if sending
        gpu_sendbuf(cache, i, n, T, len)
        raw = gc.sendbufs[i][n]
    else
        gpu_recvbuf(cache, i, n, T, len)
        raw = gc.recvbufs[i][n]
    end
    return MPI.Buffer(raw, len * sizeof(T), MPI.BYTE)
end

# The device<->host staging copies used when MPI does not report CUDA-buffer
# support. These move raw bytes directly between the two *unwrapped*
# Vector{UInt8}/CuVector{UInt8}
# buffers via the 5-argument `copyto!(dest, doffs, src, soffs, n)` form: the
# natural `copyto!(view_of_reinterpreted_host_buf, view_of_reinterpreted_dev_buf)`
# (i.e. through the same typed views pack_target/sendbuf hand out) silently
# falls back to Base's generic element-at-a-time `copyto!` and hits GPU scalar
# indexing (confirmed by direct reproduction), because GPUArrays.jl's fast
# host<->device path is only defined for the plain, unwrapped Array/CuArray
# types — not a SubArray of a ReinterpretArray. Bytes need no reinterpreting
# to move, so operating on the raw buffers sidesteps the whole issue.
# `sendbuf`/`gpu_sendbuf`/`recvbuf`/`gpu_recvbuf` are still called first, for
# their growth side effect (typed_buf! resizes the raw buffer if needed); the
# typed view each returns is otherwise unused here.
function HaloArrays.stage_send!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    if MPI.has_cuda()
        # Ensure device packing is complete before MPI reads the device buffer.
        CUDA.synchronize()
        return nothing
    end
    nbytes = len * sizeof(T)
    sendbuf(cache, i, n, T, len)
    gpu_sendbuf(cache, i, n, T, len)
    copyto!(cache.sendbufs[i][n], 1, cuda_cache!(cache).sendbufs[i][n], 1, nbytes)
    return nothing
end
function HaloArrays.stage_recv!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    if MPI.has_cuda()
        # MPI.Wait has completed the receive; synchronize before a device
        # unpack reads memory written by the MPI implementation.
        CUDA.synchronize()
        return nothing
    end
    nbytes = len * sizeof(T)
    recvbuf(cache, i, n, T, len)
    gpu_recvbuf(cache, i, n, T, len)
    copyto!(cuda_cache!(cache).recvbufs[i][n], 1, cache.recvbufs[i][n], 1, nbytes)
    return nothing
end
