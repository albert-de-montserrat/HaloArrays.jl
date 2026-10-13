import HaloGrids
import HaloGrids: ExchangeCache, sendbuf, recvbuf, MPI
using CUDA

# pack!/unpack! (HaloGrids.jl's halo_exchange.jl) are already just
# `copyto!(buf, view(A, ranges...))`/`copyto!(view(A, ranges...), buf)` —
# generic over `A`'s type. GPUArrays.jl already gives `CuArray` a strided,
# scalar-indexing-free `copyto!` for these device<->device shapes (both a
# contiguous linear buffer and a genuinely strided halo-slab view work,
# verified directly against this package's own sendrange/recvrange shapes in
# 2D and 3D), so no custom kernel is needed here: pack_target/unpack_source
# just need to hand pack!/unpack! a device-resident buffer instead of the
# host one, and pack!/unpack! themselves need no GPU awareness at all.
HaloGrids.pack_target(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    gpu_sendbuf(cache, i, n, T, len)
HaloGrids.unpack_source(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T} =
    gpu_recvbuf(cache, i, n, T, len)

# The device<->host staging copies MPI transport needs (MPI always
# sends/receives the host sendbuf/recvbuf — see backend.jl). These move raw
# bytes directly between the two *unwrapped* Vector{UInt8}/CuVector{UInt8}
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
function HaloGrids.stage_send!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    cache.gpu_aware_mpi && return nothing
    nbytes = len * sizeof(T)
    sendbuf(cache, i, n, T, len)
    gpu_sendbuf(cache, i, n, T, len)
    copyto!(cache.sendbufs[i][n], 1, cuda_cache!(cache).sendbufs[i][n], 1, nbytes)
    return nothing
end
function HaloGrids.stage_recv!(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    cache.gpu_aware_mpi && return nothing
    nbytes = len * sizeof(T)
    recvbuf(cache, i, n, T, len)
    gpu_recvbuf(cache, i, n, T, len)
    copyto!(cuda_cache!(cache).recvbufs[i][n], 1, cache.recvbufs[i][n], 1, nbytes)
    return nothing
end

# GPU-aware MPI: MPI reads and writes the raw device byte buffers directly. Both
# sides of a message use the same element type and length, so sending `nbytes`
# of `UInt8` is equivalent to sending `len` elements of `T`, and the plain
# `CuVector{UInt8}` avoids relying on MPI.jl's handling of reinterpreted views.
function HaloGrids.mpi_sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    cache.gpu_aware_mpi || return MPI.Buffer(sendbuf(cache, i, n, T, len))
    gpu_sendbuf(cache, i, n, T, len)
    return MPI.Buffer(cuda_cache!(cache).sendbufs[i][n], len * sizeof(T), MPI.Datatype(UInt8))
end
function HaloGrids.mpi_recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AnyCuArray) where {T}
    cache.gpu_aware_mpi || return MPI.Buffer(recvbuf(cache, i, n, T, len))
    gpu_recvbuf(cache, i, n, T, len)
    return MPI.Buffer(cuda_cache!(cache).recvbufs[i][n], len * sizeof(T), MPI.Datatype(UInt8))
end

# Pack kernels, and the previous dimension's unpack kernels, run asynchronously
# on the task's stream and use the same device buffers MPI is about to access.
function HaloGrids.prepare_mpi!(cache::ExchangeCache, ::AnyCuArray)
    cache.gpu_aware_mpi || return nothing
    CUDA.synchronize()
    return nothing
end
