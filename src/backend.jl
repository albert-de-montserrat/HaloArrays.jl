"""
Stub functions overridden by the CUDA/AMDGPU package extensions (`ext/HaloArraysCUDAExt.jl`,
`ext/HaloArraysAMDGPUExt.jl`). Resolved via ordinary multiple dispatch on the array
type — e.g. the CUDA extension adds a method `is_cuarray(::CuArray) = true` — so,
unlike the old package, no runtime "is the extension loaded" flag is needed at all.

GPU-specific cached state (streams, device buffers) lives behind the `Ref{Any}`
`gpu_cache` slot owned by the `Grid`'s `ExchangeCache` (see exchange_cache.jl),
never module-global.
"""
is_cuarray(::AbstractArray) = false
is_rocarray(::AbstractArray) = false
is_gpuarray(A::AbstractArray) = is_cuarray(A) || is_rocarray(A)

"""
GPU dispatch seam for pack/unpack and MPI transport, resolved by array
type exactly like `is_cuarray`/`is_rocarray` above — a GPU extension adds
methods for `AnyCuArray`/`AnyROCArray` and nothing else needs to change at the
call sites in `halo_exchange.jl`.

For a plain CPU `AbstractArray`, `pack_target`/`unpack_source` and
`mpi_sendbuf`/`mpi_recvbuf` use the host buffers and `stage_send!`/`stage_recv!`
are no-ops, so these defaults reproduce today's CPU behavior exactly, at zero
extra runtime cost once specialized per concrete array type.

For a GPU array, a device-resident buffer pool (owned by the grid's
`ExchangeCache.gpu_cache`, never module-global) is the pack/unpack target and
source — `pack!`/`unpack!` then run on device via `copyto!`/`view`. Backend
methods `mpi_sendbuf`/`mpi_recvbuf` select the transport buffers. They default
to host buffers; a GPU extension can select device buffers when its MPI
implementation reports GPU-buffer support. `stage_send!`/`stage_recv!` bridge
device and host buffers when that support is unavailable, and are no-ops for
direct device-buffer transport.
"""
pack_target(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    sendbuf(cache, i, n, T, len)
unpack_source(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} =
    recvbuf(cache, i, n, T, len)
stage_send!(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} = nothing
stage_recv!(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, ::AbstractArray) where {T} = nothing
mpi_sendbuf(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, A::AbstractArray) where {T} = sendbuf(cache, i, n, T, len)
mpi_recvbuf(cache, i::Integer, n::Integer, ::Type{T}, len::Integer, A::AbstractArray) where {T} = recvbuf(cache, i, n, T, len)
