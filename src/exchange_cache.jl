export ExchangeCache

"""
    ExchangeCache

Per-`Grid`-owned scratch state for halo exchange: raw send/recv byte buffers
(one pair per field position, per call) and the MPI request array. This is a
mutable struct whose fields are reassigned as buffers grow — never a
module-level `let`-captured variable — precisely so that two live `Grid`s
never share or clobber each other's buffers/requests, and so that growing a
buffer is a plain struct-field store rather than the reassignment of a
closure-captured local (which Julia would box, and which we measured as slower
than the allocation it would avoid).

`pending` guards the "one in-flight exchange per `Grid`" v1 restriction: it is
set by `halo_exchange_start!` and cleared by `halo_exchange_finish!`, so a
second `start!` before a matching `finish!` errors instead of corrupting
buffers shared between the two overlapping exchanges.

`threaded_pack` is the opt-in flag (set at `grid` construction, see
[`grid`](@ref)) for splitting large pack/unpack copies across
`Threads.@spawn` tasks. It defaults to `false`: real OS-thread parallelism
here has a real, unavoidable allocation cost (each spawned task allocates,
measured ~500-670B regardless of chunk size — see "Performance, memory
footprint, and GC-pressure goals" in REDESIGN_PLAN.md), so it must never
turn on under a caller who hasn't asked for it and expects the package's
zero-allocation default to hold.

`gpu_cache` is an untyped (`Ref{Any}`) slot, `nothing` until a GPU extension
(CUDA/AMDGPU) lazily constructs its own device-buffer cache struct into it on
this grid's first GPU-array exchange — see `src/backend.jl`. The core package
never inspects its contents; keeping it `Any`-typed here means the core
doesn't need to know the GPU cache's concrete type, and a CPU-only grid that
never touches a GPU array never allocates one.
"""
mutable struct ExchangeCache
    sendbufs::Vector{NTuple{2,Vector{UInt8}}}  # sendbufs[field_i][neighbor_n] -> raw bytes
    recvbufs::Vector{NTuple{2,Vector{UInt8}}}
    reqs::Array{MPI.Request,3}                 # reqs[field_i, neighbor_n, phase]; phase 1=recv, 2=send
    pending::Bool
    threaded_pack::Bool
    gpu_cache::Ref{Any}
end

ExchangeCache(threaded_pack::Bool = false) = ExchangeCache(
    NTuple{2,Vector{UInt8}}[],
    NTuple{2,Vector{UInt8}}[],
    fill(MPI.REQUEST_NULL, 0, 2, 2),
    false,
    threaded_pack,
    Ref{Any}(nothing),
)

function ensure_capacity!(cache::ExchangeCache, nfields::Integer)
    while length(cache.sendbufs) < nfields
        push!(cache.sendbufs, (UInt8[], UInt8[]))
        push!(cache.recvbufs, (UInt8[], UInt8[]))
    end
    if size(cache.reqs, 1) < nfields
        newreqs = fill(MPI.REQUEST_NULL, nfields, 2, 2)
        newreqs[1:size(cache.reqs, 1), :, :] .= cache.reqs
        cache.reqs = newreqs
    end
    return nothing
end

# Return a length-`len` view of `raw`, reinterpreted as `T`, growing `raw` first if
# needed. MPI.Buffer only knows how to unwrap `Array`/`SubArray`/`Ref`, not a
# `ReinterpretArray` (nor a `ReinterpretArray` of a `SubArray`), so `raw` must be
# reinterpreted whole (giving a plain `ReinterpretArray` over the `Vector{UInt8}`)
# and only *then* viewed down to the needed length (giving the required top-level
# `SubArray`) — never the other way around. That also means `length(raw)` must be
# an exact multiple of `sizeof(T)` at the point of reinterpreting, which may not
# already hold if this slot was last grown for a differently-sized element type.
function typed_buf!(raw::Vector{UInt8}, ::Type{T}, len::Integer) where {T}
    nbytes = len * sizeof(T)
    if length(raw) < nbytes
        resize!(raw, max(nbytes, 2 * length(raw)))
    end
    if length(raw) % sizeof(T) != 0
        resize!(raw, length(raw) + (sizeof(T) - length(raw) % sizeof(T)))
    end
    typed = reinterpret(T, raw)
    return view(typed, 1:len)
end

sendbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T} =
    typed_buf!(cache.sendbufs[i][n], T, len)
recvbuf(cache::ExchangeCache, i::Integer, n::Integer, ::Type{T}, len::Integer) where {T} =
    typed_buf!(cache.recvbufs[i][n], T, len)
