# Instrumented full exchange on CUDA. Phase synchronization is intentional;
# these measurements are not production end-to-end timings.
using MPI, CUDA, HaloArrays
const HA = HaloArrays
const GPU_PHASES = (:pack, :send_stage, :post, :recv_wait, :recv_stage,
    :unpack, :send_wait, :self_copy)

function gpu_record!(totals, phase, dim, start; device=false)
    device && CUDA.synchronize()
    totals[phase, dim] += (time_ns() - start) * 1e-9
    return nothing
end

@inline gpu_post_sends!(cache, ::Tuple{}, dim, lo, hi, comm, totals, i) = nothing
@inline function gpu_post_sends!(cache, fields::Tuple, dim, lo, hi, comm, totals, i)
    f = fields[1]
    len = HA.halo_length(f, dim)
    for (side, neighbor, tag) in ((1, lo, HA.EXCHANGE_TAG_LO), (2, hi, HA.EXCHANGE_TAG_HI))
        has_neighbor(neighbor) || continue
        t = time_ns()
        HA.stage_send!(cache, i, side, eltype(f), len, f.data)
        gpu_record!(totals, 2, dim, t; device=true)
        t = time_ns()
        cache.reqs[i, side, 2] = MPI.Isend(HA.sendbuf(cache, i, side, eltype(f), len),
            comm; dest=neighbor, tag)
        gpu_record!(totals, 3, dim, t)
    end
    gpu_post_sends!(cache, Base.tail(fields), dim, lo, hi, comm, totals, i + 1)
    return nothing
end

@inline gpu_receive!(cache, ::Tuple{}, dim, lo, hi, totals, i) = nothing
@inline function gpu_receive!(cache, fields::Tuple, dim, lo, hi, totals, i)
    f = fields[1]
    len = HA.halo_length(f, dim)
    for (side, neighbor) in ((1, lo), (2, hi))
        has_neighbor(neighbor) || continue
        t = time_ns()
        MPI.Wait(cache.reqs[i, side, 1])
        gpu_record!(totals, 4, dim, t)
        t = time_ns()
        HA.stage_recv!(cache, i, side, eltype(f), len, f.data)
        gpu_record!(totals, 5, dim, t; device=true)
        t = time_ns()
        HA.unpack!(f.data, HA.unpack_source(cache, i, side, eltype(f), len, f.data),
            HA.recvrange(f, dim, side), cache.threaded_pack)
        gpu_record!(totals, 6, dim, t; device=true)
    end
    gpu_receive!(cache, Base.tail(fields), dim, lo, hi, totals, i + 1)
    return nothing
end

function gpu_profile!(g, fields, totals)
    for dim in 1:3
        lo, hi = g.topology.neighbors[dim]
        t = time_ns()
        HA.pack_dim!(g.excache, fields, dim, lo, hi, 1)
        gpu_record!(totals, 1, dim, t; device=true)
        if lo == g.topology.rank
            t = time_ns()
            HA.copy_self_periodic!(g.excache, fields, dim, 1)
            gpu_record!(totals, 8, dim, t; device=true)
            t = time_ns()
            HA.unpack_dim!(g, fields, dim)
            gpu_record!(totals, 6, dim, t; device=true)
        else
            t = time_ns()
            HA.post_irecv!(g.excache, fields, dim, lo, hi, g.topology.comm, 1)
            gpu_record!(totals, 3, dim, t)
            gpu_post_sends!(g.excache, fields, dim, lo, hi, g.topology.comm, totals, 1)
            gpu_receive!(g.excache, fields, dim, lo, hi, totals, 1)
            t = time_ns()
            HA.wait_sends!(g, fields, dim)
            gpu_record!(totals, 7, dim, t)
        end
    end
    return nothing
end

function main()
    if !CUDA.functional()
        println("SKIP,CUDA not functional")
        return
    end
    CUDA.allowscalar(false)
    niter = isempty(ARGS) ? 300 : parse(Int, ARGS[1])
    shape = length(ARGS) >= 2 ? Tuple(parse.(Int, split(ARGS[2], 'x'))) : (128, 32, 64)
    dims = length(ARGS) >= 3 ? Tuple(parse.(Int, split(ARGS[3], 'x'))) : (0, 0, 0)
    niter > 0 && length(shape) == length(dims) == 3 && all(>=(5), shape) || error("invalid arguments")
    MPI.Init()
    try
        topo = topology(dims; periods=(true, true, true))
        g = grid(topo, shape)
        fields = ntuple(2) do i
            T = isodd(i) ? Float64 : Float32
            field(CuArray(reshape(T[T(1_000_000topo.rank + j) for j in 1:prod(shape)], shape)))
        end
        reference = map(f -> field(copy(f.data)), fields)
        update_halo!(g, reference...)
        HA.ensure_capacity!(g.excache, length(fields))
        totals = zeros(length(GPU_PHASES), 3)
        gpu_profile!(g, fields, totals)
        correct = all(Array(a.data) == Array(b.data) for (a, b) in zip(fields, reference))
        MPI.Allreduce(correct, &, topo.comm) || error("profiled halo mismatch")
        for _ in 1:10
            gpu_profile!(g, fields, totals)
        end
        fill!(totals, 0)
        CUDA.synchronize()
        MPI.Barrier(topo.comm)
        for _ in 1:niter
            gpu_profile!(g, fields, totals)
        end
        maxima = MPI.Allreduce(totals, max, topo.comm) ./ niter
        if topo.rank == 0
            label = "cuda,$(MPI.Comm_size(topo.comm)),$(join(shape, 'x')),$(join(topo.dims, 'x')),2,$niter"
            for dim in 1:3, (p, phase) in pairs(GPU_PHASES)
                println("GPU_PHASE,$label,$dim,$phase,$(maxima[p, dim])")
            end
        end
    finally
        MPI.Finalize()
    end
end

main()
