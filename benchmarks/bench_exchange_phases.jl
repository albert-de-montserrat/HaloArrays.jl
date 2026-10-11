# CPU diagnostic worker. Run with the package project, optionally under MPI.
# Instrumentation and the grouped-wait experiment live only in this harness.
using MPI, HaloArrays
const HA = HaloArrays

const PHASES = (:pack, :post, :recv_wait, :unpack, :send_wait, :self_copy)

@inline tick(::Nothing) = UInt64(0)
@inline tick(totals) = time_ns()
@inline record!(::Nothing, phase, dim, start) = nothing
@inline record!(totals, phase, dim, start) =
    (totals[phase, dim] += (time_ns() - start) * 1e-9)

function stencil!(out, A, krange)
    nx, ny, _ = size(A)
    @inbounds for k in krange, j in 2:ny-1, i in 2:nx-1
        out[i, j, k] = A[i, j, k] + 0.1 * (
            A[i-1, j, k] + A[i+1, j, k] + A[i, j-1, k] +
            A[i, j+1, k] + A[i, j, k-1] + A[i, j, k+1] - 6A[i, j, k])
    end
    return nothing
end

function public_step!(g, fields, out, overlap)
    nz = size(fields[1].data, 3)
    if overlap
        h = halo_exchange_start!(g, fields...)
        try
            stencil!(out, fields[1].data, 3:nz-2)
        finally
            halo_exchange_finish!(h)
        end
        stencil!(out, fields[1].data, (2, nz-1))
    else
        update_halo!(g, fields...)
        stencil!(out, fields[1].data, 2:nz-1)
    end
    return nothing
end

# CPU-only: no staging copies. Ordered completion mirrors the package's
# wait/unpack interleaving; grouped completion uses reusable MPI.RequestSets.
function diagnostic_exchange!(g, fields, totals, grouped, recvs, sends)
    cache = g.excache
    for dim in 1:3
        lo, hi = g.topology.neighbors[dim]
        t = tick(totals)
        HA.pack_dim!(cache, fields, dim, lo, hi, 1)
        record!(totals, 1, dim, t)
        if lo == hi == g.topology.rank
            t = tick(totals)
            HA.copy_self_periodic!(cache, fields, dim, 1)
            record!(totals, 6, dim, t)
            t = tick(totals)
            HA.unpack_dim!(g, fields, dim)
            record!(totals, 4, dim, t)
            continue
        end
        t = tick(totals)
        HA.post_irecv!(cache, fields, dim, lo, hi, g.topology.comm, 1)
        HA.post_isend!(cache, fields, dim, lo, hi, g.topology.comm, 1)
        if grouped
            for i in eachindex(fields), side in 1:2
                slot = 2(i-1) + side
                recvs[slot] = cache.reqs[i, side, 1]
                sends[slot] = cache.reqs[i, side, 2]
            end
        end
        record!(totals, 2, dim, t)
        if grouped
            t = tick(totals)
            MPI.Waitall(recvs)
            record!(totals, 3, dim, t)
        end
        for (i, f) in pairs(fields), side in 1:2
            if !grouped
                t = tick(totals)
                MPI.Wait(cache.reqs[i, side, 1])
                record!(totals, 3, dim, t)
            end
            t = tick(totals)
            len = HA.halo_length(f, dim)
            HA.unpack!(f.data, HA.recvbuf(cache, i, side, eltype(f), len),
                HA.recvrange(f, dim, side), cache.threaded_pack)
            record!(totals, 4, dim, t)
        end
        t = tick(totals)
        if grouped
            MPI.Waitall(sends)
        else
            HA.wait_sends!(g, fields, dim)
        end
        record!(totals, 5, dim, t)
    end
    return nothing
end

function sample!(work!, g, niter)
    for _ in 1:10
        work!()
    end
    GC.gc()
    MPI.Barrier(g.topology.comm)
    stats = @timed for _ in 1:niter
        work!()
    end
    elapsed = MPI.Allreduce(stats.time, max, g.topology.comm) / niter
    bytes = MPI.Allreduce(stats.bytes, max, g.topology.comm) / niter
    return elapsed, bytes
end

function run_case(shape, dims, nfields, niter)
    topo = topology(dims; periods=(true, true, true))
    g = grid(topo, shape)
    fields = ntuple(nfields) do f
        A = Array{Float64}(undef, shape)
        for i in eachindex(A)
            A[i] = sin(i + 13topo.rank + f)
        end
        field(A; halowidths=(1, 1, 1))
    end
    initial = map(f -> copy(f.data), fields)
    reference = map(f -> field(copy(f.data); halowidths=f.halowidths), fields)
    update_halo!(g, reference...)
    HA.ensure_capacity!(g.excache, nfields)
    recvs = MPI.RequestSet(fill(MPI.REQUEST_NULL, 2nfields))
    sends = MPI.RequestSet(fill(MPI.REQUEST_NULL, 2nfields))
    totals = zeros(length(PHASES), 3)
    for grouped in (false, true)
        foreach((f, A) -> copyto!(f.data, A), fields, initial)
        diagnostic_exchange!(g, fields, totals, grouped, recvs, sends)
        correct = all(a.data == b.data for (a, b) in zip(fields, reference))
        MPI.Allreduce(correct, &, topo.comm) || error("halo mismatch")
    end
    out, ref = zeros(shape), zeros(shape)
    public_step!(g, fields, ref, false)
    public_step!(g, fields, out, true)
    correct = out == ref
    MPI.Allreduce(correct, &, topo.comm) || error("stencil mismatch")
    label = "$(MPI.Comm_size(topo.comm)),$(join(shape, 'x')),$(join(topo.dims, 'x')),$nfields,$niter"
    for overlap in (false, true)
        elapsed, bytes = sample!(() -> public_step!(g, fields, out, overlap), g, niter)
        topo.rank == 0 && println("TOTAL,$label,$(overlap ? :overlap : :blocking),$elapsed,$bytes")
    end
    for grouped in (false, true)
        elapsed, bytes = sample!(() -> diagnostic_exchange!(g, fields, nothing,
            grouped, recvs, sends), g, niter)
        topo.rank == 0 && println("EXCHANGE,$label,$(grouped ? :grouped : :ordered),$elapsed,$bytes")
        for _ in 1:10
            diagnostic_exchange!(g, fields, totals, grouped, recvs, sends)
        end
        fill!(totals, 0)
        MPI.Barrier(topo.comm)
        for _ in 1:niter
            diagnostic_exchange!(g, fields, totals, grouped, recvs, sends)
        end
        maxima = MPI.Allreduce(totals, max, topo.comm) ./ niter
        if topo.rank == 0
            for dim in 1:3, (p, phase) in pairs(PHASES)
                println("PHASE,$label,$(grouped ? :grouped : :ordered),$dim,$phase,$(maxima[p, dim])")
            end
        end
    end
    return nothing
end

function main()
    niter = isempty(ARGS) ? 100 : parse(Int, ARGS[1])
    niter > 0 || error("iteration count must be positive")
    shapes = length(ARGS) >= 2 ? (Tuple(parse.(Int, split(ARGS[2], 'x'))),) :
        ((32, 32, 32), (128, 128, 128), (256, 32, 64))
    dims = length(ARGS) >= 3 ? Tuple(parse.(Int, split(ARGS[3], 'x'))) : (0, 0, 0)
    length(dims) == 3 || error("process dimensions must have three entries")
    all(s -> length(s) == 3 && all(>=(5), s), shapes) || error("shapes must be 3D with extents >= 5")
    MPI.Init()
    try
        for shape in shapes, nfields in (1, 2, 4)
            run_case(shape, dims, nfields, niter)
        end
    finally
        MPI.Finalize()
    end
end

main()
