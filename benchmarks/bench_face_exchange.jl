using MPI, HaloArrays
if !isempty(ARGS) && ARGS[1] == "cuda"
    @eval using CUDA
end

function box!(out::Array, A::Array, ir, jr, kr)
    @inbounds for k in kr, j in jr, i in ir
        out[i, j, k] = A[i, j, k] + 0.1 * (
            A[i-1, j, k] + A[i+1, j, k] + A[i, j-1, k] +
            A[i, j+1, k] + A[i, j, k-1] + A[i, j, k+1] - 6A[i, j, k])
    end
    return nothing
end

function box!(out::AbstractArray, A::AbstractArray, ir, jr, kr)
    (isempty(ir) || isempty(jr) || isempty(kr)) && return nothing
    im, ip = first(ir)-1:last(ir)-1, first(ir)+1:last(ir)+1
    jm, jp = first(jr)-1:last(jr)-1, first(jr)+1:last(jr)+1
    km, kp = first(kr)-1:last(kr)-1, first(kr)+1:last(kr)+1
    @views @. out[ir, jr, kr] = A[ir, jr, kr] + 0.1 * (
        A[im, jr, kr] + A[ip, jr, kr] + A[ir, jm, kr] +
        A[ir, jp, kr] + A[ir, jr, km] + A[ir, jr, kp] - 6 * A[ir, jr, kr])
    return nothing
end

function step!(g, fields, out, ::Val{R}, ::Val{false}) where {R}
    update_halo!(g, fields...; region=R)
    nx, ny, nz = size(fields[1].data)
    box!(out, fields[1].data, 2:nx-1, 2:ny-1, 2:nz-1)
    return nothing
end

function step!(g, fields, out, ::Val{:full}, ::Val{true})
    A = fields[1].data
    nx, ny, nz = size(A)
    h = halo_exchange_start!(g, fields...)
    try
        box!(out, A, 2:nx-1, 2:ny-1, 3:nz-2)
    finally
        halo_exchange_finish!(h)
    end
    box!(out, A, 2:nx-1, 2:ny-1, 2:2)
    box!(out, A, 2:nx-1, 2:ny-1, nz-1:nz-1)
    return nothing
end

function step!(g, fields, out, ::Val{:faces}, ::Val{true})
    A = fields[1].data
    nx, ny, nz = size(A)
    h = halo_exchange_start!(g, fields...; region=:faces)
    try
        box!(out, A, 3:nx-2, 3:ny-2, 3:nz-2)
    finally
        halo_exchange_finish!(h)
    end
    box!(out, A, 2:2, 2:ny-1, 2:nz-1)
    box!(out, A, nx-1:nx-1, 2:ny-1, 2:nz-1)
    box!(out, A, 3:nx-2, 2:2, 2:nz-1)
    box!(out, A, 3:nx-2, ny-1:ny-1, 2:nz-1)
    box!(out, A, 3:nx-2, 3:ny-2, 2:2)
    box!(out, A, 3:nx-2, 3:ny-2, nz-1:nz-1)
    return nothing
end

function measure!(work!, comm, synchronize, niter)
    for _ in 1:10
        work!()
    end
    synchronize()
    GC.gc()
    MPI.Barrier(comm)
    stats = @timed begin
        for _ in 1:niter
            work!()
        end
        synchronize()
    end
    return MPI.Allreduce(stats.time, max, comm) / niter,
        MPI.Allreduce(stats.bytes, max, comm) / niter
end

function measure_mode!(g, fields, out, region::Val{R}, overlap::Val{O}, synchronize, niter) where {R,O}
    step_time, step_bytes = measure!(() -> step!(g, fields, out, region, overlap),
        g.topology.comm, synchronize, niter)
    exchange_time, exchange_bytes = measure!(() -> update_halo!(g, fields...; region=R),
        g.topology.comm, synchronize, niter)
    return step_time, step_bytes, exchange_time, exchange_bytes
end

function buffer_bytes(cache)
    return sum(pair -> sum(length, pair), cache.sendbufs; init=0) +
        sum(pair -> sum(length, pair), cache.recvbufs; init=0)
end

function report_memory(g, label, region)
    cache = g.excache
    host = MPI.Allreduce(buffer_bytes(cache), max, g.topology.comm)
    device_cache = cache.gpu_cache[]
    device = MPI.Allreduce(device_cache === nothing ? 0 : buffer_bytes(device_cache), max, g.topology.comm)
    requests = length(cache.reqs)
    remote_sends = sum(pair -> count(r -> has_neighbor(r) && r != g.topology.rank, pair),
        g.topology.neighbors) * size(cache.reqs, 1) ÷ (region == :faces ? ndims(g) : 1)
    g.topology.rank == 0 && println("MEMORY,$label,$region,$host,$device,$requests,$remote_sends")
    return nothing
end

function main()
    backend = isempty(ARGS) ? "cpu" : ARGS[1]
    to_backend, synchronize = identity, () -> nothing
    if backend == "cuda"
        if !CUDA.functional()
            println("SKIP,CUDA not functional")
            return
        end
        CUDA.allowscalar(false)
        to_backend, synchronize = CUDA.CuArray, CUDA.synchronize
    elseif backend != "cpu"
        error("backend must be cpu or cuda")
    end
    niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
    shape = length(ARGS) >= 3 ? Tuple(parse.(Int, split(ARGS[3], 'x'))) : (128, 32, 64)
    dims = length(ARGS) >= 4 ? Tuple(parse.(Int, split(ARGS[4], 'x'))) : (0, 0, 0)
    repeats = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
    niter > 0 && repeats > 0 || error("iterations and repeats must be positive")
    length(shape) == length(dims) == 3 && all(>=(5), shape) || error("invalid shape/process dimensions")
    MPI.Init()
    try
        topo = topology(dims; periods=(true, true, true))
        for nfields in (1, 2, 4)
            g = grid(topo, shape)
            gfaces = grid(topo, shape)
            fields = ntuple(i -> field(to_backend(reshape(
                [sin(j + 13topo.rank + i) for j in 1:prod(shape)], shape))), nfields)
            initial = map(f -> Array(f.data), fields)
            out, reference = to_backend(zeros(shape)), to_backend(zeros(shape))
            step!(g, fields, reference, Val(:full), Val(false))
            modes = ((Val(:full), Val(false)), (Val(:faces), Val(false)),
                (Val(:full), Val(true)), (Val(:faces), Val(true)))
            for (region, overlap) in modes
                foreach((f, A) -> copyto!(f.data, A), fields, initial)
                step!(region isa Val{:faces} ? gfaces : g, fields, out, region, overlap)
                synchronize()
                MPI.Allreduce(Array(out) == Array(reference), &, topo.comm) || error("stencil mismatch")
            end
            label = "$backend,$(MPI.Comm_size(topo.comm)),$(join(shape, 'x')),$(join(topo.dims, 'x')),$nfields,$niter"
            for rep in 1:repeats
                for (region, overlap) in (isodd(rep) ? modes : reverse(modes))
                    measurements = measure_mode!(region isa Val{:faces} ? gfaces : g,
                        fields, out, region, overlap, synchronize, niter)
                    if topo.rank == 0
                        println("FACES,$label,$rep,$(typeof(region).parameters[1]),$(typeof(overlap).parameters[1]),$(join(measurements, ','))")
                    end
                end
            end
            report_memory(g, label, :full)
            report_memory(gfaces, label, :faces)
        end
    finally
        MPI.Finalize()
    end
end

main()
