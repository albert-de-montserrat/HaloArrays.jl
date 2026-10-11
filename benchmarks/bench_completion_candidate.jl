using MPI, HaloArrays, Test
if !isempty(ARGS) && ARGS[1] == "cuda"
    @eval using CUDA
end
include("exchange_completion_candidate.jl")
const ECC = ExchangeCompletionCandidate

function make_fields(topo, shape, nfields, to_backend; variable_widths=false)
    return ntuple(nfields) do i
        T = isodd(i) ? Float64 : Float32
        A = reshape(T[T(1_000_000topo.rank + j + 10i) for j in 1:prod(shape)], shape)
        widths = ntuple(d -> variable_widths && iseven(i) && d == 1 ? 2 : 1, length(shape))
        field(to_backend(A); halowidths=widths)
    end
end

function check_case(shape, periods, dims, to_backend, synchronize; threaded=false)
    topo = topology(ntuple(_ -> 0, length(shape)); periods)
    g = grid(topo, shape; threaded_pack=threaded)
    fields = make_fields(topo, shape, 2, to_backend; variable_widths=!threaded)
    original = map(f -> Array(f.data), fields)
    reference = map(f -> field(copy(f.data); halowidths=f.halowidths), fields)
    update_halo!(g, reference...; dims)
    state = ECC.CompletionCache(g, length(fields))
    for split in (false, true)
        foreach((f, A) -> copyto!(f.data, A), fields, original)
        if split
            h = ECC.start!(state, fields, dims)
            @test_throws ErrorException ECC.start!(state, fields, dims)
            ECC.finish!(state, h)
        else
            ECC.update!(state, fields, dims)
        end
        synchronize()
        @test all(Array(a.data) == Array(b.data) for (a, b) in zip(fields, reference))
        @test !g.excache.pending
    end
    # Smaller subsequent calls exercise buffer reuse and stale request slots.
    short = (fields[1],)
    small_state = ECC.CompletionCache(g, 1)
    ECC.update!(small_state, short, dims)
    synchronize()
    @test Array(short[1].data) == Array(reference[1].data)
    return nothing
end

function check_suite(to_backend, synchronize)
    @testset "grouped completion candidate" begin
        for N in (2, 3), periodic in (false, true)
            shape = ntuple(_ -> 8, N)
            periods = ntuple(_ -> periodic, N)
            for dims in (ntuple(identity, N), Tuple(reverse(1:N)), (1,))
                check_case(shape, periods, dims, to_backend, synchronize)
            end
        end
        topo1 = topology((0, 0); periods=(true, true))
        topo2 = topology((0, 0); periods=(true, true))
        g1, g2 = grid(topo1, (8, 9)), grid(topo2, (11, 12))
        f1 = make_fields(topo1, g1.nlocal, 2, to_backend)
        f2 = make_fields(topo2, g2.nlocal, 1, to_backend)
        r1 = map(f -> field(copy(f.data)), f1)
        r2 = map(f -> field(copy(f.data)), f2)
        update_halo!(g1, r1...)
        update_halo!(g2, r2...)
        s1, s2 = ECC.CompletionCache(g1, 2), ECC.CompletionCache(g2, 1)
        h1, h2 = ECC.start!(s1, f1, (1, 2)), ECC.start!(s2, f2, (1, 2))
        ECC.finish!(s2, h2)
        ECC.finish!(s1, h1)
        synchronize()
        @test all(Array(a.data) == Array(b.data) for (a, b) in zip(f1, r1))
        @test Array(f2[1].data) == Array(r2[1].data)
        if Threads.nthreads() > 1 && to_backend === identity
            check_case((4, 256, 256), (true, true, true), (1, 2, 3),
                to_backend, synchronize; threaded=true)
        end
    end
    return nothing
end

function measure(work!, comm, synchronize, niter)
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

function benchmark(shape, dims, niter, repeats, backend, to_backend, synchronize)
    topo = topology(dims; periods=(true, true, true))
    g = grid(topo, shape)
    order = (1, 2, 3)
    for nfields in (1, 2, 4)
        fields = make_fields(topo, shape, nfields, to_backend)
        state = ECC.CompletionCache(g, nfields)
        label = "$backend,$(MPI.Comm_size(topo.comm)),$(join(shape, 'x')),$(join(topo.dims, 'x')),$nfields,$niter"
        for rep in 1:repeats
            # Alternate order to reduce systematic first/second bias.
            modes = isodd(rep) ? (:ordered, :grouped) : (:grouped, :ordered)
            for mode in modes
                work! = mode == :ordered ? (() -> halo_exchange_finish!(HaloArrays._halo_exchange_start!(g, fields, order))) :
                    (() -> ECC.update!(state, fields, order))
                elapsed, bytes = measure(work!, topo.comm, synchronize, niter)
                topo.rank == 0 && println("COMPLETION,$label,$rep,$mode,$elapsed,$bytes")
            end
        end
    end
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
    niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
    shape = length(ARGS) >= 3 ? Tuple(parse.(Int, split(ARGS[3], 'x'))) : (128, 32, 64)
    dims = length(ARGS) >= 4 ? Tuple(parse.(Int, split(ARGS[4], 'x'))) : (0, 0, 0)
    repeats = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
    niter >= 0 && repeats > 0 || error("invalid iteration/repeat count")
    length(shape) == length(dims) == 3 && all(>=(5), shape) || error("invalid shape/process dimensions")
    MPI.Init()
    try
        check_suite(to_backend, synchronize)
        niter > 0 && benchmark(shape, dims, niter, repeats, backend, to_backend, synchronize)
    finally
        MPI.Finalize()
    end
end

main()
