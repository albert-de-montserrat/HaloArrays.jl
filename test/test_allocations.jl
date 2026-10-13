# Allocation-regression test for the exchange hot path (phase 8: performance,
# memory footprint, and GC-pressure hardening — see REDESIGN_PLAN.md).
#
# The steady-state (post-warmup) cost of `update_halo!`/`halo_exchange_start!`/
# `halo_exchange_finish!` should be allocation-free on the single-rank
# self-periodic path, which isolates this package's own code from whatever
# MPI.jl's `Irecv!`/`Isend`/`Wait` themselves allocate for real network
# transport at nprocs > 1 (measured separately, and outside this package's
# control — see REDESIGN_PLAN.md for the numbers). Measurements must run
# inside a function, never at top-level/global scope: global variables are
# dynamically typed from the compiler's perspective, and `@allocated` at
# top-level would measure that dispatch overhead instead of the package's
# real steady-state cost.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

function run_allocation_tests()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    t = topology(ntuple(_ -> 0, 2); comm=comm, periods=(true, true))
    g = grid(t, (20, 20))
    A = fill(Float64(rank), 20, 20)
    B = fill(Float32(rank), 20, 20)
    fA = field(A; halowidths=(1, 1))
    fB = field(B; halowidths=(2, 1))

    # Warmup: let buffers grow to their high-water mark and every method
    # specialization JIT-compile before measuring.
    for _ in 1:5
        update_halo!(g, fA, fB)
    end

    niter = 200
    total_update = @allocated for _ in 1:niter
        update_halo!(g, fA, fB)
    end
    bytes_per_update = total_update / niter

    total_pair = @allocated for _ in 1:niter
        h = halo_exchange_start!(g, fA, fB)
        halo_exchange_finish!(h)
    end
    bytes_per_pair = total_pair / niter

    for _ in 1:5
        update_halo!(g, fA, fB; region=:faces)
    end
    total_faces = @allocated for _ in 1:niter
        update_halo!(g, fA, fB; region=:faces)
    end
    total_face_pair = @allocated for _ in 1:niter
        h = halo_exchange_start!(g, fA, fB; region=:faces)
        halo_exchange_finish!(h)
    end
    bytes_per_faces = total_faces / niter
    bytes_per_face_pair = total_face_pair / niter

    if rank == 0
        println("nprocs=$nprocs  update_halo!: $bytes_per_update B/call  start!+finish!: $bytes_per_pair B/call")
        println("nprocs=$nprocs  faces update!: $bytes_per_faces B/call  faces start!+finish!: $bytes_per_face_pair B/call")
    end

    @testset "steady-state allocation (nprocs=$nprocs)" begin
        if nprocs == 1
            # Purely self-periodic (no real MPI transport): this package's
            # own code should be exactly allocation-free.
            @test bytes_per_update == 0
            @test bytes_per_pair == 0
            @test bytes_per_faces == 0
            @test bytes_per_face_pair == 0
        else
            # Real Irecv!/Isend/Wait transport is in play; MPI.jl's own
            # per-call overhead isn't this package's to eliminate. Bound
            # loosely enough to never be flaky against that, tightly enough
            # to catch a regression back to the ~4-6KB/call this measured
            # before the phase 8 fixes (see REDESIGN_PLAN.md).
            @test bytes_per_update < 2000
            @test bytes_per_pair < 2000
            @test bytes_per_faces < 2000
            @test bytes_per_face_pair < 2000
        end
    end

    MPI.Barrier(comm)
    rank == 0 && println("test_allocations.jl: all tests passed on $nprocs ranks.")
    MPI.Finalize()
end

run_allocation_tests()
