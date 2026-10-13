# Per-process benchmark worker for HaloGrids.jl, run under `mpiexec` by
# run_comparison.jl. Prints exactly one CSV line on rank 0:
#   RESULT,HaloGrids,<nprocs>,<N>,<niter>,<time_s>,<bytes>,<gctime_s>,<pauses>,<full_sweep>
#
# Mirrors bench_igg.jl as closely as the two APIs allow: same global domain
# size and decomposition (MPI.Dims_create picks the same split for the same
# nprocs/comm), same 2x Float64 fields, same halo width (1), fully periodic
# in every dimension so nprocs=1 is a pure self-periodic (no real network
# transport) measurement exactly like test_allocations.jl's methodology in
# HaloGrids.jl/REDESIGN_PLAN.md.
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
import MPI
using HaloGrids

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    warmup = max(5, niter ÷ 10)
    dims_order = length(ARGS) >= 3 ? Tuple(parse.(Int, split(ARGS[3], ","))) : (1, 2, 3)

    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    t  = topology((0, 0, 0); comm=comm, periods=(true, true, true))
    g  = grid(t, (N, N, N))
    A  = fill(Float64(rank), N, N, N)
    B  = fill(Float64(rank) + 0.5, N, N, N)
    fA = field(A; halowidths=(1, 1, 1))
    fB = field(B; halowidths=(1, 1, 1))

    for _ in 1:warmup
        update_halo!(g, fA, fB; dims=dims_order)
    end
    MPI.Barrier(comm)
    GC.gc()

    stats = @timed for _ in 1:niter
        update_halo!(g, fA, fB; dims=dims_order)
    end

    if rank == 0
        gs = stats.gcstats
        println("RESULT,HaloGrids,$nprocs,$N,$niter,$(stats.time),$(stats.bytes),$(stats.gctime),$(gs.pause),$(gs.full_sweep)")
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
