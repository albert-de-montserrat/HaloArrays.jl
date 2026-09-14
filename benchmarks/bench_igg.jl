# Per-process benchmark worker for ImplicitGlobalGrid.jl, run under `mpiexec`
# by run_comparison.jl. Prints exactly one CSV line on rank 0:
#   RESULT,ImplicitGlobalGrid,<nprocs>,<N>,<niter>,<time_s>,<bytes>,<gctime_s>,<pauses>,<full_sweep>
#
# Mirrors bench_haloarrays.jl as closely as the two APIs allow: same global
# domain size (N^3 local per rank, same Dims_create decomposition for the
# same nprocs/comm), same 2x Float64 arrays, same halo width (overlaps=(2,2,2)
# => halowidths=(1,1,1), identical to HaloArrays' fixed halowidths=(1,1,1)),
# fully periodic in every dimension.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "ImplicitGlobalGrid.jl"); io=devnull)
import MPI
using ImplicitGlobalGrid

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    warmup = max(5, niter ÷ 10)

    me, _, nprocs = init_global_grid(N, N, N;
        periodx=1, periody=1, periodz=1,
        overlaps=(2, 2, 2), quiet=true)

    A = fill(Float64(me), N, N, N)
    B = fill(Float64(me) + 0.5, N, N, N)

    for _ in 1:warmup
        update_halo!(A, B)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    GC.gc()

    stats = @timed for _ in 1:niter
        update_halo!(A, B)
    end

    if me == 0
        gs = stats.gcstats
        println("RESULT,ImplicitGlobalGrid,$nprocs,$N,$niter,$(stats.time),$(stats.bytes),$(stats.gctime),$(gs.pause),$(gs.full_sweep)")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    finalize_global_grid()
end

main()
