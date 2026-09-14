# Per-process "hide communication" benchmark worker for ImplicitGlobalGrid.jl,
# run under `mpiexec` by run_comparison_hide.sh. Prints exactly one CSV line
# on rank 0:
#   RESULT,ImplicitGlobalGrid,<nprocs>,<N>,<niter>,<time_s>,<bytes>,<gctime_s>,<pauses>,<full_sweep>
#
# Same workload as bench_haloarrays_hide.jl (one 3D 7-point-Laplacian
# diffusion step on A, with B exchanged alongside it for equal communication
# volume) — but there is no overlapped variant to measure here: IGG's
# `update_halo!` performs its own internal non-blocking pipelining but does
# not expose a start!/finish! split, so a caller has no hook to run
# independent compute while any part of the exchange is still in flight (this
# was the exact limitation HaloArrays.jl's split-phase API was designed to
# remove — see "Context" in HaloArrays.jl/REDESIGN_PLAN.md). The only
# available pattern is therefore the fully-serial one benchmarked below:
# block on the whole exchange, then compute the whole interior.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "ImplicitGlobalGrid.jl"); io=devnull)
import MPI
using ImplicitGlobalGrid

@inline function laplacian(A, i, j, k)
    @inbounds A[i-1, j, k] + A[i+1, j, k] + A[i, j-1, k] + A[i, j+1, k] + A[i, j, k-1] + A[i, j, k+1] - 6 * A[i, j, k]
end

function stencil_step!(Anew, A, dt, N, krange)
    @inbounds for k in krange, j in 2:N-1, i in 2:N-1
        Anew[i, j, k] = A[i, j, k] + dt * laplacian(A, i, j, k)
    end
    return nothing
end

function step_serial!(Anew, A, B, dt, N)
    update_halo!(A, B)
    stencil_step!(Anew, A, dt, N, 2:N-1)
    return nothing
end

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    warmup = max(5, niter ÷ 10)
    dt     = 0.1

    me, _, nprocs = init_global_grid(N, N, N;
        periodx=1, periody=1, periodz=1,
        overlaps=(2, 2, 2), quiet=true)

    A = fill(Float64(me), N, N, N)
    B = fill(Float64(me) + 0.5, N, N, N)
    Anew = similar(A)

    for _ in 1:warmup
        step_serial!(Anew, A, B, dt, N)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    GC.gc()

    stats = @timed for _ in 1:niter
        step_serial!(Anew, A, B, dt, N)
    end

    if me == 0
        gs = stats.gcstats
        println("RESULT,ImplicitGlobalGrid,$nprocs,$N,$niter,$(stats.time),$(stats.bytes),$(stats.gctime),$(gs.pause),$(gs.full_sweep)")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    finalize_global_grid()
end

main()
