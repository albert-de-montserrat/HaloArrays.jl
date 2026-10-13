# Per-process "hide communication" benchmark worker for HaloGrids.jl, run
# under `mpiexec` by run_comparison_hide.sh. Prints exactly one CSV line on
# rank 0:
#   RESULT,HaloGrids,<nprocs>,<N>,<niter>,<time_s>,<bytes>,<gctime_s>,<pauses>,<full_sweep>
#
# Workload: one explicit 3D 7-point-Laplacian diffusion step on field A (with
# field B exchanged alongside it purely to match bench_igg_hide.jl's
# communication volume, same as bench_halogrids.jl/bench_igg.jl do for the
# plain-blocking comparison). Unlike the plain comparison, this measures the
# *overlapped* pattern documented in HaloGrids.jl/docs/guide/halo-exchange.md
# and examples/hide_communication.jl: halo_exchange_start! leaves only the
# last dimension (3) outstanding, so every (i,j,k) whose stencil neighbors
# avoid the still-in-flight dimension-3 halo planes (k=1 and k=N) is computed
# while that exchange is in flight; only the two k-planes adjacent to those
# halo planes wait for halo_exchange_finish!.
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)
import MPI
using HaloGrids

@inline function laplacian(A, i, j, k)
    @inbounds A[i-1, j, k] + A[i+1, j, k] + A[i, j-1, k] + A[i, j+1, k] + A[i, j, k-1] + A[i, j, k+1] - 6 * A[i, j, k]
end

function stencil_step!(Anew, A, dt, N, krange)
    @inbounds for k in krange, j in 2:N-1, i in 2:N-1
        Anew[i, j, k] = A[i, j, k] + dt * laplacian(A, i, j, k)
    end
    return nothing
end

# One overlapped diffusion step: safe k-planes computed while dimension 3's
# exchange is outstanding, the two dimension-3-adjacent planes computed after
# finish!.
function step_overlapped!(g, fA, fB, Anew, A, dt, N)
    handle = halo_exchange_start!(g, fA, fB; dims=(1, 2, 3))
    stencil_step!(Anew, A, dt, N, 3:N-2)
    halo_exchange_finish!(handle)
    stencil_step!(Anew, A, dt, N, (2, N - 1))
    return nothing
end

# Reference: the straightforward blocking step (full exchange, then the
# whole interior in one pass) — used once, before timing, to prove the
# overlapped split above computes the same answer.
function step_blocking!(g, fA, fB, Anew, A, dt, N)
    update_halo!(g, fA, fB)
    stencil_step!(Anew, A, dt, N, 2:N-1)
    return nothing
end

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    warmup = max(5, niter ÷ 10)
    dt     = 0.1

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
    Anew = similar(A)

    # Correctness gate, once, before any timing: overlapped split must match
    # the blocking reference exactly.
    Aref, Bref = copy(A), copy(B)
    fAref = field(Aref; halowidths=(1, 1, 1))
    fBref = field(Bref; halowidths=(1, 1, 1))
    Anew_ref = similar(A)
    step_blocking!(g, fAref, fBref, Anew_ref, Aref, dt, N)
    step_overlapped!(g, fA, fB, Anew, A, dt, N)
    @assert Anew[2:end-1, 2:end-1, 2:end-1] == Anew_ref[2:end-1, 2:end-1, 2:end-1]

    for _ in 1:warmup
        step_overlapped!(g, fA, fB, Anew, A, dt, N)
    end
    MPI.Barrier(comm)
    GC.gc()

    stats = @timed for _ in 1:niter
        step_overlapped!(g, fA, fB, Anew, A, dt, N)
    end

    if rank == 0
        gs = stats.gcstats
        println("RESULT,HaloGrids,$nprocs,$N,$niter,$(stats.time),$(stats.bytes),$(stats.gctime),$(gs.pause),$(gs.full_sweep)")
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
