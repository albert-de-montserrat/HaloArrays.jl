# GPU (CUDA) benchmark worker for ImplicitGlobalGrid.jl's "hide
# communication" comparison point — same workload and kernel as
# bench_haloarrays_hide.jl (its twin, see its header comment for the full
# rationale, kernel, and T_eff read/write count).
#
# No overlapped variant here: IGG's `update_halo!` performs its own internal
# non-blocking pipelining but exposes no start!/finish! split, so there is no
# hook to run independent compute while any part of the exchange is still in
# flight (this is the exact limitation HaloArrays.jl's split-phase API was
# designed to remove — see "Context" in HaloArrays.jl/REDESIGN_PLAN.md, and
# the CPU benchmarks/bench_igg_hide.jl's identical note). The only available
# pattern is the fully-serial one benchmarked below.
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA
using ImplicitGlobalGrid
import MPI
include(joinpath(@__DIR__, "metrics_common.jl"))

function laplacian_kernel!(Anew, A, dt, N, kstart, kstop)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + kstart - 1
    if i <= N - 1 && j <= N - 1 && k <= kstop
        @inbounds Anew[i, j, k] = A[i, j, k] + dt * (
            A[i-1, j, k] + A[i+1, j, k] + A[i, j-1, k] + A[i, j+1, k] + A[i, j, k-1] + A[i, j, k+1] - 6 * A[i, j, k])
    end
    return nothing
end

function launch_stencil!(Anew, A, dt, N, kstart, kstop)
    nk = kstop - kstart + 1
    nk <= 0 && return nothing
    threads = (8, 8, min(4, nk))
    blocks = (cld(N - 2, threads[1]), cld(N - 2, threads[2]), cld(nk, threads[3]))
    @cuda threads = threads blocks = blocks laplacian_kernel!(Anew, A, dt, N, kstart, kstop)
    return nothing
end

function step_serial!(Anew, A, B, dt, N)
    update_halo!(A, B)
    launch_stencil!(Anew, A, dt, N, 2, N - 1)
    return nothing
end

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    warmup = max(5, niter ÷ 10)
    dt     = 0.1

    me, _, nprocs = init_global_grid(N, N, N; periodx=1, periody=1, periodz=1, quiet=true)

    A = CUDA.fill(Float64(me), N, N, N)
    B = CUDA.fill(Float64(me) + 0.5, N, N, N)
    Anew = similar(A)

    for _ in 1:warmup
        step_serial!(Anew, A, B, dt, N)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    CUDA.synchronize()
    GC.gc()

    stats = @timed for _ in 1:niter
        step_serial!(Anew, A, B, dt, N)
    end
    CUDA.synchronize()

    gpu_bytes = CUDA.@allocated for _ in 1:niter
        step_serial!(Anew, A, B, dt, N)
    end
    CUDA.synchronize()

    numel = N * N * N
    teff = t_eff_gbs(1, 1, numel, sizeof(Float64), stats.time / niter)
    peakbw = me == 0 ? measure_peak_bandwidth_gbs() : 0.0

    if me == 0
        print_result("ImplicitGlobalGrid", nprocs, N, niter, stats, gpu_bytes, teff, peakbw)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    finalize_global_grid()
end

main()
