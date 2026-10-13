# CUDA-aware-MPI benchmark matching the root IGG_hidecomm.jl loop.
import Pkg
Pkg.activate(@__DIR__; io=devnull)
using CUDA
using ParallelStencil
@init_parallel_stencil(CUDA, Float64, 3)
using ImplicitGlobalGrid
import MPI
include(joinpath(@__DIR__, "metrics_common.jl"))

@parallel_indices (i, j, k) function diffusion_step!(Tnext, T, Ci, lam, dt, idx, idy, idz)
    ii, jj, kk = i + 1, j + 1, k + 1
    center = T[ii, jj, kk]
    laplacian = (T[ii - 1, jj, kk] - 2center + T[ii + 1, jj, kk]) * idx^2 +
        (T[ii, jj - 1, kk] - 2center + T[ii, jj + 1, kk]) * idy^2 +
        (T[ii, jj, kk - 1] - 2center + T[ii, jj, kk + 1]) * idz^2
    Tnext[ii, jj, kk] = center + dt * (lam * Ci[ii, jj, kk] * laplacian)
    return nothing
end

function main()
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
    output = length(ARGS) >= 3 ? ARGS[3] : "results_igg_stencil_gpu.jld2"
    N >= 4 || error("local_size must be at least 4; got $N")
    niter > 0 || error("iterations must be positive; got $niter")
    warmup = max(10, niter ÷ 10)

        me, _, _ = init_global_grid(N, N, N; periodx=0, periody=0, periodz=0, quiet=true)
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        T = CUDA.fill(1.7, N, N, N)
        Tnext = similar(T)
        copyto!(Tnext, T)
        Ci = CUDA.fill(0.5, N, N, N)
        dx = 1.0 / (nx_g() - 1)
        dy = 1.0 / (ny_g() - 1)
        dz = 1.0 / (nz_g() - 1)
        idx, idy, idz = 1 / dx, 1 / dy, 1 / dz
        lam = 1.0
        dt = min(dx^2, dy^2, dz^2) / lam / 0.5 / 8.1
        me == 0 && println("grid: local=($N, $N, $N) global=($(nx_g()), $(ny_g()), $(nz_g())) ranks=$nprocs")

        function step!()
            @hide_communication (16, 2, 2) begin
                @parallel (1:N-2, 1:N-2, 1:N-2) diffusion_step!(Tnext, T, Ci, lam, dt, idx, idy, idz)
                update_halo!(Tnext)
            end
            T, Tnext = Tnext, T
            return nothing
        end

        for _ in 1:warmup
            step!()
        end
        CUDA.synchronize()
        max_error = maximum(abs, Array(T) .- 1.7)
        max_error <= 1e-12 || error("constant-field correctness check failed: max error=$max_error")
        MPI.Barrier(comm)
        GC.gc()
        stats = @timed begin
            for _ in 1:niter
                step!()
            end
            CUDA.synchronize()
        end
        stats = merge(stats, (time=MPI.Allreduce(stats.time, MPI.MAX, comm),))
        gpu_bytes = CUDA.@allocated for _ in 1:niter
            step!()
        end
        CUDA.synchronize()

        A_eff = 3 * N^3 * sizeof(Float64) / 1e9
        teff = A_eff / (stats.time / niter)
        peakbw = rank == 0 ? measure_peak_bandwidth_gbs() : 0.0
        rank == 0 && save_result(output, "ImplicitGlobalGridHidecomm", nprocs, N, niter, stats, gpu_bytes, teff, peakbw;
            transport="CUDA-aware MPI requested")
        MPI.Barrier(comm)
        finalize_global_grid()
end

main()
