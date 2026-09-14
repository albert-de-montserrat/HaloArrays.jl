# GPU (CUDA) benchmark worker for HaloArrays.jl: a CUDA port of
# HaloArrays.jl/examples/diffusion3D.jl's Cp-weighted 3D heat diffusion
# physics (energy-conservation assertion kept as a correctness gate before
# timing, checked after warmup rather than after the full run so the gate is
# cheap relative to a large `niter`), instrumented for this directory's
# metric set (walltime, GC pressure, device memory footprint, effective
# throughput — see metrics_common.jl). Run under `mpiexec`/`srun` by
# slurm_diffusion3d.sh. Prints exactly one CSV line on rank 0:
#   RESULT,HaloArrays,<nprocs>,<N>,<niter>,<time_s>,<bytes>,<gctime_s>,
#   <pauses>,<full_sweep>,<gpu_bytes>,<t_eff_gbs>,<peak_bw_gbs>,<pct_of_peak>
#
# T_eff: this kernel's ideal minimal fused version reads {T, Cp} and writes
# {T} => n_reads=2, n_writes=1 (see metrics_common.jl for the formula).
#
# Mirrors ImplicitGlobalGrid.jl's own
# examples/diffusion3D_multigpu_CuArrays_novis.jl GPU idiom as closely as the
# two APIs allow — see bench_igg_diffusion3d.jl for the IGG-side twin:
# `using CUDA` before the package, CuArray fields, initial conditions built
# as a host-side comprehension then uploaded in one `CuArray(...)` (scalar-
# indexing a CuArray during setup would otherwise error/be catastrophically
# slow).
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA
import MPI
using HaloArrays
include(joinpath(@__DIR__, "metrics_common.jl"))

@views d_xa(A) = A[2:end, :, :] .- A[1:end-1, :, :]
@views d_xi(A) = A[2:end, 2:end-1, 2:end-1] .- A[1:end-1, 2:end-1, 2:end-1]
@views d_ya(A) = A[:, 2:end, :] .- A[:, 1:end-1, :]
@views d_yi(A) = A[2:end-1, 2:end, 2:end-1] .- A[2:end-1, 1:end-1, 2:end-1]
@views d_za(A) = A[:, :, 2:end] .- A[:, :, 1:end-1]
@views d_zi(A) = A[2:end-1, 2:end-1, 2:end] .- A[2:end-1, 2:end-1, 1:end-1]
@views inn(A) = A[2:end-1, 2:end-1, 2:end-1]

@views function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    warmup = max(5, niter ÷ 10)

    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    lam, cp_min = 1.0, 1.0
    lx, ly, lz  = 10.0, 10.0, 10.0
    nx, ny, nz  = N, N, N
    hw = 1

    t  = topology((0, 0, 0); comm=comm, periods=(true, true, true))
    g  = grid(t, (nx, ny, nz); origin_on_vertex=true)
    Tf = field(CUDA.zeros(Float64, nx, ny, nz); halowidths=(hw, hw, hw))
    T  = Tf.data
    dx = lx / (nx_g(g, Tf) - 1)
    dy = ly / (ny_g(g, Tf) - 1)
    dz = lz / (nz_g(g, Tf) - 1)

    Cp = CuArray([cp_min +
            5 * exp(-(x_g(i, dx, g, Tf) - lx / 1.5)^2 - (y_g(j, dy, g, Tf) - ly / 2)^2 - (z_g(k, dz, g, Tf) - lz / 1.5)^2) +
            5 * exp(-(x_g(i, dx, g, Tf) - lx / 3.0)^2 - (y_g(j, dy, g, Tf) - ly / 2)^2 - (z_g(k, dz, g, Tf) - lz / 1.5)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz])
    T .= CuArray([
            100 * exp(-((x_g(i, dx, g, Tf) - lx / 2) / 2)^2 - ((y_g(j, dy, g, Tf) - ly / 2) / 2)^2 - ((z_g(k, dz, g, Tf) - lz / 3.0) / 2)^2) +
            50 * exp(-((x_g(i, dx, g, Tf) - lx / 2) / 2)^2 - ((y_g(j, dy, g, Tf) - ly / 2) / 2)^2 - ((z_g(k, dz, g, Tf) - lz / 1.5) / 2)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz])
    update_halo!(g, Tf)

    dt = min(dx * dx, dy * dy, dz * dz) * cp_min / lam / 8.1
    owned = (hw+1:nx-hw, hw+1:ny-hw, hw+1:nz-hw)
    total_energy() = MPI.Allreduce(Float64(sum(Cp[owned...] .* T[owned...])), MPI.SUM, comm)
    E0 = total_energy()

    qx = CUDA.zeros(Float64, nx - 1, ny - 2, nz - 2)
    qy = CUDA.zeros(Float64, nx - 2, ny - 1, nz - 2)
    qz = CUDA.zeros(Float64, nx - 2, ny - 2, nz - 1)
    dTedt = CUDA.zeros(Float64, nx - 2, ny - 2, nz - 2)

    function step!()
        qx .= .-lam .* d_xi(T) ./ dx
        qy .= .-lam .* d_yi(T) ./ dy
        qz .= .-lam .* d_zi(T) ./ dz
        dTedt .= (1.0 ./ inn(Cp)) .* (.-d_xa(qx) ./ dx .- d_ya(qy) ./ dy .- d_za(qz) ./ dz)
        T[2:end-1, 2:end-1, 2:end-1] .= inn(T) .+ dt .* dTedt
        update_halo!(g, Tf)
        return nothing
    end

    for _ in 1:warmup
        step!()
    end
    MPI.Barrier(comm)
    CUDA.synchronize()

    E1 = total_energy()
    rel_drift = abs(E1 - E0) / abs(E0)
    rel_drift < 1e-6 || error(
        "diffusion3D (HaloArrays, GPU): energy drifted by a relative $rel_drift after " *
        "$warmup warmup steps (correctness gate failed; not a GPU-precision quirk at this tolerance).")

    GC.gc()
    stats = @timed for _ in 1:niter
        step!()
    end
    CUDA.synchronize()

    gpu_bytes = CUDA.@allocated for _ in 1:niter
        step!()
    end
    CUDA.synchronize()

    numel = nx * ny * nz
    teff = t_eff_gbs(2, 1, numel, sizeof(Float64), stats.time / niter)
    peakbw = rank == 0 ? measure_peak_bandwidth_gbs() : 0.0

    if rank == 0
        print_result("HaloArrays", nprocs, N, niter, stats, gpu_bytes, teff, peakbw)
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
