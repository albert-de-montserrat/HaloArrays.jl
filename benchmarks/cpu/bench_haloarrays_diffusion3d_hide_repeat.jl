# Repeated-trial sibling of bench_haloarrays_diffusion3d_hide.jl (left
# completely untouched — see this file's header there for the physics/
# instrumentation/split-phase-exchange rationale). Same change as
# bench_haloarrays_diffusion3d_repeat.jl relative to its own non-repeat
# original: `ntrials` independent `@timed`/`@allocated` passes instead of
# one, each saved as its own row via save_trial_result
# (metrics_repeat_common.jl) — see that file's sibling header comment for why
# (telling a real HaloArrays-vs-ImplicitGlobalGrid gap apart from run-to-run
# noise). Run under `mpiexec`/`srun`, with `julia -tN` selecting the
# ParallelStencil thread count.
#
# T_eff: this kernel's ideal minimal fused version reads {T, Cp} and writes
# {T} => n_reads=2, n_writes=1 (see metrics_common.jl for the formula).
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)
import MPI
using HaloArrays
include(joinpath(@__DIR__, "metrics_repeat_common.jl"))

@parallel_indices (i, j, k) function flux_x!(qx, T, lam, dx)
    qx[i, j, k] = -lam * (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) / dx
    return nothing
end

@parallel_indices (i, j, k) function flux_y!(qy, T, lam, dy)
    qy[i, j, k] = -lam * (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) / dy
    return nothing
end

@parallel_indices (i, j, k) function flux_z!(qz, T, lam, dz)
    qz[i, j, k] = -lam * (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) / dz
    return nothing
end

@parallel_indices (i, j, k) function update_temperature!(T, Cp, qx, qy, qz, dt, dx, dy, dz)
    dTdt = (-(qx[i + 1, j, k] - qx[i, j, k]) / dx -
        (qy[i, j + 1, k] - qy[i, j, k]) / dy -
        (qz[i, j, k + 1] - qz[i, j, k]) / dz) / Cp[i + 1, j + 1, k + 1]
    T[i + 1, j + 1, k + 1] += dt * dTdt
    return nothing
end

@parallel_indices (i, j, k) function noflux_xlo!(T)
    T[i, j, k] = T[i + 1, j, k]
    return nothing
end

@parallel_indices (i, j, k) function noflux_xhi!(T)
    T[i, j, k] = T[i - 1, j, k]
    return nothing
end

@parallel_indices (i, j, k) function noflux_ylo!(T)
    T[i, j, k] = T[i, j + 1, k]
    return nothing
end

@parallel_indices (i, j, k) function noflux_yhi!(T)
    T[i, j, k] = T[i, j - 1, k]
    return nothing
end

function apply_vertical_noflux!(T, nx, ny, nz)
    @parallel (1:1, 1:ny, 1:nz) noflux_xlo!(T)
    @parallel (nx:nx, 1:ny, 1:nz) noflux_xhi!(T)
    @parallel (1:nx, 1:1, 1:nz) noflux_ylo!(T)
    @parallel (1:nx, ny:ny, 1:nz) noflux_yhi!(T)
    return nothing
end

@views function main()
    N       = parse(Int, ARGS[1])
    niter   = parse(Int, ARGS[2])
    output  = ARGS[3]
    ntrials = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
    warmup  = max(5, niter ÷ 10)

    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    let (base, ext) = splitext(output)
        output = "$(base)_n$(nprocs)$(ext)"
    end

    lam, cp_min = 1.0, 1.0
    lx, ly, lz  = 10.0, 10.0, 10.0
    nx, ny, nz  = N, N, N
    hw = 1

    t  = topology((0, 0, 0); comm=comm, periods=(false, false, true))
    g  = grid(t, (nx, ny, nz); origin_on_vertex=true)
    Tf = field(zeros(Float64, nx, ny, nz); halowidths=(hw, hw, hw))
    T  = Tf.data
    rank == 0 && println(
        "grid: local=($nx, $ny, $nz) global=($(nx_g(g, Tf)), $(ny_g(g, Tf)), $(nz_g(g, Tf))) nprocs=$nprocs ntrials=$ntrials")
    dx = lx / (nx_g(g, Tf) - 1)
    dy = ly / (ny_g(g, Tf) - 1)
    dz = lz / (nz_g(g, Tf) - 1)

    Cp = [cp_min +
            5 * exp(-(x_g(i, dx, g, Tf) - lx / 1.5)^2 - (y_g(j, dy, g, Tf) - ly / 2)^2 - (z_g(k, dz, g, Tf) - lz / 1.5)^2) +
            5 * exp(-(x_g(i, dx, g, Tf) - lx / 3.0)^2 - (y_g(j, dy, g, Tf) - ly / 2)^2 - (z_g(k, dz, g, Tf) - lz / 1.5)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz]
    T .= [
            100 * exp(-((x_g(i, dx, g, Tf) - lx / 2) / 2)^2 - ((y_g(j, dy, g, Tf) - ly / 2) / 2)^2 - ((z_g(k, dz, g, Tf) - lz / 3.0) / 2)^2) +
            50 * exp(-((x_g(i, dx, g, Tf) - lx / 2) / 2)^2 - ((y_g(j, dy, g, Tf) - ly / 2) / 2)^2 - ((z_g(k, dz, g, Tf) - lz / 1.5) / 2)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz]
    apply_vertical_noflux!(T, nx, ny, nz)
    update_halo!(g, Tf)

    dt = min(dx * dx, dy * dy, dz * dz) * cp_min / lam / 8.1
    owned = (hw+1:nx-hw, hw+1:ny-hw, hw+1:nz-hw)
    total_energy() = global_sum(g, Float64(sum(Cp[owned...] .* T[owned...])))
    E0 = total_energy()

    qx = zeros(Float64, nx - 1, ny - 2, nz - 2)
    qy = zeros(Float64, nx - 2, ny - 1, nz - 2)
    qz = zeros(Float64, nx - 2, ny - 2, nz - 1)
    function step!()
        @parallel (1:nx-1, 1:ny-2, 1:nz-2) flux_x!(qx, T, lam, dx)
        @parallel (1:nx-2, 1:ny-1, 1:nz-2) flux_y!(qy, T, lam, dy)
        @parallel (1:nx-2, 1:ny-2, 1:nz-1) flux_z!(qz, T, lam, dz)

        @hide_halo_communication g Tf boundary_width=(16, 2, 2) begin
            @parallel (1:nx-2, 1:ny-2, 1:nz-2) update_temperature!(
                T, Cp, qx, qy, qz, dt, dx, dy, dz)
            @parallel (1:1, 1:ny, 1:nz) noflux_xlo!(T)
            @parallel (nx:nx, 1:ny, 1:nz) noflux_xhi!(T)
            @parallel (1:nx, 1:1, 1:nz) noflux_ylo!(T)
            @parallel (1:nx, ny:ny, 1:nz) noflux_yhi!(T)
        end
        return nothing
    end

    for _ in 1:warmup
        step!()
    end
    MPI.Barrier(comm)

    E1 = total_energy()
    rel_drift = abs(E1 - E0) / abs(E0)
    rel_drift < 1e-6 || error(
        "diffusion3D (HaloArrays, CPU): energy drifted by a relative $rel_drift after " *
        "$warmup warmup steps (correctness gate failed).")

    numel = nx * ny * nz
    peakbw = rank == 0 ? measure_peak_bandwidth_gbs() : 0.0  # once: a hardware property, not per-trial noise worth re-measuring

    for trial in 1:ntrials
        GC.gc()
        stats = @timed for _ in 1:niter
            step!()
        end

        gpu_bytes = @allocated for _ in 1:niter
            step!()
        end

        teff = t_eff_gbs(2, 1, numel, sizeof(Float64), stats.time / niter)
        if rank == 0
            save_trial_result(output, "HaloArraysHide", nprocs, N, niter, trial, stats, gpu_bytes, teff, peakbw)
        end
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
