# GPU (CUDA) benchmark worker for ImplicitGlobalGrid.jl: same Cp-weighted 3D
# heat diffusion physics as bench_haloarrays_diffusion3d.jl (its twin, see
# its header comment for the full rationale and T_eff read/write count),
# mirroring ImplicitGlobalGrid.jl's own
# examples/diffusion3D_multigpu_CuArrays_novis.jl GPU idiom directly. Prints
# one CSV line on rank 0, same schema as the HaloArrays worker.
#
# No overlap variant here: IGG's `update_halo!` has no start!/finish! split
# (see bench_igg_hide.jl for the same point on the hide-communication side) —
# not applicable to this plain-blocking comparison anyway.
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA                # before ImplicitGlobalGrid, to activate its CUDA support (its own convention)
using ImplicitGlobalGrid
import MPI
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

    lam, cp_min = 1.0, 1.0
    lx, ly, lz  = 10.0, 10.0, 10.0
    nx, ny, nz  = N, N, N
    hw = 1

    me, _, nprocs = init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, quiet=true)
    dx = lx / (nx_g() - 1)
    dy = ly / (ny_g() - 1)
    dz = lz / (nz_g() - 1)

    T  = CUDA.zeros(Float64, nx, ny, nz)
    Cp = CuArray([cp_min +
            5 * exp(-(x_g(i, dx, T) - lx / 1.5)^2 - (y_g(j, dy, T) - ly / 2)^2 - (z_g(k, dz, T) - lz / 1.5)^2) +
            5 * exp(-(x_g(i, dx, T) - lx / 3.0)^2 - (y_g(j, dy, T) - ly / 2)^2 - (z_g(k, dz, T) - lz / 1.5)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz])
    T .= CuArray([
            100 * exp(-((x_g(i, dx, T) - lx / 2) / 2)^2 - ((y_g(j, dy, T) - ly / 2) / 2)^2 - ((z_g(k, dz, T) - lz / 3.0) / 2)^2) +
            50 * exp(-((x_g(i, dx, T) - lx / 2) / 2)^2 - ((y_g(j, dy, T) - ly / 2) / 2)^2 - ((z_g(k, dz, T) - lz / 1.5) / 2)^2)
        for i in 1:nx, j in 1:ny, k in 1:nz])
    update_halo!(T)

    dt = min(dx * dx, dy * dy, dz * dz) * cp_min / lam / 8.1
    owned = (hw+1:nx-hw, hw+1:ny-hw, hw+1:nz-hw)
    total_energy() = MPI.Allreduce(Float64(sum(Cp[owned...] .* T[owned...])), MPI.SUM, MPI.COMM_WORLD)
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
        update_halo!(T)
        return nothing
    end

    for _ in 1:warmup
        step!()
    end
    MPI.Barrier(MPI.COMM_WORLD)
    CUDA.synchronize()

    E1 = total_energy()
    rel_drift = abs(E1 - E0) / abs(E0)
    rel_drift < 1e-6 || error(
        "diffusion3D (ImplicitGlobalGrid, GPU): energy drifted by a relative $rel_drift after " *
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
    peakbw = me == 0 ? measure_peak_bandwidth_gbs() : 0.0

    if me == 0
        print_result("ImplicitGlobalGrid", nprocs, N, niter, stats, gpu_bytes, teff, peakbw)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    finalize_global_grid()
end

main()
