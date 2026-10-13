# 3D heat diffusion on a periodic domain, decomposed across MPI ranks with
# HaloGrids.jl — a direct port of ImplicitGlobalGrid.jl's
# examples/diffusion3D_multicpu_novis.jl onto the new API, doubling as a
# worked example of the per-field coordinate helpers (nx_g/x_g/... in
# src/coordinates.jl): initial conditions are built from *physical*
# coordinates (two Gaussian anomalies each in heat capacity and temperature),
# not local array indices, so they must be identical however the domain is
# decomposed across ranks.
#
# The domain is fully periodic in all three dimensions, which makes total
# thermal energy (sum of Cp*T over the domain) an exactly conserved quantity
# of the discrete scheme: every internal flux face is shared by exactly two
# neighboring cells with opposite sign, so the update telescopes to zero when
# summed over a closed domain, regardless of how that domain is split across
# ranks — *provided* the halo exchange correctly delivers each neighbor's
# data. This example checks that conservation directly (`@assert rel_drift <
# ...` below), which makes it a real correctness check, not just a
# doesn't-crash smoke test.
using HaloGrids
import MPI

@views d_xa(A) = A[2:end, :, :] .- A[1:end-1, :, :]
@views d_xi(A) = A[2:end, 2:end-1, 2:end-1] .- A[1:end-1, 2:end-1, 2:end-1]
@views d_ya(A) = A[:, 2:end, :] .- A[:, 1:end-1, :]
@views d_yi(A) = A[2:end-1, 2:end, 2:end-1] .- A[2:end-1, 1:end-1, 2:end-1]
@views d_za(A) = A[:, :, 2:end] .- A[:, :, 1:end-1]
@views d_zi(A) = A[2:end-1, 2:end-1, 2:end] .- A[2:end-1, 2:end-1, 1:end-1]
@views inn(A) = A[2:end-1, 2:end-1, 2:end-1]

@views function diffusion3D()
    MPI.Init()
    t = topology((0, 0, 0); periods=(true, true, true))

    # Physics
    lam, cp_min = 1.0, 1.0                      # thermal conductivity, minimal heat capacity
    lx, ly, lz = 10.0, 10.0, 10.0                # physical domain size

    # Numerics
    nx, ny, nz = 32, 32, 32                      # local (per-rank) grid size
    nt = 50                                      # number of time steps
    hw = 1
    g = grid(t, (nx, ny, nz); origin_on_vertex=true)
    Tf = field(zeros(nx, ny, nz); halowidths=(hw, hw, hw))
    T = Tf.data
    Cp = zeros(nx, ny, nz)

    dx = lx / (nx_g(g, Tf) - 1)
    dy = ly / (ny_g(g, Tf) - 1)
    dz = lz / (nz_g(g, Tf) - 1)

    # Initial conditions, built from each cell's *global* physical coordinate
    # via the per-field coordinate helpers — identical result regardless of
    # how many ranks the domain is split across.
    for k in 1:nz, j in 1:ny, i in 1:nx
        xg, yg, zg = x_g(i, dx, g, Tf), y_g(j, dy, g, Tf), z_g(k, dz, g, Tf)
        Cp[i, j, k] = cp_min +
            5 * exp(-(xg - lx / 1.5)^2 - (yg - ly / 2)^2 - (zg - lz / 1.5)^2) +
            5 * exp(-(xg - lx / 3.0)^2 - (yg - ly / 2)^2 - (zg - lz / 1.5)^2)
        T[i, j, k] =
            100 * exp(-((xg - lx / 2) / 2)^2 - ((yg - ly / 2) / 2)^2 - ((zg - lz / 3.0) / 2)^2) +
            50 * exp(-((xg - lx / 2) / 2)^2 - ((yg - ly / 2) / 2)^2 - ((zg - lz / 1.5) / 2)^2)
    end
    update_halo!(g, Tf)  # halo of T must be valid before the first flux computation

    dt = min(dx * dx, dy * dy, dz * dz) * cp_min / lam / 8.1

    owned = (hw+1:nx-hw, hw+1:ny-hw, hw+1:nz-hw)
    total_energy() = MPI.Allreduce(sum(Cp[owned...] .* T[owned...]), MPI.SUM, g.topology.comm)
    E0 = total_energy()

    qx = zeros(nx - 1, ny - 2, nz - 2)
    qy = zeros(nx - 2, ny - 1, nz - 2)
    qz = zeros(nx - 2, ny - 2, nz - 1)
    dTedt = zeros(nx - 2, ny - 2, nz - 2)

    for _ in 1:nt
        qx .= .-lam .* d_xi(T) ./ dx                                            # Fourier's law: q_x = -λ δT/δx
        qy .= .-lam .* d_yi(T) ./ dy
        qz .= .-lam .* d_zi(T) ./ dz
        dTedt .= (1.0 ./ inn(Cp)) .* (.-d_xa(qx) ./ dx .- d_ya(qy) ./ dy .- d_za(qz) ./ dz)  # δT/δt = 1/cₚ (-∇·q)
        T[2:end-1, 2:end-1, 2:end-1] .= inn(T) .+ dt .* dTedt
        update_halo!(g, Tf)
    end

    E1 = total_energy()
    rel_drift = abs(E1 - E0) / abs(E0)
    @assert rel_drift < 1e-9 "total thermal energy drifted by a relative $rel_drift over $nt steps (expected exact conservation on a periodic domain)"

    MPI.Barrier(g.topology.comm)
    t.rank == 0 && println(
        "diffusion3D: energy conserved to a relative $rel_drift after $nt steps " *
        "on $(prod(t.dims)) rank(s) (dims=$(t.dims)).")
    MPI.Finalize()
end

diffusion3D()
