# Correctness tests for the per-field coordinate helpers (nglobal/nx_g/.../
# iglobal/ix_g/.../coord_g/x_g/...).
#
# Two independent strategies are used:
#  1. Single-rank analytic cases, hand-derived and checked against the
#     documented formulas in src/coordinates.jl (covers origin/origin_on_vertex/
#     centered handling, which multi-rank cases below don't exercise directly).
#  2. A cross-rank consistency check that reuses the already-proven-correct
#     halo_exchange machinery as an oracle: fill a field so each cell holds its
#     own rank's independently-computed physical coordinate, run a real
#     (isolated, single-dimension) halo exchange, and check that what a rank
#     receives into its halo from a neighbor agrees with what that rank's own
#     coordinate formula predicts for the same physical position. This is a
#     genuine end-to-end check of the coordinate math against the topology,
#     not just internal self-consistency of the formula.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

@testset "nglobal/nx_g/ny_g/nz_g" begin
    t = topology((0, 0, 0); comm=comm)
    nlocal = (6, 8, 10)
    g = grid(t, nlocal)
    f1 = field(zeros(nlocal); halowidths=(1, 1, 1))
    f2 = field(zeros(nlocal); halowidths=(2, 2, 2))
    for d in 1:3
        dims_d = t.dims[d]
        @test nglobal(g, f1, d) == dims_d * (nlocal[d] - 2) + 2 * !t.periods[d]
        @test nglobal(g, f2, d) == dims_d * (nlocal[d] - 4) + 4 * !t.periods[d]
    end
    @test nx_g(g, f1) == nglobal(g, f1, 1)
    @test ny_g(g, f1) == nglobal(g, f1, 2)
    @test nz_g(g, f1) == nglobal(g, f1, 3)
    @test_throws Exception nglobal(g, f1, 4)

    # nprocs=1 along a dimension, non-periodic: the field sees exactly its own
    # local size as the global size (no neighbor to duplicate cells with).
    t1 = topology((1, 1, 1); comm=MPI.COMM_SELF)
    g1 = grid(t1, nlocal)
    @test nx_g(g1, f1) == nlocal[1]
    @test ny_g(g1, f1) == nlocal[2]
    @test nz_g(g1, f1) == nlocal[3]
end

@testset "coord_g single-rank analytic cases" begin
    t1 = topology((1, 1); comm=MPI.COMM_SELF)

    # Vertex-origin, not centered, dx=2, nlocal=3 (=> nglobal=3 at nprocs=1):
    # expected global coordinates 0, 2, 4 (matches the old package's own
    # documented x_g example for this exact configuration).
    g = grid(t1, (3, 3); origin=(0.0, 0.0), origin_on_vertex=true)
    f = field(zeros(3, 3); halowidths=(1, 1))
    @test [x_g(i, 2.0, g, f) for i in 1:3] ≈ [0.0, 2.0, 4.0]
    @test [y_g(i, 2.0, g, f) for i in 1:3] ≈ [0.0, 2.0, 4.0]

    # Cell-centered (origin_on_vertex=false), not centered: cell centers sit
    # dx/2 inside each edge, spanning [0, nglobal*dx].
    g2 = grid(t1, (3, 3); origin=(0.0, 0.0), origin_on_vertex=false)
    @test [x_g(i, 2.0, g2, f) for i in 1:3] ≈ [1.0, 3.0, 5.0]

    # Vertex-origin, centered on the origin, even nlocal=4 (required for
    # centered+origin_on_vertex, see grid.jl): symmetric spread around 0.
    g3 = grid(t1, (4, 4); origin=(0.0, 0.0), origin_on_vertex=true, centered=(true, true))
    f3 = field(zeros(4, 4); halowidths=(1, 1))
    @test [x_g(i, 1.0, g3, f3) for i in 1:4] ≈ [-1.5, -0.5, 0.5, 1.5]
    @test sum(x_g(i, 1.0, g3, f3) for i in 1:4) ≈ 0.0 atol=1e-12

    # A nonzero origin just shifts everything by that amount.
    g4 = grid(t1, (3, 3); origin=(10.0, 0.0), origin_on_vertex=true)
    @test [x_g(i, 2.0, g4, f) for i in 1:3] ≈ [10.0, 12.0, 14.0]
end

@testset "iglobal periodic wraparound (N=$N)" for N in (2, 3)
    dims0 = ntuple(_ -> 0, N)
    t = topology(dims0; comm=comm, periods=ntuple(_ -> true, N))
    nlocal = ntuple(_ -> 6, N)
    g = grid(t, nlocal)
    f = field(zeros(nlocal); halowidths=ntuple(_ -> 1, N))
    for d in 1:N
        ng = nglobal(g, f, d)
        for i in 1:nlocal[d]
            iw = iglobal(g, f, d, i; wrap_periodic=true)
            @test 1 <= iw <= ng
            iraw = iglobal(g, f, d, i; wrap_periodic=false)
            @test iw == iraw || iw == iraw - ng || iw == iraw + ng
        end
    end
end

# Cross-rank consistency: fill a field with each cell's own coordinate, run a
# real isolated-dimension halo exchange, and check what arrives in the halo
# against this rank's own formula for the same physical position. On a
# periodic dimension the two can differ by exactly one domain length (the
# "wrap" is physically the same point), so compare modulo L.
function check_coord_exchange(grid::Grid{N}, dim::Integer, dx::Float64, periodic::Bool) where {N}
    hw = 1
    f = field(zeros(grid.nlocal); halowidths=ntuple(_ -> hw, N))
    for ci in CartesianIndices(f.data)
        f.data[ci] = coord_g(grid, f, dim, ci[dim], dx)
    end
    update_halo!(grid, f; dims=(dim,))

    topo = grid.topology
    nlocal_d = grid.nlocal[dim]
    L = nglobal(grid, f, dim) * dx
    ok = true
    for ci in CartesianIndices(f.data)
        i = ci[dim]
        is_lo = i <= hw
        is_hi = i > nlocal_d - hw
        (is_lo || is_hi) || continue
        nb = is_lo ? topo.neighbors[dim][1] : topo.neighbors[dim][2]
        has_neighbor(nb) || continue
        received = f.data[ci]
        mine = coord_g(grid, f, dim, i, dx)
        diff = received - mine
        wrapped = mod(diff + L / 2, L) - L / 2
        if !periodic
            ok &= isapprox(received, mine; atol=1e-9)
        else
            ok &= isapprox(wrapped, 0.0; atol=1e-9)
        end
    end
    return ok
end

@testset "coord_g cross-rank consistency via real exchange (N=$N, periodic=$periodic)" for N in (2, 3), periodic in (false, true)
    dims0 = ntuple(_ -> 0, N)
    t = topology(dims0; comm=comm, periods=ntuple(_ -> periodic, N))
    nlocal = ntuple(_ -> 8, N)
    g = grid(t, nlocal; origin_on_vertex=true)
    for d in 1:N
        @test check_coord_exchange(g, d, 1.5, periodic)
    end
end

MPI.Barrier(comm)
rank == 0 && println("test_coordinates.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
