push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

@testset "grid basics (N=$N)" for N in (2, 3)
    t = topology(ntuple(_ -> 0, N); comm=comm)
    nlocal = ntuple(d -> 4 + d, N)
    g = grid(t, nlocal)
    @test g.nlocal == nlocal
    @test ndims(g) == N
    @test g.origin == ntuple(_ -> 0.0, N)
    @test g.origin_on_vertex == false
    @test g.excache isa ExchangeCache
end

@testset "grid rejects invalid nlocal" begin
    t = topology((0, 0); comm=comm)
    @test_throws Exception grid(t, (0, 4))
    @test_throws Exception grid(t, (4, -1))
end

@testset "grid centering coherence checks" begin
    t = topology((0, 0); comm=comm)
    # origin_on_vertex=true needs even nlocal to center.
    @test_throws Exception grid(t, (5, 4); origin_on_vertex=true, centered=(true, false))
    grid(t, (4, 4); origin_on_vertex=true, centered=(true, false))  # even nlocal: ok
    # origin_on_vertex=false (default) needs odd nlocal to center.
    @test_throws Exception grid(t, (4, 4); centered=(true, false))
    grid(t, (5, 4); centered=(true, false))  # odd nlocal: ok
end

@testset "grid global_size derives nlocal from a global domain size (N=$N, periods=$periods)" for N in (2, 3),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N), ntuple(d -> isodd(d), N))

    t = topology(ntuple(_ -> 0, N); comm=comm, periods=periods)
    hw = ntuple(d -> d == 1 ? 2 : 1, N)
    core = ntuple(d -> 3 + d, N)  # arbitrary per-rank interior size
    nglobal_req = ntuple(d -> t.dims[d] * core[d] + 2 * hw[d] * !periods[d], N)
    expected_nlocal = ntuple(d -> core[d] + 2 * hw[d], N)

    g = grid(t, nglobal_req; global_size=true, halowidths=hw)
    @test g.nlocal == expected_nlocal

    f = field(zeros(g.nlocal); halowidths=hw)
    for d in 1:N
        @test nglobal(g, f, d) == nglobal_req[d]
    end
end

@testset "grid global_size matches manual nlocal construction" begin
    t = topology((0, 0); comm=comm)
    hw = (1, 1)
    core = (30, 20)
    nglobal_req = (t.dims[1] * core[1] + 2 * hw[1], t.dims[2] * core[2] + 2 * hw[2])
    g_auto = grid(t, nglobal_req; global_size=true, halowidths=hw)
    g_manual = grid(t, (core[1] + 2 * hw[1], core[2] + 2 * hw[2]))
    @test g_auto.nlocal == g_manual.nlocal
end

@testset "grid global_size defaults halowidths to 1, matching field's own default" begin
    t = topology((0, 0); comm=comm)
    core = (10, 10)
    nglobal_req = (t.dims[1] * core[1] + 2, t.dims[2] * core[2] + 2)
    g = grid(t, nglobal_req; global_size=true)
    @test g.nlocal == (core[1] + 2, core[2] + 2)
end

@testset "grid global_size rejects a non-divisible global size" begin
    t = topology((0, 0); comm=comm)
    if t.dims[1] > 1
        @test_throws Exception grid(t, (t.dims[1] * 4 + 1, 8); global_size=true)
    end
end

@testset "two grids on the same topology have distinct caches" begin
    t = topology((0, 0); comm=comm)
    g1 = grid(t, (6, 6))
    g2 = grid(t, (6, 6))
    @test g1.excache !== g2.excache
end

MPI.Barrier(comm)
rank == 0 && println("test_grid.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
