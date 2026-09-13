push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloArrays

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

@testset "two grids on the same topology have distinct caches" begin
    t = topology((0, 0); comm=comm)
    g1 = grid(t, (6, 6))
    g2 = grid(t, (6, 6))
    @test g1.excache !== g2.excache
end

MPI.Barrier(comm)
rank == 0 && println("test_grid.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
