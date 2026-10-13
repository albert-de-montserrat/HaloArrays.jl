push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

@testset "topology matches raw MPI.Cart_* (N=$N, periods=$periods)" for N in (2, 3),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N))

    t = topology(ntuple(_ -> 0, N); comm=comm, periods=periods)

    ref_dims = Tuple(MPI.Dims_create(nprocs, zeros(Int, N)))
    ref_comm = MPI.Cart_create(comm, collect(ref_dims); periodic=collect(periods), reorder=true)
    ref_coords = Tuple(Int.(MPI.Cart_coords(ref_comm)))
    ref_neighbors = ntuple(d -> Int.(MPI.Cart_shift(ref_comm, d - 1, 1)), N)

    @test t.dims == ref_dims
    @test t.periods == periods
    @test t.coords == ref_coords
    @test t.neighbors == ref_neighbors
    @test ndims(t) == N
end

@testset "two topologies from the same comm don't share a communicator" begin
    t1 = topology((0, 0); comm=comm)
    t2 = topology((0, 0); comm=comm)
    @test t1.comm != t2.comm
end

@testset "invalid dims rejected" begin
    @test_throws Exception topology((1,); comm=comm)          # 1D unsupported
    @test_throws Exception topology((1, 1, 1, 1); comm=comm)  # 4D unsupported
    @test_throws Exception topology((-1, 0); comm=comm)       # negative
end

MPI.Barrier(comm)
rank == 0 && println("test_topology.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
