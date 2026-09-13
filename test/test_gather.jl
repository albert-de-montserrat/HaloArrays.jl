push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

@testset "gather! (N=$N)" for N in (2, 3)
    t = topology(ntuple(_ -> 0, N); comm=comm)
    nlocal = ntuple(_ -> 3, N)
    A = fill(Float64(rank), nlocal)
    A_global = rank == 0 ? zeros(Float64, nlocal .* t.dims) : nothing

    gather!(A, A_global, t)

    if rank == 0
        ok = true
        for I in CartesianIndices(A_global)
            coords = ntuple(d -> (Tuple(I)[d] - 1) ÷ nlocal[d], N)
            expected = MPI.Cart_rank(t.comm, collect(coords))
            ok &= (A_global[I] == expected)
        end
        @test ok
    end
end

@testset "gather! via Grid convenience method" begin
    t = topology((0, 0); comm=comm)
    g = grid(t, (3, 3))
    A = fill(Float64(rank), g.nlocal)
    A_global = rank == 0 ? zeros(Float64, g.nlocal .* t.dims) : nothing
    gather!(A, A_global, g)
    if rank == 0
        @test sum(A_global) == sum(0:nprocs-1) * 9
    end
end

MPI.Barrier(comm)
rank == 0 && println("test_gather.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
