# Flagship test for the whole rewrite: two Grids of *different* sizes (and,
# here, different field element types too, to also regression-test buffer
# reuse across differently-sized types in the same cache slot) on the same
# COMM_WORLD, with interleaved start!/finish! calls across many iterations,
# asserting neither grid's data is ever corrupted by the other's in-flight
# exchange.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Fill value encodes both the owning rank and this cell's position within
# that rank's own array (see test_halo_exchange.jl for the full rationale and
# a hand-verified derivation of why the per-dimension decomposition below is
# exact even across corners) — a rank-only fill can't tell a correctly
# sourced neighbor value from one read at the wrong offset.
const FILL_SCALE = 100_000.0

fill_value(rank::Integer, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N} =
    Float64(rank) * FILL_SCALE + Float64(LinearIndices(nlocal)[ci])

filled_array(::Type{T}, rank::Integer, nlocal::NTuple{N,Int}) where {T,N} =
    T[T(fill_value(rank, nlocal, ci)) for ci in CartesianIndices(nlocal)]

function expected_value(topo, halowidths::NTuple{N,Int}, dims, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N}
    offset = zeros(Int, N)
    source_idx = collect(Tuple(ci))
    for d in dims
        hw = halowidths[d]
        idx = ci[d]
        dir = idx <= hw ? -1 : (idx > nlocal[d] - hw ? 1 : 0)
        dir == 0 && continue
        if topo.periods[d] || (0 <= topo.coords[d] + dir <= topo.dims[d] - 1)
            offset[d] += dir
            nA = nlocal[d]
            source_idx[d] = dir == -1 ? (nA - 2hw + idx) : (idx - nA + 2hw)
        end
    end
    all(iszero, offset) && return rank, ci
    target_coords = ntuple(N) do d
        c = topo.coords[d] + offset[d]
        topo.periods[d] ? mod(c, topo.dims[d]) : c
    end
    target_rank = MPI.Cart_rank(topo.comm, collect(target_coords))
    return target_rank, CartesianIndex(source_idx...)
end

function check(topo, f, dims, nlocal, ::Type{T}) where {T}
    for ci in CartesianIndices(f.data)
        target_rank, source_ci = expected_value(topo, f.halowidths, dims, nlocal, ci)
        expected = T(fill_value(target_rank, nlocal, source_ci))
        f.data[ci] == expected || return false
    end
    return true
end

@testset "interleaved exchanges on two differently-sized grids don't cross-talk" begin
    topo1 = topology((0, 0); comm=comm, periods=(true, true))
    topo2 = topology((0, 0); comm=comm, periods=(true, true))
    g1 = grid(topo1, (6, 6))
    g2 = grid(topo2, (11, 9))  # different size AND not evenly related to g1's

    @test g1.excache !== g2.excache
    @test topo1.comm != topo2.comm

    niter = 25
    for iter in 1:niter
        A1 = filled_array(Float64, rank, g1.nlocal)
        A2 = filled_array(Float32, rank, g2.nlocal)
        f1 = field(A1; halowidths=(1, 2))
        f2 = field(A2; halowidths=(2, 1))

        h1 = halo_exchange_start!(g1, f1)
        h2 = halo_exchange_start!(g2, f2)   # concurrently in flight on a *different* grid: must not error or clobber g1's cache
        halo_exchange_finish!(h2)
        halo_exchange_finish!(h1)

        @test check(topo1, f1, 1:2, g1.nlocal, Float64)
        @test check(topo2, f2, 1:2, g2.nlocal, Float32)
    end
end

MPI.Barrier(comm)
rank == 0 && println("test_multigrid.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
