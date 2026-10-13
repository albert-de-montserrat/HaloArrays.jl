# Correctness tests for the CellArrays extension (src/CellArraysExt/,
# ext/HaloGridsCellArraysExt.jl).
#
# Strategy: same analytic-rank-fill idea as test_halo_exchange.jl's
# expected_value/check_halo, extended to distinguish cell components: cell
# component `j` (1-based, flat) is filled with `10*rank + j` in every local
# cell (rank stays in 0:7 at this suite's NPROCS_MATRIX, so this is always an
# exactly-representable, uniquely-decodable Float64 for every (rank,
# component) pair) rather than just `rank`. This additionally catches any bug
# that swaps or aliases components with each other during the per-component
# decomposition, which a rank-uniform fill (identical across every component)
# could not.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids
using CellArrays: CellArrays, CPUCellArray
using StaticArrays: SMatrix

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

const CellT = SMatrix{2,2,Float64,4}
const NCOMP = 4  # celllength(CellT) == 2*2

# Same neighbor-tracing model as test_halo_exchange.jl's expected_value, but
# returning the source rank as a plain Int (rather than pre-multiplying it
# into a value) so this file's `10*rank+j` component encoding can be built on
# top of it.
function source_rank(grid::Grid{N}, halowidths::NTuple{N,Int}, dims, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N}
    topo = grid.topology
    offset = zeros(Int, N)
    for d in dims
        hw = halowidths[d]
        idx = ci[d]
        dir = idx <= hw ? -1 : (idx > nlocal[d] - hw ? 1 : 0)
        dir == 0 && continue
        if topo.periods[d] || (0 <= topo.coords[d] + dir <= topo.dims[d] - 1)
            offset[d] += dir
        end
        # else: no neighbor on that side in dimension d; nothing changes for dim d.
    end
    all(iszero, offset) && return rank
    target_coords = ntuple(N) do d
        c = topo.coords[d] + offset[d]
        topo.periods[d] ? mod(c, topo.dims[d]) : c
    end
    return MPI.Cart_rank(topo.comm, collect(target_coords))
end

function fill_components!(A::CellArrays.CellArray, rnk::Integer)
    for j in 1:NCOMP
        CellArrays.field(A, (j,)) .= Float64(10 * rnk + j)
    end
    return A
end

function check_cellarray(grid::Grid{N}, A::CellArrays.CellArray{CellT,N}, halowidths::NTuple{N,Int}, dims, nlocal) where {N}
    ok = true
    for j in 1:NCOMP
        v = CellArrays.field(A, (j,))
        for ci in CartesianIndices(v)
            expected = Float64(10 * source_rank(grid, halowidths, dims, nlocal, ci) + j)
            if v[ci] != expected
                @error "mismatch" rank component=j ci got=v[ci] expected
                ok = false
            end
        end
    end
    return ok
end

@testset "update_halo! correctness on CellArrays (N=$N, B=$B, periods=$periods)" for N in (2, 3),
    B in (0, 1),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N))

    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=periods)
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    A = CPUCellArray{CellT,B}(undef, nlocal)
    hw = ntuple(_ -> 1, N)
    fill_components!(A, rank)

    update_halo!(g, A; halowidths=hw)

    @test check_cellarray(g, A, hw, 1:N, nlocal)
end

@testset "halo_exchange_start!/finish! matches update_halo! on a CellArray (N=$N)" for N in (2, 3)
    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=ntuple(_ -> true, N))
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    A = CPUCellArray{CellT,0}(undef, nlocal)
    hw = ntuple(d -> d == 1 ? 2 : 1, N)
    fill_components!(A, rank)

    handle = halo_exchange_start!(g, A; halowidths=hw)
    halo_exchange_finish!(handle)

    @test check_cellarray(g, A, hw, 1:N, nlocal)
end

@testset "scalar (Number) cell type is rejected with a clear error" begin
    topo = topology((0, 0); comm=comm, periods=(true, true))
    nlocal = (6, 6)
    g = grid(topo, nlocal)
    Anum = CPUCellArray{Float64,0}(undef, nlocal)
    @test_throws ErrorException update_halo!(g, Anum)
end

MPI.Barrier(comm)
rank == 0 && println("test_cellarrays.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
