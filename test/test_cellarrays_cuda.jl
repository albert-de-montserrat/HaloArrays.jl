# Correctness test for the CellArrays extension against a CuCellArray
# (src/CellArraysExt/halo_exchange.jl + src/CUDAExt/).
#
# The whole point of this file: HaloArraysCellArraysExt writes *no*
# CUDA-specific code at all — it only decomposes a CellArray into per-
# component views via CellArrays.field(A,(j,)) and hands them to HaloArrays'
# existing Field/exchange machinery. For a CuCellArray those component views
# are expected to already fall within CUDA.AnyCuArray's dispatch union
# (a view/reshape of a CuArray), meaning the existing HaloArraysCUDAExt
# pack_target/stage_send!/stage_recv! methods should apply automatically. If
# that expectation is wrong, this file fails loud (either a MethodError from
# missing GPU dispatch, or a CUDA scalar-indexing error) rather than silently
# running the slow/incorrect CPU path — see REDESIGN_PLAN.md's "Phase 9"
# writeup for the reasoning and the "measured, don't guess" rule this
# verifies.
#
# Gated on CUDA.functional(), same as test_cuda.jl: this machine has a real
# NVIDIA GPU, so this is expected to actually run and pass here, not just
# compile.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using CUDA

if !CUDA.functional()
    println("test_cellarrays_cuda.jl: CUDA not functional on this machine; skipping.")
    exit(0)
end

using HaloArrays
using CellArrays: CellArrays, CPUCellArray
using StaticArrays: SMatrix

# Defines `CuCellArray` in this (top-level/Main) module's namespace — see
# CellArrays.jl's `@define_CuCellArray` docstring: GPU type aliases are
# provided via a macro, not exported directly, to avoid CellArrays.jl itself
# depending on CUDA.
CellArrays.@define_CuCellArray()

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

const CellT = SMatrix{2,2,Float64,4}
const NCOMP = 4  # celllength(CellT) == 2*2

# Same neighbor-tracing model as test_cellarrays.jl's source_rank.
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

# Compares against a *host* copy of each component view, since a GPU field
# never leaves the device otherwise (mirrors test_cuda.jl's check_halo_gpu).
# Array(...) on the device-resident component view is itself part of what
# this file verifies: it only works without a scalar-indexing error if the
# view's type reaches GPUArrays' fast host<->device copyto!.
function check_cellarray_gpu(grid::Grid{N}, A::CellArrays.CellArray{CellT,N}, halowidths::NTuple{N,Int}, dims, nlocal) where {N}
    ok = true
    for j in 1:NCOMP
        v = Array(CellArrays.field(A, (j,)))
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

@testset "update_halo! correctness on CuCellArray (N=$N, B=$B, periods=$periods)" for N in (2, 3),
    B in (0, 1),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N))

    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=periods)
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    A = CuCellArray{CellT,B}(undef, nlocal)
    hw = ntuple(_ -> 1, N)
    fill_components!(A, rank)

    update_halo!(g, A; halowidths=hw)

    @test check_cellarray_gpu(g, A, hw, 1:N, nlocal)
end

@testset "halo_exchange_start!/finish! matches update_halo! on a CuCellArray (N=$N)" for N in (2, 3)
    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=ntuple(_ -> true, N))
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    A = CuCellArray{CellT,0}(undef, nlocal)
    hw = ntuple(d -> d == 1 ? 2 : 1, N)
    fill_components!(A, rank)

    handle = halo_exchange_start!(g, A; halowidths=hw)
    CUDA.synchronize()  # work "inserted" between start! and finish! must not corrupt the result
    halo_exchange_finish!(handle)

    @test check_cellarray_gpu(g, A, hw, 1:N, nlocal)
end

MPI.Barrier(comm)
rank == 0 && println("test_cellarrays_cuda.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
