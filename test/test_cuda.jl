# Correctness tests for the CUDA extension (src/CUDAExt/, ext/HaloArraysCUDAExt.jl).
#
# Gated on CUDA.functional(): this machine has a real NVIDIA GPU, so this is
# expected to actually run and pass here, not just compile — see
# REDESIGN_PLAN.md's phase 6 status. On a machine/CI runner without a
# functional CUDA install, this prints a skip message and exits cleanly
# instead of calling MPI at all.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using CUDA

if !CUDA.functional()
    println("test_cuda.jl: CUDA not functional on this machine; skipping.")
    exit(0)
end

using HaloArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Same analytic model as test_halo_exchange.jl's check_halo (see that file
# for the full rationale on encoding position, not just rank, and why the
# per-dimension source-index decomposition below is exact even at corners),
# generalized to compare a (possibly GPU-resident) field's *host copy*
# against the expected value, since a GPU field never leaves the device
# otherwise.
const FILL_SCALE = 100_000.0

fill_value(rank::Integer, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N} =
    Float64(rank) * FILL_SCALE + Float64(LinearIndices(nlocal)[ci])

filled_array(::Type{T}, rank::Integer, nlocal::NTuple{N,Int}) where {T,N} =
    T[T(fill_value(rank, nlocal, ci)) for ci in CartesianIndices(nlocal)]

function expected_value(grid::Grid{N}, halowidths::NTuple{N,Int}, dims, nlocal::NTuple{N,Int}, ci::CartesianIndex{N}) where {N}
    topo = grid.topology
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
    if all(iszero, offset)
        return fill_value(rank, nlocal, ci)
    end
    target_coords = ntuple(N) do d
        c = topo.coords[d] + offset[d]
        topo.periods[d] ? mod(c, topo.dims[d]) : c
    end
    target_rank = MPI.Cart_rank(topo.comm, collect(target_coords))
    return fill_value(target_rank, nlocal, CartesianIndex(source_idx...))
end

function check_halo_gpu(grid::Grid{N}, f::Field{T,N}, dims, nlocal) where {T,N}
    hostdata = Array(f.data)
    ok = true
    for ci in CartesianIndices(hostdata)
        expected = T(expected_value(grid, f.halowidths, dims, nlocal, ci))
        if hostdata[ci] != expected
            @error "mismatch" rank ci got=hostdata[ci] expected
            ok = false
        end
    end
    return ok
end

@testset "update_halo! correctness on CuArray fields (N=$N, periods=$periods)" for N in (2, 3),
    periods in (ntuple(_ -> false, N), ntuple(_ -> true, N))

    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=periods)
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)

    A = CuArray(filled_array(Float64, rank, nlocal))
    B = CuArray(filled_array(Float32, rank, nlocal))
    hwA = ntuple(_ -> 1, N)
    hwB = ntuple(d -> d == 1 ? 2 : 1, N)
    fA = field(A; halowidths=hwA)
    fB = field(B; halowidths=hwB)

    update_halo!(g, fA, fB)

    @test check_halo_gpu(g, fA, 1:N, nlocal)
    @test check_halo_gpu(g, fB, 1:N, nlocal)
end

@testset "start!/finish! matches update_halo! on CuArray fields (N=$N)" for N in (2, 3)
    topo = topology(ntuple(_ -> 0, N); comm=comm, periods=ntuple(_ -> true, N))
    nlocal = ntuple(_ -> 6, N)
    g = grid(topo, nlocal)
    A = CuArray(filled_array(Float64, rank, nlocal))
    fA = field(A; halowidths=ntuple(_ -> 1, N))

    handle = halo_exchange_start!(g, fA)
    CUDA.synchronize()  # work "inserted" between start! and finish! must not corrupt the result
    halo_exchange_finish!(handle)

    @test check_halo_gpu(g, fA, 1:N, nlocal)
end

@testset "mixed CPU and GPU fields in the same call" begin
    topo = topology((0, 0); comm=comm, periods=(true, true))
    nlocal = (6, 6)
    g = grid(topo, nlocal)

    Acpu = filled_array(Float64, rank, nlocal)
    Agpu = CuArray(filled_array(Float64, rank, nlocal))
    fcpu = field(Acpu; halowidths=(1, 1))
    fgpu = field(Agpu; halowidths=(1, 1))

    update_halo!(g, fcpu, fgpu)

    @test check_halo_gpu(g, fcpu, 1:2, nlocal)  # Array() on a CPU field's data is a no-op
    @test check_halo_gpu(g, fgpu, 1:2, nlocal)
end

@testset "GPU exchange on two concurrently-live Grids doesn't cross-talk" begin
    # The actual point of gpu_cache living on ExchangeCache (per-grid) rather
    # than module-global: interleaved start!/finish! on two independently
    # sized grids must not share or clobber each other's device buffers.
    topo1 = topology((0, 0); comm=comm, periods=(true, true))
    topo2 = topology((0, 0); comm=comm, periods=(true, true))
    g1 = grid(topo1, (6, 6))
    g2 = grid(topo2, (11, 9))

    A1 = CuArray(filled_array(Float64, rank, g1.nlocal))
    A2 = CuArray(filled_array(Float32, rank, g2.nlocal))
    f1 = field(A1; halowidths=(1, 2))
    f2 = field(A2; halowidths=(2, 1))

    h1 = halo_exchange_start!(g1, f1)
    h2 = halo_exchange_start!(g2, f2)  # concurrently in flight on a *different* grid
    halo_exchange_finish!(h2)
    halo_exchange_finish!(h1)

    @test check_halo_gpu(g1, f1, 1:2, g1.nlocal)
    @test check_halo_gpu(g2, f2, 1:2, g2.nlocal)
    @test g1.excache.gpu_cache[] !== g2.excache.gpu_cache[]
end

MPI.Barrier(comm)
rank == 0 && println("test_cuda.jl: all tests passed on $(MPI.Comm_size(comm)) ranks.")
MPI.Finalize()
