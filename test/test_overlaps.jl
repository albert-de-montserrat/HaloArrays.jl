# Tests for per-field `overlaps` (neighbors share more than 2*halowidths lines)
# and for `subtopology`.
#
# Oracle: every cell is tagged with a value that depends only on its *global*
# index, computed from `iglobal` (wrapped for periodic dimensions). Halo cells
# start as -1. After an exchange, any halo cell whose neighbor exists must hold
# the tag of its own global index -- that holds only if the neighbor sent
# exactly the lines `overlaps-halowidths+1:overlaps` (low) /
# `n-overlaps+1:n-overlaps+halowidths` (high). Cells with a halo side without a
# neighbor (physical boundary) must stay -1.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

tag(ig::NTuple{N,Int}) where {N} = Float64(sum(ig[d] * 37^d for d in 1:N))

function global_index(g, f, ci::CartesianIndex{N}) where {N}
    return ntuple(N) do d
        ig = iglobal(g, f, d, ci[d]; wrap_periodic = false)
        g.topology.periods[d] ? mod1(ig, nglobal(g, f, d)) : ig
    end
end

is_halo(f, d, i) = i <= f.halowidths[d] || i > size(f.data, d) - f.halowidths[d]
halo_side(f, d, i) = i <= f.halowidths[d] ? 1 : 2

function initial_data(g, f)
    A = fill(-1.0, size(f.data))
    for ci in CartesianIndices(A)
        any(d -> is_halo(f, d, ci[d]), 1:ndims(A)) && continue
        A[ci] = tag(global_index(g, f, ci))
    end
    return A
end

# `faces`: only cells with exactly one halo dimension are refreshed.
function expected_data(g, f; faces = false)
    A = initial_data(g, f)
    topo = g.topology
    for ci in CartesianIndices(A)
        halo_dims = [d for d in 1:ndims(A) if is_halo(f, d, ci[d])]
        isempty(halo_dims) && continue
        faces && length(halo_dims) != 1 && continue
        all(d -> has_neighbor(topo.neighbors[d][halo_side(f, d, ci[d])]), halo_dims) || continue
        A[ci] = tag(global_index(g, f, ci))
    end
    return A
end

const CASES = (
    (nlocal = (9, 10), halowidths = (1, 1), overlaps = (3, 4)),
    (nlocal = (9, 10), halowidths = (1, 1), overlaps = (2, 2)),
    (nlocal = (12, 11), halowidths = (2, 1), overlaps = (5, 3)),
    (nlocal = (8, 9, 10), halowidths = (1, 1, 1), overlaps = (2, 3, 4)),
    (nlocal = (9, 9, 9), halowidths = (2, 1, 1), overlaps = (6, 2, 3)),
)

@testset "overlaps: $(c.nlocal) hw=$(c.halowidths) ol=$(c.overlaps) periodic=$periodic" for c in CASES,
    periodic in (false, true)

    N = length(c.nlocal)
    t = topology(ntuple(_ -> 0, N); comm = comm, periods = ntuple(_ -> periodic, N))
    g = grid(t, c.nlocal)
    f = field(zeros(c.nlocal); halowidths = c.halowidths, overlaps = c.overlaps)
    @test overlaps(f) == c.overlaps

    @testset "nglobal/iglobal" begin
        for d in 1:N
            ng = nglobal(g, f, d)
            @test ng == t.dims[d] * (c.nlocal[d] - c.overlaps[d]) + c.overlaps[d] * !periodic
            if !periodic
                @test iglobal(g, f, d, 1) == t.coords[d] * (c.nlocal[d] - c.overlaps[d]) + 1
                t.coords[d] == t.dims[d] - 1 && @test iglobal(g, f, d, c.nlocal[d]) == ng
            end
        end
    end

    @testset "update_halo!" begin
        f.data .= initial_data(g, f)
        update_halo!(g, f)
        @test f.data == expected_data(g, f)
    end

    @testset "split start!/finish!" begin
        f.data .= initial_data(g, f)
        h = halo_exchange_start!(g, f)
        halo_exchange_finish!(h)
        @test f.data == expected_data(g, f)
    end

    @testset "region=:faces" begin
        f.data .= initial_data(g, f)
        update_halo!(g, f; region = :faces)
        @test f.data == expected_data(g, f; faces = true)
    end
end

@testset "mixed overlaps in one call" begin
    t = topology((0, 0); comm = comm)
    g = grid(t, (9, 9))
    fa = field(zeros(9, 9))                              # default overlaps (2, 2)
    fb = field(zeros(9, 9); overlaps = (3, 4))
    fa.data .= initial_data(g, fa)
    fb.data .= initial_data(g, fb)
    update_halo!(g, fa, fb)
    @test fa.data == expected_data(g, fa)
    @test fb.data == expected_data(g, fb)
end

@testset "overlaps validation" begin
    A = zeros(8, 8)
    @test field(A).overlaps == (2, 2)
    @test field(A; halowidths = (2, 1)).overlaps == (4, 2)
    @test field(A; overlaps = (3, 8)).overlaps == (3, 8)
    @test_throws Exception field(A; overlaps = (1, 2))        # < 2*halowidths
    @test_throws Exception field(A; halowidths = (2, 1), overlaps = (3, 2))
    @test_throws Exception field(A; overlaps = (2, 9))        # > size(A, d)
end

@testset "subtopology" begin
    t = topology((0, 0, 0); comm = comm, periods = (true, false, true))
    s = subtopology(t, (true, true, false))
    @test s isa Topology{2}
    @test s.dims == (t.dims[1], t.dims[2])
    @test s.periods == (true, false)
    @test s.coords == (t.coords[1], t.coords[2])
    @test MPI.Comm_size(s.comm) == t.dims[1] * t.dims[2]
    @test s.comm != t.comm
    for d in 1:2
        @test has_neighbor(s.neighbors[d][1]) == has_neighbor(t.neighbors[d][1])
        @test has_neighbor(s.neighbors[d][2]) == has_neighbor(t.neighbors[d][2])
    end

    # Exchange a 2D array on the sub-topology: halos hold the neighbor's
    # sub-communicator rank, and ranks in different sub-communicators never mix.
    g = grid(s, (6, 7))
    A = fill(Float64(s.rank), 6, 7)
    update_halo!(g, A)
    for d in 1:2, side in 1:2
        nb = s.neighbors[d][side]
        has_neighbor(nb) || continue
        idx = d == 1 ? (side == 1 ? (1, 4) : (6, 4)) : (side == 1 ? (3, 1) : (3, 7))
        @test A[idx...] == Float64(nb)
    end

    s3 = subtopology(t, (true, false, true))
    @test s3.dims == (t.dims[1], t.dims[3])
    @test s3.periods == (true, true)

    s_full = subtopology(t, (true, true, true))
    @test s_full isa Topology{3}
    @test s_full.dims == t.dims

    @test_throws Exception subtopology(t, (true, false, false))
end

MPI.Barrier(comm)
rank == 0 && println("test_overlaps.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
