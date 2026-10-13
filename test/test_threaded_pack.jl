# Correctness check for the opt-in `threaded_pack=true` path (see
# THREADED_PACK_MIN_LEN and the threaded pack!/unpack! dispatch in
# src/halo_exchange.jl, and `threaded_pack` on `grid` in src/grid.jl).
#
# Needs Julia launched with `-t N>1` to actually exercise `Threads.@spawn`
# (the gate requires `Threads.nthreads() > 1`); test/runtests.jl runs this
# file separately from the rest of the suite, with `-t 4`. Run directly with,
# e.g., `julia --project -t 4 test/test_threaded_pack.jl` to reproduce.
#
# Grid is sized so dim=1's halo slab clears THREADED_PACK_MIN_LEN (exercising
# the threaded path) while dims 2 and 3 stay well below it (exercising the
# plain sequential fallback in the very same exchange) — correctness must
# hold identically on both.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if Threads.nthreads() < 2
    rank == 0 && @warn "test_threaded_pack.jl needs Julia launched with -t N>1 to exercise the threaded path; skipping (Threads.nthreads()=$(Threads.nthreads()))."
    MPI.Finalize()
    exit(0)
end

topo = topology((0, 0, 0); comm=comm, periods=(true, true, true))
nlocal = (4, 256, 256)  # dim=1 halo slab (hw=1): 1*256*256=65536 elements, at THREADED_PACK_MIN_LEN
g = grid(topo, nlocal; threaded_pack=true)
hw = (1, 1, 1)

A = fill(Float64(rank), nlocal)
fA = field(A; halowidths=hw)

update_halo!(g, fA)

# Fully periodic on every dimension: the final value at any cell is the rank
# found by shifting this rank's Cartesian coords by -1/0/+1 per dimension
# according to whether that cell sits in the lo/hi halo or the interior of
# `fA.data` (which is exactly `nlocal`-sized — halos are its outermost
# `halowidths[d]` cells, not separate padding, see check_field_shapes in
# src/validation.jl) — no boundary special-casing needed, and (since every
# dimension is periodic) combining the three per-dimension shifts in one step
# is equivalent to the sequential-dimension exchange's cumulative effect.
function expected(topo, nlocal, hw, ci)
    offset = ntuple(3) do d
        idx = ci[d]
        idx <= hw[d] ? -1 : (idx > nlocal[d] - hw[d] ? 1 : 0)
    end
    all(iszero, offset) && return Float64(rank)
    coords = ntuple(d -> mod(topo.coords[d] + offset[d], topo.dims[d]), 3)
    return Float64(MPI.Cart_rank(topo.comm, collect(coords)))
end

ok = true
for ci in CartesianIndices(fA.data)
    exp = expected(topo, nlocal, hw, ci)
    if fA.data[ci] != exp
        @error "mismatch" rank ci got=fA.data[ci] exp
        global ok = false
    end
end

@testset "threaded_pack correctness (nprocs=$(MPI.Comm_size(comm)), nthreads=$(Threads.nthreads()))" begin
    @test ok
end

MPI.Barrier(comm)
rank == 0 && println("test_threaded_pack.jl: all tests passed (nthreads=$(Threads.nthreads())).")
MPI.Finalize()
