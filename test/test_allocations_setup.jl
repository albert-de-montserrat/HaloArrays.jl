# Allocation measurement for the construction/one-time-setup paths phase 8
# left unmeasured — see REDESIGN_PLAN.md ("gather! and Grid/Field/Topology
# construction ... lower priority since they're one-time setup cost, not
# per-timestep hot path"). Unlike test_allocations.jl's exchange hot path,
# these are NOT expected to be zero-allocation: `topology()`/`grid()` build a
# genuinely new value every call (that's the point — no caching/reuse is
# possible or desirable for a one-time setup call), and `gather!` constructs a
# fresh MPI derived datatype every call (no per-(topology,size) cache exists
# for it, unlike the exchange path's per-grid ExchangeCache). This test exists
# to have real numbers on record and to catch a real regression (e.g. a
# refactor that makes `field()` stop eliding to zero, or multiplies
# `topology()`'s cost by 100x), not to hold these to the hot-path bar.
#
# Measured on this machine, stable across nprocs in {1,2,4}:
#   topology(): 688 B/call (MPI.jl's own Dims_create/Cart_create/Cart_coords/
#               Cart_shift wrapping — not this package's code)
#   grid():     176 B/call (the Grid struct + a fresh empty ExchangeCache)
#   field():    0 B/call, both explicit and default halowidths — genuinely
#               elides like the exchange hot path, since Field is a thin
#               immutable wrapper and NTuple broadcasting over a Tuple
#               produces another Tuple, not a heap Vector.
#   gather!():  64-240 B/call, root vs non-root ranks differing (root builds a
#               fresh derived datatype + counts/displs every call; non-root
#               just does the send side) — not nprocs-scaling, just a real
#               per-call MPI.Types.create_subarray/create_resized/commit! cost.
#
# Also measured: the bare-array update_halo!(grid, A, B) convenience path
# (field.jl/halo_exchange.jl's `as_field`/`to_fields`) already documents that
# wrapping a bare array in a Field on every call has a real, accepted
# per-call cost, distinct from the exchange itself — measured here as a
# stable +448 B/call on top of whatever the pre-wrapped-Field exchange costs
# at that nprocs (0 at nprocs=1, matching test_allocations.jl's own residual
# at nprocs>1). This confirms that documented claim with an actual number
# instead of leaving it as an unverified "small, one-time" assertion.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloArrays

function run_setup_allocation_tests()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    niter = 200

    for _ in 1:5
        topology((0, 0); comm=comm, periods=(true, true))
    end
    total_topo = @allocated for _ in 1:niter
        topology((0, 0); comm=comm, periods=(true, true))
    end
    bytes_topo = total_topo / niter

    topo = topology((0, 0); comm=comm, periods=(true, true))

    for _ in 1:5
        grid(topo, (20, 20))
    end
    total_grid = @allocated for _ in 1:niter
        grid(topo, (20, 20))
    end
    bytes_grid = total_grid / niter

    A = fill(1.0, 20, 20)
    for _ in 1:5
        field(A; halowidths=(1, 1))
    end
    total_field = @allocated for _ in 1:niter
        field(A; halowidths=(1, 1))
    end
    bytes_field = total_field / niter

    g = grid(topo, (20, 20))
    B = fill(2.0, 20, 20)
    for _ in 1:5
        update_halo!(g, A, B)
    end
    total_conv = @allocated for _ in 1:niter
        update_halo!(g, A, B)
    end
    bytes_conv = total_conv / niter

    Ag = fill(Float64(rank), 5, 5)
    Aglobal = rank == 0 ? zeros(5 * topo.dims[1], 5 * topo.dims[2]) : nothing
    for _ in 1:5
        gather!(Ag, Aglobal, topo)
    end
    total_gather = @allocated for _ in 1:niter
        gather!(Ag, Aglobal, topo)
    end
    bytes_gather = total_gather / niter

    if rank == 0
        println("nprocs=$nprocs  topology(): $bytes_topo B/call  grid(): $bytes_grid B/call  " *
                "field(): $bytes_field B/call  update_halo!(bare arrays): $bytes_conv B/call  " *
                "gather!(): $bytes_gather B/call")
    end

    @testset "construction/setup-path allocation bound (nprocs=$nprocs)" begin
        @test bytes_topo < 2000
        @test bytes_grid < 1000
        @test bytes_field == 0
        @test bytes_conv < 3000
        @test bytes_gather < 1000
    end

    MPI.Barrier(comm)
    rank == 0 && println("test_allocations_setup.jl: all tests passed on $nprocs ranks.")
    MPI.Finalize()
end

run_setup_allocation_tests()
