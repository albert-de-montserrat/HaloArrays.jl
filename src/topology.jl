export topology, has_neighbor, Topology

"""
    Topology{N}

An explicit, independently-constructed Cartesian MPI process topology. No global
mutable state: every `Topology` (and every `Grid` built on top of it) is a plain
value threaded explicitly through calls, which is what makes multiple concurrent
grids of different sizes work by construction rather than by careful discipline.

Fields:
- `comm::MPI.Comm`: the Cartesian communicator (a fresh communicator derived from
  the input `comm`, never shared with another `Topology`, even one built from the
  same parent communicator).
- `dims::NTuple{N,Int}`: number of ranks along each dimension.
- `periods::NTuple{N,Bool}`: periodicity of each dimension.
- `rank::Int`: this process's rank within `comm`.
- `coords::NTuple{N,Int}`: this process's 0-based Cartesian coordinates.
- `neighbors::NTuple{N,NTuple{2,Int}}`: `neighbors[d] = (lo, hi)`, the ranks of
  the lower- and upper-side neighbor in dimension `d`, or `MPI.PROC_NULL` if there
  is none (non-periodic domain boundary).
"""
struct Topology{N}
    comm::MPI.Comm
    dims::NTuple{N,Int}
    periods::NTuple{N,Bool}
    rank::Int
    coords::NTuple{N,Int}
    neighbors::NTuple{N,NTuple{2,Int}}
end

Base.ndims(::Topology{N}) where {N} = N

"""
    topology(dims::NTuple{N,Integer}; comm=MPI.COMM_WORLD, periods=ntuple(_->false,N),
             disp=1, reorder=true) :: Topology{N}

Build a Cartesian MPI process `Topology`. `dims` gives the desired number of
processes along each dimension; an entry of `0` lets MPI choose (via
`MPI.Dims_create`) a balanced size for that dimension. Only 2D and 3D topologies
are supported.

Each call creates its own new communicator (via `MPI.Cart_create`), even when
given the same parent `comm` as another call — this is the actual mechanism
that lets multiple `Topology`/`Grid` instances coexist without cross-talk.
"""
function topology(dims::NTuple{N,Integer};
                   comm::MPI.Comm = MPI.COMM_WORLD,
                   periods::NTuple{N,Bool} = ntuple(_ -> false, N),
                   disp::Integer = 1,
                   reorder::Bool = true) where {N}
    N in (2, 3) || error("HaloArrays supports only 2D and 3D topologies; got N=$N from dims=$dims.")
    any(<(0), dims) && error("topology dims cannot be negative; got dims=$dims.")
    MPI.Initialized() || MPI.Init()

    nprocs = MPI.Comm_size(comm)
    newdims = Tuple(MPI.Dims_create(nprocs, collect(Int, dims)))
    comm_cart = MPI.Cart_create(comm, collect(Int, newdims); periodic=collect(periods), reorder=reorder)
    rank = MPI.Comm_rank(comm_cart)
    coords = Tuple(Int.(MPI.Cart_coords(comm_cart)))
    neighbors = ntuple(d -> Int.(MPI.Cart_shift(comm_cart, d - 1, disp)), N)

    return Topology{N}(comm_cart, newdims, periods, rank, coords, neighbors)
end

"""
    has_neighbor(rank::Integer) -> Bool

Whether `rank` (as returned in a `Topology`'s `neighbors`) denotes a real
neighbor, i.e. is not `MPI.PROC_NULL`.
"""
has_neighbor(rank::Integer) = rank != MPI.PROC_NULL

function Base.show(io::IO, ::MIME"text/plain", t::Topology{N}) where {N}
    print(io, "Topology{$N}(dims=$(t.dims), periods=$(t.periods), rank=$(t.rank), coords=$(t.coords))")
end
