# Shared HDF5/MPI extension helpers for HaloArrays.
#
# Geometry note: a Field's local array already includes its own halo padding
# (size(field.data) == grid.nlocal exactly, see grid.jl/coordinates.jl), and
# on a non-periodic domain boundary the outer halowidths-wide strip is *not*
# halo at all (it's never touched by update_halo!, since there is no
# neighbor to exchange with) — it is genuine, uniquely-owned domain data.
# `local_interior_range` below trims a side only when a real neighbor exists
# there, so the region written to disk is exactly the set of cells that are
# both non-redundant (not a copy of a neighbor's data) and complete (every
# genuine domain cell is owned by exactly one rank).

using HaloArrays
import HDF5
import MPI

const HDF5_IO_DEFAULT_GROUP = "/"

"""
    HDF5IOError

Raised for save_hdf5/load_hdf5! usage and schema problems specific to this
extension (as opposed to a plain `error(...)` from validation shared with the
rest of the package, e.g. `HaloArrays.check_fields`).
"""
struct HDF5IOError <: Exception
    msg::String
end

Base.showerror(io::IO, err::HDF5IOError) = print(io, "HDF5IOError: ", err.msg)

"""
    require_cpu_array(field, position)

Reject a field whose backing array is not host-addressable strided memory.
`StridedArray` (covering `Array`, contiguous `SubArray`/`ReshapedArray`/
`PermutedDimsArray` of one, ...) is used as the CPU/device boundary rather
than an explicit `isa Array` check so ordinary CPU views still work; GPU
array types (`CuArray`, `ROCArray`) deliberately do not subtype
`StridedArray` (it implies a valid CPU `pointer(A)`), so this stays correct
and dependency-free without importing CUDA/AMDGPU here. Stage a device field
to the host first, e.g. `field(Array(f.data); halowidths=f.halowidths)`.
"""
function require_cpu_array(field::HaloArrays.Field, position::Integer)
    field.data isa StridedArray || throw(HDF5IOError(
        "field at position $position wraps a $(typeof(field.data)), which is not a CPU " *
        "StridedArray; HDF5 I/O only ever touches host memory. Stage a device array to " *
        "the host first, e.g. `field(Array(f.data); halowidths=f.halowidths)`."))
    return nothing
end

"""
    local_interior_range(grid, field, dim) :: UnitRange{Int}

The range, into `field.data`'s dimension `dim` (1-based, halo-inclusive
indexing), of the cells this rank owns and that `save_hdf5`/`load_hdf5!`
read or write — i.e. `field.data`'s full extent in that dimension, minus
`field.halowidths[dim]` on whichever side(s) have a real neighbor.
"""
function local_interior_range(grid::HaloArrays.Grid, field::HaloArrays.Field, dim::Integer)
    hw = field.halowidths[dim]
    field.overlaps[dim] == 2hw || error(
        "save_hdf5/load_hdf5! only support fields with the default overlaps " *
        "(2*halowidths); got overlaps=$(field.overlaps), halowidths=$(field.halowidths).")
    lo_neighbor, hi_neighbor = grid.topology.neighbors[dim]
    lo_trim = HaloArrays.has_neighbor(lo_neighbor) ? hw : 0
    hi_trim = HaloArrays.has_neighbor(hi_neighbor) ? hw : 0
    nlocal_d = grid.nlocal[dim]
    return (1 + lo_trim):(nlocal_d - hi_trim)
end

"""
    local_and_global_ranges(grid, field) :: (local_ranges, global_ranges)

Per-dimension `(local, global)` `UnitRange` pairs describing this rank's
owned block: `local_ranges` indexes `field.data` directly, and
`global_ranges` (1-based) indexes the file's global, `nglobal`-shaped
dataset for the same field. Reuses [`iglobal`](@ref) (rather than
re-deriving the rank-offset arithmetic) to place the block, so this stays
consistent with the rest of the package's coordinate accounting by
construction.
"""
function local_and_global_ranges(grid::HaloArrays.Grid{N}, field::HaloArrays.Field{T,N}) where {T,N}
    local_ranges = ntuple(d -> local_interior_range(grid, field, d), N)
    global_ranges = ntuple(N) do d
        start = HaloArrays.iglobal(grid, field, d, first(local_ranges[d]); wrap_periodic = false)
        start:(start + length(local_ranges[d]) - 1)
    end
    return local_ranges, global_ranges
end

"""
    default_field_names(M) :: NTuple{M,String}

Positional fallback names (`"field1"`, `"field2"`, ...) used by both
`save_hdf5` and `load_hdf5!` when the caller doesn't pass `names` explicitly,
so a save/load pair that both omit `names` agree on the dataset labels by
construction.
"""
default_field_names(M::Integer) = ntuple(i -> "field$i", M)
