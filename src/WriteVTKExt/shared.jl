# Shared WriteVTK/MPI extension helpers for HaloArrays.
#
# Geometry note: unlike src/HDF5Ext/shared.jl's local_interior_range (which
# trims halo cells that duplicate a neighbor's data), this extension writes
# each rank's *entire* local array, halo included. Parallel VTK requires
# neighboring pieces' extents to overlap at shared boundaries — per
# WriteVTK's docs, "without it the full data cannot be visualised in
# ParaView" — and since HaloArrays already keeps halo cells exactly in sync
# with each neighbor's real data via update_halo!, writing them is what
# makes that overlap correct, not an extra step.

using HaloArrays
import WriteVTK: pvtk_grid
import MPI

"""
    PVTKIOError

Raised for save_pvtk usage and geometry problems specific to this extension
(as opposed to a plain `error(...)` from validation shared with the rest of
the package, e.g. `HaloArrays.check_fields`).
"""
struct PVTKIOError <: Exception
    msg::String
end

Base.showerror(io::IO, err::PVTKIOError) = print(io, "PVTKIOError: ", err.msg)

"""
    require_cpu_array(field, position)

Reject a field whose backing array is not host-addressable strided memory.
Same rationale and pattern as `HDF5Ext.require_cpu_array` — duplicated here
rather than imported across extensions, so this extension stays
independently loadable without `HDF5` installed (and vice versa).
"""
function require_cpu_array(field::HaloArrays.Field, position::Integer)
    field.data isa StridedArray || throw(PVTKIOError(
        "field at position $position wraps a $(typeof(field.data)), which is not a CPU " *
        "StridedArray; save_pvtk only ever touches host memory. Stage a device array to " *
        "the host first, e.g. `field(Array(f.data); halowidths=f.halowidths)`."))
    return nothing
end

"""
    local_and_global_extents(grid, field) :: (local_range, global_extent)

Unlike `HDF5Ext.local_and_global_ranges`, this does *not* trim halo cells:
`local_range` is always `1:grid.nlocal[d]` (the field's entire local array,
halo included), because parallel VTK pieces are required to overlap at
shared-neighbor boundaries. `global_extent` is the corresponding 1-based
global point-index range, computed the same way `save_hdf5` computes its
global offset — via `iglobal(grid, field, d, i; wrap_periodic=false)` at the
two ends of the local array — just without the interior trim first.

Neighboring ranks' extents overlap by exactly `2*field.halowidths[d]`
points in every non-periodic dimension (periodic topologies are rejected
before this is ever called — see `check_pvtk_batch`).
"""
function local_and_global_extents(grid::HaloArrays.Grid{N}, field::HaloArrays.Field{T,N}) where {T,N}
    local_range = ntuple(d -> 1:grid.nlocal[d], N)
    global_extent = ntuple(N) do d
        lo = HaloArrays.iglobal(grid, field, d, 1; wrap_periodic = false)
        hi = HaloArrays.iglobal(grid, field, d, grid.nlocal[d]; wrap_periodic = false)
        lo:hi
    end
    return local_range, global_extent
end

"""
    local_coordinate_range(grid, field, dim, dx) :: AbstractRange{Float64}

This rank's local coordinate vector in dimension `dim`, spanning its entire
local array (halo included) at spacing `dx`. Built directly from
[`coord_g`](@ref) at local index `1` rather than by slicing a materialized
global coordinate vector — algebraically identical (both `coord_g` and
`iglobal` are affine in the local index), but avoids allocating anything
global-sized. Always an `AbstractRange`, so WriteVTK always infers the
uniform-spacing "image data" (`.vti`/`.pvti`) format — `HaloArrays.Grid` has
no non-uniform-spacing concept to represent as a rectilinear/structured grid
instead.
"""
function local_coordinate_range(grid::HaloArrays.Grid{N}, field::HaloArrays.Field{T,N}, dim::Integer, dx::Real) where {T,N}
    x0 = HaloArrays.coord_g(grid, field, dim, 1, dx)
    return range(x0; step = Float64(dx), length = grid.nlocal[dim])
end

"""
    gather_extents(grid, field) :: Vector{NTuple{N,UnitRange{Int}}}

Collectively gather every rank's `global_extent` (from
[`local_and_global_extents`](@ref)) into one array, identical on every rank
and ordered by rank — the shape `pvtk_grid`'s `extents` keyword requires for
a structured/image-data parallel write ("the `extents` argument must be the
same for all processes"). Uses a plain `MPI.Allgather` of fixed-size
`(lo,hi)` integer pairs, not [`gather!`](@ref) (which is a full-array data
gather assuming an exact, non-overlapping tiling — the wrong tool here,
since pvtk extents deliberately overlap).
"""
function gather_extents(grid::HaloArrays.Grid{N}, field::HaloArrays.Field{T,N}) where {T,N}
    _, global_extent = local_and_global_extents(grid, field)
    local_flat = Vector{Int}(undef, 2N)
    for d in 1:N
        local_flat[2d - 1] = first(global_extent[d])
        local_flat[2d] = last(global_extent[d])
    end
    comm = grid.topology.comm
    nprocs = MPI.Comm_size(comm)
    flat = MPI.Allgather(local_flat, comm)
    return [ntuple(d -> flat[(p - 1) * 2N + 2d - 1]:flat[(p - 1) * 2N + 2d], N) for p in 1:nprocs]
end

"""
    default_field_names(M) :: NTuple{M,String}

Positional fallback names (`"field1"`, `"field2"`, ...) used by `save_pvtk`
when the caller doesn't pass `names` explicitly. Mirrors
`HDF5Ext.default_field_names`.
"""
default_field_names(M::Integer) = ntuple(i -> "field$i", M)

"""
    require_uniform_halowidths(fields)

Reject a batch whose fields don't all share the same `halowidths`. Every
field passed to one `save_pvtk` call is written against the *same* shared
coordinate grid (one `pvtk_grid` call, one `extents` array), but
`iglobal`/`nglobal` — and therefore the global position a given local index
maps to — depend on a field's own `halowidths` (see their docstrings in
`coordinates.jl`): two fields with different halowidths on the same `Grid`
imply different global domain sizes. So unlike this package's other
multi-field entry points, mixed halowidths cannot be supported here without
silently writing at least one field's data at the wrong coordinates.
"""
function require_uniform_halowidths(fields::NTuple{M,HaloArrays.Field}) where {M}
    hw1 = fields[1].halowidths
    for i in 2:M
        fields[i].halowidths == hw1 || throw(PVTKIOError(
            "save_pvtk requires every field in one call to share the same halowidths " *
            "(field 1 has halowidths=$hw1, field $i has halowidths=$(fields[i].halowidths)): " *
            "mixed halo widths imply different global geometry per field (see nglobal/iglobal), " *
            "which cannot be represented by the single shared coordinate grid one pvtk_grid file " *
            "set uses. Save fields with different halowidths in separate save_pvtk calls instead."))
    end
    return nothing
end

"""
    check_pvtk_batch(grid, fields, names, dx)

Validate a `save_pvtk` batch before any file is touched: dimensionality and
shape (via `HaloArrays.check_fields`), unique names, CPU-only arrays,
uniform halowidths across the batch, a valid `dx`, and — critically — that
the grid's topology is not periodic in any dimension.
"""
function check_pvtk_batch(grid::HaloArrays.Grid{N}, fields::NTuple{M,HaloArrays.Field},
                           names::NTuple{M,AbstractString}, dx::NTuple{N,Real}) where {N,M}
    M >= 1 || throw(PVTKIOError("at least one field is required."))
    allunique(names) || throw(PVTKIOError(
        "field names must be unique within one save_pvtk call; got $(names)."))
    any(grid.topology.periods) && throw(PVTKIOError(
        "save_pvtk does not support periodic topologies (periods=$(grid.topology.periods)): " *
        "parallel VTK extents are a flat, non-wrapping index space, so a periodic dimension's " *
        "wrap-around halo cannot be written as a contiguous extension of the same coordinate " *
        "axis without producing spatially folded (incorrect) geometry."))
    length(dx) == N || throw(PVTKIOError(
        "dx must have length $N (one entry per grid dimension); got $(dx)."))
    all(>(0), dx) || throw(PVTKIOError("dx entries must all be > 0; got dx=$dx."))
    for (i, f) in enumerate(fields)
        ndims(f) == N || throw(PVTKIOError(
            "field at position $i has $(ndims(f)) dimensions, but the grid has $N; " *
            "save_pvtk requires every field to match the grid's dimensionality exactly."))
        require_cpu_array(f, i)
    end
    require_uniform_halowidths(fields)
    HaloArrays.check_fields(grid, fields)
    return nothing
end
