# Parallel VTK (pvtk) I/O for the WriteVTK extension.
#
# No shared-file/parallel-build story is needed here, unlike src/HDF5Ext/io.jl:
# in WriteVTK.jl's pvtk_grid model, every rank writes its own independent
# serial .vti file, and only one rank (ismain) additionally writes a small
# .pvti header that just lists pointers to the per-rank pieces. There is no
# shared file and therefore nothing to serialize — every rank's
# pvtk_grid(...) do ... end block is a genuinely independent filesystem
# write. save_pvtk is still collective in the sense that every rank in the
# grid's topology must call it together (gather_extents's MPI.Allgather
# blocks otherwise), same as save_hdf5.

using HaloArrays
using WriteVTK
using MPI

function HaloArrays.save_pvtk(grid::HaloArrays.Grid{N}, fields::NTuple{M,HaloArrays.Field}, path::AbstractString;
                               names::NTuple{M,AbstractString} = default_field_names(M),
                               dx::NTuple{N,Real} = ntuple(_ -> 1.0, N)) where {N,M}
    check_pvtk_batch(grid, fields, names, dx)

    reference = fields[1]
    coords = ntuple(d -> local_coordinate_range(grid, reference, d, dx[d]), N)
    extents = gather_extents(grid, reference)
    part = grid.topology.rank + 1

    saved = pvtk_grid(path, coords...; part = part, extents = extents) do pvtk
        for i in 1:M
            pvtk[names[i]] = fields[i].data
        end
    end
    return saved
end

function HaloArrays.save_pvtk(grid::HaloArrays.Grid{N}, field::HaloArrays.Field{T,N}, path::AbstractString;
                               name::AbstractString = "field", kwargs...) where {T,N}
    return HaloArrays.save_pvtk(grid, (field,), path; names = (name,), kwargs...)
end
