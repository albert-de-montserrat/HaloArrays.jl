export save_hdf5, load_hdf5!

const HDF5EXT_INSTALL_MSG = "requires the optional HDF5 extension: run `using HDF5` " *
    "(after `]add HDF5`) to enable HaloArraysHDF5Ext."

"""
    save_hdf5(grid::Grid, field::Field, path; name="field", group="/", mode="w")
    save_hdf5(grid::Grid, fields::NTuple{M,Field}, path; names, group="/", mode="w")

Write one or more [`Field`](@ref)s' rank-local interior data (halo cells that
duplicate a neighbor's data are never stored) to an HDF5 file, using the
`Grid`'s Cartesian topology to place each rank's block at its correct offset
in a single shared, per-field global dataset. See [`load_hdf5!`](@ref) for
reading it back.

This is a stub method: calling it without `HDF5` loaded raises this error
rather than a bare `MethodError`. Loading `HDF5` (`using HDF5`) resolves the
call to `HaloArraysHDF5Ext`'s implementation via ordinary multiple dispatch —
the same "declare a stub in core, add methods from the extension" pattern as
`is_cuarray`/`is_rocarray` in `backend.jl` — so no `Base.get_extension`
lookup is needed at the call site.

See the "MPI/HDF5 file I/O" guide for the on-disk layout, the interior/halo
storage convention, and why every rank must call this together (there is no
parallel-HDF5 build to make a true collective write meaningful here, so ranks
instead take turns under an internal `MPI.Barrier`-synchronized schedule).
"""
save_hdf5(::Grid, ::Union{Field,Tuple}, ::AbstractString; kwargs...) =
    error("save_hdf5 $HDF5EXT_INSTALL_MSG")

"""
    load_hdf5!(grid::Grid, field::Field, path; name="field", group="/")
    load_hdf5!(grid::Grid, fields::NTuple{M,Field}, path; names, group="/")

Read a file written by [`save_hdf5`](@ref) back into the given `Field`(s),
mutating their arrays in place. Halo cells that have a real neighbor are
*not* restored (they were never saved) — call [`update_halo!`](@ref)
afterward to repopulate them.

This is a stub method, resolved the same way as [`save_hdf5`](@ref) once
`using HDF5` loads `HaloArraysHDF5Ext`; see its docstring and the "MPI/HDF5
file I/O" guide for the full rationale.
"""
load_hdf5!(::Grid, ::Union{Field,Tuple}, ::AbstractString; kwargs...) =
    error("load_hdf5! $HDF5EXT_INSTALL_MSG")
