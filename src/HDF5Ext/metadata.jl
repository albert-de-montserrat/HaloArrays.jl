# On-disk metadata schema for the MPI/HDF5 extension.
#
# Each field's dataset carries its own attribute bundle (rather than the file
# or group carrying one shared bundle), since a single file can hold several
# fields with independent halowidths/dtype: metadata must be looked up
# per-dataset, not per-group.

using HaloGrids
using HDF5

const HDF5_IO_SCHEMA_VERSION = "0.2.0"

"""
    HDF5Metadata

The on-disk metadata bundle for one field's dataset. Recorded as HDF5
attributes on that dataset (see [`write_metadata_attrs!`](@ref)) and checked
against the requested `Grid`/`Field` pair (see [`validate_metadata!`](@ref))
before any payload is read.

- `schema_version`: file format version tag; bumped whenever the storage
  convention changes (0.2.0: per-field attributes on the dataset, and the
  payload excludes halo cells that duplicate a neighbor's data — earlier,
  unreleased 0.1.0 files stored the full halo-inclusive local block as
  group-level attributes and are not compatible).
- `topology_dims`: the Cartesian MPI process dimensions.
- `nlocal`: the local per-rank array shape *including* halo padding, i.e.
  `size(field.data)` — used to validate that the file's rank decomposition
  matches the grid requesting the load, since the interior/halo split
  depends on it.
- `nglobal`: the global domain shape seen by this field (see
  [`nglobal`](@ref)) — the saved dataset's shape.
- `halowidths`: the field's halo width tuple.
- `field_name`: the dataset's label.
- `dtype_name`: `string(eltype(field.data))`.
"""
struct HDF5Metadata
    schema_version::String
    topology_dims::Tuple
    nlocal::Tuple
    nglobal::Tuple
    halowidths::Tuple
    field_name::String
    dtype_name::String
end

"""
    metadata_from_grid(grid::Grid{N}, field::Field{T,N}, field_name::AbstractString)

Build the `HDF5Metadata` bundle for `field` as it will be (or was) written
under `field_name`.
"""
function metadata_from_grid(grid::HaloGrids.Grid{N}, field::HaloGrids.Field{T,N}, field_name::AbstractString) where {T,N}
    nglobal_sizes = ntuple(d -> HaloGrids.nglobal(grid, field, d), N)
    return HDF5Metadata(
        HDF5_IO_SCHEMA_VERSION,
        Tuple(grid.topology.dims),
        Tuple(grid.nlocal),
        nglobal_sizes,
        Tuple(field.halowidths),
        String(field_name),
        string(T),
    )
end

"""
    validate_metadata!(meta::HDF5Metadata, grid::Grid{N}, field::Field{T,N}; repartition=false)

Check a dataset's saved metadata against the `Grid`/`Field` pair requesting
the load, before any payload is touched. Raises a plain `error(...)`
naming the saved and requested values for the first mismatch found.
With `repartition=true`, allow different topology dimensions and local sizes,
while requiring the same global shape, halo widths, type, and periodicity.
"""
function validate_metadata!(meta::HDF5Metadata, grid::HaloGrids.Grid{N}, field::HaloGrids.Field{T,N};
                            repartition::Bool = false) where {T,N}
    meta.schema_version == HDF5_IO_SCHEMA_VERSION || error(
        "HDF5 schema mismatch for field \"$(meta.field_name)\": file has " *
        "$(meta.schema_version), this extension expects $(HDF5_IO_SCHEMA_VERSION).")
    (repartition || Tuple(meta.topology_dims) == Tuple(grid.topology.dims)) || error(
        "topology_dims mismatch for field \"$(meta.field_name)\": file has " *
        "$(meta.topology_dims), grid has $(grid.topology.dims).")
    (repartition || Tuple(meta.nlocal) == Tuple(grid.nlocal)) || error(
        "nlocal mismatch for field \"$(meta.field_name)\": file has $(meta.nlocal), " *
        "grid has $(grid.nlocal).")
    Tuple(meta.halowidths) == Tuple(field.halowidths) || error(
        "halowidths mismatch for field \"$(meta.field_name)\": file has " *
        "$(meta.halowidths), field has $(field.halowidths).")
    nglobal_sizes = ntuple(d -> HaloGrids.nglobal(grid, field, d), N)
    Tuple(meta.nglobal) == nglobal_sizes || error(
        "nglobal mismatch for field \"$(meta.field_name)\": file has $(meta.nglobal), " *
        "grid/field imply $(nglobal_sizes).")
    meta.dtype_name == string(T) || error(
        "dtype mismatch for field \"$(meta.field_name)\": file has \"$(meta.dtype_name)\", " *
        "field eltype is $T.")
    if repartition
        # Schema 0.2.0 already encodes periodicity unambiguously: only
        # nonperiodic dimensions retain the two physical boundary strips.
        # Infer it from the saved geometry so existing files remain usable.
        all(values -> length(values) == N,
            (meta.topology_dims, meta.nlocal, meta.nglobal, meta.halowidths)) ||
            error("HDF5 dimensionality mismatch for field \"$(meta.field_name)\".")
        saved_periods = ntuple(N) do d
            hw = meta.halowidths[d]
            dims = meta.topology_dims[d]
            nl = meta.nlocal[d]
            dims >= 1 && hw >= 1 && nl >= 2 * hw ||
                error("Invalid saved geometry for field \"$(meta.field_name)\", dimension $d.")
            core = dims * (nl - 2 * hw)
            if meta.nglobal[d] == core
                true
            elseif meta.nglobal[d] == core + 2 * hw
                false
            else
                error("Inconsistent saved geometry for field \"$(meta.field_name)\", dimension $d.")
            end
        end
        saved_periods == grid.topology.periods || error(
            "periods mismatch for field \"$(meta.field_name)\": file implies " *
            "$(saved_periods), grid has $(grid.topology.periods).")
    end
    return nothing
end

"""
    write_metadata_attrs!(dset, meta::HDF5Metadata)

Write `meta`'s fields as HDF5 attributes on `dset`.
"""
function write_metadata_attrs!(dset, meta::HDF5Metadata)
    a = HDF5.attrs(dset)
    a["schema_version"] = meta.schema_version
    a["topology_dims"] = collect(Int, meta.topology_dims)
    a["nlocal"] = collect(Int, meta.nlocal)
    a["nglobal"] = collect(Int, meta.nglobal)
    a["halowidths"] = collect(Int, meta.halowidths)
    a["field_name"] = meta.field_name
    a["dtype_name"] = meta.dtype_name
    return nothing
end

"""
    metadata_from_dataset(dset) :: HDF5Metadata

Read back the attribute bundle [`write_metadata_attrs!`](@ref) wrote onto
`dset`.
"""
function metadata_from_dataset(dset)
    a = HDF5.attrs(dset)
    return HDF5Metadata(
        a["schema_version"],
        Tuple(Int.(a["topology_dims"])),
        Tuple(Int.(a["nlocal"])),
        Tuple(Int.(a["nglobal"])),
        Tuple(Int.(a["halowidths"])),
        a["field_name"],
        a["dtype_name"],
    )
end
