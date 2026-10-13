# Thin glue module mirroring the repository's extension pattern:
# the real optional I/O implementation lives in src/WriteVTKExt/ rather than
# inside the extension file itself. save_pvtk is declared as a stub function
# in src/PVTKIO.jl and given concrete methods here (via
# `HaloGrids.save_pvtk(...) = ...`) — see that file for why, and
# ext/HaloGridsHDF5Ext.jl for the same pattern used by save_hdf5/load_hdf5!.
module HaloGridsWriteVTKExt
    include(joinpath(@__DIR__, "..", "src", "WriteVTKExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "WriteVTKExt", "io.jl"))
end
