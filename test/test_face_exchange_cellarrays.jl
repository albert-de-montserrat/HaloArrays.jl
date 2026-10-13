using Test, MPI, HaloGrids
using CellArrays: CellArrays, CPUCellArray
using StaticArrays: SMatrix
include("face_cellarray_checks.jl")

MPI.Init()
try
    face_cellarray_checks((T, B, shape) -> CPUCellArray{T,B}(undef, shape), identity)
finally
    MPI.Finalize()
end
