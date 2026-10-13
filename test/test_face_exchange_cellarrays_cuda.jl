using Test, MPI, CUDA, HaloGrids
using CellArrays: CellArrays
using StaticArrays: SMatrix
include("face_cellarray_checks.jl")
CellArrays.@define_CuCellArray()

if !CUDA.functional()
    println("test_face_exchange_cellarrays_cuda.jl: CUDA not functional; skipping.")
    exit(0)
end
CUDA.allowscalar(false)
MPI.Init()
try
    face_cellarray_checks((T, B, shape) -> CuCellArray{T,B}(undef, shape), CUDA.CuArray)
finally
    MPI.Finalize()
end
