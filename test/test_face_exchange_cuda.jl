using Test, MPI, CUDA, HaloArrays
include("face_exchange_checks.jl")

if !CUDA.functional()
    println("test_face_exchange_cuda.jl: CUDA not functional; skipping.")
    exit(0)
end
CUDA.allowscalar(false)
MPI.Init()
try
    face_exchange_checks(CUDA.CuArray, CUDA.synchronize)
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("test_face_exchange_cuda.jl: all tests passed.")
finally
    MPI.Finalize()
end
