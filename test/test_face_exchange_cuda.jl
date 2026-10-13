using Test, MPI, CUDA, HaloGrids
include("face_exchange_checks.jl")

if !CUDA.functional()
    println("test_face_exchange_cuda.jl: CUDA not functional; skipping.")
    exit(0)
end
CUDA.allowscalar(false)
MPI.Init()
try
    face_exchange_checks(CUDA.CuArray, CUDA.synchronize)
    if MPI.has_cuda()
        face_exchange_checks(CUDA.CuArray, CUDA.synchronize; gpu_aware_mpi=true)
    elseif MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("test_face_exchange_cuda.jl: MPI is not CUDA-aware; skipping gpu_aware_mpi=true.")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("test_face_exchange_cuda.jl: all tests passed.")
finally
    MPI.Finalize()
end
