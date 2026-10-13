using Test, MPI, AMDGPU, HaloGrids
include("face_exchange_checks.jl")

if !AMDGPU.functional()
    println("test_face_exchange_amdgpu.jl: AMDGPU not functional; skipping.")
    exit(0)
end
MPI.Init()
try
    face_exchange_checks(AMDGPU.ROCArray, AMDGPU.synchronize)
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("test_face_exchange_amdgpu.jl: all tests passed.")
finally
    MPI.Finalize()
end
