using Test, MPI, HaloArrays
include("face_exchange_checks.jl")

MPI.Init()
try
    face_exchange_checks(identity, () -> nothing)
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Comm_rank(MPI.COMM_WORLD) == 0 && println("test_face_exchange.jl: all tests passed.")
finally
    MPI.Finalize()
end
