# Preflight for CUDA-aware MPI: every rank exchanges a CuArray with its ring
# neighbors and verifies the received values. Errors (non-zero exit) if MPI.jl
# does not report CUDA support or the data arrives wrong. Run with >= 2 ranks:
#   srun -n 2 julia --project=. check_cuda_aware_mpi.jl
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA
import MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
nprocs >= 2 || error("check_cuda_aware_mpi.jl needs at least 2 ranks; got $nprocs")
MPI.has_cuda() || error(
    "MPI.has_cuda() is false for $(MPI.identify_implementation()); with Cray MPICH set " *
    "JULIA_MPI_HAS_CUDA=true and MPICH_GPU_SUPPORT_ENABLED=1.")

n = 1 << 20  # 8 MiB of Float64, large enough for the rendezvous protocol
right = mod(rank + 1, nprocs)
left = mod(rank - 1, nprocs)
src = CUDA.fill(Float64(rank), n)
dst = CUDA.zeros(Float64, n)
MPI.Sendrecv!(src, dst, comm; dest=right, source=left)
all(Array(dst) .== left) || error("rank $rank: CUDA-aware Sendrecv delivered wrong data")
MPI.Barrier(comm)
rank == 0 && println("CUDA-aware MPI OK on $nprocs ranks: $(MPI.identify_implementation()), ",
    "device $(CUDA.name(CUDA.device()))")
MPI.Finalize()
