# Functional check: does this MPI + CUDA stack accept device buffers?
#
#   srun -n 2 julia --project=. check_cuda_aware_mpi.jl
#
# Prints the MPI library, what it reports about CUDA support, and then runs three
# device-buffer operations (Allreduce, ring Sendrecv, nonblocking Isend/Irecv!)
# with CuArray buffers, comparing against host results. Exit code is non-zero on
# any mismatch or error, so the SLURM driver can gate the benchmarks on it.
# Set HA_FORCE_CUDAAWARE=1 to run the device-buffer tests even if the library
# does not report CUDA support (may segfault on a non-CUDA-aware MPI).
import Pkg
Pkg.activate(@__DIR__; io = devnull)
using CUDA
import MPI
include(joinpath(@__DIR__, "cuda_aware_common.jl"))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
lrank, ndev = select_device!(comm)

for r in 0:nprocs-1
    if r == rank
        println("rank $rank/$nprocs  host=$(gethostname())  local_rank=$lrank  visible_gpus=$ndev  gpu=$(CUDA.name(CUDA.device()))")
    end
    MPI.Barrier(comm)
end

if rank == 0
    println("MPI library        : ", MPI.identify_implementation())
    println("MPI library version: ", strip(first(split(MPI.MPI_LIBRARY_VERSION_STRING, '\n'))))
    println("MPI.has_cuda()     : ", MPI.has_cuda())
    println("CUDA.jl functional : ", CUDA.functional())
    for k in ("MPICH_GPU_SUPPORT_ENABLED", "MPICH_RDMA_ENABLED_CUDA", "OMPI_MCA_opal_cuda_support",
              "UCX_TLS", "JULIA_CUDA_MEMORY_POOL", "JULIA_CUDA_USE_COMPAT", "HA_FORCE_CUDAAWARE")
        println(rpad(k, 27), ": ", get(ENV, k, "<unset>"))
    end
    flush(stdout)
end

if !cuda_aware_enabled()
    rank == 0 && println("\nMPI does not report CUDA support; skipping device-buffer tests ",
        "(set HA_FORCE_CUDAAWARE=1 to try anyway).")
    MPI.Finalize()
    exit(2)
end

ok = true
check(name, cond) = (rank == 0 && println(rpad(name, 34), cond ? "PASS" : "FAIL"); cond)

n = 1 << 16
try
    # 1. Allreduce on device memory.
    x = CUDA.fill(Float64(rank + 1), n)
    CUDA.synchronize()
    y = MPI.Allreduce(x, +, comm)
    global ok &= check("Allreduce(CuArray)", all(Array(y) .== nprocs * (nprocs + 1) / 2))

    # 2. Ring Sendrecv with device buffers.
    right = mod(rank + 1, nprocs)
    left = mod(rank - 1, nprocs)
    sbuf = CUDA.fill(Float64(rank), n)
    rbuf = CUDA.zeros(Float64, n)
    CUDA.synchronize()
    MPI.Sendrecv!(sbuf, rbuf, comm; dest = right, source = left)
    global ok &= check("Sendrecv!(CuArray) ring", all(Array(rbuf) .== left))

    # 3. Nonblocking Isend/Irecv! with device buffers, both directions.
    sbuf_l = CUDA.fill(Float64(100rank + 1), n)
    sbuf_r = CUDA.fill(Float64(100rank + 2), n)
    rbuf_l = CUDA.zeros(Float64, n)
    rbuf_r = CUDA.zeros(Float64, n)
    CUDA.synchronize()
    reqs = [MPI.Irecv!(rbuf_l, comm; source = left, tag = 1),
            MPI.Irecv!(rbuf_r, comm; source = right, tag = 2),
            MPI.Isend(sbuf_r, comm; dest = right, tag = 1),
            MPI.Isend(sbuf_l, comm; dest = left, tag = 2)]
    MPI.Waitall(reqs)
    global ok &= check("Isend/Irecv!(CuArray) pair",
        all(Array(rbuf_l) .== 100left + 2) && all(Array(rbuf_r) .== 100right + 1))
catch err
    global ok = false
    rank == 0 && (println("\nERROR during device-buffer tests:"); showerror(stdout, err); println())
end

allok = MPI.Allreduce(ok ? 1 : 0, min, comm) == 1
rank == 0 && println("\nCUDA-aware MPI check: ", allok ? "PASSED" : "FAILED")
MPI.Finalize()
exit(allok ? 0 : 1)
