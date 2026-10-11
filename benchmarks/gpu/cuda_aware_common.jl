# Shared helpers for the CUDA-aware-MPI probes in benchmarks/gpu/ (check_cuda_aware_mpi.jl,
# bench_pingpong.jl, bench_halo_cudaaware.jl). `include`d after `using CUDA, MPI`.
#
# These scripts talk to MPI.jl directly (no HaloArrays) except for the
# `haloarrays` mode of bench_halo_cudaaware.jl, which times the package's current
# host-staged CUDA exchange as the baseline. HaloArrays itself has no CUDA-aware
# transport yet, so the point of these probes is to measure what such a path
# would gain on the target cluster.

# Bind this rank to a GPU: round-robin over the devices visible to the node-local
# ranks. Under `--gpus-per-task=1` each rank sees one device and this is a no-op.
function select_device!(comm)
    lcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    lrank = MPI.Comm_rank(lcomm)
    ndev = length(CUDA.devices())
    CUDA.device!(lrank % ndev)
    return lrank, ndev
end

# `MPI.has_cuda()` only reports what the library announces (e.g. OpenMPI's
# MPIX_Query_cuda_support, or MPICH_GPU_SUPPORT_ENABLED for Cray MPICH). Passing
# a device pointer to a non-CUDA-aware MPI usually segfaults, so device-buffer
# modes are skipped unless the library reports support or the user forces it with
# HA_FORCE_CUDAAWARE=1.
function cuda_aware_enabled()
    get(ENV, "HA_FORCE_CUDAAWARE", "0") == "1" && return true
    return MPI.has_cuda()
end

# Append one CSV row from rank 0; write the header if the file is new.
function append_csv(path, header, row)
    isfile(path) || open(io -> println(io, header), path, "w")
    open(io -> println(io, join(row, ",")), path, "a")
    return nothing
end

# Maximum over ranks of a per-rank wall time (the slowest rank defines a step).
max_over_ranks(comm, t::Real) = MPI.Allreduce(Float64(t), max, comm)
