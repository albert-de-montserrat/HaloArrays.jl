# GPU (CUDA) halo-exchange-only benchmark worker for HaloGrids.jl: times
# `update_halo!` on one Float64 `N^3` field over a fully periodic 3D topology,
# so every rank exchanges all six faces. No stencil computation, so the result
# isolates packing, transport, and unpacking. Twin of bench_igg_exchange.jl.
#
# ARGS: local N, iterations, JLD2 output path. `HG_GPU_AWARE_MPI=1` passes
# device buffers to MPI (`grid(...; gpu_aware_mpi=true)`); otherwise halo
# buffers are staged through host memory. Rank 0 saves one result record.
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA
import MPI
using HaloGrids
include(joinpath(@__DIR__, "metrics_common.jl"))

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    output = ARGS[3]
    warmup = max(5, niter ÷ 10)

    MPI.Init()
    comm   = MPI.COMM_WORLD
    rank   = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    gpu_aware = env_flag("HG_GPU_AWARE_MPI")
    t = topology((0, 0, 0); comm=comm, periods=(true, true, true))
    g = grid(t, (N, N, N); gpu_aware_mpi=gpu_aware)
    f = field(CUDA.rand(Float64, N, N, N); halowidths=(1, 1, 1))
    rank == 0 && println("exchange: local=($N, $N, $N) dims=$(t.dims) nprocs=$nprocs gpu_aware_mpi=$gpu_aware")

    for _ in 1:warmup
        update_halo!(g, f)
    end
    CUDA.synchronize()
    MPI.Barrier(comm)

    GC.gc()
    stats = @timed begin
        for _ in 1:niter
            update_halo!(g, f)
        end
        CUDA.synchronize()
    end

    gpu_bytes = CUDA.@allocated for _ in 1:niter
        update_halo!(g, f)
    end
    CUDA.synchronize()

    if rank == 0
        save_result(output, gpu_aware ? "HaloGridsExchangeAware" : "HaloGridsExchange",
            nprocs, N, niter, stats, gpu_bytes, NaN, 0.0;
            transport=gpu_aware ? "device buffers" : "host-staged")
    end
    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
