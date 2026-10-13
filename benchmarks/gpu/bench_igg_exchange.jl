# GPU (CUDA) halo-exchange-only benchmark worker for ImplicitGlobalGrid.jl:
# times `update_halo!` on one Float64 `N^3` array over a fully periodic 3D
# grid. Twin of bench_halogrids_exchange.jl; IGG's default overlap of 2 sends
# and receives the same single-cell strips as a HaloGrids field with
# halowidth 1.
#
# ARGS: local N, iterations, JLD2 output path. `IGG_CUDAAWARE_MPI=1` passes
# device buffers to MPI; otherwise IGG stages through host memory. Rank 0
# saves one result record.
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CUDA                # before ImplicitGlobalGrid, to activate its CUDA support (its own convention)
using ImplicitGlobalGrid
import MPI
include(joinpath(@__DIR__, "metrics_common.jl"))

function main()
    N      = parse(Int, ARGS[1])
    niter  = parse(Int, ARGS[2])
    output = ARGS[3]
    warmup = max(5, niter ÷ 10)

    me, dims, nprocs = init_global_grid(N, N, N; periodx=1, periody=1, periodz=1, quiet=true)
    # IGG reads IGG_CUDAAWARE_MPI itself but never checks that MPI accepts
    # device buffers; fail here instead of crashing or corrupting halos.
    gpu_aware = env_flag("IGG_CUDAAWARE_MPI")
    gpu_aware && !MPI.has_cuda() && error(
        "IGG_CUDAAWARE_MPI=1 but MPI.has_cuda() is false for $(MPI.identify_implementation()); " *
        "with Cray MPICH set JULIA_MPI_HAS_CUDA=true and MPICH_GPU_SUPPORT_ENABLED=1.")
    A = CUDA.rand(Float64, N, N, N)
    me == 0 && println("exchange: local=($N, $N, $N) dims=$(Tuple(dims)) nprocs=$nprocs cudaaware=$gpu_aware")

    for _ in 1:warmup
        update_halo!(A)
    end
    CUDA.synchronize()
    MPI.Barrier(MPI.COMM_WORLD)

    GC.gc()
    stats = @timed begin
        for _ in 1:niter
            update_halo!(A)
        end
        CUDA.synchronize()
    end

    gpu_bytes = CUDA.@allocated for _ in 1:niter
        update_halo!(A)
    end
    CUDA.synchronize()

    if me == 0
        save_result(output, gpu_aware ? "ImplicitGlobalGridExchangeAware" : "ImplicitGlobalGridExchange",
            nprocs, N, niter, stats, gpu_bytes, NaN, 0.0;
            transport=gpu_aware ? "device buffers" : "host-staged")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    finalize_global_grid()
end

main()
