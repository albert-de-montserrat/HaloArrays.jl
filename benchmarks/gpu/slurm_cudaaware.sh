#!/usr/bin/env bash
# CUDA-aware MPI weak-scaling comparison of HaloGrids.jl and ImplicitGlobalGrid.jl
# on GPUs. One rank per GPU, fixed local N^3 per rank, increasing rank counts
# across nodes. For every rank count and repetition it runs:
#   bench_halogrids_exchange.jl            bench_igg_exchange.jl              (update_halo! only)
#   bench_halogrids_diffusion3d.jl         bench_igg_diffusion3d.jl           (blocking stencil step)
#   bench_halogrids_diffusion3d_hide.jl    bench_igg_diffusion3d_hide.jl      (communication hiding)
#
# MODES selects the transport: `aware` exports HG_GPU_AWARE_MPI=1 and
# IGG_CUDAAWARE_MPI=1 (device buffers passed to MPI); `staged` sets both to 0
# (halo buffers copied through host memory). A 2-rank preflight
# (check_cuda_aware_mpi.jl) aborts the job if MPI does not accept device buffers.
#
# Defaults target CSCS Alps GH200 (4 GPUs per node, Cray MPICH with GTL, the
# `julia/25.5:v1` uenv). Adjust the #SBATCH lines and UENV for other sites.
# Submit:   sbatch slurm_cudaaware.sh
# Override: N=256 NITER=200 REPS=5 MODES="aware staged" NTASKS="1 4 16" sbatch slurm_cudaaware.sh
# Plot:     julia --project=. plot_scaling.jl results_cudaaware_<jobid>.jld2
#
# benchmarks/gpu/Project.toml takes ImplicitGlobalGrid from ../../../ImplicitGlobalGrid.jl,
# so that checkout must exist next to HaloGrids.jl on the cluster.
#SBATCH --job-name=hg-vs-igg-cudaaware
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_cudaaware_%j.out

set -euo pipefail
cd "$(dirname "$0")"

export MPICH_GPU_SUPPORT_ENABLED=1     # Cray MPICH: accept device pointers (GTL)
export JULIA_MPI_HAS_CUDA=true         # tell MPI.jl that this MPICH accepts CUDA buffers
export JULIA_CUDA_MEMORY_POOL=none     # pool allocations cannot be shared over CUDA IPC
export JULIA_CUDA_USE_COMPAT=false
UENV="${UENV:-julia/25.5:v1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
N="${N:-128}"
NITER="${NITER:-150}"
REPS="${REPS:-3}"
MODES="${MODES:-aware}"
NTASKS="${NTASKS:-1 2 4 8 16}"
OUT="${OUT:-results_cudaaware_${SLURM_JOB_ID:-manual}.jld2}"
MAXTASKS="${SLURM_NTASKS:-$GPUS_PER_NODE}"

run() {  # run <ntasks> <command...>
    local n=$1; shift
    local per_node=$(( n < GPUS_PER_NODE ? n : GPUS_PER_NODE ))
    local nodes=$(( (n + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    srun -N "$nodes" -n "$n" --ntasks-per-node="$per_node" --gpus-per-task=1 \
        --uenv "$UENV" --view=juliaup "$@"
}

echo "Instantiating and precompiling benchmarks/gpu environment..." >&2
run 1 julia --project=. --startup-file=no -e \
    'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA, MPI, HaloGrids, ImplicitGlobalGrid, ParallelStencil'

echo "Preflight: CUDA-aware MPI across GPUs..." >&2
run 2 julia --project=. --startup-file=no check_cuda_aware_mpi.jl
if (( MAXTASKS > GPUS_PER_NODE )); then
    # Two ranks on different nodes: exercises the inter-node RDMA path.
    srun -N 2 -n 2 --ntasks-per-node=1 --gpus-per-task=1 --uenv "$UENV" --view=juliaup \
        julia --project=. --startup-file=no check_cuda_aware_mpi.jl
fi

SCRIPTS=(
    bench_halogrids_exchange.jl bench_igg_exchange.jl
    bench_halogrids_diffusion3d.jl bench_igg_diffusion3d.jl
    bench_halogrids_diffusion3d_hide.jl bench_igg_diffusion3d_hide.jl
)

for mode in $MODES; do
    case "$mode" in
        aware)  export HG_GPU_AWARE_MPI=1 IGG_CUDAAWARE_MPI=1 ;;
        staged) export HG_GPU_AWARE_MPI=0 IGG_CUDAAWARE_MPI=0 ;;
        *) echo "unknown mode '$mode' (expected aware or staged)" >&2; exit 1 ;;
    esac
    for n in $NTASKS; do
        if (( n > MAXTASKS )); then
            echo "-- skip ranks=$n: allocation has $MAXTASKS tasks --" >&2
            continue
        fi
        for rep in $(seq 1 "$REPS"); do
            export BENCH_REP=$rep
            for script in "${SCRIPTS[@]}"; do
                echo "-- mode=$mode ranks=$n rep=$rep $script local_N=${N}^3 niter=$NITER --" >&2
                if ! run "$n" julia --project=. --startup-file=no "$script" "$N" "$NITER" "$OUT" 2>err.log; then
                    echo "FAILED: $script mode=$mode ranks=$n rep=$rep (see err.log):" >&2
                    tail -40 err.log >&2
                    exit 1
                fi
            done
        done
    done
done

rm -f err.log
echo "Done. Wrote $OUT" >&2
