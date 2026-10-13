#!/usr/bin/env bash
# Run matched GPU workers for root IGG.jl / IGG_hidecomm.jl and their
# HaloGrids counterparts on a SLURM cluster. Adapt the resource and uenv
# directives to the target site.
#
# Submit with: sbatch slurm_igg_stencil.sh
# Optional overrides: N=512 NITER=100 OUT=results.jld2
#
# The MPICH/IGG settings enable CUDA-aware MPI for both workers on supported
# systems. JULIA_MPI_HAS_CUDA informs MPI.jl that Cray MPICH accepts CUDA
# buffers; only set it when the loaded MPICH/GTL stack provides that support.
# HaloGrids uses device buffers directly when MPI.jl reports CUDA support;
# otherwise it falls back to host staging.
#SBATCH --job-name=ha-vs-igg-stencil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_igg_stencil_%j.out

set -euo pipefail
cd "$(dirname "$0")"

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_MPI_HAS_CUDA=true
export JULIA_CUDA_USE_COMPAT=false
UENV="${UENV:-julia/25.5:v1}"
N="${N:-128}"
NITER="${NITER:-100}"
OUT="${OUT:-results_igg_stencil_gpu_${SLURM_JOB_ID:-manual}.jld2}"
NTASKS_MATRIX=(1 2 4)

echo "Instantiating benchmarks/gpu environment..."
srun -n 1 --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no -e \
    'using Pkg; Pkg.instantiate()'

for n in "${NTASKS_MATRIX[@]}"; do
  for script in \
    bench_igg_stencil.jl \
    bench_halogrids_igg.jl \
    bench_igg_stencil_hidecomm.jl \
    bench_halogrids_igg_hidecomm.jl; do
    echo "-- $script ranks=$n local_size=${N}^3 iterations=$NITER --" >&2
    if ! srun -n "$n" --uenv "$UENV" --view=juliaup julia \
      --project=. --startup-file=no "$script" "$N" "$NITER" "$OUT" 2>err.log; then
      echo "FAILED: $script at ranks=$n (see err.log):" >&2
      tail -40 err.log >&2
      exit 1
    fi
  done
done

rm -f err.log
echo "Done. Wrote $OUT" >&2
