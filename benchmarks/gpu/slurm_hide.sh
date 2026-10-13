#!/usr/bin/env bash
# SLURM submission + matrix driver for the hide-communication GPU comparison
# (bench_halogrids_hide.jl vs bench_igg_hide.jl). Submit with:
#   sbatch slurm_hide.sh
#
# See slurm_diffusion3d.sh's header comment for the cluster/env-var
# conventions (CSCS Alps GH200, --uenv, IGG_CUDAAWARE_MPI left on
# deliberately) — identical here, not repeated.
#SBATCH --job-name=ha-vs-igg-hide
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_hide_%j.out

set -euo pipefail
cd "$(dirname "$0")"

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1
export JULIA_CUDA_USE_COMPAT=false
UENV="julia/25.5:v1"

echo "Instantiating benchmarks/gpu environment (single task)..."
srun -n 1 --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no -e \
    'using Pkg; Pkg.instantiate()'

NTASKS_MATRIX=(1 2 4)
# Weak scaling: one rank/GPU and a constant 128^3 local domain per GPU.
N=128
NITER=150

OUT=results_hide_gpu.jld2
rm -f "$OUT"

for n in "${NTASKS_MATRIX[@]}"; do
  for script in bench_halogrids_hide.jl bench_igg_hide.jl; do
    echo "-- $script  gpus=$n  local_N=${N}^3 --" >&2
    if ! srun -n "$n" --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no "$script" "$N" "$NITER" "$OUT" 2>err.log; then
      echo "   FAILED (see err.log):" >&2
      tail -40 err.log >&2
      continue
    fi
  done
done

rm -f err.log
echo "Done. Wrote $OUT" >&2
