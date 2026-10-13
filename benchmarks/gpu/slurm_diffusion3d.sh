#!/usr/bin/env bash
# SLURM submission + matrix driver for the diffusion3D GPU comparison
# Runs blocking and communication-hiding variants for both HaloGrids and
# ImplicitGlobalGrid. Submit with:
#   sbatch slurm_diffusion3d.sh
#
# #SBATCH directives and env vars mirror this workspace's own CI convention
# for this cluster (see JustRelax.jl/ci/cscs-gh200.yml, JustPIC.jl/ci/cscs-gh200.yml):
# a CSCS Alps GH200 system, `--uenv julia/25.5:v1 --view=juliaup`, Cray MPICH
# GTL GPU support, IGG's CUDA-aware-MPI env var. That env var is deliberately
# left ON: ImplicitGlobalGrid gets real GPU-direct MPI transport on this
# cluster while HaloGrids.jl's CUDA extension always stages through host
# memory (documented scope gap, not a bug — see HaloGrids.jl/REDESIGN_PLAN.md).
# This is a real, expected asymmetry in the results below, not neutralized —
# see visualize.jl's summary output, which flags it explicitly.
#
# Adjust #SBATCH resources and the matrix arrays below for your allocation;
# defaults assume a single GH200 quad-GPU node.
#SBATCH --job-name=ha-vs-igg-diffusion3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_diffusion3d_%j.out

set -euo pipefail
cd "$(dirname "$0")"

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1          # ImplicitGlobalGrid: real GPU-direct MPI transport on this cluster
export JULIA_CUDA_USE_COMPAT=false  # ImplicitGlobalGrid's own convention (see cscs-gh200.yml)
UENV="julia/25.5:v1"

echo "Instantiating benchmarks/gpu environment (single task)..."
srun -n 1 --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no -e \
    'using Pkg; Pkg.instantiate()'

# Weak-scaling matrix: each MPI rank owns one GPU and a fixed 128^3 local
# domain. Increasing ntasks therefore increases both the GPU count and the
# global problem size while work and memory per GPU remain constant.
NTASKS_MATRIX=(1 2 4)
N=128
NITER=150

OUT=results_diffusion3d_gpu.jld2
rm -f "$OUT"

for n in "${NTASKS_MATRIX[@]}"; do
  for script in \
    bench_halogrids_diffusion3d.jl \
    bench_halogrids_diffusion3d_hide.jl \
    bench_igg_diffusion3d.jl \
    bench_igg_diffusion3d_hide.jl; do
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
