#!/usr/bin/env bash
# SLURM submission + matrix driver for the diffusion3D GPU comparison
# (bench_haloarrays_diffusion3d.jl vs bench_igg_diffusion3d.jl). Submit with:
#   sbatch slurm_diffusion3d.sh
#
# #SBATCH directives and env vars mirror this workspace's own CI convention
# for this cluster (see JustRelax.jl/ci/cscs-gh200.yml, JustPIC.jl/ci/cscs-gh200.yml):
# a CSCS Alps GH200 system, `--uenv julia/25.5:v1 --view=juliaup`, Cray MPICH
# GTL GPU support, IGG's CUDA-aware-MPI env var. That env var is deliberately
# left ON: ImplicitGlobalGrid gets real GPU-direct MPI transport on this
# cluster while HaloArrays.jl's CUDA extension always stages through host
# memory (documented scope gap, not a bug — see HaloArrays.jl/REDESIGN_PLAN.md).
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

# GH200 has much more HBM than the CPU harness's workstation-sized matrix
# (benchmarks/run_comparison.sh's N∈{32,64,128}) — start larger; edit freely.
NTASKS_MATRIX=(1 2 4)
SIZES=(128 256 384)
NITER=150

OUT=results_diffusion3d_gpu.csv
echo "pkg,nprocs,N,niter,time_s,bytes,gctime_s,pauses,full_sweep,gpu_bytes,t_eff_gbs,peak_bw_gbs,pct_of_peak" > "$OUT"

for n in "${NTASKS_MATRIX[@]}"; do
  for N in "${SIZES[@]}"; do
    for script in bench_haloarrays_diffusion3d.jl bench_igg_diffusion3d.jl; do
      echo "-- $script  ntasks=$n  N=$N --" >&2
      line=$(srun -n "$n" --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no "$script" "$N" "$NITER" 2>err.log | grep '^RESULT' || true)
      if [[ -z "$line" ]]; then
        echo "   FAILED (see err.log):" >&2
        tail -40 err.log >&2
        continue
      fi
      echo "${line#RESULT,}" >> "$OUT"
    done
  done
done

rm -f err.log
echo "Done. Wrote $OUT" >&2
