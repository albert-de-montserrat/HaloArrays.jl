#!/usr/bin/env bash
# SLURM submission + matrix driver for the hide-communication GPU comparison
# (bench_haloarrays_hide.jl vs bench_igg_hide.jl). Submit with:
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
SIZES=(128 256 384)
NITER=150

OUT=results_hide_gpu.csv
echo "pkg,nprocs,N,niter,time_s,bytes,gctime_s,pauses,full_sweep,gpu_bytes,t_eff_gbs,peak_bw_gbs,pct_of_peak" > "$OUT"

for n in "${NTASKS_MATRIX[@]}"; do
  for N in "${SIZES[@]}"; do
    for script in bench_haloarrays_hide.jl bench_igg_hide.jl; do
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
