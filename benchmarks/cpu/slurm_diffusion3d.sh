#!/usr/bin/env bash
# SLURM submission + matrix driver for the diffusion3D CPU comparison
# Runs blocking and communication-hiding variants for both HaloGrids and
# ImplicitGlobalGrid. Submit with:
#   sbatch slurm_diffusion3d.sh
#
# #SBATCH directives mirror this workspace's own CI convention for this
# cluster (see JustRelax.jl/ci/cscs-gh200.yml, JustPIC.jl/ci/cscs-gh200.yml):
# a CSCS Alps GH200 system, `--uenv julia/25.5:v1 --view=juliaup`. Unlike
# slurm_diffusion3d.sh in benchmarks/gpu, this variant requests no GPUs and
# runs on the node's Grace CPU cores instead (each GH200 node pairs a Hopper
# GPU with a 72-core Grace CPU, so the same `--uenv` toolchain applies) — no
# CUDA-aware-MPI env vars are needed since nothing here touches a GPU.
#
# Each rank's ParallelStencil `Threads` kernels use $SLURM_CPUS_PER_TASK
# threads (via JULIA_NUM_THREADS); adjust #SBATCH resources, --cpus-per-task,
# and the matrix arrays below to match your allocation's actual core count —
# the defaults below assume a single 72-core-per-socket-class node split
# 4 ways (18 cores/task).
#SBATCH --job-name=ha-vs-igg-diffusion3d-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --output=slurm_diffusion3d_cpu_%j.out

set -euo pipefail
cd "$(dirname "$0")"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
UENV="julia/25.5:v1"

echo "Instantiating benchmarks/cpu environment (single task)..."
srun -n 1 --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no -e \
    'using Pkg; Pkg.instantiate()'

# Weak-scaling matrix: each MPI rank owns a fixed 128^3 local domain and
# $JULIA_NUM_THREADS cores. Increasing ntasks therefore increases both the
# core count and the global problem size while work and memory per rank
# remain constant.
NTASKS_MATRIX=(1 2 4)
N=128
NITER=150

# Each bench_*_diffusion3d*.jl worker appends its own rank count to this base
# name (see save_result/output in metrics_common.jl et al.), so this run
# produces one file per NTASKS_MATRIX entry: results_diffusion3d_cpu_n1.jld2,
# _n2.jld2, _n4.jld2 — not one shared file — since ranks running at different
# times/allocations shouldn't be racing to append into the same file.
OUT_BASE=results_diffusion3d_cpu.jld2
rm -f "${OUT_BASE%.jld2}"_n*.jld2

for n in "${NTASKS_MATRIX[@]}"; do
  for script in \
    bench_halogrids_diffusion3d.jl \
    bench_halogrids_diffusion3d_hide.jl \
    bench_igg_diffusion3d.jl \
    bench_igg_diffusion3d_hide.jl; do
    echo "-- $script  ranks=$n  threads/rank=$JULIA_NUM_THREADS  local_N=${N}^3 --" >&2
    if ! srun -n "$n" --uenv "$UENV" --view=juliaup julia --project=. --startup-file=no -t "$JULIA_NUM_THREADS" "$script" "$N" "$NITER" "$OUT_BASE" 2>err.log; then
      echo "   FAILED (see err.log):" >&2
      tail -40 err.log >&2
      continue
    fi
  done
done

rm -f err.log
echo "Done. Wrote ${OUT_BASE%.jld2}_n<ranks>.jld2 for ranks in: ${NTASKS_MATRIX[*]}" >&2
