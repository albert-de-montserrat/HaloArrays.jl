#!/usr/bin/env bash
# SLURM driver for the CUDA-aware-MPI probes:
#   1. check_cuda_aware_mpi.jl : does the MPI + CUDA stack accept device buffers?
#   2. bench_pingpong.jl       : latency/bandwidth, device vs staged vs host,
#                                intra-node (2 ranks, 1 node) and inter-node
#                                (2 ranks, 2 nodes)
#   3. bench_halo_cudaaware.jl : 3D halo exchange, HaloArrays.jl vs hand-staged
#                                vs CUDA-aware, weak scaling over the rank matrix
# Submit with:
#   sbatch slurm_cuda_aware.sh
# Results land in results_cuda_aware/*.csv (one row per mode and size).
#
# Defaults follow slurm_diffusion3d.sh (CSCS Alps GH200, Cray MPICH, julia uenv).
# On another cluster edit the CLUSTER SETTINGS block: the #SBATCH lines, the
# module/uenv setup, and the CUDA-aware switch for your MPI:
#   Cray MPICH : MPICH_GPU_SUPPORT_ENABLED=1
#   OpenMPI    : built with --with-cuda (check: ompi_info --parsable | grep mpi_built_with_cuda_support)
#   MVAPICH2   : MV2_USE_CUDA=1
# MPI.jl must use the system MPI (MPIPreferences.use_system_binary), not the
# bundled one, or the CUDA-aware tests will correctly report "no CUDA support".
#
# Needs 2 nodes for the inter-node ping-pong; the halo matrix goes up to 8 ranks.
#SBATCH --job-name=ha-cuda-aware
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=slurm_cuda_aware_%j.out

set -uo pipefail
cd "$(dirname "$0")"

# ---- CLUSTER SETTINGS -------------------------------------------------------
export MPICH_GPU_SUPPORT_ENABLED=1     # Cray MPICH GPU transport layer
export JULIA_CUDA_USE_COMPAT=false
# Some stacks need CUDA.jl's stream-ordered allocator off for GPU-direct RDMA:
# export JULIA_CUDA_MEMORY_POOL=none
UENV="julia/25.5:v1"
RUN() { srun --uenv "$UENV" --view=juliaup "$@"; }
# -----------------------------------------------------------------------------

OUT=results_cuda_aware
mkdir -p "$OUT"
N=${N:-256}          # local N^3 Float64 block per GPU (128 MiB)
NITER=${NITER:-100}
HALO_NTASKS=(1 2 4 8)

echo "Instantiating benchmarks/gpu environment (single task)..."
RUN -n 1 -N 1 julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'

echo "== 1. CUDA-aware capability check (4 ranks) =="
RUN -N 1 -n 4 julia --project=. --startup-file=no check_cuda_aware_mpi.jl
check_rc=$?
if [ $check_rc -ne 0 ]; then
  echo "Capability check exit code $check_rc: device-buffer modes will be skipped or may fail." >&2
  echo "Re-run with HA_FORCE_CUDAAWARE=1 to override the library's own report." >&2
fi

echo "== 2a. ping-pong, intra-node =="
RUN -N 1 -n 2 --ntasks-per-node=2 julia --project=. --startup-file=no bench_pingpong.jl "$OUT/pingpong_intranode.csv" 200 28 || true
echo "== 2b. ping-pong, inter-node =="
RUN -N 2 -n 2 --ntasks-per-node=1 julia --project=. --startup-file=no bench_pingpong.jl "$OUT/pingpong_internode.csv" 200 28 || true

echo "== 3. halo exchange, N=$N =="
for n in "${HALO_NTASKS[@]}"; do
  echo "-- ranks=$n --" >&2
  nodes=$(( (n + 3) / 4 ))
  RUN -N "$nodes" -n "$n" julia --project=. --startup-file=no bench_halo_cudaaware.jl \
      "$OUT/halo_N${N}.csv" "$N" "$NITER" || echo "   FAILED for ranks=$n" >&2
done

echo "Done. Results in $OUT/" >&2
