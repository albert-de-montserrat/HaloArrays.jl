# GPU arrays and GPU-aware MPI

`CuArray` and `ROCArray` fields work wherever a `Field` wraps an array. Load
CUDA.jl or AMDGPU.jl before exchanging GPU fields so that the matching package
extension is active. The CUDA extension is tested on NVIDIA hardware; the
AMDGPU extension is a structurally identical port that has not been run on ROCm
hardware.

```julia
using CUDA, MPI, HaloGrids

MPI.Init()
t = topology((0, 0, 0); periods=(true, true, true))
g = grid(t, (130, 130, 130))
T = field(CUDA.zeros(Float64, 130, 130, 130))
update_halo!(g, T)
```

## Device selection

HaloGrids does not choose a GPU for each rank. Select the device before
allocating arrays, for example with `CUDA.device!(0)` when the job launcher
exposes one GPU per rank (SLURM `--gpus-per-task=1`). If a rank can see several
devices, determine its node-local rank and select a device from that rank before
allocating arrays.

## Transport: host-staged or GPU-aware

Halo strips are always packed and unpacked on the device. What happens between
packing and unpacking depends on the grid:

| `grid(...; gpu_aware_mpi=...)` | MPI sends and receives | Requirements |
|---|---|---|
| `false` (default) | host buffers; packed strips are copied device → host before sending and host → device after receiving | any MPI library |
| `true` | the device buffers themselves | an MPI library built with CUDA or ROCm support |

```julia
g = grid(t, (130, 130, 130); gpu_aware_mpi=true)
```

With `gpu_aware_mpi=true`:

- `grid` throws unless `MPI.has_cuda()` or `MPI.has_rocm()` is `true`, so a
  misconfigured MPI fails at grid construction instead of crashing inside an
  exchange.
- Before MPI accesses the device buffers of a dimension, the exchange
  synchronizes the task's GPU stream, so the pack kernels and the previous
  dimension's unpack kernels have finished.
- CPU fields on the same grid are unaffected.
- `update_halo!`, `halo_exchange_start!`/`halo_exchange_finish!`,
  `region=:faces`, and `@hide_halo_communication` all use the same transport.

The setting belongs to the grid, so one program can mix grids with both
transports.

## Configuring MPI.jl

MPI.jl's default binary (`MPICH_jll`) is not GPU-aware. Point MPI.jl at a
GPU-aware system MPI once per environment:

```julia
using MPIPreferences
MPIPreferences.use_system_binary()  # or pass library_names=..., mpiexec=... explicitly
```

Then check what MPI.jl reports:

```julia
using MPI
MPI.Init()
MPI.identify_implementation(), MPI.has_cuda()
```

How `MPI.has_cuda()` becomes `true` depends on the implementation:

- **Open MPI** built with CUDA support (for example Open MPI 5 configured with
  `--with-cuda --with-ucx` against a CUDA-enabled UCX): detected automatically.
- **Cray MPICH** (for example on HPE Cray systems): MPI.jl cannot query it.
  Enable GPU support in MPICH and tell MPI.jl about it:

  ```sh
  export MPICH_GPU_SUPPORT_ENABLED=1
  export JULIA_MPI_HAS_CUDA=true
  ```

If GPU-aware transfers fail or hang with arrays allocated from CUDA.jl's
stream-ordered memory pool, disable the pool with
`JULIA_CUDA_MEMORY_POOL=none`.

## Verifying a setup

`benchmarks/gpu/check_cuda_aware_mpi.jl` sends a `CuArray` around a ring of
ranks and checks the received values. It exits with an error if
`MPI.has_cuda()` is `false` or the data arrives wrong:

```sh
mpiexec -n 2 julia --project=benchmarks/gpu benchmarks/gpu/check_cuda_aware_mpi.jl
```

Run it across two nodes as well to exercise the inter-node path.

The GPU test files (`test/test_cuda.jl`, `test/test_face_exchange_cuda.jl`,
`test/test_allocations_cuda.jl`) add `gpu_aware_mpi=true` cases whenever
`MPI.has_cuda()` is `true`. With the default `MPICH_jll` they print a skip
message for those cases. See [Contributing](../contributing.md) for running
them against a GPU-aware MPI.

## Tested configurations

The CUDA-aware path is tested with Open MPI 5.0.9 and UCX 1.19 on a single
NVIDIA GPU shared by up to four ranks, where transfers use CUDA IPC within the
device. Multi-node GPU-aware transport (for example Cray MPICH with GPUDirect
RDMA) has not been benchmarked yet; `benchmarks/gpu/slurm_cudaaware.sh` runs the
comparison against ImplicitGlobalGrid.jl on a SLURM cluster. The ROCm-aware
path is untested.
