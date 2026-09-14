# Working on HaloArrays.jl

HaloArrays is a Julia package for explicit Cartesian MPI topologies and halo
exchange in 2D and 3D. Read the relevant implementation and tests before editing.
Treat the current source as authoritative; the workspace's `../REDESIGN_PLAN.md`
provides design context, including features that may not yet exist.

## Repository map

- `src/topology.jl`: Cartesian communicators, ranks, coordinates, neighbors.
- `src/grid.jl`: local sizes, coordinate metadata, and owned exchange cache.
- `src/field.jl`: array wrappers and per-field halo widths.
- `src/exchange_cache.jl`: reusable byte buffers, MPI requests, pending state.
- `src/halo_exchange.jl`: packing, dimension ordering, blocking and split API.
- `src/validation.jl`: shape and duplicate-array checks.
- `src/gather.jl`: gathering arrays using MPI derived datatypes.
- `src/reduce.jl`: `global_sum`/`global_max`/`global_min`/`global_reduce`/
  `global_allreduce` — thin `Topology`/`Grid`-based wrappers around
  `MPI.Reduce`/`MPI.Allreduce` for one-scalar-per-rank diagnostics; reduce
  local array data with plain Julia first, then pass the resulting scalar in.
  `global_norm` is the one exception that takes the local array itself (a
  `p`-norm's combining step, sum-of-powers then a single root at the end, is
  easy to get wrong by hand, unlike a plain associative op). None of these
  need GPU-specific code: the local step runs generically on whatever
  backend the data already lives on, and only a host scalar ever reaches MPI.
- `src/backend.jl`: backend dispatch stubs (`is_cuarray`/`is_rocarray`,
  `pack_target`/`unpack_source`/`stage_send!`/`stage_recv!`), overridden by the
  CUDA/AMDGPU extensions via ordinary multiple dispatch on the array type.
- `ext/HaloArraysCUDAExt.jl` + `src/CUDAExt/`: real, tested (GPU present on
  this machine) CUDA support — device-resident pack/unpack buffers
  (`CUDACache`) plus host<->device staging around unchanged host-only MPI
  transport. `ext/HaloArraysAMDGPUExt.jl` + `src/AMDGPUExt/`: structurally
  identical ROCArray mirror, untested here (no AMD GPU available). Neither
  manages device selection (rank-to-GPU assignment) or CUDA/ROCm-aware MPI —
  both explicitly out of scope, documented in the extension files themselves.
- `test/`: standalone Julia tests and an MPI subprocess driver. `test_cuda.jl`/
  `test_amdgpu.jl` need real (not weak) CUDA/AMDGPU dependencies, so they run
  under `test/Project.toml` (`HaloArrays` itself `Pkg.develop`ed via
  `[sources]`), not the package root — `test/runtests.jl` handles this
  automatically; each file skips gracefully via `functional()` if no usable
  GPU is present.
- `docs/`: VitePress Markdown site; `examples/`: runnable Julia examples.
- `benchmarks/`: HaloArrays.jl vs ImplicitGlobalGrid.jl comparison harness —
  `bench_haloarrays*.jl`/`bench_igg*.jl` + `run_comparison*.sh` (CPU,
  workstation-scale) and `gpu/` (CUDA, cluster-scale via `slurm_*.sh`,
  own `Project.toml` since both packages only weakdep CUDA). Not part of
  `test/runtests.jl`; run manually, results are not correctness tests.

## Implementation conventions

- Preserve explicit `Topology` / `Grid` ownership. Do not introduce a global
  current grid, shared communication cache, or runtime backend flags.
- Halo widths belong to `Field`; `Grid.nlocal` includes halo cells.
- Keep one unfinished exchange per grid. Complete earlier dimensions before
  packing later ones so corner data propagates correctly.
- Independent concurrent communication should use separate topologies, which
  own separate communicators. Cache ownership alone does not isolate MPI tags.
- Preserve tuple specialization and the thin keyword/vararg wrapper pattern in
  exchange code. Measure allocations before simplifying these hot paths.
- Packing threads are opt-in; keep MPI calls out of spawned packing tasks.
- Match the existing four-space Julia style and mutating `!` naming. Update
  docstrings and corresponding documentation when public behavior changes.
- Do not describe planned GPU support or performance targets as verified features.
- Preserve pre-existing staged and unstaged work. Do not stage or commit unless asked.

## Validation

Run commands from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no test/test_field.jl
julia --project=. --startup-file=no test/runtests.jl
```

The driver launches MPI tests at 1, 2, 4, and 8 ranks, and threaded packing tests
at 1 and 2 ranks with four Julia threads. Do not wrap the driver in `mpiexec`.
Use focused tests while developing; run the MPI matrix for communication changes.
Allocation changes need warmed-up allocation tests; threading changes need the
threaded tests. Report skipped checks and environment limitations accurately.

For documentation changes:

```sh
npm ci
npm run docs:build
```

Keep internal links valid, public signatures consistent with source, and examples
runnable from the package project. The quickstart page includes its code directly
from `examples/quickstart.jl`. Commit the npm lockfile; never commit `node_modules`,
VitePress cache, or generated site output. A docs-only edit does not require the
full MPI matrix; execute new or changed Julia examples when possible.
