# Plan: enable parallel (MPI-aware) HDF5 on Linux

## Status

Not started. This is a follow-up to [`MPI_HDF5_IO_EXTENSION_PLAN.md`](./MPI_HDF5_IO_EXTENSION_PLAN.md),
whose Phase 3 ("stencil-safe collective I/O") explicitly deferred true
simultaneous collective MPI-IO because no parallel-HDF5 build was available
to implement or validate it against on the Windows dev machine that plan was
written on. This plan is that missing piece, to be executed on a Linux
machine instead.

## Goal

Get `HDF5.has_parallel() == true` in a Julia environment that also has this
repository's `HDF5` weak dependency resolved, with MPI.jl and the parallel
HDF5 library bound to the *same* MPI implementation — then use that to
upgrade `HaloArraysHDF5Ext`'s `save_hdf5`/`load_hdf5!` from today's
rank-ordered, one-rank-in-the-file-at-a-time writes to genuinely
simultaneous collective MPI-IO.

## Why this needs a Linux machine

Investigated directly on the Windows dev machine this repo lives on (2026-09-15):

- `HDF5.jl`'s own MPI guide (https://juliaio.github.io/HDF5.jl/stable/mpi/)
  has **no Windows section at all** — every step it documents assumes a
  Unix-like system.
- This machine currently uses `MicrosoftMPI_jll` (v10.1.0), bundled by
  MPI.jl — there is no system-wide MS-MPI install, no `conda`/`mamba`, and no
  WSL distro installed (`wsl --status` reports WSL itself is not installed).
- conda-forge's `hdf5` package (the standard source of a prebuilt parallel
  HDF5) has **no `win-64` build against Microsoft MPI at all** — the only
  Windows MPI variant on conda-forge is `mpi_impi` (Intel MPI). Intel MPI's
  runtime (`impi_rt`) *is* available as a plain conda package (no installer,
  extractable directly), which makes a no-admin-rights path theoretically
  possible, but it would mean switching this project's MPI backend from
  MS-MPI to Intel MPI for at least the HDF5-enabled test environment — a much
  bigger, riskier change than "install a library," with real uncertainty
  about whether Intel MPI's `mpiexec` even runs without registering a
  Windows service (which needs admin rights this session doesn't have).
  Untested rabbit hole, not worth it for what should be a routine setup step.
- No admin/elevation is available in this shell (`net session` confirms
  "not admin"), which rules out running `MSMpiSetup.exe`/an Intel oneAPI
  installer directly even if a matching parallel HDF5 build existed.

On Linux, every one of these blockers disappears: OpenMPI/MPICH and a
parallel-HDF5 built against either are standard, well-supported package-
manager packages (`apt`, `conda-forge`), and HDF5.jl's own docs are written
for exactly this setup.

## Prerequisites

- A Linux machine (native or a VM/container) with a C toolchain not
  strictly required if using prebuilt packages (see Option A/B below).
- Julia with this repository checked out, `test/` environment resolvable
  (`HDF5` is already a weak dependency of the package root and a real
  dependency of `test/Project.toml` — see `MPI_HDF5_IO_EXTENSION_PLAN.md`).
- Either root/sudo (for an apt-based system MPI + HDF5 install) or `conda`/
  `mamba`/`micromamba` (for a fully user-local install, no sudo needed).

## Step-by-step setup

### Option A: system packages (Debian/Ubuntu-family, needs sudo)

```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin libhdf5-openmpi-dev
```

`libhdf5-openmpi-dev` is Debian/Ubuntu's parallel-HDF5-against-OpenMPI
package; find the resulting library path with:

```bash
dpkg -L libhdf5-openmpi-dev | grep 'libhdf5\.so$'
dpkg -L libhdf5-openmpi-dev | grep 'libhdf5_hl\.so$'
```

(Typically under `/usr/lib/x86_64-linux-gnu/hdf5/openmpi/`.)

### Option B: conda-forge (no sudo needed)

```bash
# micromamba/mamba/conda all work; example with micromamba:
micromamba create -n parallel-hdf5 -c conda-forge hdf5=*=mpi_openmpi* openmpi
micromamba run -n parallel-hdf5 bash -c 'echo $CONDA_PREFIX'
```

Library paths will be under `$CONDA_PREFIX/lib/libhdf5.so` and
`libhdf5_hl.so`; the matching OpenMPI install is in the same environment
(`$CONDA_PREFIX/bin/mpiexec`, `$CONDA_PREFIX/lib/libmpi.so`).

Verify conda-forge's current Linux HDF5 build matrix before relying on the
exact package spec above — package/build-string names drift over time
(confirmed on 2026-09-15 that `win-64` only ships an `mpi_impi` variant;
`linux-64` has historically shipped both `mpi_openmpi` and `mpi_mpich`
variants, but check `https://anaconda.org/conda-forge/hdf5/files` at
execution time rather than trusting this plan's memory of it).

### Configure MPIPreferences.jl

Run this **in the same Julia environment/project** that will load `HDF5`
(i.e. `test/`, given this repo's layout — `HDF5` is a weak dependency of the
package root, resolved as a real dependency only in `test/Project.toml`):

```bash
julia --project=test -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; MPIPreferences.use_system_binary()'
```

`use_system_binary()` autodetects an OpenMPI/MPICH install on `PATH`; if it
doesn't find the conda environment's copy, pass `mpiexec_path`/library paths
explicitly (see `MPIPreferences.jl`'s own docstring — `?MPIPreferences.use_system_binary`).
This writes `test/LocalPreferences.toml`. Repeat for the package root's own
`LocalPreferences.toml` too if `test/runtests.jl`'s `MPI.mpiexec()`-based
launcher (which resolves `mpiexec` from the *root* project) needs to agree
with `test/`'s worker processes — mismatched MPI implementations between
launcher and workers will fail outright, not just silently underperform.

### Point HDF5.jl at the parallel library

```julia
# --project=test
using HDF5
HDF5.API.set_libraries!("/path/to/libhdf5.so", "/path/to/libhdf5_hl.so")
```

This writes an `HDF5`-scoped entry into `test/LocalPreferences.toml`. Julia
must be restarted for a native library preference change to take effect.

### Verify

```julia
# --project=test, fresh Julia process
using MPI      # must load before HDF5, per HDF5.jl's own documented ordering
using HDF5
@assert HDF5.has_parallel()
println(HDF5.has_parallel())
```

## What changes in HaloArraysHDF5Ext once this works

Today (`src/HDF5Ext/io.jl`), `save_hdf5` has every rank take a turn, in
order, opening the file, writing its own hyperslab, and closing it
(`MPI.Barrier`/`MPI.Allreduce`-synchronized) — chosen specifically because no
parallel HDF5 was available to validate anything else. `load_hdf5!` already
does independent, uncoordinated per-rank reads (safe without parallel HDF5).
With `HDF5.has_parallel() == true`, the save path can be upgraded to genuine
collective I/O:

1. Open the file with an MPIO file access property list instead of a plain
   `h5open(path, mode)`:
   ```julia
   fapl = HDF5.FileAccessProperties(; fclose_degree=:strong)
   HDF5.set_driver!(fapl, HDF5.Drivers.MPIO(grid.topology.comm, MPI.Info()))
   h5open(path, mode; fapl) do h5
       ...
   end
   ```
   (Check current HDF5.jl API surface for the exact names —
   `HDF5.Drivers.MPIO`/`set_driver!` signatures have moved before between
   HDF5.jl versions; confirm against the installed version's docstrings
   rather than trusting this plan's names verbatim.)
2. Every rank calls `create_dataset`/`create_group` together (collective
   metadata operations — HDF5's own requirement, not this package's), then
   writes its hyperslab with a collective transfer property list
   (`dxpl_mpio = :collective` in older HDF5.jl API, or the equivalent current
   spelling).
3. Remove the `for r in 0:(nprocs-1)` turn-taking loop and its
   `MPI.Allreduce`-based error propagation (`check_field_batch`'s pre-loop
   validation stays — that part isn't turn-taking-specific).
4. Update `MPI_HDF5_IO_EXTENSION_PLAN.md`'s Phase 3 section and the "Status"
   note at its top once this lands — that plan currently documents the
   turn-taking approach as the deliberate, environment-constrained choice;
   this would be the point where that constraint is lifted.
5. Gate the collective path behind `HDF5.has_parallel()` at runtime (or keep
   both paths and pick one based on it) rather than assuming every user has
   a parallel build — most `HDF5.jl` installs via the default weak
   dependency resolution will still be serial-only, exactly like today.

## Test plan

1. Confirm `HDF5.has_parallel()` in `test/`'s environment (see Verify,
   above) before changing any extension code.
2. Re-run the existing `test/test_hdf5_io.jl` unchanged first — it should
   still pass against a parallel-enabled HDF5 build exactly as it does
   against the serial one today (the on-disk format/API contract doesn't
   change, only the transport). This is the regression gate for the
   turn-taking implementation staying correct once collective I/O is added
   alongside it.
3. Implement the collective path (see above) behind a feature flag or
   dispatch on `HDF5.has_parallel()`.
4. Add a correctness test that both paths (turn-taking and collective, when
   available) produce byte-identical files for the same input, across the
   repository's usual `-n 1/2/4/8` matrix.
5. Add a benchmark (this is also Phase 4's still-open "measure allocations
   and overhead" item from the main HDF5 plan) comparing wall-clock time for
   a representative save across both paths, to quantify whether the
   collective path is actually worth the added complexity at the problem
   sizes this package targets — file I/O may already be dominated by disk
   bandwidth rather than MPI coordination overhead at realistic sizes, in
   which case the turn-taking approach might remain the pragmatic default
   even once collective I/O is available.

## Risks and caveats

- **MPI implementation drift**: whatever MPI.jl is configured to use for
  `test/` must exactly match what the parallel HDF5 library was linked
  against (both the vendor — OpenMPI vs. MPICH — and, in stricter cases, the
  ABI-compatible version range). A mismatch typically fails loudly at
  `HDF5.has_parallel()` or at first collective call, not silently.
- **CI implications**: this repo's CI has so far deliberately stayed
  CPU-only and doesn't instantiate `test/`'s optional dependencies
  automatically (see `test/runtests.jl`'s comments on `TEST_PROJECT_CPU_FILES`).
  A parallel-HDF5 CI job would be new infrastructure (a Linux runner with
  OpenMPI + parallel HDF5 preinstalled, or an install step added to the
  workflow) — out of scope for this plan, called out here so it isn't
  discovered as a surprise later.
- **This plan is Linux-specific by design.** Do not attempt to port these
  exact steps back to Windows — see "Why this needs a Linux machine" above
  for what was already tried and ruled out there.

## Acceptance criteria

- [ ] `HDF5.has_parallel() == true` confirmed in `test/`'s Julia environment
      on the Linux machine.
- [ ] `test/test_hdf5_io.jl` still passes unchanged against the
      parallel-enabled HDF5 build.
- [ ] A genuine collective save path is implemented in
      `src/HDF5Ext/io.jl`, gated on `HDF5.has_parallel()`.
- [ ] Byte-identical output confirmed between the turn-taking and collective
      paths for the same input, across `-n 1/2/4/8`.
- [ ] `MPI_HDF5_IO_EXTENSION_PLAN.md` updated to reflect Phase 3 as
      complete, with a benchmark-backed note on whether the collective path
      is enabled by default or opt-in.
