# Rules for `test/`

Also follow the root `AGENTS.md`. Use the `run-tests` skill to choose a command.

- Name files `test_*.jl`. The driver runs every such file automatically. Register
  special cases in the constants at the top of `runtests.jl`:
  `NO_MPI_FILES`, `THREADED_FILES`, `GPU_FILES`, `TEST_PROJECT_CPU_FILES`.
- A test needing CUDA, AMDGPU, CellArrays, HDF5, ParallelStencil, WriteVTK, or
  ReadVTK runs under `--project=test`. Put it in `GPU_FILES` or
  `TEST_PROJECT_CPU_FILES`. Never add those packages to the root `[deps]`.
- GPU tests must skip with `functional()` when no GPU exists. They must not fail.
- Tests run under many rank counts (1, 2, 4, 8). Write them to pass for all of
  them. Never hard-code rank count. Test periodic and non-periodic topologies.
- Prefer a position-dependent oracle (value encodes global index) over constant
  fill. Constant fill hides wrong-neighbor and wrong-strip bugs.
- Allocation tests must warm up first, then measure. Keep them in
  `test_allocations*.jl`.
- A skipped test is not a pass. Say which files were skipped and why.
