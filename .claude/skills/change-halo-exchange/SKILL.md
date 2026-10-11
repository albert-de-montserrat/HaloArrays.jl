---
name: change-halo-exchange
description: Playbook for editing HaloArrays packing, unpacking, tags, dimension ordering, face exchange, ExchangeCache, or the start/finish API. Use before touching src/halo_exchange.jl, src/face_exchange.jl, src/exchange_cache.jl, or src/grid.jl.
---

# Change halo exchange

Read `src/AGENTS.md` first. These invariants break silently.

## Before editing

1. Read the function and the test that covers it (`test/test_halo_exchange.jl`,
   `test/face_exchange_checks.jl`, `test/test_overlaps.jl`, `test/test_multigrid.jl`).
2. Decide which paths are affected: full exchange, `region=:faces`, split
   `start!`/`finish!`, self-periodic (1 rank), 2-rank periodic, threaded pack,
   CUDA/AMDGPU staging, CellArrays.
3. Take a baseline allocation reading with `test/test_allocations.jl`.

## While editing

- Keep dimension order: later dimensions pack after earlier ones unpack (corners).
- Keep tuple recursion and thin wrapper methods. Do not collect fields into a `Vector`.
- Keep MPI calls out of `Threads.@spawn` tasks.
- Keep `pending` set/clear symmetric, including the error path.
- Directional tags only. No shared tag between lo and hi sends.
- Touching strip geometry (`halowidths`, `overlaps`)? Update `face_range`,
  `coordinates.jl`, and the I/O extensions' overlap check together.
- Touching a backend hook in `backend.jl`? Update `src/CUDAExt/` and `src/AMDGPUExt/`
  together, and `src/CellArraysExt/` if it forwards.

## Verify

1. A position-dependent correctness test at 1, 2, 4, 8 ranks, periodic and
   non-periodic (use the `run-tests` skill).
2. `test_allocations.jl` after warm-up. Compare with the baseline. A new allocation
   needs a reason in the report.
3. Threaded tests if packing changed.
4. GPU tests if backend hooks changed. State when AMDGPU was not runnable.
5. Update docstrings and `docs/src/guide/halo-exchange.md` if behavior changed.
