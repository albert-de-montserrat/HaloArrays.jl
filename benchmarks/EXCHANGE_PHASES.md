# Exchange baseline and grouped completion experiment

Run from the repository root:

```sh
julia --project=. --startup-file=no benchmarks/run_exchange_phases.jl 200
```

The driver launches 1, 2, 4, and 8 MPI ranks. Do not wrap it in mpiexec.
It covers local arrays of 32³, 128³, and 256×32×64 cells (including halos),
with 1, 2, and 4 Float64 fields. These are fixed local sizes, so changing
rank count changes the global domain. Two additional runs compare process
dimensions 1×1×2 and 2×1×1 on the same anisotropic local array.

A focused worker accepts iterations, local shape, and process dimensions:

```sh
julia --project=. --startup-file=no benchmarks/bench_exchange_phases.jl 200 128x32x64 0x0x0
```

All cases use periodic boundaries and halo width one. Position-dependent
initial values check both diagnostic implementations against public halo
exchange. A seven-point stencil checks the public overlap split against
blocking computation. Input interiors stay fixed during measurement; this
is a repeated communication/stencil workload, not a time-evolving simulation.

CSV records share the prefix `kind,ranks,local_shape,process_dims,fields,iterations`:

| Kind | Remaining columns | Meaning |
| --- | --- | --- |
| TOTAL | mode,seconds_per_step,bytes_per_step | Public blocking or overlapped exchange plus identical stencil work |
| EXCHANGE | mode,seconds_per_exchange,bytes_per_exchange | Diagnostic ordered or grouped completion, with phase timers disabled |
| PHASE | mode,dimension,phase,seconds_per_exchange | Separate instrumented run, per dimension |

Times and allocated bytes are maxima across ranks, after ten warmup calls.
Phase maxima may come from different ranks; their sum is not an end-to-end
time. Small phases include timer overhead. Neither the instrumented totals
nor the diagnostic runtime field loops establish production performance.
Use EXCHANGE to compare the two harness strategies and TOTAL to measure
actual public API timestep behavior. Repeat runs before drawing conclusions.

Ordered completion waits and unpacks each receive in turn. Grouped completion
uses cached MPI.RequestSets, waits for all receives, unpacks, and waits for
all sends. RequestSet preparation is included in posting time. Both complete
each dimension before packing the next and use package-owned buffers.
One-rank periodic dimensions use local copies and have zero MPI phases.
Only the final dimension overlaps stencil work in the public API; the
explicit decomposition runs expose the consequence of placing an
undecomposed dimension last.

For isolated repeats of just the two explicit decompositions:

```sh
julia --project=. --startup-file=no benchmarks/run_exchange_phases.jl 500 benchmarks/exchange_phases_repeats.csv focused 2
```

Repeated cases appear in run order (1×1×2 then 2×1×1 for each repeat).

This worker is CPU-only. Staging costs and GPU execution require a separate
CUDA measurement with explicit synchronization; CPU results do not justify
GPU transport changes. Production exchange code remains unchanged.

## Initial local measurements (2026-09-15)

Recorded `exchange_phases_baseline.csv` with 200 iterations, and
`exchange_phases_repeats.csv` with two isolated 500-iteration focused repeats.
Environment: Windows, Julia 1.13.0, Microsoft MPI 10.1.12498.52, CPU target
`znver4`, 16 logical processors, one Julia thread per rank. All ranks run on
one workstation; these measurements do not cover inter-node networking.
Part of the exploratory matrix overlapped allocation-check subprocesses;
use the isolated repeats to assess the focused comparison.

For two fields and local shape 128×32×64, the 1×1×2 decomposition gave:

| Measurement (µs/call) | Repeat 1 | Repeat 2 |
| --- | --- | --- |
| Public blocking stencil step | 387.0 | 419.8 |
| Public overlapped stencil step | 408.3 | 377.6 |
| Diagnostic ordered exchange | 227.7 | 246.2 |
| Diagnostic grouped exchange | 203.3 | 237.7 |

The public overlap comparison changes direction between repeats. Grouped
completion is promising in this focused case, but its benefit varies and
the larger matrix contains regressions. There is insufficient evidence to
replace production completion. A production-shaped, tuple-specialized
candidate should be measured before adoption, including GPU staging.

All benchmark halo/stencil correctness gates passed at 1, 2, 4, and 8 ranks,
and for both explicit two-rank decompositions. The existing field tests and
warmed allocation tests at 1, 2, 4, and 8 ranks also passed. GPU and threaded
tests were not run for this CPU-only harness addition.

## Tuple-specialized completion candidate

The follow-up candidate in `exchange_completion_candidate.jl` uses recursive
field/dimension tuples and the package's existing packing, MPI posting, and
backend staging hooks. It adds cached RequestSets in benchmark-owned state,
without replacing package methods. Requested dimensions still complete in
sequence, preserving full halo corners. Missing neighbors become null
requests; self-periodic exchange stays on the local-copy path.

```sh
julia --project=. --startup-file=no benchmarks/run_completion_candidate.jl cpu 300 benchmarks/completion_cpu.csv
julia --project=. --startup-file=no benchmarks/run_completion_candidate.jl cuda 300 benchmarks/completion_cuda.csv
```

The CUDA driver uses the instantiated test project. Both drivers check mixed
Float64/Float32 fields, variable widths, nonperiodic/periodic domains, subset
and reversed dimensions, cache reuse, and independent concurrent topologies.
CPU checks cover 1/2/4/8 ranks and threaded packing at 1/2 ranks; CUDA checks
cover 1/2 ranks with scalar indexing disabled. Completion benchmarks alternate
measurement order across three repeats, include GPU staging and final device
synchronization, and report maximum per-rank elapsed time and host allocations.

`COMPLETION` CSV columns are kind, backend, ranks, local_shape, process_dims,
fields, iterations, repeat, mode, seconds_per_exchange, bytes_per_exchange.
Test summaries also appear in driver output; select lines beginning with
`COMPLETION,` when parsing results.

The initial candidate checks passed on CPU and the NVIDIA GeForce RTX 5070 Ti.
Median GPU gains at two ranks were only about 1–2%; CPU results include a
14.5% regression for one field at four ranks (128×32×64 local shape), and a
3% regression for four fields at eight ranks. CPU allocations match ordered
completion, including zero on the self-periodic path. CUDA emitted an existing
runtime/precompile version mismatch message (13.4 versus 13.2), although
functional and correctness checks passed. These are workstation measurements.
**Decision: retain ordered completion in production.** The candidate remains
available for future cluster measurements.

## Optional face exchange

The implementation plan is recorded in `FACE_EXCHANGE_PLAN.md`. The public
`region=:faces` option is now implemented; `region=:full` remains the default.
Use the package guide and `examples/face_exchange.jl` for the numerical contract.

```sh
julia --project=. --startup-file=no benchmarks/run_face_exchange.jl cpu 300 benchmarks/faces_cpu.csv
julia --project=. --startup-file=no benchmarks/run_face_exchange.jl cuda 300 benchmarks/faces_cuda.csv
```

This benchmark compares full/face blocking and overlapped seven-point stencil
steps, verifies all four against full blocking exchange, and reports independent
exchange-only measurements. CPU computation uses loops; CUDA computation uses
fused broadcast boxes. The face overlap split has an inner box plus six disjoint
boundary slabs, which adds kernel launches compared with one blocking launch.
Input interiors stay fixed. Field count, shapes, warmup, repeat ordering, and
maximum-per-rank conventions match the completion measurements.

`FACES` CSV columns are kind, backend, ranks, local_shape, process_dims, fields,
iterations, repeat, region, overlap, seconds_per_step, bytes_per_step,
seconds_per_exchange, bytes_per_exchange. GPU bytes are host allocations, not
device memory usage. Times include staging and final device synchronization.
More pending dimensions and smaller slabs do not guarantee faster timesteps.
Separate grids retain full and face scratch buffers, avoiding a mixed-mode
high-water mark in memory reporting. `MEMORY` records share the prefix through
iterations, followed by region, maximum host buffer capacity in bytes, maximum
device buffer capacity in bytes, request slots, and remote sends per exchange.
These capacities exclude Julia object overhead and device allocator pooling.
Face mode keeps buffers for every dimension and can use more scratch memory
than full mode, which reuses buffers across sequential dimensions.
The CUDA driver also profiles full exchange at one rank and two ranks with
both explicit decompositions. `GPU_PHASE` records share the prefix through
iterations, followed by dimension, phase, and maximum seconds per exchange.
Packing, self-copy, receive staging, and unpacking are explicitly synchronized
to separate device work from MPI waits. Send staging and MPI posting are timed
separately while retaining field/side order. Synchronization perturbs normal
pipelining, so these phase times identify costs rather than predict timestep
speedups. Their sum is not an end-to-end time.
