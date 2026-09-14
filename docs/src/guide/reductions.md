# Global reductions

`global_sum`, `global_max`, and `global_min` combine one scalar per rank into
one result, across every rank sharing a `Topology`'s (or `Grid`'s)
communicator. All ranks participate in the call.

```julia
using HaloArrays
import MPI

t = topology((0, 0))
A = fill(Float64(t.rank), (8, 8))

local_max = maximum(A)
global_bound = global_max(t, local_max)   # every rank gets the same result
```

This snippet assumes the application manages MPI finalization. A `Grid` may be
passed instead of `t`, the same way as [`gather!`](./gather.md).

## Reduce your own local data first

These wrappers do not know about halo widths or owned-vs-halo regions, the
same way `gather!` does not remove halos automatically. Reduce your own local
array with plain Julia first — `sum(A)`, `maximum(A)`, excluding halo cells
however is appropriate for the computation — to get one scalar per rank, then
pass that scalar to `global_sum`/`global_max`/`global_min`. This keeps the
reduction itself independent of array backend: the local step can run on a
`CuArray`/`ROCArray` however it likes, and only a host scalar ever needs to
reach MPI, so no CUDA/ROCm-aware MPI transport is required.

## Every rank vs. one rank

`global_sum`/`global_max`/`global_min` are built on `global_allreduce`, which
gives every rank the same, fully reduced result — the right choice for a
value every rank needs to keep going, such as a CFL timestep bound derived
from a global max velocity, or a global convergence residual. If only one
rank needs the result (for example, to print or log it), use
`global_reduce(topology_or_grid, val, op; root=0)` instead, which reduces
onto a single rank. Both accept any reduction operator `MPI.Reduce`/
`MPI.Allreduce` support, not just `+`/`max`/`min`.

```julia
# Every rank needs the timestep bound to keep advancing.
vmax = global_max(t, maximum(abs, V))
dt = cfl * dx / vmax

# Only rank 0 needs the total for logging.
total_mass = global_reduce(t, sum(ρ), +; root=0)
t.rank == 0 && @show total_mass
```

## Global norms

`global_norm` is the one exception to "reduce locally, then pass a scalar
in": it takes each rank's own local array (or scalar) directly.

```julia
residual_norm = global_norm(t, local_residual)        # L2 by default
max_abs = global_norm(t, local_residual, Inf)          # ℓ∞ norm
```

A `p`-norm cannot be combined across ranks the same way `+`/`max`/`min` can.
The correct combination is: sum every rank's `|x|^p` values first, combine
*those* sums across ranks, and take a single `p`-th root at the very end.
Combining each rank's own already-computed local norm directly — for
example `global_sum(t, norm(local_residual))` — gives the wrong answer for
any `p` other than `1`, silently. `global_norm` does the full computation
for you so this mistake isn't available to make.

