# Migrating from ImplicitGlobalGrid.jl

HaloArrays uses explicit `Topology` and `Grid` values and per-field halo
widths. Read [core concepts](./concepts.md) before converting call sites.

**Check the field layout before changing an exchange call.** Passing a grid
is sufficient only when the old send/receive ranges match HaloArrays' ranges.
JustRelax's staggered fields do not all meet that condition.

## What maps to what

| ImplicitGlobalGrid.jl | HaloArrays.jl | Conditions |
|---|---|---|
| `init_global_grid(...)` | `topology(...); grid(t, nlocal)` | Supply dimensionality, communicator, and periodicity explicitly. |
| `update_halo!(A)` | `update_halo!(g, A)` | Matching layout, shape equal to `g.nlocal`, and halo width 1. |
| Per-array custom halo width | `field(A; halowidths=(2,2))` | Width cannot independently specify duplicated interface points. |
| `nx_g()`, `x_g(...)`, etc. | Explicit `(Grid, Field)` coordinate helpers | Verify rank stride and physical origin too. |

## Check overlap and rank stride

For a dimension with base size `n`, array size `m`, base overlap `o`, and
exchange width `h`, ImplicitGlobalGrid derives the array overlap as
`o_A = o + m - n`. Its low send strip starts at `1 + o_A - h`; its high
send strip starts at `m - o_A + 1`. HaloArrays starts these strips at
`h + 1` and `m - 2h + 1`, respectively. These agree when `o_A == 2h`.

Both receive into the outer `h` cells. If `o_A < 2h`, ImplicitGlobalGrid
skips exchange in that dimension; HaloArrays has no equivalent per-field
automatic skip based on overlap.

With JustRelax's usual `init_global_grid(nx, ny, 1)` defaults (`o=2`,
`h=1`), the representative 2D layouts are:

| Field | Array size | Effective overlap | Bare-array migration |
|---|---|---|---|
| `stokes.viscosity.η`, `ητ` | `(nx, ny)` | `(2,2)` | Exchange ranges match. |
| `stokes.τ.xy`, vertex normal stresses, `ηv` | `(nx+1, ny+1)` | `(3,3)` | Unsupported shared-interface layout. |
| `stokes.V.Vx` | `(nx+1, ny+2)` | `(3,4)` | Unsupported shared-interface layout. |
| `stokes.V.Vy` | `(nx+2, ny+1)` | `(4,3)` | Unsupported shared-interface layout. |
| `thermal.T` | `(nx+2, ny+2)` | `(4,4)` | Unsupported with the original width 1. |

For example, at `nx=8`, a vertex array has length 9. The old low/high
send indices are `3` and `7`; HaloArrays with width 1 sends indices `2`
and `8`. A rank-constant test cannot detect this shift.

Creating a separate grid for each shape satisfies shape validation but
does not fix these indices. Increasing halo width changes the receive
region and number of exchanged cells, so it is not an equivalent migration.
The old rank stride is `n-o`; HaloArrays' corresponding stride is `m-2h`.

## A supported exchange pattern

For a layout whose effective overlap equals twice its halo width in every
exchanged dimension, build a grid once and reuse it:

```julia
using HaloArrays
import MPI

# Fresh 2D topology; choose periodicity to match the application.
t = topology((0, 0); periods=(false, false))
A = zeros(8, 6)
g = grid(t, size(A))
f = field(A; halowidths=(1, 1))
update_halo!(g, f)
```

This demonstrates the exchange API, not a complete JustRelax solver port.
Converting `_solve_VS!` also requires resolving its vertex stress and
velocity layouts. A grid based on viscosity cannot accept all their shapes.

When migrating an existing communicator, explicitly preserve its process
dimensions and periodicity. JustRelax's 2D `IGG` originates from a 3D MPI
topology with a singleton z dimension; do not blindly pass its three-entry
`dims` tuple to a grid holding 2D arrays. Construct a deliberate 2D topology
or retain a consistently 3D array representation, and verify coordinate/rank
mapping if reordering is enabled.

## Multiple grids and coordinates

Each distinct array shape needs its own grid. Grids can share a topology
for completed, sequential exchanges. Independent concurrent exchanges need
separate topologies with separate communicators; per-grid caches do not
isolate MPI tags.

See [coordinate helpers](./coordinates.md) for explicit coordinate APIs.
Verify the old rank stride and location-dependent origin before substituting
those helpers. Matching function names does not establish matching coordinates.

HaloArrays currently has no explicit representation for the extra shared
interface points described above. Keep those call sites on their existing
exchange implementation until the layout is supported or the solver's
storage/ownership scheme is deliberately changed and tested.
