# Migrating from ImplicitGlobalGrid.jl

HaloGrids uses explicit `Topology` and `Grid` values and per-field halo
widths. Read [core concepts](./concepts.md) before converting call sites.

**Check the field layout before changing an exchange call.** Passing a grid
is sufficient only when the old send/receive ranges match HaloGrids' ranges.
JustRelax's staggered fields do not all meet that condition.

## What maps to what

| ImplicitGlobalGrid.jl | HaloGrids.jl | Conditions |
|---|---|---|
| `init_global_grid(...)` | `topology(...); grid(t, nlocal)` | Supply dimensionality, communicator, and periodicity explicitly. |
| `update_halo!(A)` | `update_halo!(g, A)` | Shape equal to `g.nlocal`; default halo width 1 and overlap 2. |
| Per-array halo width and overlap | `field(A; halowidths=(2,2), overlaps=(4,4))` | `overlaps` is the array overlap `o_A` below; it defaults to `2*halowidths`. |
| `nx_g()`, `x_g(...)`, etc. | Explicit `(Grid, Field)` coordinate helpers | Verify rank stride and physical origin too. |

## Check overlap and rank stride

For a dimension with base size `n`, array size `m`, base overlap `o`, and
exchange width `h`, ImplicitGlobalGrid derives the array overlap as
`o_A = o + m - n`. Its low send strip starts at `1 + o_A - h`; its high
send strip starts at `m - o_A + 1`. HaloGrids starts these strips at
`overlaps - h + 1` and `m - overlaps + 1`, respectively, so pass
`overlaps = o_A` to `field` and the ranges agree. With the default
`overlaps = 2h` they start at `h + 1` and `m - 2h + 1`.

Both receive into the outer `h` cells. If `o_A < 2h`, ImplicitGlobalGrid
skips exchange in that dimension; HaloGrids has no equivalent per-field
automatic skip based on overlap.

With JustRelax's usual `init_global_grid(nx, ny, 1)` defaults (`o=2`,
`h=1`), the representative 2D layouts are:

| Field | Array size | Effective overlap | Bare-array migration |
|---|---|---|---|
| `stokes.viscosity.η`, `ητ` | `(nx, ny)` | `(2,2)` | Exchange ranges match. |
| `stokes.τ.xy`, vertex normal stresses, `ηv` | `(nx+1, ny+1)` | `(3,3)` | `overlaps=(3,3)` on a grid of its own shape. |
| `stokes.V.Vx` | `(nx+1, ny+2)` | `(3,4)` | `overlaps=(3,4)` on a grid of its own shape. |
| `stokes.V.Vy` | `(nx+2, ny+1)` | `(4,3)` | `overlaps=(4,3)` on a grid of its own shape. |
| `thermal.T` | `(nx+2, ny+2)` | `(4,4)` | `overlaps=(4,4)` on a grid of its own shape. |

For example, at `nx=8`, a vertex array has length 9. The old low/high
send indices are `3` and `7`. HaloGrids with width 1 and the default
`overlaps=(2,)` sends indices `2` and `8`; `overlaps=(3,)` restores `3`
and `7`. A rank-constant test cannot detect this shift.

Each distinct shape still needs its own grid to satisfy shape validation.
Increasing halo width instead of overlap changes the receive region and
number of exchanged cells, so it is not an equivalent migration. The old
rank stride is `n-o`; HaloGrids uses `m-overlaps`, which is the same when
`overlaps = o_A`.

## A supported exchange pattern

Build a grid once and reuse it. Add `overlaps=` to `field` when the layout
is not the default `2*halowidths`:

```julia
using HaloGrids
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

Fields with `overlaps != 2*halowidths` are supported by `update_halo!`, the
face exchange and the coordinate helpers, but not by `save_hdf5`/`load_hdf5!`
or `save_pvtk`, which reject them. `nlocal_from_global` assumes the default
overlap. Verify any migrated call site with a position-encoded test, not a
rank-constant one.
