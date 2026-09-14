# Coordinate helpers

`Grid` and `Field` carry sizing and halo-width information, but nothing
about *where* a given local array index sits in the global domain — that is
what the functions in `src/coordinates.jl` compute. They exist mainly for
migration ease from ImplicitGlobalGrid.jl, whose `nx_g`/`x_g`/`ix_g` (and
`y_g`/`z_g`/`iy_g`/`iz_g`) are used throughout JustRelax.jl and JustPIC.jl to
build initial conditions, boundary masks, and diagnostics from physical
position rather than local index.

## Why these are per-field, not per-grid

In the old package, `nx_g()` was a grid-wide property: one active global
grid, one global size. Here, [the design deliberately has no grid-wide halo
width](./concepts.md) — each `Field` declares its own — so the *global* size a
field sees also depends on that field's own halo width: a wider halo means
more cells are duplicated across each rank boundary, so a field with
`halowidths=(2,2)` sees a smaller global domain than one with
`halowidths=(1,1)` on the very same `Grid`. Every coordinate helper therefore
takes both a `Grid` and a `Field`:

```julia
nglobal(grid, field, dim)                    # global size in dimension `dim`
iglobal(grid, field, dim, i; wrap_periodic)  # global index of local index i
coord_g(grid, field, dim, i, dx)             # global physical coordinate of local index i
```

with `nx_g`/`ny_g`/`nz_g`, `ix_g`/`iy_g`/`iz_g`, and `x_g`/`y_g`/`z_g` as
convenience wrappers for the first three dimensions, named and (for the
coordinate functions) argument-ordered to match the old package's
`x_g(ix, dx, A)` and similar.

One simplification falls out for free: every `Field` exchanged on a `Grid`
must already have `size(field.data) == grid.nlocal` exactly (enforced by
`update_halo!`/`halo_exchange_start!`). The old package's coordinate helpers
spend real complexity on arrays that are staggered relative to the base grid
(`Vx = zeros(nx+1, ny, nz)`, for a face-centered velocity component) — that
case cannot arise here, so there is no separate "array bigger/smaller than
the grid" adjustment to reason about.

## Worked example: initial conditions from physical position

`nx_g`, `x_g`, and friends let you build spatially-varying data (initial
conditions, coefficient fields, boundary masks) so that a rank's result does
not depend on how the domain happens to be decomposed — the same physical
point gets the same value whether it is computed by one rank owning the
whole axis or split across eight. The full example below (a 3D heat
diffusion step, periodic in all three dimensions) builds two Gaussian
anomalies each in heat capacity and temperature this way, then runs the
explicit finite-difference update, exchanging `T`'s halo every step.

Because the domain is fully periodic, total thermal energy (`sum(Cp .* T)`
over the whole domain) is an exactly conserved quantity of this discrete
scheme — every internal flux face is shared by two neighboring cells with
opposite sign, so summed over a closed domain the update telescopes to zero,
independent of how the domain is decomposed across ranks. The example checks
this directly, which makes it a real correctness check rather than a
doesn't-crash smoke test: a wrong halo exchange (this exact package once had
one — see the note in [halo exchange](./halo-exchange.md) about the 2-rank
periodic ring bug) breaks conservation immediately and visibly.

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "..", "examples", "diffusion3D.jl"), String) * "\n```")
```

Run it from the repository root:

```sh
julia --project=. examples/diffusion3D.jl
```

or across several ranks:

```julia
import MPI
MPI.mpiexec() do exe
    run(`$exe -n 8 $(Base.julia_cmd()) --project=. examples/diffusion3D.jl`)
end
```

The printed relative energy drift is at machine precision (typically `0.0`
or on the order of `1e-16`) regardless of rank count.
