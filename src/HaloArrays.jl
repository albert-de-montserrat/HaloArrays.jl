"""
Module HaloArrays

A ground-up redesign of ImplicitGlobalGrid.jl's halo-exchange core: explicit
`Grid` objects instead of module-global mutable state, per-grid-owned
communication caches instead of module-`let` singletons, a single per-field
halo-width concept instead of `overlaps`/`halowidths`, and an explicit
two-phase `halo_exchange_start!`/`halo_exchange_finish!` API for
communication/compute overlap.

See `REDESIGN_PLAN.md` (in the ImplicitGlobalGrid.jl / HaloArrays.jl repo root)
for the full design rationale and phased build plan this package follows.

# Core types
- [`Topology`](@ref): explicit Cartesian MPI process topology.
- [`Grid`](@ref): sizing + owned [`ExchangeCache`](@ref) layered onto a `Topology`.
- [`Field`](@ref): a local array paired with its own per-dimension halo width.

# Functions
- [`topology`](@ref), [`grid`](@ref), [`field`](@ref)
- [`halo_exchange_start!`](@ref), [`halo_exchange_finish!`](@ref), [`update_halo!`](@ref)
- [`gather!`](@ref)
- [`global_sum`](@ref)/[`global_max`](@ref)/[`global_min`](@ref)/[`global_norm`](@ref),
  [`global_reduce`](@ref)/[`global_allreduce`](@ref)
- Coordinate helpers, per field: [`nglobal`](@ref)/[`nx_g`](@ref)/[`ny_g`](@ref)/[`nz_g`](@ref),
  [`iglobal`](@ref)/[`ix_g`](@ref)/[`iy_g`](@ref)/[`iz_g`](@ref),
  [`coord_g`](@ref)/[`x_g`](@ref)/[`y_g`](@ref)/[`z_g`](@ref)
"""
module HaloArrays

import MPI

include("topology.jl")
include("field.jl")
include("exchange_cache.jl")
include("grid.jl")
include("validation.jl")
include("backend.jl")
include("halo_exchange.jl")
include("gather.jl")
include("reduce.jl")
include("coordinates.jl")

end
