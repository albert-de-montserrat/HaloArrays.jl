import HaloArrays
import HaloArrays: Grid, ExchangeHandle, _halo_exchange_start!, halo_exchange_finish!
using CellArrays: CellArrays, CellArray, ArrayCell, celllength

# --- B=0 (default, GPU-optimal): one combined Field, batched into a single
# MPI message per neighbor per dimension. ---
#
# CellArrays.jl stores every cell's component `j` contiguously for B=0 (`A.data`
# is `(prod(dims), ncomp, 1)`), so `reshape(A.data, (dims..., ncomp))` is a
# genuine zero-copy view with the component axis trailing every grid
# dimension. HaloArrays' `sendrange`/`recvrange`/`halo_length`
# (src/halo_exchange.jl) already loop over a *field's own* `ndims`, not the
# grid's, and only ever touch a dimension that's actually named in `dims` — so
# a trailing, never-named dimension is automatically "always full range,
# never packed", with no pack/unpack changes needed at all. The only core
# change this required was relaxing `check_field_shapes` (src/validation.jl)
# to allow a field with *more* dimensions than the grid (previously an exact
# match was required) — see that function's comment for why nothing else in
# the exchange path needed to change.
#
# This routes the whole CellArray through the existing, already
# allocation-tuned `_halo_exchange_start!` as ONE field instead of
# `celllength(A)` separate ones — measured up to ~5.5x faster for small
# domains/many components on this machine, never slower once both the old
# and new paths reuse buffers fairly (see REDESIGN_PLAN.md's Phase 9 update
# for the full benchmark and the buffer-reuse bug that inverted an earlier,
# unfair first measurement).
#
# `halowidths` needs one entry per dimension of the *field* passed to
# `field()`, i.e. one more than the grid's `N` — the trailing entry (for the
# component axis) is never used by the exchange (that axis is never named in
# `dims`), but `field()`'s constructor validates every dimension
# unconditionally, so a valid-but-inert placeholder (`1`) is supplied.
function cellarray_field_batched(A::CellArray{T,N,0}, halowidths::NTuple{N,Int}) where {T<:ArrayCell,N}
    ncomp = celllength(A)
    flat = reshape(A.data, (A.dims..., ncomp))
    return HaloArrays.field(flat; halowidths=(halowidths..., 1))
end

# --- B=1 (AoS): per-component decomposition, unchanged from the original design. ---
#
# For B=1, the component axis is the *fastest-varying* (leading) dimension in
# `A.data`, not trailing (CellArrays.jl's own `plain_flat` reflects this: it
# puts `celllength` first for B=1). Getting "grid dims first, component axis
# last" would need an actual data permutation, not a free reshape, defeating
# the zero-copy point of batching — and HaloArrays threads a single `dim`
# index through both `topology.neighbors[dim]` (grid-sized) and a field's own
# `sendrange`/`recvrange` (field-sized), so the exchanged dimensions must
# occupy the field's *leading* `N` slots to line up with the grid's own
# dimension numbering; a leading component axis rules that out. B=1 therefore
# keeps the original `celllength(A)`-separate-`Field`s design — still
# correct (and still reuses `CellArrays.field(A,(j,))`'s zero-copy strided
# view, so still no new pack/unpack code), just without the single-message
# win B=0 gets.
function cellarray_fields_percomponent(A::CellArray{T,N}, halowidths::NTuple{N,Int}) where {T<:ArrayCell,N}
    return ntuple(j -> HaloArrays.field(CellArrays.field(A, (j,)); halowidths=halowidths), celllength(A))
end

function cellarray_fields_percomponent(A::CellArray{T,N}, ::NTuple{N,Int}) where {T<:Number,N}
    error(
        "HaloArrays' CellArrays extension only supports multi-component cell types " *
        "(SArray/FieldArray); got a CellArray{$T,...} with a scalar (Number) cell type, " *
        "which has no zero-copy per-component view to decompose. Wrap the underlying " *
        "array with `field(A; halowidths)` and use plain `update_halo!` instead.")
end

# Single dispatch point shared by update_halo!/halo_exchange_start! below,
# picking the batched (B=0) or per-component (any other B, including the
# T<:Number rejection) path.
_cellarray_start!(grid::Grid{N}, A::CellArray{T,N,0}, hw::NTuple{N,Int}, dims::Tuple, region::Symbol) where {T<:ArrayCell,N} =
    _halo_exchange_start!(grid, (cellarray_field_batched(A, hw),), dims, region)
_cellarray_start!(grid::Grid{N}, A::CellArray{T,N,B}, hw::NTuple{N,Int}, dims::Tuple, region::Symbol) where {T<:ArrayCell,N,B} =
    _halo_exchange_start!(grid, cellarray_fields_percomponent(A, hw), dims, region)
_cellarray_start!(grid::Grid{N}, A::CellArray{T,N}, hw::NTuple{N,Int}, dims::Tuple, region::Symbol) where {T<:Number,N} =
    _halo_exchange_start!(grid, cellarray_fields_percomponent(A, hw), dims, region)  # always errors

"""
    update_halo!(grid::Grid, A::CellArray; halowidths=ntuple(_->1,N), dims=ntuple(identity,N), region=:full)

Update the halo of every cell component of `A` on `grid`, in one exchange.
`A`'s cell type must be multi-component (`SArray`/`FieldArray`, i.e.
`CellArrays.ArrayCell`) — a scalar (`Number`) cell type is rejected with a
clear error, since CellArrays.jl provides no zero-copy view to exchange for
that case. `halowidths` applies uniformly to every component; there is no
per-component override (unlike a plain multi-`Field` call, which does allow
per-field widths).

For the default blocklength `B=0` (GPU-optimal), every cell component is
batched into a *single* Field and exchanged as one MPI message per neighbor
per dimension — see `src/CellArraysExt/halo_exchange.jl` for why this is
possible with no new pack/unpack code, and REDESIGN_PLAN.md's Phase 9 update
for benchmark numbers. For `B=1` (AoS), each component is exchanged as a
separate `Field` instead (still zero-copy, just without the single-message
win) — `CellArrays.jl` provides no zero-copy view for any other blocklength,
so nothing else is supported.

A `CellArray` cannot currently be mixed inline with plain `Field`/`Array`
arguments in one call (unlike HaloArrays' `Union{Field,AbstractArray}...`
convenience form) — call this and any other fields' `update_halo!` calls
separately.

`region=:faces` exchanges only faces in the grid dimensions, retaining all
batched components and leaving halo edges/corners unchanged.
"""
function HaloArrays.update_halo!(grid::Grid{N}, A::CellArray{T,N};
        halowidths::NTuple{N,Integer} = ntuple(_ -> 1, N),
        dims::Tuple = ntuple(identity, N), region::Symbol = :full) where {T,N}
    handle = _cellarray_start!(grid, A, Int.(halowidths), dims, region)
    halo_exchange_finish!(handle)
    return nothing
end

"""
    halo_exchange_start!(grid::Grid, A::CellArray; halowidths=ntuple(_->1,N), dims=ntuple(identity,N), region=:full) :: ExchangeHandle

Two-phase counterpart to [`update_halo!`](@ref) for a `CellArray`: starts the
exchange and returns a handle for [`halo_exchange_finish!`](@ref) to
complete. See `update_halo!`'s docstring for the `CellArray`-specific
restrictions and the `B=0`-batched/`B=1`-per-component split.
"""
function HaloArrays.halo_exchange_start!(grid::Grid{N}, A::CellArray{T,N};
        halowidths::NTuple{N,Integer} = ntuple(_ -> 1, N),
        dims::Tuple = ntuple(identity, N), region::Symbol = :full) where {T,N}
    return _cellarray_start!(grid, A, Int.(halowidths), dims, region)
end
