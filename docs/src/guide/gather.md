# Gathering arrays

`gather!` assembles equally sized local arrays into a global array on one rank
of the Cartesian communicator. All ranks participate in the call.

```julia
using HaloGrids
import MPI

t = topology((0, 0))
A = fill(Float64(t.rank), (8, 8))
root = 0
A_global = t.rank == root ? zeros(size(A) .* t.dims) : nothing
gather!(A, A_global, t; root)

if t.rank == root
    @show size(A_global)
end
```

This snippet assumes the application manages MPI finalization. A `Grid` may be
passed instead of `t`; it supplies the topology to the same gather operation.
Choose the root using the Cartesian rank (`t.rank`), not a parent communicator's
rank, because topology construction can reorder ranks.

## Halos are not removed automatically

`gather!` takes an array, not a `Field`, and gathers every supplied element.
Passing `f.data` therefore includes halo cells. To gather only the interior of
a field with one halo cell per dimension, create a dense local copy first:

```julia
local_interior = f.data[2:end-1, 2:end-1]
A_global = t.rank == 0 ? zeros(eltype(local_interior),
                             size(local_interior) .* t.dims) : nothing
gather!(local_interior, A_global, t)
```

For the usual case with matching array and topology dimensionality, the global
shape is `size(local_interior) .* t.dims`, and the element types must match.
Non-root ranks pass `nothing`. The implementation uses MPI derived datatypes
and `Gatherv!` to place blocks according to the Cartesian layout.
