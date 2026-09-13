# Halo exchange

## Blocking updates

```julia
update_halo!(g, f)
update_halo!(g, temperature, pressure)
update_halo!(g, f; dims=(1,))
```

These snippets assume matching grids and fields have already been constructed.
Each side sends the interior strip adjacent to its halo and receives into the
outer halo strip. The other dimensions span the full local array, including
their halos. Processing dimensions sequentially propagates corner values.

The default dimension order is `(1, 2)` or `(1, 2, 3)`. Supply a nonempty tuple
of valid dimension indices to exchange a subset or change the order. Use matching
field order, compatible types and sizes, and the same dimension order on peers.
An exchange restricted to one dimension does not refresh the other halos.

## Overlapping independent computation ("hiding" communication)

```julia
handle = halo_exchange_start!(g, f)
try
    # Compute into a separate output array using only data that does not
    # depend on the pending halos. Keep f.data unchanged until finish!.
finally
    halo_exchange_finish!(handle)
end
# All exchanged halos can now be read.
```

There is no macro for this the way some other packages provide one — it is a
plain two-call API. `halo_exchange_start!` returns a handle; you do compute in
between; `halo_exchange_finish!` completes the handle. `update_halo!` is just
`start!` immediately followed by `finish!`, and is what you want unless you
specifically intend to overlap something.

### Why only the last dimension overlaps

Halo exchange is inherently sequential across dimensions: dimension 2's
packing has to read what dimension 1's unpacking just wrote, so that a corner
cell ends up holding the correct diagonal neighbor's value. `start!` therefore
performs every dimension in `dims` **except the last** fully synchronously —
pack, send/recv, wait, unpack — before it ever returns control to you. Only
the *last* dimension in `dims` is actually left outstanding; that is the one
piece of work `finish!` has left to do, and the only part your inserted
compute genuinely overlaps with. This is a deliberate, permanent scope limit,
not a TODO: a fundamentally different corner-communication algorithm that
would let every dimension overlap was considered and rejected in favor of
documenting the limit honestly.

Concretely, right after `start!(g, f; dims=(1, 2))` returns:

- `f`'s dimension-1 halo (the first/last slab along dimension 1) is already
  correct — that exchange ran to completion before `start!` returned.
- `f`'s dimension-2 halo (the first/last slab along dimension 2, since `2` is
  the last entry in `dims`) is **not yet correct** — the receive is still in
  flight, and the buffer it will unpack into is only safe to touch from
  `finish!`.

So the data that is safe to read while a handle is pending is: every other
field's data unconditionally, plus `f.data` itself *except* the still-in-flight
last-dimension halo slab. The data that is unsafe is `f.data`'s last-dimension
halo slab (being written by the pending receive) and, by extension, anything
you compute that reads it.

### Worked example: overlapping a stencil update with its own halo exchange

The most common real use of this API is not "do something unrelated while
waiting" but "update the part of my own field that doesn't need the halo yet,
while the part that does is still in flight." For a 5-point stencil, a point
at column `j` only reads columns `j-1` and `j+1`, so every column whose
neighbors avoid the still-pending last-dimension halo columns can be updated
immediately; only the one column on each side immediately adjacent to that
halo has to wait for `finish!`:

<<< ../../examples/hide_communication.jl

The example computes the same diffusion step twice — once the straightforward
way (`update_halo!` then the whole interior in one pass) and once with the
overlap split — and asserts the two results are identical, to make the point
that splitting the compute changes *when* work happens, not *what* it
computes. Run it the same way as [the quickstart](./getting-started):

```sh
julia --project=. examples/hide_communication.jl
```

### Progress, buffers, and the one-in-flight-exchange rule

Whether communication actually makes progress *during* your inserted compute —
as opposed to only once you call into MPI again — depends on the MPI
implementation and its progress engine (some implementations only advance
non-blocking requests inside an MPI call, so the overlap is a correctness/
scheduling convenience even where it isn't a wall-clock win). On a single-rank
periodic dimension there is no MPI call at all: the local copy is prepared
during `start!` and its unpack is simply deferred until `finish!`.

Only one exchange may be pending per `Grid` at a time: starting a second one
before finishing the first raises a clear error rather than corrupting
buffers, since the last dimension's send/recv buffers stay live in the grid's
cache until `finish!` unpacks them. Wrap `finish!` in a `try`/`finally` (as
above) so it still runs if your inserted compute throws — an unfinished
handle otherwise leaves the grid permanently "pending" and every later
exchange on it will error. Treat the handle itself as opaque. Different
`Grid`s remain fully independent — see
[multiple grids](./concepts#multiple-grids) for communicator isolation.

## Boundaries

Periodic single-rank dimensions copy the opposite interior strip locally.
Nonperiodic sides without neighbors are not received into for that dimension.
The solver remains responsible for physical boundary values and for their
consistency at edges and corners when other dimensions are exchanged.

A periodic dimension split across exactly **2** ranks is a real, common
topology (an 8-rank 2D grid with a balanced `(4, 2)` split, or simply
`mpiexec -n 2` with one periodic axis) — and, for one commit, was a
genuinely miscorrected case: the two "sides" of such a dimension are the
same physical neighbor rank, and messages were tagged only by which slot
posted them, not by which physical direction (low vs. high) they carried.
MPI's message-ordering guarantee then matched them by post order instead of
direction, so a rank's low halo silently received its neighbor's low-side
interior strip instead of the intended high-side one. Every message is now
tagged by direction rather than by slot or by which rank happens to be on
which side, which is unambiguous even when both neighbors coincide; see the
regression test in `test/test_halo_exchange.jl` (the one whose name starts
with "periodic dimension split across exactly 2 ranks") for a minimal
reproduction. This is the kind of bug a rank-uniform test fixture cannot
see — every existing correctness test filled each rank's whole array with
one repeated value, which happens to make the swapped and correct halos
identical — which is why the [coordinate helpers](./coordinates) worked
example (position-dependent data, checked against an exact conservation
law) is a better correctness oracle for this class of issue than "assert
against the rank number."

## Packing and memory

Communication buffers belong to the grid and grow for the fields and slab sizes
encountered. Reuse the grid and warm up representative exchanges before measuring
steady-state allocations or timing. Cache growth and compilation affect first calls.

```julia
g = grid(t, (258, 258, 258); threaded_pack=true) # t must be 3D
```

Threaded packing is disabled by default. Opting in uses spawned tasks for large
slabs when Julia has multiple threads (for example, `julia -t 4 --project=.`).
The current threshold is 65,536 elements per slab. Tasks allocate, and MPI calls
still run on the calling thread. Benchmark your workload before enabling this.
