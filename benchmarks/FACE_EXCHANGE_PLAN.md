# Optional face exchange

Public option: `region=:faces` on `update_halo!` and `halo_exchange_start!`.
The default `region=:full` preserves sequential corner propagation. Both
regions share the grid's pending guard; one unfinished exchange per grid
remains the rule. Independent grids need independent topologies.

For face exchange, a send/receive slab spans the field's owned interior in
all other grid dimensions. Extra trailing component dimensions span their
full extent. Edges and corners (halo cells in more than one grid dimension)
are never written. Unrequested faces and physical boundaries without peers
remain unchanged. Matching peers must use the same region, dimensions,
field order, types, widths, and compatible slab sizes.

Each Cartesian dimension uses separate grid-owned buffer slots, indexed by
`(dimension-1)*number_of_fields + field_position`. Reserve capacity before
posting communication. Reuse the existing host/device buffer dispatch and
staging hooks, including device-only periodic self-copies. Face tags encode
dimension and send direction; distinct tags are essential when two periodic
directions share the same neighbor rank. Reject repeated dimensions in this
mode so buffer slots cannot be reused while MPI owns them.

All requested dimensions pack and post before start returns. The handle
retains a specialized tuple of dimensions; finish waits, stages, and unpacks
their faces, then completes sends before clearing the pending guard. The
full-region handle keeps its existing final-dimension behavior. A face-mode
caller must treat every requested face as pending and keep input data
unchanged while computing into separate output storage.

Use this for axial stencils, such as a five/seven-point Laplacian. Mixed
derivatives, diagonal stencils, and algorithms that consume halo edges or
corners require the full-region exchange. This mode does not attempt direct
diagonal MPI exchange and does not establish asynchronous MPI progress.

Validation must cover exact source coordinates, untouched edges/corners,
owned cells, subset/reversed dimensions, mixed types and widths, periodic
one/two-rank dimensions, nonperiodic walls, cache growth/reuse, split API,
concurrent separate topologies, and guards shared with full exchange. Check
batched trailing components, real CUDA, and opt-in threaded packing. Compare
an axial stencil against full exchange and report end-to-end timestep timing
and warmed allocations separately from exchange-only timing.
