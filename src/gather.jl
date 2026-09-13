export gather!

"""
    gather!(A, A_global, topology_or_grid; root=0)

Gather local array `A` from each member of the Cartesian process topology into
one large array `A_global` on the `root` process (`A_global` may be `nothing`
on every other process). `size(A_global)` must equal `size(A) .* dims`, where
`dims` are the process-topology dimensions.

Uses the exact zero-copy technique from the old package (a derived
`MPI.Types.create_subarray` + `create_resized` + a single `Gatherv!`/`VBuffer`
call — no manual pack loop); the only change is that `dims`/`comm` are read
directly off the explicit `Topology`/`Grid` argument instead of an implicit
current-grid lookup.
"""
function gather!(A::AbstractArray{T,N2}, A_global::Union{AbstractArray{T,N},Nothing},
                  topology::Topology{ND}; root::Integer = 0) where {T,N,N2,ND}
    comm = topology.comm
    if MPI.Comm_rank(comm) == root
        A_global === nothing && error("A_global cannot be `nothing` on the root process.")
        N2 <= N || error("ndims(A)=$N2 must be <= ndims(A_global)=$N.")
        N <= ND || error("ndims(A_global)=$N must be <= the topology's number of dimensions ($ND).")
        dims = topology.dims[1:N]
        size_A = (size(A)..., ntuple(_ -> 1, N - N2)...)
        size(A_global) == dims .* size_A || error(
            "size(A_global)=$(size(A_global)) must equal size(A) .* dims = $(dims .* size_A).")
        offset = ntuple(_ -> 0, N)
        subtype = MPI.Types.create_subarray(size(A_global), size_A, offset, MPI.Datatype(eltype(A_global)))
        subtype = MPI.Types.create_resized(subtype, 0, size(A, 1) * Base.elsize(A_global))
        MPI.Types.commit!(subtype)
        counts = fill(Cint(1), reverse(dims))
        displs = zeros(Cint, reverse(dims))
        csizes = cumprod(size_A[2:end] .* dims[1:end-1])
        strides = (1, csizes...)
        for I in CartesianIndices(displs)
            off = reverse(Tuple(I - oneunit(I)))
            displs[I] = sum(off .* strides)
        end
        recvbuf = MPI.VBuffer(A_global, vec(counts), vec(displs), subtype)
        MPI.Gatherv!(A, recvbuf, comm; root)
    else
        MPI.Gatherv!(A, nothing, comm; root)
    end
    return nothing
end

gather!(A, A_global, grid::Grid; root::Integer = 0) = gather!(A, A_global, grid.topology; root = root)
