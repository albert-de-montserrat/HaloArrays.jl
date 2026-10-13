using HaloGrids
import MPI

MPI.Init()
try
    t = topology((0, 0); periods=(true, true))
    g = grid(t, (10, 10))  # 8 x 8 interior plus one halo cell on each side
    A = fill(-1.0, g.nlocal)
    A[2:end-1, 2:end-1] .= t.rank
    f = field(A; halowidths=(1, 1))

    update_halo!(g, f)

    # Check face interiors; corners propagate through both dimensions.
    @assert all(==(t.neighbors[1][1]), A[1, 2:end-1])
    @assert all(==(t.neighbors[1][2]), A[end, 2:end-1])
    @assert all(==(t.neighbors[2][1]), A[2:end-1, 1])
    @assert all(==(t.neighbors[2][2]), A[2:end-1, end])
    MPI.Barrier(t.comm)
    t.rank == 0 && println("Halo exchange complete on $(prod(t.dims)) rank(s).")
finally
    MPI.Finalize()
end
