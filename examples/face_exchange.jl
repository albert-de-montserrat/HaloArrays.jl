# Radius-one axial stencil: all face exchanges may overlap the inner box.
using MPI, HaloArrays

function stencil_box!(out, A, ir, jr, kr)
    @inbounds for k in kr, j in jr, i in ir
        out[i, j, k] = A[i, j, k] + 0.1 * (
            A[i-1, j, k] + A[i+1, j, k] + A[i, j-1, k] +
            A[i, j+1, k] + A[i, j, k-1] + A[i, j, k+1] - 6A[i, j, k])
    end
    return nothing
end

function main()
    MPI.Init()
    try
        topo = topology((0, 0, 0); periods=(true, true, true))
        shape = (16, 12, 10)
        nx, ny, nz = shape
        g = grid(topo, shape)
        A = reshape([sin(i + 13topo.rank) for i in 1:prod(shape)], shape)
        reference = copy(A)
        expected, out = zeros(shape), zeros(shape)
        update_halo!(g, field(reference))
        stencil_box!(expected, reference, 2:nx-1, 2:ny-1, 2:nz-1)

        h = halo_exchange_start!(g, field(A); region=:faces)
        try
            stencil_box!(out, A, 3:nx-2, 3:ny-2, 3:nz-2)
        finally
            halo_exchange_finish!(h)
        end
        # Disjoint slabs cover the owned boundary without recomputing cells.
        stencil_box!(out, A, (2, nx-1), 2:ny-1, 2:nz-1)
        stencil_box!(out, A, 3:nx-2, (2, ny-1), 2:nz-1)
        stencil_box!(out, A, 3:nx-2, 3:ny-2, (2, nz-1))
        @assert out == expected
        MPI.Barrier(topo.comm)
        topo.rank == 0 && println("Face-overlapped stencil matches full exchange.")
    finally
        MPI.Finalize()
    end
end

main()
