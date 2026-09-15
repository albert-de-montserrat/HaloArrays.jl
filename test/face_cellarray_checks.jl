include("face_exchange_checks.jl")

function face_cellarray_checks(make_cellarray, to_backend)
    T = SMatrix{2,2,Float64,4}
    @testset "CellArray face exchange" begin
        for N in (2, 3), B in (0, 1), periodic in (false, true), split in (false, true)
            topo = topology(ntuple(_ -> 0, N); periods=ntuple(_ -> periodic, N))
            shape = ntuple(d -> 7 + d, N)
            widths = ntuple(d -> d == 1 ? 2 : 1, N)
            g = grid(topo, shape)
            A = make_cellarray(T, B, shape)
            for c in 1:4
                CellArrays.field(A, (c,)) .= to_backend(face_initial(Float64, topo.rank, shape) .+ 10000c)
            end
            if split
                h = halo_exchange_start!(g, A; halowidths=widths, region=:faces)
                halo_exchange_finish!(h)
            else
                update_halo!(g, A; halowidths=widths, region=:faces)
            end
            for c in 1:4
                host = Array(CellArrays.field(A, (c,)))
                @test all(host[ci] == expected_face(topo, shape, widths, ntuple(identity, N), ci) + 10000c
                    for ci in CartesianIndices(host))
            end
        end
    end
    return nothing
end
