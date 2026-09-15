using Test
using MPI
using HaloArrays
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

@parallel_indices (i, j, k) function increment_owned!(A)
    A[i + 1, j + 1, k + 1] += i + 10j + 100k
    return nothing
end

MPI.Init()
@testset "ParallelStencil overlap and blocking fallback" begin
    t = topology((0, 0, 0); periods=(true, true, true))
    ext = Base.get_extension(HaloArrays, :HaloArraysParallelStencilExt)
    @test_throws ArgumentError ext.can_overlap((0, 1, 1), (1:4, 1:4, 1:4))
    @test_throws ArgumentError ext.can_overlap((1, 1), (1:4, 1:4, 1:4))
    @test_throws ArgumentError ext.can_overlap((1, 1, 1), (2:5, 1:4, 1:4))
    for (n, widths) in ((6, (1, 1, 1)), (6, (2, 1, 1)), (6, (3, 1, 1)), (32, (16, 2, 2)))
        g = grid(t, (n, n, n))
        f = field(fill(Float64(t.rank), n, n, n); halowidths=(1, 1, 1))
        reference = field(copy(f.data); halowidths=(1, 1, 1))
        A = f.data
        B = reference.data
        ranges = (1:n-2, 1:n-2, 1:n-2)
        boundary_calls = Ref(0)
        for _ in 1:2
            @parallel ranges increment_owned!(B)
            B[1, :, :] .= -7
            @synchronize
            update_halo!(g, reference)
            @hide_halo_communication g f boundary_width=widths begin
                @parallel ranges increment_owned!(A)
                A[1, :, :] .= -7
                boundary_calls[] += 1
            end
            @synchronize
            @test A == B
            @test !g.excache.pending
        end
        @test boundary_calls[] == 2
        @parallel ranges increment_owned!(B)
        @synchronize
        update_halo!(g, reference)
        @hide_halo_communication g f begin
            @parallel ranges increment_owned!(A)
        end
        @synchronize
        @test A == B
    end
end
MPI.Finalize()
