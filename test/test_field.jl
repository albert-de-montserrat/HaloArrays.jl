push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
using HaloGrids

@testset "field validation" begin
    A = zeros(6, 6)
    f = field(A; halowidths=(1, 2))
    @test f.halowidths == (1, 2)
    @test size(f) == (6, 6)
    @test ndims(f) == 2
    @test eltype(f) == Float64

    @test_throws Exception field(A; halowidths=(0, 1))   # halowidth < 1
    @test_throws Exception field(A; halowidths=(4, 1))   # 2*4 > 6
    @test field(A; halowidths=(3, 1)) isa Field  # 2*3 == 6: exactly ok

    B = zeros(Int32, 5, 5, 5)
    fB = field(B; halowidths=(1, 1, 1))
    @test eltype(fB) == Int32
    @test ndims(fB) == 3
end

@testset "field default halowidths" begin
    # No halowidths given: defaults to 1 in every dimension (field.jl); this is
    # a constructor default only, not a stored grid-wide/package-wide default,
    # so it coexists freely with any other Field using an explicit width.
    A = zeros(6, 6)
    @test field(A).halowidths == (1, 1)

    C = zeros(4, 4, 4)
    @test field(C).halowidths == (1, 1, 1)

    # Still validated: a dimension too small for the default width of 1 fails
    # the same 2*halowidths[d] <= size(A,d) check as an explicit width would.
    @test_throws Exception field(zeros(1, 4))
end

println("test_field.jl: all tests passed.")
