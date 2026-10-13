# Correctness tests for the parallel ParaView (pvtk) I/O extension
# (src/WriteVTKExt/, ext/HaloGridsWriteVTKExt.jl).
#
# Fill strategy: every rank fills its *entire* local array (halo cells
# included) from `iglobal`, so a cell's value depends only on its global
# position, not on which rank computed it — same strategy as
# test_hdf5_io.jl. Unlike that file, no update_halo! step is needed before
# saving: since a value only ever depends on global position, two
# neighboring ranks' overlapping cells are already identical by
# construction, which is exactly the overlap save_pvtk relies on.
#
# ReadVTK.jl (test-only dependency, never a package weakdep — save_pvtk has
# no load_pvtk! counterpart) verifies round trips two ways: (1) each rank
# reads back *its own* serial .vti piece and compares against field.data
# directly; (2) rank 0 reads the .pvti header, which makes ReadVTK stitch
# every piece into one array covering the whole global domain (using the
# same overlapping extents save_pvtk wrote) — comparing that assembled array
# against the expected global value at every global index simultaneously
# checks per-rank correctness, correct piece count, and correct
# `2*halowidths[d]` overlap geometry, since a mistake in any of those would
# show up as a gap or wrong value somewhere in the assembled array.
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using Test
import MPI
using HaloGrids
using WriteVTK
using ReadVTK

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

ext = Base.get_extension(HaloGrids, :HaloGridsWriteVTKExt)
@test ext !== nothing

function cleanup!(path::AbstractString)
    MPI.Barrier(comm)
    if rank == 0
        isfile(path * ".pvti") && rm(path * ".pvti"; force = true)
        isdir(path) && rm(path; recursive = true, force = true)
    end
    MPI.Barrier(comm)
end

fill_formula(gidx::NTuple{N,Integer}) where {N} = sum(gidx[d] * 1000.0^(d - 1) for d in 1:N)

global_fill_value(grid::Grid{N}, field::Field{T,N}, ci::CartesianIndex{N}) where {T,N} =
    fill_formula(ntuple(d -> HaloGrids.iglobal(grid, field, d, ci[d]), N))

function fill_globally!(grid::Grid, field::Field)
    for ci in CartesianIndices(field.data)
        field.data[ci] = eltype(field.data)(global_fill_value(grid, field, ci))
    end
    return field
end

reshape_like(a, template) = reshape(a, size(template))

@testset "single-field save_pvtk + ReadVTK round trip (N=$N)" for N in (2, 3)
    topo = topology(ntuple(_ -> 0, N); comm = comm, periods = ntuple(_ -> false, N))
    nlocal = ntuple(d -> d == 1 ? 8 : 6, N)
    hw = ntuple(d -> d == 1 ? 2 : 1, N)
    g = grid(topo, nlocal)

    f = field(zeros(nlocal...); halowidths = hw)
    fill_globally!(g, f)

    path = joinpath(tempdir(), "halogrids_pvtk_single_N$(N)_n$(nprocs)")
    cleanup!(path)

    saved = save_pvtk(g, f, path; name = "x")
    MPI.Barrier(comm)

    piece_path = saved[end]
    @test isfile(piece_path)
    vtk = ReadVTK.VTKFile(piece_path)
    pdata = ReadVTK.get_point_data(vtk)
    got = reshape_like(ReadVTK.get_data_reshaped(pdata["x"]), f.data)
    @test got == f.data

    if rank == 0
        pvtk_path = saved[1]
        @test endswith(pvtk_path, ".pvti")
        @test isfile(pvtk_path)
        pvtkfile = ReadVTK.PVTKFile(pvtk_path; dir = dirname(path))
        @test length(pvtkfile) == nprocs

        nglobal_shape = ntuple(d -> nglobal(g, f, d), N)
        pdata_p = ReadVTK.get_point_data(pvtkfile)
        assembled = reshape(ReadVTK.get_data_reshaped(pdata_p["x"]), nglobal_shape)
        for ci in CartesianIndices(assembled)
            @test assembled[ci] == eltype(f.data)(fill_formula(Tuple(ci)))
        end
    end

    MPI.Barrier(comm)
    cleanup!(path)
end

@testset "multi-field save_pvtk with explicit names and mixed dtypes (N=3)" begin
    topo = topology((0, 0, 0); comm = comm, periods = (false, false, false))
    g = grid(topo, (8, 6, 6))
    hw = (2, 1, 1)

    fu = field(zeros(Float64, 8, 6, 6); halowidths = hw)
    fv = field(zeros(Float32, 8, 6, 6); halowidths = hw)
    fill_globally!(g, fu)
    fill_globally!(g, fv)

    path = joinpath(tempdir(), "halogrids_pvtk_multi_n$(nprocs)")
    cleanup!(path)

    saved = save_pvtk(g, (fu, fv), path; names = ("u", "v"))
    MPI.Barrier(comm)

    piece_path = saved[end]
    vtk = ReadVTK.VTKFile(piece_path)
    pdata = ReadVTK.get_point_data(vtk)
    @test reshape_like(ReadVTK.get_data_reshaped(pdata["u"]), fu.data) == fu.data
    @test reshape_like(ReadVTK.get_data_reshaped(pdata["v"]), fv.data) == fv.data

    MPI.Barrier(comm)
    cleanup!(path)
end

@testset "multi-field save_pvtk falls back to positional names when omitted" begin
    topo = topology((0, 0); comm = comm, periods = (false, false))
    g = grid(topo, (6, 6))
    f1 = field(fill(1.0, 6, 6); halowidths = (1, 1))
    f2 = field(fill(2.0, 6, 6); halowidths = (1, 1))

    path = joinpath(tempdir(), "halogrids_pvtk_default_names_n$(nprocs)")
    cleanup!(path)

    saved = save_pvtk(g, (f1, f2), path)
    MPI.Barrier(comm)

    piece_path = saved[end]
    vtk = ReadVTK.VTKFile(piece_path)
    pdata = ReadVTK.get_point_data(vtk)
    @test reshape_like(ReadVTK.get_data_reshaped(pdata["field1"]), f1.data) == f1.data
    @test reshape_like(ReadVTK.get_data_reshaped(pdata["field2"]), f2.data) == f2.data

    MPI.Barrier(comm)
    cleanup!(path)
end

@testset "only the ismain rank writes the .pvti header" begin
    topo = topology((0, 0); comm = comm, periods = (false, false))
    g = grid(topo, (6, 6))
    f = field(zeros(6, 6); halowidths = (1, 1))
    fill_globally!(g, f)

    path = joinpath(tempdir(), "halogrids_pvtk_ismain_n$(nprocs)")
    cleanup!(path)

    saved = save_pvtk(g, f, path; name = "x")
    if rank == 0
        @test length(saved) == 2
        @test endswith(saved[1], ".pvti")
        pvtkfile = ReadVTK.PVTKFile(saved[1]; dir = dirname(path))
        @test length(pvtkfile) == nprocs
    else
        @test length(saved) == 1
        @test !endswith(saved[1], ".pvti")
    end

    MPI.Barrier(comm)
    cleanup!(path)
end

@testset "validation and safety" begin
    topo = topology((0, 0); comm = comm, periods = (false, false))
    g = grid(topo, (6, 6))
    f = field(zeros(6, 6); halowidths = (1, 1))
    fill_globally!(g, f)

    @testset "periodic topology rejected" begin
        ptopo = topology((0, 0); comm = comm, periods = (true, false))
        pg = grid(ptopo, (6, 6))
        pf = field(zeros(6, 6); halowidths = (1, 1))
        path = joinpath(tempdir(), "halogrids_pvtk_periodic_n$(nprocs)")
        cleanup!(path)
        @test_throws Exception save_pvtk(pg, pf, path)
        @test !isfile(path * ".pvti")
        @test !isdir(path)
    end

    @testset "duplicate field names rejected before any file is touched" begin
        f1 = field(zeros(6, 6); halowidths = (1, 1))
        f2 = field(zeros(6, 6); halowidths = (1, 1))
        path = joinpath(tempdir(), "halogrids_pvtk_dupname_n$(nprocs)")
        cleanup!(path)
        @test_throws Exception save_pvtk(g, (f1, f2), path; names = ("dup", "dup"))
        @test !isfile(path * ".pvti")
        @test !isdir(path)
    end

    @testset "mixed halowidths across a batch rejected" begin
        f1 = field(zeros(6, 6); halowidths = (1, 1))
        f2 = field(zeros(6, 6); halowidths = (2, 1))
        path = joinpath(tempdir(), "halogrids_pvtk_mixedhw_n$(nprocs)")
        cleanup!(path)
        @test_throws Exception save_pvtk(g, (f1, f2), path; names = ("a", "b"))
        @test !isfile(path * ".pvti")
        @test !isdir(path)
    end

    @testset "non-CPU-strided field data is rejected before touching the file" begin
        bad = field(falses(6, 6); halowidths = (1, 1))
        path = joinpath(tempdir(), "halogrids_pvtk_badtype_n$(nprocs)")
        cleanup!(path)
        @test_throws Exception save_pvtk(g, bad, path)
        @test !isfile(path * ".pvti")
        @test !isdir(path)
    end

    @testset "invalid dx rejected" begin
        path = joinpath(tempdir(), "halogrids_pvtk_dx_n$(nprocs)")
        cleanup!(path)
        @test_throws Exception save_pvtk(g, f, path; dx = (1.0, -1.0))
        @test_throws Exception save_pvtk(g, f, path; dx = (1.0,))
        @test !isfile(path * ".pvti")
        @test !isdir(path)
    end
end

MPI.Barrier(comm)
rank == 0 && println("test_pvtk_io.jl: all tests passed on $nprocs ranks.")
MPI.Finalize()
