# Included by the CPU and CUDA face-exchange workers. The expected value is
# computed from Cartesian coordinates and source indices, independently of
# packing helpers. Include after using Test, MPI, and HaloGrids.
function face_initial(::Type{T}, rank, shape) where {T}
    return reshape(T[T(1_000_000rank + i) for i in 1:prod(shape)], shape)
end

function expected_face(topo, shape, widths, dims, ci)
    N = ndims(topo)
    halo_dims = Int[]
    for d in 1:N
        (ci[d] <= widths[d] || ci[d] > shape[d] - widths[d]) && push!(halo_dims, d)
    end
    rank = topo.rank
    source = collect(Tuple(ci))
    if length(halo_dims) == 1 && halo_dims[1] in dims
        d = halo_dims[1]
        side = ci[d] <= widths[d] ? 1 : 2
        neighbor = topo.neighbors[d][side]
        if has_neighbor(neighbor)
            rank = neighbor
            source[d] = side == 1 ? shape[d] - 2widths[d] + ci[d] :
                ci[d] - shape[d] + 2widths[d]
        end
    end
    return 1_000_000rank + LinearIndices(shape)[CartesianIndex(source...)]
end

function check_faces(topo, f, dims)
    A = Array(f.data)
    return all(A[ci] == eltype(f)(expected_face(topo, size(A), f.halowidths, dims, ci))
        for ci in CartesianIndices(A))
end

function face_exchange_checks(to_backend, synchronize)
    @testset "face source indices and untouched cells" begin
        for N in (2, 3), periods in (ntuple(_ -> false, N), ntuple(_ -> true, N),
                ntuple(d -> isodd(d), N)),
                dims in (ntuple(identity, N), Tuple(reverse(1:N)), (1,))
            topo = topology(ntuple(_ -> 0, N); periods)
            shape = ntuple(d -> 7 + d, N)
            g = grid(topo, shape)
            a = field(to_backend(face_initial(Float64, topo.rank, shape)))
            widths = ntuple(d -> d == 1 ? 2 : 1, N)
            b = field(to_backend(face_initial(Float32, topo.rank, shape)); halowidths=widths)
            for split in (false, true)
                copyto!(a.data, face_initial(Float64, topo.rank, shape))
                copyto!(b.data, face_initial(Float32, topo.rank, shape))
                if split
                    h = halo_exchange_start!(g, a, b; dims, region=:faces)
                    synchronize()
                    # All unpacking, even self-periodic, is deferred.
                    @test Array(a.data) == face_initial(Float64, topo.rank, shape)
                    @test g.excache.pending
                    @test_throws ErrorException update_halo!(g, a)
                    @test_throws ErrorException halo_exchange_start!(g, a; region=:faces)
                    halo_exchange_finish!(h)
                else
                    update_halo!(g, a, b; dims, region=:faces)
                end
                synchronize()
                @test check_faces(topo, a, dims)
                @test check_faces(topo, b, dims)
                @test !g.excache.pending
            end
            # Changing field count remaps slots; changing region reuses slots.
            update_halo!(g, a; dims, region=:faces)
            @test check_faces(topo, a, dims)
            update_halo!(g, a)
            copyto!(a.data, face_initial(Float64, topo.rank, shape))
            update_halo!(g, a; region=:faces)
            @test check_faces(topo, a, ntuple(identity, N))
        end
    end
    @testset "face validation and shared pending guard" begin
        topo = topology((0, 0); periods=(true, true))
        g = grid(topo, (8, 9))
        f = field(to_backend(face_initial(Float64, topo.rank, g.nlocal)))
        @test_throws ErrorException update_halo!(g, f; region=:unknown)
        @test_throws ErrorException update_halo!(g, f; region=:faces, dims=(1, 1))
        @test_throws ErrorException update_halo!(g, f; region=:faces, dims=())
        @test_throws ErrorException update_halo!(g, f; region=:faces, dims=(3,))
        @test_throws ErrorException update_halo!(g, f; region=:faces, dims=(1.0,))
        @test_throws ErrorException update_halo!(g, f; region=:faces, dims=("1",))
        @test_throws ErrorException update_halo!(g, f, f; region=:faces)
        @test !g.excache.pending
        narrow = grid(topo, (4, 9))
        wide = field(to_backend(zeros(4, 9)); halowidths=(2, 1))
        @test_throws ErrorException update_halo!(narrow, wide; region=:faces, dims=(1,))
        @test !narrow.excache.pending
        h = halo_exchange_start!(g, f)
        @test_throws ErrorException halo_exchange_start!(g, f; region=:faces)
        halo_exchange_finish!(h)
        A = to_backend(face_initial(Float64, topo.rank, g.nlocal))
        update_halo!(g, A; region=:faces)
        @test check_faces(topo, field(A), (1, 2))
        h = halo_exchange_start!(g, A; region=:faces)
        halo_exchange_finish!(h)
        @test !g.excache.pending
    end
    @testset "trailing component dimensions" begin
        topo = topology((0, 0); periods=(true, true))
        shape = (8, 9, 3)
        g = grid(topo, shape[1:2])
        f = field(to_backend(face_initial(Float64, topo.rank, shape)))
        update_halo!(g, f; region=:faces)
        @test check_faces(topo, f, (1, 2))
    end
    @testset "empty transverse interior" begin
        topo = topology((0, 0); periods=(true, true))
        g = grid(topo, (8, 2))
        original = face_initial(Float64, topo.rank, g.nlocal)
        f = field(to_backend(original))
        update_halo!(g, f; dims=(1,), region=:faces)
        @test Array(f.data) == original
        @test !g.excache.pending
    end
    if to_backend !== identity
        @testset "mixed host and device fields" begin
            topo = topology((0, 0); periods=(true, true))
            g = grid(topo, (8, 9))
            host = field(face_initial(Float64, topo.rank, g.nlocal))
            device = field(to_backend(face_initial(Float32, topo.rank, g.nlocal)))
            update_halo!(g, host, device; region=:faces)
            @test check_faces(topo, host, (1, 2))
            @test check_faces(topo, device, (1, 2))
        end
    end
    @testset "concurrent independent topologies" begin
        topo1 = topology((0, 0); periods=(true, true))
        topo2 = topology((0, 0); periods=(true, true))
        g1, g2 = grid(topo1, (8, 9)), grid(topo2, (11, 12))
        f1 = field(to_backend(face_initial(Float64, topo1.rank, g1.nlocal)))
        f2 = field(to_backend(face_initial(Float32, topo2.rank, g2.nlocal)))
        h1 = halo_exchange_start!(g1, f1; region=:faces)
        h2 = halo_exchange_start!(g2, f2; region=:faces)
        halo_exchange_finish!(h2)
        halo_exchange_finish!(h1)
        @test check_faces(topo1, f1, (1, 2))
        @test check_faces(topo2, f2, (1, 2))
    end
    if Threads.nthreads() > 1 && to_backend === identity
        @testset "threaded face packing" begin
            topo = topology((0, 0, 0); periods=(true, true, true))
            g = grid(topo, (5, 258, 258); threaded_pack=true)
            f = field(face_initial(Float64, topo.rank, g.nlocal))
            update_halo!(g, f; region=:faces)
            @test check_faces(topo, f, (1, 2, 3))
        end
    end
    synchronize()
    return nothing
end
