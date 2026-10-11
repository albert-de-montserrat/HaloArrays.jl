# 3D halo exchange of one N^3 Float64 CuArray (halo width 1): CUDA-aware MPI vs
# host staging vs the HaloArrays.jl CUDA extension.
#
#   srun -n P julia --project=. bench_halo_cudaaware.jl OUT.csv [N] [niter]
#
# Modes (all exchange the same six faces, dimension by dimension so edges and
# corners propagate):
#   haloarrays : `update_halo!` from this package (host-staged transport)
#   staged     : hand-written pack kernel -> D2H -> MPI on host -> H2D -> unpack
#   cudaaware  : hand-written pack kernel -> MPI directly on the device buffers
#                -> unpack. Skipped unless the MPI library reports CUDA support
#                (or HA_FORCE_CUDAAWARE=1).
# `staged` and `cudaaware` share everything except the transfer, so their
# difference is what a CUDA-aware transport could save; `haloarrays` shows how
# close the package's current path is to `staged`.
# Every mode is verified (each face halo must hold the neighbor's rank) before it
# is timed. Output columns:
#   mode,nprocs,N,niter,t_step_ms,face_bytes,halo_gb_per_s
# `halo_gb_per_s` is bytes sent per rank per step divided by the step time.
import Pkg
Pkg.activate(@__DIR__; io = devnull)
using CUDA
import MPI
using HaloArrays
include(joinpath(@__DIR__, "cuda_aware_common.jl"))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
select_device!(comm)

out = length(ARGS) >= 1 ? ARGS[1] : "halo_cudaaware.csv"
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 256
niter = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
nwarm = 5

# ---- Cartesian layout shared by the hand-written modes ----------------------
dims = MPI.Dims_create(nprocs, zeros(Int, 3))
cart = MPI.Cart_create(comm, dims; periodic = [false, false, false], reorder = false)
neighbors = ntuple(d -> Int.(MPI.Cart_shift(cart, d - 1, 1)), 3)   # (lo, hi) ranks, PROC_NULL if none
const NONE = MPI.PROC_NULL
bytes_per_face = N * N * sizeof(Float64)
nsend = sum(d -> count(!=(NONE), neighbors[d]), 1:3)               # faces this rank sends per step

# Slab ranges along dim d (halo width 1).
slab(d, i) = ntuple(k -> k == d ? (i:i) : (1:N), 3)

struct Faces
    send::Vector{Any}      # [d][side] pack buffers (device)
    recv::Vector{Any}
    hsend::Vector{Any}     # pinned host mirrors for the staged mode
    hrecv::Vector{Any}
end
function Faces()
    mk() = [CUDA.zeros(Float64, N, N) for _ in 1:6]
    mkh() = [CUDA.pin(zeros(Float64, N, N)) for _ in 1:6]
    return Faces(mk(), mk(), mkh(), mkh())
end
slot(d, side) = 2 * (d - 1) + side     # side 1 = lo, 2 = hi

# One dimension of the exchange. Tags: a message travelling toward lo carries
# tag 2d-1, toward hi carries 2d.
function exchange_dim!(A, F::Faces, d, staged::Bool)
    lo, hi = neighbors[d]
    reqs = MPI.Request[]
    rbuf(side) = staged ? F.hrecv[slot(d, side)] : F.recv[slot(d, side)]
    sbuf(side) = staged ? F.hsend[slot(d, side)] : F.send[slot(d, side)]
    lo != NONE && push!(reqs, MPI.Irecv!(rbuf(1), cart; source = lo, tag = 2d))       # from lo, moving toward hi
    hi != NONE && push!(reqs, MPI.Irecv!(rbuf(2), cart; source = hi, tag = 2d - 1))   # from hi, moving toward lo
    # pack (interior layer next to each halo)
    for (side, idx) in ((1, 2), (2, N - 1))
        neighbors[d][side] == NONE && continue
        copyto!(reshape(F.send[slot(d, side)], ntuple(k -> k == d ? 1 : N, 3)), view(A, slab(d, idx)...))
    end
    if staged
        for side in 1:2
            neighbors[d][side] != NONE && copyto!(F.hsend[slot(d, side)], F.send[slot(d, side)])
        end
    end
    CUDA.synchronize()          # pack (and D2H) must finish before MPI reads the buffers
    lo != NONE && push!(reqs, MPI.Isend(sbuf(1), cart; dest = lo, tag = 2d - 1))
    hi != NONE && push!(reqs, MPI.Isend(sbuf(2), cart; dest = hi, tag = 2d))
    MPI.Waitall(reqs)
    for (side, idx) in ((1, 1), (2, N))
        neighbors[d][side] == NONE && continue
        staged && copyto!(F.recv[slot(d, side)], F.hrecv[slot(d, side)])
        view(A, slab(d, idx)...) .= reshape(F.recv[slot(d, side)], ntuple(k -> k == d ? 1 : N, 3))
    end
    CUDA.synchronize()
    return nothing
end

function exchange!(A, F, staged)
    for d in 1:3
        exchange_dim!(A, F, d, staged)
    end
    return nothing
end

# ---- verification -----------------------------------------------------------
# Fill the whole array with this rank's id; each face halo that has a neighbor
# must then hold the neighbor's id. Checked at the face center, where edge and
# corner propagation cannot interfere.
function verify(A, mode, nb)
    c = N ÷ 2
    ok = true
    for d in 1:3, side in 1:2
        nb[d][side] == NONE && continue
        idx = side == 1 ? 1 : N
        pos = ntuple(k -> k == d ? idx : c, 3)
        got = Array(view(A, pos...))[1]
        got == nb[d][side] || (ok = false; println("rank $rank mode=$mode: dim $d side $side got $got, want $(nb[d][side])"))
    end
    ok = MPI.Allreduce(ok ? 1 : 0, min, comm) == 1
    ok || error("verification failed in mode=$mode")
    return nothing
end

function bench(step!, A, fillA!)
    fillA!()
    for _ in 1:nwarm
        step!()
    end
    CUDA.synchronize()
    MPI.Barrier(comm)
    t0 = time_ns()
    for _ in 1:niter
        step!()
    end
    CUDA.synchronize()
    return max_over_ranks(comm, (time_ns() - t0) / 1e9 / niter)
end

function report(mode, t, nsend_mode)
    if rank == 0
        gbs = nsend_mode * bytes_per_face / 1e9 / t
        println(rpad(mode, 12), "t_step = ", round(t * 1e3; digits = 3), " ms   ", round(gbs; digits = 2), " GB/s/rank (sent)")
        append_csv(out, "mode,nprocs,N,niter,t_step_ms,face_bytes,halo_gb_per_s",
            (mode, nprocs, N, niter, t * 1e3, bytes_per_face, gbs))
    end
end

rank == 0 && println("N=$N  nprocs=$nprocs  dims=$(Tuple(dims))  MPI.has_cuda()=$(MPI.has_cuda())  niter=$niter\n")

# ---- mode: haloarrays -------------------------------------------------------
let
    topo = topology((0, 0, 0); comm = comm, periods = (false, false, false))
    g = grid(topo, (N, N, N))
    A = CUDA.zeros(Float64, N, N, N)
    fillA!() = fill!(A, Float64(topo.rank))
    fillA!(); update_halo!(g, A)
    verify(A, "haloarrays", topo.neighbors)
    t = bench(() -> update_halo!(g, A), A, fillA!)
    report("haloarrays", t, sum(d -> count(has_neighbor, topo.neighbors[d]), 1:3))
end

# ---- modes: staged, cudaaware ----------------------------------------------
cart_rank = MPI.Comm_rank(cart)
hand_modes = cuda_aware_enabled() ? ["staged", "cudaaware"] : ["staged"]
for mode in hand_modes
    staged = mode == "staged"
    A = CUDA.zeros(Float64, N, N, N)
    F = Faces()
    fillA!() = fill!(A, Float64(cart_rank))
    fillA!(); exchange!(A, F, staged)
    verify(A, mode, neighbors)
    report(mode, bench(() -> exchange!(A, F, staged), A, fillA!), nsend)
end
if rank == 0 && !cuda_aware_enabled()
    println("\nMPI does not report CUDA support: mode=cudaaware skipped (set HA_FORCE_CUDAAWARE=1 to force).")
end

MPI.Finalize()
