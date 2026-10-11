# Point-to-point ping-pong between ranks 0 and 1, device buffers vs host staging.
#
#   srun -n 2 julia --project=. bench_pingpong.jl OUT.csv [niter] [max_log2_bytes]
#
# Modes:
#   device  : CuArray handed straight to MPI (needs CUDA-aware MPI)
#   staged  : D2H copy into a pinned host buffer, MPI on host memory, H2D copy
#             on the receiver -- the transport HaloArrays' CUDA extension uses
#   host    : host buffers only, no GPU copies (MPI floor)
# Reported time is the one-way latency (half the round trip) of the full path,
# including the copies in `staged`. Run it twice to separate the interconnects:
# two ranks on one node (NVLink/PCIe) and two ranks on different nodes (network).
# Output columns: mode,bytes,t_us,gb_per_s,same_node
import Pkg
Pkg.activate(@__DIR__; io = devnull)
using CUDA
import MPI
include(joinpath(@__DIR__, "cuda_aware_common.jl"))

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
MPI.Comm_size(comm) == 2 || error("bench_pingpong.jl needs exactly 2 ranks.")
select_device!(comm)

out = length(ARGS) >= 1 ? ARGS[1] : "pingpong.csv"
niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
maxlog = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 28   # 256 MiB
nwarm = 10

lcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
same_node = MPI.Comm_size(lcomm) == 2
peer = 1 - rank

# One round trip: rank 0 sends then receives, rank 1 receives then sends.
function roundtrip!(send, recv, comm, peer, rank, pre, post)
    if rank == 0
        pre(); MPI.Send(send, comm; dest = peer, tag = 0)
        MPI.Recv!(recv, comm; source = peer, tag = 1); post()
    else
        MPI.Recv!(recv, comm; source = peer, tag = 0); post()
        pre(); MPI.Send(send, comm; dest = peer, tag = 1)
    end
end

function time_mode(mode, nbytes, niter)
    n = max(nbytes ÷ 8, 1)
    dsend = CUDA.fill(Float64(rank), n)
    drecv = CUDA.zeros(Float64, n)
    hsend = CUDA.pin(fill(Float64(rank), n))
    hrecv = CUDA.pin(zeros(Float64, n))
    CUDA.synchronize()
    if mode == "device"
        send, recv = dsend, drecv
        pre = () -> CUDA.synchronize()
        post = () -> nothing
    elseif mode == "staged"
        send, recv = hsend, hrecv
        pre = () -> (copyto!(hsend, dsend); CUDA.synchronize())
        post = () -> (copyto!(drecv, hrecv); CUDA.synchronize())
    else
        send, recv = hsend, hrecv
        pre = () -> nothing
        post = () -> nothing
    end
    for _ in 1:nwarm
        roundtrip!(send, recv, comm, peer, rank, pre, post)
    end
    MPI.Barrier(comm)
    t0 = time_ns()
    for _ in 1:niter
        roundtrip!(send, recv, comm, peer, rank, pre, post)
    end
    t = (time_ns() - t0) / 1e9 / niter / 2
    t = max_over_ranks(comm, t)
    # correctness: the receive buffer (on the device after `post`) holds the peer's rank
    got = mode == "host" ? hrecv : drecv
    all(Array(got) .== peer) || error("ping-pong data mismatch in mode=$mode, bytes=$nbytes")
    return t
end

modes = cuda_aware_enabled() ? ["device", "staged", "host"] : ["staged", "host"]
if rank == 0 && !cuda_aware_enabled()
    println("MPI does not report CUDA support: skipping mode=device (set HA_FORCE_CUDAAWARE=1 to force).")
end

rank == 0 && println(rpad("mode", 8), rpad("bytes", 12), rpad("t_us", 12), "GB/s")
for lg in 3:maxlog, mode in modes
    nbytes = 1 << lg
    reps = nbytes >= (1 << 24) ? max(niter ÷ 10, 5) : niter   # fewer iterations for huge messages
    t = time_mode(mode, nbytes, reps)
    if rank == 0
        gbs = nbytes / 1e9 / t
        println(rpad(mode, 8), rpad(nbytes, 12), rpad(round(t * 1e6; digits = 2), 12), round(gbs; digits = 3))
        append_csv(out, "mode,bytes,t_us,gb_per_s,same_node", (mode, nbytes, t * 1e6, gbs, same_node))
    end
end

MPI.Finalize()
