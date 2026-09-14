# Thin glue module, mirroring ImplicitGlobalGrid.jl's ext/*_CUDAExt.jl: the real
# code lives in src/CUDAExt/ so it reads like ordinary package source, not
# extension boilerplate.
#
# Two things this extension deliberately does NOT do:
# - Device selection: which GPU a rank uses is entirely the caller's
#   responsibility (call `CUDA.device!(...)` before building arrays), same as
#   Topology/Grid already don't manage any other MPI-rank-to-resource
#   assignment.
# - CUDA-aware (GPU-direct) MPI transport: `stage_send!`/`stage_recv!` in
#   src/CUDAExt/halo_exchange.jl are the seam where that would plug in (make
#   them no-ops and let post_isend!/post_irecv! send/recv the device buffer
#   directly), but it isn't implemented here since Microsoft MPI (this
#   project's only local MPI) isn't CUDA-aware and the path can't be validated
#   without one.
module HaloArraysCUDAExt
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "halo_exchange.jl"))
end
