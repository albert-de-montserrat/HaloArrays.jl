# Thin glue module, mirroring ImplicitGlobalGrid.jl's ext/*_CUDAExt.jl: the real
# code lives in src/CUDAExt/ so it reads like ordinary package source, not
# extension boilerplate.
#
# Two things this extension deliberately does NOT do:
# - Device selection: which GPU a rank uses is entirely the caller's
#   responsibility (call `CUDA.device!(...)` before building arrays), same as
#   Topology/Grid already don't manage any other MPI-rank-to-resource
#   assignment.
# - CUDA-aware MPI availability: the extension uses MPI.jl's `MPI.has_cuda()`
#   to choose direct device buffers or the host-staging fallback. MPI
#   implementations without automatic detection can set
#   `JULIA_MPI_HAS_CUDA=true` when their CUDA-aware support is configured.
module HaloArraysCUDAExt
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "halo_exchange.jl"))
end
