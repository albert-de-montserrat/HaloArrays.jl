# Thin glue module, mirroring ImplicitGlobalGrid.jl's ext/*_CUDAExt.jl: the real
# code lives in src/CUDAExt/ so it reads like ordinary package source, not
# extension boilerplate.
#
# Device selection is deliberately not done here: which GPU a rank uses is
# entirely the caller's responsibility (call `CUDA.device!(...)` before
# building arrays), same as Topology/Grid already don't manage any other
# MPI-rank-to-resource assignment.
module HaloGridsCUDAExt
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "CUDAExt", "halo_exchange.jl"))
end
