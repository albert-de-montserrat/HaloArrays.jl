# Thin glue module, mirroring ext/HaloGridsCUDAExt.jl: the real code lives in
# src/CellArraysExt/ so it reads like ordinary package source, not extension
# boilerplate.
#
# What this extension does NOT do:
# - No new pack/unpack/GPU-staging code. CellArrays.jl's own `field(A, (j,))`
#   already returns a real dense (B=0) or strided (B=1) plain-array view per
#   cell component; this extension only decomposes a `CellArray` into those
#   views and hands them to HaloGrids' existing `Field`/exchange machinery
#   (src/halo_exchange.jl) unmodified. A `CuCellArray`'s component views are
#   therefore expected to reach the existing HaloGridsCUDAExt dispatch
#   (AnyCuArray) automatically, with no CUDA-specific code written here — see
#   the verification note in src/CellArraysExt/halo_exchange.jl.
# - Name collision, not fixed here: both `HaloGrids.field` (the `Field`
#   constructor) and `CellArrays.field` (the per-component view accessor) are
#   exported under the same name by their respective packages. This extension
#   avoids the clash via qualified imports below, but a downstream script
#   doing `using HaloGrids, CellArrays` together will need to qualify
#   `field(...)` explicitly at the call site.
module HaloGridsCellArraysExt
    include(joinpath(@__DIR__, "..", "src", "CellArraysExt", "halo_exchange.jl"))
end
