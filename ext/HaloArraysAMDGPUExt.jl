# Thin glue module, structurally identical to ext/HaloArraysCUDAExt.jl (see
# that file's comment for the shared design rationale, including what this
# deliberately doesn't do: device selection and ROCm-aware MPI transport).
# Untested on this machine (no AMD GPU available) — see
# src/AMDGPUExt/shared.jl and REDESIGN_PLAN.md.
module HaloArraysAMDGPUExt
    include(joinpath(@__DIR__, "..", "src", "AMDGPUExt", "shared.jl"))
    include(joinpath(@__DIR__, "..", "src", "AMDGPUExt", "halo_exchange.jl"))
end
