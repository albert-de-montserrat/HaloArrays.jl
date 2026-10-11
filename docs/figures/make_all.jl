# Regenerate every documentation figure into docs/src/assets.
# Run from the repository root: julia --project=docs/figures docs/figures/make_all.jl
for f in ("topology", "overlaps", "corners", "faces", "timeline", "boundary_width")
    include(joinpath(@__DIR__, f * ".jl"))
end
