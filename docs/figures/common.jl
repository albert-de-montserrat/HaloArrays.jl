# Shared palette and helpers for the documentation figures.
# Run any figure script with: julia --project=docs/figures docs/figures/<name>.jl
# or all of them with:        julia --project=docs/figures docs/figures/make_all.jl
using CairoMakie

const C = (
    bg = "#f7f8fa",     # figure background (opaque so SVGs read in light and dark themes)
    inner = "#d9dfe7",  # interior cells
    halo = "#eda65a",   # halo cells (overwritten on receive)
    send = "#64b5cd",   # cells packed and sent to a neighbor
    ink = "#143541",    # text and arrows
    mute = "#8b97a5",   # secondary lines and unavailable cells
    accent = "#c4553a", # emphasis
    grid = "#ffffff",   # cell borders
)

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")

new_figure(w, h) = Figure(size = (w, h), backgroundcolor = C.bg, fontsize = 16, figure_padding = 24)

function blank_axis(pos; kw...)
    ax = Axis(pos; aspect = DataAspect(), backgroundcolor = :transparent, xautolimitmargin = (0.04, 0.04), yautolimitmargin = (0.04, 0.04), kw...)
    hidedecorations!(ax)
    hidespines!(ax)
    return ax
end

"""
    cells!(ax, x0, y0, colors)

Draw one unit square per entry of the color matrix `colors`. Entry `[i, j]` sits
at `(x0 + i - 1, y0 + j - 1)`.
"""
function cells!(ax, x0, y0, colors::AbstractMatrix)
    rects = [Rect2f(x0 + i - 1, y0 + j - 1, 1, 1) for i in axes(colors, 1), j in axes(colors, 2)]
    poly!(ax, vec(rects); color = Makie.to_color.(vec(colors)), strokecolor = C.grid, strokewidth = 1.5)
end

"""
    arrow!(ax, from, to; color, width)

Straight arrow from `from` to `to` (2-tuples).
"""
function arrow!(ax, from, to; color = C.ink, width = 2.5)
    p, q = Point2f(from...), Point2f(to...)
    d = q - p
    n = d / sqrt(sum(abs2, d))
    lines!(ax, [p, q - 0.17f0 * n]; color, linewidth = width)
    scatter!(ax, [q]; marker = :utriangle, rotation = atan(d[2], d[1]) - pi / 2, markersize = 14, color)
end

function save_fig(name, fig)
    mkpath(ASSETS)
    save(joinpath(ASSETS, name * ".svg"), fig)
    save(joinpath(ASSETS, name * ".png"), fig; px_per_unit = 2)
    println("wrote ", name)
end

legend_patches(pairs...) = (
    [PolyElement(color = c, strokecolor = C.grid) for (c, _) in pairs],
    [l for (_, l) in pairs],
)
