# Run with --project=docs/figures.
using CairoMakie

function main()
    colors = ["#d9dfe7", "#eda65a", "#64b5cd"]
    fig = Figure(size=(1200, 600), fontsize=18, backgroundcolor=:white)
    Label(fig[0, :], "Boundary width changes the computation shell, not the halo", fontsize=24)
    nx, ny = 14, 12
    for (panel, (bx, by)) in enumerate(((1, 1), (3, 2)))
        regions = [
            i == 1 || i == nx || j == 1 || j == ny ? 0 :
            i <= 1 + bx || i >= nx - bx || j <= 1 + by || j >= ny - by ? 1 : 2
            for i in 1:nx, j in 1:ny
        ]
        ax = Axis(fig[1, panel]; title="boundary_width = ($bx, $by)",
            xlabel="Local array x index", ylabel="Local array y index",
            xticks=1:2:nx, yticks=1:2:ny, aspect=DataAspect(),
            xgridvisible=false, ygridvisible=false)
        heatmap!(ax, 1:nx, 1:ny, regions; colormap=cgrad(colors, 3; categorical=true), colorrange=(-0.5, 2.5))
        for x in 0.5:1:nx+0.5
            lines!(ax, [x, x], [0.5, ny+0.5]; color=:white, linewidth=1.5)
        end
        for y in 0.5:1:ny+0.5
            lines!(ax, [0.5, nx+0.5], [y, y]; color=:white, linewidth=1.5)
        end
        text!(ax, (nx+1)/2, (ny+1)/2; text="Inner box\n$(nx-2-2bx) × $(ny-2-2by)",
            align=(:center, :center), fontsize=22, color="#143541")
        limits!(ax, 0.5, nx+0.5, 0.5, ny+0.5)
    end
    elements = [PolyElement(color=c) for c in colors]
    Legend(fig[2, :], elements,
        ["Halo (width 1)", "Outer: before exchange", "Inner: exchange pending"];
        orientation=:horizontal, framevisible=false)
    Label(fig[3, :], "Local array: 14 × 12    •    Kernel launch: 12 × 10    •    Both sides use the same width", fontsize=16)
    outdir = joinpath(@__DIR__, "..", "src", "assets")
    mkpath(outdir)
    save(joinpath(outdir, "boundary-width-regions.png"), fig; px_per_unit=2)
    save(joinpath(outdir, "boundary-width-regions.svg"), fig)
end

main()
