# A 3 x 2 Cartesian topology, periodic in x, non-periodic in y.
include("common.jl")

function main()
    nb = (3, 2)          # topology((3, 2); periods = (true, false))
    nl = (7, 6)          # local array size including halo
    gap = 2.2
    fig = new_figure(1100, 560)
    ax = blank_axis(fig[1, 1])
    Label(fig[0, 1], "topology((3, 2); periods = (true, false))"; fontsize = 22, font = :bold, color = C.ink)

    origin(i, j) = (i * (nl[1] + gap), j * (nl[2] + gap))
    for i in 0:nb[1]-1, j in 0:nb[2]-1
        colors = [(a == 1 || a == nl[1] || b == 1 || b == nl[2]) ? C.halo : C.inner for a in 1:nl[1], b in 1:nl[2]]
        x0, y0 = origin(i, j)
        cells!(ax, x0, y0, colors)
        text!(ax, x0 + nl[1] / 2, y0 + nl[2] / 2; text = "rank
($i, $j)", align = (:center, :center),
            color = C.ink, fontsize = 18, font = :bold)
    end

    # neighbor exchange in x (inner links) and y
    for j in 0:nb[2]-1, i in 0:nb[1]-2
        x0, y0 = origin(i, j)
        xr = x0 + nl[1]
        ym = y0 + nl[2] / 2
        arrow!(ax, (xr + 0.1, ym - 0.5), (xr + gap - 0.1, ym - 0.5);)
        arrow!(ax, (xr + gap - 0.1, ym + 0.5), (xr + 0.1, ym + 0.5))
    end
    for i in 0:nb[1]-1
        x0, y0 = origin(i, 0)
        yt = y0 + nl[2]
        xm = x0 + nl[1] / 2
        arrow!(ax, (xm - 0.5, yt + 0.1), (xm - 0.5, yt + gap - 0.1))
        arrow!(ax, (xm + 0.5, yt + gap - 0.1), (xm + 0.5, yt + 0.1))
    end

    # periodic wrap in x: dashed path below the bottom row, right edge to left edge
    xl0 = origin(0, 0)[1]
    xr0 = origin(nb[1] - 1, 0)[1] + nl[1]
    ym = origin(0, 0)[2] + nl[2] / 2
    yb = origin(0, 0)[2] - 1.8
    lines!(ax, [Point2f(xr0 + 0.1, ym), Point2f(xr0 + 1.0, ym), Point2f(xr0 + 1.0, yb), Point2f(xl0 - 1.0, yb), Point2f(xl0 - 1.0, ym), Point2f(xl0 - 0.17, ym)];
        color = C.accent, linewidth = 2.5, linestyle = :dash)
    scatter!(ax, [Point2f(xl0 - 0.1, ym)]; marker = :utriangle, rotation = -pi / 2, markersize = 14, color = C.accent)
    scatter!(ax, [Point2f(xr0 + 0.1, ym)]; marker = :utriangle, rotation = pi / 2, markersize = 14, color = C.accent)
    text!(ax, (xl0 + xr0) / 2, yb + 0.3; text = "periodic in x: the last rank neighbors the first (dashed)",
        align = (:center, :bottom), color = C.accent, fontsize = 16)

    Label(fig[2, 1], "Orange: halo cells (overwritten on receive).   Gray: interior.   Arrows: neighbor exchange.\nNon-periodic edges have no neighbor (MPI.PROC_NULL): the application sets those boundaries.";
        color = C.ink, fontsize = 15, justification = :center)
    save_fig("topology", fig)
end

main()
