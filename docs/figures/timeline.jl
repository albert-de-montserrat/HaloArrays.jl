# Schematic of the split-phase exchange. Bar lengths are illustrative, not measured.
include("common.jl")

function bar!(ax, x0, x1, y, color, label; textcolor = C.ink)
    poly!(ax, Rect2f(x0, y, x1 - x0, 0.8); color, strokecolor = C.bg, strokewidth = 2)
    text!(ax, (x0 + x1) / 2, y + 0.4; text = label, align = (:center, :center), color = textcolor, fontsize = 15)
end

function main()
    fig = new_figure(1200, 560)
    Label(fig[0, 1], "Hiding communication with halo_exchange_start! / halo_exchange_finish!"; fontsize = 22, font = :bold, color = C.ink)
    ax = Axis(fig[1, 1]; backgroundcolor = :transparent, yticks = ([0.4, 1.4, 3.4, 4.4], ["MPI", "compute", "MPI", "compute"]),
        xlabel = "time (schematic, not to scale)", xticksvisible = false, xticklabelsvisible = false,
        yticksvisible = false, leftspinevisible = false, topspinevisible = false, rightspinevisible = false,
        bottomspinecolor = C.mute, xgridvisible = false, ygridvisible = false,
        yticklabelcolor = C.ink, xlabelcolor = C.ink)
    # blocking
    bar!(ax, 0, 6, 1.0, C.inner, "kernel on the whole array")
    bar!(ax, 6, 9, 0.0, C.halo, "update_halo!")
    text!(ax, 0, 2.15; text = "blocking: update_halo!", align = (:left, :bottom), color = C.ink, fontsize = 18, font = :bold)
    # split phase
    bar!(ax, 0, 1.5, 4.0, C.inner, "outer")
    bar!(ax, 1.5, 2.7, 3.0, C.halo, "dims 1..N-1")
    bar!(ax, 2.7, 5.4, 4.0, C.inner, "inner box")
    bar!(ax, 2.7, 5.4, 3.0, C.send, "last dim in flight")
    text!(ax, 0, 5.6; text = "split phase: start!, inner work, finish!", align = (:left, :bottom), color = C.ink, fontsize = 18, font = :bold)
    for x in (1.5, 5.4)
        lines!(ax, [Point2f(x, 2.9), Point2f(x, 5.0)]; color = C.mute, linestyle = :dot, linewidth = 1.5)
    end
    text!(ax, 1.5, 5.0; text = "start!", align = (:center, :bottom), color = C.ink, fontsize = 14)
    text!(ax, 5.4, 5.0; text = "finish!", align = (:center, :bottom), color = C.ink, fontsize = 14)
    limits!(ax, -0.2, 9.4, -0.3, 6.4)
    Label(fig[2, 1], "The outer shell runs first, then start! completes every dimension except the last synchronously.\nOnly the last dimension overlaps the inner launch, so the saving depends on how large the inner box is.";
        color = C.ink, fontsize = 15, justification = :center)
    save_fig("timeline", fig)
end

main()
