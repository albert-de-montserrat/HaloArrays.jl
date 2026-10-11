# Full exchange runs one Cartesian dimension at a time, so corner halo cells receive
# data from the diagonal neighbor in two hops.
include("common.jl")

function main()
    nx, ny = 9, 8
    state(stage) = [begin
        edge_x = i == 1 || i == nx
        edge_y = j == 1 || j == ny
        if !edge_x && !edge_y
            C.inner
        elseif edge_x && !edge_y
            stage >= 1 ? C.send : C.halo
        elseif edge_y && !edge_x
            stage >= 2 ? C.send : C.halo
        else
            stage >= 2 ? C.accent : C.halo
        end
    end for i in 1:nx, j in 1:ny]

    fig = new_figure(1300, 520)
    Label(fig[0, 1:3], "Full exchange fills corners in dimension order"; fontsize = 22, font = :bold, color = C.ink)
    titles = ("before exchange", "after dimension 1 (x)", "after dimension 2 (y)")
    notes = ("halo cells hold stale data",
        "left and right halos filled\n(full columns are sent)",
        "top and bottom halos filled from\nrows that already hold x data:\ncorners (red) arrive via two hops")
    for (k, stage) in enumerate((0, 1, 2))
        ax = blank_axis(fig[1, k]; title = titles[k], titlesize = 18, titlecolor = C.ink, titlealign = :center)
        cells!(ax, 0, 0, state(stage))
        text!(ax, nx / 2, -0.8; text = notes[k], align = (:center, :top), color = C.ink, fontsize = 15)
        ylims!(ax, -3.6, ny + 0.2)
    end
    elems, labs = legend_patches((C.halo, "stale halo"), (C.send, "halo filled by a face neighbor"),
        (C.accent, "corner halo (diagonal neighbor data)"), (C.inner, "interior"))
    Legend(fig[2, 1:3], elems, labs; orientation = :horizontal, framevisible = false, labelcolor = C.ink)
    save_fig("corners", fig)
end

main()
