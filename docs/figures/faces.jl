# Full exchange (region = :full) versus face exchange (region = :faces).
# Face slabs exclude halo cells of the other Cartesian dimensions (see face_range
# in src/face_exchange.jl), so corner halos are not exchanged.
include("common.jl")

function main()
    nx, ny = 10, 8
    fig = new_figure(1100, 520)
    Label(fig[0, 1:2], "Which halo cells each exchange fills"; fontsize = 22, font = :bold, color = C.ink)
    for (k, (title, faces)) in enumerate((("region = :full", false), ("region = :faces", true)))
        colors = [begin
            ex = i == 1 || i == nx
            ey = j == 1 || j == ny
            if !ex && !ey
                C.inner
            elseif ex && ey
                faces ? C.mute : C.accent
            else
                C.send
            end
        end for i in 1:nx, j in 1:ny]
        ax = blank_axis(fig[1, k]; title, titlesize = 19, titlecolor = C.ink, titlefont = :bold)
        cells!(ax, 0, 0, colors)
        text!(ax, nx / 2, -0.8;
            text = faces ? "corners are not exchanged;\nall dimensions can stay pending together" :
                           "corners filled via dimension order;\nonly the last dimension stays pending",
            align = (:center, :top), color = C.ink, fontsize = 15)
        ylims!(ax, -3.0, ny + 0.2)
    end
    elems, labs = legend_patches((C.send, "face halo cells"), (C.accent, "corner halo, filled"),
        (C.mute, "corner halo, untouched"), (C.inner, "interior"))
    Legend(fig[2, 1:2], elems, labs; orientation = :horizontal, framevisible = false, labelcolor = C.ink)
    save_fig("faces", fig)
end

main()
