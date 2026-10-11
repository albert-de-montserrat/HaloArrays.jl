# Which cells are sent and received between two neighboring ranks along one dimension,
# for the default overlap (2 * halowidth) and a larger overlap.
# Geometry follows face_range in src/face_exchange.jl: with halowidth hw and overlap ol,
# the upper strip sent by the left rank is indices len-ol+1 : len-ol+hw, and the
# receiving halo of the right rank is 1:hw. Both name the same physical lines.
include("common.jl")

function row!(ax, x0, y, len, hw, ol; label)
    for i in 1:len
        col = i <= hw || i > len - hw ? C.halo : C.inner
        up_send = len - ol + 1 <= i <= len - ol + hw
        lo_send = ol - hw + 1 <= i <= ol
        (up_send || lo_send) && (col = C.send)
        cells!(ax, x0 + i - 1, y, fill(col, 1, 1))
    end
    text!(ax, x0 - 0.5, y + 0.5; text = label, align = (:right, :center), color = C.ink, fontsize = 18, font = :bold)
end

function panel!(ax, hw, ol, y)
    len = 12
    # the right rank starts len - ol cells after the left rank
    xB = len - ol
    row!(ax, 0, y + 1.6, len, hw, ol; label = "rank A")
    row!(ax, xB, y, len, hw, ol; label = "rank B")
    # shared physical lines
    lines!(ax, [Point2f(xB, y + 2.8), Point2f(xB, y - 0.2)]; color = C.mute, linewidth = 1.5, linestyle = :dot)
    lines!(ax, [Point2f(len, y + 2.8), Point2f(len, y - 0.2)]; color = C.mute, linewidth = 1.5, linestyle = :dot)
    bracket_y = y + 3.0
    lines!(ax, [Point2f(xB, bracket_y - 0.1), Point2f(xB, bracket_y), Point2f(len, bracket_y), Point2f(len, bracket_y - 0.1)];
        color = C.ink, linewidth = 2)
    text!(ax, (xB + len) / 2, bracket_y + 0.15; text = "$ol shared lines (overlap)", align = (:center, :bottom), color = C.ink, fontsize = 16)
    # A's upper send strip lands in B's lower halo
    sa = len - ol + hw / 2
    arrow!(ax, (sa, y + 1.5), (xB + hw / 2, y + 1.1); color = C.ink)
    end

function main()
    fig = new_figure(1200, 560)
    Label(fig[0, 1], "Sent strips and halos along one dimension"; fontsize = 22, font = :bold, color = C.ink)
    ax1 = blank_axis(fig[1, 1]; title = "halowidths = 2, default overlaps = 4 (2 * halowidths)", titlesize = 19, titlecolor = C.ink, titlealign = :left)
    panel!(ax1, 2, 4, 0.0)
    xlims!(ax1, -3, 20); ylims!(ax1, -0.2, 3.9)
    ax2 = blank_axis(fig[2, 1]; title = "halowidths = 2, overlaps = 6: the sent strip moves inward", titlesize = 19, titlecolor = C.ink, titlealign = :left)
    panel!(ax2, 2, 6, 0.0)
    xlims!(ax2, -3, 20); ylims!(ax2, -0.2, 3.9)
    elems, labs = legend_patches((C.halo, "halo (overwritten on receive)"), (C.send, "interior strip sent to the neighbor"),
        (C.inner, "other interior cells"))
    Legend(fig[3, 1], elems, labs; orientation = :horizontal, framevisible = false, labelcolor = C.ink)
    save_fig("overlaps", fig)
end

main()
