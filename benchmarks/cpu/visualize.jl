# Reads the CPU comparison JLD2 files (produced by slurm_diffusion3d.sh /
# slurm_hide.sh on the cluster, copied back locally) and renders grouped-bar
# PNGs (blocking and communication-hiding variants, grouped by problem size
# N, one chart per metric per rank count) plus a markdown summary table. Run
# this after copying results_*_cpu.jld2 back from the cluster, e.g.:
#   julia --project=benchmarks/cpu benchmarks/cpu/visualize.jl
#
# ARGS: zero or more JLD2 paths; defaults to every results_diffusion3d_cpu_n*.jld2
# (one per rank count in slurm_diffusion3d.sh's NTASKS_MATRIX, since each
# bench_*_diffusion3d*.jl run now names its own output after its rank count)
# plus results_hide_cpu.jld2, in this directory (whichever exist).
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CairoMakie
using JLD2
using Printf

const METRICS = [
    ("time_s",     "wall time / step (ms)",       row -> 1e3 * row.time_s / row.niter),
    ("gctime_s",   "GC time / step (ms)",          row -> 1e3 * row.gctime_s / row.niter),
    ("pauses",     "GC pauses / call",             row -> row.pauses / row.niter),
    ("bytes",      "host bytes allocated / call",  row -> row.bytes / row.niter),
    ("gpu_bytes",  "host bytes allocated / call (2nd pass)", row -> row.gpu_bytes / row.niter),
    ("t_eff_gbs",  "effective throughput (GB/s)",  row -> row.t_eff_gbs),
    ("pct_of_peak","effective throughput (% of measured peak BW)", row -> row.pct_of_peak),
]

read_results(path::AbstractString) = NamedTuple[
    value for (key, value) in JLD2.load(path) if startswith(key, "result__")]

function grouped_bar(rows, metricfn, ylabel, title, outfile)
    Ns = sort(unique(Int.(r.N for r in rows)))
    pkgs = sort(unique(r.pkg for r in rows))
    isempty(Ns) && return nothing

    fig = Figure(size=(600, 380))
    ax = Axis(fig[1, 1]; xlabel="local N (per rank, per dim)", ylabel=ylabel, title=title,
              xticks=(1:length(Ns), string.(Ns)))
    barwidth = 0.8 / length(pkgs)
    colors = Dict(pkgs[i] => Makie.wong_colors()[i] for i in eachindex(pkgs))
    for (pi, pkg) in enumerate(pkgs)
        xs = Float64[]
        ys = Float64[]
        for (ni, N) in enumerate(Ns)
            matches = filter(r -> r.pkg == pkg && Int(r.N) == N, rows)
            isempty(matches) && continue
            v = sum(metricfn(r) for r in matches) / length(matches)  # mean over repeated runs, if any
            push!(xs, ni + (pi - (length(pkgs) + 1) / 2) * barwidth)
            push!(ys, v)
        end
        isempty(xs) && continue
        barplot!(ax, xs, ys; width=barwidth * 0.9, color=colors[pkg], label=pkg)
    end
    axislegend(ax; position=:rt)
    save(outfile, fig)
    return outfile
end

function visualize_results(path::AbstractString, plotdir::AbstractString)
    isfile(path) || return NamedTuple[]
    rows = read_results(path)
    isempty(rows) && return rows
    tag = replace(basename(path), ".jld2" => "")
    for nprocs in sort(unique(Int.(r.nprocs for r in rows)))
        subset = filter(r -> Int(r.nprocs) == nprocs, rows)
        for (col, label, fn) in METRICS
            outfile = joinpath(plotdir, "$(tag)_n$(nprocs)_$(col).png")
            grouped_bar(subset, fn, label, "$tag — $(nprocs) rank(s)", outfile)
        end
    end
    return rows
end

function write_summary(all_results::Dict{String,Vector{<:NamedTuple}}, outpath::AbstractString)
    open(outpath, "w") do io
        println(io, "# HaloGrids.jl vs ImplicitGlobalGrid.jl — CPU blocking/overlap benchmark summary\n")
        println(io, "Both packages run ordinary host-memory MPI transport here (no GPU, no")
        println(io, "GPU-direct/CUDA-aware MPI asymmetry to account for — unlike the benchmarks/gpu")
        println(io, "comparison, see that directory's visualize.jl for that caveat), so these numbers")
        println(io, "are directly comparable as-is.\n")
        for (tag, rows) in sort(collect(all_results); by=first)
            isempty(rows) && continue
            println(io, "## $tag\n")
            println(io, "| pkg | nprocs | N | time/step (ms) | T_eff (GB/s) | % of peak | GC time/step (ms) | host B/call (2nd pass) |")
            println(io, "|---|---|---|---|---|---|---|---|")
            for r in sort(rows; by=r -> (r.pkg, r.nprocs, r.N))
                @printf(io, "| %s | %d | %d | %.4f | %.2f | %.1f%% | %.4f | %.0f |\n",
                    r.pkg, r.nprocs, r.N, 1e3 * r.time_s / r.niter, r.t_eff_gbs, r.pct_of_peak,
                    1e3 * r.gctime_s / r.niter, r.gpu_bytes / r.niter)
            end
            println(io)
        end
    end
    return outpath
end

function main()
    paths = if isempty(ARGS)
        diffusion3d = filter(f -> startswith(f, "results_diffusion3d_cpu_n") && endswith(f, ".jld2"), readdir(@__DIR__))
        vcat(joinpath.(@__DIR__, sort(diffusion3d)), joinpath(@__DIR__, "results_hide_cpu.jld2"))
    else
        collect(ARGS)
    end
    plotdir = joinpath(@__DIR__, "plots")
    mkpath(plotdir)

    all_results = Dict{String,Vector{<:NamedTuple}}()
    for path in paths
        if !isfile(path)
            println("skip (not found): $path")
            continue
        end
        tag = replace(basename(path), ".jld2" => "")
        rows = visualize_results(path, plotdir)
        all_results[tag] = rows
        println("$(length(rows)) rows -> plots written under $plotdir/ (prefix \"$tag\")")
    end

    if isempty(all_results) || all(isempty, values(all_results))
        println("No result JLD2 files found/non-empty; nothing to summarize. Run slurm_diffusion3d.sh / slurm_hide.sh first.")
        return
    end

    summarypath = joinpath(@__DIR__, "summary.md")
    write_summary(all_results, summarypath)
    println("Wrote $summarypath")
end

main()
