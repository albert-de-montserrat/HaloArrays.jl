# Weak-scaling plots for results written by slurm_cudaaware.sh. Each benchmark
# label is split into package (HaloGrids / ImplicitGlobalGrid), case
# (exchange-only, blocking diffusion3D, hidden-communication diffusion3D) and
# transport (`Aware` suffix = device buffers to MPI, otherwise host-staged).
# Repetitions are summarized by their median; the min-max spread is drawn as a
# band. Needs no GPU:
#   julia --project=benchmarks/gpu benchmarks/gpu/plot_scaling.jl results.jld2 [out.png]
import Pkg
Pkg.activate(joinpath(@__DIR__); io=devnull)
using CairoMakie
using JLD2
using Printf
using Statistics: median

const CASES = [
    ("exchange", "update_halo! only"),
    ("blocking", "diffusion3D, blocking exchange"),
    ("hide", "diffusion3D, hidden communication"),
]

function classify(label::AbstractString)
    aware = endswith(label, "Aware")
    base = aware ? chop(label; tail=5) : label
    pkg = startswith(base, "HaloGrids") ? "HaloGrids" :
          startswith(base, "ImplicitGlobalGrid") ? "IGG" :
          error("unrecognized benchmark label \"$label\"")
    rest = pkg == "HaloGrids" ? chop(base; head=9, tail=0) : chop(base; head=18, tail=0)
    case = rest == "" ? "blocking" : rest == "Hide" ? "hide" : rest == "Exchange" ? "exchange" :
           error("unrecognized benchmark label \"$label\"")
    return (; pkg, case, transport=aware ? "aware" : "staged")
end

function read_rows(path)
    rows = NamedTuple[]
    for (key, r) in JLD2.load(path)
        startswith(key, "result__") || continue
        c = classify(r.pkg)
        push!(rows, (; c..., label=r.pkg, r.nprocs, r.N, ms=1e3 * r.time_s / r.niter))
    end
    isempty(rows) && error("no result__* entries in $path")
    return rows
end

# (pkg, case, transport, N) => sorted [(nprocs, median, min, max)]
function summarize(rows)
    groups = Dict{Tuple{String,String,String,Int},Dict{Int,Vector{Float64}}}()
    for r in rows
        g = get!(groups, (r.pkg, r.case, r.transport, r.N), Dict{Int,Vector{Float64}}())
        push!(get!(g, r.nprocs, Float64[]), r.ms)
    end
    return Dict(k => sort([(n, median(v), minimum(v), maximum(v)) for (n, v) in d]; by=first)
                for (k, d) in groups)
end

function print_table(summary)
    println("| case | transport | N | ranks | HaloGrids ms/step | IGG ms/step | HG/IGG |")
    println("|---|---|---|---|---|---|---|")
    keys_ = sort(unique((k[2], k[3], k[4]) for k in keys(summary)))
    for (case, transport, N) in keys_
        hg = Dict(p[1] => p[2] for p in get(summary, ("HaloGrids", case, transport, N), []))
        igg = Dict(p[1] => p[2] for p in get(summary, ("IGG", case, transport, N), []))
        for n in sort(collect(union(keys(hg), keys(igg))))
            h = get(hg, n, NaN)
            i = get(igg, n, NaN)
            @printf("| %s | %s | %d | %d | %.4f | %.4f | %.2f |\n", case, transport, N, n, h, i, h / i)
        end
    end
end

function plot_summary(summary, outfile)
    colors = Dict("HaloGrids" => Makie.wong_colors()[1], "IGG" => Makie.wong_colors()[2])
    styles = Dict("aware" => :solid, "staged" => :dash)
    Ns = sort(unique(k[4] for k in keys(summary)))
    nps = sort(unique(p[1] for v in values(summary) for p in v))
    fig = Figure(size=(1200, 420 * length(Ns)))
    for (iN, N) in enumerate(Ns), (ic, (case, title)) in enumerate(CASES)
        ax = Axis(fig[iN, ic]; title="$title (local N=$N)", xscale=log2,
            xticks=(nps, string.(nps)), xlabel="GPUs (ranks)",
            ylabel=ic == 1 ? "ms per step" : "")
        for pkg in ("HaloGrids", "IGG"), transport in ("aware", "staged")
            pts = get(summary, (pkg, case, transport, N), nothing)
            pts === nothing && continue
            x = [p[1] for p in pts]
            band!(ax, x, [p[3] for p in pts], [p[4] for p in pts]; color=(colors[pkg], 0.2))
            scatterlines!(ax, x, [p[2] for p in pts]; color=colors[pkg],
                linestyle=styles[transport], label="$pkg $transport")
        end
        ic == 1 && axislegend(ax; position=:lt)
    end
    save(outfile, fig)
    return outfile
end

function main(args)
    isempty(args) && error("usage: plot_scaling.jl results.jld2 [out.png]")
    path = args[1]
    outfile = length(args) >= 2 ? args[2] : replace(path, r"\.jld2$" => "") * "_scaling.png"
    summary = summarize(read_rows(path))
    print_table(summary)
    println("\nWrote ", plot_summary(summary, outfile))
end

main(ARGS)
