# Shared helper for the *_repeat.jl benchmark workers (bench_haloarrays_diffusion3d_repeat.jl,
# bench_haloarrays_diffusion3d_hide_repeat.jl, bench_igg_diffusion3d_repeat.jl,
# bench_igg_diffusion3d_hide_repeat.jl). Extends metrics_common.jl (`include`d
# below, left completely untouched) with a trial-qualified save function, so a
# single script run can save several independent timing trials into the same
# JLD2 file instead of the original single-trial save_result's one row per
# run — needed to tell a real HaloArrays-vs-ImplicitGlobalGrid performance
# gap apart from ordinary run-to-run noise, which a single 50-iteration timed
# block (the original bench_*.jl behavior) has no way to quantify.
include(joinpath(@__DIR__, "metrics_common.jl"))

# Identical to save_result except the JLD2 key also carries `trial`, so
# repeated calls for the same (pkg, nprocs, N) don't collide (JLD2 refuses to
# silently overwrite an existing key). The saved row also carries `trial`
# itself, purely informational — visualize.jl's existing "average over
# matching rows" behavior (see its grouped_bar/speedup_bar) already treats
# multiple rows for the same (pkg, nprocs, N) as repeated trials to be
# averaged, with no changes needed there to consume this.
function save_trial_result(path, pkg, nprocs, N, niter, trial, stats, gpu_bytes, teff, peakbw)
    gs = stats.gcstats
    pct = peakbw > 0 ? 100 * teff / peakbw : NaN
    result = (; pkg=String(pkg), nprocs, N, niter, trial, time_s=stats.time,
        bytes=stats.bytes, gctime_s=stats.gctime, pauses=gs.pause,
        full_sweep=gs.full_sweep, gpu_bytes, t_eff_gbs=teff,
        peak_bw_gbs=peakbw, pct_of_peak=pct)
    key = "result__$(pkg)__n$(nprocs)__N$(N)__trial$(trial)"
    jldopen(path, "a+") do file
        file[key] = result
    end
    println("Saved $key to $path")
    return result
end
