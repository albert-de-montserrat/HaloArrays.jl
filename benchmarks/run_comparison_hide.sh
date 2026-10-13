#!/usr/bin/env bash
# Drives bench_halogrids_hide.jl / bench_igg_hide.jl across a {nprocs} x {N}
# matrix and collects their single-line CSV output into results_hide.csv.
#
# Both workers run one 3D 7-point-Laplacian diffusion step per iteration on a
# fully periodic domain with 2x Float64 fields and halo width 1 (same domain
# as run_comparison.sh's plain-blocking comparison), but HaloGrids.jl's
# worker overlaps the interior compute with dimension 3's in-flight exchange
# (halo_exchange_start!/finish!) while ImplicitGlobalGrid.jl's worker can only
# run the two serially (update_halo! has no start!/finish! split) — see the
# header comments in each worker file. Measured with @timed (wall time, bytes
# allocated, GC time, GC pause count, full-sweep count) on rank 0 only, same
# methodology as run_comparison.sh.
set -euo pipefail
cd "$(dirname "$0")"

MPIEXEC=$(julia --project=".." --startup-file=no -e \
    'import MPI; MPI.mpiexec() do exe; println(exe); end' 2>/dev/null | tail -1)
echo "Using mpiexec: $MPIEXEC" >&2

NPROCS_MATRIX=(1 2 4 8)
SIZES=(32 64 128)
NITER=150

OUT=results_hide.csv
echo "pkg,nprocs,N,niter,time_s,bytes,gctime_s,pauses,full_sweep" > "$OUT"

for n in "${NPROCS_MATRIX[@]}"; do
  for N in "${SIZES[@]}"; do
    for script in bench_halogrids_hide.jl bench_igg_hide.jl; do
      echo "-- $script  nprocs=$n  N=$N --" >&2
      line=$("$MPIEXEC" -n "$n" julia --startup-file=no "$script" "$N" "$NITER" 2>err.log | grep '^RESULT' || true)
      if [[ -z "$line" ]]; then
        echo "   FAILED (see err.log):" >&2
        tail -20 err.log >&2
        continue
      fi
      echo "${line#RESULT,}" >> "$OUT"
    done
  done
done

rm -f err.log
echo "Done. Wrote $OUT" >&2
