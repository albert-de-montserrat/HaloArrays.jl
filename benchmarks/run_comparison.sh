#!/usr/bin/env bash
# Drives bench_halogrids.jl / bench_igg.jl across a {nprocs} x {N} matrix and
# collects their single-line CSV output into results.csv.
#
# Both benchmark workers do the same thing: a 3D, fully periodic domain,
# 2x Float64 fields, halo width 1, niter update_halo! calls after a warmup,
# measured with @timed (wall time, bytes allocated, GC time, GC pause count,
# full-sweep count) on rank 0 only.
set -euo pipefail
cd "$(dirname "$0")"

MPIEXEC=$(julia --project=".." --startup-file=no -e \
    'import MPI; MPI.mpiexec() do exe; println(exe); end' 2>/dev/null | tail -1)
echo "Using mpiexec: $MPIEXEC" >&2

NPROCS_MATRIX=(1 2 4 8)
SIZES=(32 64 128)
NITER=150

OUT=results.csv
echo "pkg,nprocs,N,niter,time_s,bytes,gctime_s,pauses,full_sweep" > "$OUT"

for n in "${NPROCS_MATRIX[@]}"; do
  for N in "${SIZES[@]}"; do
    for script in bench_halogrids.jl bench_igg.jl; do
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
