# Subprocess-per-test-file driver, mirroring ImplicitGlobalGrid.jl's
# test/runtests.jl, but explicitly looping an `mpiexec -n N` matrix per file
# instead of trusting whatever N invoked the driver (test_field.jl needs no
# MPI ranks at all and is run once, plainly).
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
import MPI

const NPROCS_MATRIX = [1, 2, 4, 8]
const NO_MPI_FILES = ["test_field.jl"]
# Needs real OS threads to exercise Threads.@spawn-based packing (see
# test_threaded_pack.jl); run separately from the rest of the suite with
# -t 4, over a small nprocs matrix (it isn't testing topology variety, just
# that the threaded and sequential pack/unpack paths agree).
const THREADED_FILES = ["test_threaded_pack.jl"]
# CUDA/AMDGPU are weakdeps of the package itself (Project.toml), not real
# dependencies, so a subprocess launched with `--project=<pkg root>` (like
# every other file below) cannot `using CUDA`/`using AMDGPU` at all. test/
# has its own Project.toml (CUDA, AMDGPU, CellArrays, MPI, StaticArrays, plus
# HaloArrays itself as a `[sources]`-path dependency) specifically so these
# files can; see test_cuda.jl/test_amdgpu.jl, which each skip gracefully via
# CUDA.functional()/AMDGPU.functional() when no usable GPU is present. Run
# over a small nprocs matrix, same reasoning as THREADED_FILES: this project
# has at most one physical GPU to test against, so higher rank counts add
# process-oversubscription time, not correctness coverage.
const GPU_FILES = ["test_cuda.jl", "test_amdgpu.jl", "test_cellarrays_cuda.jl", "test_allocations_cuda.jl"]
const GPU_NPROCS_MATRIX = [1, 2]
# test_cellarrays.jl needs test/'s CellArrays dependency for the same reason
# (CellArrays is only a weakdep of the package root), but has no GPU-hardware
# requirement or functional()-style gate, so it runs the same full
# NPROCS_MATRIX as an ordinary CPU test file — just launched with
# `--project=test` instead of `--project=<pkg root>`.
# test_hdf5_io.jl follows the same optional weakdep pattern with a stronger
# environment gate, because the HDF5 package is a weak dependency of the core
# package and therefore must be made visible through the test/Project.toml.
const TEST_PROJECT_CPU_FILES = ["test_cellarrays.jl", "test_hdf5_io.jl"]

testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
testfiles = sort(filter(istest, readdir(testdir; join=true)))

nfail = 0
printstyled("Testing package HaloArrays.jl\n"; bold=true, color=:white)

exename = joinpath(Sys.BINDIR, Base.julia_exename())
for f in testfiles
    if basename(f) in NO_MPI_FILES
        println("\n-- ", basename(f), " (no MPI) --")
        try
            run(`$exename --project=$(joinpath(@__DIR__, "..")) --startup-file=no $f`)
        catch
            global nfail += 1
        end
        continue
    end
    if basename(f) in THREADED_FILES
        for n in (1, 2)
            println("\n-- ", basename(f), " -n $n -t 4 --")
            try
                MPI.mpiexec() do exe
                    run(`$exe -n $n $exename --project=$(joinpath(@__DIR__, "..")) --startup-file=no -t 4 $f`)
                end
            catch
                global nfail += 1
            end
        end
        continue
    end
    if basename(f) in GPU_FILES
        # test/Project.toml's own Manifest.toml is gitignored (like the
        # package root's, only not force-committed the way that one is), so a
        # fresh checkout/CI runner won't have CUDA/AMDGPU/HaloArrays resolved
        # there yet. Rather than auto-instantiate here (a real, un-opted-into
        # time and network cost added to every plain test run/CI job — this
        # project's CI has so far deliberately stayed CPU-only), skip cleanly
        # and say so, exactly like a real CUDA.functional()==false skip would.
        # Run `julia --project=test -e 'using Pkg; Pkg.instantiate()'` once
        # (from test/) to enable this category locally.
        if !isfile(joinpath(@__DIR__, "Manifest.toml"))
            println("\n-- ", basename(f), " (skipped: test/ dependencies not instantiated) --")
            continue
        end
        for n in GPU_NPROCS_MATRIX
            println("\n-- ", basename(f), " -n $n (test/ project) --")
            try
                MPI.mpiexec() do exe
                    run(`$exe -n $n $exename --project=$(joinpath(@__DIR__)) --startup-file=no $f`)
                end
            catch
                global nfail += 1
            end
        end
        continue
    end
    if basename(f) in TEST_PROJECT_CPU_FILES
        # Same "skip if test/ isn't instantiated" gate as GPU_FILES, but the
        # full NPROCS_MATRIX and no functional()-style runtime check: this
        # category exists purely because of the weakdep/real-dep split, not
        # because of any hardware requirement.
        if !isfile(joinpath(@__DIR__, "Manifest.toml"))
            println("\n-- ", basename(f), " (skipped: test/ dependencies not instantiated) --")
            continue
        end
        for n in NPROCS_MATRIX
            println("\n-- ", basename(f), " -n $n (test/ project) --")
            try
                MPI.mpiexec() do exe
                    run(`$exe -n $n $exename --project=$(joinpath(@__DIR__)) --startup-file=no $f`)
                end
            catch
                global nfail += 1
            end
        end
        continue
    end
    for n in NPROCS_MATRIX
        println("\n-- ", basename(f), " -n $n --")
        try
            MPI.mpiexec() do exe
                run(`$exe -n $n $exename --project=$(joinpath(@__DIR__, "..")) --startup-file=no $f`)
            end
        catch
            global nfail += 1
        end
    end
end

exit(nfail)
