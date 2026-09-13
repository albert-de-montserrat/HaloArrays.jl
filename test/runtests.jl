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
