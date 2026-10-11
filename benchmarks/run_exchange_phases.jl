# Run from the repository root with julia --project=. --startup-file=no.
using MPI

function main()
    niter = isempty(ARGS) ? 200 : parse(Int, ARGS[1])
    output = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(@__DIR__, "exchange_phases.csv")
    worker = joinpath(@__DIR__, "bench_exchange_phases.jl")
    project = dirname(@__DIR__)
    executable = joinpath(Sys.BINDIR, Base.julia_exename())
    focused = length(ARGS) >= 3 && ARGS[3] == "focused"
    repeats = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    repeats > 0 || error("repeat count must be positive")
    open(output, "w") do io
        for n in (focused ? () : (1, 2, 4, 8))
            println("Benchmarking $n ranks")
            MPI.mpiexec() do exe
                run(pipeline(`$exe -n $n $executable --project=$project --startup-file=no $worker $niter`; stdout=io))
            end
        end
        # Same rank count, different final dimension: one leaves real MPI
        # outstanding; the other leaves only a local periodic copy outstanding.
        for rep in 1:repeats, dims in ("1x1x2", "2x1x1")
            println("Benchmarking 2 ranks, process dimensions $dims, repeat $rep")
            MPI.mpiexec() do exe
                run(pipeline(`$exe -n 2 $executable --project=$project --startup-file=no $worker $niter 128x32x64 $dims`; stdout=io))
            end
        end
    end
    println("Saved $output")
end

main()
