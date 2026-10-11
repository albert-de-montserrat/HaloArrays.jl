using MPI

function main()
    backend = isempty(ARGS) ? "cpu" : ARGS[1]
    backend in ("cpu", "cuda") || error("backend must be cpu or cuda")
    niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
    output = length(ARGS) >= 3 ? abspath(ARGS[3]) : joinpath(@__DIR__, "completion_$(backend).csv")
    worker = joinpath(@__DIR__, "bench_completion_candidate.jl")
    root = dirname(@__DIR__)
    project = backend == "cuda" ? joinpath(root, "test") : root
    executable = joinpath(Sys.BINDIR, Base.julia_exename())
    ranks = backend == "cuda" ? (1, 2) : (1, 2, 4, 8)
    open(output, "w") do io
        for n in ranks
            println("Checking and benchmarking $backend at $n ranks")
            MPI.mpiexec() do exe
                run(pipeline(`$exe -n $n $executable --project=$project --startup-file=no $worker $backend $niter 128x32x64 0x0x0 3`; stdout=io))
            end
        end
        if backend == "cpu"
            for dims in ("1x1x2", "2x1x1"), shape in ("32x32x32", "128x128x128")
                println("Checking and benchmarking $backend at 2 ranks, $dims, $shape")
                MPI.mpiexec() do exe
                    run(pipeline(`$exe -n 2 $executable --project=$project --startup-file=no $worker $backend $niter $shape $dims 3`; stdout=io))
                end
            end
            println("Checking threaded packing at 1 and 2 ranks")
            for n in (1, 2)
                MPI.mpiexec() do exe
                    run(pipeline(`$exe -n $n $executable --project=$project --startup-file=no -t 4 $worker cpu 0`; stdout=io))
                end
            end
        end
    end
    println("Saved $output")
end

main()
