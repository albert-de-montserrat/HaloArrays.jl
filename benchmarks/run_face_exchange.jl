using MPI

function main()
    backend = isempty(ARGS) ? "cpu" : ARGS[1]
    backend in ("cpu", "cuda") || error("backend must be cpu or cuda")
    niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
    output = length(ARGS) >= 3 ? abspath(ARGS[3]) : joinpath(@__DIR__, "faces_$backend.csv")
    root = dirname(@__DIR__)
    project = backend == "cuda" ? joinpath(root, "test") : root
    worker = joinpath(@__DIR__, "bench_face_exchange.jl")
    executable = joinpath(Sys.BINDIR, Base.julia_exename())
    ranks = backend == "cuda" ? (1, 2) : (1, 2, 4, 8)
    open(output, "w") do io
        for n in ranks
            println("Benchmarking face/full exchange, $backend, $n ranks")
            MPI.mpiexec() do exe
                run(pipeline(`$exe -n $n $executable --project=$project --startup-file=no $worker $backend $niter 128x32x64 0x0x0 3`; stdout=io))
            end
        end
        if backend == "cpu"
            for dims in ("1x1x2", "2x1x1")
                println("Benchmarking face/full exchange, cpu, 2 ranks, $dims")
                MPI.mpiexec() do exe
                    run(pipeline(`$exe -n 2 $executable --project=$project --startup-file=no $worker cpu $niter 128x32x64 $dims 3`; stdout=io))
                end
            end
        else
            phase_worker = joinpath(@__DIR__, "bench_gpu_exchange_phases.jl")
            for (n, dims) in ((1, "1x1x1"), (2, "2x1x1"), (2, "1x1x2"))
                println("Profiling GPU phases, $n ranks, $dims")
                MPI.mpiexec() do exe
                    run(pipeline(`$exe -n $n $executable --project=$project --startup-file=no $phase_worker $niter 128x32x64 $dims`; stdout=io))
                end
            end
        end
    end
    println("Saved $output")
end

main()
