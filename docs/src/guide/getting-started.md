# Getting started

Use Julia 1.9 or newer and an instantiated checkout of HaloGrids.jl. MPI.jl is
the package's required runtime dependency. From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

To use this checkout in another Julia environment, run
`Pkg.develop(path="/path/to/HaloGrids.jl")` from that environment.

## A complete periodic exchange

Each rank stores an array of size `(10, 10)`: an `(8, 8)` interior surrounded by
one layer of halo cells. Interior values identify the rank that owns them.
Periodic boundaries ensure every face receives data, even in a one-rank run.

```@eval
using Markdown
Markdown.parse("```julia\n" * read(joinpath(@__DIR__, "..", "..", "..", "examples", "quickstart.jl"), String) * "\n```")
```

Run the example from the repository root:

```sh
julia --project=. examples/quickstart.jl
```

For multiple processes, use the launcher selected by MPI.jl. In a Julia REPL
started with `julia --project=.`:

```julia
import MPI
MPI.mpiexec() do exe
    run(`$exe -n 4 $(Base.julia_cmd()) --project=. examples/quickstart.jl`)
end
```

The example asserts that received face values match the neighboring ranks.
`topology((0, 0))` lets MPI choose the process layout; it does not describe the
number of cells. Read [the core concepts](./concepts.md) for sizing and ownership.

## MPI lifetime

`topology` initializes MPI if it has not been initialized. The example initializes
and finalizes MPI explicitly because it owns the application lifetime. Libraries
embedded in a larger MPI application should leave finalization to that application.
All participating ranks must execute matching communication calls in the same order.
