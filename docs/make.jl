using Documenter
using DocumenterVitepress
using HaloArrays

makedocs(;
    sitename = "HaloArrays.jl",
    authors = "Albert de Montserrat",
    modules = [HaloArrays],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/albert-de-montserrat/HaloArrays.jl",
        devbranch = "master",
        devurl = "dev",
    ),
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting started" => "guide/getting-started.md",
            "Topologies, grids, and fields" => "guide/concepts.md",
            "Halo exchange" => "guide/halo-exchange.md",
            "Hiding communication and boundary widths" => "guide/hide-communication.md",
            "Gathering arrays" => "guide/gather.md",
            "MPI/HDF5 file I/O" => "guide/hdf5-io.md",
            "Global reductions" => "guide/reductions.md",
            "Coordinate helpers" => "guide/coordinates.md",
            "Migrating from ImplicitGlobalGrid.jl" => "guide/migration.md",
        ],
        "Reference" => [
            "Public API" => "api.md",
            "Development and documentation" => "contributing.md",
        ],
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/albert-de-montserrat/HaloArrays.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true,
)
