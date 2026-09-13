# HaloArrays.jl

Explicit Cartesian MPI topologies, per-grid communication caches, and per-field
halo widths for Julia arrays in 2D and 3D.

Start with the [getting-started guide](docs/guide/getting-started.md), explore the
[public API](docs/api.md), or run [the periodic exchange example](examples/quickstart.jl):

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/quickstart.jl
```

The current implementation targets CPU arrays. GPU extension declarations are
present in the package metadata, but the extension implementations are not yet available.

## Documentation

With Node.js 22 or newer and npm installed:

```sh
npm ci
npm run docs:dev
```

Build with `npm run docs:build`. See [development instructions](docs/contributing.md)
for the MPI test matrix and documentation deployment settings.
