# HaloArrays.jl

Explicit Cartesian MPI topologies, per-grid communication caches, and per-field
halo widths for Julia arrays in 2D and 3D.

Start with the [getting-started guide](docs/guide/getting-started.md), explore the
[public API](docs/api.md), or run [the periodic exchange example](examples/quickstart.jl):

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/quickstart.jl
```

CUDA support (`HaloArraysCUDAExt`) is implemented and tested on real hardware,
including a fast path for `CellArrays.jl`'s `CellArray` type
(`HaloArraysCellArraysExt`). AMDGPU support (`HaloArraysAMDGPUExt`) is
implemented but has not been run on real ROCm hardware. See
[AGENTS.md](AGENTS.md) for the current per-backend test status.

## Documentation

With Node.js 22 or newer and npm installed:

```sh
npm ci
npm run docs:dev
```

Build with `npm run docs:build`. See [development instructions](docs/contributing.md)
for the MPI test matrix and documentation deployment settings.
