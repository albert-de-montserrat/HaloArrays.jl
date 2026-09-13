# Development and documentation

## Julia checks

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no test/test_field.jl
julia --project=. --startup-file=no test/runtests.jl
```

The full driver starts its own MPI subprocesses at 1, 2, 4, and 8 ranks. Threaded
packing checks run separately with four Julia threads at 1 and 2 ranks. Run the
driver directly, without an outer MPI launcher. Allow enough process slots for
eight ranks, or run focused files with an appropriate MPI.jl launcher while developing.

## Continuous integration

GitHub Actions runs on pushes, pull requests, and manual dispatches. The Julia
jobs test Julia 1.9 and the latest stable 1.x release on Linux, using the full
MPI and threaded test driver and the quickstart example. Dependencies are resolved
from `Project.toml` in each job rather than the development manifest. The CPU
driver is invoked directly, without installing the unimplemented GPU test extras.

A separate Node.js 22 job installs the npm lockfile dependencies and builds this
site, including VitePress's internal link checks. These checks do not deploy the site.

## Work on this site

Install Node.js 22 or newer, including npm. The site pins VitePress 1.6.4 and
uses its default theme with local search. From the repository root:

```sh
npm ci
npm run docs:dev
```

The development command prints the local URL. On PowerShell systems that block
`npm.ps1`, use `npm.cmd` for these commands.

```sh
npm run docs:build
npm run docs:preview
```

The build checks internal links and writes the static site to
`docs/.vitepress/dist`. Preview serves that built output locally. The
quickstart, halo-exchange overlap, and coordinate-helper pages include
`examples/quickstart.jl`, `examples/hide_communication.jl`, and
`examples/diffusion3D.jl` directly, so edits to their executable source also
update the page.

## Deployment base

The default base is `/`. When hosting in a subdirectory, set `DOCS_BASE` to its
path, including the leading and trailing slash, before building. For PowerShell:

```powershell
$env:DOCS_BASE = '/HaloArrays.jl/'
npm run docs:build
```

Publish the generated `docs/.vitepress/dist` directory using your chosen static
host. No deployment target is assumed by this repository.

See the [VitePress setup guide](https://vuejs.github.io/vitepress/v1/guide/getting-started)
for the documentation tooling and [site configuration](https://vuejs.github.io/vitepress/v1/reference/site-config)
for hosting options. Keep package behavior documented from `src/` and verified examples.
