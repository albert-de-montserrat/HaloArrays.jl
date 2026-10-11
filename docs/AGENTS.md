# Rules for `docs/`

Also follow the root `AGENTS.md`. Use the `update-docs` skill.

- Site is Documenter.jl + DocumenterVitepress. Build: `julia --project=docs docs/make.jl`.
  Do not add `package.json`, `npm`, or a Node setup.
- `docs/src/api.md` is built from `@docs` blocks. Change the docstring in `src/`.
  Do not hand-write signatures.
- New page: add the file under `docs/src/guide/` and register it in `pages` in
  `docs/make.jl`.
- Quickstart, hide-communication, ParallelStencil, and coordinate pages embed
  `examples/*.jl` through `@eval` blocks. Edit the example, not a copy.
- Code blocks in guides must run from the package project, or say which extra
  package they need (`using HDF5`, `using WriteVTK`).
- Keep claims about GPU and performance to what tests or benchmarks show. Do not
  document planned features as available.
- Never commit `docs/build/`, `node_modules/`, or caches.

## Figures

- Figures are Makie scripts in `docs/figures/`, sharing `common.jl` (palette, helpers).
  Outputs (SVG and PNG) are committed in `docs/src/assets/`. The docs build does not run them.
- Regenerate: `julia --project=docs/figures docs/figures/make_all.jl`. Register new
  scripts in `make_all.jl`. The environment's `Manifest.toml` is gitignored.
- Draw geometry from the real rules (for example `face_range` in `src/face_exchange.jl`).
  Label schematic timelines as schematic. Never put invented numbers in a figure.
- Use the palette in `common.jl` and an opaque background so SVGs read in light and dark themes.
- Embed with alt text that states what the figure shows:
  `![alt](../assets/name.png)` from `docs/src/guide/`, `assets/name.png` from `docs/src/`.
- After generating, open the PNG and check for clipped text and overlaps.
