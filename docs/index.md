---
layout: home
hero:
  name: HaloArrays.jl
  text: Halo exchange with explicit grids.
  tagline: Build Cartesian MPI topologies, wrap local arrays, and exchange boundary data in Julia.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Explore the API
      link: /api
features:
  - title: Explicit ownership
    details: Topologies own communicators. Grids own communication caches. Fields declare their halo widths.
  - title: Two or three dimensions
    details: Exchange multiple fields across Cartesian process layouts with periodic or nonperiodic boundaries.
  - title: Split-phase exchange
    details: Start an exchange, perform independent work, then finish the final dimension's communication.
---

## Current scope

These pages describe the implementation in this checkout of version 0.1.0.
The examples use CPU arrays. `CuArray`/`ROCArray` fields work as drop-in
replacements wherever a `Field` wraps an array (see `ext/HaloArraysCUDAExt.jl`
and `ext/HaloArraysAMDGPUExt.jl`); the CUDA extension is tested against real
hardware, while the AMDGPU extension is a structurally identical port that has
not been run against real ROCm hardware yet.
