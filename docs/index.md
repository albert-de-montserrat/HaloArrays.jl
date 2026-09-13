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
The examples use CPU arrays. CUDA and AMDGPU extension entries exist in the
package metadata, but their implementations are not present yet.
