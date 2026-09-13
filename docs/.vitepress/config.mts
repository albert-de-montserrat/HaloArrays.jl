import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'HaloArrays.jl',
  description: 'Explicit grids and MPI halo exchange for Julia.',
  lang: 'en-US',
  base: process.env.DOCS_BASE || '/',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api' },
      { text: 'Contributing', link: '/contributing' }
    ],
    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting started', link: '/guide/getting-started' },
          { text: 'Topologies, grids, and fields', link: '/guide/concepts' },
          { text: 'Halo exchange', link: '/guide/halo-exchange' },
          { text: 'Gathering arrays', link: '/guide/gather' },
          { text: 'Coordinate helpers', link: '/guide/coordinates' },
          { text: 'Migrating from ImplicitGlobalGrid.jl', link: '/guide/migration' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Public API', link: '/api' },
          { text: 'Development and documentation', link: '/contributing' }
        ]
      }
    ],
    search: { provider: 'local' },
    outline: [2, 3]
  }
})
