using Pkg
Pkg.activate(temp=true)
Pkg.add("PDFIO")
using PDFIO

p = "C:\\Users\\Albert\\Documents\\DevPkgs\\HaloArrays.jl\\Concurrency and Computation - 2026 - Machado - MPI‐Based Approaches to Overlap Computation and I O in Geophysical.pdf"
doc = pdDocOpen(p)
pages = pdDocGetPageCount(doc)
println("pages=", pages)
for i in 1:pages
    page = pdDocGetPage(doc, i)
    io = IOBuffer()
    pdPageExtractText(io, page)
    txt = String(take!(io))
    println("\n--- PAGE ", i, " ---\n", txt)
end
pdDocClose(doc)
