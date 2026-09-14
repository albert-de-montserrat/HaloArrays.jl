module HaloArraysParallelStencilExt

using HaloArrays
using ParallelStencil

boundary_width(fields...) = ntuple(d -> maximum(HaloArrays.halowidths(f)[d] for f in fields), ndims(first(fields)))

function hide_halo_communication_expr(args...)
    block = args[end]
    positional = collect(args[1:end-1])
    kw = findfirst(x -> x isa Expr && x.head == :(=) && x.args[1] == :boundary_width, positional)
    width = kw === nothing ? nothing : popat!(positional, kw).args[2]
    length(positional) >= 2 || error("expected grid, fields, and a block")
    grid, fields = positional[1], positional[2:end]
    statements = filter(x -> !(x isa LineNumberNode), block.args)
    launch, boundary_updates = statements[1], statements[2:end]
    launch isa Expr && launch.head == :macrocall && launch.args[1] == Symbol("@parallel") && length(launch.args) == 4 || error("first statement must be `@parallel ranges kernel!(...)`")
    ranges, kernel = launch.args[3:4]
    width_expr = width === nothing ? :(HaloArraysParallelStencilExt.boundary_width($(fields...))) : width
    outer, inner, range, handle = gensym.((:outer, :inner, :range, :handle))
    return esc(quote
        $outer = ParallelStencil.ParallelKernel.get_ranges_outer($width_expr, $ranges)
        $inner = ParallelStencil.ParallelKernel.get_ranges_inner($width_expr, $ranges)
        for $range in $outer
            ParallelStencil.@parallel $range $kernel
        end
        $(boundary_updates...)
        ParallelStencil.@synchronize
        $handle = HaloArrays.halo_exchange_start!($grid, $(fields...))
        try
            for $range in $inner
                ParallelStencil.@parallel $range $kernel
            end
        finally
            HaloArrays.halo_exchange_finish!($handle)
        end
    end)
end

end
