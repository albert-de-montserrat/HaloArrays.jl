# HaloArrays.jl port of bench_PS.jl (3D heat diffusion, ParallelStencil kernel).
# Usage: julia --project=benchmarks/gpu bench_HA.jl [n=32] [nt=100000]
const USE_GPU = get(ENV, "USE_GPU", "0") == "1"
import MPI
@static if USE_GPU; using CUDA; end
using HaloArrays
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)*_dx^2 + @d2_yi(T)*_dy^2 + @d2_zi(T)*_dz^2));
    return
end

function diffusion3D(n, nt)
# Physics
lam        = 1.0;
cp_min     = 1.0;
lx, ly, lz = 10.0, 10.0, 10.0;

# Numerics
nx, ny, nz = n, n, n;
MPI.Initialized() || MPI.Init()
t  = topology((0, 0, 0); periods=(false, false, false))
g  = grid(t, (nx, ny, nz); origin_on_vertex=true)
Tf  = field(@zeros(nx, ny, nz); halowidths=(1, 1, 1))
T2f = field(@zeros(nx, ny, nz); halowidths=(1, 1, 1))
Ci  = @zeros(nx, ny, nz);
me  = t.rank
dx  = lx/(nx_g(g, Tf)-1);
dy  = ly/(ny_g(g, Tf)-1);
dz  = lz/(nz_g(g, Tf)-1);
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;

# Initial conditions
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-((x_g(ix,dx,g,Tf)-lx/1.5))^2-((y_g(iy,dy,g,Tf)-ly/2))^2-((z_g(iz,dz,g,Tf)-lz/1.5))^2) +
                                   5*exp(-((x_g(ix,dx,g,Tf)-lx/3.0))^2-((y_g(iy,dy,g,Tf)-ly/2))^2-((z_g(iz,dz,g,Tf)-lz/1.5))^2) for ix=1:nx, iy=1:ny, iz=1:nz]) )
Tf.data  .= Data.Array([100*exp(-((x_g(ix,dx,g,Tf)-lx/2)/2)^2-((y_g(iy,dy,g,Tf)-ly/2)/2)^2-((z_g(iz,dz,g,Tf)-lz/3.0)/2)^2) +
                         50*exp(-((x_g(ix,dx,g,Tf)-lx/2)/2)^2-((y_g(iy,dy,g,Tf)-ly/2)/2)^2-((z_g(iz,dz,g,Tf)-lz/1.5)/2)^2) for ix=1:nx, iy=1:ny, iz=1:nz])
T2f.data .= Tf.data;

bw = min(16, n ÷ 4);
dt = min(dx^2,dy^2,dz^2)*cp_min/lam/8.1;
HIDE = get(ENV, "HIDE", "1") == "1"
function step!(Tf, T2f)
    if !HIDE
        @parallel (1:nx-2, 1:ny-2, 1:nz-2) diffusion3D_step!(T2f.data, Tf.data, Ci, lam, dt, _dx, _dy, _dz); update_halo!(g, T2f); return
    end
    @hide_halo_communication g T2f boundary_width=(bw, 2, 2) begin
        @parallel (1:nx-2, 1:ny-2, 1:nz-2) diffusion3D_step!(T2f.data, Tf.data, Ci, lam, dt, _dx, _dy, _dz);
    end
end

# Time loop (10 warmup steps, as in bench_PS.jl)
for it = 1:10
    step!(Tf, T2f); Tf, T2f = T2f, Tf
end
USE_GPU && CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD)
time_s = @elapsed begin
    for it = 11:nt
        step!(Tf, T2f); Tf, T2f = T2f, Tf
    end
    USE_GPU && CUDA.synchronize()
end

# Performance
A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Data.Number);
t_it  = time_s/(nt-10);
T_eff = A_eff/t_it;
if (me==0) println("HaloArrays n=$n nprocs=$(prod(t.dims)) time_s=$time_s t_it_ms=$(1e3*t_it) T_eff=$T_eff"); end
MPI.Finalize()
end

diffusion3D(isempty(ARGS) ? 32 : parse(Int, ARGS[1]), length(ARGS) < 2 ? 100000 : parse(Int, ARGS[2]))
