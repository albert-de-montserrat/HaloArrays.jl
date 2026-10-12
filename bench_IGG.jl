# Timed, no-plot variant of bench_PS.jl (ImplicitGlobalGrid) for comparison with bench_HA.jl.
# Usage: julia --project=benchmarks/gpu bench_IGG.jl [n=32] [nt=100000]
const USE_GPU = get(ENV, "USE_GPU", "0") == "1"
@static if USE_GPU; using CUDA; end
using ImplicitGlobalGrid
import MPI
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
lam        = 1.0;
cp_min     = 1.0;
lx, ly, lz = 10.0, 10.0, 10.0;
nx, ny, nz = n, n, n;
me, dims   = init_global_grid(nx, ny, nz; quiet=true);
dx         = lx/(nx_g()-1);
dy         = ly/(ny_g()-1);
dz         = lz/(nz_g()-1);
_dx, _dy, _dz = 1.0/dx, 1.0/dy, 1.0/dz;

T   = @zeros(nx, ny, nz);
T2  = @zeros(nx, ny, nz);
Ci  = @zeros(nx, ny, nz);
Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-((x_g(ix,dx,Ci)-lx/1.5))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) +
                                   5*exp(-((x_g(ix,dx,Ci)-lx/3.0))^2-((y_g(iy,dy,Ci)-ly/2))^2-((z_g(iz,dz,Ci)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]) )
T  .= Data.Array([100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
                   50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])
T2 .= T;

bw = min(16, n ÷ 4);
dt = min(dx^2,dy^2,dz^2)*cp_min/lam/8.1;
t0 = 0
for it = 1:nt
    if (it == 11) USE_GPU && CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD); t0 = time_ns(); end
    @hide_communication (bw, 2, 2) begin
        @parallel diffusion3D_step!(T2, T, Ci, lam, dt, _dx, _dy, _dz);
        update_halo!(T2);
    end
    T, T2 = T2, T;
end
USE_GPU && CUDA.synchronize()
time_s = (time_ns()-t0)/1e9

A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Data.Number);
t_it  = time_s/(nt-10);
T_eff = A_eff/t_it;
if (me==0) println("IGG n=$n nprocs=$(prod(dims)) time_s=$time_s t_it_ms=$(1e3*t_it) T_eff=$T_eff"); end
finalize_global_grid();
end

diffusion3D(isempty(ARGS) ? 32 : parse(Int, ARGS[1]), length(ARGS) < 2 ? 100000 : parse(Int, ARGS[2]))
