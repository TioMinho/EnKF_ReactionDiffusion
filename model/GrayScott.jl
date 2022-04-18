# Libraries ______________
using LinearAlgebra
include("../utils/pde_utils.jl")
# ________________________

# Functions ______________
"""
Gray-scott reaction-diffusion system

Arguments
    t (float) : time [t > 0]
    s (1D array) : position [x,y > 0]
    x (3D array) : current state [concentrations U(t,s) and V(t,s)]
    p (1D array) : parameters (D_u, D_v, F, k, Sₓ, Sᵧ)

Returns
    dx (array) : the time-derivative dx(t,s)/dt
"""
function GrayScott(t,x,p)
    # Parameters
    Dᵤ,Dᵥ,F,k,Sₓ,Sᵧ,Δs = p;
    S² = Int(Sₓ*Sᵧ);

    # Dynamics
    return [    F * (1 .- x[1:S²]) .- x[1:S²].*x[S²+1:end].^2 .+ Dᵤ*∇²(x[1:S²],    (Sₓ,Sᵧ),Δs) 
             -(F+k) * x[S²+1:end]  .+ x[1:S²].*x[S²+1:end].^2 .+ Dᵥ*∇²(x[S²+1:end],(Sₓ,Sᵧ),Δs)];
end
# ________________________