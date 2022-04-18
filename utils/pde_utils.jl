# Libraries ______________
using LinearAlgebra
# ________________________

# Functions ______________
"""
Finite-different approximation (5p stencil) of the Laplacian operator

Arguments
    x (array) : current state [concentrations X(t,s)]
    S (double) : Number of discretised cells (Sₓ, Sᵧ)

Returns
    ∇²(x,s) (array) : the space-derivative dx(t,s)/dt
"""
function ∇²(x,S,Δs)
    x = reshape(x, Int.(S)); M = zeros(size(x));
    #                            Top         /        Bot       /         Left       /        Right     /          Center
    M[2:end-1,2:end-1] = (x[1:end-2,2:end-1] + x[3:end,2:end-1] + x[2:end-1,1:end-2] + x[2:end-1,3:end] - 4*x[2:end-1,2:end-1]);
    return M[:]
end

"""
Apply a absorving (Neumann) boundary condition given a 2D space

Arguments
    x (array) : current state [concentrations X(t,s)]
    S (double) : Number of discretised cells (Sₓ, Sᵧ)
"""
function ∂Ω_Neumann(x,S)
    x = reshape(x, Int.((S..., 2)));
    x[1,:,:] = x[2,:,:];  x[end,:,:] = x[end-1,:,:];    # Left / Right
    x[:,1,:] = x[:,2,:];  x[:,end,:] = x[:,end-1,:];    # Top / Bottom
    return x[:]
end