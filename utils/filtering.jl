# Libraries ____________________
using LinearAlgebra, Statistics, Distributions
# ______________________________

# Functions ____________________
"""
    EnKF((f,g),y,x₀,Σ₀; Q=1,R=1)

Solves a state estimation problem using the *Ensemble Kalman Filter (EnKF)*
for stochastic linear discrete-time state-space systems
``` math
    \\begin{aligned}
        x_k &= f(x_{k-1}) + v_k ,\\quad   v_k \\sim \\mathcal{N}(0,Q) \\\\
        y_k &= g(x_k)     + z_k ,\\quad   z_k \\sim \\mathcal{N}(0,R)
    \\end{aligned}
```
with prior distribution \$x_0 \\sim \\mathcal{N}(\\mu_0,\\Sigma_0)\$.
"""
function EnKF(sys, y, Xₛ; Q=1, R=1, verbose=false)
    f,H   = sys;          # Retrieves the model matrices
    Nᵧ,K  = size(y);      # Total number of steps
    Nₓ,Nₛ = size(Xₛ);      # Size of state-space
    
    # Initialize the state and covariance estimate sequences
    μₑ = [mean(Xₛ,dims=2) zeros(Nₓ, K-1)]; 

    # Efficient samplers for Gaussian noise
    v(k) = cholesky(Q).L*randn(Nₓ,Nₛ);
    z(k) = cholesky(R).L*randn(Nᵧ,Nₛ);
    
    # Iterates over all measurements
    p = Progress(K-1, desc="Filtering (EnKF):", dt=0.5, barlen=50, showspeed=true, color=:white);
    for k = 2:K;  if verbose; next!(p); end;
        # Prediction step _______________
        Xₛ = hcat(f.(eachcol(Xₛ))...) .+ v(k);    # Prior state estimate
        Lₖ = Xₛ .- mean(Xₛ, dims=2);             # Covariance (square-root) matrix estimate

        # Update step ___________________
        ϵₖ = z(k);
        U,Σ,_ = svd(1.2(H*Lₖ .+ ϵₖ));  Σ = Diagonal(Σ);    #

        K = pinv(Σ*Σ')*U';             #
        K = K * (y[:,k].+ϵₖ .- H*Xₛ);                    #
        K = U * K;                    #
        K = (H*Lₖ)' * K;              #
        
        Xₛ .+= Lₖ*K;               #
        μₑ[:,k] .= mean(Xₛ, dims=2);    # Posterior state estimate
    end
    # -- --
    return μₑ
end

