
# Libraries ______________
using LinearAlgebra, ProgressMeter, Random
include("pde_utils.jl")
# ________________________

# Functions ______________
function simulate(sys, x₀, K; Q=0, R=0, μₓ=0, μᵧ=0, verbose=false)
    # Retrieves the model functions
    (f,h) = sys;

    # Initialize the state and output sequences
    x = [ x₀   zeros(size(x₀,1),   K-1)];   Nₓ = size(x,1);
    y = [h(x₀) zeros(size(h(x₀),1),K-1)];   Nᵧ = size(y,1);

    # Efficient samplers for Gaussian noise
    v(k) = rand(MvNormal(zeros(Nₓ),Q));
    z(k) = rand(MvNormal(zeros(Nᵧ),R));
    
    # -- Simulation loop --
    p = Progress(K-1, desc="Simulation:", dt=0.5, barlen=50, showspeed=true, color=:white);
    for k = 2:K;  if verbose; next!(p); end;
        x[:,k] .= f(x[:,k-1]) .+ v(k);          
        y[:,k] .= h(x[:,k])   .+ z(k);
	end  
    # -- --
    return (1:K), x, y
end;

function makeIntegrator(f, dt; ∂Ω=∂Ω_Neumann, S=(128,128), method="euler")
    if lowercase(method) == "euler"
        return (x) -> ∂Ω(x .+ dt*f(x), S);
    elseif lowercase(method) == "rk4"
        return (x) -> begin
            k₁ = f(x);
            k₂ = f(x .+ dt/2*k₁);
            k₃ = f(x .+ dt/2*k₂);
            k₄ = f(x .+ dt*k₃);
            ∂Ω(x .+ dt/6*(k₁ + 2k₂ + 2k₃ + k₄), S)
        end
    end
end

function init_droplets(S; N=5, seed_=-1)
    # Auxiliary variables
    x₀ = zeros(S[1], S[2], 2);  # Initial state (tensor form)
    dₛ = S .÷ 16;               # Droplets size

    # Random range sampler 
    if seed_>0; RNG = (s) -> MersenneTwister(seed_+s)
    else        RNG = (s) -> MersenneTwister(rand(1:99999)+s)
    end

    # Randomly position the droplets
    for k in 1:N
        p = [rand(RNG(k),1+dₛ[1]:S[1]-dₛ[1]-1), rand(RNG(2k),1+dₛ[2]:S[2]-dₛ[2]-1)];    # Random position for the droplet
        dₛᵢ = Int.(round.(dₛ .* rand(RNG(3k),2)));                                     # Random size for inner part of droplet

        x₀[p[1]-dₛ[1]:p[1]+dₛ[1], p[2]-dₛ[2]:p[2]+dₛ[2], 1] .= 1;
        x₀[p[1]-dₛᵢ[1]:p[1]+dₛᵢ[1], p[2]-dₛᵢ[2]:p[2]+dₛᵢ[2], 2] .= 0.5;
    end
    # -- --
    return x₀[:];
end
# ________________________