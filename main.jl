# Libraries ______________
using LinearAlgebra, JLD2, ToeplitzMatrices, SparseArrays
include("model/GrayScott.jl");
[include("utils/"*f) for f in ["simulation.jl", "plotting.jl", "filtering.jl"]];
# ________________________

# Variables ______________
T  = 5000; Δt = 1;      # Total timesteps / Discretisation interval

# Model parameters
S  = (128, 128);                                # Space dimensions
Θ  = [.1, .05, 0.042, 0.06075, S..., 1];        # Parameter vector

Ĉ = I(S[1]*S[2]); C = [Ĉ 0Ĉ];                   # Output matrix 
Q = 1e-5I(2S[1]*S[2]); R=0.001I(S[1]*S[2]);      # Covariances

# Model equations (state/output)
f = makeIntegrator((x)->GrayScott([],x,Θ), Δt);
g = (x) -> C*x;
# ________________________

# Script _________________
# FIGURE: True State (Initial vs Final time)
x₀ = zeros(S...,2); x₀[44:84,44:84,1] .= 1; x₀[54:74,54:74,2] .= 0.5;    # Initial state for simulation
t,x,_ = simulate((f,g), x₀[:], T, Q=0Q, R=0R, verbose=true);            # Simulates the system to generate data

plot_results(x, S=S, fpi=15, figsize=(550,860), snapshots=1:T, format="png")

# FIGURE: Summary of ensemble size experiments (RMSE and execution time)
data = load("codes/data/Metrics_Ensemble_2.jld2")
λ = ["" "0"; "0" ""; "0.5" "0.5"]; Nₚ = ["10" "25" "50" "75" "100"];

p = [violin(data["RMSE"][:,j,:], xlabel="", title=latexstring("\$C=[$(λ[j,1])I_{N_x/2}\\quad $(λ[j,2])I_{N_x/2}]\$")) for j=1:3]
xlabel!(p[end], L"Ensemble size - $P$"); ylabel!(p[2], L"$RMSE(x,\hat{x})$")
ylims!(p[1],(0,0.003)), ylims!(p[2],(0,0.003)), ylims!(p[3],(0,0.01))
ff = plot(p..., layout=(3,1), xticks=(1:5,Nₚ), minorgrid=nothing, legend=nothing, size=(600,500))
savefig(ff, "codes/res/RMSE_EnsembleSizes.pdf")

p = violin(mean(data["TIMES"],dims=2)[:,1,:], xlabel=L"Ensemble size - $P$", ylabel=L"Execution times $[s]$")
ff = plot(p, xticks=(1:5,Nₚ), ylims=(0,2.5e3), minorgrid=nothing, margin=5mm, legend=nothing, size=(600,250))
savefig(ff, "codes/res/TIMES_EnsembleSizes.pdf")

# FIGURE: Estimates with different output scenarios
T = 1000;                                                   # Total number of timesteps
x₀ = init_droplets(S, N=100, seed_=616);                    # Initial state for simulation
_,x,y = simulate((f,g), x₀, T, Q=Q, R=R, verbose=true);     # Simulates the system to generate data

plot_results(x, fpi=15, figsize=(550,860), snapshots=1:T, format="png", title="True state")  # Plots the true state

Xₛ = x₀ .+ 1e-2*randn(size(x₀,1), 50);                   # - Obtains state estimates 
for (i,λᵢ) = enumerate(eachrow([1 0; 0 1; 0.5 0.5]))
    Cᵢ = [λᵢ[1]*Ĉ λᵢ[2]*Ĉ];
    _,_,y = simulate((f,(x)->Cᵢ*x), x₀, T, Q=Q, R=R, verbose=true);     # Simulates the system to generate data
    xₑ = EnKF((f,Cᵢ), y, Xₛ, Q=Q, R=R, verbose=true);                   #  from the Ensemble Kalman filter -

    plot_results(xₑ, fpi=15, figsize=(550,860), snapshots=1:T, format="png",
                    title=latexstring("EnKF Estimate, \$\\lambda = $(λᵢ[1])\$"))
end

# FIGURE: Downsampling of concentration profile
T = 5000; α = 2;
H = sparse(kron(I(S[1]÷α), Circulant(repeat([ones(α); zeros(S[1]-α)],α,1))'[1:α:S[1],:]))

x₀ = zeros(S...,2); x₀[44:84,44:84,1] .= 1; x₀[54:74,54:74,2] .= 0.5;    # Initial state for simulation
t,x,_ = simulate((f,g), x₀[:], T, Q=0Q, R=0R, verbose=true);            # Simulates the system to generate data

plot_results(C*x, fpi=15, figsize=(550,480), single=true, snapshots=1:T, format=".png")
plot_results(H*C*x, S=(64,64), fpi=15, figsize=(550,480), single=true, snapshots=1:T, format=".png")

# FIGURE: Estimates with different output densities
T = 1000;                                                   # Total number of timesteps
x₀ = init_droplets(S, N=100, seed_=616);                    # Initial state for simulation

Xₛ = x₀ .+ 1e-2*randn(size(x₀,1), 50);                   # - Obtains state estimates 
for (i,αᵢ) = enumerate([2 4])
    C = [Ĉ 0Ĉ];
    H = sparse(kron(I(S[1]÷αᵢ), Circulant(repeat([ones(αᵢ); zeros(S[1]-αᵢ)],αᵢ,1))'[1:αᵢ:S[1],:]))
    
    _,_,y = simulate((f,(x)->H*C*x), x₀, T, Q=Q, R=R[1]*I(size(H,1)), verbose=true);    # Simulates the system to generate data
    xₑ = EnKF((f,H*C), y, Xₛ, Q=Q, R=R[1]*I(size(H,1)), verbose=true);                  #  from the Ensemble Kalman filter -

    plot_results(xₑ, fpi=15, figsize=(550,860), snapshots=1:T, format="png",
                    title=latexstring("EnKF Estimate, \$(\\lambda = 1, \\alpha = $(αᵢ))\$"))
end

# FIGURE: Summary of output density experiments (RMSE and execution time)
data_e = load("codes/data/Metrics_Ensemble_2.jld2")
data_r = load("codes/data/Metrics_Resolution.jld2")
data["RMSE"]  = cat(data_e["RMSE"][:,[1],:],  data_r["RMSE"],  dims=2);
data["TIMES"] = cat(data_e["TIMES"][:,[1],:], data_r["TIMES"], dims=2);
α = ["1" "2" "4"]; Nₚ = ["10" "25" "50" "75" "100"];

p = [violin(data["RMSE"][:,j,:], xlabel="", title=latexstring("\$C=[I_{N_x/2}\\quad 0I_{N_x/2}],\\quad \\alpha=$(α[j])\$")) for j=1:3]
xlabel!(p[end], L"Ensemble size - $P$"); ylabel!(p[2], L"$RMSE(x,\hat{x})$")
ylims!(p[1],(0,0.003)), ylims!(p[2],(0,0.003)), ylims!(p[3],(0,0.015))

ff = plot(p..., layout=(3,1), xticks=(1:5,Nₚ), minorgrid=nothing, legend=nothing, size=(600,500))
savefig(ff, "codes/res/RMSE_EnsembleDensity.pdf")

p = [violin(data["TIMES"][:,j,:], xlabel="", title=latexstring("\$C=[I_{N_x/2}\\quad 0I_{N_x/2}],\\quad \\alpha=$(α[j])\$")) for j=1:3]
xlabel!(p[end], L"Ensemble size - $P$"); ylabel!(p[2],L"Execution times $[s]$")
ylims!(p[1],(0,26e2)), ylims!(p[2],(0,15e2)), ylims!(p[3],(0,13e2))
yticks!(p[1],0:800:2800), yticks!(p[3],0:400:1200)

ff = plot(p..., layout=(3,1), xticks=(1:5,Nₚ), minorgrid=nothing, margin=5mm, legend=nothing, size=(600,500))
savefig(ff, "codes/res/TIMES_EnsembleDensity.pdf")

# ________________________

