
# Libraries ______________
using LinearAlgebra, Plots, StatsPlots, LaTeXStrings, Measures, Dates

# Plotting settings
gr(show = true)
theme(:bright, grid=nothing)
# ________________________

# Functions ______________
function plot_results(x; S=(128,128), fpi=5, isGif=false, snapshots=[], figsize=(900,400), 
                            title="", single=false, labs=["U" "V"], format="pdf")
    # Auxiliary variables
    folder = joinpath(@__DIR__, "../res/", string(now()));

    # In case of GIF, generates the gif object using the macro
    if isGif 
        @gif for _ = 5:size(x,2)
            _plot_results(x[:,n],S,figsize,title,single,labs)
        end every fpi
    
    # In case of live-plotting, displays each frame in the sequence
    else; for n = [1; fpi:fpi:size(x,2); size(x,2)]
        p = _plot_results(x[:,n],S,figsize,title,single,labs)
        display(p)
        
        if n ∈ snapshots; 
            if !ispath(folder); mkdir(folder); end
            savefig(p, folder*"/frame_$(n)."*format)
        end
    end; end
end

function plot_metrics(M)
    # 

end

function _plot_results(x,S,figsize,title,single,labs)
    if single
        # Create the plot figures
        h = heatmap(reshape(x,S), xlabel=L"$x_1$", ylabel=L"$x_2$", colorbar_title=latexstring("Concentration - \$$(labs[1])\$"), c=:berlin);
        
        return plot(h, size=figsize, margin=2mm, guidefontsize=24, colorbar_titlefontsize=16, 
                        colorbar=:best, ticks=false, bbox_inches="tight")
    else
        # Reshapes the state-vector as a tensor, and auxiliary variables
        x₁,x₂ = get_grids(x,S)

        # Create the plot figures
        h1 = heatmap(x₁, xlabel=""      , ylabel=L"$x_2$",colorbar_title=L"Concentration - $U$", c=:berlin, title=title);
        h2 = heatmap(x₂, xlabel=L"$x_1$", ylabel=L"$x_2$",colorbar_title=L"Concentration - $V$", c=:berlin);
        
        return plot(h1, h2, layout=(2,1), size=figsize, margin=2mm, guidefontsize=24, colorbar_titlefontsize=16, 
                        colorbar=:best, ticks=false, bbox_inches="tight")
    end
end

function get_grids(x,S)
    return (reshape(x[1:Int(S[1]*S[2])], Int.(S)), reshape(x[Int(S[1]*S[2])+1:end], Int.(S)))
end
# ________________________