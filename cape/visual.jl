using Plots
using Measures
using JLD2
try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape")
end
using Pkg
Pkg.activate("..")

@load "visual_snapshot/cg_lambda_plot_200_FLAT_UP_1_4.jld2" lambdas cgs non_zero_lengths nleaves_list
lambdas
nleaves_list
cgs
# perm = sortperm(nleaves_list)

line_colors = [:red, :green, :blue]

scatter_colors = [:darkred, :darkgreen, :darkblue]

gray_colors = [:lightcoral, :darkseagreen, :lightsteelblue]
function get_plot_settings(width_cm, height_cm, xlabel, ylabel)
    dpi = 150
    width_px = round(Int, width_cm * dpi / 2.54)
    height_px = round(Int, height_cm * dpi / 2.54)
    p = plot(size=(width_px, height_px), xlabel=xlabel, ylabel=ylabel, margin=10mm,
        titlefontsize=15,
        guidefontsize=12,
        tickfontsize=8,
        legendfontsize=8
    )

    return p
end

function get_plot(n, type, dir)
    p = get_plot_settings(7.5, 8, "Explanation size", "Confidence gap",)
    plot!(p, xlims=(1, 10000), xticks=[1, 10, 100, 1000], xscale=:log10, legend=:bottomright)
    if type == "FLAT"
        plot!(p, title="Flat TreeLIME optimization\ncompared to explanation size\nwith n=$(n)")
    else
        plot!(p, title="Layered TreeLIME optimization\ncompared to explanation size\nwith n=$(n) and dir=$(dir)")
    end
    return p
end

function get_lambda_plot(n, type, dir)
    p = get_plot_settings(7.5, 8, "λ", "Confidence gap")
    if type == "FLAT"
        plot!(p, title="Flat TreeLIME optimization\ncompared to λ\nwith n=$(n)")
    else
        plot!(p, title="Layered TreeLIME optimization\ncompared to λ\nwith n=$(n) and dir=$(dir)")
    end
    return p
end

function plot_out!(p, x, cgs, l, findminorfindmax, min_cg)

    above_zero_indices = cgs .>= min_cg
    below_zero_indices = .!above_zero_indices
    if (findminorfindmax == findmin)
        x = map(i -> i == 0 ? i + 1 : i, x)
    end

    plot!(p, x, cgs, label=nothing, color=gray_colors[l], linewidth=2)
    # xlims!(p, minimum(float.(x)), maximum(float.(x)))

    parts = [[i, i + 1] for i in 1:length(x)-1]

    parts = filter(part -> (cgs[part[1]] >= min_cg) && cgs[part[2]] >= min_cg, parts)

    for part in parts
        plot!(p, x[part], cgs[part], label=nothing, color=line_colors[l], linewidth=2)
    end






    # plot!(p, x, cgs, label="Layer $(l)", color=line_colors[l], linewidth=1)#, xticks=ticks)
    # Add small dots where the points are
    scatter!(p, x, cgs, markersize=3, markercolor=scatter_colors[l], markerstrokecolor=scatter_colors[l], label=nothing)
    scatter!(p, x[cgs.<min_cg], cgs[cgs.<min_cg], color=gray_colors[l], markersize=3, markerstrokecolor=gray_colors[l], label=nothing)

    # Highlight the line which represents zero confidence gap

    positive_cgs = cgs .> min_cg
    if any(positive_cgs)
        best_index = findminorfindmax(x[positive_cgs])[2]

        max_cg = cgs[positive_cgs][best_index]
        result_label = nothing
        if l == 1
            result_label = "Result"
        end

        # Add the point to the plot
        # scatter!(p, [x[positive_cgs][best_index]], [max_cg], color=:red, markersize=5, markerstrokecolor=:orange, label=result_label)
    end
    hlinelabel = nothing
    if l == 1
        hlinelabel = "Min CG"
    end
    hline!(p, [min_cg], color=:gray, linewidth=2, label=hlinelabel, linestyle=:dash)
    hline!(p, [0], color=:gray, linewidth=1, label=hlinelabel)
    return p
end


for rel_tol in [50, 75, 90, 99]
    for type in ["FLAT", "LAYERED"]
        for n in [50, 200, 400, 1000]
            plots = []
            possible_dir = ["UP", "DOWN"]
            if type == "FLAT"
                possible_dir = ["UP"]
            end
            try
                for dir in possible_dir
                    possible_layers = [1, 2]#, 3]
                    if type == "FLAT"
                        possible_layers = [1]
                    end

                    p = get_plot(n, type, dir)
                    p_lambda = get_lambda_plot(n, type, dir)
                    for l in possible_layers

                        @load "visual_snapshot/cg_lambda_plot_$(n)_$(type)_$(dir)_$(l)_$(rel_tol/100)_4.jld2" lambdas cgs non_zero_lengths nleaves_list min_cg
                        # perm = sortperm(nleaves_list)
                        println("cg_lambda_plot_$(n)_$(type)_$(dir)_$(l)_4")

                        # nleaves_list = nleaves_list[perm]
                        # cgs_sorted = cgs[perm]

                        plot_out!(p, nleaves_list, cgs, l, findmin, min_cg)

                        plot_out!(p_lambda, lambdas, cgs, l, findmax, min_cg)




                    end
                    push!(plots, p)
                    push!(plots, p_lambda)
                end
                width_px = round(Int, 20 * 150 / 2.54)
                height_px = round(Int, 15 * 150 / 2.54)
                p = plot(plots..., size=(width_px, height_px))

                # Display the plot
                display(p)
                savefig(p, "optimization_visual/$(type)_optimization_$(n)_$(rel_tol).pdf")
            catch
            end
        end
    end
end

# Combine the plots in a 2 by 3 grid
