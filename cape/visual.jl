using Plots
using Plots.PlotMeasures
using JLD2
try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape")
end
using Pkg
Pkg.activate("..")
# @load "visual_snapshot/cg_lambda_plot_200_FLAT_UP_1_4.jld2" lambdas cgs non_zero_lengths nleaves_list
# lambdas
# nleaves_list
# cgs
# perm = sortperm(nleaves_list)

line_colors = [:red, :green, :blue, :purple1]

scatter_colors = [:darkred, :darkgreen, :darkblue, :purple]

gray_colors = [:lightcoral, :darkseagreen, :lightsteelblue, :gray]
#gray_colors = [RGB(227 / 255, 197 / 255, 197 / 255), RGB(174 / 255, 189 / 255, 174 / 255), RGB(197 / 255, 208 / 255, 222 / 255), :gray]

function get_plot_settings(width_cm, height_cm, xlabel, ylabel)
    dpi = 150
    width_px = round(Int, width_cm * dpi / 2.54)
    height_px = round(Int, height_cm * dpi / 2.54)
    p = plot(size=(width_px, height_px), xlabel=xlabel, ylabel=ylabel, margin=3mm, #left right top bottom
        titlefontsize=15,
        guidefontsize=12,
        tickfontsize=8,
        legendfontsize=8
    )

    return p
end

function get_plot(n, type, dir, rt)
    p = get_plot_settings(7.5, 8, "Explanation size", "Confidence gap",)
    plot!(p, #xlims=(1, 10000), xticks=[1, 10, 100, 1000],
        xscale=:log10, legend=:bottomleft)
    if type == "FLAT"
        # plot!(p, title="Flat TreeLIME optimization\ncompared to explanation size\nwith n=$(n) and rel_tol=$(rt)%")
    else
        # plot!(p, title="Layered TreeLIME optimization\ncompared to explanation size\nwith n=$(n), dir=$(dir) and rel_tol=$(rt)%")
    end
    return p
end

function get_lambda_plot(n, type, dir, rt)
    p = get_plot_settings(7.5, 8, "λ", "")
    plot!(p, legend=false)
    if type == "FLAT"
        # plot!(p, title="Flat TreeLIME optimization\ncompared to λ\nwith n=$(n) and rel_tol=$(rt)%")
    else
        # plot!(p, title="Layered TreeLIME optimization\ncompared to λ\nwith n=$(n), dir=$(dir) and rel_tol=$(rt)%")
    end
    return p
end

function plot_out!(p, x, cgs, l, findminorfindmax, min_cg, type, dir)
    line_color = line_colors[l]
    scatter_color = scatter_colors[l]
    gray_color = gray_colors[l]
    if type == "FLAT"
        line_color = line_colors[4]
        gray_color = gray_colors[4]
        scatter_color = scatter_colors[4]
    end
    above_zero_indices = cgs .>= min_cg
    below_zero_indices = .!above_zero_indices
    if (findminorfindmax == findmin)
        x = map(i -> i == 0 ? i + 1 : i, x)
    end


    plot!(p, x, cgs, label=nothing, color=gray_color, linewidth=2)
    # xlims!(p, minimum(float.(x)), maximum(float.(x)))

    parts = [[i, i + 1] for i in 1:length(x)-1]

    parts = filter(part -> (cgs[part[1]] >= min_cg) && cgs[part[2]] >= min_cg, parts)
    first = true
    layer_label = "Layer $(l)"
    if type == "FLAT"
        layer_label = "All layers"
    end
    for part in parts
        if !first
            layer_label = nothing
        end

        plot!(p, x[part], cgs[part], label=layer_label, color=line_color, linewidth=2)
        first = false
    end

    # plot!(p, x, cgs, label="Layer $(l)", color=line_colors[l], linewidth=1)#, xticks=ticks)
    # Add small dots where the points are
    scatter!(p, x, cgs, markersize=3, markercolor=scatter_color, markerstrokecolor=scatter_color, label=nothing)
    scatter!(p, x[cgs.<min_cg], cgs[cgs.<min_cg], color=gray_color, markersize=3, markerstrokecolor=gray_color, label=nothing)

    # Highlight the line which represents zero confidence gap

    positive_cgs = cgs .> min_cg
    if any(positive_cgs)
        best_index = findminorfindmax(x[positive_cgs])[2]

        max_cg = cgs[positive_cgs][best_index]
        result_label = nothing

        if (type == "FLAT") || (type == "LAYERED" && dir == "UP" && l == 1) || (type == "LAYERED" && dir == "DOWN" && l == 3)
            result_label = "Final result"
            scatter!(p, [x[positive_cgs][best_index]], [max_cg], color=:red, markersize=5, markerstrokecolor=:orange, label=result_label)
        else
            if l == 2
                result_label = "Partial result"
            end
            scatter!(p, [x[positive_cgs][best_index]], [max_cg], color=:blue, markersize=4, markerstrokecolor=:purple, label=result_label)
        end
    end
    hlinelabel = nothing
    zerolabel = nothing
    if l == 1
        hlinelabel = "Min CG"
        zerolabel = "0 CG"
    end
    hline!(p, [min_cg], color=:gray, linewidth=2, label=hlinelabel, linestyle=:dash)
    hline!(p, [0], color=:gray, linewidth=1, label=zerolabel)
    return p
end

for class in collect(1:10)
    for rel_tol in [50, 75, 90, 99]
        for type in ["FLAT", "LAYERED"]
            for n in [50, 200, 400, 1000]
                possible_dir = ["UP", "DOWN"]
                if type == "FLAT"
                    possible_dir = ["UP"]
                end
                for dir in possible_dir
                    try
                        plots = []
                        p = get_plot(n, type, dir, rel_tol)
                        p_lambda = get_lambda_plot(n, type, dir, rel_tol)
                        possible_layers = [1, 2, 3]
                        if type == "FLAT"
                            possible_layers = [1]
                        end

                        for l in possible_layers

                            @load "visual_snapshot/cg_lambda_plot_$(n)_$(type)_$(dir)_$(l)_$(rel_tol/100)_$(class).jld2" lambdas cgs non_zero_lengths nleaves_list min_cg og_class
                            # perm = sortperm(nleaves_list)
                            println("cg_lambda_plot_$(n)_$(type)_$(dir)_$(l)_4")

                            # nleaves_list = nleaves_list[perm]
                            # cgs_sorted = cgs[perm]

                            plot_out!(p, nleaves_list, cgs, l, findmin, min_cg, type, dir)

                            plot_out!(p_lambda, lambdas, cgs, l, findmax, min_cg, type, dir)

                            xflip!(p)


                        end
                        push!(plots, p)
                        push!(plots, p_lambda)
                        width_px = round(Int, 15 * 150 / 2.54)
                        height_px = round(Int, 8 * 150 / 2.54)
                        title = nothing
                        if type == "FLAT"
                            title = "Flat TreeLIME optimization compared to λ and Explnation size \nwith n=$(n) and relative tolerance=$(rel_tol)%"
                        else
                            title = "Layered TreeLIME optimization compared to λ and explnation size\nwith n=$(n), dir=$(dir) and relative tolerance=$(rel_tol)%"
                        end
                        title = plot(title=title, grid=false, showaxis=false, bottom_margin=-50Plots.px, titlefontsize=15)
                        p_comp = plot(title, plots..., size=(width_px, height_px), layout=@layout([A{0.01h}; [B C]]), top_margin=5mm)# layout=(length(plots), 1))

                        display(p_comp)
                        savefig(p_comp, "optimization_visual/$(type)_$(dir)_optimization_$(n)_$(rel_tol/100)_$(class).pdf")
                    catch e
                        println(e)
                        println("fail")
                    end
                end
            end
        end
    end
end

# Combine the plots in a 2 by 3 grid
