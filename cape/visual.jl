using Plots
using Measures
using JLD2

@load "cg_lambda_plot_50_LAYERED_UP_1.jld2" lambdas cgs non_zero_lengths nleaves_list
nleaves_list
cgs
non_zero_lengths
lambdas


# Get the indices of the parts where both points are positive




line_colors = [:red, :green, :blue]

scatter_colors = [:darkred, :darkgreen, :darkblue]

gray_colors = [:lightcoral, :darkseagreen, :lightsteelblue]

# for n in [10, 50, 100, 200, 400, 1000, 5000, 10000]
n = 50
plots = []
for n in [50]#[200, 1000]
    for dir in ["UP", "DOWN"]
        p = plot(xlabel="lenghts", ylabel="Confidence Gaps", title="Layered $(n)_$(dir)", legend=false, xlims=(1, 10000), xticks=[1, 10, 100, 1000], xscale=:log10)
        hline!(p, [0], color=:black, linewidth=1, label="Zero confidance gap")
        for l in [1, 2, 3]
            @load "cg_lambda_plot_$(n)_LAYERED_$(dir)_$(l).jld2" lambdas cgs non_zero_lengths nleaves_list
            ticks = range(minimum(non_zero_lengths), maximum(non_zero_lengths), step=10)
            non_zero_lengths = non_zero_lengths .+ 1

            above_zero_indices = cgs .>= 0
            below_zero_indices = .!above_zero_indices

            plot!(p, nleaves_list, cgs, label=nothing, color=gray_colors[l], linewidth=1)

            parts = [[i, i + 1] for i in 1:length(nleaves_list)-1]

            parts = filter(part -> (cgs[part[1]] >= 0) && cgs[part[2]] >= 0, parts)

            for part in parts
                plot!(p, nleaves_list[part], cgs[part], label=nothing, color=line_colors[l], linewidth=1)
            end






            # plot!(p, nleaves_list, cgs, label="Layer $(l)", color=line_colors[l], linewidth=1)#, xticks=ticks)
            # Add small dots where the points are
            scatter!(p, nleaves_list, cgs, markersize=2, markercolor=scatter_colors[l], markerstrokecolor=scatter_colors[l], label=nothing)
            scatter!(p, nleaves_list[cgs.<0], cgs[cgs.<0], color=gray_colors[l], markersize=2, markerstrokecolor=gray_colors[l], label=nothing)

            # Highlight the line which represents zero confidence gap

            positive_cgs = cgs .> 0
            if any(positive_cgs)
                best_index = findmin(non_zero_lengths[positive_cgs])[2]

                max_cg = cgs[positive_cgs][best_index]


                # Add the point to the plot
                scatter!(p, [nleaves_list[positive_cgs][best_index]], [max_cg], color=:orange, markersize=3, markerstrokecolor=:orange, label=nothing)
            end

        end
        push!(plots, p)
    end
end

# Combine the plots in a 2 by 3 grid
p = plot(plots..., layout=grid(1, 3), size=(1500, 3000))

# Display the plot
display(p)