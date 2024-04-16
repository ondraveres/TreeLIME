using Plots
using Measures
using JLD2


@load "cg_lambda_plot_200_LAYERED_UP_.jld2" lambdas cgs non_zero_lengths nleaves_list
nleaves_list


# for n in [10, 50, 100, 200, 400, 1000, 5000, 10000]
plots = []
for dir in ["UP"]
    for n in [200]
        for l in [1, 2, 3]
            @load "cg_lambda_plot_$(n)_LAYERED_$(dir)_$(l).jld2" lambdas cgs non_zero_lengths nleaves_list
            ticks = range(minimum(non_zero_lengths), maximum(non_zero_lengths), step=10)
            non_zero_lengths = non_zero_lengths .+ 1
            p = plot(nleaves_list, cgs, xlabel="lenghts", ylabel="Confidence Gaps", title="$(n)_$(dir)_$(l)", legend=false, xlims=(1, 10000), xticks=[1, 10, 100, 1000], xscale=:log10)#, xticks=ticks)
            # Add small dots where the points are
            scatter!(p, nleaves_list, cgs, markersize=2, markercolor=:darkblue, markerstrokecolor=:darkblue)

            # Highlight the line which represents zero confidence gap
            hline!(p, [0], color=:green, linewidth=1)

            positive_cgs = cgs .> 0
            if any(positive_cgs)
                best_index = findmin(non_zero_lengths[positive_cgs])[2]

                max_cg = cgs[positive_cgs][best_index]


                # Add the point to the plot
                scatter!(p, [non_zero_lengths[positive_cgs][best_index]], [max_cg], color=:red, markersize=4)
            end

            push!(plots, p)
        end
    end
end

# Combine the plots in a 2 by 3 grid
p = plot(plots..., layout=grid(10, 3), size=(1500, 2000))

# Display the plot
display(p)