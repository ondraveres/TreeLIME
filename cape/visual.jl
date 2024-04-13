using Plots
using JLD2
@load "cg_lambda_plot.jld2" lambdas cgs non_zero_lengths

ticks = range(minimum(non_zero_lengths), maximum(non_zero_lengths), step=10)

plot(non_zero_lengths, cgs, xlabel="lenghts", ylabel="Confidence Gaps", title="Confidence Gaps vs Lambdas", legend=false, xticks=ticks)
# plot(lambdas, cgs, xlabel="lambdas", ylabel="Confidence Gaps", title="Confidence Gaps vs Lambdas", legend=false)
