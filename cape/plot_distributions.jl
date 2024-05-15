using Plots
using Distributions
using Measures

function get_plot(d, t)
    width_cm = 5
    height_cm = 5
    dpi = 150
    width_px = round(Int, width_cm * dpi / 2.54)
    height_px = round(Int, height_cm * dpi / 2.54)


    x = range(0, stop=1, length=1000)
    y = pdf.(d, x)

    return plot(x, y, size=(width_px, height_px), xlabel="γ value", ylabel="Probability Density", margin=5mm, title=t, legend=false, xticks=[0, 0.5, 1],
        titlefontsize=12,
        guidefontsize=10,
        tickfontsize=8,
        legendfontsize=10
    )
end

function get_plot_c(constant, t)
    width_cm = 5
    height_cm = 5
    dpi = 150
    width_px = round(Int, width_cm * dpi / 2.54)
    height_px = round(Int, height_cm * dpi / 2.54)

    constant_distribution = x -> 0

    # Generate x values
    x_values = 0:0.01:1

    # Generate y values using the constant function
    y_values = constant_distribution.(x_values)

    c = palette(:default)[1]

    # Plot the constant function
    p = plot(x_values, y_values, ize=(width_px, height_px), xlabel="γ value", ylabel="Probability Density", margin=5mm, title=t, legend=false, xticks=[0, 0.5, 1],
        titlefontsize=12,
        guidefontsize=10,
        tickfontsize=8,
        legendfontsize=10)
    # Add an arrow at point 0.9
    annotate!(p, [(constant, 0.1, Plots.text("↑", color=c, pointsize=18))])
    scatter!(p, [constant], [0], color=c, markersize=3, markerstrokecolor=c)
    scatter!(p, [constant], [0], color=:white, markersize=2, markerstrokecolor=:white)

    return p
end

possible_d = [
    (Truncated(Normal(0.0, 0.01), 0, 1), "Truncated distribution D\nwith μ = 0, σ = 0.01"),
    (Truncated(Normal(0.0, 0.1), 0, 1), "Truncated distribution D\nwith μ = 0, σ = 0.1"),
    (Truncated(Normal(0.0, 0.2), 0, 1), "Truncated distribution D\nwith μ = 0, σ = 0.2"),
    (Truncated(Normal(1.0, 0.01), 0, 1), "Truncated distribution D\nwith μ = 1, σ = 0.01"),
    (Truncated(Normal(1.0, 0.1), 0, 1), "Truncated distribution D\nwith μ = 1, σ = 0.1"),
    (Truncated(Normal(1.0, 0.2), 0, 1), "Truncated distribution D\nwith μ = 1, σ = 0.2"),
    (Uniform(0, 1), "Uniform distribution")
]



width_cm = 15
height_cm = 15
dpi = 150
width_px = round(Int, width_cm * dpi / 2.54)
height_px = round(Int, height_cm * dpi / 2.54)

plots = [get_plot(d, t) for (d, t) in possible_d]
push!(plots, plot(get_plot_c(0.1, "Const(0.1)\ndistribution")))
pb1 = plot(legend=false, grid=false, foreground_color_subplot=:white);
push!(plots, pb1)
push!(plots, plot(get_plot_c(0.5, "Const(0.5)\ndistribution")))
push!(plots, plot(get_plot_c(0.9, "Const(0.9)\ndistribution")))
pb2 = plot(legend=false, grid=false, foreground_color_subplot=:white);


ylims!(plots[7], 0, 1.2)
l = @layout [grid(2, 4); grid(1, 4, widths=[1 / 6, 1 / 3, 1 / 3, 1 / 6])]
m = plot(plots..., layout=(3, 4), size=(width_px, height_px))

savefig(m, "distributions.pdf")


