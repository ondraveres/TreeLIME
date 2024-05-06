using Plots
using Distributions
using Measures

function get_plot(perturbation_chance)
    width_cm = 5
    height_cm = 5
    dpi = 150
    width_px = round(Int, width_cm * dpi / 2.54)
    height_px = round(Int, height_cm * dpi / 2.54)

    d = Truncated(Normal(0.0, perturbation_chance), 0, 1)
    x = range(0, stop=1, length=1000)
    y = pdf.(d, x)

    return plot(x, y, size=(width_px, height_px), xlabel="γ value", ylabel="Probability Density", margin=5mm, title="Truncated distribution D\nwith μ = 0, σ = $(perturbation_chance)",
        titlefontsize=12,
        guidefontsize=10,
        tickfontsize=8,
        legendfontsize=10
    )
end

possible_perturbation_chance = [0.01, 0.1, 0.2]
width_cm = 15
height_cm = 5
dpi = 150
width_px = round(Int, width_cm * dpi / 2.54)
height_px = round(Int, height_cm * dpi / 2.54)

plots = [get_plot(perturbation_chance) for perturbation_chance in possible_perturbation_chance]
m = plot(plots..., layout=(1, length(plots)), size=(width_px, height_px))

savefig(m, "distributions.pdf")