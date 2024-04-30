using Plots
using Distributions

possible_perturbation_chance = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
for perturbation_chance in possible_perturbation_chance
    d = Truncated(Normal(0.0, perturbation_chance), 0, 1)
    x = range(0, stop=1, length=1000)
    y = pdf.(d, x)
    p = plot(x, y, title="α = $perturbation_chance")
    display(p)
end

