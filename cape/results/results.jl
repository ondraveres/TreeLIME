try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape/results")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape/results")
end
using Pkg
Pkg.activate("../..")
using ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
using StatsPlots

exdfs = []

for task in 1:100
    try
        @load "./layered_and_flat_exdf_$(task).bson" exdf
        push!(exdfs, exdf)
    catch
    end
end

exdf = vcat(exdfs...)

exdf.name .== "lime_50_1_Flat_UP_0.0_CONST"
# vscodedisplay(exdf)

new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json, :sampleno)
# new_df.nleaves = new_df.nleaves .+ 1
new_df = filter(row -> row[:nleaves] != 0, new_df)
transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)
# vscodedisplay(new_df)



filtered_df = filter(row -> occursin(r"lime_\d+_1_layered_UP_0.1_CONST", row[:name]), new_df)
filtered_df[!, :perturbations] = [parse(Int, m.match) for m in match.(r"\d+", filtered_df[!, :name])]
sort!(filtered_df, :perturbations)

xorder = unique(filtered_df.name)
Nx = length(xorder)
str = fill("", length(filtered_df.name))
for (i, xi) in enumerate(xorder)
    j = findall(x -> x == xi, filtered_df.name)
    si = " "^(Nx - i)
    @. str[j] = si * string(filtered_df.name[j]) * si
end
function extract_value(s)
    m = match(r"\d+", s)
    return m !== nothing ? parse(Int, m.match) : missing
end

using Measures
possible_methods = ["lime", "banz", "shap"]
possible_perturbations = [50, 100, 200, 400, 1000]
possible_type = ["Flat", "layered"]
possible_direction = ["UP", "DOWN"]
possible_perturbation_chance = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
possible_dist = ["CONST", "JSONDIFF"]
new_df.pruning_method
for variable in ["method", "perturbations", "flat_or_layered", "direction", "perturbation_chance", "dist"]
    if variable == "method"
        continue
        for pertubation_count in possible_perturbations
            for type in possible_type
                for perturbation_chance in possible_perturbation_chance
                    for dist in possible_dist
                        title = "n=$(pertubation_count), type=$(type), perturbation_chance = $(perturbation_chance), dist = $(dist)"
                        filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                        filtered_df6 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                        filtered_df2 = filter(row -> occursin(Regex("shap_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                        filtered_df3 = filter(row -> occursin(Regex("banz_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                        filtered_df4 = filter(row -> occursin(Regex("const"), row[:name]), new_df)
                        filtered_df5 = filter(row -> occursin(Regex("stochastic"), row[:name]), new_df)
                        combined_df = vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5, filtered_df6)
                        println("lime_$(pertubation_count)_1_$(type)_UP_$(perturbation_chance)_$(dist)")
                        # print(filtered_df2)
                        p = plot(size=(1000, 600), yscale=:log10, yticks=[1, 10, 100, 1000], title=title)
                        @df combined_df dotplot!(p, :name, :nleaves, marker=(:black, stroke(0)), label="Hard ones", xrotation=30, bottom_margin=10mm)
                        savefig(p, "plots/methods/$(title).pdf")
                    end
                end
            end
        end

    elseif variable == "perturbations"
        println("Action for perturbations")
        for method in possible_methods
            for type in possible_type
                possible_direction_local = possible_direction
                if method != "lime"
                    possible_direction_local = "X"
                elseif type == "Flat"
                    possible_direction_local = ["UP"]
                end
                for direction in possible_direction_local
                    perturbation_chance_local = possible_perturbation_chance
                    if method != "lime"
                        perturbation_chance_local = "X"
                    end
                    for perturbation_chance in perturbation_chance_local
                        possible_dist_local = possible_dist
                        if method != "lime"
                            possible_dist_local = "X"
                        end
                        for dist in possible_dist_local
                            title = "method=$(method), type=$(type), direction = $(direction) perturbation_chance = $(perturbation_chance), dist = $(dist)"
                            println(title)
                            filtered_df = nothing
                            if method == "lime"
                                filtered_df = filter(row -> occursin(Regex("lime_\\d+_1_$(type)_$(direction)_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                                transform!(filtered_df, :name => (x -> "TreeLIME \n n=" .* string.(extract_value.(x)) .* " α=" .* string(perturbation_chance) .* ", dist = " .* string(dist)) => :Formatted_name)
                            else
                                filtered_df = filter(row -> occursin(Regex("$(method)_\\d+"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                                transform!(filtered_df, :name => (x -> "$(method) \n n=" .* string.(extract_value.(x))) => :Formatted_name)
                            end
                            filtered_df[!, :perturbations] = [parse(Int, m.match) for m in match.(r"\d+", filtered_df[!, :name])]
                            sort!(filtered_df, :perturbations)

                            xorder = unique(filtered_df.Formatted_name)
                            Nx = length(xorder)
                            str = fill("", length(filtered_df.Formatted_name))
                            for (i, xi) in enumerate(xorder)
                                j = findall(x -> x == xi, filtered_df.Formatted_name)
                                si = " "^(Nx - i)
                                @. str[j] = si * string(filtered_df.Formatted_name[j]) * si
                            end

                            p = plot(size=(1000, 600), yscale=:log10, yticks=[1, 10, 100, 1000], title=title)
                            @df filtered_df dotplot!(p, str, :nleaves, marker=(:black, stroke(0)), label="Hard ones", xrotation=30, bottom_margin=10mm)
                            savefig(p, "plots/perturbations/$(title).pdf")
                        end
                    end
                end
            end
        end

    elseif variable == "flat_or_layered"
        println("Action for flat_or_layered")
    elseif variable == "direction"
        println("Action for direction")
    elseif variable == "perturbation_chance"
        println("Action for perturbation_chance")
    elseif variable == "dist"
        println("Action for dist")
    else
        println("No action defined for $variable")
    end
end

@df new_df violin(string.(:name), :nleaves, linewidth=0, yscale=:log10, size=(1200, 400));
@df new_df boxplot!(string.(:name), :nleaves, fillalpha=0.75, linewidth=2, yscale=:log10, size=(1200, 400))
p = plot(size=(1000, 600), yscale=:log10, yticks=[1, 10, 100, 1000]);
# @df filtered_df dotplot!(p, string.(:name), :nleaves, marker=(:black, stroke(0)), label="Hard ones", xrotation=45)
@df filtered_df dotplot!(p, str, :nleaves, marker=(:black, stroke(0)), label="Hard ones", xrotation=30, bottom_margin=10mm)

@df filtered_df dotplot!(p, string.(:name), :nleaves, marker=(:green, stroke(0)), label="Easy ones")


