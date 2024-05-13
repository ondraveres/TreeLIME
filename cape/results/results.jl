try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape/results")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape/results")
end
using Pkg
Pkg.activate("../..")
using ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
using StatsPlots

function parse_lime_type(s::Union{String,SubString{String}})
    symbol = Symbol(uppercase(String(s)))
    if haskey(LIME_TYPE_DICT, symbol)
        return LIME_TYPE_DICT[symbol]
    else
        error("Invalid LimeType: $s")
    end
end
function parse_direction(s::Union{String,SubString{String}})
    symbol = Symbol(uppercase(String(s)))
    if haskey(DIRECTION_DICT, symbol)
        return DIRECTION_DICT[symbol]
    else
        return DIRECTION_DICT[:UP]
    end
end

function parse_distance(s::Union{String,SubString{String}})
    symbol = Symbol(uppercase(String(s)))
    if haskey(DISTANCE_DICT, symbol)
        return DISTANCE_DICT[symbol]
    else
        return DISTANCE_DICT[:CONST]
    end
end

LIME_TYPE_DICT = Dict(:FLAT => ExplainMill.FLAT, :LAYERED => ExplainMill.LAYERED)
DIRECTION_DICT = Dict(:UP => ExplainMill.UP, :DOWN => ExplainMill.DOWN)
DISTANCE_DICT = Dict(:JSONDIFF => ExplainMill.JSONDIFF, :CONST => ExplainMill.CONST)


exdfs = []

for task in 1:10000
    try
        @load "./data/layered_and_flat_exdf_$(task).bson" exdf
        push!(exdfs, exdf)
    catch
    end
end

exdf = vcat(exdfs...)

exdf[!, :class]

exdf = filter(row -> row[:class] != 9, exdf)

# vscodedisplay(exdf)

new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json, :sampleno, :rel_tol)
# new_df.nleaves = new_df.nleaves .+ 1
new_df.nleaves[new_df.nleaves.==0] .+= 1
transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)



function get_plot()
    width_cm = 15
    height_cm = 8
    dpi = 150
    width_px = round(Int, width_cm * dpi / 2.54)
    height_px = round(Int, height_cm * dpi / 2.54)
    return plot(size=(width_px, height_px), yscale=:log10, yticks=[1, 10, 100, 1000], ylabel="Explanation size", margin=10mm,
        titlefontsize=17,
        guidefontsize=12,
        tickfontsize=8,
        legendfontsize=10
    )
end

function plot_out(title, filename, df, category)

    folder_path = "plots/$(category)/simple"
    p = get_plot()
    title!(p, title)
    @df df dotplot!(p, :Formatted_name, :nleaves, marker=(:black, stroke(0)), legend=false, markersize=2)
    @df df boxplot!(p, :Formatted_name, :nleaves, fillalpha=0.75, linewidth=3, linecolor=:black, marker=(:black, stroke(3)), legend=false, outliers=false)
    mkpath(folder_path)
    path = "$(folder_path)/$(filename)"
    path = replace(path, " " => "")
    path = replace(path, "," => "-")
    path = replace(path, "." => "")
    path = replace(path, "=" => "is")
    savefig(p, path * ".pdf")


    folder_path = "plots/$(category)/time"
    p = get_plot()
    title!(p, title)
    try
        scatter!(p, df.time, df.nleaves, group=df.Formatted_name, legend=:outertopright, xlabel="Time in seconds", m=(:auto))
    catch
    end
    mkpath(folder_path)
    path = "$(folder_path)/$(filename)"
    path = replace(path, " " => "")
    path = replace(path, "," => "-")
    path = replace(path, "." => "")
    path = replace(path, "=" => "is")
    savefig(p, path * ".pdf")
end

function name_to_props(name)
    split_name = split(name, "_")
    perturbation_count = parse(Int, split_name[2])
    rel_tol = parse(Float64, split_name[3]) / 100
    lime_type = parse_lime_type(split_name[4])
    direction = parse_direction(split_name[5])
    perturbation_chance = parse(Float64, split_name[6])
    distance = parse_distance(split_name[6])
    return perturbation_count, rel_tol, lime_type, direction, perturbation_chance, distance
end

t = Dict(
    "lime" => "TreeLIME",
    "banz" => "Banzhaf heuristic",
    "shap" => "Shapley heuristic",
    "const" => "Constant heuristic",
    "stochastic" => "Stochastic heuristic",
    "banz\n" => "Banzhaf\nheuristic",
    "shap\n" => "Shapley\nheuristic",
    "const\n" => "Constant\nheuristic",
    "stochastic\n" => "Stochastic\nheuristic",
    "Flat" => "Flat",
    "layered" => "Level by level",
    "UP" => "Up",
    "DOWN" => "Down",
    0.0 => "Random",
    "CONST" => "Const",
    "JSONDIFF" => "JsonDiff",
)
function tr(key)
    return get(t, key, key)
end

function extract_value(s)
    m = match(r"\d+", s)
    return m !== nothing ? parse(Int, m.match) : missing
end

using Measures
possible_methods = ["lime", "banz", "shap", "const", "stochastic"]
possible_methods_with_perturbations = ["lime", "banz", "shap"]
possible_perturbations = [200, 1000]
possible_type = ["Flat", "layered"]
possible_direction = ["UP", "DOWN"]
possible_perturbation_chance = [0.0,
    0.01, 0.1, 0.2,
    1.01, 1.1, 1.2,
    2.1, 2.5, 2.9]
possible_dist = ["CONST", "JSONDIFF"]
possible_rel_tols = [50, 99]
# print(new_df.name)
# print(new_df)
i = 0
for variable in ["best_treelime", "method", "perturbations", "flat_or_layered", "perturbation_chance", "dist", "time", "rel_tol"]
    if variable == "best_treelime"
        for pertubation_count in possible_perturbations
            for type in possible_type
                # for perturbation_chance in possible_perturbation_chance
                for dist in possible_dist
                    for rel_tol in possible_rel_tols
                        filename = "n=$(pertubation_count), type=$(type), rel_tol = $(rel_tol/100), dist = $(dist)"
                        title = "Comparison of methods in $(tr(type)) mode\n with n=$(pertubation_count), relative tolerance = $(rel_tol/100) and  δ = $(tr(dist))"
                        filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_UP_1.2_$(dist)"), row[:name]), new_df)
                        perturbation_chances = map(x -> x[5], name_to_props.(filtered_df1.name))
                        if type == "layered"
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME\ndir = $(tr("UP")) \nσ = " .* string.(perturbation_chances) .* "\nδ = $(tr(dist))"
                            ) => :Formatted_name)
                        else
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME\nσ = " .* string.(perturbation_chances) .* "\nδ = $(tr(dist))"
                            ) => :Formatted_name)
                        end

                        filtered_df6 = nothing
                        if type == "layered"
                            filtered_df6 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_DOWN_1.2_$(dist)"), row[:name]), new_df)
                            perturbation_chances = map(x -> x[5], name_to_props.(filtered_df6.name))
                            transform!(filtered_df6, :name => (
                                x -> "TreeLIME\ndir = $(tr("DOWN")) \nσ = " .* string.(perturbation_chances) .* "\nδ = $(tr(dist))"
                            ) => :Formatted_name)
                        end
                        # print(filtered_df6)
                        filtered_df2 = filter(row -> occursin(Regex("shap_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df2, :name => (
                            x -> "$(tr("shap\n"))"
                        ) => :Formatted_name)
                        filtered_df3 = filter(row -> occursin(Regex("banz_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df3, :name => (
                            x -> "$(tr("banz\n"))"
                        ) => :Formatted_name)
                        filtered_df4 = filter(row -> occursin(Regex("const"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df4, :name => (
                            x -> "$(tr("const\n"))"
                        ) => :Formatted_name)
                        filtered_df5 = filter(row -> occursin(Regex("stochastic"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df5, :name => (
                            x -> "$(tr("stochastic\n"))"
                        ) => :Formatted_name)
                        combined_df = if filtered_df6 === nothing
                            vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5)
                        else
                            vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5, filtered_df6)
                        end
                        println("lime_$(pertubation_count)_$(rel_tol)_$(type)_UP_$(dist)")
                        # print(filtered_df2)
                        plot_out(title, filename, combined_df, "best_treelime")
                    end
                end
            end
        end

    elseif variable == "method"
        continue
        for pertubation_count in possible_perturbations
            for type in possible_type
                # for perturbation_chance in possible_perturbation_chance
                for dist in possible_dist
                    for rel_tol in possible_rel_tols
                        filename = "n=$(pertubation_count), type=$(type), rel_tol = $(rel_tol/100), dist = $(dist)"
                        title = "Comparison of methods in $(tr(type)) mode\n with n=$(pertubation_count), relative tolerance = $(rel_tol/100) and  δ = $(tr(dist))"
                        filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_UP_([0-9]*\\.[0-9]+)_$(dist)"), row[:name]), new_df)
                        perturbation_chances = map(x -> x[5], name_to_props.(filtered_df1.name))
                        if type == "layered"
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME\ndir = $(tr("UP")) \nσ = " .* string.(perturbation_chances) .* "\nδ = $(tr(dist))"
                            ) => :Formatted_name)
                        else
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME\nσ = " .* string.(perturbation_chances) .* "\nδ = $(tr(dist))"
                            ) => :Formatted_name)
                        end

                        filtered_df6 = nothing
                        if type == "layered"
                            filtered_df6 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_DOWN_([0-9]*\\.[0-9]+)_$(dist)"), row[:name]), new_df)
                            perturbation_chances = map(x -> x[5], name_to_props.(filtered_df6.name))
                            transform!(filtered_df6, :name => (
                                x -> "TreeLIME\ndir = $(tr("DOWN")) \nσ = " .* string.(perturbation_chances) .* "\nδ = $(tr(dist))"
                            ) => :Formatted_name)
                        end
                        # print(filtered_df6)
                        filtered_df2 = filter(row -> occursin(Regex("shap_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df2, :name => (
                            x -> "$(tr("shap\n"))"
                        ) => :Formatted_name)
                        filtered_df3 = filter(row -> occursin(Regex("banz_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df3, :name => (
                            x -> "$(tr("banz\n"))"
                        ) => :Formatted_name)
                        filtered_df4 = filter(row -> occursin(Regex("const"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df4, :name => (
                            x -> "$(tr("const\n"))"
                        ) => :Formatted_name)
                        filtered_df5 = filter(row -> occursin(Regex("stochastic"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                        transform!(filtered_df5, :name => (
                            x -> "$(tr("stochastic\n"))"
                        ) => :Formatted_name)
                        combined_df = if filtered_df6 === nothing
                            vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5)
                        else
                            vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5, filtered_df6)
                        end
                        println("lime_$(pertubation_count)_$(rel_tol)_$(type)_UP_$(dist)")
                        # print(filtered_df2)
                        plot_out(title, filename, combined_df, "methods")
                        # end
                    end
                end
            end
        end


    elseif variable == "perturbations"
        continue
        println("Action for perturbations")
        for method in possible_methods_with_perturbations
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
                            for rel_tol in possible_rel_tols
                                filename = "method=$((method)), type=$((type)), direction = $(direction) perturbation_chance = $(perturbation_chance), dist = $(dist), rel_tol = $(rel_tol)"
                                title = "$(tr(method)) in $(tr(type)) mode\nwith σ = $(tr(perturbation_chance)), δ = $(tr(dist)) and relative threshold = $(rel_tol/100)"
                                println(title)
                                filtered_df = nothing
                                if method == "lime"
                                    filtered_df = filter(row -> occursin(Regex("lime_\\d+_$(rel_tol)_$(type)_$(direction)_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                                    transform!(filtered_df, :name => (
                                        x -> "TreeLIME\nn=" .* string.(extract_value.(x))
                                        # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                                    ) => :Formatted_name)
                                else
                                    filtered_df = filter(row -> occursin(Regex("$(method)_\\d+"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd) && row[:rel_tol] == rel_tol / 100, new_df)
                                    transform!(filtered_df, :name => (x -> "$(tr(method*"\n"))\nn=" .* string.(extract_value.(x))) => :Formatted_name)
                                end
                                filtered_df[!, :perturbations] = [parse(Int, m.match) for m in match.(r"\d+", filtered_df[!, :name])]
                                sort!(filtered_df, :perturbations)

                                xorder = unique(filtered_df.Formatted_name)
                                Nx = length(xorder)
                                str = fill("", length(filtered_df.Formatted_name))
                                for (i, xi) in enumerate(xorder)
                                    j = findall(x -> x == xi, filtered_df.Formatted_name)
                                    si = " "^(Nx - i)
                                    lines = split(string(unique(filtered_df.Formatted_name[j])[1]), '\n')
                                    modified_lines = (si .* line .* si for line in lines)
                                    new_string = join(modified_lines, '\n')
                                    @. str[j] .= new_string#si * string(filtered_df.Formatted_name[j]) * si
                                end
                                filtered_df.Formatted_name = str
                                plot_out(title, filename, filtered_df, "perturbations")
                            end
                        end
                    end
                end
            end
        end


    elseif variable == "flat_or_layered"
        continue
        println("Action for flat_or_layered")
        for method in possible_methods
            for pertubation_count in possible_perturbations
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
                        for rel_tol in possible_rel_tols
                            filename = "method=$((method)), n=$(pertubation_count), perturbation_chance = $(perturbation_chance), dist = $(dist), rel_tol = $(rel_tol)"
                            title = nothing
                            if method == "lime"
                                title = "$(tr(method)) with n=$(pertubation_count)\n σ = $(tr(perturbation_chance)), δ = $(tr(dist)) and relative threshold = $(rel_tol/100)"
                            else
                                title = "$(tr(method))\n with n=$(pertubation_count) and relative threshold = $(rel_tol/100)"
                            end
                            println(title)
                            filtered_df1 = nothing
                            filtered_df2 = nothing
                            filtered_df3 = nothing
                            if method == "lime"
                                filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_Flat_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                                transform!(filtered_df1, :name => (
                                    x -> "TreeLIME in $(tr("Flat")) mode"
                                    # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                                ) => :Formatted_name)
                                filtered_df2 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_layered_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                                transform!(filtered_df2, :name => (
                                    x -> "TreeLIME in $(tr("layered"))-$(tr("UP")) mode"
                                    # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                                ) => :Formatted_name)
                                filtered_df3 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_layered_DOWN_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                                transform!(filtered_df3, :name => (
                                    x -> "TreeLIME in $(tr("layered"))-$(tr("DOWN")) mode"
                                    # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                                ) => :Formatted_name)
                            else
                                filtered_df1 = filter(row -> occursin(Regex("$(method)_$(pertubation_count)"), row[:name]) && row[:pruning_method] == :Flat_HAdd && row[:rel_tol] == rel_tol / 100, new_df)
                                transform!(filtered_df1, :name => (x -> "$(tr(method)) in $(tr("Flat")) mode") => :Formatted_name)

                                filtered_df2 = filter(row -> occursin(Regex("$(method)_$(pertubation_count)"), row[:name]) && row[:pruning_method] == :LbyLo_HAdd && row[:rel_tol] == rel_tol / 100, new_df)
                                transform!(filtered_df2, :name => (x -> "$(tr(method)) in $(tr("layered")) mode") => :Formatted_name)
                            end
                            combined_df = if filtered_df3 === nothing
                                vcat(filtered_df1, filtered_df2)
                            else
                                vcat(filtered_df1, filtered_df2, filtered_df3)
                            end

                            plot_out(title, filename, combined_df, "Flat_or_layered")
                        end
                    end
                end
            end
        end

    elseif variable == "rel_tol"
        continue
        println("Action for rel_tol")
        for method in possible_methods
            for pertubation_count in possible_perturbations
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
                        filename = "method=$((method)), n=$(pertubation_count), perturbation_chance = $(perturbation_chance), dist = $(dist)"
                        title = nothing
                        if method == "lime"
                            title = "$(tr(method)) with n=$(pertubation_count)\n σ = $(tr(perturbation_chance)), δ = $(tr(dist))"
                        else
                            title = "$(tr(method))\n with n=$(pertubation_count)"
                        end
                        println(title)
                        filtered_df1 = nothing
                        filtered_df2 = nothing
                        filtered_df3 = nothing
                        if method == "lime"
                            filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_\\d+_Flat_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME\nin $(tr("Flat")) mode\n with rel_tol=" .* string.(map(x -> x[2], name_to_props.(filtered_df1.name)))
                                # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                            filtered_df2 = filter(row -> occursin(Regex("lime_$(pertubation_count)_\\d+_layered_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df2, :name => (
                                x -> "TreeLIME\nin $(tr("layered"))-$(tr("UP")) mode\n with rel_tol=" .* string.(map(x -> x[2], name_to_props.(filtered_df2.name)))
                                # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                            filtered_df3 = filter(row -> occursin(Regex("lime_$(pertubation_count)_\\d+_layered_DOWN_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df3, :name => (
                                x -> "TreeLIME\nin $(tr("layered"))-$(tr("DOWN")) mode\n with rel_tol=" .* string.(map(x -> x[2], name_to_props.(filtered_df3.name)))
                                # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                        else
                            filtered_df1 = filter(row -> occursin(Regex("$(method)_$(pertubation_count)"), row[:name]) && row[:pruning_method] == :Flat_HAdd, new_df)
                            transform!(filtered_df1, :name => (x -> "$(tr(method)) in $(tr("Flat")) mode\n with rel_tol=" .* string.(filtered_df1[!, :rel_tol])) => :Formatted_name)

                            filtered_df2 = filter(row -> occursin(Regex("$(method)_$(pertubation_count)"), row[:name]) && row[:pruning_method] == :LbyLo_HAdd, new_df)
                            transform!(filtered_df2, :name => (x -> "$(tr(method)) in $(tr("layered")) mode\n with rel_tol=" .* string.(filtered_df2[!, :rel_tol])) => :Formatted_name)
                        end
                        combined_df = if filtered_df3 === nothing
                            vcat(filtered_df1, filtered_df2)
                        else
                            vcat(filtered_df1, filtered_df2, filtered_df3)
                        end

                        plot_out(title, filename, combined_df, "rel_tol")
                    end
                end
            end
        end



    elseif variable == "perturbation_chance"
        continue
        println("Action for perturbation_chance")
        for pertubation_count in possible_perturbations
            for type in possible_type
                possible_direction_local = possible_direction
                if type == "Flat"
                    possible_direction_local = ["UP"]
                end
                for direction in possible_direction_local


                    possible_dist_local = possible_dist
                    for dist in possible_dist_local
                        for rel_tol in possible_rel_tols
                            filename = "n=$(pertubation_count),type=$((type)), direction = $(direction), dist = $(dist), rel_tol = $(rel_tol)"
                            title = nothing
                            if type == "Flat"
                                title = "$(tr("lime")) in $(tr(type)) mode\nwith n=$(pertubation_count), δ = $(tr(dist)) and relative tolerance = $(rel_tol/100)"
                            else
                                title = "$(tr("lime")) in $(tr(type))-$(tr(direction)) mode\n with n=$(pertubation_count), δ = $(tr(dist)) and relative tolerance = $(rel_tol/100)"
                            end
                            println(title)

                            filtered_df = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_$(direction)_([0-9]*\\.[0-9]+)_$(dist)"), row[:name]), new_df)



                            filtered_df[!, :perturbation_chance] = [parse(Float64, match(Regex("([0-9]*\\.[0-9]+)"), row[:name]).match) for row in eachrow(filtered_df)]
                            sort!(filtered_df, :perturbation_chance)

                            transform!(filtered_df, [:name, :perturbation_chance] =>
                                ((name, perturbation_chance) -> "TreeLIME" .* "\nσ=" .* string.(tr.(perturbation_chance))) => :Formatted_name)

                            xorder = unique(filtered_df.Formatted_name)
                            Nx = length(xorder)
                            str = fill("", length(filtered_df.Formatted_name))
                            for (i, xi) in enumerate(xorder)
                                j = findall(x -> x == xi, filtered_df.Formatted_name)
                                si = " "^(Nx - i)
                                lines = split(string(unique(filtered_df.Formatted_name[j])[1]), '\n')
                                modified_lines = (si .* line .* si for line in lines)
                                new_string = join(modified_lines, '\n')
                                @. str[j] = new_string
                            end
                            filtered_df.Formatted_name = str
                            plot_out(title, filename, filtered_df, "perturbation_chance")
                        end
                    end
                end
            end
        end


    elseif variable == "dist"
        continue
        for pertubation_count in possible_perturbations
            for perturbation_chance in possible_perturbation_chance
                for type in possible_type
                    possible_direction_local = possible_direction
                    if type == "Flat"
                        possible_direction_local = ["UP"]
                    end
                    for direction in possible_direction_local
                        for rel_tol in possible_rel_tols
                            filename = "n=$(pertubation_count), perturbation_chance = $(perturbation_chance), type=$((type)), direction = $(direction), rel_tol = $(rel_tol)"
                            title = "$(tr("lime")) in $(tr(type))-$(tr(direction)) mode\n with n=$(pertubation_count), σ = $(tr(perturbation_chance)) and relative tolerance = $(rel_tol/100)"
                            println(title)

                            filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_$(direction)_$(perturbation_chance)_CONST"), row[:name]), new_df)
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME with δ = $(tr("CONST"))"
                                # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)

                            filtered_df2 = filter(row -> occursin(Regex("lime_$(pertubation_count)_$(rel_tol)_$(type)_$(direction)_$(perturbation_chance)_JSONDIFF"), row[:name]), new_df)
                            transform!(filtered_df2, :name => (
                                x -> "TreeLIME with δ = $(tr("JSONDIFF"))"
                                # .* "\nσ=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                            combined_df = vcat(filtered_df1, filtered_df2)
                            plot_out(title, filename, combined_df, "dist")
                        end
                    end
                end
            end
        end

    else
        println("No action defined for $variable")
    end
end


