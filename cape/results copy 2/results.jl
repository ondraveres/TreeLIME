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


# vscodedisplay(exdf)

new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json, :sampleno)
# new_df.nleaves = new_df.nleaves .+ 1
new_df = filter(row -> row[:nleaves] != 0, new_df)
transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)
# vscodedisplay(new_df)


function get_plot()
    return plot(size=(1000, 600), yscale=:log10, yticks=[1, 10, 100, 1000], ylabel="Explanation size", margin=8mm)
end

function plot_out(title, filename, df, category)

    folder_path = "plots/$(category)/simple"
    p = get_plot()
    title!(p, title, titlefontsize=20)
    @df df dotplot!(p, :Formatted_name, :nleaves, marker=(:black, stroke(0)))
    mkpath(folder_path)
    savefig(p, "$(folder_path)/$(filename).pdf")


    folder_path = "plots/$(category)/time"
    p = get_plot()
    title!(p, title, titlefontsize=20)
    scatter!(p, df.time, df.nleaves, group=df.Formatted_name, legend=:outertopright, xlabel="Time in seconds", m=(:auto))
    mkpath(folder_path)
    savefig(p, "$(folder_path)/$(filename).pdf")
end


t = Dict(
    "lime" => "TreeLIME",
    "banz" => "Banzhaf",
    "shap" => "Shapley",
    "Flat" => "Flat",
    "layered" => "Level by level",
    "UP" => "Up",
    "DOWN" => "Down",
    0.0 => "Random",
    "CONST" => "Constant",
    "JSONDIFF" => "JsonDiff")
function tr(key)
    return get(t, key, key)
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
possible_perturbation_chance = [0.0, 0.1, 0.3, 0.5]#, 0.7, 0.9]
possible_dist = ["CONST", "JSONDIFF"]
new_df.pruning_method
for variable in ["method", "perturbations", "flat_or_layered", "perturbation_chance", "dist", "time"]
    if variable == "method"
        for pertubation_count in possible_perturbations
            for type in possible_type
                for perturbation_chance in possible_perturbation_chance
                    for dist in possible_dist
                        filename = "n=$(pertubation_count), type=$(type), perturbation_chance = $(perturbation_chance), dist = $(dist)"
                        title = "Comparison of methods in $(tr(type)) mode\n with n=$(pertubation_count)"
                        filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                        transform!(filtered_df1, :name => (
                            x -> "TreeLIME\ndirection = $(tr("UP")) \nα = $(tr(perturbation_chance)) and δ = $(tr(dist))"
                        ) => :Formatted_name)
                        filtered_df6 = nothing
                        if type == "layered"
                            filtered_df6 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_DOWN_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df6, :name => (
                                x -> "TreeLIME\ndirection = $(tr("DOWN")) \nα = $(tr(perturbation_chance)) and δ = $(tr(dist))"
                            ) => :Formatted_name)
                        end
                        filtered_df2 = filter(row -> occursin(Regex("shap_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                        transform!(filtered_df2, :name => (
                            x -> "$(tr("shap"))"
                        ) => :Formatted_name)
                        filtered_df3 = filter(row -> occursin(Regex("banz_$(pertubation_count)"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                        transform!(filtered_df3, :name => (
                            x -> "$(tr("banz"))"
                        ) => :Formatted_name)
                        filtered_df4 = filter(row -> occursin(Regex("const"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                        transform!(filtered_df4, :name => (
                            x -> "$(tr("const"))"
                        ) => :Formatted_name)
                        filtered_df5 = filter(row -> occursin(Regex("stochastic"), row[:name]) && row[:pruning_method] == (type == "Flat" ? :Flat_HAdd : :LbyLo_HAdd), new_df)
                        transform!(filtered_df5, :name => (
                            x -> "$(tr("stochastic"))"
                        ) => :Formatted_name)
                        combined_df = if filtered_df6 === nothing
                            vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5)
                        else
                            vcat(filtered_df1, filtered_df2, filtered_df3, filtered_df4, filtered_df5, filtered_df3)
                        end
                        println("lime_$(pertubation_count)_1_$(type)_UP_$(perturbation_chance)_$(dist)")
                        # print(filtered_df2)
                        plot_out(title, filename, combined_df, "methods")
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
                            filename = "method=$((method)), type=$((type)), direction = $(direction) perturbation_chance = $(perturbation_chance), dist = $(dist)"
                            title = "$(tr(method)) in $(tr(type)) mode\n with α = $(tr(perturbation_chance)) and δ = $(tr(dist))"
                            println(title)
                            filtered_df = nothing
                            if method == "lime"
                                filtered_df = filter(row -> occursin(Regex("lime_\\d+_1_$(type)_$(direction)_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                                transform!(filtered_df, :name => (
                                    x -> "TreeLIME\nn=" .* string.(extract_value.(x))
                                    # .* "\nα=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                                ) => :Formatted_name)
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
                            filtered_df.Formatted_name = str
                            plot_out(title, filename, filtered_df, "perturbations")
                        end
                    end
                end
            end
        end


    elseif variable == "flat_or_layered"
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
                        filename = "method=$((method)), n=$(pertubation_count), perturbation_chance = $(perturbation_chance), dist = $(dist)"
                        title = "$(tr(method))\n with n=$(pertubation_count), α = $(tr(perturbation_chance)) and δ = $(tr(dist))"
                        println(title)
                        filtered_df1 = nothing
                        filtered_df2 = nothing
                        filtered_df3 = nothing
                        if method == "lime"
                            filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_Flat_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df1, :name => (
                                x -> "TreeLIME in $(tr("Flat")) mode"
                                # .* "\nα=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                            filtered_df2 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_layered_UP_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df2, :name => (
                                x -> "TreeLIME in $(tr("layered"))-$(tr("UP")) mode"
                                # .* "\nα=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                            filtered_df3 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_layered_DOWN_$(perturbation_chance)_$(dist)"), row[:name]), new_df)
                            transform!(filtered_df3, :name => (
                                x -> "TreeLIME in $(tr("layered"))-$(tr("DOWN")) mode"
                                # .* "\nα=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                            ) => :Formatted_name)
                        else
                            filtered_df1 = filter(row -> occursin(Regex("$(method)_$(pertubation_count)"), row[:name]) && row[:pruning_method] == :Flat_HAdd, new_df)
                            transform!(filtered_df1, :name => (x -> "$(method) in $(tr("Flat")) mode") => :Formatted_name)

                            filtered_df2 = filter(row -> occursin(Regex("$(method)_$(pertubation_count)"), row[:name]) && row[:pruning_method] == :LbyLo_HAdd, new_df)
                            transform!(filtered_df2, :name => (x -> "$(method) in $(tr("layered")) mode") => :Formatted_name)
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


    elseif variable == "perturbation_chance"
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
                        filename = "n=$(pertubation_count),type=$((type)), direction = $(direction), dist = $(dist)"
                        title = "$(tr("lime")) in $(tr(type)) mode\n with n=$(pertubation_count) and δ = $(tr(dist))"
                        println(title)

                        filtered_df = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_$(direction)_([0-9]*\\.[0-9]+)_$(dist)"), row[:name]), new_df)



                        filtered_df[!, :perturbation_chance] = [parse(Float64, match(Regex("([0-9]*\\.[0-9]+)"), row[:name]).match) for row in eachrow(filtered_df)]
                        sort!(filtered_df, :perturbation_chance)

                        transform!(filtered_df, [:name, :perturbation_chance] =>
                            ((name, perturbation_chance) -> "TreeLIME\n" .* "\nα=" .* string.(tr.(perturbation_chance))) => :Formatted_name)

                        xorder = unique(filtered_df.Formatted_name)
                        Nx = length(xorder)
                        str = fill("", length(filtered_df.Formatted_name))
                        for (i, xi) in enumerate(xorder)
                            j = findall(x -> x == xi, filtered_df.Formatted_name)
                            si = " "^(Nx - i)
                            @. str[j] = si * string(filtered_df.Formatted_name[j]) * si
                        end
                        filtered_df.Formatted_name = str
                        plot_out(title, filename, filtered_df, "perturbation_chance")

                    end
                end
            end
        end


    elseif variable == "dist"
        for pertubation_count in possible_perturbations
            for perturbation_chance in possible_perturbation_chance
                for type in possible_type
                    possible_direction_local = possible_direction
                    if type == "Flat"
                        possible_direction_local = ["UP"]
                    end
                    for direction in possible_direction_local
                        filename = "n=$(pertubation_count), perturbation_chance = $(perturbation_chance), type=$((type)), direction = $(direction)"
                        title = "$(tr("lime")) in $(tr(type))-$(tr(direction)) mode\n with n=$(pertubation_count) and α = $(tr(perturbation_chance))"
                        println(title)

                        filtered_df1 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_$(direction)_$(perturbation_chance)_CONST"), row[:name]), new_df)
                        transform!(filtered_df1, :name => (
                            x -> "TreeLIME with δ = $(tr("CONST"))"
                            # .* "\nα=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                        ) => :Formatted_name)

                        filtered_df2 = filter(row -> occursin(Regex("lime_$(pertubation_count)_1_$(type)_$(direction)_$(perturbation_chance)_JSONDIFF"), row[:name]), new_df)
                        transform!(filtered_df2, :name => (
                            x -> "TreeLIME with δ = $(tr("JSONDIFF"))"
                            # .* "\nα=" .* string(perturbation_chance) .* "\ndist = " .* string(dist)
                        ) => :Formatted_name)
                        combined_df = vcat(filtered_df1, filtered_df2)
                        plot_out(title, filename, combined_df, "dist")
                    end
                end
            end
        end

    else
        println("No action defined for $variable")
    end
end


