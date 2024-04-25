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

for task in 1:3
    @load "./layered_and_flat_exdf_$(task).bson" exdf
    push!(exdfs, exdf)
end

exdf = vcat(exdfs...)

new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json, :sampleno)
new_df.nleaves = new_df.nleaves .+ 1
transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)
vscodedisplay(new_df)


hard_df = filter(row -> !(row[:sampleno] in indices), new_df)
easy_df = filter(row -> (row[:sampleno] in indices), new_df)



# Extract the number after "_" in the name
new_df[!, :number] = [
    try
        split_name = split(name, "_")
        perturbation_count = parse(Int32, split_name[2])
    catch
        100
    end for name in new_df.name
]


@df new_df violin(string.(:name), :nleaves, linewidth=0, yscale=:log10, size=(1200, 400))
# @df new_df boxplot!(string.(:name), :nleaves, fillalpha=0.75, linewidth=2, yscale=:log10, size=(1200, 400))
p = plot(size=(1200, 400), yscale=:log10, yticks=[1, 10, 100, 1000]);
@df new_df dotplot!(p, string.(:name), :nleaves, marker=(:red, stroke(0)), label="Hard ones")
@df easy_df dotplot!(p, string.(:name), :nleaves, marker=(:green, stroke(0)), label="Easy ones")