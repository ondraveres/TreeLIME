try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape")
end
using Pkg
Pkg.activate("..")

using ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
using ExplainMill: jsondiff, nnodes, nleaves
using ProgressMeter

@time @load "cape_equal_dropout_extractor2.jld2" extractor sch model data #takes 11 minutes

include("../datasets/common.jl")
include("../datasets/loader.jl")
include("../datasets/stats.jl")


sample_num = 1000
_s = ArgParseSettings()
@add_arg_table! _s begin
    ("--task"; default = "1"; arg_type = Int)
end
settings = parse_args(ARGS, _s; as_symbols=true)
settings = NamedTuple{Tuple(keys(settings))}(values(settings))
task = settings.task
println(task);


statlayer = StatsLayer()
model = @set model.m = Chain(model.m, statlayer)
# soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)
Random.seed!(1)

ds = data[1:min(numobs(data), sample_num)]
model_variant_k = 3

predictions = Flux.onecold((model(ds)))
println(predictions)



variants = []
for n in [200, 1000]
    push!(variants, ("lime_$(n)_1_Flat_UP", :Flat_HAdd))
    push!(variants, ("lime_$(n)_1_layered_DOWN", :Flat_HAdd))
    push!(variants, ("lime_$(n)_1_layered_UP", :Flat_HAdd))
end

push!(variants, ("banz", :Flat_HAdd))



function getexplainer(name; sch=nothing, extractor=nothing)
    if name == "stochastic"
        return ExplainMill.StochasticExplainer()
    elseif name == "grad"
        return ExplainMill.GradExplainer()
    elseif name == "const"
        return ExplainMill.ConstExplainer()
    elseif name == "gnn"
        return ExplainMill.GnnExplainer()
    elseif name == "banz"
        return ExplainMill.DafExplainer(100)
    elseif startswith(name, "lime")
        split_name = split(name, "_")
        perturbation_count = parse(Float64, split_name[2])
        round_count = parse(Float64, split_name[3])
        lime_type = parse_lime_type(split_name[4])
        direction = parse_direction(split_name[5])
        return ExplainMill.TreeLimeExplainer(perturbation_count, round_count, lime_type, direction)
    else
        error("unknown eplainer $name")
    end
end
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
LIME_TYPE_DICT = Dict(:FLAT => ExplainMill.FLAT, :LAYERED => ExplainMill.LAYERED)
DIRECTION_DICT = Dict(:UP => ExplainMill.UP, :DOWN => ExplainMill.DOWN)
exdf = DataFrame()
variants
for (name, pruning_method) in variants # vcat(variants, ("nothing", "nothing"))
    e = getexplainer(name;)
    @info "explainer $e on $name with $pruning_method"
    for j in [1]
        global exdf
        if e isa ExplainMill.TreeLimeExplainer
            exdf = add_cape_treelime_experiment(exdf, e, ds[j][1], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        else
            exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        end
    end
end
# printtree(ds[1])
# exdf
# vscodedisplay(exdf)

# filtered_df = filter(row -> row[:nleaves] == 0, new_df)
# sampleno_values = filtered_df[!, :sampleno]
# unique(predictions[sampleno_values])
# predictions[2]
# indices = findall(x -> x == 9, predictions)

#     end
# end
# exdf
# printtree(ds[10])
# logsoft_model(ds[10])
# vscodedisplay(exdf)
@save "./results/layered_and_flat_exdf_$(task).bson" exdf
# @load "../datasets/extra_valuable_cape_ex_big.bson" exdf
# exdf
#@load "layered_exdf.bson" exdf

# exdf


# new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json, :sampleno)
# new_df.nleaves = new_df.nleaves .+ 1
# transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
# transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)
# vscodedisplay(new_df)


# hard_df = filter(row -> !(row[:sampleno] in indices), new_df)
# easy_df = filter(row -> (row[:sampleno] in indices), new_df)



# # Extract the number after "_" in the name
# new_df[!, :number] = [
#     try
#         split_name = split(name, "_")
#         perturbation_count = parse(Int32, split_name[2])
#     catch
#         100
#     end for name in new_df.name
# ]
# using StatsPlots


# @df new_df violin(string.(:name), :nleaves, linewidth=0, yscale=:log10, size=(1200, 400))
# # @df new_df boxplot!(string.(:name), :nleaves, fillalpha=0.75, linewidth=2, yscale=:log10, size=(1200, 400))
# p = plot(size=(1200, 400), yscale=:log10, yticks=[1, 10, 100, 1000]);
# @df hard_df dotplot!(p, string.(:name), :nleaves, marker=(:red, stroke(0)), label="Hard ones");
# @df easy_df dotplot!(p, string.(:name), :nleaves, marker=(:green, stroke(0)), label="Easy ones")

