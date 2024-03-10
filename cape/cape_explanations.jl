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

@time @load "cape_model_variables_equal.jld2" labelnames df_labels time_split_complete_schema extractor data model #takes 11 minutes

@load "cape_equal_extractor.jld2" extractor sch model data

include("../datasets/treelime.jl")
include("../datasets/common.jl")
include("../datasets/loader.jl")
include("../datasets/stats.jl")

sample_num = 10000

labelnames

statlayer = StatsLayer()
model = @set model.m = Chain(model.m, statlayer)
soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)
Random.seed!(1)

heuristic = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft]
uninformative = []#[:Flat_Gadd, :Flat_Garr, :Flat_Garrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
variants = vcat(
    collect(Iterators.product(["stochastic"], vcat(uninformative, heuristic)))[:],
    collect(Iterators.product(["gnn", "banz"], vcat(heuristic)))[:],
)#grad missing
ds = data[1:min(numobs(data), sample_num)]
exdf = DataFrame()
model_variant_k = 1
predictions = Flux.onecold((model(ds)))

n = length(length(variants) * sample_num)
p = Progress(n, 1)  # 
# for (name, pruning_method) in variants
#     e = getexplainer(name)
#     @info "explainer $e on $name with $pruning_method"
#     flush(stdout)

#     for j in 1:numobs(ds)
#         println(j)
#         global exdf
#         exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.8, name, :Flat_HAdd, j, statlayer, extractor, model_variant_k)
#         next!(p)  # upd
#     end
# end
# for j in 1:numobs(ds)
#     global exdf
#     exdf = add_cape_treelime_experiment(exdf, ds[429], logsoft_model, predictions[j], j, statlayer, extractor, time_split_complete_schema, 1000, model_variant_k)
# end

predictions = Flux.onecold((model(ds)))

first_indices = Dict(value => findall(==(value), predictions)[1:min(end, 3)] for value in unique(predictions))
first_indices
sorted = sort(first_indices)

for (class, sample_indexes) in pairs(sorted)
    for sample_index in sample_indexes[1]
        lables = treelime(ds[sample_index], logsoft_model, extractor, time_split_complete_schema, 10, 0.5)
        println("exploration rate: ", 1 - mean(lables .== class), ", for class ", class, "  and index ", sample_index)
        # if mean(lables .== class) == 1
        #     println("no exploration")
        # else
        #     println(lables)
        #     println("index: ", sample_index, " class: ", class)
        #     println("exploration found")
        #     return
        # end
    end
end

lables = treelime(ds[37], logsoft_model, extractor, sch, 1000, 0.05)
mask = lables
logical = ExplainMill.e2boolean(ds[37], mask, extractor)
logical_json = JSON.json(logical)
filename = "explanation.json"
# next!(p)  # upd
open(filename, "w") do f
    write(f, logical_json)
end
printtree(ds[1])
printtree(ds[1].data[:behavior][:summary])

time_split_complete_schema

for i in 1:1
    println("This is iteration number $i")
end
# ms = ExplainMill.explain(ExplainMill.GradExplainer(), ds[1], logsoft_model, predictions[1], pruning_method=:Flat_HAdd, abs_tol=0.8)
# logical = ExplainMill.e2boolean(ds[1], ms, extractor)

# OneHotArrays

# BSON.@save resultsdir(stats_filename) exdf

# Flux.OneHotMatrix <: Mill.MaybeHotMatrix
# variants[1]

# my_ngram = nothing

# my_ngramx