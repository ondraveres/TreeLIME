try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape")
end
using Pkg
Pkg.activate("..")

using ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
using ExplainMill: jsondiff, nnodes, nleaves

@load "cape_model_variables.jld2" labelnames df_labels time_split_complete_schema extractor data model

include("../datasets/treelime.jl")
include("../datasets/common.jl")
include("../datasets/loader.jl")
include("../datasets/stats.jl")

sample_num = 10

labelnames

statlayer = StatsLayer()
model = @set model.m = Chain(model.m, statlayer)
soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)
Random.seed!(1)

heuristic = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft]
uninformative = [:Flat_Gadd, :Flat_Garr, :Flat_Garrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
variants = vcat(
    collect(Iterators.product(["stochastic"], vcat(uninformative, heuristic)))[:],
    collect(Iterators.product(["grad", "gnn", "banz"], vcat(heuristic)))[:],
)
ds = data[1:min(numobs(data), sample_num)]
exdf = DataFrame()
model_variant_k = 1
predictions = Flux.onecold(softmax(model(ds)))
# for (name, pruning_method) in variants
#     e = getexplainer(name)
#     @info "explainer $e on $name with $pruning_method"
#     flush(stdout)

#     for j in 1:numobs(ds)
#         global exdf
#         exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.9, name, pruning_method, j, statlayer, extractor, model_variant_k)
#     end
# end
# for j in 1:numobs(ds)
#     global exdf
#     exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, model_variant_k)
# end

ms = ExplainMill.explain(ExplainMill.StochasticExplainer(), ds[1], logsoft_model, predictions[1], pruning_method=:Flat_HAdd, rel_tol=0.8)

logical = ExplainMill.e2boolean(ds[1], ms, extractor)

OneHotArrays

BSON.@save resultsdir(stats_filename) exdf

Flux.OneHotMatrix <: Mill.MaybeHotMatrix