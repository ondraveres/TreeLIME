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

# @time @load "cape_model_variables_equal.jld2" labelnames df_labels time_split_complete_schema extractor data model #takes 11 minutes
# supertype(supertype(typeof(sch)))
@time @load "cape_equal_extractor.jld2" extractor sch model data



include("../datasets/common.jl")
include("../datasets/loader.jl")
include("../datasets/stats.jl")

sample_num = 1



statlayer = StatsLayer()
model = @set model.m = Chain(model.m, statlayer)
soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)
Random.seed!(1)



ds = data[1:min(numobs(data), sample_num)]
#exdf = DataFrame()
model_variant_k = 3
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

# predictions = Flux.onecold((model(ds)))
predictions = Flux.onecold((model(ds)))

first_indices = Dict(value => findall(==(value), predictions)[1:min(end, 100)] for value in unique(predictions))
first_indices
sorted = sort(first_indices)

#exdf = DataFrame()
variants = getVariants()
pairs(sorted)
ds
# @showprogress 1 "Processing..." for (class, sample_indexes) in pairs(sorted)
#     @showprogress "Processing samples..." for sample_index in sample_indexes
# @showprogress "Processing observations..." for j in 1:numobs(ds)
#     for c in [0.5]
#         global exdf
#         exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 10, model_variant_k, c, "sample")
#         exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 100, model_variant_k, c, "missing")
#         exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 100, model_variant_k, c, "sample")
#         exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 10, model_variant_k, c, "missing")
#         #exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 1000, model_variant_k, c, "missing")
#         #exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 1000, model_variant_k, c, "sample")
#         #exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 10000, model_variant_k, c, "missing")
#         #exdf = add_cape_treelime_experiment(exdf, ds[j], logsoft_model, predictions[j], j, statlayer, extractor, sch, 10000, model_variant_k, c, "sample")
#     end
# end
@showprogress "Processing variants..." for (name, pruning_method) in variants
    e = getexplainer(name; sch, extractor)
    @info "explainer $e on $name with $pruning_method"
    flush(stdout)
    @showprogress "Processing observations for variant..." for j in 1:numobs(ds)
        global exdf
        try
            exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.8, name, pruning_method, j, statlayer, extractor, model_variant_k)
        catch e
            println("fail")
            println(e)
        end
    end
end
#     end
# end
exdf

vscodedisplay(exdf)
@save "extra_valuable_cape_ex_big.bson" exdf
@load "../datasets/extra_valuable_cape_ex_big.bson" exdf
exdf

new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json)
transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)
vscodedisplay(new_df)



# mask = treelime(ds[18], logsoft_model, extractor, sch, 10, 0.28, "missing")
# logical = ExplainMill.e2boolean(ds[18], mask, extractor)
# logical_json = JSON.json(logical)
# filename = "explanation_with_inner.json"
# open(filename, "w") do f
#     write(f, logical_json)
# end
# ExplainMill.confidencegap(soft_model, extractor(logical), 6)[1, 1]

# e = LimeExplainer(sch, extractor, 100, 0.5)
# dd = ds[18]
# pruning_method = :Flat_HAdd
# rel_tol = 0.9
# lime_mask = ExplainMill.explain(e, ds[18], logsoft_model, pruning_method=pruning_method, rel_tol=rel_tol)
# open("lime.json", "w") do f
#     write(f, JSON.json(ExplainMill.e2boolean(dd, lime_mask, extractor)))
# end
# ExplainMill.confidencegap(soft_model, extractor(ExplainMill.e2boolean(dd, lime_mask, extractor)), 6)[1, 1]


# printtree(ds[1])
# printtree(ds[1].data[:behavior][:summary])

# time_split_complete_schema

# for i in 1:1
#     println("This is iteration number $i")
# end
# ms = ExplainMill.explain(ExplainMill.GradExplainer(), ds[1], logsoft_model, predictions[1], pruning_method=:Flat_HAdd, abs_tol=0.8)
# logical = ExplainMill.e2boolean(ds[1], ms, extractor)

# OneHotArrays

# BSON.@save resultsdir(stats_filename) exdf

# Flux.OneHotMatrix <: Mill.MaybeHotMatrix
# variants[1]

# my_ngram = nothing

# my_ngramx