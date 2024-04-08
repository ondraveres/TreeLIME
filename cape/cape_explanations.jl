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
@time @load "cape_equal_extractor.jld2" extractor sch model data #takes 15 minutes
printtree(sch)
printtree(extractor)


include("../datasets/common.jl")
include("../datasets/loader.jl")
include("../datasets/stats.jl")

sample_num = 100

model
soft_model
logsoft_model
statlayer = StatsLayer()
model = @set model.m = Chain(model.m, statlayer)
soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)
Random.seed!(1)



ds = data[1:min(numobs(data), sample_num)]
model_variant_k = 3

predictions = Flux.onecold((model(ds)))
# first_indices = Dict(value => findall(==(value), predictions)[1:min(end, 100)] for value in unique(predictions))
# first_indices
# sorted = sort(first_indices)

# exdf = DataFrame()

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
# Base.eps(::Type{Any}) = eps(Float32)
# Base.typemin(::Type{Any}) = typemin(Float32)
variants = [
    #("l2-distance_10", :Flat_HAdd),
    #("l2-distance_100", :Flat_HAdd),
    ("l2-distance_1000", :Flat_HAdd),
    # ("l2-distance_5000", :Flat_HAdd),
    # ("stochastic", :Flat_HAdd)
    #("banz", :Flat_HAdd),
    #("banz", :Flat_HAdd),
    #("banz", :Flat_HAdd),
    #("banz", :Flat_HAdd),
    #("banz", :LbyL_HAdd),
    # ("lime_3000_s_0.005_b", :Flat_HAdd),
    # ("lime_100_s_0.05_b", :Flat_HAdd),
    # ("lime_1000_s_0.05_b", :Flat_HAdd),
    # ("lime_10_m_0.1_a", :Flat_HAdd),
    # ("lime_100_m_0.1_a", :Flat_HAdd),
    # ("lime_1000_m_0.1_a", :Flat_HAdd),
]
# variants = getVariants()
# ds
# @showprogress "Processing variants..."
printtree(ds[3])
og_mk = ExplainMill.create_mask_structure(ds[1], d -> SimpleMask(d))
printtree(og_mk)
numobs(ds)
for (name, pruning_method) in variants # vcat(variants, ("nothing", "nothing"))
    e = getexplainer(name; sch, extractor)
    @info "explainer $e on $name with $pruning_method"
    for j in 10:20
        global exdf
        # try
        exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        # exdf = add_cape_treelime_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        # catch e
        #     println("fail")
        #     println(e)
        # end
    end
end
printtree(ds[1])
exdf
#     end
# end
# exdf
# printtree(ds[10])
# logsoft_model(ds[10])
# vscodedisplay(exdf)
@save "valuable_cape_ex_big.bson" exdf
# @load "../datasets/extra_valuable_cape_ex_big.bson" exdf
# exdf
@load "valuable_cape_ex_big.bson" exdf

# exdf


new_df = select(exdf, :name, :pruning_method, :time, :gap, :original_confidence_gap, :nleaves, :explanation_json)
transform!(new_df, :time => (x -> round.(x, digits=2)) => :time)
transform!(new_df, :gap => (x -> first.(x)) => :gap, :original_confidence_gap => (x -> first.(x)) => :original_confidence_gap)
vscodedisplay(new_df)


# Extract the number after "_" in the name
new_df[!, :number] = [
    try
        parse(Int, replace(name, r".*_" => ""))
    catch
        1
    end for name in new_df.name
]

# Create a new column for the color
new_df[!, :color] = ifelse.(startswith.(new_df.name, "stochastic"), "red", "blue")
jitter_factor = 0.8# adjust this as needed
new_df[!, :number_jittered] = new_df.number .* (1 .+ jitter_factor .* (rand(size(new_df, 1)) .- 0.5))
new_df[!, :nleaves_jittered] = new_df.nleaves .+ jitter_factor .* (rand(size(new_df, 1)) .- 0.5)

new_df = filter(row -> row.nleaves != 0, new_df)
# Create the scatter plot with more ticks on the x and y axes
scatter(new_df[new_df.color.=="red", :number_jittered], new_df[new_df.color.=="red", :nleaves], color="red", label="stochastic", xscale=:log10, yscale=:log10, markersize=2)
scatter!(new_df[new_df.color.=="blue", :number_jittered], new_df[new_df.color.=="blue", :nleaves], color="blue", label="non-stochastic", xscale=:log10, yscale=:log10, markersize=2,
    xticks=[1, 10, 100, 1000],
    yticks=[1, 10, 100, 1000])
xlabel!("# perturbations")
ylabel!("# leaves")
