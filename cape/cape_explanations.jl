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


sample_num = 3000
_s = ArgParseSettings()
@add_arg_table! _s begin
    ("--task"; default = 1; arg_type = Int)
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
data = shuffle(data)

ds = data[1:min(numobs(data), sample_num)]
model_variant_k = 3

predictions = Flux.onecold((model(ds)))

variants = []

# push!(variants, ("lime_50_10_layered_UP_0.01_JSONDIFF", :Flat_HAdd))
possible_rel_tols = [50, 75, 90, 99]

for n in [50, 200, 400, 1000]
    for c in [0.01, 0.1, 0.2, 5.0]
        for rel_tol in possible_rel_tols
            push!(variants, ("lime_$(n)_$(rel_tol)_Flat_UP_$(c)_JSONDIFF", :Flat_HAdd))
            push!(variants, ("lime_$(n)_$(rel_tol)_Flat_UP_$(c)_CONST", :Flat_HAdd))
            push!(variants, ("lime_$(n)_$(rel_tol)_layered_DOWN_$(c)_JSONDIFF", :Flat_HAdd))
            push!(variants, ("lime_$(n)_$(rel_tol)_layered_DOWN_$(c)_CONST", :Flat_HAdd))
            push!(variants, ("lime_$(n)_$(rel_tol)_layered_UP_$(c)_JSONDIFF", :Flat_HAdd))
            push!(variants, ("lime_$(n)_$(rel_tol)_layered_UP_$(c)_CONST", :Flat_HAdd))
        end
    end
    push!(variants, ("banz_$(n)", :Flat_HAdd))
    push!(variants, ("banz_$(n)", :LbyLo_HAdd))
    push!(variants, ("shap_$(n)", :Flat_HAdd))
    push!(variants, ("shap_$(n)", :LbyLo_HAdd))
end

push!(variants, ("stochastic", :Flat_HAdd))
push!(variants, ("stochastic", :LbyLo_HAdd))
# # push!(variants, ("grad", :Flat_HAdd))
push!(variants, ("const", :Flat_HAdd))
push!(variants, ("const", :LbyLo_HAdd))
push!(variants, ("const", :Flat_Gadd))

function getexplainer(name; sch=nothing, extractor=nothing)
    if name == "stochastic"
        return ExplainMill.StochasticExplainer()
    elseif name == "grad"
        return ExplainMill.GradExplainer()
    elseif name == "const"
        return ExplainMill.ConstExplainer()
    elseif startswith(name, "gnn")
        split_name = split(name, "_")
        perturbation_count = parse(Int, split_name[2])
        return ExplainMill.GnnExplainer(perturbation_count)
    elseif startswith(name, "banz")
        split_name = split(name, "_")
        perturbation_count = parse(Int, split_name[2])
        return ExplainMill.BanzExplainer(perturbation_count)
    elseif startswith(name, "shap")
        split_name = split(name, "_")
        perturbation_count = parse(Int, split_name[2])
        return ExplainMill.ShapExplainer(perturbation_count)
    elseif startswith(name, "lime")
        split_name = split(name, "_")
        perturbation_count = parse(Int, split_name[2])
        rel_tol = parse(Float64, split_name[3]) / 100
        lime_type = parse_lime_type(split_name[4])
        direction = parse_direction(split_name[5])
        perturbation_chance = parse(Float64, split_name[6])
        distance = parse_distance(split_name[6])
        return ExplainMill.TreeLimeExplainer(perturbation_count, rel_tol, lime_type, direction, perturbation_chance, distance)
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


exdf = DataFrame()
for (name, pruning_method) in variants # vcat(variants, ("nothing", "nothing"))
    e = getexplainer(name;)
    @info "explainer $e on $name with $pruning_method"
    for j in [task]
        global exdf
        if e isa ExplainMill.TreeLimeExplainer
            exdf = add_cape_treelime_experiment(exdf, e, ds[j][1], logsoft_model, predictions[j], NaN, name, pruning_method, j, statlayer, extractor, model_variant_k)
        else
            for rel_tol in possible_rel_tols

                exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], Float64(rel_tol) / 100, name, pruning_method, j, statlayer, extractor, model_variant_k)
            end

        end
    end
end

try
    @save "./results/data/layered_and_flat_exdf_$(task+3000).bson" exdf
catch
    @save "./results/data/layered_and_flat_exdf_$(task+3000).bson" exdf
end

# predictions
# unique_predictions = sort(unique(predictions))
# indices = [findall(x -> x == prediction, predictions) for prediction in unique_predictions]

# first_items = [vec[1:5] for vec in indices]
# classes_ratio = []
# classes_lengths = []
# malware_names = [
#     "Adload",
#     "Emotet",
#     "HarHar",
#     "Lokibot",
#     "njRAT",
#     "Qakbot",
#     "Swisyn",
#     "Trickbot",
#     "Ursnif",
#     "Zeus",]
# ind = 0
# for class in first_items
#     ind += 1
#     println("\n\nLeaves from class $(ind)-$(malware_names[ind])\n\n", ind)
#     ratios = []
#     sizes = []
#     for item in class
#         no_mk = ExplainMill.create_mask_structure(ds[item], d -> SimpleMask(fill(true, d)))
#         og_class = Flux.onecold((model(ds[item])))[1]
#         mk = ExplainMill.create_mask_structure(ds[item], d -> SimpleMask(fill(false, d)))
#         og_cg = ExplainMill.logitconfgap(logsoft_model, ds[item][no_mk], og_class)[1]
#         mk
#         my_cgs = []
#         myrecursion(mk, my_cgs, item, mk, og_class, og_cg)

#         histogram(my_cgs, bins=50, xlabel="Confidence Gap", ylabel="Frequency", label="Histogram")
#         total_leaves = nleaves(ExplainMill.e2boolean(ds[item], no_mk, extractor))
#         positive_cgs = count(x -> x > 0, my_cgs)
#         ratio = positive_cgs / total_leaves
#         push!(ratios, ratio)
#         push!(sizes, positive_cgs)
#     end
#     push!(classes_ratio, mean(ratios))
#     push!(classes_lengths, mean(sizes))
# end
# println(predictions[1:400])
# classes_ratio
# classes_lengths
# println(classes_ratio)
# bar(ratios, yscale=:log10, yticks=[1, 10, 100, 1000])
# function remove_empty!(dict)
#     keys_to_delete = []
#     for (key, value) in dict
#         if isa(value, Dict)
#             remove_empty!(value)
#             isempty(value) && push!(keys_to_delete, key)
#         elseif isa(value, Array) && isempty(value)
#             push!(keys_to_delete, key)
#         end
#     end
#     for key in keys_to_delete
#         delete!(dict, key)
#     end
#     return dict
# end
# function myrecursion(mask, my_cgs, item, mk, og_class, og_cg, i=1)
#     if mask isa ExplainMill.ProductMask
#         # print(mask)
#         ch = children(mask)
#         for (name, value) in pairs(ch)
#             # print(name)
#             myrecursion(value, my_cgs, item, mk, og_class, og_cg)
#         end
#     elseif mask isa ExplainMill.BagMask
#         # print(mask)
#         for i in 1:length(mask.mask.x)
#             mask.mask.x[i] = true
#             myrecursion(children(mask)[1], my_cgs, item, mk, og_class, og_cg, i)
#             mask.mask.x[i] = false
#         end
#     else
#         # print(mask)
#         mask.mask.x[i] = true
#         n = nleaves(ExplainMill.e2boolean(ds[item], mk, extractor))
#         # print("N=", n)
#         if n == 1
#             cg = ExplainMill.logitconfgap(logsoft_model, ds[item][mk], og_class)[1]
#             if cg > (0.5 * og_cg)
#                 println(JSON.json(remove_empty!(ExplainMill.e2boolean(ds[item], mk, extractor))))
#                 push!(my_cgs, cg)
#             end
#         end
#         # println(i)
#         mask.mask.x[i] = false
#     end
# end
# myrecursion(mk)
# histogram(cgs)
# total_cgs = length(cgs)
# positive_cgs = count(x -> x > 0, cgs)
# ratio = positive_cgs / total_cgs
# a = 1



