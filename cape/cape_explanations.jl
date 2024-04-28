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

ds = data[1:min(numobs(data), sample_num)]
model_variant_k = 3

predictions = Flux.onecold((model(ds)))

variants = []

for n in [50, 100, 200, 400, 1000]
    for c in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
        push!(variants, ("lime_$(n)_1_Flat_UP_$(c)_JSONDIFF", :Flat_HAdd))
        push!(variants, ("lime_$(n)_1_Flat_UP_$(c)_CONST", :Flat_HAdd))
        push!(variants, ("lime_$(n)_1_layered_DOWN_$(c)_JSONDIFF", :Flat_HAdd))
        push!(variants, ("lime_$(n)_1_layered_DOWN_$(c)_CONST", :Flat_HAdd))
        push!(variants, ("lime_$(n)_1_layered_UP_$(c)_JSONDIFF", :Flat_HAdd))
        push!(variants, ("lime_$(n)_1_layered_UP_$(c)_CONST", :Flat_HAdd))
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
        round_count = parse(Int, split_name[3])
        lime_type = parse_lime_type(split_name[4])
        direction = parse_direction(split_name[5])
        perturbation_chance = parse(Float64, split_name[6])
        distance = parse_distance(split_name[6])
        return ExplainMill.TreeLimeExplainer(perturbation_count, round_count, lime_type, direction, perturbation_chance, distance)
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
variants
for (name, pruning_method) in variants # vcat(variants, ("nothing", "nothing"))
    e = getexplainer(name;)
    @info "explainer $e on $name with $pruning_method"
    for j in [task]
        global exdf
        if e isa ExplainMill.TreeLimeExplainer
            exdf = add_cape_treelime_experiment(exdf, e, ds[j][1], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        else
            exdf = add_cape_experiment(exdf, e, ds[j], logsoft_model, predictions[j], 0.0005, name, pruning_method, j, statlayer, extractor, model_variant_k)
        end
    end
end

@save "./results/layered_and_flat_exdf_$(task).bson" exdf


