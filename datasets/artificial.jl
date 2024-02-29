# for ((i=1;i<=20;i+=1)); do  for d in  one_of_1_2trees  one_of_1_5trees  one_of_1_paths  one_of_2_5trees  one_of_2_paths  one_of_5_paths ; do  julia -p 24 artificial.jl --dataset $d --incarnation $i ; done ; done
using Pkg, ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
using ExplainMill: jsondiff, nnodes, nleaves

try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/datasets")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/datasets")
end
Pkg.activate("..")
include("common.jl")
include("loader.jl")
include("stats.jl")
include("treelime.jl")

function mypredict(mymodel::Mill.AbstractMillModel, ds::Mill.AbstractMillNode, ikeyvalmap)
    o = mapslices(x -> ikeyvalmap[argmax(x)], mymodel(ds), dims=1)
end
_s = ArgParseSettings()
@add_arg_table! _s begin
    ("--dataset"; default = "mutagenesis"; arg_type = String)
    ("--task"; default = "one_of_1_5trees"; arg_type = String)
    ("--incarnation"; default = 8; arg_type = Int)
    ("-k"; default = 5; arg_type = Int)
end


settings = parse_args(ARGS, _s; as_symbols=true)

settings = NamedTuple{Tuple(keys(settings))}(values(settings))



###############################################################
# start by loading all samples
###############################################################

samples, labels, concepts = loaddata(settings)
labels = vcat(labels, fill(2, length(concepts)))
samples = vcat(samples, concepts)

resultsdir(s...) = joinpath("..", "..", "data", "sims", settings.dataset, settings.task, "$(settings.incarnation)", s...)
###############################################################
# create schema of the JSON
###############################################################
schema_file = resultsdir("schema.jdl2")
global sch = nothing
if isfile(schema_file)
    @info "Schema file exists, loading from file"
    global sch = load(schema_file, "sch")
else
    @info "Schema file does not exist, creating new schema"
    global sch = JsonGrinder.schema(vcat(samples, concepts, Dict()))
    @save schema_file sch = sch
end
exdf = DataFrame()

extractor = suggestextractor(sch)
model_variant_k = 3
model_name = "my-24-feb-model-variant-$(model_variant_k).bson"
if !isfile(resultsdir(model_name))
    !isdir(resultsdir()) && mkpath(resultsdir())
    trndata = extractbatch(extractor, samples)
    function makebatch()
        i = rand(1:2000, 100)
        trndata[i], Flux.onehotbatch(labels[i], 1:2)
    end
    ds = extractor(JsonGrinder.sample_synthetic(sch))
    good_model, concept_gap = nothing, 0
    random_useless = 10
    # good_model, concept_gap
    model = reflectinmodel(
        sch,
        extractor,
        d -> Dense(d, model_variant_k, relu),
        all_imputing=true,
        # b = Dict("" => d -> Chain(Dense(d, k, relu), Dense(k, 2)))
    )
    model = @set model.m = Chain(model.m, Dense(model_variant_k, 2))
    for i in 1:100
        @info "start of epoch $i"
        ###############################################################
        #  train
        ###############################################################

        opt = ADAM()
        ps = Flux.params(model)
        loss = (x, y) -> Flux.logitcrossentropy(model(x), y)
        data_loader = Flux.DataLoader((trndata, Flux.onehotbatch(labels, 1:2)), batchsize=100, shuffle=true)


        Flux.Optimise.train!(loss, ps, data_loader, opt)

        soft_model = @set model.m = Chain(model.m, softmax)
        cg = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), 2)[1, 1], concepts))
        eg = ExplainMill.confidencegap(soft_model, extractor(JSON.parse("{}")), 1)[1, 1]
        predictions = model(trndata)
        accuracy(ds, y) = mean(Flux.onecold(model(ds)) .== y)
        acc = mean(Flux.onecold(predictions) .== labels)
        @info "crossentropy on all samples = ", Flux.logitcrossentropy(predictions, Flux.onehotbatch(labels, 1:2)),
        @info "accuracy on all samples = ", acc
        @info "minimum gap on concepts = $(cg) on empty sample = $(eg)"
        @info "accuracy on concepts = $( accuracy(extractor.(concepts), 2)))"
        @info "end of epoch $i"
        flush(stdout)

        mean(Flux.onecold(predictions) .== labels)

        if (acc > 0.999)
            break
        end
        # if cg > 0 && eg > 0
        #     if cg > concept_gap
        #         good_model, concept_gap = model, cg
        #     end
        # end
        # concept_gap > 0.95 && break
    end
    if concept_gap < 0
        error("Failed to train a model")
    end
    BSON.@save resultsdir(model_name) model extractor sch
end
for model_variant_k in []#[3, 4, 5]
    model_name = "my-24-feb-model-variant-$(model_variant_k).bson"
    d = BSON.load(resultsdir(model_name))
    (model, extractor, sch) = d[:model], d[:extractor], d[:sch]
    statlayer = StatsLayer()

    model = @set model.m = Chain(model.m, statlayer)
    soft_model = @set model.m = Chain(model.m, softmax)
    logsoft_model = @set model.m = Chain(model.m, logsoftmax)


    ###############################################################
    #  Helper functions for explainability
    ###############################################################
    my_class_indexes = PrayTools.classindexes(labels)

    function loadclass(k, n=typemax(Int))
        dss = map(s -> extractor(s, store_input=true), sample(samples[my_class_indexes[k]], min(n, length(my_class_indexes[k])), replace=false))
        reduce(catobs, dss)
    end


    function onlycorrect(dss, i, min_confidence=0)
        correct = mypredict(soft_model, dss, [1, 2]) .== i
        dss = dss[correct[:]]
        min_confidence == 0 && return (dss)
        correct = ExplainMill.confidencegap(soft_model, dss, i) .>= min_confidence
        dss[correct[:]]
    end

    function getexplainer(name)
        if name == "stochastic"
            return ExplainMill.StochasticExplainer()
        elseif name == "grad"
            return ExplainMill.GradExplainer2()
        elseif name == "gnn"
            return ExplainMill.GnnExplainer()
        elseif name == "gnn2"
            return ExplainMill.GnnExplainer()
        elseif name == "banz"
            return ExplainMill.DafExplainer()
        else
            error("unknown eplainer $name")
        end
    end

    ###############################################################
    #  Explainability experiments
    ###############################################################

    Random.seed!(settings.incarnation)
    strain = 2
    ds = loadclass(strain, 100)

    i = strain
    concept_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i)[1, 1], concepts))
    sample_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i)[1, 1], samples[labels.==2]))
    threshold_gap = 0.2

    correct_ds = onlycorrect(ds, strain, 0.1)
    ds = correct_ds
    @info "minimum gap on concepts = $(concept_gap) on samples = $(sample_gap)"

    heuristic = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft]
    uninformative = [:Flat_Gadd, :Flat_Garr, :Flat_Garrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
    variants = vcat(
        collect(Iterators.product(["stochastic"], vcat(uninformative, heuristic)))[:],
        collect(Iterators.product(["grad", "gnn", "gnn2", "banz"], vcat(heuristic)))[:],
    )
    ds = ds[1:min(numobs(ds), 100)]

    # if !isfile(resultsdir("stats_" * model_name))
    collect(Iterators.product(["stochastic"], vcat(uninformative, heuristic)))[:]
    print(variants)

    for (name, pruning_method) in variants
        e = getexplainer(name)
        @info "explainer $e on $name with $pruning_method"
        flush(stdout)
        for j in 1:numobs(ds)
            global exdf
            exdf = addexperiment(exdf, e, ds[j], logsoft_model, 2, 0.9, name, pruning_method, j, settings, statlayer, model_variant_k, extractor)
        end
        BSON.@save resultsdir("triple_stats_" * model_name) exdf
    end
    for j in 1:numobs(ds)
        global exdf
        exdf = add_treelime_experiment(exdf, ds[j], logsoft_model, 2, j, settings, statlayer, model_variant_k, extractor)
        BSON.@save resultsdir("triple_stats_" * model_name) exdf
    end
    # t = @elapsed ms = ExplainMill.explain(ExplainMill.StochasticExplainer(), ds[1], logsoft_model, 2, pruning_method=:Flat_Gadd, abs_tol=0.1)
end
# vscodedisplay(exdf)
# @save "stability_data.bson" exdf

@load "stability_data.bson" exdf

exdf2 = DataFrame(exdf)


function dice_coefficient(a, b)
    size_a = nnodes(a)
    size_b = nnodes(b)
    ce = jsondiff(a, b)
    ec = jsondiff(b, a)

    misses_nodes = nnodes(ce)
    excess_nodes = nnodes(ec)

    dc = 2 * (size_a + size_b - misses_nodes - excess_nodes) / (size_a + size_b)
    return dc
end


grouped_df = DataFrames.groupby(exdf, [:name, :pruning_method, :sampleno, :incarnation])

json_string1 = collect(grouped_df)[1][!, :explanation_json][1]
json_string2 = collect(grouped_df)[1][!, :explanation_json][2]
json_string3 = collect(grouped_df)[1][!, :explanation_json][3]

dict1 = JSON.parse(json_string1)
dict2 = JSON.parse(json_string2)
dict3 = JSON.parse(json_string3)

ce = jsondiff(dict1, dict2)
ec = jsondiff(dict2, dict1)

misses_nodes = nnodes(ce)
excess_nodes = nnodes(ec)
length([])

nnodes(dict1)
nnodes(dict2)

dice_coefficient(dict1, dict3)

dict1
dict2