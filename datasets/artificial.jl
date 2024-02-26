# for ((i=1;i<=20;i+=1)); do  for d in  one_of_1_2trees  one_of_1_5trees  one_of_1_paths  one_of_2_5trees  one_of_2_paths  one_of_5_paths ; do  julia -p 24 artificial.jl --dataset $d --incarnation $i ; done ; done
@time using Pkg, ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
@time using ExplainMill: jsondiff, nnodes, nleaves
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

model_name = "my-24-feb-model.bson"

###############################################################
# start by loading all samples
###############################################################

@time samples, labels, concepts = loaddata(settings);
labels = vcat(labels, fill(2, length(concepts)))
samples = vcat(samples, concepts)

resultsdir(s...) = joinpath("..", "..", "data", "sims", settings.dataset, settings.task, "$(settings.incarnation)", s...)
###############################################################
# create schema of the JSON
###############################################################
schema_file = resultsdir("schema.jdl2")
if isfile(schema_file)
    @info "Schema file exists, loading from file"
    @time sch = load(schema_file, "sch")
else
    @info "Schema file does not exist, creating new schema"
    sch = JsonGrinder.schema(vcat(samples, concepts, Dict()))
    @save schema_file sch = sch
end
@time extractor = suggestextractor(sch)
local model = reflectinmodel(
    sch,
    extractor,
    d -> Dense(d, settings.k, relu),
    all_imputing=true,
    # b = Dict("" => d -> Chain(Dense(d, settings.k, relu), Dense(settings.k, 2)))
)
@which reflectinmodel(
    sch,
    extractor,
    d -> Dense(d, settings.k, relu),
    all_imputing=true,
    # b = Dict("" => d -> Chain(Dense(d, settings.k, relu), Dense(settings.k, 2)))
)
# sch = JsonGrinder.schema(vcat(samples, concepts, Dict()))
if !isfile(resultsdir(model_name))
    !isdir(resultsdir()) && mkpath(resultsdir())
    @time extractor = suggestextractor(sch)
    trndata = extractbatch(extractor, samples)
    function makebatch()
        i = rand(1:2000, 100)
        trndata[i], Flux.onehotbatch(labels[i], 1:2)
    end
    ds = extractor(JsonGrinder.sample_synthetic(sch))
    good_model, concept_gap = nothing, 0
    random_useless = 10
    # good_model, concept_gap
    local model = reflectinmodel(
        sch,
        extractor,
        d -> Dense(d, settings.k, relu),
        all_imputing=true,
        # b = Dict("" => d -> Chain(Dense(d, settings.k, relu), Dense(settings.k, 2)))
    )
    model = @set model.m = Chain(model.m, Dense(settings.k, 2))
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


d = BSON.load(resultsdir(model_name))
(model, extractor, sch) = d[:model], d[:extractor], d[:sch]
statlayer = StatsLayer()

model = @set model.m = Chain(model.m, statlayer)
soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)


###############################################################
#  Helper functions for explainability
###############################################################
const my_class_indexes = PrayTools.classindexes(labels)

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
ds = ds[1:min(numobs(ds), 12)]
exdf = DataFrame()


# if !isfile(resultsdir("stats_" * model_name))

for (name, pruning_method) in variants
    e = getexplainer(name)
    @info "explainer $e on $name with $pruning_method"
    flush(stdout)
    for j in 1:numobs(ds)
        global exdf
        exdf = addexperiment(exdf, e, ds[j], logsoft_model, 2, 0.8, name, pruning_method, j, settings, statlayer)
    end
    BSON.@save resultsdir("stats_" * model_name) exdf
end
for j in 1:numobs(ds)
    global exdf
    exdf = add_treelime_experiment(exdf, ds[j], logsoft_model, 2, j, settings, statlayer)
    BSON.@save resultsdir("stats_" * model_name) exdf
end


t = @elapsed ms = ExplainMill.explain(ExplainMill.StochasticExplainer(), ds[1], logsoft_model, 2, pruning_method=:Flat_Gadd, abs_tol=0.1)
vscodedisplay(exdf)
