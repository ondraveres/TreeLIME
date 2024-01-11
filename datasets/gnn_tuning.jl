# for ((i=1;i<=20;i+=1)); do  for d in  one_of_1_2trees  one_of_1_5trees  one_of_1_paths  one_of_2_5trees  one_of_2_paths  one_of_5_paths ; do  julia -p 24 artificial.jl --dataset $d --incarnation $i ; done ; done
using Pkg
Pkg.activate("..")

using ArgParse
using Flux
using Mill
using JsonGrinder
using JSON
using BSON
using Statistics
using IterTools
using TrainTools
using StatsBase
using ExplainMill
using Serialization
using Setfield
using Zygote
using Printf
using DataFrames
using ExplainMill: jsondiff, nnodes, nleaves
include("common.jl")
include("loader.jl")
include("stats.jl")

_s = ArgParseSettings()
@add_arg_table! _s begin
    ("--dataset"; default = "hepatitis"; arg_type = String)
    ("--task"; default = "one_of_1_2trees"; arg_type = String)
    ("--incarnation"; default = 2; arg_type = Int)
end
settings = parse_args(ARGS, _s; as_symbols=true)
settings = NamedTuple{Tuple(keys(settings))}(values(settings))

###############################################################
# start by loading all samples
###############################################################
samples, labels, concepts = loaddata(settings);
resultsdir(s...) = simdir(settings.dataset, settings.task, "$(settings.incarnation)", s...)

d = BSON.load(resultsdir("newmodel.bson"))
(model, extractor, sch) = d[:model], d[:extractor], d[:schema]
statlayer = StatsLayer()
model = @set model.m.m = Chain(model.m.m..., statlayer);
soft_model = @set model.m.m = Chain(model.m.m..., softmax);
soft_model = @set model.m.m = Chain(model.m.m..., softmax);
logsoft_model = @set model.m.m = Chain(model.m.m..., logsoftmax);


###############################################################
#  Helper functions for explainability
###############################################################
const ci = TrainTools.classindexes(labels);

function loadclass(k, n=typemax(Int))
    dss = map(extractor, sample(samples[ci[k]], min(n, length(ci[k])), replace=false))
    reduce(catobs, dss)
end

function onlycorrect(dss, i, min_confidence=0)
    correct = ExplainMill.predict(soft_model, dss, [1, 2]) .== i
    dss = dss[correct[:]]
    min_confidence == 0 && return (dss)
    correct = ExplainMill.confidencegap(soft_model, dss, i) .>= min_confidence
    dss[correct[:]]
end

###############################################################
#  Demonstration of explainability
###############################################################
# strain = "IP_PHONE", "GAME_CONSOLE", "SURVEILLANCE", "NAS", "HOME_AUTOMATION", "VOICE_ASSISTANT", "PC", "AUDIO", "MEDIA_BOX", "GENERIC_IOT", "IP_PHONE", "TV", "PRINTER", "MOBILE"
strain = 2
Random.seed!(settings.incarnation)
ds = loadclass(strain, 1000)
i = strain
concept_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i), concepts))
sample_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i), samples[labels.==2]))
threshold_gap = floor(0.9 * concept_gap, digits=2)
ds = onlycorrect(ds, strain, threshold_gap)
ds = ds[1:min(nobs(ds), 100)]
@info "minimum gap on concepts = $(concept_gap) on samples = $(sample_gap)"


pruning_method = :Flat_HAdd
variants = collect(Iterators.product([1.0f0, 0.1f0, 0.01f0], [1.0f0, 0.5f0, 0.1f0, 0.05f0, 0.01f0, 0.005f0]))[:]
exdf = DataFrame()
for gnnparams in collect(Iterators.product([1.0f0, 0.1f0, 0.01f0], [1.0f0, 0.5f0]))[:]
    e, n = ExplainMill.GnnExplainer(gnnparams...), 200
    name = @sprintf("gnn_%.2g_%.g", gnnparams...)

    addexperiment(DataFrame(), e, ds[1], logsoft_model, i, n, threshold_gap, name, pruning_method, 1, settings, statlayer)
    for j in 1:nobs(ds)
        global exdf
        exdf = addexperiment(exdf, e, ds[j], logsoft_model, i, n, threshold_gap, name, pruning_method, j, settings, statlayer)
    end
    BSON.@save resultsdir("gnn.bson") exdf
end