# for ((i=1;i<=20;i+=1)); do  for d in  one_of_1_2trees  one_of_1_5trees  one_of_1_paths  one_of_2_5trees  one_of_2_paths  one_of_5_paths ; do  julia -p 24 artificial.jl --dataset $d --incarnation $i ; done ; done
try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/datasets")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/datasets")
end
using Pkg
Pkg.activate("..")
using ArgParse, Flux, Mill, JsonGrinder, JSON, BSON, Statistics, IterTools, PrayTools, StatsBase, ExplainMill, Serialization, Setfield, DataFrames, HierarchicalUtils, Random, JLD2, GLMNet, Plots, Zygote
using ExplainMill: jsondiff, nnodes, nleaves


sample_num = 20
iter_count = 50
k_variants = [3, 4, 5]
stats_filename = "stability_data4.bson"

include("common.jl")
include("loader.jl")
include("stats.jl")


_s = ArgParseSettings()
@add_arg_table! _s begin
    ("--dataset"; default = "deviceid"; arg_type = String)
    ("--task"; default = "one_of_1_1trees"; arg_type = String)
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

samples[3]

resultsdir(s...) = joinpath("..", "..", "data", "sims", settings.dataset, settings.task, "$(settings.incarnation)", s...)
###############################################################
# create schema of the JSON
###############################################################
schema_file = resultsdir("schema.jdl2")
sch = nothing
if isfile(schema_file)
    @info "Schema file exists, loading from file"
    global sch = load(schema_file, "sch")
else
    @info "Schema file does not exist, creating new schema"
    global sch = JsonGrinder.schema(vcat(samples, concepts, Dict()))
    @save schema_file sch = sch
end

sch
printtree(sch)
exdf = DataFrame()
extractor = suggestextractor(sch)
dd = extractor(samples[10012])
model_variant_k = 3
model_name = "my-2-march-model-variant-$(model_variant_k).bson"

samples[10099]
index = findfirst(sample -> get(sample, "ip", nothing) == "172.30.100.28", samples)
ds = extractor(samples[10099])
mask = ExplainMill.create_mask_structure(ds, d -> SimpleMask(fill(true, d)))

printtree(mask)
fv = ExplainMill.FlatView(mask)
size(fv.itemmap)
fv.itemmap
new_mask_bool_vector = [fv[i] for i in 1:length(fv.itemmap)]





global extractor
global sch
model_variant_k = 10
model_name = "my-2-march-model-variant-$(model_variant_k).bson"
# if true || !isfile(resultsdir(model_name))
!isdir(resultsdir()) && mkpath(resultsdir())
trndata = extractbatch(extractor, samples)
ds = extractor(JsonGrinder.sample_synthetic(sch))
good_model, concept_gap = nothing, 0
# good_model, concept_gap
labels[1]
JSON.parse("{}")
# model = reflectinmodel(
#     sch,
#     extractor,
#     d -> Dense(d, model_variant_k, relu),
#     all_imputing=true
# )
# model = @set model.m = Chain(model.m, Dense(model_variant_k, 2))
# for i in 1:200
#     @info "start of epoch $i"
#     opt = ADAM()
#     ps = Flux.params(model)
#     loss = (x, y) -> Flux.logitcrossentropy(model(x), y)
#     data_loader = Flux.DataLoader((trndata, Flux.onehotbatch(labels, 1:2)), batchsize=2000, shuffle=true)
#     Flux.Optimise.train!(loss, ps, data_loader, opt)
#     soft_model = @set model.m = Chain(model.m, softmax)
#     cg = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), 2)[1, 1], concepts))
#     eg = ExplainMill.confidencegap(soft_model, extractor(JSON.parse("{}")), 1)[1, 1]
#     predictions = model(trndata)
#     accuracy(ds, y) = mean(Flux.onecold(model(ds)) .== y)
#     acc = mean(Flux.onecold(predictions) .== labels)
#     @info "crossentropy on all samples = ", Flux.logitcrossentropy(predictions, Flux.onehotbatch(labels, 1:2)),
#     @info "accuracy on all samples = ", acc
#     @info "minimum gap on concepts = $(cg) on empty sample = $(eg)"
#     @info "accuracy on concepts = $( accuracy(extractor.(concepts), 2)))"
#     @info "end of epoch $i"
#     flush(stdout)
#     if (acc > 0.999)
#         break
#     end
# end
# if concept_gap < 0
#     error("Failed to train a model")
# end
# BSON.@save resultsdir(model_name) model extractor sch
# end

labels
d = BSON.load(resultsdir(model_name))
(model, extractor, sch) = d[:model], d[:extractor], d[:sch]
statlayer = StatsLayer()
model = @set model.m = Chain(model.m, statlayer)
soft_model = @set model.m = Chain(model.m, softmax)
logsoft_model = @set model.m = Chain(model.m, logsoftmax)
my_class_indexes = PrayTools.classindexes(labels)
Random.seed!(120)
strain = 2
ds = loadclass(strain, my_class_indexes, sample_num)
i = strain
concept_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i)[1, 1], concepts))
sample_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i)[1, 1], samples[labels.==2]))
threshold_gap = 0.2
correct_ds = onlycorrect(ds, strain, soft_model, 0.1)
# ds = correct_ds
@info "minimum gap on concepts = $(concept_gap) on samples = $(sample_gap)"


heuristic = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft]
uninformative = [:Flat_Gadd, :Flat_Garr, :Flat_Garrft, :LbyL_Gadd, :LbyL_Garr, :LbyL_Garrft]
variants = vcat(
    collect(Iterators.product(["stochastic"], vcat(uninformative, heuristic)))[:],
    collect(Iterators.product(["grad", "gnn", "banz"], vcat(heuristic)))[:],
)
ds = ds[1:min(numobs(ds), sample_num)]
mask = ExplainMill.create_mask_structure(ds[1], d -> SimpleMask(fill(true, d)))
printtree(mask)
# for (name, pruning_method) in variants
#     e = getexplainer(name)
#     @info "explainer $e on $name with $pruning_method"
#     flush(stdout)
#     for j in 1:numobs(ds)
#         global exdf
#         exdf = addexperiment(exdf, e, ds[j], logsoft_model, 2, 0.9, name, pruning_method, j, settings, statlayer, model_variant_k, extractor)
#     end
# end
# for j in 1:numobs(ds)
#     global exdf
#     exdf = add_treelime_experiment(exdf, ds[j], logsoft_model, 2, j, settings, statlayer, model_variant_k, extractor)
# end





dd = extractor(samples[10099], store_input=true)
mask = ExplainMill.create_mask_structure(dd, d -> SimpleMask(fill(true, d)))
mask = ExplainMill.create_mask_structure(dd, d -> SimpleMask(fill(true, d)))

mk = ExplainMill.create_mask_structure(dd, d -> SimpleMask(ones(Bool, d)))

printtree(mk)

model(dd)

fv = ExplainMill.FlatView(mk)

ranking = [15, 30, 5, 12, 17, 2, 10, 23, 19, 18, 26, 24, 11, 25, 7, 21, 31, 4, 6, 13, 14, 1, 20, 9, 27, 16, 29, 8, 3, 22, 28]
length(fv.itemmap)
length(ranking)
for i in 1:31
    fv[i] = ranking[i]
end
printtree(mk)

printtree(mask)
pruning_method = :Flat_HAdd
rel_tol = 0.9
Random.seed!(1)
stochastic_mask = ExplainMill.explain(StochasticExplainer(), dd, logsoft_model, i, pruning_method=pruning_method, rel_tol=rel_tol)

printtree(stochastic_mask)

grad_mask = ExplainMill.explain(DafExplainer(200, true, false, extractor), dd, logsoft_model, i, pruning_method=pruning_method, rel_tol=rel_tol)
printtree(grad_mask)

model()

const_mask = ExplainMill.explain(ConstExplainer(), dd, logsoft_model, i, pruning_method=pruning_method, rel_tol=rel_tol)
lime_m_mask = ExplainMill.explain(LimeExplainer(sch, extractor, 3, 0.5, "missing"), dd, logsoft_model, i, pruning_method=pruning_method, rel_tol=rel_tol)
lime_s_mask = ExplainMill.explain(LimeExplainer(sch, extractor, 3, 0.5, "sample"), dd, logsoft_model, i, pruning_method=pruning_method, rel_tol=rel_tol)
open("lime_m.json", "w") do f
    write(f, JSON.json(ExplainMill.e2boolean(dd, mask, extractor)))
end
open("lime_s.json", "w") do f
    write(f, JSON.json(ExplainMill.e2boolean(dd, lime_s_mask, extractor)))
end
open("treelime.json", "w") do f
    write(f, JSON.json(ExplainMill.e2boolean(dd, treelime_mask, extractor)))
end
open("stochastic.json", "w") do f
    write(f, JSON.json(ExplainMill.e2boolean(dd, stochastic_mask, extractor)))
end
open("grad.json", "w") do f
    write(f, JSON.json(ExplainMill.e2boolean(dd, grad_mask, extractor)))
end
open("const.json", "w") do f
    write(f, JSON.json(ExplainMill.e2boolean(dd, const_mask, extractor)))
end


mean([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0] .== 1)
my_values = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]

my_values2 = [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]



mask2 = ExplainMill.create_mask_structure(ds[1], d -> SimpleMask(fill(true, d)))
fv = ExplainMill.FlatView(mask2)
for i in 1:length(my_values)
    fv[i] = my_values2[i]
end
result = ExplainMill.e2boolean(ds[1], mask2, extractor)
json_string = JSON.json(result)

print(json_string)

mi = fv.itemmap[6]
mask2
fv.masks[11]
size(fv.masks[6].x)
