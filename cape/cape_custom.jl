using Pkg
using Flux, MLDataPattern, Mill, JsonGrinder, JSON, Statistics, IterTools, StatsBase, ThreadTools
using JsonGrinder: suggestextractor, ExtractDict
using Mill: reflectinmodel
using CSV, DataFrames
using Random
using Dates
using Plots
using Printf
using JLD2
using ExplainMill

iterations = 40

THREADS = Threads.nthreads()

PATH_TO_REPORTS = "/mnt/data/jsonlearning/Avast_cuckoo/"
PATH_TO_REDUCED_REPORTS = PATH_TO_REPORTS * "public_small_reports/"
PATH_TO_FULL_REPORTS = PATH_TO_REPORTS * "public_full_reports/"
PATH_TO_LABELS = "./";

df_labels = CSV.read(PATH_TO_REPORTS * "public_labels.csv", DataFrame);

grouped = DataFrames.groupby(df_labels, :classification_family)

train_samples = DataFrame()
test_samples = DataFrame()

for sub_df in grouped
    my_df = sub_df[shuffle(1:end), :]
    train = my_df[1:500, :]
    test = my_df[501:600, :]

    append!(train_samples, train)
    append!(test_samples, test)
end

df_labels = vcat(train_samples, test_samples)

jsons = tmap(df_labels.sha256) do s
    try
        open(JSON.parse, "$(PATH_TO_REDUCED_REPORTS)$(s).json")
    catch e
        @error "Error when processing sha $s: $e"
    end
end;

sch = JsonGrinder.schema(vcat(jsons, Dict()))

extractor = suggestextractor(sch)
data = tmap(json -> extractor(json, store_input=true), jsons);


labelnames = sort(unique(df_labels.classification_family))
neurons = 32
model = reflectinmodel(sch, extractor,
    k -> Dense(k, neurons, relu),
    all_imputing=true
)
model = @set model.m = Chain(model.m, Dense(neurons, length(labelnames)))

model(data[2])

stochastic_mask = ExplainMill.explain(GradExplainer(), data[1], model, pruning_method=:Flat_HAdd, rel_tol=0.1)
