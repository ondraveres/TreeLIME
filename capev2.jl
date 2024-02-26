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

THREADS = Threads.nthreads()

PATH_TO_REPORTS = "/Users/ondrejveres/Diplomka/ExplainMill.jl/data/Avast_cuckoo/"
PATH_TO_REDUCED_REPORTS = PATH_TO_REPORTS * "public_small_reports/"
PATH_TO_FULL_REPORTS = PATH_TO_REPORTS * "public_full_reports/"
PATH_TO_LABELS = "./";

df_labels = CSV.read(PATH_TO_REPORTS * "public_labels.csv", DataFrame);
df_labels = df_labels[1:min(nrow(df_labels), 5000), :]
df_labels
all_samples_count = size(df_labels, 1)
println("All samples: $(all_samples_count)")
println("Malware families: ")
[println(k => v) for (k, v) in countmap(df_labels.classification_family)];

df_labels[!, :month] = map(i -> string(year(i), "-", month(i) < 10 ? "0$(month(i))" : month(i)), df_labels.date);
month_counts = sort(countmap(df_labels.month) |> collect, by=x -> x[1])
index2017 = findfirst(j -> j[1] == "2017-01", month_counts)
previous_months = sum(map(j -> j[2], month_counts[1:index2017-1]))
month_counts[index2017] = Pair("≤" * month_counts[index2017][1], month_counts[index2017][2] + previous_months)
deleteat!(month_counts, 1:64)
bar(getindex.(month_counts, 2), xticks=(1:length(month_counts), getindex.(month_counts, 1)), xtickfontsize=5, ytickfontsize=5, xrotation=45, yguidefontsize=8, xguidefontsize=8, legend=false,
    xlabel="Month and year of the first evidence of a sample", ylabel="Number of samples for each month", size=(900, 400),
    left_margin=5Plots.mm, bottom_margin=10Plots.mm)


timesplit = Date(2019, 8, 1)
train_indexes = findall(i -> df_labels.date[i] < timesplit, 1:all_samples_count)
test_indexes = [setdiff(Set(1:all_samples_count), Set(train_indexes))...];

train_size = length(train_indexes)
test_size = length(test_indexes)

println("Train size: $(train_size)")
println("Test size: $(test_size)")


jsons = tmap(df_labels.sha256) do s
    try
        open(JSON.parse, "$(PATH_TO_REDUCED_REPORTS)$(s).json")
    catch e
        @error "Error when processing sha $s: $e"
    end
end;
@assert size(jsons, 1) == all_samples_count # verifying that all samples loaded correctly

chunks = Iterators.partition(train_indexes, div(train_size, THREADS))
sch_parts = tmap(chunks) do ch
    JsonGrinder.schema(jsons[ch])
end
time_split_complete_schema = merge(sch_parts...)
printtree(time_split_complete_schema)

extractor = suggestextractor(time_split_complete_schema)
data = tmap(extractor, jsons);
@save "extractedCapeData.jld2" data

@load "extractedCapeData.jld2" data

labelnames = sort(unique(df_labels.classification_family))
neurons = 32
model = reflectinmodel(time_split_complete_schema, extractor,
    k -> Dense(k, neurons, relu),
    d -> SegmentedMeanMax(d),
    fsm=Dict("" => k -> Dense(k, length(labelnames))),
)

minibatchsize = 500
function minibatch()
    idx = sample(train_indexes, minibatchsize, replace=false)
    reduce(catobs, data[idx]), Flux.onehotbatch(df_labels.classification_family[idx], labelnames)
end

iterations = 30

function accuracy(x, y)
    vals = tmap(x) do s
        Flux.onecold(softmax(model(s)), labelnames)[1]
    end
    mean(vals .== y)
end


eval_trainset = shuffle(train_indexes)[1:1000]
eval_testset = shuffle(test_indexes)[1:1000]

cb = () -> begin
    train_acc = accuracy(data[eval_trainset], df_labels.classification_family[eval_trainset])
    test_acc = accuracy(data[eval_testset], df_labels.classification_family[eval_testset])
    println("accuracy: train = $train_acc, test = $test_acc")
end
ps = Flux.params(model)
loss = (x, y) -> Flux.logitcrossentropy(model(x), y)
opt = ADAM()

Flux.Optimise.train!(loss, ps, repeatedly(minibatch, iterations), opt, cb=Flux.throttle(cb, 2))

full_test_accuracy = accuracy(data[test_indexes], df_labels.classification_family[test_indexes])
println("Final evaluation:")
println("Accuratcy on test data: $(full_test_accuracy)")

test_predictions = Dict()
for true_label in labelnames
    current_predictions = Dict()
    [current_predictions[pl] = 0.0 for pl in labelnames]
    family_indexes = filter(i -> df_labels.classification_family[i] == true_label, test_indexes)
    predictions = tmap(data[family_indexes]) do s
        Flux.onecold(softmax(model(s)), labelnames)[1]
    end
    [current_predictions[pl] += 1.0 for pl in predictions]
    [current_predictions[pl] = current_predictions[pl] ./ length(predictions) for pl in labelnames]
    test_predictions[true_label] = current_predictions
end

@printf "%8s\t" "TL\\PL"
[@printf " %8s" s for s in labelnames]
print("\n")
for tl in labelnames
    @printf "%8s\t" tl
    for pl in labelnames
        @printf "%9s" @sprintf "%.2f" test_predictions[tl][pl] * 100
    end
    print("\n")
end