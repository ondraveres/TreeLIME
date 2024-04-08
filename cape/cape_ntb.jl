try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/cape")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/cape")
end
using Pkg
Pkg.activate("..")

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

# num_samples = 10000
iterations = 10

THREADS = Threads.nthreads()

PATH_TO_REPORTS_LOCAL = "/Users/ondrejveres/Diplomka/ExplainMill.jl/data/Avast_cuckoo/"
PATH_TO_REPORTS = "/mnt/data/jsonlearning/Avast_cuckoo/"
PATH_TO_REDUCED_REPORTS_LOCAL = PATH_TO_REPORTS_LOCAL * "public_small_reports/"
PATH_TO_REDUCED_REPORTS = PATH_TO_REPORTS * "public_small_reports/"
PATH_TO_FULL_REPORTS = PATH_TO_REPORTS * "public_full_reports/"
PATH_TO_LABELS = "./";

df_labels = CSV.read(PATH_TO_REPORTS_LOCAL * "public_labels.csv", DataFrame);

# Group the data by classification_family
grouped = DataFrames.groupby(df_labels, :classification_family)

grouped

# Initialize an empty DataFrame to store the samples
train_samples = DataFrame()
test_samples = DataFrame()

# Select a thousand samples from each group
for sub_df in grouped
    my_df = sub_df[shuffle(1:end), :]
    train = my_df[1:500, :]
    test = my_df[501:600, :]

    append!(train_samples, train)
    append!(test_samples, test)
end

df_labels = vcat(train_samples, test_samples)

# df_labels
# df_labels = df_labels[1:min(nrow(df_labels), num_samples), :]
# df_labels

all_samples_count = size(df_labels, 1)
println("All samples: $(all_samples_count)")
println("Malware families: ")
[println(k => v) for (k, v) in countmap(df_labels.classification_family)];

# df_labels[!, :month] = map(i -> string(year(i), "-", month(i) < 10 ? "0$(month(i))" : month(i)), df_labels.date);
# month_counts = sort(countmap(df_labels.month) |> collect, by=x -> x[1])
# index2017 = findfirst(j -> j[1] == "2017-01", month_counts)
# previous_months = sum(map(j -> j[2], month_counts[1:index2017-1]))
# month_counts[index2017] = Pair("≤" * month_counts[index2017][1], month_counts[index2017][2] + previous_months)
# df_labels[!, :month] = map(i -> string(year(i), "-", month(i) < 10 ? "0$(month(i))" : month(i)), df_labels.date);
# month_counts = sort(countmap(df_labels.month) |> collect, by=x -> x[1])
# index2017 = findfirst(j -> j[1] == "2017-01", month_counts)
# previous_months = sum(map(j -> j[2], month_counts[1:index2017-1]))
# month_counts[index2017] = Pair("≤" * month_counts[index2017][1], month_counts[index2017][2] + previous_months)
# deleteat!(month_counts, 1:64)
# bar(getindex.(month_counts, 2), xticks=(1:length(month_counts), getindex.(month_counts, 1)), xtickfontsize=5, ytickfontsize=5, xrotation=45, yguidefontsize=8, xguidefontsize=8, legend=false,
#     xlabel="Month and year of the first evidence of a sample", ylabel="Number of samples for each month", size=(900, 400),
#     left_margin=5Plots.mm, bottom_margin=10Plots.mm)


timesplit = Date(2019, 8, 1)
train_indexes = 1:5000
test_indexes = [setdiff(Set(1:all_samples_count), Set(train_indexes))...];

# println("Malware families: ")
# [println(k => v) for (k, v) in countmap(df_labels.classification_family[train_indexes])];

train_size = length(train_indexes)
test_size = length(test_indexes)

println("Train size: $(train_size)")
println("Test size: $(test_size)")






jsons = tmap(df_labels.sha256) do s
    try
        open(JSON.parse, "$(PATH_TO_REDUCED_REPORTS_LOCAL)$(s).json")
    catch e
        @error "Error when processing sha $s: $e"
    end
end;
@assert size(jsons, 1) == all_samples_count # verifying that all samples loaded correctly

chunks = Iterators.partition(train_indexes, div(train_size, THREADS))
sch_parts = tmap(chunks) do ch
    JsonGrinder.schema(vcat(jsons[ch], Dict()))
end
time_split_complete_schema = merge(sch_parts...)
time_split_complete_schema
sch = JsonGrinder.schema(vcat(jsons, Dict()))

# printtree(time_split_complete_schema)
# import JsonGrinder: generate_html
# generate_html("recipes_max_vals=100.html", time_split_complete_schema, max_vals=100000)
extractor = suggestextractor(sch)
data = tmap(json -> extractor(json, store_input=true), jsons);
# data = tmap(json -> extractor(json), jsons);
# catobs(data)
function Mill.catobs(a::Any, b::Any)
    cat(a, b, dims=1)
end

labelnames = sort(unique(df_labels.classification_family))
neurons = 32
model = reflectinmodel(time_split_complete_schema, extractor,
    k -> Dense(k, neurons, relu),
    d -> SegmentedMeanMax(d),
    fsm=Dict("" => k -> Dense(k, length(labelnames))),
)

minibatchsize = 1
idx = sample(train_indexes, minibatchsize, replace=false)
idx_repeated = repeat(idx, inner=10)
function minibatch()
    idx = sample(train_indexes, minibatchsize, replace=false)
    dropout_data = []
    for i in idx
        push!(dropout_data, data[i])
        mk = ExplainMill.create_mask_structure(data[i], d -> ExplainMill.ParticipationTracker(ExplainMill.DafMask(d)))
        for j in 1:9
            sample!(mk, Weights([0.5, 0.5]))
            ExplainMill.updateparticipation!(mk)
            push!(dropout_data, data[i][mk])

        end
    end
    idx_repeated = repeat(idx, inner=10)
    reduce(catobs, dropout_data), Flux.onehotbatch(df_labels.classification_family[idx_repeated], labelnames)
end



function accuracy(x, y)
    vals = tmap(x) do s
        Flux.onecold(softmax(model(s)), labelnames)[1]
    end
    mean(vals .== y)
end


eval_trainset = shuffle(train_indexes)[1:50]
eval_testset = shuffle(test_indexes)[1:50]

cb = () -> begin
    train_acc = accuracy(data[eval_trainset], df_labels.classification_family[eval_trainset])
    test_acc = accuracy(data[eval_testset], df_labels.classification_family[eval_testset])
    println("accuracy: train = $train_acc, test = $test_acc")
end
ps = Flux.params(model)
loss = (x, y) -> Flux.logitcrossentropy(model(x), y)
opt = ADAM()

# data_loader = Flux.DataLoader((data[train_indexes], Flux.onehotbatch(df_labels.classification_family[train_indexes], labelnames)), batchsize=100, shuffle=true)


Flux.Optimise.train!(loss, ps, repeatedly(minibatch, iterations), opt, cb=Flux.throttle(cb, 2))
# Flux.Optimise.train!(loss, ps, data_loader, opt, cb=Flux.throttle(cb, 2))

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

# @printf "%8s\t" "TL\\PL"
# [@printf " %8s" s for s in labelnames]
# print("\n")
# for tl in labelnames
#     @printf "%8s\t" tl
#     for pl in labelnames
#         @printf "%9s" @sprintf "%.2f" test_predictions[tl][pl] * 100
#     end
#     print("\n")
# end

labelnames
df_labels
time_split_complete_schema
extractor

data

model
@save "cape_equal_drop_extractor.jld2" extractor sch data model
@save "cape_model_variables_equal_small.jld2" labelnames df_labels time_split_complete_schema extractor data model
# using Plots
# predictions = Flux.onecold(softmax(model(data)))
# histogram(predictions, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency")


# @save "extractedCapeData.jld2" data
# @save "capeExtractor.jld2" extractor

# @load "extractedCapeData.jld2" data
# @load "capeExtractor.jld2" extractor