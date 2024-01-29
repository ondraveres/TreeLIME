# for ((i=1;i<=20;i+=1)); do  for d in  one_of_1_2trees  one_of_1_5trees  one_of_1_paths  one_of_2_5trees  one_of_2_paths  one_of_5_paths ; do  julia -p 24 artificial.jl --dataset $d --incarnation $i ; done ; done
using Revise
using Pkg;
cd("/home/veresond/ExplainMill.jl/myscripts/datasets");
Pkg.activate("..");
using ArgParse;
using Flux;
using Mill;
using JsonGrinder;
using JSON;
using BSON;
using Statistics;
using IterTools;
using PrayTools;
using StatsBase;
using ExplainMill;
using Serialization;
using Setfield;
using DataFrames;
using ExplainMill: jsondiff, nnodes, nleaves;
include("common.jl");
include("loader.jl");
include("stats.jl");
using PrintTypesTersely;
function StatsBase.predict(mymodel::Mill.AbstractMillModel, ds::Mill.AbstractMillNode, ikeyvalmap)
    o = mapslices(x -> ikeyvalmap[argmax(x)], mymodel(ds), dims=1)
end;
PrintTypesTersely.off();
_s = ArgParseSettings();
@add_arg_table! _s begin
    ("--dataset"; default = "mutagenesis"; arg_type = String)
    ("--task"; default = "one_of_1_5trees"; arg_type = String)
    ("--incarnation"; default = 1; arg_type = Int)
    ("-k"; default = 5; arg_type = Int)
end
;
settings = parse_args(ARGS, _s; as_symbols=true);



settings = NamedTuple{Tuple(keys(settings))}(values(settings));

model_name = "hundreditermodel22_1.bson"

###############################################################
# start by loading all samples
###############################################################
;
samples, labels, concepts = loaddata(settings);
loaddata(settings)[3];
concepts;
labels = vcat(labels, fill(2, length(concepts)));
samples = vcat(samples, concepts);

resultsdir(s...) = joinpath("..", "..", "data", "sims", settings.dataset, settings.task, "$(settings.incarnation)", s...);
println("start");
println("resultsdir() = ", resultsdir());
###############################################################
# create schema of the JSON
###############################################################
sch = JsonGrinder.schema(vcat(samples, concepts, Dict()));
if !isfile(resultsdir(model_name))
    !isdir(resultsdir()) && mkpath(resultsdir())
    sch = JsonGrinder.schema(vcat(samples, concepts, Dict()))
    extractor = suggestextractor(sch)

    trndata = extractbatch(extractor, samples)
    function makebatch()
        i = rand(1:2000, 100)
        trndata[i], Flux.onehotbatch(labels[i], 1:2)
    end
    ds = extractor(JsonGrinder.sample_synthetic(sch))
    good_model, concept_gap = nothing, 0
    # good_model, concept_gap
    local model = reflectinmodel(
        sch,
        extractor,
        d -> Dense(d, settings.k, relu),
        all_imputing=true,
        # b = Dict("" => d -> Chain(Dense(d, settings.k, relu), Dense(settings.k, 2)))
    )
    model = @set model.m = Chain(model.m, Dense(settings.k, 2))
    for i in 1:2
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

        if (acc > 0.95)
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
    #model = good_model
    BSON.@save resultsdir(model_name) model extractor sch
end;


resultsdir();
using Flux;
isfile(resultsdir(model_name));
d = BSON.load(resultsdir(model_name));




(model, extractor, sch) = d[:model], d[:extractor], d[:sch];
statlayer = StatsLayer()
;
model = @set model.m = Chain(model.m, statlayer);
soft_model = @set model.m = Chain(model.m, softmax);
logsoft_model = @set model.m = Chain(model.m, logsoftmax);


###############################################################
#  Helper functions for explainability
###############################################################
const ci = PrayTools.classindexes(labels)
ci

ci

function loadclass(k, n=typemax(Int))
    dss = map(s -> extractor(s, store_input=true), sample(samples[ci[k]], min(n, length(ci[k])), replace=false))
    reduce(catobs, dss)
end


function onlycorrect(dss, i, min_confidence=0)
    correct = predict(soft_model, dss, [1, 2]) .== i
    dss = dss[correct[:]]
    min_confidence == 0 && return (dss)
    correct = ExplainMill.confidencegap(soft_model, dss, i) .>= min_confidence
    dss[correct[:]]
end

Random.seed!(settings.incarnation)
strain = 2
ds = loadclass(strain, 1000)
if false

    extractor(samples[10111], store_input=true).metadata
    i = strain
    concept_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i)[1, 1], concepts))
    sample_gap = minimum(map(c -> ExplainMill.confidencegap(soft_model, extractor(c), i)[1, 1], samples[labels.==2]))
    threshold_gap = 0.2#floor(0.9 * concept_gap, digits=2)
    # correct = predict(soft_model, ds, [1, 2])
    # argmax(soft_model(ds[1]))
    # soft_model(ds)
    # mean(Flux.onecold(model(ds) .== labels))
    # mean(labels)
    # mean(Flux.onecold(soft_model(ds)) .== labels)
    # mean(correct)


    # ExplainMill.confidencegap(soft_model, ds, 2)
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

    PrintTypesTersely.on()

    ExplainMill.DafExplainer()
    exdf = DataFrame()
    numobs(ds)
    variants
    if false
        if !isfile(resultsdir("stats_" * model_name))
            for (name, pruning_method) in variants
                e = getexplainer(name)
                @info "explainer $e on $name with $pruning_method"
                flush(stdout)
                #addexperiment(DataFrame(), e, ds[1], logsoft_model, i, n, threshold_gap, name, pruning_method, 1, settings, statlayer)
                for j in 1:numobs(ds)
                    global exdf
                    exdf = addexperiment(exdf, e, ds[j], logsoft_model, 2, 0, 0.1, name, pruning_method, j, settings, statlayer)
                end
                BSON.@save resultsdir("stats_" * model_name) exdf
            end
        end
    end
end


using HierarchicalUtils

### the players
PrintTypesTersely.off()

mysample = ds[1]


mask = ExplainMill.create_mask_structure(mysample, d -> SimpleMask(fill(true, d)))
mysample_copy = deepcopy(mysample)
mask_copy = deepcopy(mask)
sch_copy = deepcopy(sch)
extractor_copy = deepcopy(extractor)

children(mysample_copy)
children(mysample)
children(mask)
mysample_copy[:atoms].bags
mysample_copy[:atoms] isa ProductNode

function my_recursion(data_node, mask_node, extractor_node, schema_node)
    if data_node isa ProductNode
        children_names = []
        modified_data_ch_nodes = []
        modified_mask_ch_nodes = []
        for (
            (data_ch_name, data_ch_node),
            (mask_ch_name, mask_ch_node)
        ) in zip(
            pairs(children(data_node)),
            pairs(children(mask_node))
        )
            push!(children_names, data_ch_name)
            (modified_child_data, modified_child_mask) = my_recursion(data_ch_node, mask_ch_node, extractor_node[data_ch_name], schema_node[data_ch_name])
            push!(modified_data_ch_nodes, modified_child_data)
            push!(modified_mask_ch_nodes, modified_child_mask)
        end
        nt_data = NamedTuple{Tuple(children_names)}(modified_data_ch_nodes)
        nt_mask = NamedTuple{Tuple(children_names)}(modified_mask_ch_nodes)

        return ProductNode(nt_data), ExplainMill.ProductMask(nt_mask)
    end
    if data_node isa BagNode
        child_node = Mill.data(data_node)
        (modified_data_child_node, modified_child_mask) = my_recursion(child_node, mask_node.child, extractor_node.item, schema_node.items)

        return BagNode(modified_data_child_node, data_node.bags, data_node.metadata), ExplainMill.BagMask(modified_child_mask, mask_node.bags, mask_node.mask)
    end
    if data_node isa ArrayNode
        total = sum(values(schema_node.counts))
        normalized_probs = [v / total for v in values(schema_node.counts)]
        n = length(normalized_probs)  # Get the number of elements
        w = Weights(ones(n))
        vals = collect(keys(schema_node.counts))
        @info collect(keys(schema_node.counts))
        if extractor_node isa ExtractCategorical
            @info data_node.data
            @info numobs(data_node)

            new_hot_vectors = []
            new_random_keys = []
            for i in 1:numobs(data_node)
                original_hot_vector = data_node.data[:, i]
                random_key = sample(vals, w)
                extracted_random_key = extractor_node.keyvalemap[random_key]
                new_hot_vector = MaybeHotVector(extracted_random_key, extractor_node.n)
                while original_hot_vector == new_hot_vector
                    global random_key = sample(vals, w)
                    extracted_random_key = extractor_node.keyvalemap[random_key]
                    new_hot_vector = MaybeHotMatrix(MaybeHotVector(extracted_random_key, extractor_node.n))
                    @info "while"
                    @info original_hot_vector new_hot_vector
                end
                push!(new_random_keys, random_key)
                push!(new_hot_vectors, new_hot_vector)
                @info "appending"
            end
            maybe_hot_matrix = hcat(new_hot_vectors...)
            new_array_node = ArrayNode(maybe_hot_matrix, data_node.metadata)
            @info "numobs origo" numobs(data_node)
            @info "numobs" numobs(new_array_node)
            @info "new data" new_array_node.data
            return new_array_node, mask_node
        end
        return ArrayNode(Mill.data(data_node), data_node.metadata), mask_node
    end
end
(s, m) = my_recursion(mysample_copy, mask_copy, extractor_copy, sch_copy)



a = mask_copy[:lumo]
mysample_copy[:lumo]
mask_copy[:atoms].mask
s == mysample_copy
mysample_copy[:lumo]
s[:lumo]
m == mask_copy
s


e = ExtractDict(Dict(:a => ExtractScalar(Float32, 2, 3),
    :b => ExtractCategorical(1:5)))
sc = ExtractArray(e([Dict("a" => 1, "b" => 1), Dict("a" => 1, "b" => 1), Dict("a" => 1, "b" => 1)]))
typeof(sc)
mysample_copy == my_recursion(mysample_copy)
mask_copy
N = 1  # Number of copies
copies = Array{Tuple{typeof(mysample),typeof(mask)},1}(undef, N)
my_data_node = nothing
my_mask_node = nothing
my_extractor_node = nothing
for i in 1:N

    mysample_copy = deepcopy(mysample)
    mask_copy = deepcopy(mask)
    sch_copy = deepcopy(sch)
    extractor_copy = deepcopy(extractor)

    leafmap!(mysample_copy, mask_copy, sch_copy, extractor_copy; complete=false, order=LevelOrder()) do (data_node, mask_node, schema_node, extractor_node)
        total = sum(values(schema_node.counts))
        normalized_probs = [v / total for v in values(schema_node.counts)]
        n = length(normalized_probs)  # Get the number of elements
        w = Weights(ones(n))
        #w = Weights(normalized_probs)
        vals = collect(keys(schema_node.counts))

        if extractor_node isa ExtractCategorical
            for i in 1:length(mask_node.mask.x)
                if rand() > 0.5
                    extracted_random_key = extractor_node.keyvalemap[sample(vals, w)]
                    mask_node.mask.x[i] = true
                    original_hot_vector = data_node.data[:, i]
                    new_hot_vector = MaybeHotVector(extracted_random_key, extractor_node.n)
                    while original_hot_vector == new_hot_vector
                        extracted_random_key = extractor_node.keyvalemap[sample(vals, w)]
                        new_hot_vector = MaybeHotVector(extracted_random_key, extractor_node.n)
                    end
                    new_hot_matrix = MaybeHotMatrix(new_hot_vector)
                    new_data = hcat(data_node.data[:, 1:i-1], new_hot_matrix, data_node.data[:, i+1:end])
                    data_node.data = new_data
                end
            end
            global my_data_node = data_node
            global my_mask_node = mask_node
            global my_extractor_node = extractor_node

        elseif extractor_node isa ExtractScalar

            if rand() > 0.0
                extracted_random_key = extractor_node(sample(vals, w))
                @info "start"
                @info extracted_random_key
                @info extractor_node
                @info data_node.data
                @info mask_node
                @info mask_node.cols
                @info mask_node.rows
                @info length(mask_node.mask.x)
                # data_node.data = extracted_random_key.data
                # mask_node.mask.m.x[1] = true

            end
        else
            @error "unknown extractor type"
        end
    end
    copies[i] = (mysample_copy, mask_copy)
end


my_mask.mask
my_sch
mask_labels = []
model(mysample_copy)
mysample
copies[1][1]
for i in 1:N
    push!(mask_labels, argmax(model(copies[i][1]))[1])
end
mean(mask_labels)

flat_view = ExplainMill.FlatView(mask)
new_flat_view = ExplainMill.FlatView(mask_copy)

mask_bool_vector = [flat_view[i] for i in 1:length(flat_view.itemmap)]
new_mask_bool_vector = [new_flat_view[i] for i in 1:length(new_flat_view.itemmap)]

mean(mask_bool_vector)
mean(new_mask_bool_vector)


model(mysample)
model(mysample_copy)

mysample[:lumo]
mysample_copy[:lumo]

mysample_copy
mysample

global_mask_node.mask.m.x




# ret = treemap(mysample_copy, mask_copy, extractor_copy, sch_copy; complete=false) do n, ch
#     (data_node, mask_node, extractor_node, schema_node) = n
#     if (isleaf(data_node))
#         data_node.data
#         data_node.metadata
#         schema_node.counts

#         if extractor_node isa ExtractCategorical


#             total = sum(values(schema_node.counts))
#             normalized_probs = [v / total for v in values(schema_node.counts)]
#             w = Weights(normalized_probs)
#             vals = collect(keys(schema_node.counts))

#             new_data = []
#             for i in 1:length(mask_node.mask.m.x)
#                 if rand() < 0.0
#                     random_key = sample(vals, w)
#                     extracted_random_key = extractor_node.keyvalemap[random_key]
#                     mask_node.mask.m.x[i] = true
#                     new_hot_vector = MaybeHotVector(extracted_random_key, extractor_node.n)
#                     new_hot_matrix = MaybeHotMatrix(new_hot_vector)
#                     push!(new_data, new_hot_matrix)
#                 else
#                     push!(new_data, data_node.data[:, i])
#                 end
#             end
#             concatenated_data = hcat(new_data...)
#             new_array_node = ArrayNode(concatenated_data, data_node.metadata)
#             @info "origo data" data_node.data
#             @info "změněný data" new_array_node.data
#             return (new_array_node, mask_node, extractor_node, schema_node)
#         elseif extractor_node isa ExtractScalar
#             @info "scalar"
#         else
#             @error "unknown extractor type"
#         end
#     end
#     return (data_node, mask_node, extractor_node, schema_node)
# end
