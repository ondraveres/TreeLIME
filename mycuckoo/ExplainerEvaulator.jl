using JsonGrinder, Mill, Flux, MLDatasets, Statistics, Random, JSON3, OneHotArrays, JSON, Accessors, JLD2, PrintTypesTersely, CSV, BSON
PrintTypesTersely.off()
using DataFrames
using PrettyTables
using Pkg
using ExplainMill
using Cuckoo


abstract type AbstractProblem end
abstract type MutagenesisProblem <: AbstractProblem end
abstract type RecipesProblem <: AbstractProblem end
abstract type CuckooProblem <: AbstractProblem end

struct DoesRecipeContainSaltAndOil <: RecipesProblem end

struct DoesItContainMoreThan14Carbons <: MutagenesisProblem end
struct IsLumoLargerThanConst <: MutagenesisProblem end
struct IsMalware <: CuckooProblem end

problems = [
    # DoesItContainMoreThan14Carbons(),
    # DoesRecipeContainSaltAndOil(),
    # IsLumoLargerThanConst(),
    IsMalware(),
]


explainers = [
    ConstExplainer(100),
    # StochasticExplainer(),
    # GnnExplainer(2),
    # GradExplainer(),
    # DafExplainer(),
]
string(explainers[1])
cuckoo = Dataset("cuckoo", full=false)
Cuckoo.load_samples(cuckoo, inds=1:3)


function convert_keys_to_symbols(dict)
    if typeof(dict) <: Dict
        return Dict(Symbol(k) => convert_keys_to_symbols(v) for (k, v) in dict)
    elseif typeof(dict) <: Array
        return [convert_keys_to_symbols(v) for v in dict]
    else
        return dict
    end
end

function load_train_x_data(::RecipesProblem)
    data_file = "data/recipes_enhanced_x_only.json"
    samples = open(data_file, "r") do fid
        (JSON.parse(read(fid, String)))
    end
    return map(sample -> Dict(Symbol(k) => v for (k, v) in sample), samples[1:3500])
end



function load_test_x_data(::CuckooProblem)
    inds = 4:6
    return load_cuckoo_data(inds)

end

function load_cuckoo_data(inds)
    files = cuckoo.samples[inds]
    n = length(files)
    #dicts = Vector{Union{ProductNode,Missing}}(missing, n)
    dicts = [Dict() for i in 1:n]
    read_json(file) = JSON.parse(read(file, String))

    if typeof(inds) == Colon || length(inds) > 32
        p = Progress(n; desc="Extracting JSONs: ")
        Threads.@threads for i in 1:n
            # in case the loading errors, returns only the samples it does not error on
            try
                dicts[i] = d.extractor(read_json(files[i]))
            catch e
                @info "Error in filepath = $(files[i])"
                @warn e
            end
            next!(p)
        end
    else
        for i in 1:n
            # try
            println("this works")
            dicts[i] = read_json(files[i])
            #println(dicts[i])
            # catch e
            #     @info "Error in filepath = $(files[i])"
            #     @error e
            # end
        end
    end
    return dicts
end

function load_train_x_data(::CuckooProblem)
    inds = 1:3
    load_cuckoo_data(inds)

end

function load_test_x_data(::RecipesProblem)
    data_file = "data/recipes_enhanced_x_only.json"
    samples = open(data_file, "r") do fid
        (JSON.parse(read(fid, String)))
    end
    return map(sample -> Dict(Symbol(k) => v for (k, v) in sample), samples[3501:end])
end

function load_train_x_data(::MutagenesisProblem)
    base = MLDatasets.Mutagenesis(split=:train).features
    return base
end
load_test_x_data(::MutagenesisProblem) = MLDatasets.Mutagenesis(split=:test).features;
MLDatasets.Mutagenesis(split=:test).features;
function condition(::DoesRecipeContainSaltAndOil, dict)
    sampleIngredients = dict[:ingredients]
    requiredIngredients = ["salt", "oil"]
    for requiredIngredient in requiredIngredients
        if !any(s -> occursin(requiredIngredient, s), sampleIngredients)
            return false
        end
    end
    return true
end

function condition(::DoesItContainMoreThan14Carbons, dict)
    atoms = dict[:atoms]
    carbon_count = count(atom -> atom[:element] == "c", atoms)
    return carbon_count > 14
end

function condition(::IsLumoLargerThanConst, dict)
    lumo = dict[:lumo]
    return lumo > -1.0
end

function condition(::IsMalware, dict)
    return true
end

generate_y(problem::AbstractProblem, x_train) = map(x -> condition(problem, x) ? 1 : 0, x_train)

function get_model(::MutagenesisProblem, sch, extractor)
    model = reflectinmodel(sch, extractor, d -> Dense(d, 10, relu), all_imputing=true)
    model = @set model.m = Chain(model.m, Dense(10, 2))
    return model
end

function get_model(::RecipesProblem, sch, extractor)
    model = reflectinmodel(sch, extractor,
        input_dim -> Dense(input_dim, 20, relu),
        input_dim -> SegmentedMeanMax(input_dim),
        fsm=Dict("" => input_dim -> Dense(input_dim, 2)), all_imputing=true,
    )
    return model
end

function get_model(::CuckooProblem, sch, extractor)
    model = reflectinmodel(sch, extractor, d -> Dense(d, 10, relu), all_imputing=true)
    model = @set model.m = Chain(model.m, Dense(10, 2))
    return model
end

problem = IsMalware()
#for problem in problems
extracted_train = nothing
extractor = nothing
sch = nothing

x_train = load_train_x_data(problem)

x_test = load_test_x_data(problem)
x_train

y_test = generate_y(problem, x_test)
sch = JsonGrinder.schema(x_train)
x_train
extractor = suggestextractor(sch)
y_train = generate_y(problem, x_train)
y_test = generate_y(problem, x_test)
n = length(x_train)

extracted_train = Vector{Union{ProductNode,Missing}}(missing, n)

extracted_train = extractor.(x_train, store_input=true)




model = get_model(problem, sch, extractor)
problem_name = string(typeof(problem))
model_name = problem_name * ".jld2"
model_state = nothing

Flux.onecold.(model(extracted_train))

function Mill.catobs(a::Nothing, b::Any)
    return b
end

function Mill.catobs(a::Any, b::Nothing)
    return a
end

# try
#     model_state = JLD2.load(model_name, "model_state")
#     Flux.loadmodel!(model, model_state)
# catch e

catobs
red = nothing

if sum(ismissing.(extracted_train)) > 0
    mask = .!ismissing.(extracted_train)
    red = reduce(catobs, extracted_train[mask|>BitVector]), mask
else
    red = reduce(catobs, extracted_train)
end

extracted_train
reduced = reduce(catobs, extracted_train)

typeof(extracted_train[2])

typeof(extracted_train[1])

catobs(extracted_train[1], extracted_train[2])
loss(ds, y) = Flux.Losses.logitbinarycrossentropy(Flux.onecold(model(ds)), y)
accuracy(ds, y) = mean(Flux.onecold.(model.(ds)) .== y .+ 1)

opt = AdaBelief()
ps = Flux.params(model)

Flux.onecold(model(extracted_train[1]))
OneHotArrays.onehotbatch(y_train .+ 1, 1:2)
y_train
model.(extracted_train)

# Lastly we turn our training data to minibatches, and we can start training:
data_loader = Flux.DataLoader((extracted_train, y_train), batchsize=2, shuffle=true)
Flux.Optimise.train!(loss, ps, data_loader, opt)
@show accuracy(extracted_train, y_train)

# We can see the accuracy rising and obtaining over 80% quite quickly:
i = 1
while accuracy(extracted_train, y_train) < 0.999
    @info "Epoch 1"
    Flux.Optimise.train!(loss, ps, data_loader, opt)
    @show accuracy(extracted_train, y_train)
    i += 1
end

# save the model
model_state = Flux.state(model)
jldsave(model_name; model_state)
# end


## Explain the the positive class
positives = x_test[y_test.==1]

print("positives: ", length(positives), "\n")
print("x_test: ", length(x_test), "\n")

extracted_positives = extractor.(positives, store_input=true)

result = DataFrame(Explainer=String[], ExplanationsMatchCondition=Float64[], ExplanationsClassifiedCorrectly=Float64[])

for explainer in explainers
    println("Explainer: ", explainer)
    masks = map(d -> explain(explainer, d, model; rel_tol=0.9), extracted_positives)
    explanations = map((ds, mk) -> ExplainMill.yarason(ds, mk, extractor)[1], extracted_positives, masks)
    explanations_y = generate_y(problem, explanations)
    correctly_explained = explanations[explanations_y.==1]
    wrongly_explained = explanations[explanations_y.==0]

    acc = length(correctly_explained) / length(extracted_positives)

    # model.(extractor.(explanations))

    masked_data = map((d, m) -> d[m], extracted_positives, masks)

    classified_correctly = (mean(Flux.onecold.(model.(masked_data)))[1] - 1)

    # println(problem_name * " Explanations Match Condition: ", acc)
    # println(problem_name * " Explanations Classified Correctly By The Model: ", classified_correctly) ## minus one because 2 is the positive class

    push!(result, [string(explainer), acc, classified_correctly])

    open(problem_name * "_wrong.json", "w") do f
        JSON.print(f, wrongly_explained)
    end
    open(problem_name * "_correct.json", "w") do f
        JSON.print(f, correctly_explained)
    end
    open(problem_name * "_explanations.json", "w") do f
        JSON.print(f, explanations)
    end
end
pretty_table(result, title=problem_name)

#end

