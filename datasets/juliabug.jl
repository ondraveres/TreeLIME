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



###############################################################
# start by loading all samples
###############################################################

@time samples, labels, concepts = loaddata(settings);
labels = vcat(labels, fill(2, length(concepts)));
samples = vcat(samples, concepts);

resultsdir(s...) = joinpath("..", "..", "data", "sims", settings.dataset, settings.task, "$(settings.incarnation)", s...)
###############################################################
# create schema of the JSON
###############################################################
schema_file = resultsdir("schema.jdl2")
sch = nothing
if isfile(schema_file)
    @info "Schema file exists, loading from file"
    @time global sch = load(schema_file, "sch")
else
    @info "Schema file does not exist, creating new schema"
    global sch = JsonGrinder.schema(vcat(samples, concepts, Dict()))
    @save schema_file sch = sch
end
exdf = DataFrame()
for model_variant_k in [3, 4, 5, 6]
    extractor = suggestextractor(sch)

end
