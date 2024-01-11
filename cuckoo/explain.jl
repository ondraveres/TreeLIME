using Pkg
Pkg.activate("..")
using Flux, Mill, JsonGrinder, ArgParse, Setfield, Serialization, ExplainMill, DataFrames, BSON
using Mill: nobs

include("common.jl")

_s = ArgParseSettings()
@add_arg_table! _s begin
  ("--name"; default="banz";arg_type=String);
  ("--pruning_method"; default="LbyL_HArr";arg_type=String);
  ("-i"; default=4;arg_type=Int);
end

settings = parse_args(ARGS, _s; as_symbols=true)
settings = NamedTuple{Tuple(keys(settings))}(values(settings))
modeldir(s...) = joinpath("../../data/sims/cuckoo/", s...)

extractor = deserialize(modeldir("extractor.jls"));
smodel = deserialize(modeldir("model.jls"));
model = smodel.m;

model = f32(@set model.m.m = Chain(model.m.m, smodel.severity));
statlayer = StatsLayer()
model = @set model.m.m = Chain(model.m.m...,  statlayer);
soft_model = @set model.m.m = Chain(model.m.m...,  softmax);
logsoft_model = @set model.m.m = Chain(model.m.m...,  logsoftmax);

name, pruning_method = settings[:name], Symbol(settings[:pruning_method])
e, n = getexplainer(name)

samplesizes = BSON.load("samplesizes.bson")[:samplesizes]
sort!(samplesizes, :length)

####
# Let's calculation of statistics
###
r = samplesizes[settings[:i], :]
ds = ff32(deserialize(modeldir(r.filename))[r.j]);
ExplainMill.stats(e, ds, model, 1, 1);

r = samplesizes[settings[:i], :]
ds = ff32(deserialize(modeldir(r.filename))[r.j]);
i = Int(startswith(r.filename, "extracted_mal")) + 1
gap = ExplainMill.confidencegap(soft_model, ds, i)
exdf = addexperiment(DataFrame(), e, ds, logsoft_model, i, n, 0.9*gap, name, pruning_method, merge(settings, (sampleno = r.j, filename = r.filename)), statlayer)
BSON.@save "../../data/sims/cuckoo/stats/$(name)_$(pruning_method)_$(settings[:i]).bson" exdf
