using Revise
using Pkg
Pkg.activate(".")
using Flux, Mill, JsonGrinder, Setfield, Serialization, ExplainMill, DataFrames
using Mill: nobs

include("../public_cuckoo/common.jl")

modeldir(s...) = joinpath("/home/viliam.lisy/rcc_v2", s...)

extractor = deserialize(modeldir("extractor.jls"));
smodel = deserialize(modeldir("train","model2.jls"));
model = smodel.m;

model = f32(@set model.m.m = Chain(model.m.m, smodel.severity));
statlayer = StatsLayer()
model = @set model.m.m = Chain(model.m.m...,  statlayer);
soft_model = @set model.m.m = Chain(model.m.m...,  softmax);
logsoft_model = @set model.m.m = Chain(model.m.m...,  logsoftmax);

samplesizes = mapreduce(vcat, ["extracted_ben","extracted_mal"]) do d
	mapreduce(vcat, readdir(modeldir(d))) do f
		ds = deserialize(modeldir(d, f));
		os = map(1:nobs(ds)) do j 
			dd = ds[j]
			gap = ExplainMill.confidencegap(soft_model, dd, 2)
			ms = ExplainMill.stats(ExplainMill.ConstExplainer(), dd, model, 0, 0);
			l = length(ExplainMill.FlatView(ms));
			(filename = joinpath(d,f), j = j, length = l, gap = gap)
		end
		DataFrame(os)
	end
end
samplesizes = DataFrame(samplesizes)
BSON.@save "../reddir_v2/samplesizes.bson" samplesizes

gaps = ExplainMill.confidencegap(soft_model, malware, 2)[:]
ds = malware[gaps .> 0.9]

exdf = DataFrame()
sampleno = 1


logical1 = map(1:nobs(ds)) do j 
	dd = ff32(ds[j])
	gap = ExplainMill.confidencegap(soft_model, dd, 2)
	e = ExplainMill.GnnExplainer()
	# ms1 = ExplainMill.stats(e, dd, model, 2, 200);
	# f = () -> ExplainMill.confidencegap(soft_model, prune(dd, ms1), 2) - 0.9*gap
	# ExplainMill.greedy!(f, ms1, x -> ExplainMill.scorefun(e, x))

	t1 = @elapsed ms1 = ExplainMill.explain(e, dd, logsoft_model, 2, 200, pruning_method = :greedy, threshold = 0.9*gap)
	exdf = addexperiment(exdf, e, ds[j], logsoft_model, 2, 200, 0.9*gap, name, pruning_method, sampleno, statlayer)
	logical = ExplainMill.e2boolean(ms1, dd, extractor)
	open("/tmp/gnn_$(j).json","w") do io 
		JSON.print(io, logical, 2)
	end
	logical
end
logical1 = unique(logical1)

logical2 = map(1:nobs(ds)) do j 
	dd = ff32(ds[j])
	gap = ExplainMill.confidencegap(soft_model, dd, 2)
	e = ExplainMill.GradExplainer2()
	# ms2 = ExplainMill.stats(e, dd, model, 2, 0)
	# f = () -> ExplainMill.confidencegap(soft_model, prune(dd, ms2), 2) - 0.9*gap
	# ExplainMill.importantfirst!(f, ms2, x -> ExplainMill.scorefun(e, x), oscilate = false)
	t2 = @elapsed ms2 = ExplainMill.explain(e, dd, logsoft_model, 2, 200, pruning_method = :importantfirst, threshold = 0.9*gap)
	ExplainMill.e2boolean(ms2, dd, extractor)
	logical = ExplainMill.e2boolean(ms2, dd, extractor)
	open("/tmp/HArr_$(j).json","w") do io 
		JSON.print(io, logical, 2)
	end

end
logical2 = unique(logical2)

