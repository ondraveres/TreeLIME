using Pkg
Pkg.activate("..")
using DataFrames, Statistics, Serialization, PrettyTables, Printf, HypothesisTests, CSV, CodecZlib, BSON

srcdir = "../../data/sims/"

function loadstats(p, filename)
	!isfile(joinpath(srcdir, p, filename)) && return(DataFrame())
	BSON.load(joinpath(srcdir, p, filename))[:exdf]
end

function meanandconfidence(x)
	x = skipmissing(x)
	ci = confint(OneSampleTTest(Float64.(collect(x))))
	s = @sprintf("%.2f",ci[2] - ci[1])
	s = s[2:end]
	v = @sprintf("%.2f",mean(x))
	@sprintf("\$%s\\pm %s\$", v, s)
end

#####
#	Basic visualization
#####
function filtercase(df, ranking::Nothing, level_by_level)
	df[!,:pruning_method] .= String.(df[!,:pruning_method])
	pms = level_by_level ? ["LbyL_GAdd", "LbyL_GArr", "LbyL_GArrft"] : ["Flat_GAdd", "Flat_GArr", "Flat_GArrft"]
	df = filter(r -> r.pruning_method ∈ pms, df)
	exportcase(df, pms)
end

function filtercase(df, ranking::String, level_by_level)
	df[!,:pruning_method] .= String.(df[!,:pruning_method])
	pms = level_by_level ? ["LbyL_HAdd", "LbyL_HArr", "LbyL_HArrft"] : ["Flat_HAdd", "Flat_HArr", "Flat_HArrft"]
	df = filter(r -> r.name == ranking && r.pruning_method ∈ pms, df)
	exportcase(df, pms)
end

function exportcase(df, pms)
	DataFrame(
	 ranking = "",
	 add_e = meanandconfidence(filter(r -> r.pruning_method == pms[1], df)[!,:excess_leaves]),
	 add_t = meanandconfidence(filter(r -> r.pruning_method == pms[1], df)[!,:time]),
	 addrr_e = meanandconfidence(filter(r -> r.pruning_method == pms[2], df)[!,:excess_leaves]),
	 addrr_t = meanandconfidence(filter(r -> r.pruning_method == pms[2], df)[!,:time]),
	 addrrft_e = meanandconfidence(filter(r -> r.pruning_method == pms[3], df)[!,:excess_leaves]),
	 addrrft_t = meanandconfidence(filter(r -> r.pruning_method == pms[3], df)[!,:time]),
	)
end

####
#	Aggregate results to the final table
####
# df = mapreduce(vcat, ["deviceid", "hepatitis", "mutagenesis"]) do problem
# 	mapreduce(vcat, readdir(joinpath(srcdir, problem))) do  task
# 		mapreduce(vcat, readdir(joinpath(srcdir,problem, task))) do i
# 			loadstats(joinpath(problem, task, i), "stats.bson")
# 		end
# 	end
# end

#####
# Exporting data to Table 1
#####

df = open("../../data/table1.csv.gz") do io
    CSV.read(GzipDecompressorStream(io))
end

function maketable(df)
	mapreduce(vcat, [false, true]) do b
		uninformative = filtercase(df, nothing, b)
		heuristic = mapreduce(r -> filtercase(df, r, b), vcat, ["GNN", "GNN2", "Grad", "Banz", "Rnd" ])
		vcat(uninformative, heuristic)		
	end
end

display(maketable(df))
pretty_table(maketable(df), backend = :latex)

#####
# Exporting data to Table 2
#####

df = open("../../data/cuckoo.csv.gz") do io
    CSV.read(GzipDecompressorStream(io))
end
df = filter(r -> startswith(r.filename,"extracted_ben"), df)
ks = [:time, :gradients, :inferences, :nleaves, :flatlength]
adf = by(df, [:name, :pruning_method], dff -> DataFrame([k => mean(dff[!,k]) for k in ks]...))
for (k,d) in [(:time, 0), (:gradients, 0), (:inferences, 0), (:nleaves, 4), (:flatlength, 0)]
	adf[!,k] .= round.(adf[!,k], digits = d)
end
@warn "Table contains results from more samples than reported in the paper. Therefore the numbers are slightly different."
display(adf)
pretty_table(adf, backend = :latex)

#####
# Exporting data selection of best hyperparameters of GNN explainer
#####

gf = open("../../data/gnn.csv.gz") do io
    CSV.read(GzipDecompressorStream(io))
end
gf = by(gf, :name, dff -> DataFrame(:excess_leaves => mean(skipmissing(dff[!,:excess_leaves]))))
gf[!,:alpha] .= map(s -> split(s, "_")[2], gf[!,:name])
gf[!,:beta] .= map(s -> split(s, "_")[3], gf[!,:name])
gf[!,:name] .= map(s -> split(s, "_")[1], gf[!,:name])
display(gf)
pretty_table(maketable(gf), backend = :latex)