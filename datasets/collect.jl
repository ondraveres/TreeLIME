using Pkg
Pkg.activate("..")
using BSON
using DataFrames
using Statistics
using Serialization

srcdir = "../../data/sims/"


function loadstats(p)
	isfile(joinpath(dstdir, p, "stats.bson")) && return(DataFrame())
	BSON.load(joinpath(dstdir, p, "stats.bson"))[:exdf]
end


df = mapreduce(vcat, ["deviceid", "hepatitis", "mutagenesis"]) do problem
	mapreduce(vcat, readdir(joinpath(srcdir, problem))) do  task
		mapreduce(vcat, readdir(joinpath(srcdir,problem, task))) do i
			loadproblem(joinpath(problem, task, i))
		end
	end
end
