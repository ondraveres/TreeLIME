using Pkg
try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/datasets")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/datasets")
end
Pkg.activate("..")
using DataFrames, Statistics, Serialization, PrettyTables, Printf, HypothesisTests, CSV, CodecZlib, BSON, JLD2

srcdir = "../../data/sims/"
misses = 0
hits = 0

function loadstats(p, filename)
    if !isfile(joinpath(srcdir, p, filename))
        @info "Loading $(joinpath(srcdir, p, filename)) failed"
        global misses += 1
        return (DataFrame())
    end
    global hits += 1
    @info "paths is $(joinpath(srcdir, p, filename))"
    BSON.load(joinpath(srcdir, p, filename))[:exdf]
    # @load joinpath(srcdir, p, filename) exdf #for jdl2
    # return exdf
end

function meanandconfidence(x)
    x = skipmissing(x)
    ci = [0, 0]
    try
        ci = confint(OneSampleTTest(Float64.(collect(x))))
    catch
        @show x
    end
    s = @sprintf("%.2f", ci[2] - ci[1])
    s = s[2:end]
    v = @sprintf("%.2f", mean(x))
    @sprintf("%s ± %s", v, s)
end

#####
#	Basic visualization
#####
function filtercase(df, ranking::Nothing, level_by_level)
    df[!, :pruning_method] .= String.(df[!, :pruning_method])
    pms = level_by_level ? ["LbyL_Gadd", "LbyL_Garr", "LbyL_Garrft"] : ["Flat_Gadd", "Flat_Garr", "Flat_Garrft"]
    df = filter(r -> r.pruning_method ∈ pms, df)
    exportcase(df, pms, level_by_level)
end
function filter_treelime(df)
    df[:, :pruning_method] .= String.(df[:, :pruning_method])
    pms = ["treelime"]
    df_s = filter(r -> r.name == "treelime_sample", df)
    df_m = filter(r -> r.name == "treelime_missing", df)
    vcat(
        exportcase(df_s, pms, "treelime_sample", "treelime_sample"),
        exportcase(df_m, pms, "treelime_missing", "treelime_missing")
    )
end

function filtercase(df, ranking::String, level_by_level)
    df[!, :pruning_method] .= String.(df[!, :pruning_method])
    pms = level_by_level ? ["LbyL_HAdd", "LbyL_HArr", "LbyL_HArrft"] : ["Flat_HAdd", "Flat_HArr", "Flat_HArrft"]
    df = filter(r -> r.name == ranking && r.pruning_method ∈ pms, df)
    exportcase(df, pms, level_by_level, ranking)
end

function exportcase(df, pms, level_by_level, ranking="none")
    data = Dict(
        :ranking => ranking,
        :level_by_level => string(level_by_level),
        :add_e => meanandconfidence(filter(r -> r.pruning_method == pms[1], df)[!, :excess_leaves]),
        :add_m => meanandconfidence(filter(r -> r.pruning_method == pms[1], df)[!, :misses_leaves]),
        :add_t => meanandconfidence(filter(r -> r.pruning_method == pms[1], df)[!, :time]),
        :addrr_e => "-",
        :addrr_t => "-",
        :addrrft_e => "-",
        :addrrft_t => "-"
    )
    if length(pms) >= 3
        data[:addrr_e] = meanandconfidence(filter(r -> r.pruning_method == pms[2], df)[!, :excess_leaves])
        data[:addrr_t] = meanandconfidence(filter(r -> r.pruning_method == pms[2], df)[!, :time])
        data[:addrrft_e] = meanandconfidence(filter(r -> r.pruning_method == pms[3], df)[!, :excess_leaves])
        data[:addrrft_t] = meanandconfidence(filter(r -> r.pruning_method == pms[3], df)[!, :time])
    end
    return select(DataFrame(data), :ranking, :level_by_level, :add_e, :add_m, :add_t, :addrr_e, :addrr_t, :addrrft_e, :addrrft_t)
end

####
#	Aggregate results to the final table
####
misses = 0
hits = 0
df = mapreduce(vcat, ["mutagenesis", "deviceid", "hepatitis"]) do problem
    mapreduce(vcat, readdir(joinpath(srcdir, problem))) do task
        mapreduce(vcat, readdir(joinpath(srcdir, problem, task))) do i
            loadstats(joinpath(problem, task, i), "stability_data7.bson")
        end
    end
end
misses
hits

df



hits / (hits + misses)
#vscodedisplay(df)
BSON.@save "aggregated_merged_data.bson" t1
# mytreelime = filter_treelime(df)

function maketable(df)
    vcat(
        mapreduce(vcat, [false, true]) do b
            uninformative = filtercase(df, nothing, b)
            heuristic = mapreduce(r -> filtercase(df, r, b), vcat, ["gnn", "grad", "banz", "stochastic",
                "lime_m_0.1", "lime_s_0.1", "lime_m_0.2", "lime_s_0.2", "lime_m_0.3", "lime_s_0.3", "lime_m_0.4", "lime_s_0.4", "lime_m_0.5", "lime_s_0.5", "lime_m_0.6", "lime_s_0.6", "lime_m_0.7", "lime_s_0.7", "lime_m_0.8", "lime_s_0.8", "lime_m_0.9", "lime_s_0.9"])
            vcat(uninformative, heuristic)
        end,
        filter_treelime(df)
    )
end
df
t1 = maketable(df)
vscodedisplay(t1)

dontjump

BSON.@load "aggregated_merged_data.bson" t1
t1
using Plots

y = parse.(Float64, first.(split.(t1[!, :add_e], "±")))

group =first.(t1[!, :ranking], 6)
x_s = last.(t1[!, :ranking], 3)
using Random
x_p = [coalesce(tryparse(Float64, x), rand()) for x in x_s]
x_p
y
group
t1[!, :ranking]
plot(t1[!, :ranking], y, group=group)
plot(t1[!, :ranking], y, group=group, size=(800, 600), legend=:top)

xlims!(0, 1)
xlims!(0, 1)
dont jump




# display(maketable(df))
# pretty_table(maketable(df), backend=Val(:latex))

# myyy = BSON.load(joinpath(srcdir, "mutagenesis/one_of_1_5trees/1", "2ninefivestats.bson"))[:exdf]


# myyy.pruning_method
# xdd = filter(r -> r.pruning_method == "LbyL_HAdd", myyy)[!, :excess_leaves]
# df = myyy
# df[!, :pruning_method] .= String.(df[!, :pruning_method])
# pms = true ? ["LbyL_HAdd", "LbyL_HArr", "LbyL_HArrft"] : ["Flat_HAdd", "Flat_HArr", "Flat_HArrft"]
# df = filter(r -> r.pruning_method ∈ pms, df)
# exportcase(df, pms, true)
# [!, :excess_leaves]



#####
# Exporting data to Table 1
#####

# df = open("../../data/table1.csv.gz") do io
#     CSV.read(GzipDecompressorStream(io), DataFrame)
# end





# uninformative = filtercase(df, nothing, true)
# construct = df[!, :pruning_method]
# heuristic = mapreduce(r -> filtercase(df, r, true), vcat, ["GNN", "GNN2", "Grad", "Banz", "Rnd"])
# typeof(t1)


# # # # #####
# # # # # Exporting data to Table 2
# # # # #####

# # # # df = open("../../data/cuckoo.csv.gz") do io
# # # #     CSV.read(GzipDecompressorStream(io))
# # # # end
# # # # df = filter(r -> startswith(r.filename, "extracted_ben"), df)
# # # # ks = [:time, :gradients, :inferences, :nleaves, :flatlength]
# # # # adf = by(df, [:name, :pruning_method], dff -> DataFrame([k => mean(dff[!, k]) for k in ks]...))
# # # # for (k, d) in [(:time, 0), (:gradients, 0), (:inferences, 0), (:nleaves, 4), (:flatlength, 0)]
# # # #     adf[!, k] .= round.(adf[!, k], digits=d)
# # # # end
# # # # @warn "Table contains results from more samples than reported in the paper. Therefore the numbers are slightly different."
# # # # display(adf)
# # # # pretty_table(adf, backend=:latex)

# # # # #####
# # # # # Exporting data selection of best hyperparameters of GNN explainer
# # # # #####

# # # # gf = open("../../data/gnn.csv.gz") do io
# # # #     CSV.read(GzipDecompressorStream(io))
# # # # end
# # # # gf = by(gf, :name, dff -> DataFrame(:excess_leaves => mean(skipmissing(dff[!, :excess_leaves]))))
# # # # gf[!, :alpha] .= map(s -> split(s, "_")[2], gf[!, :name])
# # # # gf[!, :beta] .= map(s -> split(s, "_")[3], gf[!, :name])
# # # # gf[!, :name] .= map(s -> split(s, "_")[1], gf[!, :name])
# # # # display(gf)
# # # # pretty_table(maketable(gf), backend=:latex)