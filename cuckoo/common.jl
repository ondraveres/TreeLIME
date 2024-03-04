# ensure that derivatives against sparse are not computed
using JSON
using ExplainMill: jsondiff, nnodes, nleaves
using Zygote, SparseArrays
using Flux, Mill
Sparse = Union{Flux.OneHotMatrix,SparseMatrixCSC,Mill.NGramMatrix}
Zygote.@adjoint Base.:*(A::AbstractMatrix, B::Sparse) =
    A * B, d -> (d * B', nothing)

ff32(n) = n
ff32(n::ArrayNode{Matrix{Float64}}) = ArrayNode(convert(Matrix{Float32}, n.data))
ff32(n::BagNode) = @set n.data = ff32(n.data)
ff32(n::ProductNode) = @set n.data = map(ff32, n.data)



mutable struct StatsLayer
    f::Int
    b::Int
end

StatsLayer() = StatsLayer(0, 0)

function reset!(s::StatsLayer)
    s.f = 0
    s.b = 0
end

function (s::StatsLayer)(x)
    s.f += 1
    x
end

Zygote.@adjoint function (s::StatsLayer)(x)
    s.f += 1
    s.b += 1
    x, Δ -> (nothing, Δ)
end

function isdone(exdf, name, pruning_method, n, info)
    fdf = filter(row ->
            (row.name == name) &&
                (row.pruning_method == pruning_method) &&
                (row.sampleno == info.sampleno) &&
                (row.filename == info.filename) &&
                (row.n == n),
        exdf)
    !isempty(fdf)
end


function stats(dd, ms, extractor, soft_model, i)
    gap = ExplainMill.confidencegap(soft_model, prune(dd, ms), i)
    fv = ExplainMill.FlatView(ms)
    logical = ExplainMill.e2boolean(ms, dd, extractor)[1]
    (gap=gap,
        nleaves=nleaves(logical),
        nnodes=nnodes(logical),
        selected=length(ExplainMill.useditems(fv)),
        flatlength=length(fv)
    )
end

function addexperiment(exdf, e, dd, logsoft_model, i, n, threshold_gap, name, pruning_method, info, statlayer::StatsLayer)
    isdone(exdf, name, pruning_method, n, info) && return (exdf)
    reset!(statlayer)
    t = @elapsed ms = ExplainMill.explain(e, dd, logsoft_model, i, n, pruning_method=pruning_method, threshold=threshold_gap)
    s = merge((
            name=name,
            pruning_method=pruning_method,
            n=n,
            time=t,
            inferences=statlayer.f,
            gradients=statlayer.b,
        ),
        info,
        stats(dd, ms, extractor, soft_model, i)
    )
    jsonname = info.filename[1:end-4]
    jsonname = occursin("/", jsonname) ? split(jsonname, "/")[end] : jsonname
    open("../../data/sims/cuckoo/jsons/$(jsonname)_$(info.sampleno)_$(name)_$(pruning_method).json", "w") do io
        JSON.print(io, ExplainMill.e2boolean(ms, dd, extractor)[1], 2)
    end
    vcat(exdf, DataFrame([s]))
end

function getexplainer(name)
    if name == "Rnd"
        return (ExplainMill.StochasticExplainer(), 0)
    elseif name == "grad"
        return (ExplainMill.GradExplainer2(), 0)
    elseif name == "GNN"
        return (ExplainMill.GnnExplainer(1.0f0, 0.1f0), 200)
    elseif name == "banz"
        return (ExplainMill.DafExplainer(true, true), 200)
    else
        error("unknown explainer $name")
    end
end

