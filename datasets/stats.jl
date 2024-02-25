
function stats(dd, ms, extractor, soft_model, i, concepts)
    gap = ExplainMill.confidencegap(soft_model, dd[ms], i)
    fv = ExplainMill.FlatView(ms)
    logical = ExplainMill.e2boolean(dd, ms, extractor)
    original_confidence_gap = ExplainMill.confidencegap(soft_model, dd, i)
    explanation_confidence_gap = ExplainMill.confidencegap(soft_model, dd[ms], i)

    ce = map(c -> jsondiff(c, logical), concepts)
    ec = map(c -> jsondiff(logical, c), concepts)
    second_concept = "no second concept"
    try
        second_concept = JSON.json(concepts[2])
    catch
    end
    return (
        (
        gap=gap,
        original_confidence_gap=original_confidence_gap,
        explanation_confidence_gap=explanation_confidence_gap,
        misses_nodes=minimum(nnodes(c) for c in ce),
        misses_leaves=minimum(nleaves(c) for c in ce),
        excess_nodes=minimum(nnodes(c) for c in ec),
        excess_leaves=minimum(nleaves(c) for c in ec),
        nleaves=nleaves(logical),
        nnodes=nnodes(logical),
        cleaves=nleaves(concepts),
        cnodes=nnodes(concepts),
        selected=length(ExplainMill.useditems(fv)),
        explanation_json=JSON.json(logical),
        concepts_1_json=JSON.json(concepts[1]),
        concepts_2_json=second_concept,
        flatlength=length(fv),
    )
    )
end




###############################################################
#  checking if experiment has been already done
###############################################################
function isdone(exdf, name, pruning_method, sampleno, n)
    fdf = filter(row ->
            (row.name == name) &&
                (row.pruning_method == pruning_method) &&
                (row.sampleno == sampleno) &&
                (row.n == n),
        exdf)
    !isempty(fdf)
end

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

function addexperiment(exdf, e, dd, logsoft_model, i, n, threshold_gap, name, pruning_method, sampleno, settings, statlayer::StatsLayer)
    isdone(exdf, name, pruning_method, sampleno, n) && return (exdf)
    reset!(statlayer)
    t = @elapsed ms = ExplainMill.explain(e, dd, logsoft_model, i, pruning_method=pruning_method, rel_tol=0.9)
    s = merge((
            name=name,
            pruning_method=pruning_method,
            sampleno=sampleno,
            n=n,
            time=t,
            dataset=settings.dataset,
            task=settings.task,
            incarnation=settings.incarnation,
            inferences=statlayer.f,
            gradients=statlayer.b,
        ),
        stats(dd, ms, extractor, soft_model, i, concepts)
    )
    vcat(exdf, DataFrame([s]))
end

function add_treelime_experiment(exdf, dd, logsoft_model, i, n, sampleno, settings, statlayer::StatsLayer)
    reset!(statlayer)
    t = @elapsed ms = treelime(dd, logsoft_model, extractor, schema)
    s = merge((
            name="treelime",
            pruning_method="treelime",
            sampleno=sampleno,
            n=n,
            time=t,
            dataset=settings.dataset,
            task=settings.task,
            incarnation=settings.incarnation,
            inferences=statlayer.f,
            gradients=statlayer.b
        ),
        stats(dd, ms, extractor, soft_model, i, concepts)
    )
    vcat(exdf, DataFrame([s]))
end

