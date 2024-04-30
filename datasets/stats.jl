
function stats(dd, ms, extractor, soft_model, i, concepts)
    gap = ExplainMill.confidencegap(soft_model, dd[ms], i)
    fv = ExplainMill.FlatView(ms)
    logical = ExplainMill.e2boolean(dd, ms, extractor)
    println("explanation size $(nnodes(logical))")
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
        class=i,
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


function add_cape_experiment(exdf, e, dd, logsoft_model, class_to_explain, rel_tol, name, pruning_method, sampleno, statlayer::StatsLayer, extractor, model_variant_k=1)
    reset!(statlayer)
    t = @elapsed my_ms = ExplainMill.explain(e, dd, logsoft_model, class_to_explain, pruning_method=pruning_method, rel_tol=rel_tol)
    logical = ExplainMill.e2boolean(dd, my_ms, extractor)
    s = merge((
            name=name,
            pruning_method=pruning_method,
            sampleno=sampleno,
            time=t,
            dataset="cape",
            task="malware family clasification",
            incarnation=1,
            inferences=statlayer.f,
            gradients=statlayer.b,
            model_variant_k=model_variant_k,
            explanation_json=JSON.json(logical),
            rel_tol=rel_tol
        ),
        stats(dd, my_ms, extractor, logsoft_model, class_to_explain, [Dict()])
    )
    vcat(exdf, DataFrame([s]))
end

function add_cape_treelime_experiment(exdf, e::ExplainMill.TreeLimeExplainer, dd::AbstractMillNode, logsoft_model::AbstractMillModel, class_to_explain, rel_tol, name, pruning_method, sampleno, statlayer::StatsLayer, extractor, model_variant_k=1)
    reset!(statlayer)
    t = @elapsed ms = ExplainMill.treelime(e::ExplainMill.TreeLimeExplainer, dd::AbstractMillNode, model::AbstractMillModel, extractor)
    s = merge((
            name=name,
            pruning_method="treelime",
            sampleno=sampleno,
            time=t,
            dataset="cape",
            task="10 class",
            incarnation="single",
            inferences=statlayer.f,
            gradients=statlayer.b,
            model_variant_k=model_variant_k,
            rel_tol=rel_tol
        ),
        stats(dd, ms, extractor, logsoft_model, class_to_explain, [Dict()])
    )
    vcat(exdf, DataFrame([s]))
end


