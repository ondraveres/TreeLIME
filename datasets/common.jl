using Random
using JsonGrinder: is_intable, is_floatable, unify_types, extractscalar

function loadclass(k, n=typemax(Int))
    dss = map(s -> extractor(s, store_input=true), sample(samples[my_class_indexes[k]], min(n, length(my_class_indexes[k])), replace=false))
    reduce(catobs, dss)
end


function onlycorrect(dss, i, min_confidence=0)
    correct = mypredict(soft_model, dss, [1, 2]) .== i
    dss = dss[correct[:]]
    min_confidence == 0 && return (dss)
    correct = ExplainMill.confidencegap(soft_model, dss, i) .>= min_confidence
    dss[correct[:]]
end

function getexplainer(name)
    if name == "stochastic"
        return ExplainMill.StochasticExplainer()
    elseif name == "grad"
        return ExplainMill.GradExplainer2()
    elseif name == "gnn"
        return ExplainMill.GnnExplainer()
    elseif name == "banz"
        return ExplainMill.DafExplainer()
    else
        error("unknown eplainer $name")
    end
end

function scalar_extractor()
    [(e -> length(keys(e)) <= 100,
            e -> ExtractCategorical(keys(e))),
        (e -> is_intable(e),
            e -> extractscalar(Int64, e)),
        (e -> is_floatable(e),
            e -> extractscalar(Float64, e)),
        (e -> true,
            e -> extractscalar(unify_types(e), e)),]
end

function loaddata()
    samples = open(datadir("data.json"), "r") do fio
        JSON.parse(fio)
    end

    meta = open(datadir("meta.json"), "r") do fio
        JSON.parse(fio)
    end
    samples, meta
end

function splitindices(samples, meta)
    ii = randperm(length(samples))
    i = meta["val_samples"]
    j = meta["val_samples"] + meta["test_samples"]
    ii[j+1:end], ii[1:i], ii[i+1:j]
end

function tweakextractor!(extractor, sch, settings)
    if settings.dataset == "sap_balanced"
        extractor.vec[:age] = extractscalar(Float32, sch[:age])
        delete!(extractor.other, :age)
        extractor.vec[:educationnum] = extractscalar(Float32, sch[:educationnum])
        delete!(extractor.other, :educationnum)
        foreach(k -> delete!(extractor.other, k), [:nom1, :nom2, :nom3])
    end

    if settings.dataset == "hepatitis"
        delete!(extractor.other, :age)
        ExtractDict(Dict(:age => extractscalar(Float32, sch[:age])), extractor.other)
    end

    if settings.dataset == "stats_raw"
        extractor.vec[:down_votes] = extractscalar(Float32, sch[:down_votes])
        delete!(extractor.other, :down_votes)
    end

    # if settings.dataset == "carcinogenesis"
    # 	for k in keys(sch[:bond1].items)
    # 		for j in [:bond2, :bond3, :bond7]
    # 			merge!(sch[:bond1].items[k].counts,sch[j].items[k].counts)
    # 		end
    # 	end
    # 	e = suggestextractor(sch, (scalar_extractors = scalar_extractor(),))
    # 	for j in keys(extractor.other)
    # 		extractor.other[j] = e[:bond1]
    # 	end
    # end

end

###############################################################
#  train
###############################################################
function initmbprovider(data, target, train_idxs, n)
    makebatch = initbatchprovider(data, target, train_idxs, n)
    () -> begin
        x, y = makebatch()
        reduce(catobs, x), y
    end
end

function accuracy(model, idxs)
    ŷ = map(i -> argmax(softmax(model(data[i]).data)[:]), idxs)
    y = target[idxs]
    mean(ŷ .== y)
end

function accuracy(model, idxs, oclasses)
    ŷ = map(i -> oclasses[argmax(softmax(model(data[i]).data)[:])], idxs)
    y = target[idxs]
    mean(ŷ .== y)
end



function printyaras(yara)
    for j in 1:length(yara)
        printstyled("Sample ", j, "\n"; color=:green, bold=true)
        JSON.print(yara[j], 2)
        println()
    end
end

function printyaras(io, yara)
    for j in 1:length(yara)
        println(io, "Sample ", j)
        JSON.print(io, yara[j], 2)
        println(io)
    end
end


