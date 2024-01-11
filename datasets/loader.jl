using JSON
const _jsonpath = "../../data/raw/"
const _simpath = "../../data/sims"

jsondir(s...) = joinpath(_jsonpath, s...)
simdir(s...) = joinpath(_simpath, s...)

loaddata(s::NamedTuple) = loaddata(s.dataset, s.task, s.incarnation)
function loaddata(dataset, task, fold)
    concepts = loadjsonl(jsondir(dataset, task, "$(fold)_concept.jsonl"))
    p = loadjsonl(jsondir(dataset, task, "$(fold)_positive.jsonl"))
    n = loadjsonl(jsondir(dataset, task, "$(fold)_negative.jsonl"))
    y = vcat(fill(1, length(n)), fill(2, length(p)))
    vcat(n, p), y, concepts
end

function loadjsonl(f)
    map(readlines(f)) do s
        JSON.parse(s)
    end
end

