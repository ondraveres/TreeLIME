using Pkg
try
    cd("/Users/ondrejveres/Diplomka/ExplainMill.jl/myscripts/datasets")
catch
    cd("/home/veresond/ExplainMill.jl/myscripts/datasets")
end
Pkg.activate("..")

using BSON, JSON, DataFrames, Statistics
using ExplainMill: jsondiff, nnodes, nleaves
using ProgressMeter

BSON.@load "merged_data.bson" df

exdf = df

# vscodedisplay(exdf)

function dice_coefficient(a, b)
    size_a = nnodes(a)
    size_b = nnodes(b)
    ce = jsondiff(a, b)
    ec = jsondiff(b, a)

    misses_nodes = nnodes(ce)
    excess_nodes = nnodes(ec)

    dc = (size_a + size_b - misses_nodes - excess_nodes) / (size_a + size_b)
    return dc
end


grouped_df = DataFrames.groupby(exdf, [:name, :pruning_method, :sampleno, :incarnation, :dataset, :task])

function average_operation(jsons)
    total = 0
    count = 0

    for i in 1:length(jsons)
        for j in i+1:length(jsons)
            total += dice_coefficient(jsons[i], jsons[j])
            count += 1
        end
    end

    return total / count
end


dice_coefficients = DataFrame(name=String[], pruning_method=String[], average_dice=Union{Float16,Missing}[])

n = length(grouped_df)
p = Progress(n, 1)  # 

for df in grouped_df
    jsons = map(JSON.parse, df[!, :explanation_json])
    average_dice = average_operation(jsons)
    push!(dice_coefficients, (name=df[1, "name"], pruning_method=string(df[1, "pruning_method"]), average_dice=average_dice))
    next!(p)  # upd
end

findall(isnan.(dice_coefficients[:, :average_dice]))
nan_indexes = isnan.(dice_coefficients[:, :average_dice])
dice_coefficients
dice_coefficients[nan_indexes, :average_dice] .= missing
average_df = combine(DataFrames.groupby(dice_coefficients, [:name, :pruning_method]), :average_dice => x -> mean(skipmissing(x)))

average_df
reshaped_df = unstack(average_df, :name, :pruning_method, :average_dice_function)

vscodedisplay(reshaped_df)

BSON.@save "big_stability_full.bson" reshaped_df

selected_df = select(reshaped_df, ["name", "treelime", "Flat_HAdd", "Flat_HArr", "Flat_HArrft", "LbyL_HAdd", "LbyL_HArr", "LbyL_HArrft"])

selected_df
filtered_df = filter(row -> row[:name] != "gnn2", selected_df)

vscodedisplay(filtered_df)

BSON.@save "big_stability_shortened.bson" filtered_df


# json_string1 = collect(grouped_df)[1][!, :explanation_json][1]
# json_string2 = collect(grouped_df)[1][!, :explanation_json][2]
# json_string3 = collect(grouped_df)[1][!, :explanation_json][3]

# dict1 = JSON.parse(json_string1)
# dict2 = JSON.parse(json_string2)
# dict3 = JSON.parse(json_string3)

# ce = jsondiff(dict1, dict2)
# ec = jsondiff(dict2, dict1)

# misses_nodes = nnodes(ce)
# excess_nodes = nnodes(ec)
# length([])

# nnodes(dict1)
# nnodes(dict2)



# dict1
# dict2

# dice_coefficient(dict1, dict3)

# dfs = collect(grouped_df)
# for df in dfs


# end
