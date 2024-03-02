@load "stability_data.bson" exdf


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


grouped_df = DataFrames.groupby(exdf, [:name, :pruning_method, :sampleno, :incarnation])

json_string1 = collect(grouped_df)[1][!, :explanation_json][1]
json_string2 = collect(grouped_df)[1][!, :explanation_json][2]
json_string3 = collect(grouped_df)[1][!, :explanation_json][3]

dict1 = JSON.parse(json_string1)
dict2 = JSON.parse(json_string2)
dict3 = JSON.parse(json_string3)

ce = jsondiff(dict1, dict2)
ec = jsondiff(dict2, dict1)

misses_nodes = nnodes(ce)
excess_nodes = nnodes(ec)
length([])

nnodes(dict1)
nnodes(dict2)



dict1
dict2

dice_coefficient(dict1, dict3)

dfs = collect(grouped_df)
for df in dfs


end




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


dice_coefficients = DataFrame(name=String[], pruning_method=String[], average_dice=Float16[])
for df in dfs
    jsons = map(JSON.parse, df[!, :explanation_json])
    average_dice = average_operation(jsons)
    push!(dice_coefficients, (name=df[1, "name"], pruning_method=string(df[1, "pruning_method"]), average_dice=average_dice))
end
dfs[1][1, "pruning_method"]

average_df = combine(DataFrames.groupby(dice_coefficients, [:name, :pruning_method]), :average_dice => mean => :average_dice)

vscodedisplay(average_df)

reshaped_df = unstack(average_df, :name, :pruning_method, :average_dice)

vscodedisplay(reshaped_df)

selected_df = select(reshaped_df, ["name", "treelime", "Flat_HAdd", "Flat_HArr", "Flat_HArrft", "LbyL_HAdd", "LbyL_HArr", "LbyL_HArrft"])

selected_df
filtered_df = filter(row -> row[:name] != "gnn2", selected_df)

vscodedisplay(filtered_df)
