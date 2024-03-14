using ProgressMeter
function treelime(ds, model, extractor, sch, perturbation_count, perturbation_chance, perturbation_strategy)
    mask = ExplainMill.create_mask_structure(ds, d -> HeuristicMask(fill(0.7, d)))
    # p = Progress(perturbation_count, 1)  # 
    modified_samples = []
    modification_masks = []
    flat_modification_masks = []
    labels = []
    samples = []

    og_class = Flux.onecold((model(ds)))[1]
    println("og_class is ", og_class)
    # o_logical = ExplainMill.e2boolean(ds, mask, extractor)
    # o_logical_json = JSON.json(o_logical)
    # open("original.json", "w") do f
    #     write(f, o_logical_json)
    # end
    for i in 1:perturbation_count
        mysample_copy = deepcopy(ds)
        mask_copy = deepcopy(mask)
        sch_copy = deepcopy(sch)
        extractor_copy = deepcopy(extractor)
        (s, m) = my_recursion(mysample_copy, mask_copy, extractor_copy, sch_copy, 1 - perturbation_chance, perturbation_strategy)
        local new_flat_view = ExplainMill.FlatView(m)
        new_mask_bool_vector = [new_flat_view[i] for i in 1:length(new_flat_view.itemmap)]
        push!(flat_modification_masks, new_mask_bool_vector)
        push!(labels, argmax(model(s))[1])
        push!(samples, s)
        # println(model(s))
        # println(model(s))
        push!(modified_samples, s)
        push!(modification_masks, m)
        # logical = ExplainMill.e2boolean(s, mask, extractor)
        # logical_json = JSON.json(logical)
        # filename = "logical_$(i).json"
        # next!(p)  # upd
        # open(filename, "w") do f
        #     write(f, logical_json)
        # end
    end
    # dss = reduce(catobs, samples)
    # labels = Flux.onecold((model(dss)))
    println("labels are ", labels)
    # results = tmap(model, samples)
    # labels = Flux.onecold.(results)
    println("exploration rate: ", 1 - mean(labels .== og_class))
    # return labels
    labels = ifelse.(labels .== og_class, 2, 1)
    mean(labels .== 2)

    println("lenght ", length(flat_modification_masks[1]))

    X = hcat(flat_modification_masks...)
    y = labels

    # println("y is ", y)

    Xmatrix = convert(Matrix, X')  # transpose X because glmnet assumes features are in columns
    yvector = convert(Vector, y)

    # Fit the model

    cv = glmnetcv(Xmatrix, yvector; alpha=1.0)  # alpha=1.0 for lasso
    βs = cv.path.betas
    λs = cv.lambda
    βs

    sharedOpts = (legend=false, xlabel="lambda", xscale=:log10)
    # p2 = plot(λs, βs', title="Across Cross Validation runs"; sharedOpts...)
    #p2

    # The fitted coefficients at the best lambda can be accessed as follows:
    coef = GLMNet.coef(cv)


    non_zero_indices = findall(x -> abs(x) > 0, coef)


    y_pred = GLMNet.predict(cv, Xmatrix)
    # println("y_pred is ", y_pred)
    y_pred_labels = ifelse.(y_pred .>= 1.5, 2, 1)
    println("mean prediction label ", mean(y_pred_labels))
    my_accuracy = mean(y_pred_labels .== yvector)
    println("Accuracy: $my_accuracy, Non-zero indexes: $(length(non_zero_indices))")


    # leafmap!(mask) do mask_node
    #     mask_node.mask.x .= false
    #     return mask_node
    # end

    new_flat_view = ExplainMill.FlatView(mask)
    # new_flat_view[non_zero_indices] = true
    y_pred_inverted = 1 .- y_pred
    for i in 1:length(flat_modification_masks[1])
        mi = new_flat_view.itemmap[i]
        new_flat_view.masks[mi.maskid].h[mi.innerid] = abs(coef[i])
        # new_flat_view.masks[i].h = 0.123
    end

    # ex = ExplainMill.e2boolean(ds, mask, extractor)


    return mask
end



function extractbatch_andstore(extractor, samples; store_input=false)
    mapreduce(s -> extractor(s, store_input=store_input), catobs, samples)
end

function my_recursion(data_node, mask_node, extractor_node, schema_node, perturbation_chance, perturbation_strategy)
    # println("recursion start")
    # printtree(data_node)
    # println("end of log")
    if data_node isa ProductNode
        children_names = []
        modified_data_ch_nodes = []
        modified_mask_ch_nodes = []
        for (
            (data_ch_name, data_ch_node),
            (mask_ch_name, mask_ch_node)
        ) in zip(
            pairs(children(data_node)),
            pairs(children(mask_node))
        )
            push!(children_names, data_ch_name)
            extractor_child_node = extractor_node[data_ch_name]
            scheme_child_node = nothing
            try
                scheme_child_node = schema_node[data_ch_name]
            catch e
                printtree(schema_node)
                @error e
                return
            end
            (modified_child_data, modified_child_mask) = my_recursion(data_ch_node, mask_ch_node, extractor_child_node, scheme_child_node, perturbation_chance, perturbation_strategy)
            push!(modified_data_ch_nodes, modified_child_data)
            push!(modified_mask_ch_nodes, modified_child_mask)
        end
        nt_data = NamedTuple{Tuple(children_names)}(modified_data_ch_nodes)
        nt_mask = NamedTuple{Tuple(children_names)}(modified_mask_ch_nodes)

        return ProductNode(nt_data), ExplainMill.ProductMask(nt_mask)
    end
    if data_node isa BagNode

        child_node = Mill.data(data_node)
        (modified_data_child_node, modified_child_mask) = my_recursion(child_node, mask_node.child, extractor_node.item, schema_node.items, perturbation_chance, perturbation_strategy)

        return BagNode(modified_data_child_node, data_node.bags, data_node.metadata), ExplainMill.BagMask(modified_child_mask, mask_node.bags, mask_node.mask)
    end
    if data_node isa ArrayNode
        if numobs(data_node) == 0
            return (data_node, mask_node)
        end
        total = sum(values(schema_node.counts))
        normalized_probs = [v / total for v in values(schema_node.counts)]
        n = length(normalized_probs)  # Get the number of elements
        w = Weights(ones(n))#collect(values(schema_node.counts)))
        vals = collect(keys(schema_node.counts))

        if mask_node isa ExplainMill.NGramMatrixMask
            # println(vals)
        end

        if mask_node isa ExplainMill.CategoricalMask || mask_node isa ExplainMill.NGramMatrixMask
            new_hot_vectors = []
            new_random_keys = []
            global my_mask_node = mask_node
            global my_extractor_node = extractor_node
            global my_schema_node = schema_node
            global my_data_node = data_node
            new_values = []
            # println("new values start ", numobs(data_node), " -> ", data_node)
            for i in 1:numobs(data_node)
                # original_hot_vector = data_node.data[:, i]
                if rand() > perturbation_chance

                    random_val = sample(vals, w)
                    # if (random_val == data_node.metadata[i])
                    #     println("MATCH")
                    # else
                    #     println("MISS")
                    # end
                    if perturbation_strategy == "missing"
                        random_val = missing
                    end
                    push!(new_values, random_val)
                    # println("pushing ", random_val)

                    mask_node.mask.x[i] = true
                else
                    # println("pushing ", data_node.metadata[i])

                    push!(new_values, data_node.metadata[i])
                    mask_node.mask.x[i] = false
                end

            end
            # println("new values done ", new_values)

            new_array_node = extractbatch_andstore(extractor_node, new_values; store_input=true)
            return new_array_node, mask_node
        elseif mask_node isa ExplainMill.FeatureMask
            if rand() > perturbation_chance
                new_hot_vectors = []
                new_random_keys = []
                global my_mask_node = mask_node
                global my_extractor_node = extractor_node
                global my_schema_node = schema_node
                global my_data_node = data_node
                for i in 1:numobs(data_node)
                    random_key = sample(vals, w)
                    if perturbation_strategy == "missing"
                        random_key = missing
                    end
                    push!(new_random_keys, random_key)
                end
                mask_node.mask.x[1] = true
                new_array_node = extractbatch_andstore(extractor_node, new_random_keys; store_input=true)
            else
                mask_node.mask.x[1] = false
                return data_node, mask_node
            end
            return new_array_node, mask_node
        elseif mask_node isa ExplainMill.EmptyMask

        else
            @warn typeof(mask_node)
        end
        return ArrayNode(Mill.data(data_node), data_node.metadata), mask_node
    end
end

