function treelime(ds, model, extractor, schema)
    mask = ExplainMill.create_mask_structure(ds, d -> SimpleMask(fill(false, d)))
    N = 100
    modified_samples = []
    modification_masks = []
    flat_modification_masks = []
    labels = []
    for i in 1:N
        mysample_copy = deepcopy(ds)
        mask_copy = deepcopy(mask)
        sch_copy = deepcopy(sch)
        extractor_copy = deepcopy(extractor)
        (s, m) = my_recursion(mysample_copy, mask_copy, extractor_copy, sch_copy)
        local new_flat_view = ExplainMill.FlatView(m)
        new_mask_bool_vector = [new_flat_view[i] for i in 1:length(new_flat_view.itemmap)]
        push!(flat_modification_masks, new_mask_bool_vector)
        push!(labels, argmax(model(s))[1])
        push!(modified_samples, s)
        push!(modification_masks, m)
    end
    mean(labels .== 2)

    mean(flat_modification_masks)





    X = hcat(flat_modification_masks...)
    y = labels

    Xmatrix = convert(Matrix, X')  # transpose X because glmnet assumes features are in columns
    yvector = convert(Vector, y)

    # Fit the model
    cv = glmnetcv(Xmatrix, yvector; alpha=1.0)  # alpha=1.0 for lasso
    βs = cv.path.betas
    λs = cv.lambda
    βs

    sharedOpts = (legend=false, xlabel="lambda", xscale=:log10)
    p2 = plot(λs, βs', title="Across Cross Validation runs"; sharedOpts...)
    #p2

    # The fitted coefficients at the best lambda can be accessed as follows:
    coef = GLMNet.coef(cv)


    non_zero_indices = findall(x -> abs(x) > 0, coef)

    coef

    for (i, c) in enumerate(coef)
        println("Feature $i has coefficient $c")
    end

    y_pred = GLMNet.predict(cv, Xmatrix)


    y_pred_labels = ifelse.(y_pred .>= 0.5, 2, 1)


    my_accuracy = mean(y_pred_labels .== yvector)

    println("Accuracy: $my_accuracy")

    new_mask = ExplainMill.create_mask_structure(mysample, d -> SimpleMask(fill(true, d)))

    leafmap!(new_mask) do mask_node
        mask_node.mask.x .= false
        return mask_node
    end

    new_flat_view = ExplainMill.FlatView(new_mask)
    new_flat_view[non_zero_indices] = true



    ex = ExplainMill.e2boolean(ds, new_mask, extractor)


    # json_str = JSON.json(ex)
    return ex

    # # Write the JSON string to a file
    # open(resultsdir("my_explanation.json"), "w") do f
    #     write(f, json_str)
    # end

end



function extractbatch_andstore(extractor, samples; store_input=false)
    mapreduce(s -> extractor(s, store_input=store_input), catobs, samples)
end

function my_recursion(data_node, mask_node, extractor_node, schema_node)
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
            (modified_child_data, modified_child_mask) = my_recursion(data_ch_node, mask_ch_node, extractor_node[data_ch_name], schema_node[data_ch_name])
            push!(modified_data_ch_nodes, modified_child_data)
            push!(modified_mask_ch_nodes, modified_child_mask)
        end
        nt_data = NamedTuple{Tuple(children_names)}(modified_data_ch_nodes)
        nt_mask = NamedTuple{Tuple(children_names)}(modified_mask_ch_nodes)

        return ProductNode(nt_data), ExplainMill.ProductMask(nt_mask)
    end
    if data_node isa BagNode
        child_node = Mill.data(data_node)
        (modified_data_child_node, modified_child_mask) = my_recursion(child_node, mask_node.child, extractor_node.item, schema_node.items)

        return BagNode(modified_data_child_node, data_node.bags, data_node.metadata), ExplainMill.BagMask(modified_child_mask, mask_node.bags, mask_node.mask)
    end
    if data_node isa ArrayNode
        total = sum(values(schema_node.counts))
        normalized_probs = [v / total for v in values(schema_node.counts)]
        n = length(normalized_probs)  # Get the number of elements
        w = Weights(ones(n))
        vals = collect(keys(schema_node.counts))

        if mask_node isa ExplainMill.CategoricalMask
            new_hot_vectors = []
            new_random_keys = []
            global my_mask_node = mask_node
            global my_extractor_node = extractor_node
            global my_schema_node = schema_node
            global my_data_node = data_node
            new_values = []
            for i in 1:numobs(data_node)
                # original_hot_vector = data_node.data[:, i]
                if rand() > 0.5
                    random_val = sample(vals, w)
                    push!(new_values, random_val)
                    mask_node.mask.x[i] = true
                else
                    push!(new_values, data_node.metadata[i])
                    mask_node.mask.x[i] = false
                end

            end

            new_array_node = extractbatch_andstore(extractor_node, new_values; store_input=true)
            return new_array_node, mask_node
        end
        if mask_node isa ExplainMill.FeatureMask
            if rand() > 0.5
                new_hot_vectors = []
                new_random_keys = []
                global my_mask_node = mask_node
                global my_extractor_node = extractor_node
                global my_schema_node = schema_node
                global my_data_node = data_node
                for i in 1:numobs(data_node)
                    random_key = sample(vals, w)
                    push!(new_random_keys, random_key)
                end
                mask_node.mask.x[1] = true
                new_array_node = extractbatch_andstore(extractor_node, new_random_keys; store_input=true)
            else
                mask_node.mask.x[1] = false
                return data_node, mask_node
            end
            return new_array_node, mask_node
        end
        @error ExplainMill.CategoricalMask
        return ArrayNode(Mill.data(data_node), data_node.metadata), mask_node
    end
end
