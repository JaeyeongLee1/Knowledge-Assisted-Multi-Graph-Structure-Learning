def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        assert feature in data.columns

        res.append(data.loc[:, feature].values.tolist())

    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


def build_loc_net(struc, all_features, feature_map):

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        assert node_name in all_features
        assert node_name in index_feature_map

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            assert child in all_features
            assert child in index_feature_map

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes
