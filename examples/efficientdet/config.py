import itertools


def bifpn_configuration(min_level, max_level, weight_method):
    """A dynamic bifpn configuration that can adapt to
    different min/max levels.
    # Arguments
        min_level: Integer. EfficientNet feature minimum level.
        Level decides the activation map size,
        eg: For an input image of 640 x 640,
        the activation map resolution at level 3 is
        (640 / (2 ^ 3)) x (640 / (2 ^ 3)).
        max_level: Integer. EfficientNet feature maximum level.
        Level decides the activation map size,
        eg: For an input image of 640 x 640,
        the activation map resolution at level 3 is
        (640 / (2 ^ 3)) x (640 / (2 ^ 3)).
        weight_method: String. FPN weighting methods for feature fusion.
    # Returns
        bifpn_config: Dictionary. Contains the nodes and offsets
        for building the FPN.
    """
    bifpn_config = dict()
    bifpn_config['weight_method'] = weight_method or 'fastattn'

    # Node id starts from the input features and monotically
    # increase whenever a new node is added.
    # Example for P3 - P7:
    # Px dimension = input_image_dimension / (2 ** x)
    #     Input    Top-down   Bottom-up
    #     -----    --------   ---------
    #     P7 (4)              P7" (12)
    #     P6 (3)    P6' (5)   P6" (11)
    #     P5 (2)    P5' (6)   P5" (10)
    #     P4 (1)    P4' (7)   P4" (9)
    #     P3 (0)              P3" (8)
    # The node ids are provided in the brackets for a particular
    # feature level x, denoted by Px.
    # So output would be like:
    # [
    #   {'feature_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
    #   {'feature_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
    #   {'feature_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
    #   {'feature_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
    #   {'feature_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
    #   {'feature_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
    #   {'feature_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
    #   {'feature_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
    # ]
    num_levels = max_level - min_level + 1
    node_ids = {min_level + level_id: [level_id]
                for level_id in range(num_levels)}
    id_count = itertools.count(num_levels)
    nodes = []
    # Top-down path
    for level in range(max_level - 1, min_level - 1, -1):
        nodes.append({
            'feature_level': level,
            'inputs_offsets': [node_ids[level][-1],
                               node_ids[level + 1][-1]]
        })
        node_ids[level].append(next(id_count))
    # Bottom-up path
    for level in range(min_level + 1, max_level + 1):
        nodes.append({
            'feature_level': level,
            'inputs_offsets': node_ids[level] + [node_ids[level - 1][-1]]
        })
        node_ids[level].append(next(id_count))
    bifpn_config['nodes'] = nodes
    return bifpn_config


def get_fpn_configuration(fpn_name, min_level, max_level, fpn_weight_method):
    """Provides the fpn configuration that can adapt to
    different min/max levels.
    # Arguments
        fpn_name: String. Name of the FPN, eg: BiFPN.
        min_level: Integer. EfficientNet feature minimum level.
        Level decides the activation map size,
        eg: For an input image of 640 x 640,
        the activation map resolution at level 3 is
        (640 / (2 ^ 3)) x (640 / (2 ^ 3)).
        max_level: Integer. EfficientNet feature maximum level.
        Level decides the activation map size,
        eg: For an input image of 640 x 640,
        the activation map resolution at level 3 is
        (640 / (2 ^ 3)) x (640 / (2 ^ 3)).
        weight_method: String. FPN weighting methods for feature fusion.
    # Returns
        bifpn_config: Dictionary. Contains the nodes and offsets
        for building the FPN.
    """
    if fpn_name == 'BiFPN':
        return bifpn_configuration(min_level, max_level, fpn_weight_method)
    else:
        raise NotImplementedError('FPN name not found.')


def get_efficientdet_default_params(model_name):
    efficientdet_model_param_dict = {
        'efficientdet-d0':
            dict(
                backbone_name='efficientnet-b0',
                image_size=512,
                fpn_num_filters=64,
                fpn_cell_repeats=3,
                box_class_repeats=3,
                anchor_scale=4.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='fastattn',
            ),
        'efficientdet-d1':
            dict(
                backbone_name='efficientnet-b1',
                image_size=640,
                fpn_num_filters=88,
                fpn_cell_repeats=4,
                box_class_repeats=3,
                anchor_scale=4.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='fastattn',
            ),
        'efficientdet-d2':
            dict(
                backbone_name='efficientnet-b2',
                image_size=768,
                fpn_num_filters=112,
                fpn_cell_repeats=5,
                box_class_repeats=3,
                anchor_scale=4.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='fastattn',
            ),
        'efficientdet-d3':
            dict(
                backbone_name='efficientnet-b3',
                image_size=896,
                fpn_num_filters=160,
                fpn_cell_repeats=6,
                box_class_repeats=4,
                anchor_scale=4.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='fastattn',
            ),
        'efficientdet-d4':
            dict(
                backbone_name='efficientnet-b4',
                image_size=1024,
                fpn_num_filters=224,
                fpn_cell_repeats=7,
                box_class_repeats=4,
                anchor_scale=4.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='fastattn',
            ),
        'efficientdet-d5':
            dict(
                backbone_name='efficientnet-b5',
                image_size=1280,
                fpn_num_filters=288,
                fpn_cell_repeats=7,
                box_class_repeats=4,
                anchor_scale=4.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='fastattn',
            ),
        'efficientdet-d6':
            dict(
                backbone_name='efficientnet-b6',
                image_size=1280,
                fpn_num_filters=384,
                fpn_cell_repeats=8,
                box_class_repeats=5,
                anchor_scale=5.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='sum',  # Use unweighted sum for stability.
            ),
        'efficientdet-d7':
            dict(
                backbone_name='efficientnet-b6',
                image_size=1536,
                fpn_num_filters=384,
                fpn_cell_repeats=8,
                box_class_repeats=5,
                anchor_scale=5.0,
                min_level=3,
                max_level=7,
                fpn_weight_method='sum',  # Use unweighted sum for stability.
            ),
        'efficientdet-d7x':
            dict(
                backbone_name='efficientnet-b7',
                image_size=1536,
                fpn_num_filters=384,
                fpn_cell_repeats=8,
                box_class_repeats=5,
                anchor_scale=4.0,
                min_level=3,
                max_level=8,
                fpn_weight_method='sum',  # Use unweighted sum for stability.
            ),
    }
    return efficientdet_model_param_dict[model_name]
