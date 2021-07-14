from efficientdet_model import EfficientDet


WEIGHT_PATH = '/media/deepan/externaldrive1/project_repos/' \
              'paz/paz_efficientdet_weights/'


def get_EFFICIENTDET(
        model_name,
        backbone,
        image_size,
        fpn_num_filters,
        fpn_cell_repeats,
        box_class_repeats,
        anchor_scale,
        min_level,
        max_level,
        fpn_weight_method):
    """ Generates an EfficientDet model with the parameter
    values passed as argument.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet model
    """

    model = EfficientDet(model_name,
                         backbone,
                         image_size,
                         fpn_num_filters,
                         fpn_cell_repeats,
                         box_class_repeats,
                         anchor_scale,
                         min_level,
                         max_level,
                         fpn_weight_method)
    weights_path = WEIGHT_PATH + model_name + '.h5'
    model.build((1, image_size, image_size, 3))
    model.summary()
    model.load_weights(weights_path)
    return model


def EFFICIENTDET_D0(model_name='efficientdet-d0',
                    backbone='efficientnet-b0',
                    image_size=512,
                    fpn_num_filters=64,
                    fpn_cell_repeats=3,
                    box_class_repeats=3,
                    anchor_scale=4.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='fastattn'):
    """ Instantiates EfficientDet-D0 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D0 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D1(model_name='efficientdet-d1',
                    backbone='efficientnet-b1',
                    image_size=640,
                    fpn_num_filters=88,
                    fpn_cell_repeats=4,
                    box_class_repeats=3,
                    anchor_scale=4.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='fastattn'):
    """ Instantiates EfficientDet-D1 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D1 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D2(model_name='efficientdet-d2',
                    backbone='efficientnet-b2',
                    image_size=768,
                    fpn_num_filters=112,
                    fpn_cell_repeats=5,
                    box_class_repeats=3,
                    anchor_scale=4.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='fastattn'):
    """ Instantiates EfficientDet-D2 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D2 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D3(model_name='efficientdet-d3',
                    backbone='efficientnet-b3',
                    image_size=896,
                    fpn_num_filters=160,
                    fpn_cell_repeats=6,
                    box_class_repeats=4,
                    anchor_scale=4.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='fastattn'):
    """ Instantiates EfficientDet-D3 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D3 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D4(model_name='efficientdet-d4',
                    backbone='efficientnet-b4',
                    image_size=1024,
                    fpn_num_filters=224,
                    fpn_cell_repeats=7,
                    box_class_repeats=4,
                    anchor_scale=4.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='fastattn'):
    """ Instantiates EfficientDet-D4 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D4 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D5(model_name='efficientdet-d5',
                    backbone='efficientnet-b5',
                    image_size=1280,
                    fpn_num_filters=288,
                    fpn_cell_repeats=7,
                    box_class_repeats=4,
                    anchor_scale=4.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='fastattn'):
    """ Instantiates EfficientDet-D5 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D5 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D6(model_name='efficientdet-d6',
                    backbone='efficientnet-b6',
                    image_size=1280,
                    fpn_num_filters=384,
                    fpn_cell_repeats=8,
                    box_class_repeats=5,
                    anchor_scale=5.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='sum'):
    """ Instantiates EfficientDet-D6 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D6 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D7(model_name='efficientdet-d3',
                    backbone='efficientnet-b7',
                    image_size=1536,
                    fpn_num_filters=384,
                    fpn_cell_repeats=8,
                    box_class_repeats=5,
                    anchor_scale=5.0,
                    min_level=3,
                    max_level=7,
                    fpn_weight_method='sum'):
    """ Instantiates EfficientDet-D7 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D7 model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model


def EFFICIENTDET_D7x(model_name='efficientdet-d7x',
                     backbone='efficientnet-b7',
                     image_size=1536,
                     fpn_num_filters=384,
                     fpn_cell_repeats=8,
                     box_class_repeats=5,
                     anchor_scale=4.0,
                     min_level=3,
                     max_level=8,
                     fpn_weight_method='sum'):
    """ Instantiates EfficientDet-D7x model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D7x model
    """
    model = get_EFFICIENTDET(model_name,
                             backbone,
                             image_size,
                             fpn_num_filters,
                             fpn_cell_repeats,
                             box_class_repeats,
                             anchor_scale,
                             min_level,
                             max_level,
                             fpn_weight_method)
    return model
