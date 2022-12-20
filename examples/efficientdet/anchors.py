import numpy as np
from paz.backend.boxes import to_center_form


def build_prior_boxes(model, num_scales, aspect_ratios, anchor_scale):
    """Generates anchor boxes in centre form for all feature levels.

    # Arguments:
        min_level: Int, first EfficientNet layer index.
        max_level: Int, last EfficientNet layer index.
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        anchor_scales: Array of shape `(9,)`, anchor box scales.

    # Returns:
        anchors: Array of shape `(49104, 4)`.
    """
    config = make_configuration(model, num_scales, aspect_ratios, anchor_scale)
    _, _, _, _, _, num_scale_aspect = config
    image_size = model.input.shape[1:3].as_list()
    boxes_all = []
    for args in zip(*config[:-1]):
        boxes_level = generate_level_boxes(*args, num_scale_aspect, image_size)
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    prior_boxes = np.concatenate(boxes_all, axis=0).astype('float32')
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes


def make_configuration(model, num_scales, aspect_ratios, anchor_scale):
    """Generates anchor box parameter combinations.

    # Arguments:
        min_level: Int, first EfficientNet layer index.
        max_level: Int, last EfficientNet layer index.
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        anchor_scale: Array of shape `(5,)`, anchor box scales.

    # Returns:
        Tuple: being generated configuarations.
    """
    num_levels = len(model.branches)
    num_scale_aspect = num_scales * len(aspect_ratios)
    features_H, features_W = get_feature_dims(model, num_scale_aspect)
    octave_scales = build_octaves(num_scales, aspect_ratios, num_levels)
    aspects = build_aspects(aspect_ratios, num_scales, num_levels)
    strides = build_strides(model, features_H, features_W, num_levels)
    anchor_scales = build_scales(model, anchor_scale, num_scale_aspect)
    strides_y, strides_x = strides
    return (strides_y, strides_x, octave_scales, aspects, anchor_scales,
            num_scale_aspect)


def get_feature_dims(model, num_scale_aspect):
    """Calculates layer-wise EfficientNet feature height and width.

    # Arguments:
        min_level: Int, being first EfficientNet layer index.
        max_level: Int, being last EfficientNet layer index.
        num_scale_aspect: Int, number of scales aspect ratios
            combinations.

    # Returns:
        Tuple: Containing feature height and width.
    """
    feature_W, feature_H = [], []
    for branch in model.branches:
        feature_W.extend([branch.shape[1]])
        feature_H.extend([branch.shape[2]])
    features_H = np.repeat(feature_H, num_scale_aspect).astype(np.float64)
    features_W = np.repeat(feature_W, num_scale_aspect).astype(np.float64)
    return features_H, features_W


def build_octaves(num_scales, aspect_ratios, num_levels):
    """Generates layer-wise EfficientNet anchor box octaves.

    # Arguments:
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        num_levels: Int, number of feature levels.

    # Returns:
        octave_scales: Array of shape `(5, 9)`.
    """
    scale_octaves = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octaves_tiled = np.tile(scale_octaves, num_levels)
    octaves_standardized = octaves_tiled / float(num_scales)
    octave_scales = octaves_standardized.reshape(num_levels, -1)
    return octave_scales


def build_aspects(aspect_ratios, num_scales, num_levels):
    """Generates layer-wise EfficientNet anchor box aspect ratios.

    # Arguments:
        aspect_ratios: List, anchor boxes aspect ratios.
        num_scales: Int, number of anchor box scales.
        num_levels: Int, number of feature levels.

    # Returns:
        aspects: Array of shape `(5, 9)`.
    """
    aspect = np.tile(aspect_ratios, len(range(num_scales)))
    aspects_tiled = np.tile(aspect, num_levels)
    aspects = aspects_tiled.reshape(num_levels, -1)
    return aspects


def build_strides(model, features_H, features_W, num_levels):
    """Generates layer-wise EfficientNet anchor box strides.

    # Arguments:
        features_H: Array of shape `(45,)`, input feature height.
        features_W: Array of shape `(45,)`, input feature width.
        num_levels: Int, number of feature levels.

    # Returns:
        Tuple: Containing strides in y and x direction.
    """
    base_feature_H, base_feature_W = model.input.shape[1:3].as_list()
    H_inverse = np.reciprocal(features_H)
    strides_y = np.reshape(base_feature_H * H_inverse, (num_levels, -1))
    W_inverse = np.reciprocal(features_W)
    strides_x = np.reshape(base_feature_W * W_inverse, (num_levels, -1))
    return strides_y, strides_x


def build_scales(model, anchor_scale, num_scale_aspect):
    """Generates layer-wise EfficientNet anchor box scales.

    # Arguments:
        anchor_scale: Array of shape `(5,)`, anchor box scales.
        num_scale_aspect:  Int, number of scale aspect ratio
            combinations.
        num_levels: Int, number of feature levels.

    # Returns:
        anchor_scales: Array of shape ``(5, 9)``.
    """
    num_levels = len(model.branches)
    anchor_scale = np.repeat(anchor_scale, len(model.branches))
    anchors_repeated = np.repeat(anchor_scale, num_scale_aspect)
    anchor_scales = np.reshape(anchors_repeated, (num_levels, -1))
    return anchor_scales


def generate_level_boxes(strides_y, strides_x, octave_scales, aspects,
                         anchor_scales, num_scale_aspect, image_size):
    """Generates anchor box in centre form per feature level.

    # Arguments:
        stride_y: Array of shape `()`, y-direction stride.
        stride_x: Array of shape `()`, x-direction stride.
        octave_scales: Array of shape `(9,)`, anchor box octave scales.
        aspects: Array of shape `(9,)`, anchor box aspects.
        anchor_scales: Array of shape `(9,)`, anchor box scales.
        image_size: Tuple, being input image size.
        num_scale_aspect: Int, number of scale aspect ratio
            combinations.

    # Returns:
        boxes_level: List containing anchor boxes in centre form.
    """
    boxes_level = []
    for combination in range(num_scale_aspect):
        box_coordinates = compute_box_coordinates(
            strides_y[combination], strides_x[combination],
            octave_scales[combination], aspects[combination],
            anchor_scales[combination], image_size)
        center_x, center_y, anchor_x, anchor_y = box_coordinates
        boxes = np.concatenate(([center_x - anchor_x], [center_y - anchor_y],
                                [center_x + anchor_x], [center_y + anchor_y]),
                               axis=0)
        boxes = np.swapaxes(boxes, 0, 1)
        boxes_level.append(np.expand_dims(boxes, axis=1))
    return boxes_level


def compute_box_coordinates(stride_y, stride_x, octave_scale, aspect,
                            anchor_scale, image_size):
    """Calculates anchor box coordinates in centre form.

    # Arguments:
        stride_y: Array of shape `()`, y-direction stride.
        stride_x: Array of shape `()`, x-direction stride.
        octave_scale: Array of shape `()`, anchor box octave scale.
        aspect: Array of shape `()`, anchor box aspect ratio.
        anchor_scale: Array of shape `()`, anchor box scales.
        image_size: Tuple, being input image size.

    # Returns:
        Tuple: holding anchor box centre, width and height.
    """
    W, H = image_size
    base_anchor_x = anchor_scale * stride_x * (2 ** octave_scale)
    base_anchor_y = anchor_scale * stride_y * (2 ** octave_scale)
    aspect_x, aspect_y = compute_aspect_ratio(aspect)
    anchor_x = (base_anchor_x * aspect_x / 2.0) / H
    anchor_y = (base_anchor_y * aspect_y / 2.0) / W
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    center_x = center_x.reshape(-1) / H
    center_y = center_y.reshape(-1) / W
    return center_x, center_y, anchor_x, anchor_y


def compute_aspect_ratio(aspect):
    """Calculates anchor box aspect ratio.

    # Arguments:
        aspect: Float, anchor box aspect.

    # Returns:
        Tuple: containing anchor box aspect ratios.
    """
    aspect_x = np.sqrt(aspect)
    aspect_y = 1 / aspect_x
    return aspect_x, aspect_y
