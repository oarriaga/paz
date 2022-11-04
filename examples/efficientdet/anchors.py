import numpy as np
from paz.backend.boxes import to_center_form


def compute_feature_sizes(image_size, max_level):
    """Computes the size of features that are given as input to each of
        EfficientNet layers.

    # Arguments
        image_size: Tuple, (Height, Width) of the input image.
        max_level: Int, maximum level for features.

    # Returns
        feature_sizes: Numpy array of shape ``(8, 2)``.
    """
    feature_H, feature_W = image_size
    feature_sizes = np.array([feature_H, feature_W], dtype=np.float64)
    for _ in range(1, max_level + 1):
        feature_H = (feature_H - 1) // 2 + 1
        feature_W = (feature_W - 1) // 2 + 1
        feature_size = np.array([feature_H, feature_W])
        feature_sizes = np.vstack((feature_sizes, feature_size))
    return feature_sizes


def generate_configurations(feature_sizes, min_level, max_level,
                            num_scales, aspect_ratios, anchor_scale):
    """Generates configurations or in other words different combinations of
        strides, octave scales, aspect ratios and anchor scales for each level
        of the EfficientNet layers that feeds BiFPN network.

    # Arguments:
        feature_sizes: Numpy array representing the feature sizes of inputs to
            EfficientNet layers.
        min_level: Int, Number representing the index of the earliest
            EfficientNet layer that feeds the BiFPN layers.
        max_level: Int, Number representing the index of the last EfficientNet
            layer that feeds the BiFPN layers.
        num_scales: Int, specifying the number of scales in the
            anchor boxes.
        aspect_ratios: List, specifying the aspect rations of the anchor boxes.
        anchor_scale: Numpy array representing the scales of the anchor box.

    # Returns:
        Tuple: Containing configuarations of strides, octave scales, aspect
            ratios and anchor scales.
    """
    num_levels = max_level + 1 - min_level
    scale_aspect_ratio_combinations = (
        len(range(num_scales)) * len(aspect_ratios))
    features_H, features_W = build_features(
        feature_sizes, min_level, max_level, scale_aspect_ratio_combinations)
    octave_scales = build_octaves(num_scales, aspect_ratios, num_levels)
    aspects = build_aspects(aspect_ratios, num_scales, num_levels)
    strides_y, strides_x = build_strides(
        feature_sizes, features_H, features_W, num_levels)
    anchor_scales = build_scales(
        anchor_scale, scale_aspect_ratio_combinations, num_levels)
    return ((strides_y, strides_x, octave_scales, aspects, anchor_scales),
            num_levels, scale_aspect_ratio_combinations)


def build_strides(feature_sizes, features_H, features_W, num_levels):
    """Generates strides for each EfficientNet layer that feeds BiFPN network.

    # Arguments:
        feature_sizes: Numpy array representing the feature sizes of inputs to
            EfficientNet layers.
        features_H: Numpy array representing the height of the input features.
        features_W: Numpy array representing the width of the input features.
        num_levels: Int, representing the number of feature levels.

    # Returns:
        Tuple: Containing strides in y and x direction.
    """
    base_feature_H, base_feature_W = feature_sizes[0]
    strides_y = np.reshape(
        base_feature_H*np.reciprocal(features_H), (num_levels, -1))
    strides_x = np.reshape(
        base_feature_W*np.reciprocal(features_W), (num_levels, -1))
    return strides_y, strides_x


def build_features(feature_sizes, min_level, max_level,
                   scale_aspect_ratio_combinations):
    """Calculates feature height and width for each EfficientNet layer that
    feeds BiFPN network.

    # Arguments:
        feature_sizes: Numpy array representing the feature sizes of inputs to
            EfficientNet layers.
        min_level: Int, Number representing the index of the earliest
            EfficientNet layer that feeds the BiFPN layers.
        max_level: Int, Number representing the index of the last EfficientNet
            layer that feeds the BiFPN layers.
        scale_aspect_ratio_combinations: Int, representing the number of
        possible combinations of scales and aspect ratios.

    # Returns:
        Tuple: Containing feature height and width.
    """
    feature_H = feature_sizes[min_level: max_level + 1][:, 0]
    feature_W = feature_sizes[min_level: max_level + 1][:, 1]
    features_H = np.repeat(feature_H, scale_aspect_ratio_combinations)
    features_W = np.repeat(feature_W, scale_aspect_ratio_combinations)
    return features_H, features_W


def build_octaves(num_scales, aspect_ratios, num_levels):
    """Calculates octave scales for each EfficientNet layer that feeds
    BiFPN network.

    # Arguments:
        num_scales: Int, specifying the number of scales in the
            anchor boxes.
        aspect_ratios: List, specifying the aspect rations of the anchor boxes.
        num_levels: Int, representing the number of feature levels.

    # Returns:
        octave_scales: Numpy array of shape ``(5, 9)``.
    """
    scale_octaves = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octave_scales = (np.tile(scale_octaves, num_levels)
                     / float(num_scales)).reshape(num_levels, -1)
    return octave_scales


def build_aspects(aspect_ratios, num_scales, num_levels):
    """Calculates aspect ratios for each EfficientNet layer that feeds
    BiFPN network.

    # Arguments:
        aspect_ratios: List, specifying the aspect rations of the anchor boxes.
        num_scales: Int, specifying the number of scales in the anchor boxes.
        num_levels: Int, representing the number of feature levels.

    # Returns:
        aspects: Numpy array of shape ``(5, 9)``.
    """
    aspect = np.tile(aspect_ratios, len(range(num_scales)))
    aspects = np.tile(aspect, num_levels).reshape(num_levels, -1)
    return aspects


def build_scales(anchor_scale, scale_aspect_ratio_combinations, num_levels):
    """Calculates anchor scales for each EfficientNet layer that feeds
    BiFPN network.

    # Arguments:
        anchor_scale: Numpy array representing the scales of the anchor box.
        scale_aspect_ratio_combinations: Int, representing the number of
            possible combinations of scales and aspect ratios.
        num_levels: Int, representing the number of feature levels.

    # Returns:
        anchor_scales: Numpy array of shape ``(5, 9)``.
    """
    anchor_scales = np.reshape(
        np.repeat(anchor_scale, scale_aspect_ratio_combinations),
        (num_levels, -1))
    return anchor_scales


def compute_aspect_ratio(aspect):
    """Calculates the aspect ratio of the anchor box.

    # Arguments:
        aspect: Numpy array representing the aspect of the anchor box.

    # Returns:
        Tuple: Containing the aspect ratio values of the anchor box.
    """
    aspect_x = np.sqrt(aspect)
    aspect_y = 1 / aspect_x
    return aspect_x, aspect_y


def compute_box_coordinates(stride_y, stride_x, octave_scale, aspect,
                            anchor_scale, image_size):
    """Calculates the coordinates of the anchor box in centre form.

    # Arguments:
        stride_y: Numpy array representing the stride value in y direction.
        stride_x: Numpy array representing the stride value in x direction.
        octave_scale: Numpy array representing the octave scale of the
            anchor box.
        aspect: Numpy array representing the aspect value.
        anchor_scale: Numpy array representing the scale of anchor box.
        image_size: Tuple, representing the size of input image.

    # Returns:
        Tuple: Containing the centre as well as width and height of
            anchor box in centre form.
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


def generate_level_boxes(strides_y, strides_x, octave_scales, aspects,
                         anchor_scales, image_size,
                         scale_aspect_ratio_combinations):
    """Generates anchor box in centre form for every feature level.

    # Arguments:
        strides_y: Numpy array representing the stride value in y direction.
        strides_x: Numpy array representing the stride value in x direction.
        octave_scales: Numpy array representing the octave scale of the
            anchor box.
        aspects: Numpy array representing the aspect value.
        anchor_scaless: Numpy array representing the scale of anchor box.
        image_size: Tuple, representing the size of input image.
        scale_aspect_ratio_combinations: Int, representing the number of
            combinations of scale and aspect ratio.

    # Returns:
        boxes_level: List containing ancho9r boxes in centre form for every
            feature level.
    """
    boxes_level = []
    for combination in range(scale_aspect_ratio_combinations):
        box_coordinates = compute_box_coordinates(
            strides_y[combination], strides_x[combination],
            octave_scales[combination], aspects[combination],
            anchor_scales[combination], image_size)
        center_x, center_y, anchor_x, anchor_y = box_coordinates
        boxes = np.vstack((center_x - anchor_x, center_y - anchor_y,
                           center_x + anchor_x, center_y + anchor_y))
        boxes = np.swapaxes(boxes, 0, 1)
        boxes_level.append(np.expand_dims(boxes, axis=1))
    return boxes_level


def generate_anchors(feature_sizes, min_level, max_level, num_scales,
                     aspect_ratios, image_size, anchor_scales):
    """Generates anchor boxes in centre form for all feature levels.

    # Arguments:
        feature_sizes: Numpy array of shape ``(8, 2)``.
        min_level: Int, Number representing the index of the earliest
            EfficientNet layer that feeds the BiFPN layers.
        max_level: Int, Number representing the index of the last EfficientNet
            layer that feeds the BiFPN layers.
        num_scales: Int, specifying the number of scales in the
            anchor boxes.
        aspect_ratios: List, specifying the aspect rations of the anchor boxes.
        image_size: Tuple, representing the size of input image.
        anchor_scales: Numpy array representing the scale of anchor box.

    # Returns:
        anchors: Numpy array of shape ``(49104, 4)``.
    """
    configuration = generate_configurations(
        feature_sizes, min_level, max_level, num_scales,
        aspect_ratios, anchor_scales)
    ((strides_y, strides_x, octave_scales, aspects, anchor_scales),
        num_levels, scale_aspect_ratio_combinations) = configuration
    boxes_all = []
    for level in range(num_levels):
        boxes_level = generate_level_boxes(
            strides_y[level], strides_x[level], octave_scales[level],
            aspects[level], anchor_scales[level], image_size,
            scale_aspect_ratio_combinations)
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchors = np.vstack(boxes_all).astype('float32')
    return anchors


def build_prior_boxes(min_level, max_level, num_scales,
                      aspect_ratios, anchor_scale, image_size):
    """Function to generate prior boxes.

    # Arguments
        min_level: Int, Number representing the index of the earliest
            EfficientNet layer that feeds the BiFPN layers.
        max_level: Int, Number representing the index of the last EfficientNet
            layer that feeds the BiFPN layers.
        num_scales: Int, specifying the number of scales in the
            anchor boxes.
        aspect_ratios: List, specifying the aspect rations of the anchor boxes.
        anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level. Or a list, one value per
            layer.
        image_size: Tuple, representing the size of input image.

    # Returns
        prior_boxes: Numpy array of shape ``(49104, 4)``.
    """
    anchor_scales = np.repeat(anchor_scale, max_level - min_level + 1)
    feature_sizes = compute_feature_sizes(image_size, max_level)
    prior_boxes = generate_anchors(
        feature_sizes, min_level, max_level, num_scales, aspect_ratios,
        image_size, anchor_scales)
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes
