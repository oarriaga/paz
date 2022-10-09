import numpy as np
from paz.backend.boxes import to_center_form


def compute_feature_sizes(image_size, max_level):
    """
    # Arguments
        image_size: Tuple, (Height, Width) of the input image.
        max_level: Int, maximum level for features.

    # Returns
        feature_sizes: List, feature sizes with height and width values
        for a given image size and max level.
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
    num_levels = max_level + 1 - min_level
    base_feature_H, base_feature_W = feature_sizes[0]
    scale_aspect_ratio_combinations = (len(range(num_scales))
                                       * len(aspect_ratios))
    feature_H = feature_sizes[min_level: max_level + 1][:, 0]
    feature_W = feature_sizes[min_level: max_level + 1][:, 1]
    features_H = np.repeat(feature_H, scale_aspect_ratio_combinations)
    features_W = np.repeat(feature_W, scale_aspect_ratio_combinations)
    scale_octaves = np.repeat(list(range(num_scales)), len(aspect_ratios))
    aspect = np.tile(aspect_ratios, len(range(num_scales)))
    strides_y = np.reshape(base_feature_H*np.reciprocal(features_H),
                           (num_levels, -1))
    strides_x = np.reshape(base_feature_W*np.reciprocal(features_W),
                           (num_levels, -1))
    octave_scales = (np.tile(scale_octaves, num_levels)
                     / float(num_scales)).reshape(num_levels, -1)
    aspects = np.tile(aspect, num_levels).reshape(num_levels, -1)
    anchor_scales = np.reshape(np.repeat(anchor_scale,
                               scale_aspect_ratio_combinations),
                               (num_levels, -1))
    return ((strides_y, strides_x, octave_scales, aspects, anchor_scales),
            num_levels, scale_aspect_ratio_combinations)


def compute_aspect_ratio(aspect):
    aspect_x = np.sqrt(aspect)
    aspect_y = 1 / aspect_x
    return aspect_x, aspect_y


def compute_box_coordinates(stride_y, stride_x, octave_scale, aspect,
                            anchor_scale, image_size):
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
    boxes_level = []
    for combination in range(scale_aspect_ratio_combinations):
        box_coordinates = compute_box_coordinates(strides_y[combination],
                                                  strides_x[combination],
                                                  octave_scales[combination],
                                                  aspects[combination],
                                                  anchor_scales[combination],
                                                  image_size)
        center_x, center_y, anchor_x, anchor_y = box_coordinates
        boxes = np.vstack((center_x - anchor_x, center_y - anchor_y,
                           center_x + anchor_x, center_y + anchor_y))
        boxes = np.swapaxes(boxes, 0, 1)
        boxes_level.append(np.expand_dims(boxes, axis=1))
    return boxes_level


def generate_anchors(feature_sizes, min_level, max_level, num_scales,
                     aspect_ratios, image_size, anchor_scales):
    ((strides_y, strides_x, octave_scales, aspects, anchor_scales), num_levels,
     scale_aspect_ratio_combinations) = generate_configurations(feature_sizes,
                                                                min_level,
                                                                max_level,
                                                                num_scales,
                                                                aspect_ratios,
                                                                anchor_scales)
    boxes_all = []
    for level in range(num_levels):
        boxes_level = generate_level_boxes(strides_y[level], strides_x[level],
                                           octave_scales[level],
                                           aspects[level],
                                           anchor_scales[level],
                                           image_size,
                                           scale_aspect_ratio_combinations)
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchors = np.vstack(boxes_all).astype('float32')
    return anchors


def build_prior_boxes(min_level, max_level, num_scales,
                      aspect_ratios, anchor_scale, image_size):
    """Function to generate prior boxes.

    # Arguments
    min_level: Int, minimum level for features.
    max_level: Int, maximum level for features.
    num_scales: Int, specifying the number of scales in the anchor boxes.
    aspect_ratios: List, specifying the aspect ratio of the
    default anchor boxes. Computed with k-mean on COCO dataset.
    anchor_scale: float number representing the scale of size of the base
    anchor to the feature stride 2^level. Or a list, one value per layer.
    image_size: Int, size of the input image.

    # Returns
    prior_boxes: Numpy, Prior anchor boxes corresponding to the
    feature map size of each feature level.
    """
    anchor_scales = np.repeat(anchor_scale, max_level - min_level + 1)
    feature_sizes = compute_feature_sizes(image_size, max_level)
    prior_boxes = generate_anchors(feature_sizes, min_level, max_level,
                                   num_scales, aspect_ratios, image_size,
                                   anchor_scales)
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes
