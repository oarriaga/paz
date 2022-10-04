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
    feature_sizes = [{'height': image_size[0], 'width': image_size[1]}]
    feature_size = image_size
    for _ in range(1, max_level + 1):
        feature_size_y = (feature_size[0] - 1) // 2 + 1
        feature_size_x = (feature_size[1] - 1) // 2 + 1
        feature_size = [feature_size_y, feature_size_x]
        feature_sizes.append(
            {'height': feature_size[0], 'width': feature_size[1]})
    return feature_sizes


def generate_configurations(feature_sizes, min_level, max_level,
                            num_scales, aspect_ratios, anchor_scales):
    anchor_configurations = {}
    feature_sizes = feature_sizes
    for level in range(min_level, max_level + 1):
        anchor_configurations[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configurations[level].append(
                    ((feature_sizes[0]['height'] /
                        float(feature_sizes[level]['height']),
                        feature_sizes[0]['width'] /
                        float(feature_sizes[level]['width'])),
                        scale_octave / float(num_scales),
                        aspect, anchor_scales[level - min_level]))
    return anchor_configurations


def compute_aspect_ratio(aspect):
    if isinstance(aspect, list):
        aspect_x, aspect_y = aspect
    else:
        aspect_x = np.sqrt(aspect)
        aspect_y = 1 / aspect_x
    return aspect_x, aspect_y


def compute_box_coordinates(configuration, image_size):
    stride, octave_scale, aspect, anchor_scale = configuration
    stride_y, stride_x = stride
    W, H = image_size
    base_anchor_x = anchor_scale * stride_x * 2 ** octave_scale
    base_anchor_y = anchor_scale * stride_y * 2 ** octave_scale
    aspect_x, aspect_y = compute_aspect_ratio(aspect)
    anchor_x = (base_anchor_x * aspect_x / 2.0) / H
    anchor_y = (base_anchor_y * aspect_y / 2.0) / W
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    center_x = center_x.reshape(-1) / H
    center_y = center_y.reshape(-1) / W
    return center_x, center_y, anchor_x, anchor_y


def generate_level_boxes(configurations, image_size):
    boxes_level = []
    for configuration in configurations:
        box_coordinates = compute_box_coordinates(configuration, image_size)
        center_x, center_y, anchor_x, anchor_y = box_coordinates
        boxes = np.vstack((center_y - anchor_y, center_x - anchor_x,
                           center_y + anchor_y, center_x + anchor_x))
        boxes = np.swapaxes(boxes, 0, 1)
        boxes_level.append(np.expand_dims(boxes, axis=1))
    return boxes_level


def generate_anchors(configuration, image_size):
    boxes_all = []
    for _, configurations in configuration.items():
        boxes_level = generate_level_boxes(configurations, image_size)
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchors = np.vstack(boxes_all)
    return anchors


def generate_boxes(configuration, image_size):
    anchor_boxes = generate_anchors(configuration, image_size)
    anchor_boxes = anchor_boxes.astype('float32')
    return anchor_boxes


def generate_anchors_per_location(num_scales, aspect_ratios):
    return num_scales * len(aspect_ratios)


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
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if isinstance(anchor_scale, (list, tuple)):
        assert len(anchor_scale) == max_level - min_level + 1
    else:
        anchor_scales = [anchor_scale] * (max_level - min_level + 1)
    feature_sizes = compute_feature_sizes(image_size, max_level)
    configuration = generate_configurations(feature_sizes, min_level,
                                            max_level, num_scales,
                                            aspect_ratios, anchor_scales)
    prior_boxes = generate_boxes(configuration, image_size)
    a1, a2, a3, a4 = np.hsplit(prior_boxes, 4)
    prior_boxes = np.concatenate([a2, a1, a4, a3], axis=1)
    prior_boxes = to_center_form(prior_boxes)
    return prior_boxes
