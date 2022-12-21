import numpy as np
from paz.backend.boxes import to_center_form


def build_prior_boxes(model, *args):
    """Generates anchor boxes in centre form for all feature levels.

    # Arguments:
        model:
        args"

    # Returns:
        anchors: Array of shape `(49104, 4)`.
    """
    boxes_all = []
    for level_arg in range(len(model.branches)):
        level_config = build_level_configuration(model, *args, level_arg)
        boxes_level = generate_level_boxes(model, *level_config)
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    prior_boxes = np.concatenate(boxes_all, axis=0).astype('float32')
    return to_center_form(prior_boxes)


def build_level_configuration(model, num_scales, aspect_ratios,
                              anchor_scale, level_arg):
    """Generates anchor box parameter combinations.

    # Arguments:
        model
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        anchor_scale: Array of shape `(5,)`, anchor box scales.

    # Returns:
        Tuple: being generated configuarations.
    """
    num_scale_aspect = num_scales * len(aspect_ratios)
    stride = build_strides(model, num_scale_aspect, level_arg)
    octave = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octave = octave / float(num_scales)
    aspect = np.tile(aspect_ratios, num_scales)
    scales = np.repeat(anchor_scale, num_scale_aspect)
    stride_y, stride_x = stride
    return stride_y, stride_x, octave, aspect, scales


def generate_level_boxes(model, *args):
    """Generates anchor box in centre form per feature level.

    # Arguments:
        args:
        image_size: Tuple, being input image size.

    # Returns:
        boxes_level: List containing anchor boxes in centre form.
    """
    boxes_level = []
    for arg in zip(*args):
        box_coordinates = compute_box_coordinates(model, *arg)
        center_x, center_y, anchor_x, anchor_y = box_coordinates
        boxes = np.concatenate(([center_x - anchor_x], [center_y - anchor_y],
                                [center_x + anchor_x], [center_y + anchor_y]),
                               axis=0)
        boxes_level.append(np.expand_dims(boxes.T, axis=1))
    return boxes_level


def build_strides(model, num_scale_aspect, level_arg):
    """Generates layer-wise EfficientNet anchor box strides.

    # Arguments:
        model:
        features_H: Array of shape `(45,)`, input feature height.
        features_W: Array of shape `(45,)`, input feature width.
        num_levels: Int, number of feature levels.

    # Returns:
        Tuple: Containing strides in y and x direction.
    """
    base_feature_H, base_feature_W = model.input.shape[1:3]
    feature_H, feature_W = model.branches[level_arg].shape[1:3]
    features_H = np.repeat(feature_H, num_scale_aspect).astype(np.float32)
    features_W = np.repeat(feature_W, num_scale_aspect).astype(np.float32)
    H_inverse = np.reciprocal(features_H)
    W_inverse = np.reciprocal(features_W)
    strides_y = base_feature_H * H_inverse
    strides_x = base_feature_W * W_inverse
    return strides_y, strides_x


def compute_box_coordinates(model, stride_y, stride_x, octave_scale, aspect,
                            anchor_scale):
    """Calculates anchor box coordinates in centre form.

    # Arguments:
        model:
        stride_y: Array of shape `()`, y-direction stride.
        stride_x: Array of shape `()`, x-direction stride.
        octave_scale: Array of shape `()`, anchor box octave scale.
        aspect: Array of shape `()`, anchor box aspect ratio.
        anchor_scale: Array of shape `()`, anchor box scales.

    # Returns:
        Tuple: holding anchor box centre, width and height.
    """
    W, H = model.input.shape[1:3]
    base_anchor_x = anchor_scale * stride_x * (2 ** octave_scale)
    base_anchor_y = anchor_scale * stride_y * (2 ** octave_scale)
    aspect_x, aspect_y = np.sqrt(aspect), 1/np.sqrt(aspect)
    anchor_x = (base_anchor_x * aspect_x / 2.0) / H
    anchor_y = (base_anchor_y * aspect_y / 2.0) / W
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    center_x = center_x.reshape(-1) / H
    center_y = center_y.reshape(-1) / W
    return center_x, center_y, anchor_x, anchor_y
