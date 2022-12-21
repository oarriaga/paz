import numpy as np
from paz.backend.boxes import to_center_form


def build_prior_boxes(model, *args):
    """Builds anchor boxes in centre form for given model.

    # Arguments:
        model: Keras/tensorflow model.
        args: List, containing num_scales, aspect_ratios, anchor_scale

    # Returns:
        anchors: Array of shape `(num_boxes, 4)`.
    """
    boxes_all = []
    for level_arg in range(len(model.branches)):
        level_configs = build_level_configurations(model, *args, level_arg)
        boxes_level = []
        for level_config in zip(*level_configs):
            boxes = compute_box_coordinates(model, *level_config)
            boxes_level.append(np.expand_dims(boxes.T, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    prior_boxes = np.concatenate(boxes_all, axis=0).astype('float32')
    return to_center_form(prior_boxes)


def build_level_configurations(model, num_scales, aspect_ratios,
                               anchor_scale, level_arg):
    """Builds anchor box parameter combinations.

    # Arguments:
        model: Keras/tensorflow model.
        num_scales: Int, number of anchor box scales.
        aspect_ratios: List, anchor boxes aspect ratios.
        anchor_scale: Array of shape `(5,)`, anchor box scales.
        level_arg: Int, level index.

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


def build_strides(model, num_scale_aspect, level_arg):
    """Builds level-wise strides.

    # Arguments:
        model: Keras/tensorflow model.
        num_scale_aspect: Int, count of scale aspect ratio combinations.
        level_arg: Int, level index.

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


def compute_box_coordinates(model, stride_y, stride_x, octave_scale,
                            aspect, anchor_scale):
    """Calculates anchor box coordinates in centre form.

    # Arguments:
        model: Keras/tensorflow model.
        stride_y: Array of shape `()`, y-direction stride.
        stride_x: Array of shape `()`, x-direction stride.
        octave_scale: Array of shape `()`, anchor box octave scale.
        aspect: Array of shape `()`, anchor box aspect ratio.
        anchor_scale: Array of shape `()`, anchor box scales.

    # Returns:
        Tuple: holding anchor box centre, width and height.
    """
    W, H = model.input.shape[1:3]
    base_anchor_W = anchor_scale * stride_x * (2 ** octave_scale)
    base_anchor_H = anchor_scale * stride_y * (2 ** octave_scale)
    aspect_W, aspect_H = np.sqrt(aspect), 1/np.sqrt(aspect)
    anchor_W = (base_anchor_W * aspect_W / 2.0) / H
    anchor_H = (base_anchor_H * aspect_H / 2.0) / W
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    center_x = center_x.reshape(-1) / H
    center_y = center_y.reshape(-1) / W
    arg = ([center_x - anchor_W], [center_y - anchor_H],
           [center_x + anchor_W], [center_y + anchor_H])
    box_coordinates = np.concatenate(arg, axis=0)
    return box_coordinates
