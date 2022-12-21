import numpy as np
from paz.backend.boxes import to_center_form


def build_prior_boxes(image_shape, branches, num_scales, aspect_ratios, scale):
    """Builds prior boxes in centre form for given model.

    # Arguments:
        image_shape
        branches:
        num_scales:
        aspect_ratios:
        scale:

    # Returns:
        priors: Array of shape `(num_boxes, 4)`.
    """
    args = (image_shape, branches, num_scales, aspect_ratios, scale)
    prior_boxes = []
    for level_arg in range(len(branches)):
        level_configs = build_level_configurations(level_arg, *args)
        boxes_level = []
        for level_config in zip(*level_configs):
            boxes = compute_box_coordinates(image_shape, *level_config)
            boxes_level.append(np.expand_dims(boxes.T, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        prior_boxes.append(boxes_level.reshape([-1, 4]))
    prior_boxes = np.concatenate(prior_boxes, axis=0).astype('float32')
    return to_center_form(prior_boxes)


def build_level_configurations(level_arg, image_shape, branches, num_scales,
                               aspect_ratios, scale):
    """Builds prior box parameter combinations per level.

    # Arguments:
        model: Keras/tensorflow model.
        num_scales: Int, number of prior box scales.
        aspect_ratios: List, prior boxes aspect ratios.
        scale: Array of shape `(5,)`, prior box scales.
        level_arg: Int, level index.

    # Returns:
        Tuple: being generated configuarations.
    """
    num_scale_aspect = num_scales * len(aspect_ratios)
    stride = build_strides(image_shape, branches, num_scale_aspect, level_arg)
    octave = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octave = octave / float(num_scales)
    aspect = np.tile(aspect_ratios, num_scales)
    scales = np.repeat(scale, num_scale_aspect)
    stride_y, stride_x = stride
    return stride_y, stride_x, octave, aspect, scales


def build_strides(image_shape, branches, num_scale_aspect, level_arg):
    """Builds level-wise strides.

    # Arguments:
        model: Keras/tensorflow model.
        num_scale_aspect: Int, count of scale aspect ratio combinations.
        level_arg: Int, level index.

    # Returns:
        Tuple: Containing strides in y and x direction.
    """
    base_feature_H, base_feature_W = image_shape
    feature_H, feature_W = branches[level_arg].shape[1:3]
    features_H = np.repeat(feature_H, num_scale_aspect).astype('float32')
    features_W = np.repeat(feature_W, num_scale_aspect).astype('float32')
    H_inverse = np.reciprocal(features_H)
    W_inverse = np.reciprocal(features_W)
    strides_y = base_feature_H * H_inverse
    strides_x = base_feature_W * W_inverse
    return strides_y, strides_x


def compute_box_coordinates(image_shape, stride_y, stride_x, octave_scale,
                            aspect, scale):
    """Calculates prior box coordinates in centre form.

    # Arguments:
        model: Keras/tensorflow model.
        stride_y: Array of shape `()`, y-direction stride.
        stride_x: Array of shape `()`, x-direction stride.
        octave_scale: Array of shape `()`, prior box octave scale.
        aspect: Array of shape `()`, prior box aspect ratio.
        scale: Array of shape `()`, prior box scales.

    # Returns:
        Tuple: holding prior box centre, width and height.
    """
    W, H = image_shape
    base_prior_W = scale * stride_x * (2 ** octave_scale)
    base_prior_H = scale * stride_y * (2 ** octave_scale)
    aspect_W, aspect_H = np.sqrt(aspect), 1/np.sqrt(aspect)
    prior_W = (base_prior_W * aspect_W / 2.0) / H
    prior_H = (base_prior_H * aspect_H / 2.0) / W
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    center_x = center_x.reshape(-1) / H
    center_y = center_y.reshape(-1) / W
    arg = ([center_x - prior_W], [center_y - prior_H],
           [center_x + prior_W], [center_y + prior_H])
    box_coordinates = np.concatenate(arg, axis=0)
    return box_coordinates
