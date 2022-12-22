import numpy as np
from paz.backend.boxes import to_center_form


def build_anchors(image_shape, branches, num_scales, aspect_ratios, scale):
    """Builds anchor boxes in centre form for given model.

    # Arguments:
        image_shape
        branches:
        num_scales:
        aspect_ratios:
        scale:

    # Returns:
        anchors: Array of shape `(num_boxes, 4)`.
    """
    num_scale_aspect = num_scales * len(aspect_ratios)
    args = (image_shape, branches, num_scale_aspect)
    anchor_boxes = []
    for branch_arg in range(len(branches)):
        stride = build_strides(branch_arg, *args)
        octave = build_octaves(num_scales, aspect_ratios)
        aspect = build_aspect(aspect_ratios, num_scales)
        scales = build_scales(scale, num_scale_aspect)
        stride_y, stride_x = stride
        branch_boxes = build_branch_boxes(
            stride_y, stride_x, octave, aspect, scales, image_shape)
        anchor_boxes.append(branch_boxes.reshape([-1, 4]))
    anchor_boxes = np.concatenate(anchor_boxes, axis=0).astype('float32')
    return to_center_form(anchor_boxes)


def build_branch_boxes(stride_y, stride_x, octave, aspect, scales,
                       image_shape):
    """_summary_

    Args:
        stride_y (_type_): _description_
        stride_x (_type_): _description_
        octave (_type_): _description_
        aspect (_type_): _description_
        scales (_type_): _description_
        image_shape (_type_): _description_

    Returns:
        _type_: _description_
    """
    branch_boxes = []
    for branch_config in zip(stride_y, stride_x, octave, aspect, scales):
        boxes = compute_box_coordinates(image_shape, *branch_config)
        branch_boxes.append(np.expand_dims(boxes.T, axis=1))
    branch_boxes = np.concatenate(branch_boxes, axis=1)
    return branch_boxes


def build_octaves(num_scales, aspect_ratios):
    """_summary_

    Args:
        num_scales (_type_): _description_
        aspect_ratios (_type_): _description_

    Returns:
        _type_: _description_
    """
    octave = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octave = octave / float(num_scales)
    return octave


def build_aspect(aspect_ratios, num_scales):
    """_summary_

    Args:
        aspect_ratios (_type_): _description_
        num_scales (_type_): _description_

    Returns:
        _type_: _description_
    """
    aspect = np.tile(aspect_ratios, num_scales)
    return aspect


def build_scales(scale, num_scale_aspect):
    """_summary_

    Args:
        scale (_type_): _description_
        num_scale_aspect (_type_): _description_

    Returns:
        _type_: _description_
    """
    scales = np.repeat(scale, num_scale_aspect)
    return scales


def build_strides(branch_arg, image_shape, branches, num_scale_aspect):
    """Builds branch-wise strides.

    # Arguments:
        model: Keras/tensorflow model.
        num_scale_aspect: Int, count of scale aspect ratio combinations.
        branch_arg: Int, branch index.

    # Returns:
        Tuple: Containing strides in y and x direction.
    """
    H_image, W_image = image_shape
    feature_H, feature_W = branches[branch_arg].shape[1:3]
    features_H = np.repeat(feature_H, num_scale_aspect).astype('float32')
    features_W = np.repeat(feature_W, num_scale_aspect).astype('float32')
    strides_y = H_image / features_H
    strides_x = W_image / features_W
    return strides_y, strides_x


def compute_box_coordinates(image_shape, stride_y, stride_x, octave_scale,
                            aspect, scale):
    """Calculates anchor box coordinates in centre form.

    # Arguments:
        model: Keras/tensorflow model.
        stride_y: Array of shape `()`, y-direction stride.
        stride_x: Array of shape `()`, x-direction stride.
        octave_scale: Array of shape `()`, anchor box octave scale.
        aspect: Array of shape `()`, anchor box aspect ratio.
        scale: Array of shape `()`, anchor box scales.

    # Returns:
        Tuple: holding anchor box centre, width and height.
    """
    W, H = image_shape
    base_anchor = build_base_anchor(scale, stride_x, stride_y, octave_scale)
    base_anchor_W, base_anchor_H = base_anchor
    aspect_W, aspect_H = compute_aspect_ratio(aspect)
    anchor_half_W, anchor_half_H = compute_anchor_dims(
        base_anchor_W, base_anchor_H, aspect_W, aspect_H, image_shape)
    center_x, center_y = compute_anchor_centres(
        stride_x, stride_y, image_shape)
    x_min, y_min = [center_x - anchor_half_W], [center_y - anchor_half_H]
    x_max, y_max = [center_x + anchor_half_W], [center_y + anchor_half_H]
    box_coordinates = np.concatenate((x_min, y_min, x_max, y_max), axis=0)
    return box_coordinates


def build_base_anchor(scale, stride_x, stride_y, octave_scale):
    """_summary_

    Args:
        scale (_type_): _description_
        stride_x (_type_): _description_
        stride_y (_type_): _description_
        octave_scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    base_anchor_W = scale * stride_x * (2 ** octave_scale)
    base_anchor_H = scale * stride_y * (2 ** octave_scale)
    return base_anchor_W, base_anchor_H


def compute_aspect_ratio(aspect):
    """_summary_

    Args:
        aspect (_type_): _description_
    """
    aspect_W, aspect_H = np.sqrt(aspect), 1/np.sqrt(aspect)
    return aspect_W, aspect_H


def compute_anchor_dims(base_anchor_W, base_anchor_H, aspect_W, aspect_H,
                        image_shape):
    """_summary_

    Args:
        base_anchor_W (_type_): _description_
        base_anchor_H (_type_): _description_
        aspect_W (_type_): _description_
        aspect_H (_type_): _description_
        W (_type_): _description_
        H (_type_): _description_

    Returns:
        _type_: _description_
    """
    W, H = image_shape
    anchor_half_W = (base_anchor_W * aspect_W / 2.0) / H
    anchor_half_H = (base_anchor_H * aspect_H / 2.0) / W
    return anchor_half_W, anchor_half_H


def compute_anchor_centres(stride_x, stride_y, image_shape):
    """_summary_

    Args:
        stride_x (_type_): _description_
        stride_y (_type_): _description_
        image_shape (_type_): _description_

    Returns:
        _type_: _description_
    """
    W, H = image_shape
    x = np.arange(stride_x / 2, H, stride_x)
    y = np.arange(stride_y / 2, W, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    normalized_center_x = center_x.flatten() / H
    normalized_center_y = center_y.flatten() / W
    return normalized_center_x, normalized_center_y
