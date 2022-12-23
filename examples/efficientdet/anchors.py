import numpy as np
from paz.backend.boxes import to_center_form


def build_anchors(image_shape, branches, num_scales, aspect_ratios, scale):
    """Builds anchor boxes in centre form for given model.

    # Arguments:
        image_shape: List, input image shape.
        branches: List, EfficientNet branch tensors.
        num_scales: Int, number of anchor scales.
        aspect_ratios: List, anchor box aspect ratios.
        scale: Float, anchor box scale.

    # Returns:
        anchor_boxes: Array of shape `(num_boxes, 4)`.
    """
    num_scale_aspect = num_scales * len(aspect_ratios)
    args = (image_shape, branches, num_scale_aspect)
    octave = build_octaves(num_scales, aspect_ratios)
    aspect = build_aspect(num_scales, aspect_ratios)
    scales = build_scales(scale, num_scale_aspect)
    anchor_boxes = []
    for branch_arg in range(len(branches)):
        stride = build_strides(branch_arg, *args)
        boxes = make_branch_boxes(*stride, octave, aspect, scales, image_shape)
        anchor_boxes.append(boxes.reshape([-1, 4]))
    anchor_boxes = np.concatenate(anchor_boxes, axis=0).astype('float32')
    return to_center_form(anchor_boxes)


def build_strides(branch_arg, image_shape, branches, num_scale_aspect):
    """Builds branch-wise strides.

    # Arguments:
        branch_arg: Int, branch index.
        image_shape: List, input image shape.
        branches: List, EfficientNet branch tensors.
        num_scale_aspect: Int, count of scale aspect ratio combinations.

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


def build_octaves(num_scales, aspect_ratios):
    """Builds branch-wise EfficientNet anchor box octaves.

    # Arguments:
        num_scales: Int, number of anchor scales.
        aspect_ratios: List, anchor box aspect ratios.

    # Returns:
        octave_normalized: Array of shape `(num_scale_aspect,)`.
    """
    octave = np.repeat(list(range(num_scales)), len(aspect_ratios))
    octave_normalized = octave / float(num_scales)
    return octave_normalized


def build_aspect(num_scales, aspect_ratios):
    """Builds branch-wise EfficientNet anchor box aspect ratios.

    # Arguments:
        aspect_ratios: List, anchor box aspect ratios.
        num_scales: Int, number of anchor scales.

    # Returns:
        Array of shape `(num_scale_aspect,)`.
    """
    return np.tile(aspect_ratios, num_scales)


def build_scales(scale, num_scale_aspect):
    """Builds branch-wise EfficientNet anchor box scales.

    # Arguments:
        scale: Float, anchor box scale.
        num_scale_aspect: Int, number of scale and aspect combinations.

    # Returns:
        Array of shape `(num_scale_aspect,)`.
    """
    return np.repeat(scale, num_scale_aspect)


def make_branch_boxes(stride_y, stride_x, octave,
                      aspect, scales, image_shape):
    """Builds anchor boxes per EfficientNet branch.

    # Arguments:
        stride_y: Array of shape `(num_scale_aspect,)` y-axis stride.
        stride_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        octave: Array of shape `(num_scale_aspect,)` octave scale.
        aspect: Array of shape `(num_scale_aspect,)` aspect ratio.
        scales: Array of shape `(num_scale_aspect,)` anchor box scales.
        image_shape: List, input image shape.

    # Returns:
        branch_boxes: Array of shape `(num_boxes,num_scale_aspect,4)`.
    """
    branch_boxes = []
    for branch_config in zip(stride_y, stride_x, octave, aspect, scales):
        boxes = compute_box_coordinates(image_shape, *branch_config)
        branch_boxes.append(np.expand_dims(boxes.T, axis=1))
    branch_boxes = np.concatenate(branch_boxes, axis=1)
    return branch_boxes


def compute_box_coordinates(image_shape, stride_y, stride_x,
                            octave_scale, aspect, scale):
    """Computes anchor box coordinates in centre form.

    # Arguments:
        image_shape: List, input image shape.
        stride_y: Array of shape `(num_scale_aspect,)` y-axis stride.
        stride_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        octave_scale: Array of shape `()`, anchor box octave scale.
        aspect: Array of shape `()`, anchor box aspect ratio.
        scale: Array of shape `()`, anchor box scales.

    # Returns:
        Tuple: Box coordinates in corner form.
    """
    base_anchor = build_base_anchor(scale, stride_y, stride_x, octave_scale)
    aspect_size = compute_aspect_size(aspect)
    anchor_half_W, anchor_half_H = compute_anchor_dims(
        *base_anchor, *aspect_size, image_shape)
    center_x, center_y = compute_anchor_centres(
        stride_y, stride_x, image_shape)
    x_min, y_min = [center_x - anchor_half_W], [center_y - anchor_half_H]
    x_max, y_max = [center_x + anchor_half_W], [center_y + anchor_half_H]
    box_coordinates = np.concatenate((x_min, y_min, x_max, y_max), axis=0)
    return box_coordinates


def build_base_anchor(scale, stride_y, stride_x, octave_scale):
    """Builds base anchor.

    # Arguments:
        scale: Float, anchor box scale.
        stride_y: Array of shape `(num_scale_aspect,)` y-axis stride.
        stride_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        octave_scale: Array of shape `()`, anchor box octave scale.

    # Returns:
        Tuple: Base anchor width and height.
    """
    base_anchor_W = scale * stride_x * (2 ** octave_scale)
    base_anchor_H = scale * stride_y * (2 ** octave_scale)
    return base_anchor_W, base_anchor_H


def compute_aspect_size(aspect):
    """Computes aspect width and height.

    # Arguments:
        aspect: Array of shape `()`, anchor box aspect ratio.

    # Returns:
        Tuple: Aspect width and height.
    """
    return np.sqrt(aspect), 1/np.sqrt(aspect)


def compute_anchor_dims(base_anchor_W, base_anchor_H,
                        aspect_W, aspect_H, image_shape):
    """Compute anchor's half width and height.

    # Arguments:
        base_anchor_W: Array of shape (), base anchor width.
        base_anchor_H: Array of shape (), base anchor height.
        aspect_W: Array of shape (), aspect width.
        aspect_H: Array of shape (), aspect height.
        image_shape: List, input image shape.

    # Returns:
        Tuple: Anchor's half width and height.
    """
    H, W = image_shape
    anchor_half_W = (base_anchor_W * aspect_W / 2.0)
    anchor_half_H = (base_anchor_H * aspect_H / 2.0)
    anchor_half_W_normalized = anchor_half_W / W
    anchor_half_H_normalized = anchor_half_H / H
    return anchor_half_W_normalized, anchor_half_H_normalized


def compute_anchor_centres(stride_y, stride_x, image_shape):
    """Compute anchor centres.

    # Arguments:
        stride_y: Array of shape `(num_scale_aspect,)` y-axis stride.
        stride_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        image_shape: List, input image shape.

    # Returns:
        Tuple: Normalized anchor centres.
    """
    H, W = image_shape
    x = np.arange(stride_x / 2, W, stride_x)
    y = np.arange(stride_y / 2, H, stride_y)
    center_x, center_y = np.meshgrid(x, y)
    normalized_center_x = center_x.flatten() / W
    normalized_center_y = center_y.flatten() / H
    return normalized_center_x, normalized_center_y
