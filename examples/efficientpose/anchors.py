import numpy as np
from paz.backend.anchors import build_strides


def build_translation_anchors(image_shape, branches,
                              num_scales, aspect_ratios):
    """Builds translation anchors in centre form for given model.
    Translation anchors are reference boxes built with various
    scales and aspect ratio centered over every pixel in the input
    image and branch tensors. They can be strided. Anchors define
    regions of image where objects are likely to be found. They help
    to accurately estimate the translation of the objects in the image
    the same time handling variations in object size and shape.

    # Arguments
        image_shape: List, input image shape.
        branches: List, EfficientNet branch tensors.
        num_scales: Int, number of anchor scales.
        aspect_ratios: List, anchor box aspect ratios.

    # Returns
        translation_anchors: Array of shape `(num_boxes, 3)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    num_scale_aspect = num_scales * len(aspect_ratios)
    args = (branches, num_scale_aspect)
    translation_anchors = []
    for branch_arg in range(len(branches)):
        strides = build_strides(branch_arg, image_shape, *args)
        anchors = make_branch_anchors(*strides, branch_arg, *args)
        translation_anchors.append(anchors)
    translation_anchors = np.concatenate(translation_anchors, axis=0)
    return translation_anchors.astype('float32')


def make_branch_anchors(strides_y, strides_x, branch_arg,
                        branches, num_scale_aspect):
    """Builds branch-wise EfficientPose translation anchors.

    # Arguments
        strides_y: Array of shape `(num_scale_aspect,)` y-axis stride.
        strides_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        branch_arg: Int, index of the branch.
        branches: List, EfficientNet branch tensors.
        num_scale_aspect: Int, count of scale aspect ratio combinations.

    # Returns
        translation_anchors: Array of shape `(num_branch_boxes, 3)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    args_1 = (strides_y, strides_x, branch_arg, branches)
    centers = compute_translation_centers(*args_1)
    args_2 = (centers, strides_x, num_scale_aspect)
    translation_anchors = append_stride_to_centre(*args_2)
    return translation_anchors


def compute_translation_centers(strides_y, strides_x, branch_arg, branches):
    """Compute anchor centres.

    # Arguments
        strides_y: Array of shape `(num_scale_aspect,)` y-axis stride.
        strides_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        branch_arg: Int, index of the branch.
        branches: List, EfficientNet branch tensors.

    # Returns
        centers: Array of shape `(num_boxes, 2)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    feature_H, feature_W = branches[branch_arg].shape[1:3]
    center_x = (np.arange(0, feature_H) + 0.5) * strides_x[0]
    center_y = (np.arange(0, feature_W) + 0.5) * strides_y[0]
    center_x, center_y = np.meshgrid(center_x, center_y)
    center_x_flattened = center_x.reshape(-1, 1)
    center_y_flattened = center_y.reshape(-1, 1)
    centers = np.concatenate((center_x_flattened, center_y_flattened), axis=1)
    return centers


def append_stride_to_centre(centers, strides_x, num_scale_aspect):
    """Appends anchor strides to anchor centers.

    # Arguments
        centers: Array of shape `(num_boxes, 2)`, anchor centers.
        strides_x: Array of shape `(num_scale_aspect,)` x-axis stride.
        num_scale_aspect: Int, count of scale aspect ratio combinations.

    # Returns
        translation_anchors: Array of shape `(num_branch_boxes, 3)`.

    # References
        This module is derived based on [EfficientPose](
            https://github.com/ybkscht/EfficientPose)
    """
    centers = np.repeat(centers, num_scale_aspect, axis=0)
    num_translation_anchors = centers.shape[0]
    stride_array = np.full((num_translation_anchors, 1), strides_x[0])
    translation_anchors = np.concatenate((centers, stride_array), axis=-1)
    return translation_anchors
