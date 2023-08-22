import numpy as np
from paz.backend.anchors import (build_octaves, build_aspect, build_scales,
                                 build_strides, make_branch_boxes)
from paz.backend.boxes import to_center_form


def build_translation_anchors(image_shape, branches, num_scales,
                              aspect_ratios, scale):
    """Builds anchor boxes in centre form for given model.
    Anchor boxes a.k.a prior boxes are reference boxes built with
    various scales and aspect ratio centered over every pixel in the
    input image and branch tensors. They can be strided. Anchor boxes
    define regions of image where objects are likely to be found. They
    help object detector to accurately localize and classify objects at
    the same time handling variations in object size and shape.

    # Arguments
        image_shape: List, input image shape.
        branches: List, EfficientNet branch tensors.
        num_scales: Int, number of anchor scales.
        aspect_ratios: List, anchor box aspect ratios.
        scale: Float, anchor box scale.

    # Returns
        anchor_boxes: Array of shape `(num_boxes, 4)`.
    """
    num_scale_aspect = num_scales * len(aspect_ratios)
    args = (image_shape, branches, num_scale_aspect)
    translation_boxes = []
    for branch_arg in range(len(branches)):
        stride = build_strides(branch_arg, *args)
        boxes = make_branch_translation_boxes(
            *stride, branch_arg, branches, num_scale_aspect)
        translation_boxes.append(boxes.reshape([-1, 4]))
    translation_boxes = np.concatenate(
        translation_boxes, axis=0).astype('float32')
    return to_center_form(translation_boxes)


def make_branch_translation_boxes(strides_y, strides_x, branch_arg,
                                  branches, num_scale_aspect):
    feature_H, feature_W = branches[branch_arg].shape[1:3]
    center_x = (np.arange(0, feature_H) + 0.5) * strides_x[0]
    center_y = (np.arange(0, feature_W) + 0.5) * strides_y[0]
    center_x, center_y = np.meshgrid(center_x, center_y)
    center_x_flattened = center_x.ravel()[np.newaxis, :].T
    center_y_flattened = center_y.ravel()[np.newaxis, :].T
    centers = np.concatenate((center_x_flattened, center_y_flattened), axis=1)
    centers = np.expand_dims(centers, axis=1)

    translation_anchors = np.zeros((1, num_scale_aspect, 2))
    translation_anchors = translation_anchors + centers
    translation_anchors = translation_anchors.reshape((centers.shape[0] * num_scale_aspect, 2))
    stride_array = np.full((translation_anchors.shape[0], 1), strides_x[0])
    translation_anchors = np.concatenate((translation_anchors, stride_array), axis=-1)
    print('?j')