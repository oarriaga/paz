import numpy as np
from paz.backend.anchors import build_strides


def build_translation_anchors(image_shape, branches,
                              num_scales, aspect_ratios):

    num_scale_aspect = num_scales * len(aspect_ratios)
    args = (branches, num_scale_aspect)
    translation_anchors = []
    for branch_arg in range(len(branches)):
        stride = build_strides(branch_arg, image_shape, *args)
        anchors = make_branch_anchors(*stride, branch_arg, *args)
        translation_anchors.append(anchors)
    translation_anchors = np.concatenate(translation_anchors, axis=0)
    return translation_anchors.astype('float32')


def make_branch_anchors(strides_y, strides_x, branch_arg,
                        branches, num_scale_aspect):
    args_1 = (strides_y, strides_x, branch_arg, branches)
    centers = compute_translation_centers(*args_1)
    args_2 = (centers, strides_x, num_scale_aspect)
    translation_anchors = append_stride_to_centre(*args_2)
    return translation_anchors


def compute_translation_centers(strides_y, strides_x, branch_arg, branches):
    feature_H, feature_W = branches[branch_arg].shape[1:3]
    center_x = (np.arange(0, feature_H) + 0.5) * strides_x[0]
    center_y = (np.arange(0, feature_W) + 0.5) * strides_y[0]
    center_x, center_y = np.meshgrid(center_x, center_y)
    center_x_flattened = center_x.reshape(-1, 1)
    center_y_flattened = center_y.reshape(-1, 1)
    centers = np.concatenate((center_x_flattened, center_y_flattened), axis=1)
    return centers


def append_stride_to_centre(centers, strides_x, num_scale_aspect):
    centers = np.repeat(centers, num_scale_aspect, axis=0)
    num_translation_anchors = centers.shape[0]
    stride_array = np.full((num_translation_anchors, 1), strides_x[0])
    translation_anchors = np.concatenate((centers, stride_array), axis=-1)
    return translation_anchors
