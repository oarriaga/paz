import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


def compute_output_shape(input_shape, arg, layer='None'):
    """Used by to compute output shapes for different layers #keras output layer shape

    # Arguments:
        input_shape
        arg: arguments for layers
        layer : layer used
    """

    if layer == 'DetectionLayer':
        detection_max_instances = arg
        return (None, detection_max_instances, 6)
    elif layer == 'ProposalLayer':
        proposal_count = arg
        return (None, proposal_count, 4)
    elif layer == 'DetectionTargetLayer':
        train_rois_per_image, mask_shape = arg
        return [
            (None, train_rois_per_image, 4),  # ROIs
            (None, train_rois_per_image),  # class_ids
            (None, train_rois_per_image, 4),  # deltas
            (None, train_rois_per_image, mask_shape[0],
             mask_shape[1])  # masks
        ]
    elif layer == 'PyramidROIAlign':
        pool_shape = arg
        return input_shape[0][:2] + pool_shape + (input_shape[2][-1],)

    else:
        print("Invalid layer name")
        return 0


