import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from mask_rcnn.layer_utils import slice_batch, compute_targets_from_groundtruth_values


class DetectionTargetLayer(Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
       and masks for each.

    # Arguments:
        proposals: Normalized proposals
                   [batch, N, (y_min, x_min, y_max, x_max)]
        prior_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        prior_boxes: Normalized ground-truth boxes
                     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        prior_masks: [batch, height, width, MAX_GT_INSTANCES] of Boolean type

    # Returns:
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y_min, x_min, y_max, x_max)]
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]

    # Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, images_per_gpu, mask_shape, train_rois_per_image, roi_positive_ratio,
                 bbox_std_dev, use_mini_mask, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.images_per_gpu = images_per_gpu
        self.mask_shape = mask_shape
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.batch_size = batch_size

    def call(self, inputs):
        proposals, prior_class_ids, prior_boxes, prior_masks = inputs
        names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
        outputs = slice_batch([proposals, prior_class_ids, prior_boxes, prior_masks],
                              [self.train_rois_per_image, self.roi_positive_ratio,
                               self.mask_shape, self.use_mini_mask,
                               tf.cast(self.bbox_std_dev, dtype=tf.float32)],
                              compute_targets_from_groundtruth_values, self.images_per_gpu, names=names)
        return outputs
