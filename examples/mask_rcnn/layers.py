import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from .layer_utils import slice_batch
from .layer_utils import filter_low_confidence, apply_NMS,trim_by_score
from .layer_utils import get_top_detections, zero_pad_detections, apply_box_delta, \
                         clip_image_boundaries, refine_instances
from .layer_utils import compute_overlaps_graph, compute_ROI_overlaps, update_priors, \
                         compute_target_masks, pad_ROI, pad_ROI_priors
from .layer_utils import compute_ROI_level, apply_ROI_pooling, rearrange_pooled_features
from .layer_utils import clip_boxes, apply_box_deltas, NMS, refine_detections_graph
from .layer_utils import trim_zeros, box_refinement, detection_targets_graph


class DetectionLayer(Layer):
    """Detects final bounding boxes and masks for given proposals

    # Arguments:
        config: instance of base configuration class

    # Returns:
        [batch, num_detections, (y_min, x_min, y_max, x_max, class_id,
         class_score)] where coordinates are normalized.
    """

    def __init__(self, batch_size, window, bbox_std_dev, images_per_gpu, detection_max_instances,
                 detection_min_confidence, detection_nms_threshold, **kwargs,):
        super(DetectionLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.window = window
        self.bbox_std_dev = bbox_std_dev
        self.images_per_gpu = images_per_gpu
        self.detection_max_instances = detection_max_instances
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_threshold = detection_nms_threshold

    def __call__(self, inputs):
        rois, mrcnn_class, mrcnn_bbox = inputs

        std_dev_batch = tf.repeat(self.bbox_std_dev, self.batch_size)
        std_dev_batch = tf.cast(std_dev_batch, dtype=tf.float32)
        window_batch = tf.repeat(self.window, self.batch_size)
        detection_min_confidence_batch = tf.repeat(self.detection_min_confidence, self.batch_size)
        detection_max_instances_batch = tf.repeat(self.detection_max_instances, self.batch_size)
        detection_nms_threshold_batch = tf.repeat(self.detection_nms_threshold, self.batch_size)

        detections_batch = slice_batch(
            [rois, mrcnn_class, mrcnn_bbox, std_dev_batch, window_batch,
             detection_min_confidence_batch, detection_max_instances_batch,
             detection_nms_threshold_batch ],
            refine_detections_graph,
            self.images_per_gpu)
        return tf.reshape(detections_batch,
                          [self.batch_size, self.detection_max_instances, 6])


class ProposalLayer(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
       to the second stage. Filtering is done based on anchor scores and
       non-max suppression to remove overlaps. It also applies bounding
       box refinement deltas to anchors.

    # Arguments:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y_min, x_min, y_max, x_max)] anchors
                 in normalized coordinates

    # Returns:
        Normalized proposals [batch, rois, (y_min, x_min, y_max, x_max)]
    """

    def __init__(self, proposal_count, nms_threshold, rpn_bbox_std_dev,
                 pre_nms_limit, images_per_gpu, batch_size, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.pre_nms_limit = pre_nms_limit
        self.images_per_gpu = images_per_gpu
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

    def __call__(self, inputs):
        scores, deltas, anchors = inputs
        scores = scores[:, :, 1]
        deltas = deltas * np.reshape(self.rpn_bbox_std_dev, [1, 1, 4])

        scores, deltas, pre_nms_anchors = trim_by_score(scores, deltas,
                                                        anchors, self.images_per_gpu,
                                                        self.pre_nms_limit)
        boxes = apply_box_delta(pre_nms_anchors, deltas, self.images_per_gpu)
        boxes = clip_image_boundaries(boxes, self.images_per_gpu)

        proposal_count = tf.repeat(self.proposal_count, self.batch_size)
        threshold = tf.repeat(self.nms_threshold, self.batch_size)

        proposals = slice_batch([boxes, scores, proposal_count, threshold], NMS, self.images_per_gpu)
        return proposals


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
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.images_per_gpu = images_per_gpu
        self.mask_shape = mask_shape
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask
        self.batch_size = batch_size

    def __call__(self, inputs):
        proposals, prior_class_ids, prior_boxes, prior_masks = inputs
        names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
        train_rois_per_image = tf.repeat(self.train_rois_per_image, self.batch_size)
        roi_positive_ratio = tf.repeat(self.roi_positive_ratio, self.batch_size)
        roi_positive_ratio = tf.cast(roi_positive_ratio, dtype=tf.int32)
        mask_shape = tf.repeat(self.mask_shape, self.batch_size)
        use_mini_mask = tf.repeat(self.use_mini_mask, self.batch_size)
        bbox_std_dev = tf.repeat(self.bbox_std_dev, self.batch_size)
        bbox_std_dev = tf.cast(bbox_std_dev, dtype=tf.float32)

        outputs = slice_batch(
            [proposals, prior_class_ids, prior_boxes, prior_masks, train_rois_per_image,
             roi_positive_ratio.numpy(), mask_shape, use_mini_mask, bbox_std_dev],
            detection_targets_graph, self.images_per_gpu, names=names)
        return outputs


class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    # Arguments:
        pool_shape: [pool_height, pool_width] of the output pooled regions
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array.
        image_shape: shape of image
        feature_maps: List of feature maps from different levels
                      of the pyramid. Each is [batch, height, width, channels]

    # Returns:
        Pooled regions: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def __call__(self, inputs):
        boxes, image_shape = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        roi_level =  compute_ROI_level(boxes, image_shape)
        pooled, box_to_level = apply_ROI_pooling(roi_level, boxes,
                                                 feature_maps, self.pool_shape)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        pooled = rearrange_pooled_features(pooled, box_to_level, boxes)
        return pooled