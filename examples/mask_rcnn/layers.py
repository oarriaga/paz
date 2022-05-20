import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from .layer_utils import batch_slice, trim_zeros_graph
from .layer_utils import box_refinement_graph, filter_low_confidence, apply_NMS,trim_by_score
from .layer_utils import get_top_detections, zero_pad_detections, apply_box_delta, \
                         clip_image_boundaries, refine_instances
from .layer_utils import compute_overlaps_graph, compute_ROI_overlaps, update_priors, \
                         compute_target_masks, pad_ROI, pad_ROI_priors
from .layer_utils import compute_ROI_level, apply_ROI_pooling, rearrange_pooled_features
from .layer_utils import clip_boxes_graph, apply_box_deltas_graph


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
        detections_batch = batch_slice(
            [rois, mrcnn_class, mrcnn_bbox],
            self.refine_detections_graph,
            self.images_per_gpu)
        return tf.reshape(detections_batch,
                          [self.batch_size, self.detection_max_instances, 6])

    def refine_detections_graph(self, rois, probs, deltas):
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        deltas_specific = tf.gather_nd(deltas, indices)

        refined_rois = apply_box_deltas_graph(
            rois, deltas_specific * self.bbox_std_dev)
        refined_rois = clip_boxes_graph(refined_rois, self.window)
        keep = tf.where(class_ids > 0)[:, 0]

        if self.detection_min_confidence:
            keep = filter_low_confidence(class_scores, keep, self.detection_min_confidence)

        nms_keep = apply_NMS(class_ids, class_scores, refined_rois, keep,
                              self.detection_max_instances, self.detection_nms_threshold)
        keep = get_top_detections(class_scores, keep, nms_keep, self.detection_max_instances)

        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep),
                     dtype=tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]], axis=1)

        return zero_pad_detections(detections, self.detection_max_instances)


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
                 pre_nms_limit, images_per_gpu, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.pre_nms_limit = pre_nms_limit
        self.images_per_gpu = images_per_gpu
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def __call__(self, inputs):
        scores, deltas, anchors = inputs
        scores = scores[:, :, 1]
        deltas = deltas * np.reshape(self.rpn_bbox_std_dev, [1, 1, 4])

        scores, deltas, pre_nms_anchors = trim_by_score(scores, deltas,
                                                        anchors, self.images_per_gpu,
                                                        self.pre_nms_limit)
        boxes = apply_box_delta(pre_nms_anchors, deltas, self.images_per_gpu)
        boxes = clip_image_boundaries(boxes, self.images_per_gpu)

        proposals = batch_slice([boxes, scores], self.NMS, self.images_per_gpu)
        return proposals

    def NMS(self, boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, self.proposal_count,
            self.nms_threshold, name='rpn_non_max_suppression')
        proposals = tf.gather(boxes, indices)
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
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
                 bbox_std_dev, use_mini_mask, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.images_per_gpu = images_per_gpu
        self.mask_shape = mask_shape
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask

    def __call__(self, inputs):
        proposals, prior_class_ids, prior_boxes, prior_masks = inputs
        names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
        outputs = batch_slice(
            [proposals, prior_class_ids, prior_boxes, prior_masks],
            self.detection_targets_graph, self.images_per_gpu, names=names)
        return outputs

    def detection_targets_graph(self, proposals, prior_class_ids, prior_boxes,
                                prior_masks):
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                      name='roi_assertion'),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)
        ground_truth = [prior_class_ids, prior_boxes, prior_masks]
        refined_priors, crowd_boxes = refine_instances(proposals, ground_truth)
        _, refined_boxes, _ = refined_priors

        overlaps = compute_overlaps_graph(proposals, refined_boxes)
        positive_indices, positive_rois, negative_rois = \
            compute_ROI_overlaps(proposals, refined_boxes, crowd_boxes,
                                 overlaps, self.train_rois_per_image, self.roi_positive_ratio)
        deltas, roi_priors = update_priors(overlaps, positive_indices,
                                           positive_rois, refined_priors, self.bbox_std_dev)
        masks = compute_target_masks(positive_rois, roi_priors, self.mask_shape, self.use_mini_mask)
        rois, num_negatives, num_positives = pad_ROI(positive_rois,
                                                     negative_rois, self.train_rois_per_image)
        roi_class_ids, deltas, masks = pad_ROI_priors(num_positives, num_negatives, roi_priors,
                                                      deltas, masks)
        return rois, roi_class_ids, deltas, masks


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

        roi_level = compute_ROI_level(boxes, image_shape)
        pooled, box_to_level = apply_ROI_pooling(roi_level, boxes,
                                                 feature_maps, self.pool_shape)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        pooled = rearrange_pooled_features(pooled, box_to_level, boxes)
        return pooled



