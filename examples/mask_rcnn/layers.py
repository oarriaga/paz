import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda

from mask_rcnn import utils


class DetectionLayer(Layer):
    """Detects final bounding boxes and masks for given proposals

    # Arguments:
        config: instance of base configuration class

    # Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.BATCH_SIZE = config.BATCH_SIZE
        self.WINDOW = config.WINDOW
        self.BBOX_STD_DEV = config.BBOX_STD_DEV
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.DETECTION_MAX_INSTANCES = config.DETECTION_MAX_INSTANCES
        self.DETECTION_MIN_CONFIDENCE = config.DETECTION_MIN_CONFIDENCE
        self.DETECTION_NMS_THRESHOLD = config.DETECTION_NMS_THRESHOLD


    def call(self, inputs):
        ROIs, mrcnn_class, mrcnn_bbox = inputs

        detections_batch = utils.batch_slice(
            [ROIs, mrcnn_class, mrcnn_bbox],
            lambda x, y, w: self.refine_detections_graph(x, y, w),
            self.IMAGES_PER_GPU)

        return tf.reshape(
            detections_batch,
            [self.BATCH_SIZE, self.DETECTION_MAX_INSTANCES, 6])


    def refine_detections_graph(self, rois, probs, deltas):
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        deltas_specific = tf.gather_nd(deltas, indices)

        refined_rois = apply_box_deltas_graph(
            rois, deltas_specific * self.BBOX_STD_DEV)
        refined_rois = clip_boxes_graph(refined_rois, self.WINDOW)
        keep = tf.where(class_ids > 0)[:, 0]

        if self.DETECTION_MIN_CONFIDENCE:
            keep = self.filter_low_confidence(class_scores, keep)

        nms_keep = self.apply_NMS(class_ids, class_scores, refined_rois, keep)
        keep = self.get_top_detections(class_scores, keep, nms_keep)

        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), dtype=tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)

        return self.zero_pad_detections(detections)


    def get_top_detections(self, scores, keep, nms_keep):
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        roi_count = self.DETECTION_MAX_INSTANCES
        class_scores_keep = tf.gather(scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        
        return tf.gather(keep, top_ids)


    def NMS_map(self, pre_nms_elements, keep, unique_class_id):
        class_ids, scores, ROIs = pre_nms_elements
        ixs = tf.where(tf.equal(class_ids, unique_class_id))[:, 0]
        
        class_keep = tf.image.non_max_suppression(
                tf.gather(ROIs, ixs),
                tf.gather(scores, ixs),
                max_output_size=self.DETECTION_MAX_INSTANCES,
                iou_threshold=self.DETECTION_NMS_THRESHOLD)
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        
        gap = self.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        class_keep.set_shape([self.DETECTION_MAX_INSTANCES])
        
        return class_keep


    def apply_NMS(self, class_ids, scores, refined_rois, keep):
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]
        
        pre_nms_elements = [pre_nms_class_ids, pre_nms_scores, pre_nms_rois]
        nms_keep = tf.map_fn(lambda x:self.NMS_map(pre_nms_elements, keep, x), 
                             unique_pre_nms_class_ids,
                             dtype=tf.int64)

        return self.merge_results(nms_keep)


    def filter_low_confidence(self, class_scores, keep):
        conf_keep = tf.where(class_scores >= self.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        return tf.sparse.to_dense(keep)[0]


    def merge_results(self, nms_keep):
        nms_keep = tf.reshape(nms_keep, [-1])
        return tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])


    def zero_pad_detections(self, detections):
        gap = self.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
        return tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')


    def compute_output_shape(self, input_shape):
        return (None, self.DETECTION_MAX_INSTANCES, 6)


class ProposalLayer(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
       to the second stage. Filtering is done based on anchor scores and
       non-max suppression to remove overlaps. It also applies bounding
       box refinement deltas to anchors.

    # Arguments:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    # Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.RPN_BBOX_STD_DEV = config.RPN_BBOX_STD_DEV
        self.PRE_NMS_LIMIT = config.PRE_NMS_LIMIT
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold


    def call(self, inputs):
        scores, deltas, anchors = inputs
        scores = scores[:, :, 1]
        deltas = deltas * np.reshape(self.RPN_BBOX_STD_DEV, [1, 1, 4])

        scores, deltas, pre_nms_anchors = self.trim_by_score(scores, deltas, anchors)
        boxes = self.apply_box_delta(pre_nms_anchors, deltas)
        boxes = self.clip_image_boundaries(boxes)

        proposals = utils.batch_slice([boxes, scores], self.NMS,
                                self.IMAGES_PER_GPU)
        return proposals


    def NMS(self, boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, self.proposal_count,
            self.nms_threshold, name='rpn_non_max_suppression')
        proposals = tf.gather(boxes, indices)
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals


    def trim_by_score(self, scores, deltas, anchors):
        pre_nms_limit = tf.minimum(self.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        indices = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name='top_anchors').indices
        scores = utils.batch_slice([scores, indices], lambda x, y: tf.gather(x, y),
                                   self.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, indices], lambda x, y: tf.gather(x, y),
                                   self.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, indices], lambda a, x: tf.gather(a, x),
                                    self.IMAGES_PER_GPU,
                                    names=['pre_nms_anchors'])
        return scores, deltas, pre_nms_anchors


    def apply_box_delta(self, pre_nms_anchors, deltas):
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.IMAGES_PER_GPU,
                                  names=['refined_anchors'])
        return boxes


    def clip_image_boundaries(self, boxes):
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes, lambda x: clip_boxes_graph(x, window),
                            self.IMAGES_PER_GPU,
                            names=['refined_anchors_clipped'])
        return boxes


    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


#Detection Target Layer
def overlaps_graph(boxes_A, boxes_B):
    """Computes IoU overlaps between two sets of boxes.

    # Arguments:
        boxesA: [N, (y_min, x_min, y_max, x_max)]
        boxesB = [N, (y_min, x_min, y_max, x_max)]
    """
    box_A = tf.reshape(tf.tile(tf.expand_dims(boxes_A, 1),
                            [1, 1, tf.shape(boxes_B)[0]]), [-1, 4])
    box_B = tf.tile(boxes_B, [tf.shape(boxes_A)[0], 1])
    # 2. Compute intersections
    box_A_y_min, box_A_x_min, box_A_y_max, box_A_x_max = tf.split(box_A, 4, axis=1)
    box_B_y_min, box_B_x_min, box_B_y_max, box_B_x_max = tf.split(box_B, 4, axis=1)
    y_min = tf.maximum(box_A_y_min, box_B_y_min)
    x_min = tf.maximum(box_A_x_min, box_B_x_min)
    y_max = tf.minimum(box_A_y_max, box_B_y_max)
    x_max = tf.minimum(box_A_x_max, box_B_x_max)
    intersection = tf.maximum(x_max - x_min, 0) * tf.maximum(y_max - y_min, 0)
    # 3. Compute unions
    box_A_area = (box_A_y_max - box_A_y_min) * (box_A_x_max - box_A_x_min)
    box_B_area = (box_B_y_max - box_B_y_min) * (box_B_x_max - box_B_x_min)
    union = box_A_area + box_B_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes_A)[0], tf.shape(boxes_B)[0]])
    return overlaps


class DetectionTargetLayer(Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
       and masks for each.

    # Arguments:
        proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        prior_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        prior_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                   coordinates.
        prior_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    # Returns: 
        rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
               coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
        target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                      Masks cropped to bbox boundaries and resized to neural
                      network output size.

    # Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.MASK_SHAPE = config.MASK_SHAPE
        self.TRAIN_ROIS_PER_IMAGE = config.TRAIN_ROIS_PER_IMAGE
        self.ROI_POSITIVE_RATIO = config.ROI_POSITIVE_RATIO
        self.BBOX_STD_DEV = config.BBOX_STD_DEV
        self.USE_MINI_MASK = config.USE_MINI_MASK

    def call(self, inputs):
        proposals, prior_class_ids, prior_boxes, prior_masks = inputs

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ['rois', 'target_class_ids', 'target_bbox', 'target_mask']
        outputs = batch_slice(
            [proposals, prior_class_ids, prior_boxes, prior_masks],
            lambda w, x, y, z: self.detection_targets_graph(
                w, x, y, z),
            self.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.TRAIN_ROIS_PER_IMAGE, self.MASK_SHAPE[0],
             self.MASK_SHAPE[1])  # masks
        ]


    def detection_targets_graph(self, proposals, prior_class_ids, prior_boxes, prior_masks):
        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                    name='roi_assertion'),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        ground_truth = [prior_class_ids, prior_boxes, prior_masks]
        proposals, ground_truth = self.remove_zero_padding(proposals, ground_truth)
        refined_ground_truth, crowd_boxes = self.refine_instances(ground_truth)
        _, refined_boxes, _ = refined_ground_truth

        overlaps = overlaps_graph(proposals, refined_boxes)
        positive_indices, positive_ROI, negative_ROI = self.compute_ROI_overlaps(
                                                proposals, refined_boxes,
                                                crowd_boxes, overlaps)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
        )
        roi_prior_boxes = tf.gather(prior_boxes, roi_gt_box_assignment)
        roi_prior_class_ids = tf.gather(prior_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement_graph(positive_rois, roi_prior_boxes)
        deltas /= self.BBOX_STD_DEV

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(prior_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois
        if self.USE_MINI_MASK:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_prior_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                        box_ids,
                                        self.MASK_SHAPE)
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_prior_boxes = tf.pad(roi_prior_boxes, [(0, N + P), (0, 0)])
        roi_prior_class_ids = tf.pad(roi_prior_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_prior_class_ids, deltas, masks


    def remove_zero_padding(self, proposals, ground_truth):
        class_ids, boxes, masks = ground_truth
        proposals, _ = utils.trim_zeros_graph(proposals, name='trim_proposals')
        boxes, non_zeros = utils.trim_zeros_graph(boxes,
                                        name='trim_prior_boxes')
        class_ids = tf.boolean_mask(class_ids, non_zeros,
                                          name='trim_prior_class_ids')
        masks = tf.gather(masks, tf.where(non_zeros)[:, 0], axis=2,
                          name='trim_prior_masks')

        return proposals, [class_ids, boxes, masks]


    def refine_instances(self, ground_truth):
        class_ids, boxes, masks = ground_truth
        crowd_ix = tf.where(class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(boxes, crowd_ix)
        class_ids = tf.gather(class_ids, non_crowd_ix)
        boxes = tf.gather(boxes, non_crowd_ix)
        masks = tf.gather(masks, non_crowd_ix, axis=2)
        return [class_ids, boxes, masks], crowd_boxes


    def compute_ROI_overlaps(self, proposals, boxes, crowd_boxes, overlaps):
        crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        return self.gather_ROIs(positive_indices, negative_indices)


    def gather_ROIs(self, positive_indices, negative_indices):
        positive_count = int(self.TRAIN_ROIS_PER_IMAGE *
                             self.ROI_POSITIVE_RATIO)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]

        ratio = 1.0 / self.ROI_POSITIVE_RATIO
        negative_count = tf.cast(ratio * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)
        return positive_indices, positive_rois, negative_rois


    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


#Proposal layer
def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name='apply_box_deltas_out')
    return result

def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

#Region proposal layer
def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.
    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = Activation(
        'softmax', name='rpn_class_xxx')(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    # Arguments:
        pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array.
        image_shape: shape of image
        feature_maps: List of feature maps from different levels of the pyramid.
                      Each is [batch, height, width, channels]

    # Returns:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)


    def call(self, inputs):
        boxes, image_shape = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        ROI_level = self.compute_ROI_level(boxes, image_shape)
        pooled, box_to_level = self.apply_ROI_pooling(ROI_level, boxes, feature_maps)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        pooled = self.rearrange_pooled_features(pooled, box_to_level, boxes)
        return pooled


    def compute_ROI_level(self, boxes, image_shape):
        y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=2)
        height = y_max - y_min
        width = x_max - x_min
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

        ROI_level = self.log2_graph(tf.sqrt(height * width) / (224.0 / tf.sqrt(image_area)))
        ROI_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(ROI_level), tf.int32)))
        return tf.squeeze(ROI_level, 2)


    def apply_ROI_pooling(self, ROI_level, boxes, feature_maps):
        pooled, box_to_level = [], []
        for index, level in enumerate(range(2, 6)):
            level_index = tf.where(tf.equal(ROI_level, level))
            level_boxes = tf.gather_nd(boxes, level_index)
            box_indices = tf.cast(level_index[:, 0], tf.int32)
            box_to_level.append(level_index)

            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            pooled.append(tf.image.crop_and_resize(
                feature_maps[index], level_boxes, box_indices, self.pool_shape,
                method='bilinear'))
        pooled = tf.concat(pooled, axis=0)
        box_to_level = tf.concat(box_to_level, axis=0)
        return pooled, box_to_level


    def rearrange_pooled_features(self, pooled, box_to_level, boxes):
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        top_k_indices = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        top_k_indices = tf.gather(box_to_level[:, 2], top_k_indices)
        pooled = tf.gather(pooled, top_k_indices)
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        return tf.reshape(pooled, shape)


    def log2_graph(self, x):
        return tf.math.log(x) / tf.math.log(2.0)


    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )
