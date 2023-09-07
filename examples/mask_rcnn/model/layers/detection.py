import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import context

from mask_rcnn.model.layer_utils import apply_box_delta, slice_batch
from mask_rcnn.model.layer_utils import clip_boxes


def get_top_detections(scores, keep, nms_keep, detection_max_instances):
    """Selects the detection with highest score values and retains it.
    Also conversion from sparse to dense form of detection ROIs.

    # Arguments:
        scores: [N] Predicted target class values.
        keep: [N] ROIs after supression.
        nms_keep: ROIs after non-maximal suppresion.
        detection_max_instances: Max number of final detections.

    # Returns :
        keep : indices of the predictions with higher score to be kept.
    """
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(nms_keep, 0))

    keep = tf.sparse.to_dense(keep)[0]
    ROI_count = detection_max_instances
    class_scores_keep = tf.gather(scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], ROI_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    return tf.gather(keep, top_ids)


def NMS_map(class_ids, scores, ROIs, keep, detection_max_instances,
            detection_nms_threshold, unique_class_id):
    """Greedily selects a subset of bounding boxes in descending order of
    score. The output of is a set of integers indexing into the input
    collection of bounding boxes representing the selected boxes.

    # Arguments:
        box_data: contains class ids, scores and ROIs.
        keep: ROIs after suppression.
        unique_class_id : unique class ids [1x No. of classes].
        detection_max_instances: Max number of final detections.
        detection_nms_threshold: Non-maximum suppression threshold for
                                 detection.

    # Returns:
        class_keep: detected instances kept after nms.
    """
    ids = tf.where(tf.equal(class_ids, unique_class_id))[:, 0]

    class_keep = tf.image.non_max_suppression(tf.gather(ROIs, ids),
                                              tf.gather(scores, ids),
                                              max_output_size=detection_max_instances,
                                              iou_threshold=detection_nms_threshold)

    class_keep = tf.gather(keep, tf.gather(ids, class_keep))
    gap = detection_max_instances - tf.shape(class_keep)[0]
    class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT',
                        constant_values=-1)
    class_keep.set_shape([detection_max_instances])
    return class_keep


def NMS_map_call(pre_nms_class_ids, pre_nms_scores, pre_nms_ROIs, keeps,
                 max_instances, nms_threshold, unique_pre_nms_class_ids):
    """Decorator function used to call to NMS_map function.
    """
    def _NMS_map_call(class_id):
        return NMS_map(pre_nms_class_ids, pre_nms_scores, pre_nms_ROIs, keeps,
                       max_instances, nms_threshold, class_id)

    return tf.map_fn(_NMS_map_call, unique_pre_nms_class_ids, dtype=tf.int64)


def apply_NMS(class_ids, scores, refined_ROIs, keep,
              detection_max_instances, detection_nms_threshold):
    """Performs NMS on the detected instances per detected instances.

    # Arguments:
        class_ids  : [1x N] array values.
        scores : probability scores for all classes.
        refined_ROIs : ROIs after NMS.
        keep : ROIs after suppression.
        detection_max_instances: Max no. of instances the model can detect.
                                 Default value is 100.
        detection_nms_threshold: Threshold value below which instances are
                                 ignored.

    # Return:
        nms_keep: Detected instances kept after performing nms.
    """
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(scores, keep)
    pre_nms_ROIs = tf.gather(refined_ROIs, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    nms_keep = NMS_map_call(pre_nms_class_ids, pre_nms_scores, pre_nms_ROIs,
                            keep, detection_max_instances,
                            detection_nms_threshold, unique_pre_nms_class_ids)
    return merge_results(nms_keep)


def merge_results(nms_keep):
    """Used by top detection layer in apply_NMS to reshape keep values
    and remove negative indices if any.

    # Arguments:
        nms_keep : ROIs after nms.
    """
    nms_keep = tf.reshape(nms_keep, [-1])

    return tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])


def filter_low_confidence(class_scores, keep, detection_min_confidence):
    """Computes proposals with highest confidence and filters the lower ones
    based on the given threshold value.

    # Arguments:
        class_scores: probability scores for all classes
        keep : ROIs after supression
        detection_min_confidence: Minimum probability value to accept a
                                  detected instance

    # Return:
        keep: ROIs after thresholding.
    """
    confidence = tf.where(class_scores >= detection_min_confidence)[:, 0]
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(confidence, 0))
    return tf.sparse.to_dense(keep)[0]


def zero_pad_detections(detections, detection_max_instances):
    """Matches keep detection shape same as the max instance by zero padding.

    # Arguments:
        detections: num of detections.
        detection_max_instances: Max number of final detections.

    # Return:
        detections: num of detections after zero padding.
    """
    gap = detection_max_instances - tf.shape(detections)[0]
    return tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')


def compute_delta(probabilities, deltas):
    """Used by Detection Layer to compute the top class_ids, scores and deltas
    from the normalised proposals.

    # Arguments:
        probs: Normalized proposed classes.
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply.

    # Returns:
        class_ids: [N] Top class ids detected.
        class_scores: [N] Scores of the top detected classes.
        delta specific: [N, (dy, dx, log(dh), log(dw))].
    """
    class_ids = tf.argmax(probabilities, axis=1, output_type=tf.int32)
    indices = tf.stack([tf.range(tf.shape(probabilities)[0]), class_ids],
                       axis=1)
    class_scores = tf.gather_nd(probabilities, indices)
    deltas_specific = tf.gather_nd(deltas, indices)
    return class_ids, class_scores, deltas_specific


def compute_refined_ROIs(ROIs, deltas, windows):
    """Used by Detection Layer to apply changes to bounding box and clips the
    bounding boxes to specific window size.

    # Arguments:
        ROIs: Proposed regions form the  proposal layer.
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply.
        windows: [1x4] Default None.

    # Returns:
        ROIs: Normalized proposals.
    """
    refined_ROIs = apply_box_delta(ROIs, deltas)
    refined_ROIs = clip_boxes(refined_ROIs, windows)
    return refined_ROIs


def compute_keep(class_ids, class_scores, refined_ROIs,
                 detection_min_confidence, detection_max_instances,
                 detection_nms_threshold):
    """Used by Detection Layer to keep proposed regions and class_ids after
    non-maximum suppresion.

    # Arguments:
        class_ids: predicted class ids.
        class_scores: probability scores for all classes.
        refined_ROIs: ROIs after NMS.
        detection_min_confidence: Minimum probability value
                                  to accept a detected instance.
        detection_max_instances: Max number of final detections.
        detection_nms_threshold: Non-maximum suppression threshold for
                                 detection.

    # Returns:
        keep: class ids after NMS
    """
    keep = tf.where(class_ids > 0)[:, 0]
    if detection_min_confidence:
        keep = filter_low_confidence(class_scores, keep,
                                     detection_min_confidence)

    nms_keep = apply_NMS(class_ids, class_scores, refined_ROIs, keep,
                         detection_max_instances, detection_nms_threshold)
    keep = get_top_detections(class_scores, keep, nms_keep,
                              detection_max_instances)
    return keep


def refine_detections(ROIs, probs, deltas, bounding_box_std_dev, windows,
                      detection_min_confidence, detection_max_instances,
                      detection_nms_threshold):
    """Used by Detection Layer to keep detections with high scores and
    confidence. Also removes the detections with low confidence and scores.

    # Arguments:
        ROIs: Proposed regions form the  proposal layer
        probs: Normalized proposed classes for the proposed regions
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        bounding_box_std_dev : Bounding box refinement standard deviation for
                               final detections.
        windows:  [1x4] Default None
        detection_min_confidence: Minimum probability value to accept a
                                  detected instance, ROIs below this
                                  threshold are skipped.
        detection_max_instances: Max number of final detections.
        detection_nms_threshold: Non-maximum suppression threshold
                                 for detection.

    # Return:
        detections: num of detections after zero padding.
    """
    class_ids, class_scores, deltas_specific = compute_delta(probs, deltas)
    refined_ROIs = compute_refined_ROIs(ROIs,
                                        deltas_specific * bounding_box_std_dev,
                                        windows)
    keep = compute_keep(class_ids, class_scores, refined_ROIs,
                        detection_min_confidence,
                        detection_max_instances, detection_nms_threshold)

    gather_refined_ROIs = tf.gather(refined_ROIs, keep)
    gather_class_ids = tf.cast(
        tf.gather(class_ids, keep), dtype=tf.float32)[..., tf.newaxis]
    gather_class_scores = tf.gather(class_scores, keep)[..., tf.newaxis]
    detections = tf.concat(
        [gather_refined_ROIs, gather_class_ids, gather_class_scores], axis=1)
    return zero_pad_detections(detections, detection_max_instances)


class DetectionLayer(Layer):
    """Detects final bounding boxes and masks for given proposals.

    # Arguments:
        ROIs: Proposed regions form the  proposal layer
              [batch, N, (y_min, x_min, y_max, x_max)].
        mrcnn_class: [batch, train_ROIs_per_image]. Integer class IDs.
        mrcnn_bounding_box: Normalized ground-truth boxes
                            [batch, max_ground_truth_instances,
                            (x1, y1, x2, y2)].

    # Returns:
        [batch, num_detections, (y_min, x_min, y_max, x_max, class_id,
         class_score)] where coordinates are normalized.
    """

    def __init__(self, batch_size, bounding_box_std_dev, images_per_gpu,
                 detection_max_instances, detection_min_confidence,
                 detection_nms_threshold, image_shape, window, **kwargs,):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.bounding_box_std_dev = bounding_box_std_dev
        self.images_per_gpu = images_per_gpu
        self.detection_max_instances = detection_max_instances
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_threshold = detection_nms_threshold
        self.image_shape = image_shape
        self.window = window

    def call(self, inputs):
        ROIs, mrcnn_class, mrcnn_bounding_box = inputs
        detections_batch = slice_batch([ROIs, mrcnn_class, mrcnn_bounding_box],
                                       [tf.cast(self.bounding_box_std_dev,
                                                dtype=tf.float32),
                                       self.window,
                                       self.detection_min_confidence,
                                       self.detection_max_instances,
                                       tf.cast(self.detection_nms_threshold,
                                               dtype=tf.float32)],
                                       refine_detections,
                                       self.images_per_gpu)

        return tf.reshape(detections_batch,
                          [self.batch_size, self.detection_max_instances, 6])
